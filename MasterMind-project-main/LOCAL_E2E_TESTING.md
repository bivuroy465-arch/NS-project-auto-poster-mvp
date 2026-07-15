# Local E2E Testing — Full Lifecycle Walkthrough

This guide walks through every step of the full request lifecycle:
from minting a JWT, to hitting the Edge Gateway, to watching the Worker
consume the job from the Redis Stream.

## Prerequisites

- Docker Desktop (or Docker Engine + Compose V2) installed and running.
- `openssl` and `python3` available on your `$PATH` (standard on macOS/Linux).
- Port `6379`, `5432`, and `8000` free on your host.

---

## Step 1 — Start the cluster

```sh
cd MasterMind-project-main
docker compose up --build
```

Wait until you see all four services reach a healthy state. The output
will include lines similar to:

```
postgres  | database system is ready to accept connections
redis     | Ready to accept connections
worker    | [entrypoint] Migration applied successfully (or already up to date)
worker    | Worker <hostname>-<id> started.
gateway   | Application startup complete.
```

> **Tip:** Run `docker compose up --build -d` to detach, then
> `docker compose logs -f` to tail all logs simultaneously.

---

## Step 2 — Mint a local JWT

The gateway enforces strict JWT validation (HS256, `iss`, `exp`, `iat`).
Use this one-liner to mint a valid 15-minute token against the local secret:

```sh
python3 - <<'EOF'
import base64, hmac, hashlib, json, time, struct

secret = b"local_dev_jwt_secret_do_not_use_in_production_00000000000000"
now = int(time.time())

header  = base64.urlsafe_b64encode(json.dumps({"alg":"HS256","typ":"JWT"}).encode()).rstrip(b"=")
payload = base64.urlsafe_b64encode(json.dumps({
    "sub": "local-test-user",
    "iss": "autoposter",
    "iat": now,
    "exp": now + 900,   # 15 minutes
}).encode()).rstrip(b"=")

sig_input = header + b"." + payload
sig = hmac.new(secret, sig_input, hashlib.sha256).digest()
sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=")

print((sig_input + b"." + sig_b64).decode())
EOF
```

Copy the printed token — you will use it as `$TOKEN` in the steps below.

```sh
TOKEN="<paste token here>"
```

---

## Step 3 — Submit a job to the Gateway

```sh
curl -s -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "idempotency_key": "e2e-test-001",
    "platforms": ["twitter"],
    "dry_run": true
  }' | python3 -m json.tool
```

**Expected response (202 Accepted):**

```json
{
  "job_id": "<deterministic-uuid>"
}
```

Note the `job_id` — you will poll it in Step 5.

```sh
JOB_ID="<paste job_id here>"
```

---

## Step 4 — Watch the Worker consume the job

In your compose log stream (or a separate terminal with
`docker compose logs -f worker`) you will see:

```
worker  | Processing job_id=<id> stream_id=<stream-id>
worker  | [pipeline] Starting dry-run for platforms: ['twitter']
worker  | Job <id> completed successfully.
```

The Worker follows the critical-path sequence:
idempotency guard → lock acquire → PROCESSING → pipeline → XACK → COMPLETE.

---

## Step 5 — Poll job status from the Gateway

```sh
curl -s http://localhost:8000/v1/jobs/$JOB_ID \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

**Expected response (200 OK):**

```json
{
  "job_id": "<id>",
  "status": "COMPLETE",
  "result": "{\"published\": true}",
  "completed_at": "<iso-timestamp>"
}
```

---

## Step 6 — Inspect Redis directly (optional)

```sh
docker compose exec redis redis-cli

# See the Stream entries:
127.0.0.1:6379> XRANGE autoposter:jobs - +

# See the job status hash:
127.0.0.1:6379> HGETALL job:<job_id>

# See the Consumer Group lag:
127.0.0.1:6379> XINFO GROUPS autoposter:jobs
```

---

## Step 7 — Test idempotency (replay protection)

Resubmit the exact same request with the same `idempotency_key`:

```sh
curl -s -X POST http://localhost:8000/v1/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"idempotency_key": "e2e-test-001", "platforms": ["twitter"], "dry_run": true}' \
  | python3 -m json.tool
```

**Expected response (409 Conflict):**

```json
{
  "error": "duplicate_request",
  "job_id": "<same deterministic uuid>",
  "detail": "This idempotency key was already accepted. Poll the job_id for status."
}
```

The same `job_id` is returned — the caller can safely poll it.

---

## Step 8 — Test rate limiting

The gateway enforces per-IP and per-JWT-subject sliding-window limits.
Fire more than 60 requests within 60 seconds to trigger a 429:

```sh
for i in $(seq 1 65); do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST http://localhost:8000/v1/jobs \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"idempotency_key\": \"ratelimit-test-$i\", \"dry_run\": true}"
done
```

You will see `202` for the first ~60 requests and `429` thereafter, with
a `Retry-After` header indicating when the window resets.

---

## Tear down

```sh
docker compose down          # stop containers, keep volumes
docker compose down -v       # stop containers AND delete volumes (full reset)
```

---

## Using a real API key (live test)

1. Open `docker-compose.yml` and set `DRY_RUN: "false"` for the `worker` service.
2. Replace the dummy `OPENAI_API_KEY` (and the relevant social platform keys)
   with real credentials.
3. Re-run `docker compose up --build` and submit a job.

For production-grade credential management, use the Credential Enclave
provisioner instead of plaintext env vars:

```sh
python3 scripts/provision_enclave.py --key OPENAI_API_KEY --value "sk-..."
```

Then set `ENCRYPTED_OPENAI_API_KEY` in `docker-compose.yml` and remove
the plaintext `OPENAI_API_KEY` entry.
