#!/usr/bin/env sh
# =============================================================================
# docker-entrypoint.sh — Processing Fleet container entrypoint
#
# Responsibilities (in order):
#   1. Validate that every required environment variable is present.
#      The script aborts with a non-zero exit code rather than starting with
#      a partial or broken configuration — fail loud, fail early.
#   2. Apply the credential-enclave database migration idempotently using the
#      bundled asyncpg-backed migration runner.  The migration SQL is safe to
#      re-run (all DDL uses IF NOT EXISTS / idempotent indexes).
#   3. Exec the Processing Fleet worker, replacing the shell process so the
#      worker receives SIGTERM directly (no double-signal delay).
#
# Security notes:
#   - Uses POSIX /bin/sh (not bash) for maximum portability on slim images.
#   - Secrets are never echoed; only their *presence* is checked.
#   - `exec` at the end ensures PID 1 is the Python process for correct
#     signal handling (SIGTERM → graceful shutdown) and zombie reaping.
# =============================================================================

set -eu

# ─── 1. Required-variable guard ──────────────────────────────────────────────
# Add or remove variable names from this list as the application evolves.
# Intentionally does NOT print values — only names of missing variables.

REQUIRED_VARS="
  ENCLAVE_MASTER_KEY
  UPSTASH_REDIS_REST_URL
  UPSTASH_REDIS_REST_TOKEN
"

_missing=""
for _var in $REQUIRED_VARS; do
    # POSIX-compatible indirect expansion check.
    eval "_val=\${${_var}:-}"
    if [ -z "$_val" ]; then
        _missing="${_missing} ${_var}"
    fi
done

if [ -n "$_missing" ]; then
    echo "[entrypoint] FATAL: the following required environment variables are not set:${_missing}" >&2
    echo "[entrypoint] Set them in your orchestrator (Docker Compose, Kubernetes Secret, etc.) and restart." >&2
    exit 1
fi

echo "[entrypoint] All required environment variables present."

# ─── 2. Database migration ───────────────────────────────────────────────────
# The migration runner is a thin Python script that:
#   a) Opens a connection pool using DATABASE_URL (asyncpg).
#   b) Reads migrations/001_credential_enclave.sql.
#   c) Runs it inside a transaction (the SQL itself uses BEGIN/COMMIT, so the
#      outer connection just executes the whole file in one shot).
#   d) Exits 0 on success (including "already applied" — DDL is idempotent).
#   e) Exits non-zero on connection error / SQL failure, which aborts startup.
#
# Skip migration if DATABASE_URL is unset (unit-test / Redis-only deployments).

if [ -n "${DATABASE_URL:-}" ]; then
    echo "[entrypoint] Applying database migrations..."
    python - <<'PYEOF'
import asyncio
import os
import pathlib
import sys


async def apply_migration() -> None:
    try:
        import asyncpg  # type: ignore[import]
    except ImportError:
        print("[entrypoint] asyncpg not installed — skipping migration.", file=sys.stderr)
        return

    db_url: str = os.environ["DATABASE_URL"]
    sql_path = pathlib.Path("/app/migrations/001_credential_enclave.sql")

    if not sql_path.exists():
        print(f"[entrypoint] Migration file not found: {sql_path}", file=sys.stderr)
        sys.exit(1)

    sql = sql_path.read_text()

    try:
        conn: asyncpg.Connection = await asyncpg.connect(db_url)
    except Exception as exc:  # noqa: BLE001
        print(f"[entrypoint] Cannot connect to database: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        await conn.execute(sql)
        print("[entrypoint] Migration applied successfully (or already up to date).")
    except Exception as exc:  # noqa: BLE001
        print(f"[entrypoint] Migration failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        await conn.close()


asyncio.run(apply_migration())
PYEOF
    echo "[entrypoint] Migration step complete."
else
    echo "[entrypoint] DATABASE_URL not set — skipping migration (Redis-only mode)."
fi

# ─── 3. Exec fleet worker ────────────────────────────────────────────────────
# `exec` replaces the shell with the Python process so it becomes PID 1.
# Kubernetes / Docker send SIGTERM to PID 1 on graceful shutdown; the worker's
# _shutdown() handler intercepts it and drains in-flight jobs before exiting.

echo "[entrypoint] Starting Processing Fleet worker..."
exec python -m src.fleet "$@"
