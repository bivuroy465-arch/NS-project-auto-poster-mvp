# AI Social Media Auto-Poster

Post every day. Write nothing. The system picks a fresh topic, writes
platform-specific posts, generates an anime-style image, runs every post
through a quality gate, publishes to Twitter / LinkedIn / Facebook, logs to
Google Sheets, and alerts you on failures — all on a daily GitLab CI schedule.

## How it works

```
validate -> read recent topics -> pick unique topic -> image (fallback chain)
         -> write post -> quality gate (rules + AI editor) -> publish -> log
```

Providers are plug-in style. Choose them in `config.yaml`; no code changes needed.

- **Text:** `openai`, `claude`, `gemini`, `groq`
- **Image:** `openai` (DALL-E 3), `stability` (anime-friendly)
- **Platforms:** `twitter`, `linkedin`, `facebook`

## Production features

- Structured logging and retry with exponential backoff on every API call
- Per-platform fault isolation: one failure won't stop the others
- Config + environment validation at startup
- Topic **duplicate guard** (reads recent topics from Google Sheets)
- **Image fallback chain** with graceful text-only degradation
- **Quality gate** (rule layer + AI editor) with a circuit breaker
- **Failure alerting** via a fail-safe webhook notifier
- **Dry-run mode** to test content generation without publishing

## Quality Gate (AI Editor)

Every post must pass before it is published.

1. **Rule layer** (fast, deterministic, platform-aware): rejects empty content,
   posts over the platform character limit, banned words, placeholders
   (e.g. `[insert link]`), AI-isms, and Twitter posts with no hashtag.
2. **AI editor** (optional): a ruthless editor prompt sent to a fast/cheap
   provider (Groq by default) that rejects robotic tone, hallucinations,
   AI-isms and placeholders, replying `PASS` or `FAIL: <reason>`.

If a post is rejected, it is regenerated up to `max_attempts`. After that the
**circuit breaker** trips: the platform is skipped and an alert is sent
(`Quality Gate rejected content N times for topic X. Last Reason: Y...`).

```yaml
quality_gate:
  enabled: true
  editor_provider: groq        # any registered text provider
  editor_model: llama-3.3-70b-versatile
  max_attempts: 3
  banned_words: []
```

Set `enabled: false` to publish raw output (rule layer still runs).

## Image Fallback Chain

Image providers are tried in order. If one fails, the exact reason is logged, a
non-fatal alert is sent, and the next provider is tried. If all fail, an alert
is sent and the post is published **text-only** (the run never crashes).

```yaml
image_providers:    # tried in order
  - openai
  - stability
image_model: dall-e-3
image_style: anime
```

Backward compatible: a single `image_provider: openai` string still works.

## Failure Alerting (Notifier)

A fail-safe webhook notifier sends alerts on platform failures, quality-gate
skips, image total-failure, fatal config errors, and CI job crashes
(via `after_script`). It never raises and is a no-op if unconfigured.

Set `ALERT_WEBHOOK_URL` (and `TELEGRAM_CHAT_ID` for Telegram). The payload shape
is auto-detected:

- **Telegram:** `https://api.telegram.org/bot<TOKEN>/sendMessage` + `TELEGRAM_CHAT_ID`
- **Discord:** `https://discord.com/api/webhooks/<id>/<token>`
- **Slack / generic:** any URL accepting `{"text": ...}`

## Setup

1. **API keys** — add the values from `.env.example` as GitLab CI/CD variables
   (Settings > CI/CD > Variables). Mark secrets as *Masked*.
2. **Choose providers** — edit `config.yaml`.
3. **First test (recommended)** — set `dry_run: true` in `config.yaml`
   (or CI/CD variable `DRY_RUN=true`) and run a manual pipeline.
4. **Go live** — set `dry_run: false`, then create a schedule
   (Build > Pipeline schedules). Example cron: `0 22 * * *` (22:00 UTC daily).

## CI pipeline

- `lint` (ruff), `unit_test` (pytest), `smoke_test` run on every push / MR.
- `auto_post` runs on schedules and manual web runs, and alerts on failure.

## Local testing

```bash
pip install -r requirements-dev.txt
cp .env.example .env   # fill in your keys
DRY_RUN=true python -m src.main   # generate only, no publishing
pytest -q                          # run the full test suite
```

## Add a new AI provider

1. Create a file under `src/providers/text/` (or `image/`) implementing the
   `base.py` interface (with `@with_retry()` on the API call).
2. Register it in the matching `factory.py`.
3. Set its name in `config.yaml`. Done.
