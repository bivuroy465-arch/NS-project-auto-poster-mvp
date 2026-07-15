"""Vercel serverless function: POST /api/gateway/enqueue

Thin HTTP adapter over GatewayRouter. All business logic lives in
src/gateway/gateway.py — this module only handles HTTP parsing and
response serialisation.

Returns:
    202 Accepted  — job enqueued, body is EnqueueResponse JSON.
    400 Bad Request — malformed JSON body.
    401 Unauthorized — JWT missing, expired, or invalid signature.
    409 Conflict — idempotency key already consumed.
    422 Unprocessable Entity — request body fails Pydantic validation.
    429 Too Many Requests — rate limit exceeded.
    500 Internal Server Error — unexpected error (Redis unavailable, etc.).
"""
from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler

from pydantic import ValidationError

from src.gateway import (
    EnqueueRequest,
    GatewayRouter,
    IdempotencyConflictError,
    RateLimitExceededError,
    TokenInvalidError,
)
from src.logging_setup import get_logger

log = get_logger("api.gateway.enqueue")

_router = GatewayRouter()


class handler(BaseHTTPRequestHandler):  # noqa: N801 — Vercel expects lowercase `handler`
    def do_POST(self) -> None:  # noqa: N802
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length) if content_length else b"{}"
            body = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError):
            self._send(400, {"error": "Invalid JSON body."})
            return

        auth_header = self.headers.get("Authorization", "")
        client_ip = (
            self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or self.client_address[0]
        )

        try:
            request = EnqueueRequest(**body)
        except ValidationError as exc:
            self._send(422, {"error": "Validation error.", "detail": exc.errors()})
            return

        try:
            response = _router.enqueue(auth_header, client_ip, request)
            self._send(202, response.model_dump())
        except TokenInvalidError as exc:
            self._send(401, {"error": str(exc)})
        except RateLimitExceededError as exc:
            self._send(
                429,
                {"error": str(exc)},
                extra_headers={"Retry-After": str(exc.window_seconds)},
            )
        except IdempotencyConflictError as exc:
            self._send(409, {"error": str(exc)})
        except Exception:
            log.error("Unhandled error in enqueue handler:\n%s", traceback.format_exc())
            self._send(500, {"error": "Internal server error."})

    def _send(
        self,
        status: int,
        body: dict,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for header, value in (extra_headers or {}).items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:  # suppress default HTTP server logs
        pass
