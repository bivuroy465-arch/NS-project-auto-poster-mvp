"""Vercel serverless function: GET /api/gateway/status/[job_id]

Fetches the current processing status of a job from the Redis Hash.

Returns:
    200 OK     — job found, body is JobStatusResponse JSON.
    401        — JWT invalid or missing.
    404        — job_id not found (never enqueued or expired).
    500        — unexpected error.
"""
from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

from src.gateway import GatewayRouter, TokenInvalidError
from src.logging_setup import get_logger

log = get_logger("api.gateway.status")

_router = GatewayRouter()


class handler(BaseHTTPRequestHandler):  # noqa: N801
    def do_GET(self) -> None:  # noqa: N802
        # Extract job_id from path: /api/gateway/status/<job_id>
        path = urlparse(self.path).path
        parts = [p for p in path.split("/") if p]
        job_id = parts[-1] if parts else ""

        auth_header = self.headers.get("Authorization", "")

        try:
            result = _router.status(auth_header, job_id)
        except TokenInvalidError as exc:
            self._send(401, {"error": str(exc)})
            return
        except Exception:
            log.error("Unhandled error in status handler:\n%s", traceback.format_exc())
            self._send(500, {"error": "Internal server error."})
            return

        if result is None:
            self._send(404, {"error": f"Job {job_id!r} not found."})
            return

        self._send(200, result.model_dump())

    def _send(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        pass
