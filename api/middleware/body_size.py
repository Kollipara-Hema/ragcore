"""
ASGI body-size limit for ingest routes.

Pure ASGI middleware (not BaseHTTPMiddleware — BaseHTTPMiddleware itself
buffers the request body and would defeat the entire point of this guard).
Sits at the wire layer so oversized payloads are rejected before any
downstream component (Starlette's multipart parser, pydantic body binding,
rate-limiter, auth, handler) reads a single body byte.

Two-stage check:
  1. Content-Length header — single integer compare, zero body read.
  2. Wrapped receive() — bounds actual bytes for missing/lying
     Content-Length (e.g. chunked transfer-encoding). Total in-memory peak
     is bounded by max_bytes + one ASGI chunk (typically 16-64 KB on uvicorn).

This middleware must be registered as the OUTERMOST middleware. Its whole
value is being cheaper than anything it protects; any inward move defeats
that. See the onion-order comment in api/main.py at the registration site.

Returns 413 for oversized bodies, 400 for malformed Content-Length headers.
"""
from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyTooLarge(Exception):
    """Internal sentinel raised by limited_receive when the running byte
    counter exceeds the cap. Caught in __call__ to send the 413."""


class BodySizeLimitMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int,
        protected_prefixes: tuple[str, ...],
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.protected_prefixes = protected_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._is_protected(scope.get("path", "")):
            await self.app(scope, receive, send)
            return

        # Stage 1: Content-Length header check (zero body read).
        cl_raw = self._get_content_length(scope)
        if cl_raw is not None:
            try:
                cl = int(cl_raw)
            except ValueError:
                await self._send_error(
                    send, 400, "Malformed Content-Length header.",
                )
                return
            if cl > self.max_bytes:
                await self._send_error(
                    send, 413,
                    f"Request body exceeds {self.max_bytes} bytes.",
                )
                return

        # Stage 2: bound actual bytes regardless of Content-Length presence
        # or accuracy. Wrapped receive() counts every http.request body byte;
        # an http.request stream that exceeds the cap raises a sentinel
        # caught below to send the 413.
        total = 0

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        # Track whether the downstream app has started a response. If our
        # sentinel fires AFTER the app sent http.response.start, ASGI forbids
        # sending a second one — we swallow and let the connection finish.
        # For ingest handlers (which buffer everything before responding),
        # this branch isn't expected to fire.
        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._send_error(
                    send, 413,
                    f"Request body exceeds {self.max_bytes} bytes.",
                )

    def _is_protected(self, path: str) -> bool:
        return any(
            path == p or path.startswith(p + "/")
            for p in self.protected_prefixes
        )

    @staticmethod
    def _get_content_length(scope: Scope) -> bytes | None:
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                return value
        return None

    @staticmethod
    async def _send_error(send: Send, status: int, detail: str) -> None:
        body = json.dumps({"error": detail}).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        })
