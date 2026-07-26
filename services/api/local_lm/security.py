from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import HTTPException, Request, Response, WebSocket, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings

SESSION_COOKIE = "local_lm_session"
CSRF_HEADER = "x-local-lm-csrf"
MAX_JSON_BODY_BYTES = 4 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
SECURITY_HEADERS = {
    b"content-security-policy": (
        b"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        b"img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self'; "
        b"connect-src 'self' ws://127.0.0.1:* ws://localhost:* ws://[::1]:*; "
        b"worker-src 'self' blob:; object-src 'none'; base-uri 'none'; "
        b"form-action 'self'; frame-ancestors 'none'"
    ),
    b"cross-origin-opener-policy": b"same-origin",
    b"cross-origin-resource-policy": b"same-origin",
    b"permissions-policy": (
        b"camera=(), microphone=(), geolocation=(), payment=(), usb=(), serial=(), bluetooth=()"
    ),
    b"referrer-policy": b"no-referrer",
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
}


def _write_private(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
    except OSError:
        if path.exists():
            return
        raise


class JsonBodyLimitMiddleware:
    """Bound JSON bodies while leaving separately streamed upload endpoints intact."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_JSON_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or not str(scope.get("path", "")).startswith("/api")
            or not self._is_json(scope)
        ):
            await self.app(scope, receive, send)
            return

        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                pass

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        messages = self._replay(bytes(body))
        await self.app(scope, messages.__anext__, send)

    @staticmethod
    def _is_json(scope: Scope) -> bool:
        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        media_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
        return media_type == b"application/json" or media_type.endswith(b"+json")

    @staticmethod
    async def _replay(body: bytes) -> AsyncIterator[Message]:
        yield {"type": "http.request", "body": body, "more_body": False}
        while True:
            yield {"type": "http.disconnect"}

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": f"JSON request body exceeds the {self.max_bytes}-byte limit"},
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        )
        await response(scope, receive, send)


class UploadBodyLimitMiddleware:
    """Bound multipart requests before Starlette materializes uploaded files."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        artifact_max_bytes: int,
        project_max_bytes: int,
    ) -> None:
        self.app = app
        self.limits = {
            "/api/artifacts": artifact_max_bytes + MULTIPART_OVERHEAD_BYTES,
            "/api/projects/import": project_max_bytes + MULTIPART_OVERHEAD_BYTES,
        }

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self._limit(scope)
        if limit is None:
            await self.app(scope, receive, send)
            return

        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > limit:
                    await self._reject(scope, receive, send, limit)
                    return
            except ValueError:
                pass

        received = 0
        exceeded = False
        rejection_sent = False

        async def limited_receive() -> Message:
            nonlocal exceeded, received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _UploadLimitExceeded
            return message

        async def limited_send(message: Message) -> None:
            nonlocal rejection_sent
            if exceeded:
                if message["type"] == "http.response.start" and not rejection_sent:
                    rejection_sent = True
                    await self._reject(scope, receive, send, limit)
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except _UploadLimitExceeded:
            if not rejection_sent:
                await self._reject(scope, receive, send, limit)

    def _limit(self, scope: Scope) -> int | None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            return None
        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        media_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
        if media_type != b"multipart/form-data":
            return None
        return self.limits.get(str(scope.get("path", "")))

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        payload_limit = max(0, limit - MULTIPART_OVERHEAD_BYTES)
        response = JSONResponse(
            {"detail": f"upload request exceeds the {payload_limit}-byte limit"},
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        )
        await response(scope, receive, send)


class SecurityHeadersMiddleware:
    """Apply browser isolation headers without replacing stricter route headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {name.lower() for name, _value in headers}
                headers.extend(
                    (name, value)
                    for name, value in SECURITY_HEADERS.items()
                    if name not in existing
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class _UploadLimitExceeded(Exception):
    pass


class SessionSecurity:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.secret = self._load_or_create_secret(settings.session_secret_path)
        self.csrf_token = hmac.new(
            self.secret.encode(), b"local-lm-csrf-v1", hashlib.sha256
        ).hexdigest()

    @staticmethod
    def _load_or_create_secret(path: Path) -> str:
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if len(value) >= 32:
                return value
        value = secrets.token_urlsafe(48)
        _write_private(path, value)
        return value

    def issue_session(self, response: Response) -> str:
        response.set_cookie(
            SESSION_COOKIE,
            self.secret,
            httponly=True,
            secure=False,
            samesite="strict",
            path="/",
        )
        return self.csrf_token

    def _valid_cookie(self, cookie: str | None) -> bool:
        return bool(cookie and hmac.compare_digest(cookie, self.secret))

    def _valid_origin(self, origin: str | None) -> bool:
        if origin is None:
            return True
        allowed = {
            f"http://127.0.0.1:{self.settings.port}",
            f"http://localhost:{self.settings.port}",
            f"http://[::1]:{self.settings.port}",
        }
        if self.settings.dev:
            allowed.update(
                {
                    "http://127.0.0.1:5173",
                    "http://localhost:5173",
                    "http://[::1]:5173",
                }
            )
        return origin.lower() in allowed

    def validate_origin(self, origin: str | None) -> None:
        if not self._valid_origin(origin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="untrusted browser origin",
            )

    def validate_request(self, request: Request) -> None:
        if self.settings.dev and request.client and request.client.host == "testclient":
            return
        self.validate_origin(request.headers.get("origin"))
        if not self._valid_cookie(request.cookies.get(SESSION_COOKIE)):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="session required")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            csrf = request.headers.get(CSRF_HEADER)
            if not csrf or not hmac.compare_digest(csrf, self.csrf_token):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="CSRF check failed"
                )

    async def validate_websocket(self, websocket: WebSocket) -> bool:
        if self.settings.dev and websocket.client and websocket.client.host == "testclient":
            return True
        return self._valid_origin(websocket.headers.get("origin")) and self._valid_cookie(
            websocket.cookies.get(SESSION_COOKIE)
        )
