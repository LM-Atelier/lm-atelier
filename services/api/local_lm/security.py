from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path

from fastapi import HTTPException, Request, Response, WebSocket, status

from .config import Settings

SESSION_COOKIE = "local_lm_session"
CSRF_HEADER = "x-local-lm-csrf"


def _write_private(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
    except OSError:
        if path.exists():
            return
        raise


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

    def validate_request(self, request: Request) -> None:
        if self.settings.dev and request.client and request.client.host == "testclient":
            return
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
        return self._valid_cookie(websocket.cookies.get(SESSION_COOKIE))
