from __future__ import annotations

import ast
from pathlib import Path
from typing import Annotated

import pytest
from fastapi import FastAPI, File, Response, UploadFile
from httpx2 import ASGITransport, AsyncClient

from local_lm.config import Settings
from local_lm.main import create_app
from local_lm.security import (
    MAX_JSON_BODY_BYTES,
    MULTIPART_OVERHEAD_BYTES,
    SecurityHeadersMiddleware,
    SessionSecurity,
    UploadBodyLimitMiddleware,
)


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_all_application_subprocesses_use_an_explicit_environment() -> None:
    package_root = Path(__file__).resolve().parents[1] / "local_lm"
    subprocess_calls = {
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.run",
    }
    missing: list[str] = []

    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _qualified_name(node.func) not in subprocess_calls:
                continue
            if not any(keyword.arg == "env" for keyword in node.keywords):
                missing.append(f"{source_path.relative_to(package_root)}:{node.lineno}")

    assert missing == [], "subprocesses without an explicit safe environment: " + ", ".join(missing)


def test_public_preview_rejects_non_loopback_binding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LOCAL_LM_ALLOW_LAN", "true")
    settings = Settings(data_dir=tmp_path / "lan", host="0.0.0.0")

    with pytest.raises(ValueError, match="loopback binding only"):
        settings.prepare()


@pytest.mark.parametrize(
    "worker_url",
    [
        "https://127.0.0.1:12341",
        "http://example.com:12341",
        "http://127.0.0.1:12341/api",
        "http://user:secret@127.0.0.1:12341",
        "http://127.0.0.1:12341?token=secret",
    ],
)
def test_worker_urls_are_restricted_to_plain_loopback_origins(worker_url: str) -> None:
    with pytest.raises(ValueError, match="worker URLs"):
        Settings(llama_url=worker_url)


def test_worker_url_normalizes_a_trailing_slash() -> None:
    settings = Settings(
        llama_url="http://localhost:12341/",
        comfy_url="http://[::1]:8188/",
    )

    assert settings.llama_url == "http://localhost:12341"
    assert settings.comfy_url == "http://[::1]:8188"


async def test_non_dev_api_requires_cookie_and_csrf(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "secure", dev=False)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (await client.get("/api/projects")).status_code == 401
            session = await client.post("/api/session")
            assert session.status_code == 200
            csrf = session.json()["csrf_token"]
            assert session.json()["event_sequence"] == 0
            assert isinstance(session.json()["event_epoch"], str)
            assert session.json()["event_epoch"]
            assert "frame-ancestors 'none'" in session.headers["content-security-policy"]
            assert session.headers["referrer-policy"] == "no-referrer"
            assert (await client.get("/api/projects")).status_code == 200
            denied = await client.post("/api/projects", json={"name": "Denied"})
            assert denied.status_code == 403
            allowed = await client.post(
                "/api/projects",
                json={"name": "Allowed"},
                headers={"x-local-lm-csrf": csrf},
            )
            assert allowed.status_code == 201


async def test_browser_origins_are_limited_to_the_local_application(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "origins", dev=False, port=12340)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            rejected = await client.post(
                "/api/session",
                headers={"origin": "https://malicious.example"},
            )
            assert rejected.status_code == 403
            assert rejected.json()["detail"] == "untrusted browser origin"
            assert rejected.headers["x-frame-options"] == "DENY"
            assert "frame-ancestors 'none'" in rejected.headers["content-security-policy"]

            allowed = await client.post(
                "/api/session",
                headers={"origin": "http://127.0.0.1:12340"},
            )
            assert allowed.status_code == 200


def test_websocket_origin_policy_allows_non_browser_clients_and_local_ui(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    security = SessionSecurity(
        Settings(data_dir=tmp_path / "websocket-origins", dev=False, port=12340)
    )

    assert security._valid_origin(None)
    assert security._valid_origin("http://localhost:12340")
    assert security._valid_origin("http://[::1]:12340")
    assert not security._valid_origin("null")
    assert not security._valid_origin("https://127.0.0.1:12340")
    assert not security._valid_origin("http://127.0.0.1.attacker.test:12340")


async def test_json_request_body_limit_covers_streamed_bodies_without_content_length(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        data_dir=tmp_path / "json-body-limit",
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )
    app = create_app(settings)

    async def oversized_body():  # type: ignore[no-untyped-def]
        yield b'{"name":"'
        yield b"x" * MAX_JSON_BODY_BYTES
        yield b'"}'

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            session = await client.post("/api/session")
            response = await client.post(
                "/api/projects",
                content=oversized_body(),
                headers={
                    "content-type": "application/json",
                    "x-local-lm-csrf": session.json()["csrf_token"],
                },
            )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        f"JSON request body exceeds the {MAX_JSON_BODY_BYTES}-byte limit"
    )


async def test_upload_limit_rejects_streamed_multipart_before_endpoint_runs() -> None:
    app = FastAPI()
    endpoint_ran = False

    @app.post("/api/artifacts")
    async def consume_upload(file: Annotated[UploadFile, File()]) -> dict[str, str]:
        nonlocal endpoint_ran
        endpoint_ran = True
        return {"name": file.filename or ""}

    app.add_middleware(
        UploadBodyLimitMiddleware,
        artifact_max_bytes=8,
        project_max_bytes=16,
    )

    async def oversized_body():  # type: ignore[no-untyped-def]
        yield (
            b"--test\r\n"
            b'Content-Disposition: form-data; name="file"; filename="oversized.bin"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n"
        )
        yield b"x" * (MULTIPART_OVERHEAD_BYTES + 9)
        yield b"\r\n--test--\r\n"

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/artifacts",
            content=oversized_body(),
            headers={"content-type": "multipart/form-data; boundary=test"},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "upload request exceeds the 8-byte limit"
    assert endpoint_ran is False


async def test_security_headers_block_remote_active_content_and_preserve_stricter_csp() -> None:
    app = FastAPI()

    @app.get("/")
    async def index() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/artifact")
    async def artifact() -> Response:
        return Response(
            "image",
            headers={"Content-Security-Policy": "sandbox; default-src 'none'"},
        )

    app.add_middleware(SecurityHeadersMiddleware)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/")
        artifact_response = await client.get("/artifact")

    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "img-src 'self' data: blob:" in csp
    assert "connect-src 'self' ws://127.0.0.1:* ws://localhost:*" in csp
    assert "frame-ancestors 'none'" in csp
    assert "https:" not in csp
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"
    assert "camera=()" in response.headers["permissions-policy"]
    assert artifact_response.headers["content-security-policy"] == "sandbox; default-src 'none'"
