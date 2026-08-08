from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient

import local_lm.workflow_editor_shell as editor_shell
from local_lm.comfy_editor_bridge import (
    WORKFLOW_EDITOR_BRIDGE_PROTOCOL_VERSION,
    ComfyEditorBridgeError,
)
from local_lm.workflow_editor_sessions import WORKFLOW_EDITOR_PROTOCOL_VERSION
from local_lm.workflow_editor_shell import (
    workflow_editor_comfy_origin,
    workflow_editor_origins_conflict,
    workflow_editor_shell_asset,
    workflow_editor_shell_csp,
    workflow_editor_shell_document,
)


def test_shell_document_frames_only_the_configured_runtime() -> None:
    document = workflow_editor_shell_document("http://[::1]:8188")
    csp = workflow_editor_shell_csp("http://[::1]:8188")

    assert 'src="http://[::1]:8188/"' in document
    assert 'referrerpolicy="no-referrer"' in document
    assert "allow-top-navigation" not in document
    assert "allow-popups" not in document
    assert "frame-src http://[::1]:8188" in csp
    assert "frame-src *" not in csp
    assert "connect-src 'none'" in csp


@pytest.mark.parametrize(
    "value",
    (
        "https://127.0.0.1:8188",
        "http://example.com:8188",
        "http://127.0.0.1:8188/path",
        "http://user@127.0.0.1:8188",
        "http://127.0.0.1:8188?token=secret",
        "http://127.0.0.1",
    ),
)
def test_shell_refuses_any_origin_outside_plain_loopback_http(value: str) -> None:
    with pytest.raises(ComfyEditorBridgeError) as refused:
        workflow_editor_comfy_origin(value)

    assert refused.value.code == "workflow-editor-origin-invalid"


def test_shell_refuses_a_same_origin_media_runtime() -> None:
    assert workflow_editor_origins_conflict(
        "http://127.0.0.1:12340",
        "http://127.0.0.1:12340",
    )
    assert not workflow_editor_origins_conflict(
        "http://127.0.0.1:8188",
        "http://127.0.0.1:12340",
    )


def test_shell_and_session_protocols_match_the_pinned_bridge() -> None:
    assert WORKFLOW_EDITOR_PROTOCOL_VERSION == WORKFLOW_EDITOR_BRIDGE_PROTOCOL_VERSION == 2


def test_shell_assets_are_integrity_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    script = workflow_editor_shell_asset("shell.js")
    style = workflow_editor_shell_asset("shell.css")
    assert b"event.source === frame.contentWindow" in script
    assert b"event.source === openerWindow" in script
    assert b"event.origin !== comfyOrigin" in script
    assert b"event.origin !== window.location.origin" in script
    assert b"event.ports.length !== 1" in script
    assert b"frame.contentWindow.postMessage" in script
    assert b"localStorage" not in script
    assert b"location.search" not in script
    assert b"fetch(" not in script
    assert b"#workflow-editor-frame" in style

    monkeypatch.setitem(editor_shell._ASSET_HASHES, "shell.js", "0" * 64)
    with pytest.raises(ComfyEditorBridgeError) as refused:
        workflow_editor_shell_asset("shell.js")
    assert refused.value.code == "workflow-editor-shell-integrity-failed"

    monkeypatch.setitem(editor_shell._ASSET_HASHES, "shell.css", "0" * 64)
    with pytest.raises(ComfyEditorBridgeError) as refused_style:
        workflow_editor_shell_asset("shell.css")
    assert refused_style.value.code == "workflow-editor-shell-integrity-failed"


async def test_shell_route_preserves_strict_isolation_and_exposes_no_authority(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app.state.services.processes,
        "workflow_editor_runtime_identity",
        lambda: "runtime_authority_must_stay_server_side",
    )

    response = await client.get("/api/workflow-editor/shell")
    script = await client.get("/api/workflow-editor/shell.js")
    style = await client.get("/api/workflow-editor/shell.css")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["cross-origin-opener-policy"] == "same-origin"
    assert response.headers["cross-origin-resource-policy"] == "same-origin"
    assert response.headers["x-frame-options"] == "DENY"
    csp = response.headers["content-security-policy"]
    assert "frame-src http://127.0.0.1:8188" in csp
    assert "frame-ancestors 'none'" in csp
    assert "script-src 'self'" in csp
    assert "runtime_authority_must_stay_server_side" not in response.text
    assert "nonce" not in response.text.casefold()
    assert "receipt" not in response.text.casefold()
    assert script.status_code == 200
    assert script.headers["cache-control"] == "no-store"
    assert script.headers["content-type"].startswith("application/javascript")
    assert style.status_code == 200
    assert style.headers["cache-control"] == "no-store"
    assert style.headers["content-type"].startswith("text/css")


async def test_shell_route_fails_closed_until_the_verified_runtime_is_ready(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app.state.services.processes,
        "workflow_editor_runtime_identity",
        lambda: None,
    )

    response = await client.get("/api/workflow-editor/shell")

    assert response.status_code == 409
    assert response.json()["code"] == "workflow-editor-runtime-not-ready"


async def test_shell_route_refuses_same_origin_comfyui(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app.state.services.processes,
        "workflow_editor_runtime_identity",
        lambda: "runtime_one",
    )
    monkeypatch.setattr(app.state.services.settings, "comfy_url", "http://127.0.0.1:8188")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://127.0.0.1:8188") as client:
        await client.post("/api/session")
        response = await client.get("/api/workflow-editor/shell")

    assert response.status_code == 409
    assert response.json()["code"] == "workflow-editor-origin-conflict"


def test_shell_assets_are_packaged_beside_the_module() -> None:
    root = Path(editor_shell.__file__).with_name("workflow_editor_shell_assets")
    assert {path.name for path in root.iterdir()} == {"shell.css", "shell.js"}
