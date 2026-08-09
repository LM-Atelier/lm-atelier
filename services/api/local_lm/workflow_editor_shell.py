from __future__ import annotations

import hashlib
import hmac
import ipaddress
from html import escape
from pathlib import Path
from urllib.parse import urlparse

from .comfy_editor_bridge import ComfyEditorBridgeError

MAX_SHELL_ASSET_BYTES = 128 * 1024
SHELL_SCRIPT_ROUTE = "/api/workflow-editor/shell.js"
SHELL_STYLE_ROUTE = "/api/workflow-editor/shell.css"

_ASSET_DIRECTORY = Path(__file__).with_name("workflow_editor_shell_assets")
_ASSET_HASHES = {
    "shell.css": "ff22be620f20b119c4d89e024562bf3ae4720ed169c4089216ac6bea5a923d3b",
    "shell.js": "e2b20ce9bb636625d74b60f4782349d0f95a0fb8224ccd8967570f37c7962f55",
}


def workflow_editor_shell_document(comfy_url: str) -> str:
    origin = workflow_editor_comfy_origin(comfy_url)
    source = escape(f"{origin}/", quote=True)
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            "  <title>LM Atelier Workflow Editor</title>",
            f'  <link rel="stylesheet" href="{SHELL_STYLE_ROUTE}">',
            "</head>",
            "<body>",
            '  <p id="workflow-editor-status" role="status">Connecting to ComfyUI...</p>',
            (
                '  <iframe id="workflow-editor-frame" title="ComfyUI workflow editor" '
                f'src="{source}" referrerpolicy="no-referrer" '
                'sandbox="allow-downloads allow-forms allow-modals allow-same-origin '
                'allow-scripts"></iframe>'
            ),
            f'  <script src="{SHELL_SCRIPT_ROUTE}" defer></script>',
            "</body>",
            "</html>",
            "",
        )
    )


def workflow_editor_shell_csp(comfy_url: str) -> str:
    origin = workflow_editor_comfy_origin(comfy_url)
    return (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        f"frame-src {origin}; connect-src 'none'; img-src 'none'; "
        "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


def workflow_editor_comfy_origin(comfy_url: str) -> str:
    parsed = urlparse(comfy_url)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ComfyEditorBridgeError(
            "workflow-editor-origin-invalid",
            "The configured media runtime origin is invalid for native editing.",
        )
    try:
        loopback = (
            parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
        )
        port = parsed.port
    except ValueError:
        loopback = False
        port = None
    if not loopback or port is None:
        raise ComfyEditorBridgeError(
            "workflow-editor-origin-invalid",
            "The configured media runtime origin is invalid for native editing.",
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def workflow_editor_origins_conflict(comfy_url: str, shell_url: str) -> bool:
    comfy = urlparse(workflow_editor_comfy_origin(comfy_url))
    shell = urlparse(shell_url)
    try:
        shell_port = shell.port or (80 if shell.scheme == "http" else 443)
    except ValueError:
        return True
    return (
        comfy.scheme.casefold(),
        (comfy.hostname or "").casefold(),
        comfy.port,
    ) == (
        shell.scheme.casefold(),
        (shell.hostname or "").casefold(),
        shell_port,
    )


def workflow_editor_shell_asset(name: str) -> bytes:
    expected = _ASSET_HASHES.get(name)
    if expected is None:
        raise ComfyEditorBridgeError(
            "workflow-editor-shell-asset-invalid",
            "The workflow editor shell asset is unavailable.",
        )
    path = _ASSET_DIRECTORY / name
    try:
        with path.open("rb") as handle:
            content = handle.read(MAX_SHELL_ASSET_BYTES + 1)
    except OSError as exc:
        raise ComfyEditorBridgeError(
            "workflow-editor-shell-integrity-failed",
            "The bundled workflow editor shell failed its integrity check.",
        ) from exc
    if len(content) > MAX_SHELL_ASSET_BYTES or not hmac.compare_digest(
        hashlib.sha256(content).hexdigest(), expected
    ):
        raise ComfyEditorBridgeError(
            "workflow-editor-shell-integrity-failed",
            "The bundled workflow editor shell failed its integrity check.",
        )
    return content
