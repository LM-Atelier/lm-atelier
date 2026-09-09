from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from . import __version__
from .instance_identity import (
    INSTANCE_ID_HEADER,
    InstanceIdentityError,
    load_or_create_instance_identity,
)
from .instance_lock import DataDirectoryLockError
from .runtime_config import (
    RUNTIME_ENV_KEYS,
    RuntimeConfigError,
    configure_persisted_runtime,
    persist_runtime_values,
)

_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class InstanceProbe:
    state: Literal["owned", "unavailable", "conflict"]
    reason: str = ""


def default_data_dir() -> Path:
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "LMAtelier" / "data"
        return Path.home() / "AppData" / "Local" / "LMAtelier" / "data"
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "lm-atelier"
    return Path.home() / ".local" / "share" / "lm-atelier"


def bundled_web_dir() -> Path | None:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return None
    candidate = Path(str(bundle_root)) / "web"
    return candidate if candidate.is_dir() else None


def _apply_source_environment_file() -> None:
    """Honor documented .env runtime overrides for non-frozen source launches."""

    if getattr(sys, "frozen", False):
        return
    from .config import Settings

    configured = Settings()
    configured_fields = configured.model_fields_set
    field_environment = {key.removeprefix("LOCAL_LM_").lower(): key for key in RUNTIME_ENV_KEYS}
    field_environment["data_dir"] = "LOCAL_LM_DATA_DIR"
    for field, environment_key in field_environment.items():
        if field not in configured_fields or os.environ.get(environment_key, "").strip():
            continue
        value = getattr(configured, field)
        if value is not None and str(value).strip():
            os.environ[environment_key] = str(value)


def configure_desktop_environment() -> None:
    _apply_source_environment_file()
    os.environ.setdefault("LOCAL_LM_DATA_DIR", str(default_data_dir()))
    explicit_chat_engine = bool(os.environ.get("LOCAL_LM_CHAT_ENGINE", "").strip())
    explicit_media_engine = bool(os.environ.get("LOCAL_LM_MEDIA_ENGINE", "").strip())
    configure_persisted_runtime(Path(os.environ["LOCAL_LM_DATA_DIR"]))
    if not explicit_chat_engine and os.environ.get("LOCAL_LM_CHAT_ENGINE", "mock") == "mock":
        os.environ["LOCAL_LM_CHAT_ENGINE"] = "llama.cpp"
    if not explicit_media_engine and os.environ.get("LOCAL_LM_MEDIA_ENGINE", "mock") == "mock":
        os.environ["LOCAL_LM_MEDIA_ENGINE"] = "comfyui"
    persist_runtime_values(
        Path(os.environ["LOCAL_LM_DATA_DIR"]),
        {
            "LOCAL_LM_CHAT_ENGINE": os.environ["LOCAL_LM_CHAT_ENGINE"],
            "LOCAL_LM_MEDIA_ENGINE": os.environ["LOCAL_LM_MEDIA_ENGINE"],
        },
    )
    os.environ.setdefault("LOCAL_LM_DEV", "false")
    if web_dir := bundled_web_dir():
        os.environ.setdefault("LOCAL_LM_WEB_DIST_DIR", str(web_dir))


def _probe_instance(url: str, expected_identity: str) -> InstanceProbe:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{url}/api/ready", timeout=2) as response:
            try:
                payload = json.load(response)
            except (OSError, ValueError):
                return InstanceProbe("conflict", "the service returned an invalid readiness reply")
            headers = getattr(response, "headers", {})
            identity = headers.get(INSTANCE_ID_HEADER)
    except urllib.error.HTTPError as exc:
        return InstanceProbe("conflict", f"the service returned HTTP {exc.code}")
    except (OSError, urllib.error.URLError):
        if _port_accepts_connections(url):
            return InstanceProbe("conflict", "another process is accepting connections")
        return InstanceProbe("unavailable")
    if not isinstance(payload, dict) or payload.get("version") != __version__:
        return InstanceProbe("conflict", "a different service or LM Atelier version is running")
    if identity != expected_identity:
        return InstanceProbe("conflict", "LM Atelier is using a different data folder")
    return InstanceProbe("owned")


def _health_matches(url: str, expected_identity: str) -> bool:
    return _probe_instance(url, expected_identity).state == "owned"


def _port_accepts_connections(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.port:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=0.25):
            return True
    except OSError:
        return False


def _port_conflict_message(
    url: str,
    data_dir: Path,
    reason: str,
) -> str:
    detail = f" ({reason})" if reason else ""
    return (
        f"LM Atelier did not open {url} because that address belongs to another "
        f"running service{detail}. Close the other LM Atelier/service and try again. "
        f"This launch is configured for the data folder {data_dir.resolve()}. "
        "Advanced users can select another loopback port with LOCAL_LM_PORT."
    )


def _browser_enabled() -> bool:
    value = os.environ.get("LOCAL_LM_OPEN_BROWSER", "true")
    return value.strip().lower() not in _FALSE_VALUES


def launch_url(base_url: str, arguments: Sequence[str]) -> str:
    """The URL the browser opens, honoring the installer's setup hand-off.

    The installer's final step runs the app with --first-run-setup so the
    person lands in setup - choosing models, watching real download progress,
    and preparing workers - before the workspace is ever shown. The flag
    travels as a query parameter the web app reads; everything it triggers is
    the same provisioning machinery the in-app wizard uses, not an
    installer-side copy of it.
    """

    if "--first-run-setup" in arguments:
        return f"{base_url}/?firstRunSetup=1"
    return base_url


def _open_browser_when_ready(
    url: str,
    expected_identity: str,
    data_dir: Path,
    timeout_seconds: float = 30,
    *,
    open_url: str | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        probe = _probe_instance(url, expected_identity)
        if probe.state == "owned":
            webbrowser.open(open_url or url)
            return
        if probe.state == "conflict":
            print(_port_conflict_message(url, data_dir, probe.reason), file=sys.stderr)
            return
        time.sleep(0.25)


def _set_console_title() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleTitleW("LM Atelier")
    except (AttributeError, OSError):
        return


def main() -> int:
    try:
        configure_desktop_environment()
    except RuntimeConfigError as exc:
        # A redirected state folder is refused rather than written through, and
        # this is the first thing that touches it - so without this the refusal
        # would arrive as a traceback at startup instead of the same clean
        # message and exit code the sibling containment refusal already uses.
        print(
            f"LM Atelier could not establish ownership of its data folder: {exc}.",
            file=sys.stderr,
        )
        return 2
    if "--download-worker" in sys.argv[1:]:
        from .download_worker import main as download_worker_main

        return download_worker_main()
    if "--runtime-self-test" in sys.argv[1:]:
        from .credentials import CredentialStore
        from .runtime_provisioning import default_engine_manifest_path

        credential_vault_available = CredentialStore().vault_available()
        print(
            json.dumps(
                {
                    "version": __version__,
                    "credential_vault_available": credential_vault_available,
                    "chat_engine": os.environ["LOCAL_LM_CHAT_ENGINE"],
                    "media_engine": os.environ["LOCAL_LM_MEDIA_ENGINE"],
                    "engine_manifest_available": default_engine_manifest_path().is_file(),
                }
            )
        )
        if sys.platform == "win32" and not credential_vault_available:
            return 1
        return 0

    from .config import get_settings

    settings = get_settings()
    host = f"[{settings.host}]" if ":" in settings.host else settings.host
    url = f"http://{host}:{settings.port}"
    try:
        expected_identity = load_or_create_instance_identity(settings.data_dir)
    except InstanceIdentityError as exc:
        print(
            f"LM Atelier could not establish ownership of its data folder: {exc}.",
            file=sys.stderr,
        )
        return 2
    probe = _probe_instance(url, expected_identity)
    if probe.state == "owned":
        if _browser_enabled():
            webbrowser.open(launch_url(url, sys.argv[1:]))
        return 0
    if probe.state == "conflict":
        print(
            _port_conflict_message(url, settings.data_dir, probe.reason),
            file=sys.stderr,
        )
        return 2

    try:
        from .main import run
    except DataDirectoryLockError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _set_console_title()
    if _browser_enabled():
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, expected_identity, settings.data_dir),
            kwargs={"open_url": launch_url(url, sys.argv[1:])},
            name="open-lm-atelier",
            daemon=True,
        ).start()

    run()
    return 0
