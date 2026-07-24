from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from . import __version__
from .runtime_config import configure_persisted_runtime

_FALSE_VALUES = {"0", "false", "no", "off"}


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


def configure_desktop_environment() -> None:
    os.environ.setdefault("LOCAL_LM_DATA_DIR", str(default_data_dir()))
    configure_persisted_runtime(Path(os.environ["LOCAL_LM_DATA_DIR"]))
    os.environ.setdefault("LOCAL_LM_DEV", "false")
    if web_dir := bundled_web_dir():
        os.environ.setdefault("LOCAL_LM_WEB_DIST_DIR", str(web_dir))


def _health_matches(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/ready", timeout=2) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return payload.get("version") == __version__


def _browser_enabled() -> bool:
    value = os.environ.get("LOCAL_LM_OPEN_BROWSER", "true")
    return value.strip().lower() not in _FALSE_VALUES


def _open_browser_when_ready(url: str, timeout_seconds: float = 30) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _health_matches(url):
            webbrowser.open(url)
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
    configure_desktop_environment()
    if "--download-worker" in sys.argv[1:]:
        from .download_worker import main as download_worker_main

        return download_worker_main()
    if "--runtime-self-test" in sys.argv[1:]:
        from .credentials import CredentialStore

        credential_state = CredentialStore().state()
        print(
            json.dumps(
                {
                    "version": __version__,
                    "credential_vault_available": credential_state.vault_available,
                }
            )
        )
        if sys.platform == "win32" and not credential_state.vault_available:
            return 1
        return 0

    from .config import get_settings

    settings = get_settings()
    url = f"http://{settings.host}:{settings.port}"
    if _health_matches(url):
        if _browser_enabled():
            webbrowser.open(url)
        return 0

    _set_console_title()
    if _browser_enabled():
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url,),
            name="open-lm-atelier",
            daemon=True,
        ).start()

    from .main import run

    run()
    return 0
