from __future__ import annotations

import sys
from pathlib import Path

from local_lm import desktop
from local_lm.downloads import download_worker_command


def test_default_data_dir_uses_windows_local_app_data(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")

    assert desktop.default_data_dir() == Path(r"C:\Users\Tester\AppData\Local\LMAtelier\data")


def test_default_data_dir_uses_xdg_data_home(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/xdg-data")

    assert desktop.default_data_dir() == Path("/tmp/xdg-data/lm-atelier")


def test_download_worker_uses_frozen_executable_dispatch(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "/opt/lm-atelier/lm-atelier")

    assert download_worker_command() == ["/opt/lm-atelier/lm-atelier", "--download-worker"]
