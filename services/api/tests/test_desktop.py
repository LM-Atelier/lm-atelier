from __future__ import annotations

import sys
from pathlib import Path

from local_lm import desktop
from local_lm.downloads import download_worker_command
from local_lm.runtime_config import configure_persisted_runtime, runtime_config_path


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


def test_desktop_console_script_uses_persistence_aware_launcher() -> None:
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert 'lm-atelier = "local_lm.desktop:main"' in project.read_text(encoding="utf-8")


def test_runtime_configuration_survives_desktop_relaunch(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    first_launch = {
        "LOCAL_LM_CHAT_ENGINE": "llama.cpp",
        "LOCAL_LM_LLAMA_EXECUTABLE": str(tmp_path / "llama-server"),
        "LOCAL_LM_MEDIA_ENGINE": "comfyui",
        "LOCAL_LM_COMFY_EXECUTABLE": str(tmp_path / "python"),
        "LOCAL_LM_COMFY_DIRECTORY": str(tmp_path / "ComfyUI"),
    }

    configure_persisted_runtime(data_dir, first_launch)
    second_launch: dict[str, str] = {}
    configure_persisted_runtime(data_dir, second_launch)

    assert second_launch == first_launch
    payload = runtime_config_path(data_dir).read_text(encoding="utf-8")
    assert "TOKEN" not in payload


def test_explicit_runtime_configuration_overrides_saved_value(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    configure_persisted_runtime(
        data_dir,
        {"LOCAL_LM_LLAMA_EXECUTABLE": str(tmp_path / "old-llama-server")},
    )
    environment = {"LOCAL_LM_LLAMA_EXECUTABLE": str(tmp_path / "new-llama-server")}

    configure_persisted_runtime(data_dir, environment)

    assert environment["LOCAL_LM_LLAMA_EXECUTABLE"].endswith("new-llama-server")
    assert "new-llama-server" in runtime_config_path(data_dir).read_text(encoding="utf-8")
