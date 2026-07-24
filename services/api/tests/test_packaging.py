from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from local_lm import __version__

ROOT = Path(__file__).resolve().parents[3]


def test_release_and_engine_manifests_are_pinned_and_versioned() -> None:
    release = json.loads((ROOT / "packaging/release-manifest.json").read_text())
    engines = json.loads((ROOT / "packaging/engines.json").read_text())

    assert release["version"] == __version__
    assert release["data_policy"]["rollback"]
    assert release["bundled_engines"] is False
    assert engines["engines"]["llama.cpp"]["pinned_release"] != "latest"
    assert engines["engines"]["comfyui"]["pinned_release"] != "latest"
    assert all(
        engine["certification"] == "hardware-pending" for engine in engines["engines"].values()
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Bash syntax is checked by Ubuntu CI")
def test_linux_release_scripts_pass_shell_syntax_check() -> None:
    scripts = [ROOT / "scripts/package.sh", *sorted((ROOT / "packaging/linux").glob("*.sh"))]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_installers_preserve_data_unless_purge_is_explicit() -> None:
    linux = (ROOT / "packaging/linux/uninstall.sh").read_text()
    windows = (ROOT / "packaging/windows/uninstall.ps1").read_text()

    assert "--purge-data" in linux
    assert "PurgeData" in windows
    assert "versions" in linux and "versions" in windows


def test_installed_launchers_avoid_relocated_console_scripts() -> None:
    linux = (ROOT / "packaging/linux/start-installed.sh").read_text()
    windows = (ROOT / "packaging/windows/start-local-lm.ps1").read_text()

    assert ".venv/bin/python" in linux
    assert '.venv\\Scripts\\python.exe"' in windows
    assert "from local_lm.main import run; run()" in linux
    assert "from local_lm.main import run; run()" in windows
    assert ".venv/bin/lm-atelier" not in linux
    assert ".venv\\Scripts\\lm-atelier.exe" not in windows


def test_windows_install_creates_and_removes_a_start_menu_launcher() -> None:
    installer = (ROOT / "packaging/windows/install.ps1").read_text()
    launcher = (ROOT / "packaging/windows/start-local-lm.ps1").read_text()
    uninstaller = (ROOT / "packaging/windows/uninstall.ps1").read_text()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert '[Environment]::GetFolderPath("Programs")' in installer
    assert '"LM Atelier.lnk"' in installer
    assert "CreateShortcut" in installer
    assert '$ApplicationLauncher = Join-Path $VersionRoot "LM Atelier.exe"' in installer
    assert "$Shortcut.TargetPath = $ApplicationLauncher" in installer
    assert "Windows Start menu" in installer
    assert "Invoke-WebRequest" in launcher
    assert "Start-Process $Url" in launcher
    assert '"LM Atelier.lnk"' in uninstaller
    assert r".\.venv\Scripts\lm-atelier.exe" in readme
    assert "Windows Start menu" in readme


@pytest.mark.skipif(sys.platform != "win32", reason="Native launchers build on Windows")
def test_windows_release_builds_top_level_native_launchers(tmp_path: Path) -> None:
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "packaging/windows/build-launchers.ps1"),
            "-OutputDirectory",
            str(tmp_path),
        ],
        check=True,
    )

    setup = tmp_path / "Setup LM Atelier.exe"
    application = tmp_path / "LM Atelier.exe"
    assert setup.read_bytes().startswith(b"MZ")
    assert application.read_bytes().startswith(b"MZ")
    assert not (tmp_path / "Start LM Atelier.exe").exists()


def test_release_workflow_packages_the_top_level_windows_applications() -> None:
    package = (ROOT / "scripts/package.ps1").read_text()
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "build-launchers.ps1" in package
    assert "Setup LM Atelier.exe" in readme
    assert "LM Atelier.exe" in readme
    assert "Start LM Atelier.exe" not in package
    assert "windows-2025" in workflow
    assert r".\scripts\package.ps1" in workflow
