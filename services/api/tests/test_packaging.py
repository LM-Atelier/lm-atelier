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
    assert release["prerequisites"]["python"] == "Bundled in official installers"
    assert engines["engines"]["llama.cpp"]["pinned_release"] != "latest"
    assert engines["engines"]["comfyui"]["pinned_release"] != "latest"
    assert all(
        engine["certification"] == "hardware-pending" for engine in engines["engines"].values()
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Bash syntax is checked by Ubuntu CI")
def test_linux_release_scripts_pass_shell_syntax_check() -> None:
    scripts = [
        ROOT / "scripts/package.sh",
        ROOT / "scripts/build-linux-installer.sh",
        *sorted((ROOT / "packaging/linux").glob("*.sh")),
    ]
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


def test_windows_installer_creates_start_menu_and_application_launchers() -> None:
    installer = (ROOT / "packaging/windows/LMAtelier.iss").read_text()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert r"DefaultDirName={localappdata}\Programs\LM Atelier" in installer
    assert 'Name: "{group}\\LM Atelier"' in installer
    assert 'Filename: "{app}\\{#MyAppExeName}"' in installer
    assert "LM Atelier terminal" in readme


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


def test_release_workflow_builds_self_contained_platform_installers() -> None:
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "build-windows-installer.ps1" in workflow
    assert "build-linux-installer.sh" in workflow
    assert "LM-Atelier-Setup-<version>-windows-x86_64.exe" in readme
    assert "LM-Atelier-Setup-<version>-linux-x86_64.run" in readme
    assert "windows-2025" in workflow
    assert "ubuntu-24.04" in workflow


def test_frozen_installer_contracts_are_explicit() -> None:
    spec = (ROOT / "packaging/LMAtelier.spec").read_text()
    windows = (ROOT / "packaging/windows/LMAtelier.iss").read_text()
    linux = (ROOT / "packaging/linux/self-extracting-installer.sh").read_text()
    linux_uninstall = (ROOT / "packaging/linux/frozen-uninstall.sh").read_text()

    assert '"apps" / "web" / "dist"' in spec
    assert '"local_lm/migrations"' in spec
    assert "PrivilegesRequired=lowest" in windows
    assert r"DefaultDirName={localappdata}\Programs\LM Atelier" in windows
    assert r'Filename: "{app}\{#MyAppExeName}"' in windows
    assert "__LM_ATELIER_PAYLOAD_BELOW__" in linux
    assert "$HOME/.local/bin/lm-atelier" in linux
    assert "--purge-data" in linux_uninstall
