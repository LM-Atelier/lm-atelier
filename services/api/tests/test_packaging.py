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
