"""Each worker of a parallel run gets a data folder of its own, outside what git tracks.

The folder is named before the application is imported, by the conftest beside
the test package. It runs here in a fresh interpreter, so the choice is made
exactly as a worker makes it and this process's own folder is left alone.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
NAMING_CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"
_NAMED = (
    "import os, runpy, sys; runpy.run_path(sys.argv[1]); "
    "print(os.environ.get('LOCAL_LM_DATA_DIR', ''))"
)


def _named_folder(tmp_path: Path, *, worker: str | None, configured: str | None) -> str:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTEST_XDIST_WORKER", "LOCAL_LM_DATA_DIR"}
    }
    if worker is not None:
        environment["PYTEST_XDIST_WORKER"] = worker
    if configured is not None:
        environment["LOCAL_LM_DATA_DIR"] = configured
    named = subprocess.run(
        [sys.executable, "-c", _NAMED, str(NAMING_CONFTEST)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    return named.stdout.strip()


def _ignored(folder: str) -> bool:
    checked = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", f"{Path(folder).as_posix()}/"],
        cwd=REPOSITORY,
        capture_output=True,
        check=False,
    )
    assert checked.returncode in {0, 1}, checked.stderr
    return checked.returncode == 0


def test_a_worker_with_no_folder_named_writes_where_git_ignores_it(tmp_path: Path) -> None:
    folder = _named_folder(tmp_path, worker="gw3", configured=None)

    assert folder
    assert "gw3" in folder
    assert Path(folder).parts[0] != "data"
    assert _ignored(folder)


def test_a_worker_keeps_the_name_its_caller_chose(tmp_path: Path) -> None:
    held = str(tmp_path / "held")

    assert _named_folder(tmp_path, worker="gw3", configured=held) == f"{held}-gw3"


def test_a_single_process_keeps_whatever_it_was_given(tmp_path: Path) -> None:
    held = str(tmp_path / "held")

    assert _named_folder(tmp_path, worker=None, configured=held) == held
    assert _named_folder(tmp_path, worker=None, configured=None) == ""
