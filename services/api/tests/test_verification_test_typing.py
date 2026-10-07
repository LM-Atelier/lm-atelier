from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _fixture(directory: Path, *, broken: bool) -> None:
    package = directory / "services/api/local_lm"
    tests = directory / "services/api/tests"
    scripts = directory / "scripts"
    for path in (package, tests, scripts):
        path.mkdir(parents=True)
    (package / "__init__.py").write_text("def value() -> int:\n    return 1\n")
    (tests / "typing_helpers.py").write_text(
        "def passthrough(value: int) -> int:\n    return value\n"
    )
    source = (
        "def test_value(value):\n    return value\n"
        if broken
        else "from typing_helpers import passthrough\n"
        "from local_lm import value\n"
        "def test_value() -> None:\n    assert passthrough(value()) == 1\n"
    )
    (tests / "test_value.py").write_text(source)
    (scripts / "check.py").write_text("from local_lm import value\nassert value() == 1\n")
    (directory / "services/api/pyproject.toml").write_text("[tool.mypy]\nstrict = true\n")


def _mypy_tool() -> Path:
    name = "mypy.exe" if os.name == "nt" else "mypy"
    tool = Path(sys.executable).parent / name
    assert tool.is_file()
    return tool


def _bash_path(path: Path) -> str:
    value = path.as_posix()
    if os.name == "nt":
        assert len(path.drive) == 2 and path.drive.endswith(":")
        return "/" + path.drive[0].lower() + value[2:]
    return value


def _powershell_stage(directory: Path) -> subprocess.CompletedProcess[str]:
    source = (ROOT / "scripts/verify.ps1").read_text()
    start = source.find("    $PreviousMypyPath = ")
    if start < 0:
        start = source.index('    Invoke-Checked "Strict mypy"')
    end = source.index('    Invoke-Checked "Bandit high-severity scan"', start)
    stage = source[start:end]
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell is not None
    root = directory.as_posix().replace("'", "''")
    tool = _mypy_tool().as_posix().replace("'", "''")
    previous = (directory / "previous-import-path").as_posix().replace("'", "''")
    script = f"""
$ErrorActionPreference = 'Stop'
function Invoke-Checked {{
    param([string]$Label, [string]$FilePath, [string[]]$ArgumentList = @())
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {{ throw 'Type check refused' }}
}}
$RepositoryRoot = '{root}'
$Mypy = '{tool}'
$env:MYPYPATH = '{previous}'
$status = 0
try {{
{stage}
}}
catch {{ $status = 1 }}
if ($env:MYPYPATH -ne '{previous}') {{ exit 2 }}
exit $status
"""
    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _bash_stage(directory: Path) -> subprocess.CompletedProcess[str]:
    source = (ROOT / "scripts/verify.sh").read_text()
    start = source.index('run_checked "Strict mypy"')
    end = source.index('run_checked "Bandit high-severity scan"', start)
    stage = source[start:end].rstrip()
    shell = shutil.which("bash")
    if os.name == "nt":
        installed = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
        if installed.is_file():
            shell = str(installed)
    assert shell is not None
    tools = directory / "shell-tools"
    tools.mkdir()
    wrapper = tools / "mypy"
    wrapper.write_text(
        "#!/usr/bin/env bash\nexec " + shlex.quote(_mypy_tool().as_posix()) + ' "$@"\n'
    )
    wrapper.chmod(0o700)
    previous = (directory / "previous-import-path").as_posix()
    script = f"""
root={shlex.quote(_bash_path(directory))}
python_tools={shlex.quote(_bash_path(tools))}
export MYPYPATH={shlex.quote(previous)}
run_checked() {{ shift; "$@"; }}
status=0
{stage} || status=$?
[[ "$MYPYPATH" == {shlex.quote(previous)} ]] || exit 2
exit "$status"
"""
    return subprocess.run(
        [shell, "-c", script],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.mark.parametrize("shell", ["powershell", "bash"])
def test_verification_typing_commands_reject_untyped_test_sources(
    tmp_path: Path, shell: str
) -> None:
    _fixture(tmp_path, broken=True)
    run = _powershell_stage if shell == "powershell" else _bash_stage
    result = run(tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "no-untyped-def" in result.stdout
    assert "test_value.py" in result.stdout


@pytest.mark.parametrize("shell", ["powershell", "bash"])
def test_verification_typing_commands_accept_typed_test_helper_imports(
    tmp_path: Path, shell: str
) -> None:
    _fixture(tmp_path, broken=False)
    run = _powershell_stage if shell == "powershell" else _bash_stage
    result = run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Success: no issues found" in result.stdout
