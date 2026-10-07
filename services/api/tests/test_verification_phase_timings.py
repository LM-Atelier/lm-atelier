from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest
from _pytest.terminal import TerminalReporter
from verification_phase_timings import (
    VerificationPhaseTimings,
    register_verification_phase_timings,
)

ROOT = Path(__file__).resolve().parents[3]


class Capture:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_sep(self, separator: str, title: str) -> None:
        self.lines.append(title)

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def report(
    module: str,
    phase: Literal["setup", "call", "teardown"],
    seconds: float,
    outcome: Literal["passed", "failed", "skipped"] = "passed",
) -> pytest.TestReport:
    return pytest.TestReport(
        nodeid=f"tests/{module}::case[paramA]",
        location=(module, 0, "case"),
        keywords={},
        outcome=outcome,
        longrepr=None,
        when=phase,
        duration=seconds,
    )


def summary(plugin: VerificationPhaseTimings) -> list[str]:
    capture = Capture()
    plugin.pytest_terminal_summary(cast(TerminalReporter, capture))
    return capture.lines


def rows(lines: list[str]) -> list[list[str]]:
    return [line.split() for line in lines if line.startswith(("test_", "other"))]


def test_cumulative_fixture_cost_ranks_above_an_individually_slower_call() -> None:
    plugin = VerificationPhaseTimings()
    values: tuple[tuple[Literal["setup", "call", "teardown"], float, float], ...] = (
        ("setup", 1.0, 2.0),
        ("call", 0.25, 0.5),
        ("teardown", 0.125, 0.375),
    )
    for phase, first, second in values:
        plugin.pytest_runtest_logreport(report("test_alpha.py", phase, first))
        plugin.pytest_runtest_logreport(report("test_alpha.py", phase, second))
    plugin.pytest_runtest_logreport(report("test_beta.py", "call", 3.0))
    lines = summary(plugin)
    assert rows(lines)[0] == ["test_alpha.py", "3.00", "0.75", "0.50", "4.25", "2", "2", "2"]
    assert rows(lines)[1][0] == "test_beta.py"
    assert "setup 3.00s / 2 reports; call 3.75s / 3 reports; teardown 0.50s / 2 reports" in lines[1]


def test_failed_calls_and_skipped_setups_still_contribute_their_actual_phases() -> None:
    plugin = VerificationPhaseTimings()
    plugin.pytest_runtest_logreport(report("test_failure.py", "setup", 0.5))
    plugin.pytest_runtest_logreport(report("test_failure.py", "call", 1.0, "failed"))
    plugin.pytest_runtest_logreport(report("test_failure.py", "teardown", 0.1))
    plugin.pytest_runtest_logreport(report("test_skip.py", "setup", 2.0, "skipped"))
    assert rows(summary(plugin)) == [
        ["test_skip.py", "2.00", "0.00", "0.00", "2.00", "1", "0", "0"],
        ["test_failure.py", "0.50", "1.00", "0.10", "1.60", "1", "1", "1"],
    ]


def test_diagnostics_omit_parameter_values_and_unrecognized_module_names() -> None:
    plugin = VerificationPhaseTimings()
    plugin.pytest_runtest_logreport(report("test_alpha.py", "call", 1.0))
    plugin.pytest_runtest_logreport(report("auxiliary.py", "call", 2.0))
    lines = summary(plugin)
    assert rows(lines)[0][0] == "other"
    assert "auxiliary.py" not in "\n".join(lines)
    assert "paramA" not in "\n".join(lines)


def test_module_ranking_is_bounded_and_equal_costs_sort_by_name() -> None:
    plugin = VerificationPhaseTimings()
    for number in reversed(range(21)):
        plugin.pytest_runtest_logreport(report(f"test_case_{number:02}.py", "call", 1.0))
    table = rows(summary(plugin))
    assert len(table) == 20
    assert table[0][0] == "test_case_00.py"
    assert table[-1][0] == "test_case_19.py"
    assert "call 21.00s / 21 reports" in summary(plugin)[1]


def test_an_empty_session_has_no_timing_summary() -> None:
    assert summary(VerificationPhaseTimings()) == []


def test_registration_counts_reports_only_at_the_controller_and_is_idempotent() -> None:
    manager = pytest.PytestPluginManager()
    initial_plugins = manager.get_plugins()
    config = cast(pytest.Config, SimpleNamespace(pluginmanager=manager))
    register_verification_phase_timings(config)
    register_verification_phase_timings(config)
    added_plugins = manager.get_plugins() - initial_plugins
    assert len(added_plugins) == 1
    assert isinstance(next(iter(added_plugins)), VerificationPhaseTimings)
    worker_manager = pytest.PytestPluginManager()
    initial_worker_plugins = worker_manager.get_plugins()
    worker = cast(pytest.Config, SimpleNamespace(pluginmanager=worker_manager, workerinput={}))
    register_verification_phase_timings(worker)
    assert worker_manager.get_plugins() == initial_worker_plugins


@pytest.mark.parametrize("distributed", [False, True])
def test_actual_api_collection_reports_each_phase_once_in_serial_and_distributed_runs(
    tmp_path: Path, distributed: bool
) -> None:
    environment = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "services/api"),
        PYTEST_ADDOPTS="",
        LOCAL_LM_DATA_DIR=str(tmp_path / "nested-data"),
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "--basetemp=" + str(tmp_path / "nested"),
        "services/api/tests/test_domain.py::"
        "test_elapsed_milliseconds_treats_naive_database_timestamp_as_utc",
        "services/api/tests/test_api_errors.py::test_error_codes_are_kebab_case_slugs",
    ]
    if distributed:
        command.extend(["-n", "2", "--dist", "loadfile"])
    result = subprocess.run(
        command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    table = rows(result.stdout.splitlines())
    assert {row[0] for row in table} == {"test_domain.py", "test_api_errors.py"}
    assert len(table) == 2
    assert all(row[-3:] == ["1", "1", "1"] for row in table)
    assert "2 passed" in result.stdout
