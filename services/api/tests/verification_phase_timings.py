"""Report cumulative setup, call and teardown costs without changing test execution."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest
from _pytest.terminal import TerminalReporter

_PHASES = ("setup", "call", "teardown")
_MODULE_NAME = re.compile(r"test_[a-zA-Z0-9_]+\.py")
_PLUGIN_NAME = "api-verification-phase-timings"


@dataclass
class PhaseTotal:
    seconds: float = 0.0
    count: int = 0


class VerificationPhaseTimings:
    def __init__(self) -> None:
        self._modules: dict[str, dict[str, PhaseTotal]] = {}

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when not in _PHASES:
            return
        module = report.nodeid.split("::", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
        if _MODULE_NAME.fullmatch(module) is None:
            module = "other"
        phases = self._modules.get(module)
        if phases is None:
            phases = {name: PhaseTotal() for name in _PHASES}
            self._modules[module] = phases
        phase = phases[report.when]
        phase.seconds += report.duration
        phase.count += 1

    def pytest_terminal_summary(self, terminalreporter: TerminalReporter) -> None:
        if not self._modules:
            return
        terminalreporter.write_sep("=", "API phase costs (summed elapsed; workers overlap)")
        totals = {
            name: PhaseTotal(
                seconds=sum(phases[name].seconds for phases in self._modules.values()),
                count=sum(phases[name].count for phases in self._modules.values()),
            )
            for name in _PHASES
        }
        terminalreporter.write_line(
            "All reported phases: "
            + "; ".join(
                f"{name} {total.seconds:.2f}s / {total.count} reports"
                for name, total in totals.items()
            )
        )
        terminalreporter.write_line(
            f"{'module':48} {'setup_s':>10} {'call_s':>10} {'teardown_s':>10} "
            f"{'total_s':>10} {'setup_n':>8} {'call_n':>8} {'teardown_n':>10}"
        )
        ranked = sorted(
            self._modules.items(),
            key=lambda item: (-sum(phase.seconds for phase in item[1].values()), item[0]),
        )
        for module, phases in ranked[:20]:
            setup, call, teardown = (phases[name] for name in _PHASES)
            total = setup.seconds + call.seconds + teardown.seconds
            terminalreporter.write_line(
                f"{module:48} {setup.seconds:10.2f} {call.seconds:10.2f} "
                f"{teardown.seconds:10.2f} {total:10.2f} "
                f"{setup.count:8} {call.count:8} {teardown.count:10}"
            )


def register_verification_phase_timings(config: pytest.Config) -> None:
    if not hasattr(config, "workerinput") and not config.pluginmanager.hasplugin(_PLUGIN_NAME):
        config.pluginmanager.register(VerificationPhaseTimings(), _PLUGIN_NAME)
