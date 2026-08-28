"""Tests for the bounded startup-stage watchdog.

Written against the failure the watchdog exists to diagnose rather than against
its happy path. A start that hangs never reaches the line that would report a
duration, so the only useful evidence is emitted while the stage is still
stuck; the central test below deadlocks rather than passes if the
implementation ever regresses to timing a stage and reporting afterwards.
"""

from __future__ import annotations

import logging
import sys
import threading
import time

import pytest

from local_lm import main
from local_lm.main import _startup_stage


def test_a_stuck_stage_is_named_while_it_is_still_stuck(capsys) -> None:
    """The whole point: evidence arrives during the hang, not after it.

    The stage blocks on `release`, and `release` is only set once the watchdog
    has already announced the stage. An implementation that reported on exit
    would never announce, so `announced.wait` would time out and this test
    would fail rather than quietly pass.
    """
    release = threading.Event()
    announced = threading.Event()

    class Watcher(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "still running after" in record.getMessage():
                announced.set()

    watcher = Watcher()
    logging.getLogger("local_lm").addHandler(watcher)
    try:
        with _startup_stage("wedged-stage", warn_after=0.05):
            assert announced.wait(timeout=10), "no notice while the stage was still running"
            release.set()
    finally:
        logging.getLogger("local_lm").removeHandler(watcher)

    assert release.is_set()
    err = capsys.readouterr().err
    assert "wedged-stage" in err
    assert "still running after" in err


def test_the_notice_repeats_so_a_wait_is_visibly_growing(capsys) -> None:
    """One line could be read as a step that completed; repetition cannot."""
    seen = threading.Semaphore(0)

    class Counter(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "still running after" in record.getMessage():
                seen.release()

    counter = Counter()
    logging.getLogger("local_lm").addHandler(counter)
    try:
        with _startup_stage("slow-stage", warn_after=0.05):
            assert seen.acquire(timeout=10)
            assert seen.acquire(timeout=10)
    finally:
        logging.getLogger("local_lm").removeHandler(counter)

    assert capsys.readouterr().err.count("slow-stage") >= 2


def test_a_healthy_stage_says_nothing_at_all(capsys) -> None:
    """Bounded means silent when there is nothing wrong to report."""
    with _startup_stage("quick-stage", warn_after=30.0):
        pass
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_the_notice_reaches_stderr_without_any_logging_handler(capsys) -> None:
    """The stages that hang worst run before api.log exists.

    Five stages run before the lifespan installs the file handler, so a stage
    that reports only through logging is invisible for exactly the window this
    exists to cover. Emitting with the local_lm logger disabled proves stderr
    carries it on its own.
    """
    local = logging.getLogger("local_lm")
    previous = local.disabled
    local.disabled = True
    try:
        with _startup_stage("pre-logging-stage", warn_after=0.05):
            deadline = threading.Event()
            deadline.wait(0.3)
    finally:
        local.disabled = previous

    err = capsys.readouterr().err
    assert "pre-logging-stage" in err
    assert "local_lm:" in err


def test_a_stage_that_raises_still_stops_its_watchdog() -> None:
    """A failed stage must not leave a thread announcing it forever."""
    before = {t.name for t in threading.enumerate()}
    with pytest.raises(RuntimeError), _startup_stage("failing-stage", warn_after=0.05):
        raise RuntimeError("stage failed")
    for _ in range(100):
        if not any(t.name == "startup-stage-failing-stage" for t in threading.enumerate()):
            break
        threading.Event().wait(0.05)
    still = {t.name for t in threading.enumerate()} - before
    assert "startup-stage-failing-stage" not in still


def test_the_watchdog_thread_is_a_daemon_and_cannot_hold_shutdown() -> None:
    started: list[threading.Thread] = []
    real = threading.Thread

    class Recording(real):  # type: ignore[misc,valid-type]
        def start(self) -> None:
            started.append(self)
            super().start()

    threading.Thread = Recording  # type: ignore[misc]
    try:
        with _startup_stage("daemon-check", warn_after=30.0):
            pass
    finally:
        threading.Thread = real  # type: ignore[misc]

    assert started, "no watchdog thread was created"
    assert all(thread.daemon for thread in started)


def test_no_startup_statement_runs_outside_a_stage() -> None:
    """A stage nobody wrapped is a stage a hang can hide in.

    The point of this instrumentation is that every part of startup names
    itself. An unwrapped call in create_app would be exactly the blind spot
    that made a seventy-minute hang undiagnosable, and it would be invisible
    in review because the surrounding lines all look instrumented. So the
    property is asserted against the source rather than trusted.

    `get_settings()` is the reason this exists: it calls `Settings.prepare()`
    itself, so on the production path it ran before the first stage began.
    """
    import ast
    import inspect

    import local_lm.main as module

    tree = ast.parse(inspect.getsource(module))
    create_app = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "create_app"
    )

    unwrapped: list[str] = []
    for statement in create_app.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            break
        if isinstance(statement, ast.With):
            names = [
                item.context_expr.func.id
                for item in statement.items
                if isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
            ]
            if "_startup_stage" in names:
                continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "_configure_console_logging"
        ):
            continue
        unwrapped.append(ast.dump(statement)[:70])

    assert not unwrapped, f"startup statements outside any stage: {unwrapped}"


def test_every_stage_has_a_distinct_name() -> None:
    """Two stages sharing a name would make the evidence ambiguous."""
    import ast
    import inspect

    import local_lm.main as module

    names = [
        node.args[0].value
        for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_startup_stage"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    assert names, "no startup stages found"
    assert len(names) == len(set(names)), f"duplicate stage names: {names}"


def test_a_still_running_notice_cannot_land_after_the_stage_has_finished() -> None:
    """The last name emitted must still be a stage that is running.

    `finished.set()` stops another wait cycle but says nothing about an
    emission already past the wait. A watchdog parked between returning from
    `wait` and writing its line can otherwise print after the stage has exited
    and recorded its own finished notice, which makes the final diagnostic name
    a stage that is no longer running - the one thing this is for.

    The watchdog is parked inside the emitter, the stage is then allowed to
    exit, and only afterwards is the watchdog released. The assertion is on the
    ORDER of the recorded notices rather than on any timing, so it is decided
    by the implementation and not by the scheduler.
    """
    notices: list[str] = []
    record_lock = threading.Lock()
    parked = threading.Event()
    release = threading.Event()
    finished_seen = threading.Event()
    late_recorded = threading.Event()

    def recording_notice(message: str) -> None:
        if "still running" in message:
            parked.set()
            release.wait(timeout=10)
        with record_lock:
            notices.append(message)
        if "finished" in message:
            finished_seen.set()
        else:
            late_recorded.set()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main, "_emit_startup_notice", recording_notice)
    try:

        def run_stage() -> None:
            with main._startup_stage("late-notice", warn_after=0.05):
                parked.wait(timeout=10)

        stage = threading.Thread(target=run_stage, name="late-notice-stage")
        stage.start()
        assert parked.wait(timeout=10), "the watchdog never reached the emitter"
        # An implementation that does not serialize lets the stage finish here.
        # One that does will block until the parked notice is released, and this
        # wait simply times out; either way the release below is what decides
        # the order that is asserted.
        finished_seen.wait(timeout=1.0)
        release.set()
        assert late_recorded.wait(timeout=10), "the parked notice never landed"
        stage.join(timeout=15)
        assert not stage.is_alive(), "the stage never exited"
    finally:
        monkeypatch.undo()

    with record_lock:
        recorded = list(notices)
    assert any("still running" in message for message in recorded), recorded
    assert any("finished" in message for message in recorded), recorded
    last_running = max(
        index for index, message in enumerate(recorded) if "still running" in message
    )
    first_finished = min(index for index, message in enumerate(recorded) if "finished" in message)
    assert last_running < first_finished, (
        f"a still-running notice landed after the stage finished: {recorded}"
    )


def test_a_watchdog_that_wakes_after_the_stage_finished_stays_silent() -> None:
    """The other half of the window: past the wait, but not yet speaking.

    The lock alone orders a watchdog that is already inside the emitter. It
    does nothing for one that has returned from `wait` and has not yet taken
    the lock: by the time it does, the stage can have finished and recorded so,
    and it would then name a stage that has stopped. Setting the event before
    the lock and rechecking it under the lock is what closes this half, and
    without that recheck this test fails.

    The interleaving cannot be waited for, so it is forced. Every lock built
    while the factory is installed is a gated one, and they behave as ordinary
    locks for everybody else; the gate opens only for the lock the watchdog
    loop enters directly, which is the one it serializes speaking on. If the
    loop is ever renamed or stops taking that lock the gate never opens and
    this fails on the wait below, rather than gating nothing and passing.
    """
    notices: list[str] = []
    record_lock = threading.Lock()
    at_the_lock = threading.Event()
    let_through = threading.Event()
    real_lock = threading.Lock
    gated_by: list[str] = []

    class GatedLock:
        """Holds the watchdog at the door once, after its wait has returned.

        The Event's own lock is entered from inside `Event.wait`, so its
        immediate caller is that machinery rather than `watch`, and it is left
        alone. Without that distinction the gate would close on the very first
        wait and the watchdog would never reach the emitter at all. Counting
        which lock was handed out is the obvious alternative and is wrong: any
        other thread building a lock between the stage's Event and its Lock
        moves the number and gates something else, which is a flake rather
        than a failure.
        """

        def __init__(self) -> None:
            self._lock = real_lock()

        def acquire(self, *arguments: object, **keywords: object) -> bool:
            # Forward the arguments. Condition._is_owned probes a lock with
            # acquire(False) when the lock does not supply _is_owned itself,
            # and a signature that swallows that argument turns the probe into
            # a blocking wait, which deadlocks the Event this stage waits on.
            return self._lock.acquire(*arguments, **keywords)  # type: ignore[arg-type]

        def release(self) -> None:
            self._lock.release()

        def __enter__(self) -> bool:
            if sys._getframe(1).f_code.co_name == "watch" and not at_the_lock.is_set():
                gated_by.append("watchdog")
                at_the_lock.set()
                let_through.wait(timeout=10)
            return self.acquire()

        def __exit__(self, *exception: object) -> None:
            self.release()

    def lock_factory() -> object:
        return GatedLock()

    def recording_notice(message: str) -> None:
        with record_lock:
            notices.append(message)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main, "_emit_startup_notice", recording_notice)
    monkeypatch.setattr(main.threading, "Lock", lock_factory)
    try:
        with main._startup_stage("woken-late", warn_after=0.05):
            assert at_the_lock.wait(timeout=10), "the watchdog never reached the lock"
        # The stage has exited and recorded that it finished while the watchdog
        # was held at the door. Only now is the watchdog allowed to proceed.
        let_through.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with record_lock:
                if any("still running" in message for message in notices):
                    break
            time.sleep(0.01)
    finally:
        let_through.set()
        monkeypatch.undo()

    assert gated_by == ["watchdog"], (
        f"the gated lock was not the one the watchdog serializes on: {gated_by}"
    )
    with record_lock:
        recorded = list(notices)
    assert any("finished" in message for message in recorded), recorded
    assert not any("still running" in message for message in recorded), (
        f"a watchdog woken after the stage finished still announced it: {recorded}"
    )
