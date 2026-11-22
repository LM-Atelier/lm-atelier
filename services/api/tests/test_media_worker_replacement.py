"""A media run waits, queued, before it restarts ComfyUI under someone else's work.

A run that needs ComfyUI started with a different set of models and custom
nodes stops the worker that is running. When that worker is rendering for
another application, or a workflow editor is open on it, the stop ends their
work. Such a run waits before it claims anything: it stays queued, holds no
device slot, and leaves chat loaded, so text turns and other jobs go ahead of
it. It starts once the other work has ended, or fails after the configured wait
without ending it. A prompt LM Atelier itself left behind is not a reason to
wait. Work that begins while the run is being prepared is caught at the restart
itself, and the run fails then rather than stop it.

A run that is stopped before it generates anything is not reported to the media
engine. ComfyUI keeps a stop for a run it is not generating and applies it to
that run's next attempt, so a retry of a run stopped while it was queued would
end at once without a result.

After someone else's prompts, ComfyUI must stay free for a quiet spell before a
run restarts it. The spell is one for every run, so runs held back by the same
work start in their order in the queue, and while a run that met that work
could start, a run that did not wait for it does not start inside the spell
either. A database another writer holds while a run shows why it waits keeps
the run waiting, and a restart shows a run that was waiting as queued again.

A run that is paused or held while it waits keeps its wait and the time it has
counted, but it starts only once it is let go, so it keeps no window open: once
ComfyUI is free, the device goes back to chat as if nothing waited. What it saw
of someone else's work may be long over by then, so it holds back no other run.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS, TERMINAL_STATUSES, wait_until
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from test_chat_handoff_across_retries import finished
from test_generation_retry import read_run, start_media
from test_media_warm_window import SCOPE, _warm_orchestrator
from test_scheduler import _real_sqlite_error
from test_workflow_editor_api import _create_workflow

from local_lm import orchestrator as orchestrator_module
from local_lm import scheduler as scheduler_module
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal, database_is_contended
from local_lm.models import Job, Message, Run
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.schemas import WorkerStatus

if TYPE_CHECKING:
    from local_lm.media_worker_use import MediaUse

PROMPTS_LABEL = "Waiting for other media work to finish"
EDITOR_LABEL = "Waiting for the workflow editor to close"
RESTART_NEEDED = (
    "This request needs ComfyUI restarted with a different set of models and custom nodes. "
)


class ReplacementWorld:
    """A ready ComfyUI worker that an image run would replace, with chat loaded beside it.

    The live worker was launched for another workflow's own models and custom
    nodes, and the run uses a workflow without a dependency contract, so the run
    needs the broad worker in its place. The supervisor's worker effects are
    recorded rather than made. ComfyUI holds one prompt LM Atelier left behind
    and `foreign` more that someone else queued: the total it reports counts
    both, and the adapter's own count only theirs. `answers` are counts the
    adapter gives once each, before it goes back to `foreign`.
    """

    def __init__(
        self,
        app: FastAPI,
        settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
        *,
        settle_seconds: float = 0.3,
    ) -> None:
        services = app.state.services
        self.services = services
        self.orchestrator: ConversationOrchestrator = services.orchestrator
        self.order: list[str] = []
        self.chat_running = True
        self.media_running = True
        self.media_pid = 31
        self.scope: str | None = SCOPE
        self.foreign = 0
        self.answers: deque[int] = deque()
        self.probes = 0
        self.on_chat_stop: Callable[[], None] | None = None
        self.starting = asyncio.Event()
        self.start_gate: asyncio.Event | None = None
        settings.media_engine = "comfyui"
        processes = services.processes
        monkeypatch.setattr(processes, "statuses", self.statuses)
        monkeypatch.setattr(processes, "launch_scope_sha256", self.launch_scope)
        monkeypatch.setattr(processes, "stop", self.stop)
        monkeypatch.setattr(processes, "start_media", self.start_media)
        media = services.engines.media
        monkeypatch.setattr(media, "foreign_prompt_count", self.foreign_prompt_count, raising=False)
        monkeypatch.setattr(media, "queue_remaining", self.queue_remaining, raising=False)
        monkeypatch.setattr(self.orchestrator, "_resume_chat_worker", self.resume)
        # Shorter than the application's, so a wait ends within a test.
        monkeypatch.setattr(
            orchestrator_module, "MEDIA_IDLE_SETTLE_SECONDS", settle_seconds, raising=False
        )
        monkeypatch.setattr(scheduler_module, "_ADMISSION_RECHECK_SECONDS", 0.05, raising=False)

    def statuses(self) -> list[WorkerStatus]:
        return [
            WorkerStatus(
                name="chat",
                state="ready" if self.chat_running else "stopped",
                managed=True,
                running=self.chat_running,
                pid=11 if self.chat_running else None,
                profile_id="profile-chat" if self.chat_running else None,
            ),
            WorkerStatus(
                name="media",
                state="ready" if self.media_running else "stopped",
                managed=True,
                running=self.media_running,
                pid=self.media_pid if self.media_running else None,
            ),
        ]

    def launch_scope(self, name: str) -> str | None:
        return self.scope if name == "media" and self.media_running else None

    async def stop(self, name: str, **_options: object) -> WorkerStatus:
        self.order.append(f"stop {name}")
        if name == "chat":
            self.chat_running = False
            if self.on_chat_stop is not None:
                self.on_chat_stop()
        else:
            self.media_running = False
        return next(item for item in self.statuses() if item.name == name)

    async def start_media(self, *_args: object, **options: Any) -> WorkerStatus:
        guard = options.get("replacement_guard")
        if self.media_running and guard is not None:
            # The supervisor asks under the worker's lock, just before its stop.
            await guard()
        self.starting.set()
        if self.start_gate is not None:
            await self.start_gate.wait()
        self.order.append("start media")
        self.scope = getattr(options.get("activation_scope"), "launch_sha256", None)
        self.media_running = True
        self.media_pid += 1
        return self.statuses()[1]

    async def resume(self, profile_id: str) -> None:
        self.order.append(f"resume {profile_id}")
        self.chat_running = True

    async def foreign_prompt_count(self) -> int:
        self.probes += 1
        if self.answers:
            return self.answers.popleft()
        return self.foreign

    async def queue_remaining(self) -> int:
        return self.foreign + 1


async def _image_setup(client: AsyncClient) -> dict[str, str]:
    """Create a workflow without a dependency contract, and an image profile to run it with."""

    workflow = await _create_workflow(client)
    profile = await client.post(
        "/api/profiles",
        json={"name": "Replacement image profile", "role": "image", "engine": "comfyui"},
    )
    assert profile.status_code == 201, profile.text
    return {"revision": workflow["current_revision_id"], "profile": profile.json()["id"]}


async def _image_turn(client: AsyncClient, setup: dict[str, str] | None = None) -> tuple[str, str]:
    """Send an image turn in a new chat, on `setup` or a new one; (chat, run)."""

    if setup is None:
        setup = await _image_setup(client)
    chat = (await client.post("/api/chats", json={"title": "Replacement"})).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}", json={"active_image_profile_id": setup["profile"]}
    )
    assert selected.status_code == 200, selected.text
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Draw a blue bowl",
            "mode": "image",
            "workflow_revision_id": setup["revision"],
        },
    )
    assert accepted.status_code == 202, accepted.text
    return str(chat["id"]), str(accepted.json()["run"]["id"])


def _job_id(run_id: str) -> str:
    with SessionLocal() as session:
        job_id = session.scalar(select(Job.id).where(Job.run_id == run_id))
    assert job_id is not None
    return job_id


async def _waiting_state(run_id: str) -> dict[str, Any]:
    """The run's job as the queue records it, and what its conversation shows."""

    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        run = session.get(Run, run_id)
        assert job is not None and run is not None
        message = session.get(Message, run.assistant_message_id)
        assert message is not None
        progress = job.progress_json if isinstance(job.progress_json, dict) else {}
        return {
            "status": job.status,
            "claimed": job.claim_owner is not None,
            "attempt": job.attempt,
            "stage": progress.get("stage"),
            "shown": next((part.text for part in message.parts if part.type == "progress"), None),
        }


async def _shown_waiting(world: ReplacementWorld, run_id: str, label: str) -> dict[str, Any]:
    """Wait for the run to show `label`, failing at once if a worker moved first."""

    async def read() -> dict[str, Any]:
        assert world.order == [], f"a worker moved before the run waited: {world.order}"
        return await _waiting_state(run_id)

    return await wait_until(
        read,
        lambda state: state["stage"] == label and state["shown"] == label.title(),
        what=f"the run to show {label!r}",
    )


async def _execution_ended(orchestrator: ConversationOrchestrator, job_id: str) -> None:
    async def running() -> bool:
        return job_id in orchestrator._tasks

    await wait_until(running, lambda still: not still, what="the job's execution to end")


async def _worker_order(world: ReplacementWorld, count: int, what: str) -> list[str]:
    """Wait for the world to record at least `count` worker effects, and return them."""

    async def order() -> list[str]:
        return list(world.order)

    return await wait_until(order, lambda done: len(done) >= count, what=what)


def _generation_order(world: ReplacementWorld, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The runs in the order they begin generating, which is the order they start in."""

    media = world.services.engines.media
    generate = media.generate
    order: list[str] = []

    async def recorded(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        order.append(request.run_id)
        async for event in generate(request):
            yield event

    monkeypatch.setattr(media, "generate", recorded)
    return order


class _Checks:
    """Every answer the runs' admission checks give, and a gate that can hold one run's checks.

    An answer is recorded as (job, answer, whether ComfyUI held someone else's
    prompts when the check began). A check of a run whose gate is closed waits
    at the gate until the test opens it, as a check that comes round later would.
    """

    def __init__(self, world: ReplacementWorld, monkeypatch: pytest.MonkeyPatch) -> None:
        self.answers: list[tuple[str, str | None, bool]] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.at_gate: set[str] = set()
        check = world.orchestrator._admit_media_run

        async def recorded(job_id: str, run_id: str) -> str | None:
            gate = self.gates.get(job_id)
            if gate is not None and not gate.is_set():
                self.at_gate.add(job_id)
                await gate.wait()
                self.at_gate.discard(job_id)
            busy = world.foreign > 0
            answer = await check(job_id, run_id)
            self.answers.append((job_id, answer, busy))
            return answer

        monkeypatch.setattr(world.orchestrator, "_admit_media_run", recorded)

    async def held_at_gate(self) -> set[str]:
        return set(self.at_gate)

    def answers_once_free(self, job_id: str) -> Callable[[], Awaitable[list[str | None]]]:
        """Return a reader of `job_id`'s answers from checks that began while ComfyUI was free."""

        async def read() -> list[str | None]:
            return [answer for job, answer, busy in self.answers if job == job_id and not busy]

        return read


class _HeldDatabase:
    """Another writer that takes the database in a thread of its own and keeps it a while."""

    def __init__(self, path: Path, seconds: float) -> None:
        self.path = path
        self.seconds = seconds
        self.holding = threading.Event()
        self.thread: threading.Thread | None = None

    def take(self) -> None:
        """Take the database once, returning as soon as it is held."""

        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._hold, name="held-database", daemon=True)
        self.thread.start()
        assert self.holding.wait(PATIENCE_SECONDS), "the other writer never took the database"

    def _hold(self) -> None:
        connection = sqlite3.connect(self.path, timeout=PATIENCE_SECONDS)
        try:
            connection.execute("BEGIN IMMEDIATE")
            self.holding.set()
            time.sleep(self.seconds)
            connection.rollback()
        finally:
            connection.close()

    def released(self) -> bool:
        """Whether the writer took the database and has let it go again."""

        if self.thread is None:
            return False
        self.thread.join(PATIENCE_SECONDS)
        return not self.thread.is_alive()


def _stage_writes(world: ReplacementWorld, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record how each write of a queued run's stage ends.

    Each write is "written", "held" while another writer holds the database, or
    "failed".
    """

    write = world.orchestrator._set_queued_media_phase
    outcomes: list[str] = []

    async def recorded(job_id: str, run_id: str, label: str | None) -> None:
        try:
            await write(job_id, run_id, label)
        except OperationalError as error:
            outcomes.append("held" if database_is_contended(error) else "failed")
            raise
        outcomes.append("written")

    monkeypatch.setattr(world.orchestrator, "_set_queued_media_phase", recorded)
    return outcomes


async def _progress(client: AsyncClient, run_id: str) -> dict[str, Any]:
    """The progress part the run's conversation shows, as the client reads it."""

    message_id = (await read_run(client, run_id))["assistant_message_id"]
    message = await client.get(f"/api/messages/{message_id}")
    assert message.status_code == 200, message.text
    parts = [part for part in message.json()["parts"] if part["type"] == "progress"]
    assert len(parts) == 1, parts
    return {"text": parts[0]["text"], "metadata": parts[0]["metadata_json"]}


async def test_a_run_that_would_replace_a_busy_worker_waits_queued_while_chat_works(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A settle long enough that two checks in a row always fall inside it.
    world = ReplacementWorld(app, settings, monkeypatch, settle_seconds=2.0)
    # Another application is rendering in ComfyUI.
    world.foreign = 1
    _chat_id, run_id = await _image_turn(client)

    waiting = await _shown_waiting(world, run_id, PROMPTS_LABEL)
    assert waiting == {
        "status": "queued",
        "claimed": False,
        "attempt": 0,
        "stage": PROMPTS_LABEL,
        "shown": "Waiting For Other Media Work To Finish",
    }

    # Chat stays loaded, and a text turn goes ahead of the waiting run.
    other = (await client.post("/api/chats", json={"title": "Meanwhile"})).json()
    text = await client.post(
        f"/api/chats/{other['id']}/turns",
        json={"text": "Name three primary colors.", "mode": "text"},
    )
    assert text.status_code == 202, text.text
    await finished(client, text.json()["run"]["id"])
    assert world.order == []
    assert (await _waiting_state(run_id))["status"] == "queued"

    # A moment with nothing of anyone else's in ComfyUI, shorter than the settle,
    # is a gap between two of their prompts rather than the end of their work,
    # however many checks fall inside it.
    world.answers.extend([0, 0])
    seen = world.probes

    async def probes() -> int:
        return world.probes

    await wait_until(probes, lambda count: count >= seen + 3, what="three more checks")
    assert world.order == []
    assert (await _waiting_state(run_id))["shown"] == "Waiting For Other Media Work To Finish"

    # Their work ends. ComfyUI's own count still includes the prompt LM Atelier
    # left behind, and that is no reason to go on waiting.
    world.foreign = 0

    async def order() -> list[str]:
        return list(world.order)

    moved = await wait_until(order, lambda done: len(done) >= 2, what="the run to start")
    assert moved[:2] == ["stop chat", "start media"]
    await finished(client, run_id)


async def test_a_run_that_keeps_the_live_worker_does_not_wait(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = ReplacementWorld(app, settings, monkeypatch)
    # The live worker is the broad one this run uses, so nothing is replaced and
    # the other application's render goes on beside the run.
    world.scope = None
    world.foreign = 1
    _chat_id, run_id = await _image_turn(client)

    await finished(client, run_id)

    assert world.order == ["stop chat"]


@pytest.mark.parametrize(("route", "answer"), [("cancel", 200), ("cancel-all", 204)])
async def test_stopping_a_waiting_run_moves_no_worker_and_tells_the_engine_nothing(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    answer: int,
) -> None:
    world = ReplacementWorld(app, settings, monkeypatch)
    world.foreign = 1
    engine_cancel = AsyncMock()
    monkeypatch.setattr(world.services.engines.media, "cancel", engine_cancel)
    chat_id, run_id = await _image_turn(client)
    await _shown_waiting(world, run_id, PROMPTS_LABEL)

    stopped = await client.post(f"/api/chats/{chat_id}/{route}")
    assert stopped.status_code == answer, stopped.text
    await finished(client, run_id, "cancelled")
    await _execution_ended(world.orchestrator, _job_id(run_id))

    # It never claimed, so no worker moved, and nothing of it reached the engine.
    assert world.order == []
    engine_cancel.assert_not_awaited()
    # Its wait ended with it, so nothing keeps a window open or stands aside for it.
    assert world.orchestrator._media_waits == {}
    assert world.services.scheduler._deferred == {}


@pytest.mark.parametrize(
    ("limit", "waited"),
    [(0.3, "after 0.3 seconds"), (0, "and LM Atelier is set not to wait for it")],
)
async def test_a_run_still_waiting_at_the_limit_fails_before_claiming(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    limit: float,
    waited: str,
) -> None:
    world = ReplacementWorld(app, settings, monkeypatch)
    world.foreign = 1
    settings.media_replace_wait_seconds = limit
    handoffs = AsyncMock()
    monkeypatch.setattr(world.orchestrator, "_complete_media_handoff", handoffs)
    _chat_id, run_id = await _image_turn(client)

    run = await finished(client, run_id, "failed")

    assert run["error"] == (
        f"{RESTART_NEEDED}ComfyUI was still running work that LM Atelier did not start "
        f"{waited}, so that work was left running and this request was stopped. Try again "
        "once it has finished."
    )
    # A run that fails before any claim reserves no retry, so none of the
    # allowance is spent.
    assert run["provenance_json"]["failure_retries"] == {"limit": 3, "used": 0, "pending": False}
    assert (await _waiting_state(run_id))["attempt"] == 0
    assert world.order == []
    handoffs.assert_not_awaited()


async def test_an_editor_left_open_holds_the_run_with_its_own_reason(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = ReplacementWorld(app, settings, monkeypatch)
    # ComfyUI runs nothing, but a workflow editor is open on its launch.
    world.orchestrator.media_worker_in_use = lambda: True
    _chat_id, run_id = await _image_turn(client)

    waiting = await _shown_waiting(world, run_id, EDITOR_LABEL)
    assert waiting["status"] == "queued"
    assert waiting["shown"] == "Waiting For The Workflow Editor To Close"

    settings.media_replace_wait_seconds = 0.3
    run = await finished(client, run_id, "failed")

    assert run["error"] == (
        f"{RESTART_NEEDED}The ComfyUI workflow editor was still open after 0.3 seconds, so it "
        "was left open and this request was stopped. Save or close the editor, then try again."
    )
    assert world.order == []


def _ended_or_retrying(run: dict[str, Any]) -> bool:
    """Whether the run has ended, or a failure has queued it for a retry."""

    retries = run["provenance_json"]["failure_retries"]
    return run["status"] in TERMINAL_STATUSES or retries["used"] > 0 or retries["pending"]


async def test_work_that_starts_while_the_run_prepares_is_not_stopped(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = ReplacementWorld(app, settings, monkeypatch)

    def someone_queues_a_render() -> None:
        world.foreign = 1

    # ComfyUI is free when the run starts, and another application queues a
    # render while the run releases chat.
    world.on_chat_stop = someone_queues_a_render
    _chat_id, run_id = await _image_turn(client)

    # The run is refused after its claim, where a failure can reserve a retry,
    # so this looks for a retry as well as for the run's end.
    run = await wait_until(
        lambda: read_run(client, run_id),
        _ended_or_retrying,
        what="the run to end or be queued again",
    )

    # A retry would meet the same work, so the refusal reserves none.
    assert (run["status"], run["provenance_json"]["failure_retries"]) == (
        "failed",
        {"limit": 3, "used": 0, "pending": False},
    )
    assert run["error"] == (
        "ComfyUI was running work that LM Atelier did not start, so it was not restarted for "
        "this request and that work was left running. Try again once it has finished."
    )
    assert world.order == ["stop chat"]
    # ComfyUI is held for its other user, and chat is owed back once they finish.
    assert world.orchestrator._media_warm_window is not None
    assert world.orchestrator._displaced_chat_profile_id == "profile-chat"


async def test_cancelling_a_run_while_its_worker_starts_tells_the_engine_nothing(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = ReplacementWorld(app, settings, monkeypatch)
    world.start_gate = asyncio.Event()
    engine_cancel = AsyncMock()
    monkeypatch.setattr(world.services.engines.media, "cancel", engine_cancel)
    chat_id, run_id = await _image_turn(client)

    async def starting() -> bool:
        return world.starting.is_set()

    await wait_until(starting, bool, what="the media worker to start")
    stopped = await client.post(f"/api/chats/{chat_id}/cancel")
    assert stopped.status_code == 200, stopped.text
    await finished(client, run_id, "cancelled")

    # The run had sent nothing to the engine, so there was nothing to tell it.
    engine_cancel.assert_not_awaited()


async def test_a_media_job_cancelled_while_queued_can_be_retried(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    media = services.engines.media
    original = media.generate
    generating: set[str] = set()
    remembered: set[str] = set()

    async def cancel(run_id: str) -> None:
        # As ComfyUI does: a stop for a run it is not generating is kept, and
        # applied to that run's next generation.
        if run_id not in generating:
            remembered.add(run_id)

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        if request.run_id in remembered:
            remembered.discard(request.run_id)
            yield MediaEvent(type="cancelled")
            return
        generating.add(request.run_id)
        try:
            async for event in original(request):
                yield event
        finally:
            generating.discard(request.run_id)

    monkeypatch.setattr(media, "cancel", cancel)
    monkeypatch.setattr(media, "generate", generate)

    # Another job holds the device, so the picture is still queued when it is stopped.
    async with services.scheduler.lease("primary"):
        run_id = await start_media(client)
        chat_id = (await read_run(client, run_id))["chat_id"]
        stopped = await client.post(f"/api/chats/{chat_id}/cancel")
        assert stopped.status_code == 200, stopped.text
    await finished(client, run_id, "cancelled")
    job_id = _job_id(run_id)
    await _execution_ended(services.orchestrator, job_id)

    retried = await client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200, retried.text
    await _execution_ended(services.orchestrator, job_id)

    # The retry ran to its result rather than ending at once on the old stop.
    assert (await read_run(client, run_id))["status"] == "complete"


async def test_stopping_a_run_that_is_generating_tells_the_engine(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = app.state.services.engines.media
    generating = asyncio.Event()
    engine_cancel = AsyncMock()

    async def generate(_request: MediaRequest) -> AsyncIterator[MediaEvent]:
        yield MediaEvent(type="progress", progress=0.1, phase="sampling")
        generating.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(media, "generate", generate)
    monkeypatch.setattr(media, "cancel", engine_cancel)
    run_id = await start_media(client)
    await asyncio.wait_for(generating.wait(), timeout=PATIENCE_SECONDS)

    stopped = await client.post(f"/api/jobs/{_job_id(run_id)}/cancel")
    assert stopped.status_code == 200, stopped.text
    await finished(client, run_id, "cancelled")

    # Its work may already be running in the engine, and the engine stops it at once.
    engine_cancel.assert_awaited_once_with(run_id)


async def test_runs_held_back_by_the_same_prompts_start_in_queue_order(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settle = 2.0
    world = ReplacementWorld(app, settings, monkeypatch, settle_seconds=settle)
    checks = _Checks(world, monkeypatch)
    generated = _generation_order(world, monkeypatch)
    # Another application is rendering in ComfyUI, and two runs wait for it.
    world.foreign = 1
    setup = await _image_setup(client)
    _chat, first = await _image_turn(client, setup)
    await _shown_waiting(world, first, PROMPTS_LABEL)
    _chat, second = await _image_turn(client, setup)
    await _shown_waiting(world, second, PROMPTS_LABEL)
    older = _job_id(first)

    # The older run's next check comes round late. The newer run's check is the
    # first to find the other work over, and the older run finds it over only
    # partway through the quiet spell that began then.
    gate = checks.gates[older] = asyncio.Event()
    await wait_until(checks.held_at_gate, lambda held: older in held, what="the older run's check")
    world.foreign = 0
    newer_once_free = await wait_until(
        checks.answers_once_free(_job_id(second)), bool, what="the newer run finding it free"
    )
    assert newer_once_free[0] == PROMPTS_LABEL
    await asyncio.sleep(settle / 2)
    gate.set()

    await finished(client, first)
    await finished(client, second)
    assert generated == [first, second], "the newer run started first"


async def test_a_run_first_asked_inside_the_quiet_spell_waits_and_starts_behind_the_older_run(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A quiet spell that lasts until the test ends it.
    world = ReplacementWorld(app, settings, monkeypatch, settle_seconds=PATIENCE_SECONDS)
    generated = _generation_order(world, monkeypatch)
    world.foreign = 1
    setup = await _image_setup(client)
    _chat, first = await _image_turn(client, setup)
    await _shown_waiting(world, first, PROMPTS_LABEL)

    # The other application's work ends, and a newer run is sent while ComfyUI
    # has been free for less than the quiet spell. It waits too, and says why.
    world.foreign = 0
    _chat, second = await _image_turn(client, setup)
    waiting = await _shown_waiting(world, second, PROMPTS_LABEL)
    assert waiting == {
        "status": "queued",
        "claimed": False,
        "attempt": 0,
        "stage": PROMPTS_LABEL,
        "shown": "Waiting For Other Media Work To Finish",
    }

    # The spell passes, and the older run starts first.
    monkeypatch.setattr(orchestrator_module, "MEDIA_IDLE_SETTLE_SECONDS", 0.2)
    await finished(client, first)
    await finished(client, second)
    assert generated == [first, second], "the newer run started first"
    assert world.order[:2] == ["stop chat", "start media"]


async def test_runs_stopped_inside_the_quiet_spell_leave_nothing_to_hold_back_a_later_run(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = ReplacementWorld(app, settings, monkeypatch, settle_seconds=PATIENCE_SECONDS)
    world.foreign = 1
    setup = await _image_setup(client)
    waiting = [await _image_turn(client, setup)]
    await _shown_waiting(world, waiting[0][1], PROMPTS_LABEL)
    world.foreign = 0
    waiting.append(await _image_turn(client, setup))
    await _shown_waiting(world, waiting[1][1], PROMPTS_LABEL)

    # Both are stopped while they sit out the quiet spell.
    for chat_id, run_id in waiting:
        stopped = await client.post(f"/api/chats/{chat_id}/cancel")
        assert stopped.status_code == 200, stopped.text
        await finished(client, run_id, "cancelled")
        await _execution_ended(world.orchestrator, _job_id(run_id))

    # What their checks saw went with them, so a later run starts at once,
    # although the spell they sat out would still be running.
    _chat, later = await _image_turn(client, setup)
    await finished(client, later)
    assert world.order[:2] == ["stop chat", "start media"]


async def test_a_run_waits_on_through_a_database_held_while_it_shows_why(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = ReplacementWorld(app, settings, monkeypatch)
    world.foreign = 1
    with SessionLocal() as session:
        busy_timeout = session.execute(text("PRAGMA busy_timeout")).scalar_one() / 1000
    # Another writer takes the database as the run's first check looks at
    # ComfyUI, and keeps it past the busy timeout of the write that shows why
    # the run waits.
    other_writer = _HeldDatabase(settings.state_dir / "local-lm.sqlite3", busy_timeout + 1)
    probe = world.foreign_prompt_count

    async def first_check_meets_a_held_database() -> int:
        other_writer.take()
        return await probe()

    monkeypatch.setattr(
        world.services.engines.media,
        "foreign_prompt_count",
        first_check_meets_a_held_database,
        raising=False,
    )
    writes = _stage_writes(world, monkeypatch)
    _chat_id, run_id = await _image_turn(client)

    # The run goes on waiting, unclaimed, and a later check shows why once the
    # database is free again.
    waiting = await _shown_waiting(world, run_id, PROMPTS_LABEL)
    assert waiting == {
        "status": "queued",
        "claimed": False,
        "attempt": 0,
        "stage": PROMPTS_LABEL,
        "shown": "Waiting For Other Media Work To Finish",
    }
    assert writes[:2] == ["held", "written"]
    assert other_writer.released()

    # It then starts like any run whose wait is over.
    world.foreign = 0
    await finished(client, run_id)
    assert world.order[:2] == ["stop chat", "start media"]


async def test_a_run_set_not_to_wait_still_gives_owed_chat_back_once_comfyui_is_free(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator_module, "MEDIA_BUSY_RECHECK_SECONDS", 0.05)
    world = ReplacementWorld(app, settings, monkeypatch)
    # An earlier media run left chat down and owed for the run queued behind it.
    world.chat_running = False
    world.orchestrator._displaced_chat_profile_id = "profile-chat"
    world.foreign = 1
    settings.media_replace_wait_seconds = 0
    _chat_id, run_id = await _image_turn(client)

    run = await finished(client, run_id, "failed")
    assert "LM Atelier is set not to wait for it" in run["error"]
    assert world.order == []
    # ComfyUI is held for its other user, and chat is owed back once they finish.
    assert world.orchestrator._media_warm_window is not None

    world.foreign = 0
    assert await _worker_order(world, 2, "chat to come back") == [
        "stop media",
        "resume profile-chat",
    ]
    assert world.chat_running


async def _set_aside(
    client: AsyncClient, run_id: str, how: str
) -> tuple[str, Callable[[], Awaitable[None]]]:
    """Pause the generation lane, or hold the run's own work, with the queue's controls.

    Returns the stage the queue then shows the run at, and what lets it go again.
    """

    if how == "pause":
        paused = await client.post(
            "/api/queue/lanes/generation/pause-after-current",
            json={"expected_revision": 0, "idempotency_key": "pause-while-waiting"},
        )
        assert paused.status_code == 200, paused.text
        lane_revision = paused.json()["revision"]

        async def resume() -> None:
            resumed = await client.post(
                "/api/queue/lanes/generation/resume",
                json={"expected_revision": lane_revision, "idempotency_key": "resume-waiting"},
            )
            assert resumed.status_code == 200, resumed.text

        return "generation paused", resume
    assert how == "hold"
    plan_id = (await read_run(client, run_id))["work_plan_id"]
    held = await client.post(
        f"/api/queue/items/{plan_id}/hold",
        json={"expected_revision": 0, "idempotency_key": "hold-while-waiting"},
    )
    assert held.status_code == 200, held.text
    plan_revision = held.json()["control_revision"]

    async def release() -> None:
        released = await client.post(
            f"/api/queue/items/{plan_id}/release",
            json={"expected_revision": plan_revision, "idempotency_key": "release-waiting"},
        )
        assert released.status_code == 200, released.text

    return "held", release


async def _queue_shows(run_id: str, stage: str) -> dict[str, Any]:
    """Wait for the queue to show the run's job at `stage`."""

    return await wait_until(
        lambda: _waiting_state(run_id),
        lambda state: state["stage"] == stage,
        what=f"the queue to show the run {stage!r}",
    )


@pytest.mark.parametrize("how", ["pause", "hold"])
async def test_owed_chat_comes_back_while_a_waiting_run_is_paused_or_held(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
) -> None:
    monkeypatch.setattr(orchestrator_module, "MEDIA_BUSY_RECHECK_SECONDS", 0.05)
    world = ReplacementWorld(app, settings, monkeypatch)
    # An earlier media run left chat down and owed for the run queued behind it.
    world.chat_running = False
    world.orchestrator._displaced_chat_profile_id = "profile-chat"
    world.foreign = 1
    _chat_id, run_id = await _image_turn(client)
    await _shown_waiting(world, run_id, PROMPTS_LABEL)
    # ComfyUI is held for its other user, and chat is owed back once they finish.
    assert world.orchestrator._media_warm_window is not None
    job_id = _job_id(run_id)
    wait = world.orchestrator._media_waits[job_id]

    # Paused or held, the run starts only once it is let go, however long that is.
    stage, let_go = await _set_aside(client, run_id, how)
    await _queue_shows(run_id, stage)

    # The other work ends, and the device goes back to chat with the run still waiting.
    world.foreign = 0
    assert await _worker_order(world, 2, "chat to come back") == [
        "stop media",
        "resume profile-chat",
    ]
    assert world.chat_running
    assert world.orchestrator._media_warm_window is None
    waiting = await _waiting_state(run_id)
    assert (waiting["status"], waiting["claimed"], waiting["stage"]) == ("queued", False, stage)
    # It keeps its wait, and with it the time it has counted.
    assert world.orchestrator._media_waits.get(job_id) is wait

    # Let go, it is asked again and starts.
    await let_go()
    await finished(client, run_id)
    assert world.order[:4] == ["stop media", "resume profile-chat", "stop chat", "start media"]


async def test_a_later_runs_window_hands_the_device_back_while_a_waiting_run_is_held(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator_module, "MEDIA_BUSY_RECHECK_SECONDS", 0.05)
    settings.media_keep_warm_seconds = 0.2
    world = ReplacementWorld(app, settings, monkeypatch)
    world.foreign = 1
    setup = await _image_setup(client)
    _chat, earlier = await _image_turn(client, setup)
    await _shown_waiting(world, earlier, PROMPTS_LABEL)
    stage, release = await _set_aside(client, earlier, "hold")
    await _queue_shows(earlier, stage)

    # The other work ends, and a later run in another chat goes ahead of the
    # held one. It finishes with nothing behind it that could start, so it
    # keeps its worker warm.
    world.foreign = 0
    _chat, later = await _image_turn(client, setup)
    await finished(client, later)

    # Nobody uses the worker in that window, so it ends as it would with nothing
    # waiting, and chat comes back.
    order = await _worker_order(world, 4, "the later run's window to end")
    assert order[:4] == ["stop chat", "start media", "stop media", "resume profile-chat"]
    assert world.chat_running
    waiting = await _waiting_state(earlier)
    assert (waiting["status"], waiting["claimed"], waiting["stage"]) == ("queued", False, stage)
    assert _job_id(earlier) in world.orchestrator._media_waits

    # Released, the held run starts.
    await release()
    await finished(client, earlier)


async def _hold_a_run_that_met_other_work(
    client: AsyncClient, world: ReplacementWorld, setup: dict[str, str]
) -> tuple[str, Callable[[], Awaitable[None]]]:
    """Send a run that waits for another application's render, hold it, and end the render.

    The run's last check saw the render. Returns the run and what releases it.
    """

    world.foreign = 1
    _chat, run_id = await _image_turn(client, setup)
    await _shown_waiting(world, run_id, PROMPTS_LABEL)
    stage, release = await _set_aside(client, run_id, "hold")
    await _queue_shows(run_id, stage)
    world.foreign = 0
    return run_id, release


async def test_what_a_held_run_saw_of_other_work_holds_back_no_later_run(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settle = 0.3
    world = ReplacementWorld(app, settings, monkeypatch, settle_seconds=settle)
    checks = _Checks(world, monkeypatch)
    setup = await _image_setup(client)
    held, release = await _hold_a_run_that_met_other_work(client, world, setup)
    # ComfyUI then stays free for longer than the quiet spell, with no run
    # asked about it in the meantime.
    await asyncio.sleep(2 * settle)

    # A later run that needs ComfyUI restarted starts at its first check,
    # without sitting out a spell or saying it waits for other work.
    _chat, later = await _image_turn(client, setup)
    await finished(client, later)
    assert [answer for job, answer, _busy in checks.answers if job == _job_id(later)] == [None]
    assert world.order[:2] == ["stop chat", "start media"]

    # Released, the held run starts on the worker the later run left.
    await release()
    await finished(client, held)


async def test_a_run_waiting_only_on_an_editor_starts_once_it_closes_beside_a_held_run(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = ReplacementWorld(app, settings, monkeypatch)
    checks = _Checks(world, monkeypatch)
    setup = await _image_setup(client)
    held, release = await _hold_a_run_that_met_other_work(client, world, setup)
    # A workflow editor opens on ComfyUI's launch, and a later run waits for it.
    editing = {"open": True}
    world.orchestrator.media_worker_in_use = lambda: editing["open"]
    _chat, later = await _image_turn(client, setup)
    await _shown_waiting(world, later, EDITOR_LABEL)
    later_job = _job_id(later)

    # The editor closes just before the later run is asked again.
    gate = checks.gates[later_job] = asyncio.Event()
    await wait_until(checks.held_at_gate, lambda held: later_job in held, what="the next check")
    asked = sum(job == later_job for job, _answer, _busy in checks.answers)
    editing["open"] = False
    gate.set()

    # That check lets it start: a closed editor does not come back by itself,
    # and what the held run saw holds back no other run.
    await finished(client, later)
    answers = [answer for job, answer, _busy in checks.answers if job == later_job]
    assert answers[asked] is None
    assert PROMPTS_LABEL not in answers
    assert world.order[:2] == ["stop chat", "start media"]

    await release()
    await finished(client, held)


async def test_a_restart_shows_a_run_that_was_waiting_as_queued(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    orchestrator = services.orchestrator
    # Another job holds the device, so the picture stays queued throughout.
    async with services.scheduler.lease("primary"):
        run_id = await start_media(client)
        job_id = _job_id(run_id)
        # When the application stopped, the run was waiting for someone else's
        # work in ComfyUI, and its conversation said so.
        await orchestrator._set_queued_media_phase(job_id, run_id, PROMPTS_LABEL)
        assert (await _progress(client, run_id))["text"] == "Waiting For Other Media Work To Finish"
        dispatched: list[tuple[str, str | None]] = []
        monkeypatch.setattr(orchestrator, "start", lambda job, run: dispatched.append((job, run)))

        orchestrator.recover_interrupted()

        assert dispatched == [(job_id, run_id)]
        # Nothing waits in a new process, so the run shows it is queued.
        assert await _progress(client, run_id) == {
            "text": "Queued",
            "metadata": {"progress": 0, "phase": "queued", "indeterminate": True},
        }
    await finished(client, run_id)


def _admission(
    monkeypatch: pytest.MonkeyPatch, *, replace_wait_seconds: float = 3600
) -> tuple[ConversationOrchestrator, dict[str, Any], AsyncMock]:
    """A run whose admission would replace the live scoped worker, and what it shows."""

    from local_lm.orchestrator import _NeededLaunch

    orchestrator, record, _ = _warm_orchestrator(
        60, scope=SCOPE, replace_wait_seconds=replace_wait_seconds
    )
    # Chat is loaded, so a wait owes nothing back.
    record["chat_running"] = True
    monkeypatch.setattr(orchestrator, "_needed_media_launch", lambda _run_id: _NeededLaunch(None))
    shown = AsyncMock()
    monkeypatch.setattr(orchestrator, "_set_queued_media_phase", shown)
    return orchestrator, record, shown


async def test_a_wait_adds_up_the_gaps_between_its_checks_each_at_most_one_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm.media_worker_use import MediaWorkerBusy

    orchestrator, record, _ = _admission(monkeypatch, replace_wait_seconds=15)
    record["comfy_queue"] = 1
    admit = orchestrator._media_admission("job-waiting", "run-waiting")

    assert await admit() == PROMPTS_LABEL
    wait = orchestrator._media_waits["job-waiting"]
    # Moved back, the last busy check stands for ten minutes in which the run
    # was not asked, as while it is held, paused or waiting for earlier steps.
    # The check that ends them adds one step, no more than a late check would.
    wait.last_busy = (wait.last_busy or 0) - 600
    assert await admit() == PROMPTS_LABEL
    assert wait.counted == 10

    wait.last_busy = (wait.last_busy or 0) - 5
    with pytest.raises(MediaWorkerBusy) as refused:
        await admit()
    assert refused.value.late is False
    assert "after 15 seconds" in str(refused.value)
    await orchestrator.close()


async def test_a_run_set_not_to_wait_fails_at_its_first_busy_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm.media_worker_use import MediaWorkerBusy

    orchestrator, record, shown = _admission(monkeypatch, replace_wait_seconds=0)
    record["comfy_queue"] = 1
    admit = orchestrator._media_admission("job-waiting", "run-waiting")

    with pytest.raises(MediaWorkerBusy, match="LM Atelier is set not to wait for it"):
        await admit()
    # It never showed itself waiting, since it did not wait.
    shown.assert_not_awaited()
    await orchestrator.close()


async def test_no_run_restarts_comfyui_until_it_stays_free_after_someone_elses_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, record, shown = _admission(monkeypatch)
    record["comfy_queue"] = 1
    waited = orchestrator._media_admission("job-waited", "run-waited")
    assert await waited() == PROMPTS_LABEL

    # Free for a moment only, which can fall between two of someone's prompts.
    record["comfy_queue"] = 0
    assert await waited() == PROMPTS_LABEL
    # A run that did not wait for those prompts does not restart ComfyUI inside
    # that moment either: it waits for the same reason, and says so.
    newer = orchestrator._media_admission("job-newer", "run-newer")
    assert await newer() == PROMPTS_LABEL

    # Once ComfyUI has stayed free for the quiet spell, both may start.
    monkeypatch.setattr(orchestrator_module, "MEDIA_IDLE_SETTLE_SECONDS", 0.0)
    assert await waited() is None
    assert await newer() is None
    assert orchestrator._media_waits == {}
    # Each conversation shows its run as queued again rather than waiting.
    assert [call.args for call in shown.await_args_list] == [
        ("job-waited", "run-waited", PROMPTS_LABEL),
        ("job-newer", "run-newer", PROMPTS_LABEL),
        ("job-waited", "run-waited", None),
        ("job-newer", "run-newer", None),
    ]
    await orchestrator.close()


async def test_a_queue_that_cannot_be_read_keeps_a_newer_run_to_the_quiet_spell(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    orchestrator, comfy, _shown = _admission(monkeypatch)
    comfy["comfy_queue"] = 1
    waited = orchestrator._media_admission("job-waited", "run-waited")
    assert await waited() == PROMPTS_LABEL
    comfy["comfy_queue"] = 0

    def unreadable(_job_ids: object) -> frozenset[str]:
        raise RuntimeError("The queue could not be read")

    monkeypatch.setattr(orchestrator.scheduler, "startable_jobs", unreadable)
    newer = orchestrator._media_admission("job-newer", "run-newer")
    with caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
        # Whether the run that met those prompts could start cannot be told, so
        # what it saw counts, and the newer run sits out the spell with it.
        assert await newer() == PROMPTS_LABEL
        # Once the spell has passed, no check needs the queue.
        monkeypatch.setattr(orchestrator_module, "MEDIA_IDLE_SETTLE_SECONDS", 0.0)
        assert await newer() is None
    assert [
        record.levelno for record in caplog.records if record.name == "local_lm.orchestrator"
    ] == [logging.ERROR]
    await orchestrator.close()


async def test_a_run_whose_wait_ends_while_the_database_is_held_waits_until_it_shows_queued(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    orchestrator, _record, shown = _admission(monkeypatch)
    editing = {"open": True}
    orchestrator.media_worker_in_use = lambda: editing["open"]
    admit = orchestrator._media_admission("job-waiting", "run-waiting")
    assert await admit() == EDITOR_LABEL

    # The editor closes while another writer holds the database, so the
    # conversation cannot show the run queued again yet: it waits on, still
    # saying why, rather than starting as if the check could not tell.
    editing["open"] = False
    shown.side_effect = _real_sqlite_error(tmp_path, contended=True)
    with caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
        assert await admit() == EDITOR_LABEL
    assert "job-waiting" in orchestrator._media_waits
    assert [
        record.levelno for record in caplog.records if record.name == "local_lm.orchestrator"
    ] == [logging.INFO]

    # Once the database is free, the next check shows it queued and lets it start.
    shown.side_effect = None
    assert await admit() is None
    assert [call.args[2] for call in shown.await_args_list] == [EDITOR_LABEL, None, None]
    assert orchestrator._media_waits == {}
    await orchestrator.close()


async def test_a_check_that_cannot_answer_shows_the_run_it_lets_start_as_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, record, shown = _admission(monkeypatch)
    record["comfy_queue"] = 1
    admit = orchestrator._media_admission("job-waiting", "run-waiting")
    assert await admit() == PROMPTS_LABEL

    # The supervisor cannot say which worker runs, so the check cannot tell
    # whether the run would restart ComfyUI and lets it start, as it would have
    # without the check.
    def unreadable() -> list[WorkerStatus]:
        raise RuntimeError("The worker statuses could not be read")

    monkeypatch.setattr(orchestrator.processes, "statuses", unreadable)
    assert await admit() is None

    # It no longer waits, so its conversation no longer says it does.
    assert shown.await_args_list[-1].args == ("job-waiting", "run-waiting", None)
    assert orchestrator._media_waits == {}
    await orchestrator.close()


async def test_a_run_that_waited_for_an_editor_starts_once_it_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, _record, shown = _admission(monkeypatch)
    editing = {"open": True}
    orchestrator.media_worker_in_use = lambda: editing["open"]
    admit = orchestrator._media_admission("job-waiting", "run-waiting")
    assert await admit() == EDITOR_LABEL

    # A closed editor does not reopen by itself, so nothing needs to settle.
    editing["open"] = False
    assert await admit() is None
    assert [call.args[2] for call in shown.await_args_list] == [EDITOR_LABEL, None]
    await orchestrator.close()


@pytest.mark.parametrize(
    ("use", "late", "limit", "message"),
    [
        (
            "prompts",
            False,
            3600,
            f"{RESTART_NEEDED}ComfyUI was still running work that LM Atelier did not start "
            "after 60 minutes, so that work was left running and this request was stopped. "
            "Try again once it has finished.",
        ),
        (
            "editor",
            False,
            60,
            f"{RESTART_NEEDED}The ComfyUI workflow editor was still open after 1 minute, so it "
            "was left open and this request was stopped. Save or close the editor, then try "
            "again.",
        ),
        (
            "prompts",
            False,
            90,
            f"{RESTART_NEEDED}ComfyUI was still running work that LM Atelier did not start "
            "after 90 seconds, so that work was left running and this request was stopped. "
            "Try again once it has finished.",
        ),
        (
            "editor",
            False,
            0,
            f"{RESTART_NEEDED}The ComfyUI workflow editor was still open and LM Atelier is set "
            "not to wait for it, so it was left open and this request was stopped. Save or "
            "close the editor, then try again.",
        ),
        (
            "prompts",
            True,
            3600,
            "ComfyUI was running work that LM Atelier did not start, so it was not restarted "
            "for this request and that work was left running. Try again once it has finished.",
        ),
        (
            "editor",
            True,
            3600,
            "A ComfyUI workflow editor was open, so ComfyUI was not restarted under it. Save or "
            "close the editor, then try again.",
        ),
    ],
)
def test_each_refusal_says_what_was_left_running_and_what_to_do(
    use: MediaUse, late: bool, limit: float, message: str
) -> None:
    from local_lm.media_worker_use import MediaWorkerBusy

    refusal = MediaWorkerBusy(use, late=late, limit_seconds=limit)

    assert str(refusal) == message
    assert refusal.code == "media-worker-busy"
