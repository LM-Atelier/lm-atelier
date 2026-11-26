"""ComfyUI is emptied in place, not stopped, when a render gives the device back to chat.

Stopping ComfyUI to give the device to chat means starting an empty one again
afterwards, so the next render waits for a ComfyUI start as well as for its
models. ComfyUI can unload its models and free its cache in place: on a 16 GB
card about 10 GB came back within half a second. So after a render that
finished or failed on its own workflow, and when a window that kept ComfyUI
warm ends, it is emptied and the chat model loads beside it, provided the
memory really came back and the chat model fits in what is free.

A failed render's handoff still stops it, since only a new process clears some
faults, and so does a worker launched for one activation, which nothing else
may run on. A chat model that does not come up beside the emptied worker gets
the device as before: the worker is stopped, the model loaded again and an
empty worker started. Someone who begins using ComfyUI while it is emptied, or
while a chat model fails to load beside it, keeps it until they are done, as
anyone using it when the handoff begins does; it is stopped for chat then.

The supervisor's worker effects are recorded rather than made, and ComfyUI is a
constructed backend for the adapter's own part.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS, wait_until
from test_chat_handoff_across_retries import finished
from test_comfyui_kept_after_a_workflow_failure import (
    REFUSED_GRAPH,
    _adapter,
    _ComfyUI,
    _Generation,
    _node_failed,
    _ready_world,
)
from test_generation_retry import choose_retries, failing_media
from test_media_worker_replacement import ReplacementWorld, _image_turn

from local_lm import hardware
from local_lm import orchestrator as orchestrator_module
from local_lm.adapters import comfyui as comfyui_module
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import ModelInstall, ModelProfile
from local_lm.processes import ProcessSupervisor
from local_lm.runtime_provisioning import RuntimeProvisioner
from local_lm.schemas import DeviceInfo, WorkerStopReason

#: What the workers did once a run's device handoff was over.
EMPTIED = ["stop chat", "empty media", "resume profile-chat"]
STOPPED = ["stop chat", "stop media", "resume profile-chat", "start media"]
GIB = 1 << 30
MIB = 1 << 20


def _empties(
    world: ReplacementWorld,
    monkeypatch: pytest.MonkeyPatch,
    answer: bool | Exception = True,
    *,
    queued_meanwhile: int = 0,
) -> None:
    """Give the media engine a way to empty itself, answering `answer`.

    `queued_meanwhile` is how many prompts someone queues from ComfyUI's own
    page while it empties.
    """

    async def release_memory() -> bool:
        world.order.append("empty media")
        world.foreign += queued_meanwhile
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(
        world.services.engines.media, "release_memory", release_memory, raising=False
    )


def _fits(world: ReplacementWorld, monkeypatch: pytest.MonkeyPatch, fits: bool = True) -> None:
    """Say whether the chat model owed back fits beside an emptied worker."""

    async def chat_fits(_profile_id: str) -> bool:
        return fits

    monkeypatch.setattr(world.orchestrator, "_chat_fits_beside_media", chat_fits)


async def _settled(world: ReplacementWorld, effects: int) -> list[str]:
    """The workers' effects once there are `effects` of them and chat is owed nothing."""

    async def read() -> tuple[list[str], str | None]:
        return list(world.order), world.orchestrator._displaced_chat_profile_id

    order, owed = await wait_until(
        read,
        lambda seen: len(seen[0]) >= effects and seen[1] is None,
        what="the device to go back to chat",
    )
    # Anything the handoff starts afterwards, such as an empty worker, has had
    # its moment.
    await asyncio.sleep(0.3)
    assert owed is None
    return list(world.order)


async def _a_finished_render(client: AsyncClient) -> None:
    _chat_id, run_id = await _image_turn(client)
    await finished(client, run_id)


async def test_a_finished_renders_handoff_empties_comfyui_and_loads_chat_beside_it(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)

    await _a_finished_render(client)

    assert await _settled(world, len(EMPTIED)) == EMPTIED
    # The same process, never restarted.
    assert world.media_pid == 31


async def test_a_warm_window_that_ends_empties_comfyui_too(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0.3)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)

    await _a_finished_render(client)

    assert await _settled(world, len(EMPTIED)) == EMPTIED
    assert world.media_pid == 31


async def test_a_window_that_ends_owing_nothing_empties_comfyui_without_a_restart(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat model was loaded beside a worker someone was using; the window ends."""

    world = _ready_world(app, settings, monkeypatch)
    _empties(world, monkeypatch)
    orchestrator = world.orchestrator
    assert orchestrator._open_media_warm_window()
    orchestrator._displaced_chat_profile_id = None

    await orchestrator._end_media_warm_window()
    await asyncio.sleep(0.3)

    assert world.order == ["empty media"]
    assert world.media_pid == 31


async def test_a_window_that_ends_owing_nothing_still_stops_a_worker_for_one_activation(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing else may run on it, so it is not kept; no broad worker replaces it."""

    world = ReplacementWorld(app, settings, monkeypatch)
    _empties(world, monkeypatch)
    orchestrator = world.orchestrator
    assert orchestrator._open_media_warm_window()
    orchestrator._displaced_chat_profile_id = None

    await orchestrator._end_media_warm_window()
    await asyncio.sleep(0.3)

    assert world.order == ["stop media"]


@pytest.mark.parametrize(
    ("emptiable", "owed", "handed_back"),
    [
        pytest.param(True, True, ["empty media", "resume profile-chat"], id="after-a-finished-run"),
        pytest.param(
            False,
            True,
            ["stop media", "resume profile-chat", "start media"],
            id="after-a-failed-attempt",
        ),
        pytest.param(True, False, ["empty media"], id="after-a-finished-run-owing-nothing"),
        pytest.param(
            False, False, ["stop media", "start media"], id="after-a-failed-attempt-owing-nothing"
        ),
    ],
)
async def test_a_worker_held_for_someone_else_is_emptied_only_as_its_handoff_allowed(
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    emptiable: bool,
    owed: bool,
    handed_back: list[str],
) -> None:
    """However long it was held, a failed attempt's worker is stopped once the hold ends."""

    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)
    orchestrator = world.orchestrator
    world.foreign = 1
    await orchestrator._complete_media_handoff(
        "profile-chat", reason=WorkerStopReason.RUN_HANDOFF, emptiable=emptiable
    )
    assert orchestrator._media_warm_window is not None
    assert world.order == []
    world.foreign = 0
    if not owed:
        # A chat model was loaded beside the worker while it was held.
        orchestrator._displaced_chat_profile_id = None

    await orchestrator._end_media_warm_window()

    assert await _settled(world, len(handed_back)) == handed_back


async def test_a_failed_renders_handoff_still_stops_comfyui(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fault may live in the process itself, and only a new one clears it."""

    world = _ready_world(app, settings, monkeypatch)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)
    await choose_retries(client, 0)
    failing_media(app, monkeypatch, 1)

    _chat_id, run_id = await _image_turn(client)
    await finished(client, run_id, "failed")

    assert await _settled(world, len(STOPPED)) == STOPPED


async def test_a_failed_attempt_handed_back_before_its_retry_still_stops_comfyui(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A text turn queued behind a failed attempt gets the device before the retry."""

    world = _ready_world(app, settings, monkeypatch)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)
    media = world.services.engines.media
    original = media.generate
    started = asyncio.Event()
    fail_now = asyncio.Event()
    attempts: list[str] = []

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        attempts.append(request.run_id)
        if len(attempts) == 1:
            started.set()
            await fail_now.wait()
            raise RuntimeError("Temporary media engine failure")
        async for event in original(request):
            yield event

    monkeypatch.setattr(media, "generate", generate)
    _chat_id, run_id = await _image_turn(client)
    await asyncio.wait_for(started.wait(), timeout=PATIENCE_SECONDS)
    other = (await client.post("/api/chats", json={"title": "Meanwhile"})).json()
    question = await client.post(
        f"/api/chats/{other['id']}/turns",
        json={"text": "Name three primary colors.", "mode": "text"},
    )
    assert question.status_code == 202, question.text
    fail_now.set()
    await finished(client, question.json()["run"]["id"])
    await finished(client, run_id)

    # The handoff before the retry is the second effect: a stop, not an emptying.
    assert world.order[:2] == ["stop chat", "stop media"]


async def test_a_worker_launched_for_one_activation_is_still_stopped(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing else may run on it, so it is not kept; no broad worker replaces it."""

    world = ReplacementWorld(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)

    await world.orchestrator._complete_media_handoff(
        "profile-chat", reason=WorkerStopReason.RUN_HANDOFF, emptiable=True
    )
    await asyncio.sleep(0.3)

    assert world.order == ["stop media", "resume profile-chat"]


@pytest.mark.parametrize(
    "answer",
    [False, RuntimeError("ComfyUI did not answer")],
    ids=["did-not-empty", "could-not-ask"],
)
async def test_a_worker_that_is_not_emptied_is_stopped_as_before(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    answer: bool | Exception,
) -> None:
    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch, answer)
    _fits(world, monkeypatch)

    await _a_finished_render(client)

    assert await _settled(world, 1 + len(STOPPED)) == ["stop chat", "empty media", *STOPPED[1:]]


async def test_a_chat_model_that_does_not_fit_beside_the_emptied_worker_gets_the_device(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It would otherwise run from shared system memory, slowly."""

    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch, fits=False)

    await _a_finished_render(client)

    assert await _settled(world, 1 + len(STOPPED)) == ["stop chat", "empty media", *STOPPED[1:]]


async def test_a_chat_model_that_does_not_come_up_beside_the_emptied_worker_gets_the_device(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)
    tries: list[str] = []

    async def load_once_the_device_is_free(profile_id: str) -> None:
        tries.append(profile_id)
        if world.media_running:
            world.order.append(f"resume {profile_id} failed")
            return
        await world.resume(profile_id)

    monkeypatch.setattr(world.orchestrator, "_resume_chat_worker", load_once_the_device_is_free)

    await _a_finished_render(client)

    assert await _settled(world, 6) == [
        "stop chat",
        "empty media",
        "resume profile-chat failed",
        "stop media",
        "resume profile-chat",
        "start media",
    ]
    assert tries == ["profile-chat", "profile-chat"]


async def test_a_failed_stop_after_chat_does_not_come_up_leaves_chat_owed(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)
    world.chat_running = False

    async def does_not_come_up(profile_id: str) -> None:
        world.order.append(f"resume {profile_id} failed")

    async def cannot_stop(name: str, **_options: object) -> Any:
        world.order.append(f"stop {name} failed")
        raise RuntimeError("The worker did not stop")

    monkeypatch.setattr(world.orchestrator, "_resume_chat_worker", does_not_come_up)
    monkeypatch.setattr(world.services.processes, "stop", cannot_stop)

    await world.orchestrator._complete_media_handoff(
        "profile-chat", reason=WorkerStopReason.RUN_HANDOFF, emptiable=True
    )

    assert world.order == ["empty media", "resume profile-chat failed", "stop media failed"]
    assert world.orchestrator._displaced_chat_profile_id == "profile-chat"


@pytest.mark.parametrize(
    ("answer", "fits", "comes_up", "handoff_emptied", "window_emptied"),
    [
        pytest.param(True, True, True, 1, 1, id="emptied"),
        pytest.param(
            True, False, True, 0, 1, id="emptied-then-stopped-for-a-chat-that-does-not-fit"
        ),
        pytest.param(
            True, True, False, 0, 1, id="emptied-then-stopped-for-a-chat-that-fails-there"
        ),
        pytest.param(False, True, True, 0, 0, id="did-not-empty"),
        pytest.param(RuntimeError("ComfyUI did not answer"), True, True, 0, 0, id="could-not-ask"),
    ],
)
async def test_an_emptying_counts_only_where_it_stood_in_for_a_stop(
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    answer: bool | Exception,
    fits: bool,
    comes_up: bool,
    handoff_emptied: int,
    window_emptied: int,
) -> None:
    """A run's handoff, then a window that ends owing nothing, each under its own reason.

    A worker emptied and then stopped anyway, for a chat model that does not
    fit beside it or does not come up there, counts that stop alone.
    """

    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch, answer)
    _fits(world, monkeypatch, fits=fits)
    world.chat_running = False
    orchestrator = world.orchestrator
    supervisor = world.services.processes
    if not comes_up:

        async def load_once_the_device_is_free(profile_id: str) -> None:
            if world.media_running:
                world.order.append(f"resume {profile_id} failed")
                return
            await world.resume(profile_id)

        monkeypatch.setattr(orchestrator, "_resume_chat_worker", load_once_the_device_is_free)

    def emptied() -> dict[WorkerStopReason, int]:
        # The supervisor's own status, not the world's stand-in for it.
        media = next(
            item for item in ProcessSupervisor.statuses(supervisor) if item.name == "media"
        )
        assert media.stops_by_reason == no_stops, "the world's stops reached the supervisor"
        return media.emptied_by_reason

    no_stops = dict.fromkeys(WorkerStopReason, 0)
    assert emptied() == no_stops
    caplog.set_level(logging.INFO, logger="local_lm.processes")

    await orchestrator._complete_media_handoff(
        "profile-chat", reason=WorkerStopReason.RUN_HANDOFF, emptiable=True
    )
    assert emptied() == {**no_stops, WorkerStopReason.RUN_HANDOFF: handoff_emptied}
    stopped = "stop media" in world.order
    assert stopped is not bool(handoff_emptied), world.order

    # The worker a stop restarted is running again before the window opens.
    if orchestrator._media_restart_task is not None:
        await orchestrator._media_restart_task
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0.3)
    assert orchestrator._open_media_warm_window()
    orchestrator._displaced_chat_profile_id = None
    handed_back = len(world.order)
    await orchestrator._end_media_warm_window()

    assert emptied() == {
        **no_stops,
        WorkerStopReason.RUN_HANDOFF: handoff_emptied,
        WorkerStopReason.KEEP_WARM_EXPIRY: window_emptied,
    }
    stopped = "stop media" in world.order[handed_back:]
    assert stopped is not bool(window_emptied), world.order
    lines = [
        record.getMessage() for record in caplog.records if record.name == "local_lm.processes"
    ]
    assert [line for line in lines if "emptied in place" in line] == [
        *["Media worker emptied in place (reason: run_handoff)"] * handoff_emptied,
        *["Media worker emptied in place (reason: keep_warm_expiry)"] * window_emptied,
    ]


async def test_a_chat_model_that_does_not_come_up_with_text_queued_restarts_comfyui_after_it(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The text turn runs on the device first; an empty worker starts once chat is done."""

    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)
    world.chat_running = False
    orchestrator = world.orchestrator

    async def load_once_the_device_is_free(profile_id: str) -> None:
        if world.media_running:
            world.order.append(f"resume {profile_id} failed")
            return
        await world.resume(profile_id)

    monkeypatch.setattr(orchestrator, "_resume_chat_worker", load_once_the_device_is_free)
    monkeypatch.setattr(
        orchestrator, "_handoff_chat_target", lambda _fallback: ("profile-chat", True, False)
    )

    await orchestrator._complete_media_handoff(
        "profile-chat", reason=WorkerStopReason.RUN_HANDOFF, emptiable=True
    )

    assert await _settled(world, 4) == [
        "empty media",
        "resume profile-chat failed",
        "stop media",
        "resume profile-chat",
    ]
    assert orchestrator._media_restart_after_chat_activity is True


@pytest.mark.parametrize(
    ("answer", "fits"),
    [
        pytest.param(False, True, id="did-not-empty"),
        pytest.param(RuntimeError("ComfyUI did not answer"), True, id="could-not-ask"),
        pytest.param(True, False, id="chat-does-not-fit"),
    ],
)
async def test_a_worker_someone_begins_using_while_it_is_emptied_is_held_for_them(
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    answer: bool | Exception,
    fits: bool,
) -> None:
    """Once they are done, it is stopped for chat rather than emptied again."""

    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch, answer, queued_meanwhile=1)
    _fits(world, monkeypatch, fits=fits)
    orchestrator = world.orchestrator

    await orchestrator._complete_media_handoff(
        "profile-chat", reason=WorkerStopReason.RUN_HANDOFF, emptiable=True
    )

    assert world.order == ["empty media"], "their prompt was ended by a stop"
    assert orchestrator._media_warm_window is not None
    assert orchestrator._displaced_chat_profile_id == "profile-chat"
    world.foreign = 0

    await orchestrator._end_media_warm_window()

    assert await _settled(world, 4) == ["empty media", *STOPPED[1:]]


async def test_a_worker_someone_begins_using_while_chat_fails_to_load_beside_it_is_held(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once they are done, it is stopped for chat, the model loaded and an empty worker started."""

    world = _ready_world(app, settings, monkeypatch)
    monkeypatch.setattr(settings, "media_keep_warm_seconds", 0)
    _empties(world, monkeypatch)
    _fits(world, monkeypatch)
    world.chat_running = False
    orchestrator = world.orchestrator

    async def load_once_the_device_is_free(profile_id: str) -> None:
        if world.media_running:
            world.order.append(f"resume {profile_id} failed")
            # Someone queues a prompt from ComfyUI's own page meanwhile.
            world.foreign = 1
            return
        await world.resume(profile_id)

    monkeypatch.setattr(orchestrator, "_resume_chat_worker", load_once_the_device_is_free)

    await orchestrator._complete_media_handoff(
        "profile-chat", reason=WorkerStopReason.RUN_HANDOFF, emptiable=True
    )

    assert world.order == ["empty media", "resume profile-chat failed"]
    assert orchestrator._media_warm_window is not None
    assert orchestrator._media_warm_stop_reason == WorkerStopReason.CHAT_LOAD
    assert orchestrator._displaced_chat_profile_id == "profile-chat"
    world.foreign = 0

    await orchestrator._end_media_warm_window()

    assert await _settled(world, 5) == [
        "empty media",
        "resume profile-chat failed",
        "stop media",
        "resume profile-chat",
        "start media",
    ]


@pytest.mark.parametrize(
    ("queued_meanwhile", "held"),
    [pytest.param(1, True, id="someone-begins-meanwhile"), pytest.param(0, False, id="nobody")],
)
async def test_a_window_owing_nothing_whose_worker_is_not_emptied_is_held_only_for_someone(
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    queued_meanwhile: int,
    held: bool,
) -> None:
    """A chat model was loaded beside the worker; the window ends and the emptying fails."""

    world = _ready_world(app, settings, monkeypatch)
    _empties(world, monkeypatch, False, queued_meanwhile=queued_meanwhile)
    orchestrator = world.orchestrator
    assert orchestrator._open_media_warm_window()
    orchestrator._displaced_chat_profile_id = None

    await orchestrator._end_media_warm_window()

    if held:
        assert world.order == ["empty media"], "their prompt was ended by a stop"
        assert orchestrator._media_warm_window is not None
        world.foreign = 0
        await orchestrator._end_media_warm_window()
    assert await _settled(world, 3) == ["empty media", "stop media", "start media"]


async def test_a_window_owing_nothing_whose_worker_is_emptied_holds_it_for_nobody(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Someone begins using it as it empties, and there is nothing left to stop."""

    world = _ready_world(app, settings, monkeypatch)
    _empties(world, monkeypatch, queued_meanwhile=1)
    orchestrator = world.orchestrator
    assert orchestrator._open_media_warm_window()
    orchestrator._displaced_chat_profile_id = None

    await orchestrator._end_media_warm_window()
    await asyncio.sleep(0.3)

    assert world.order == ["empty media"]
    assert orchestrator._media_warm_window is None
    assert world.media_pid == 31


@pytest.mark.parametrize(
    ("estimate", "free", "fits"),
    [
        pytest.param(8 * GIB, 12 * GIB, True, id="room-to-spare"),
        pytest.param(8 * GIB, 8 * GIB, False, id="no-margin"),
        pytest.param(8 * GIB, None, False, id="free-memory-unknown"),
        pytest.param(None, 12 * GIB, False, id="estimate-unknown"),
    ],
)
async def test_a_chat_model_fits_beside_the_emptied_worker_only_with_room_to_spare(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    estimate: int | None,
    free: int | None,
    fits: bool,
) -> None:
    with SessionLocal() as session:
        install = ModelInstall(
            id="model_beside_media",
            name="Chat beside media",
            role="chat",
            engine="llama.cpp",
            local_path="synthetic",
            active=True,
        )
        profile = ModelProfile(
            id="profile_beside_media",
            name="Chat beside media",
            role="chat",
            engine="llama.cpp",
            model_install_id=install.id,
            load_settings_json={"context_length": 4096},
        )
        session.add_all([install, profile])
        session.commit()
    services = app.state.services
    monkeypatch.setattr(
        services.processes, "chat_memory_estimate", lambda _profile, _install: estimate
    )
    monkeypatch.setattr(orchestrator_module, "first_device_free_memory", lambda: free)

    assert await services.orchestrator._chat_fits_beside_media("profile_beside_media") is fits


async def test_the_chat_memory_estimate_is_what_a_chat_load_records(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + bytes(4096))
    executable = tmp_path / "llama-server.exe"
    executable.write_bytes(b"runtime")
    settings.llama_executable = executable
    supervisor = ProcessSupervisor(settings, Mock(spec=RuntimeProvisioner))
    recorded: list[int | None] = []

    async def replace(
        _name: str,
        _command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
        **_options: object,
    ) -> None:
        recorded.append(estimated_memory_bytes)

    monkeypatch.setattr(supervisor, "_replace", replace)
    profile = ModelProfile(
        id="profile_estimate",
        model_install_id="model_estimate",
        name="Estimate",
        role="chat",
        engine="llama.cpp",
        load_settings_json={"context_length": 8192},
    )
    install = ModelInstall(
        id="model_estimate",
        name="Estimate",
        role="chat",
        engine="llama.cpp",
        local_path=str(model),
        active=True,
    )

    await supervisor.load_chat(profile, install)

    assert recorded == [supervisor.chat_memory_estimate(profile, install)]
    assert isinstance(recorded[0], int) and recorded[0] > model.stat().st_size
    other_engine = ModelProfile(
        id="profile_other", name="Other", role="chat", engine="vllm", load_settings_json={}
    )
    assert supervisor.chat_memory_estimate(other_engine, install) is None
    missing = ModelInstall(
        id="model_missing",
        name="Missing",
        role="chat",
        engine="llama.cpp",
        local_path=str(tmp_path / "missing.gguf"),
        active=True,
    )
    assert supervisor.chat_memory_estimate(profile, missing) is None


def test_the_free_memory_beside_chat_is_read_from_the_first_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = [
        DeviceInfo(id=f"cuda:{index}", name="GPU", kind="gpu", available_memory_bytes=free)
        for index, free in enumerate((5 * GIB, 9 * GIB))
    ]
    monkeypatch.setattr(hardware, "_nvidia_devices", lambda: devices)
    assert hardware.first_device_free_memory() == 5 * GIB
    monkeypatch.setattr(hardware, "_nvidia_devices", lambda: [])
    assert hardware.first_device_free_memory() is None


class _FreeingComfyUI(_ComfyUI):
    """The constructed ComfyUI, answering requests to free its memory.

    `free` is how it answers each request, `counts` the free device memory
    each read of its statistics reports in turn, the last one repeating, a
    count of None being an answer without one, and `stats_status` how the
    statistics are answered.
    """

    def __init__(
        self,
        *,
        free: int = 200,
        counts: Sequence[int | None] = (),
        stats_status: int = 200,
        **fields: Any,
    ) -> None:
        super().__init__(**fields)
        self.free = free
        self.counts = list(counts)
        self.stats_status = stats_status
        self.asked: list[Any] = []
        self.reads = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        route = (request.method, request.url.path)
        if route == ("POST", "/free"):
            self.asked.append(json.loads(request.content or b"{}"))
            return httpx.Response(self.free, json={})
        if route == ("GET", "/system_stats"):
            self.reads += 1
            count = self.counts[min(self.reads, len(self.counts)) - 1] if self.counts else None
            device: dict[str, Any] = {"name": "cuda:0", "type": "cuda", "index": 0}
            if count is not None:
                device["vram_free"] = count
            return httpx.Response(self.stats_status, json={"system": {}, "devices": [device]})
        return await super().__call__(request)


class _MissedFirstAsk(_FreeingComfyUI):
    """A ComfyUI whose worker missed the first request and acts on the second."""

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if (request.method, request.url.path) == ("GET", "/system_stats"):
            self.reads += 1
            count = 14 * GIB if len(self.asked) >= 2 else 4 * GIB
            device = {"name": "cuda:0", "type": "cuda", "index": 0, "vram_free": count}
            return httpx.Response(200, json={"system": {}, "devices": [device]})
        return await super().__call__(request)


ASK = {"unload_models": True, "free_memory": True}


@pytest.mark.parametrize(
    ("comfy", "emptied", "asks"),
    [
        pytest.param(
            lambda: _FreeingComfyUI(counts=[4 * GIB, 14 * GIB, 14 * GIB]),
            True,
            1,
            id="rose-and-settled",
        ),
        pytest.param(
            lambda: _FreeingComfyUI(counts=[4 * GIB, 9 * GIB, 14 * GIB, 12 * GIB, 12 * GIB]),
            True,
            1,
            id="settles-only-once-it-stops-moving",
        ),
        pytest.param(lambda: _MissedFirstAsk(), True, 2, id="first-ask-missed"),
        pytest.param(
            lambda: _FreeingComfyUI(counts=[14 * GIB]), False, 2, id="never-rose-asked-twice"
        ),
        pytest.param(lambda: _FreeingComfyUI(free=500, counts=[4 * GIB]), False, 1, id="refused"),
        pytest.param(lambda: _FreeingComfyUI(counts=[None]), False, 0, id="count-unreadable"),
        pytest.param(
            lambda: _FreeingComfyUI(counts=[4 * GIB], stats_status=500),
            False,
            0,
            id="statistics-refused",
        ),
        pytest.param(
            lambda: _FreeingComfyUI(counts=[GIB * (n + 1) for n in range(200)]),
            False,
            1,
            id="still-rising-at-the-deadline",
        ),
        pytest.param(
            lambda: _FreeingComfyUI(
                counts=[4 * GIB, 14 * GIB, *(14 * GIB - n * 128 * MIB for n in range(1, 60))]
            ),
            False,
            1,
            id="falling-again-after-its-rise",
        ),
    ],
)
async def test_comfyui_counts_as_emptied_only_once_its_free_memory_rose_and_settled(
    monkeypatch: pytest.MonkeyPatch,
    comfy: Callable[[], _FreeingComfyUI],
    emptied: bool,
    asks: int,
) -> None:
    monkeypatch.setattr(comfyui_module, "RELEASE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(comfyui_module, "RELEASE_ASK_AGAIN_SECONDS", 0.05)
    monkeypatch.setattr(comfyui_module, "RELEASE_GIVE_UP_SECONDS", 0.3)
    monkeypatch.setattr(comfyui_module, "RELEASE_SECONDS", 0.6)
    backend = comfy()
    adapter = await _adapter(monkeypatch, backend)
    # As after a render ComfyUI took, whose models it may still hold.
    adapter._prompts_since_emptied = 1
    try:
        assert await adapter.release_memory() is emptied
    finally:
        await adapter.close()

    assert backend.asked == [ASK] * asks


async def test_a_count_that_crept_past_the_mark_must_still_hold_still(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two readings past the mark, close together, show the release; one is not enough."""

    monkeypatch.setattr(comfyui_module, "RELEASE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(comfyui_module, "RELEASE_SECONDS", 0.6)
    backend = _FreeingComfyUI(
        counts=[
            4 * GIB,
            4 * GIB + 200 * MIB,
            4 * GIB + 260 * MIB,
            *(5 * GIB + n * GIB for n in range(200)),
        ]
    )
    adapter = await _adapter(monkeypatch, backend)
    adapter._prompts_since_emptied = 1
    try:
        assert await adapter.release_memory() is False
    finally:
        await adapter.close()

    assert backend.asked == [ASK]


async def test_comfyui_once_emptied_has_nothing_to_wait_for_until_it_takes_another_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comfyui_module, "RELEASE_POLL_SECONDS", 0.01)
    backend = _FreeingComfyUI(counts=[4 * GIB, 14 * GIB])
    adapter = await _adapter(monkeypatch, backend)
    adapter._prompts_since_emptied = 1
    try:
        assert await adapter.release_memory() is True
        reads = backend.reads
        assert await adapter.release_memory() is True
    finally:
        await adapter.close()

    assert backend.asked == [ASK, ASK]
    assert backend.reads == reads


async def test_a_count_that_has_not_risen_by_the_give_up_is_not_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing came back after both requests, so the worker is stopped rather than waited on."""

    monkeypatch.setattr(comfyui_module, "RELEASE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(comfyui_module, "RELEASE_ASK_AGAIN_SECONDS", 0.05)
    monkeypatch.setattr(comfyui_module, "RELEASE_GIVE_UP_SECONDS", 0.3)
    monkeypatch.setattr(comfyui_module, "RELEASE_SECONDS", 5.0)
    # It would rise only after a hundred readings, a second at the least.
    backend = _FreeingComfyUI(counts=[4 * GIB] * 100 + [14 * GIB])
    adapter = await _adapter(monkeypatch, backend)
    adapter._prompts_since_emptied = 1
    try:
        assert await adapter.release_memory() is False
    finally:
        await adapter.close()

    assert backend.asked == [ASK, ASK]
    assert backend.reads < 100


async def test_a_comfyui_that_took_nothing_of_ours_since_it_was_emptied_is_not_waited_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No rise would come; whatever someone else left there is still asked to go."""

    backend = _FreeingComfyUI(counts=[14 * GIB])
    adapter = await _adapter(monkeypatch, backend)
    try:
        assert await adapter.release_memory() is True
    finally:
        await adapter.close()

    assert backend.asked == [ASK]
    assert backend.reads == 0


@pytest.mark.parametrize(
    ("generation", "emptied", "asks"),
    [
        pytest.param(
            _Generation(answer=lambda: httpx.Response(400, json=REFUSED_GRAPH)),
            True,
            1,
            id="graph-refused",
        ),
        pytest.param(
            _Generation(events=(_node_failed("ValueError", "The scheduler is not one it knows"),)),
            False,
            2,
            id="prompt-taken",
        ),
    ],
)
async def test_only_a_prompt_comfyui_took_leaves_a_release_to_wait_for(
    monkeypatch: pytest.MonkeyPatch, generation: _Generation, emptied: bool, asks: int
) -> None:
    """A graph refused at validation loads nothing; a prompt taken may have loaded its models."""

    monkeypatch.setattr(comfyui_module, "_HISTORY_SETTLE_SECONDS", 0.05)
    monkeypatch.setattr(comfyui_module, "_HISTORY_POLL_SECONDS", 0.01)
    monkeypatch.setattr(comfyui_module, "RELEASE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(comfyui_module, "RELEASE_ASK_AGAIN_SECONDS", 0.05)
    monkeypatch.setattr(comfyui_module, "RELEASE_GIVE_UP_SECONDS", 0.3)
    # The count never moves, as when nothing comes back.
    backend = _FreeingComfyUI(counts=[14 * GIB], answer=generation.answer)
    adapter = await _adapter(monkeypatch, backend, generation.events)
    try:
        with pytest.raises(RuntimeError):
            [event async for event in adapter.generate(generation.request)]
        assert await adapter.release_memory() is emptied
    finally:
        await adapter.close()

    assert backend.asked == [ASK] * asks
