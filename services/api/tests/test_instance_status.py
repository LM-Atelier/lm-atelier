"""One status read tells a script where the running instance stands.

`GET /api/instance/status` says which instance and which start this is,
whether startup has finished, whether the workspace is locked, whether each
worker is up, on its way, down on purpose or down by failure, how each queue
lane dispatches, how much room is left, and, for a named chat, whether a plain
turn would run each workflow. Where another route already gives an answer, the
two are compared here so they cannot drift, and the read is held to asking no
engine anything and to carrying no title, prompt or file path.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import Any, get_args

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from run_waits import PATIENCE_SECONDS, wait_for_terminal_status
from sqlalchemy import event, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from test_guide_video_selection import _family as _video_family
from test_guide_video_selection import _graph as _video_graph
from test_media_worker_start_and_stop_reasons import (
    HEALTH,
    LAUNCH,
    _fake_workers,
    _StubbornProcess,
)
from test_setup_verification import seed_ready_role
from test_worker_unload import _StartupRestore
from test_workflow_revision_review import _approved
from test_workspace_lock_api import CHANGED, EPOCH, LOCKED, _choose, _lock
from workflow_fixtures import seed_workflow_trust

import local_lm
from local_lm import __version__, api, db, worker_startup
from local_lm import main as application
from local_lm.artifact_deletion_authority import ArtifactReferenceIndexStale
from local_lm.config import Settings
from local_lm.db import SQLITE_BUSY_TIMEOUT_MS, SessionLocal
from local_lm.domain import Operation
from local_lm.filesystem_links import AnchoredDirectoryError
from local_lm.generation_experiments_v1 import EXPERIMENT_CHAT_SCOPE
from local_lm.instance_identity import INSTANCE_ID_HEADER
from local_lm.main import PUBLIC_API_PATHS, create_app
from local_lm.models import (
    Chat,
    GenerationQueuePolicy,
    Job,
    ModelInstall,
    ModelProfile,
    Project,
    SetupVerification,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.processes import ProcessSupervisor, WorkerStopIncomplete
from local_lm.schemas import TurnRequest, WorkerStartReason, WorkerStopReason
from local_lm.settings_registry import builtin_settings_for_role
from local_lm.setup_verification import SETUP_VERIFICATION_SCOPE
from local_lm.workspace_lock import REACHABLE_WHILE_LOCKED

STATUS = "/api/instance/status"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TITLE = "Neutral title sentinel 41c7"
DRAFT = "Neutral draft sentinel 9e2b"
FAMILY = "Neutral family sentinel 5d8a"
STATUS_KEYS = {
    "instance_id",
    "boot_id",
    "started_at",
    "build",
    "startup",
    "workspace_lock",
    "workers",
    "lanes",
    "storage",
    "chat",
}


@asynccontextmanager
async def _serving(
    app: FastAPI, *, settled: bool = True
) -> AsyncIterator[tuple[AsyncClient, dict[str, Any]]]:
    """The application started, and a client holding a session, with that session's answer.

    Settled waits for the retention sweep as the shared client does, so the
    store is quiet; a test of a sweep that never ends does not wait for it.
    """

    async with app.router.lifespan_context(app):
        if settled:
            await asyncio.wait_for(app.state.retention_sweep, timeout=PATIENCE_SECONDS)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            session = await client.post("/api/session")
            assert session.status_code == 200
            client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
            yield client, session.json()


async def _status(client: AsyncClient, chat_id: str | None = None) -> dict[str, Any]:
    response = await client.get(STATUS, params={} if chat_id is None else {"chat_id": chat_id})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def _worker(body: dict[str, Any], name: str) -> dict[str, Any]:
    found: dict[str, Any] = next(worker for worker in body["workers"] if worker["name"] == name)
    return found


def _tasks(body: dict[str, Any]) -> dict[str, str]:
    return {task["task"]: task["state"] for task in body["startup"]["tasks"]}


def _operations(body: dict[str, Any], capability: str) -> dict[str, dict[str, Any]]:
    entry = next(
        workflow
        for workflow in body["chat"]["workflows"]
        if workflow["selector_capability"] == capability
    )
    return {operation["operation"]: operation for operation in entry["operations"]}


async def _restore_ended(app: FastAPI) -> None:
    """Wait for the startup restore, and the runtime check if there is one, to end."""

    progress = app.state.startup
    await asyncio.wait_for(asyncio.shield(progress.worker_restore), timeout=PATIENCE_SECONDS)
    if progress.runtime_verification is not None:
        await asyncio.wait_for(
            asyncio.shield(progress.runtime_verification), timeout=PATIENCE_SECONDS
        )


def _settings_beside(settings: Settings, template: Path, name: str) -> Settings:
    """Settings for another data folder next to this test's, already migrated."""

    data_dir = settings.data_dir.parent / name
    (data_dir / "state").mkdir(parents=True)
    shutil.copy2(template, data_dir / "state" / "local-lm.sqlite3")
    return Settings(data_dir=data_dir, dev=True, chat_engine="mock", media_engine="mock")


async def _refusing_media_engine(app: FastAPI) -> list[str]:
    """Answer every request the ComfyUI adapter makes with a refusal, noting each path."""

    sent: list[str] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.path)
        return httpx.Response(503)

    media = app.state.services.engines.media
    await media._client.aclose()
    media._client = httpx.AsyncClient(
        base_url="http://comfy.test", transport=httpx.MockTransport(refuse)
    )
    return sent


async def _family_of(client: AsyncClient, workflow_id: str) -> dict[str, Any]:
    families = await client.get("/api/workflow-families", params={"selector_capability": "image"})
    assert families.status_code == 200, families.text
    family: dict[str, Any] = next(
        family
        for family in families.json()
        if any(variant["id"] == workflow_id for variant in family["variants"])
    )
    return family


async def _select_family(client: AsyncClient, chat_id: str, family_id: str) -> None:
    selected = await client.put(
        f"/api/chats/{chat_id}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )
    assert selected.status_code == 200, selected.text


def _reviewing_runtime(app: FastAPI, patch: pytest.MonkeyPatch) -> None:
    """A media worker that reads as ready and describes the one node `_approved` builds on.

    The same runtime the revision review tests approve against, applied only
    for as long as `patch` is held.
    """

    services = app.state.services
    statuses = services.processes.statuses
    patch.setattr(
        services.processes,
        "statuses",
        lambda: [
            worker.model_copy(
                update={"managed": True, "running": True, "state": "ready", "pid": 43210}
            )
            if worker.name == "media"
            else worker
            for worker in statuses()
        ],
    )

    async def object_info() -> dict[str, Any]:
        return {
            "EmptyLatentImage": {
                "python_module": "nodes",
                "input": {
                    "required": {
                        "width": ["INT", {"default": 512}],
                        "height": ["INT", {"default": 512}],
                        "batch_size": ["INT", {"default": 1}],
                    }
                },
                "output": ["LATENT"],
            }
        }

    patch.setattr(services.engines.media, "object_info", object_info, raising=False)


#: What a statement that writes, or takes the writer, begins with.
_WRITING = ("INSERT", "UPDATE", "DELETE", "REPLACE", "BEGIN IMMEDIATE", "BEGIN EXCLUSIVE")


@contextmanager
def _statements_the_read_writes(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Every statement the status read's database work sends that writes or takes the writer.

    Only what runs inside that work counts: the application's own upkeep goes
    on writing beside it, as it drains what a test's setup recorded.
    """

    written: list[str] = []
    inside = threading.local()
    reads = api._instance_reads

    def marked(*args: Any, **options: Any) -> Any:
        inside.reading = True
        try:
            return reads(*args, **options)
        finally:
            inside.reading = False

    def note(_connection: object, _cursor: object, statement: str, *_rest: object) -> None:
        spoken = " ".join(statement.split())
        if getattr(inside, "reading", False) and spoken.upper().startswith(_WRITING):
            written.append(spoken[:80])

    engine = db.engine
    with monkeypatch.context() as patch:
        patch.setattr(api, "_instance_reads", marked)
        event.listen(engine, "before_cursor_execute", note)
        try:
            yield written
        finally:
            event.remove(engine, "before_cursor_execute", note)


async def _pinned_chat(client: AsyncClient) -> tuple[str, str, str, str]:
    """A chat whose project pins a revision that a later, identical revision has succeeded.

    Returns the chat, the project, the pinned revision and its successor.
    """

    graph = {"1": {"class_type": "SaveImage", "inputs": {}}}
    workflow = await _created_workflow(client, "Neutral pinned workflow", api_graph=graph)
    pinned = workflow["current_revision_id"]
    seed_workflow_trust(pinned)
    chat_id, project_id = await _pinning_chat(client, pinned)
    successor = await client.post(
        f"/api/workflows/{workflow['id']}/revisions", json={"api_graph": graph}
    )
    assert successor.status_code == 201, successor.text
    seed_workflow_trust(successor.json()["id"])
    assert successor.json()["id"] != pinned
    return chat_id, project_id, pinned, successor.json()["id"]


async def _pinning_chat(client: AsyncClient, revision_id: str) -> tuple[str, str]:
    """A chat that takes its pictures from its project, which pins this revision.

    Returns the chat and the project.
    """

    project = await client.post(
        "/api/projects", json={"name": "Neutral", "image_workflow_revision_id": revision_id}
    )
    assert project.status_code == 201, project.text
    project_id: str = project.json()["id"]
    chat = (
        await client.post("/api/chats", json={"title": "Neutral", "project_id": project_id})
    ).json()
    inherited = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image", json={"mode": "default"}
    )
    assert inherited.status_code == 200, inherited.text
    chat_id: str = chat["id"]
    return chat_id, project_id


async def _created_workflow(client: AsyncClient, name: str, **fields: Any) -> dict[str, Any]:
    created = await client.post(
        "/api/workflows",
        json={
            "name": name,
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "Mock"}},
            **fields,
        },
    )
    assert created.status_code == 201, created.text
    workflow: dict[str, Any] = created.json()
    return workflow


async def _ended_run(client: AsyncClient, run_id: str, expected: str) -> dict[str, Any]:
    async def read() -> dict[str, Any]:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        run: dict[str, Any] = response.json()
        return run

    return dict(await wait_for_terminal_status(read, what=f"run {run_id}", expected=expected))


async def _automatic(client: AsyncClient, chat_id: str, capability: str) -> None:
    chosen = await client.put(
        f"/api/chats/{chat_id}/workflow-selections/{capability}", json={"mode": "automatic"}
    )
    assert chosen.status_code == 200, chosen.text


def _only_new_families_offered(session: Session, capability: str) -> None:
    """Withdraw every family offered for this selector until now, so a test offers its own.

    A default must stay offered, so the default is withdrawn with it.
    """

    session.execute(
        update(WorkflowPreference)
        .where(WorkflowPreference.selector_capability == capability)
        .values(enabled=False, is_default=False)
    )


def _edit_family(
    session: Session, name: str, input_schema: dict[str, Any], *, is_default: bool
) -> WorkflowRevision:
    """A picture edit family of its own, offered for the image selector."""

    family = WorkflowFamily(name=name, use_case="neutral edits")
    definition = WorkflowDefinition(
        family=family,
        variant_key="edit",
        name=f"{name} edit",
        operation=Operation.IMAGE_TO_IMAGE.value,
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="mock",
        api_graph_json={},
        input_schema_json=input_schema,
        trusted=True,
    )
    preference = WorkflowPreference(
        family=family, selector_capability="image", is_default=is_default
    )
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return revision


async def test_the_instance_id_outlives_a_restart_and_each_start_has_its_own_boot(
    settings: Settings, migrated_database_template: Path
) -> None:
    starts: list[tuple[dict[str, Any], str, str, str]] = []
    for _start in range(2):
        app = create_app(settings)
        async with _serving(app) as (client, session):
            ready = await client.get("/api/ready")
            workers = (await client.get("/api/workers")).json()
            starts.append(
                (
                    await _status(client),
                    ready.headers[INSTANCE_ID_HEADER],
                    session["event_epoch"],
                    workers[0]["counted_since"],
                )
            )

    for body, header, epoch, counted_since in starts:
        assert body["instance_id"] == header
        assert body["boot_id"] == epoch
        assert body["started_at"] == counted_since
    (first, _, _, _), (second, _, _, _) = starts
    assert first["instance_id"] == second["instance_id"]
    assert first["boot_id"] != second["boot_id"]

    elsewhere = create_app(_settings_beside(settings, migrated_database_template, "elsewhere"))
    async with _serving(elsewhere) as (client, _session):
        other = await _status(client)
    assert other["instance_id"] != first["instance_id"]


async def test_startup_runs_while_the_worker_restore_is_held_and_completes_once_it_ends(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    restore = _StartupRestore(app, monkeypatch)

    async with _serving(app) as (client, _session):
        if app.state.startup.runtime_verification is not None:
            await asyncio.wait_for(
                asyncio.shield(app.state.startup.runtime_verification), timeout=PATIENCE_SECONDS
            )
        held = await _status(client)
        restore.go.set()
        await _restore_ended(app)
        ended = await _status(client)

    assert held["startup"]["state"] == "running"
    assert _tasks(held) == {
        "worker_restore": "running",
        "runtime_verification": "done",
        "reference_index": "done",
        "retention_sweep": "done",
    }
    assert [(worker["restore"], worker["condition"]) for worker in held["workers"]] == [
        ("pending", "not_started"),
        ("pending", "not_started"),
    ]
    assert ended["startup"]["state"] == "complete"
    assert set(_tasks(ended).values()) == {"done"}
    assert [(worker["restore"], worker["condition"]) for worker in ended["workers"]] == [
        ("not_configured", "not_started"),
        ("not_configured", "not_started"),
    ]


async def test_startup_stalls_when_reference_index_upkeep_ends_before_the_index_is_ready(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def upkeep_that_gives_up(*_args: object, **_options: object) -> None:
        # As upkeep ends when the index cannot be certified: readiness is never set.
        return None

    monkeypatch.setattr(application, "maintain_reference_index", upkeep_that_gives_up)

    async with _serving(app, settled=False) as (client, _session):
        await _restore_ended(app)
        await asyncio.wait_for(
            asyncio.shield(app.state.startup.reference_index), timeout=PATIENCE_SECONDS
        )
        body = await _status(client)

    assert body["startup"]["state"] == "stalled"
    assert _tasks(body) == {
        "worker_restore": "done",
        "runtime_verification": "done",
        "reference_index": "gave_up",
        "retention_sweep": "waiting",
    }


@pytest.mark.parametrize("outcome", ["failed", "skipped_for_release"])
async def test_a_restore_that_failed_reads_failing_and_one_a_release_skipped_reads_not_started(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    restore = _StartupRestore(app, monkeypatch)
    if outcome == "failed":
        restore.start_media.side_effect = RuntimeError("The neutral worker could not start.")
        restore.load_chat.side_effect = RuntimeError("The neutral model could not load.")

    async with _serving(app) as (client, _session):
        restore.configure_both(settings, monkeypatch)
        if outcome == "skipped_for_release":
            unloaded = await client.post("/api/workers/unload")
            assert unloaded.status_code == 200, unloaded.text
        restore.go.set()
        await _restore_ended(app)
        body = await _status(client)

    condition = "failing" if outcome == "failed" else "not_started"
    for name in ("chat", "media"):
        worker = _worker(body, name)
        assert (worker["restore"], worker["condition"], worker["last_end"]) == (
            outcome,
            condition,
            None,
        ), name


async def test_the_lock_refuses_the_status_read_as_it_refuses_the_rest_of_the_api(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()
    assert (await _choose(client, 0)).status_code == 200
    opened_epoch = (await client.get("/api/privacy/status")).json()["lock_epoch"]
    await _lock(client)

    for params in ({}, {"chat_id": chat["id"]}, {"chat_id": "chat_missing"}):
        refused = await client.get(STATUS, params=params)
        assert (refused.status_code, refused.json()) == (423, LOCKED), params
        assert refused.headers["cache-control"] == "no-store", params

    assert (await client.post("/api/privacy/unlock")).status_code == 200
    stale = await client.get(STATUS, headers={EPOCH: opened_epoch})
    assert (stale.status_code, stale.json()) == (423, CHANGED)

    opened = await client.get(STATUS)
    privacy = (await client.get("/api/privacy/status")).json()
    assert opened.status_code == 200, opened.text
    assert opened.headers["cache-control"] == "no-store"
    assert opened.json()["workspace_lock"] == privacy
    assert (privacy["enabled"], privacy["locked"]) == (True, False)


async def test_a_worker_stopped_on_purpose_reads_stopped_and_one_that_failed_reads_failing(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor: ProcessSupervisor = app.state.services.processes
    spawned = _fake_workers(supervisor, monkeypatch)
    first = await _status(client)
    for name in ("chat", "media"):
        assert _worker(first, name) == {
            "name": name,
            "condition": "not_started",
            "restore": "not_configured",
            "last_stop_reason": None,
            "last_end": None,
            "failure_code": None,
        }

    on_purpose = [reason for reason in WorkerStopReason if reason is not WorkerStopReason.UNHEALTHY]
    for reason in on_purpose:
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
        assert _worker(await _status(client), "media")["condition"] == "ready", reason
        await supervisor.stop("media", reason=reason)
        stopped = _worker(await _status(client), "media")
        assert (stopped["condition"], stopped["last_stop_reason"], stopped["last_end"]) == (
            "stopped",
            reason.value,
            "stop",
        ), reason

    # Ending by itself after a stop on purpose leaves that stop as the last
    # stop reason, which is why the end is read as well: once the ended
    # process is cleared away, which counts no stop, nothing else says it failed.
    await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
    spawned[-1].end(1)
    crashed = _worker(await _status(client), "media")
    await supervisor.stop("media", reason=WorkerStopReason.MANUAL)
    cleared = _worker(await _status(client), "media")
    for after_the_crash in (crashed, cleared):
        assert (
            after_the_crash["condition"],
            after_the_crash["last_stop_reason"],
            after_the_crash["last_end"],
        ) == ("failing", on_purpose[-1].value, "exit")
    workers = {worker["name"]: worker for worker in (await client.get("/api/workers")).json()}
    assert workers["media"]["state"] == "stopped", "the cleared worker reads only as stopped"

    await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
    await supervisor.stop("media", reason=WorkerStopReason.UNHEALTHY)
    unhealthy = _worker(await _status(client), "media")
    assert (unhealthy["condition"], unhealthy["last_stop_reason"], unhealthy["last_end"]) == (
        "failing",
        "unhealthy",
        "stop",
    )


async def test_the_health_probe_degrades_where_the_status_read_reports_a_worker_stopped_on_purpose(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    comfyui = Settings(
        data_dir=settings.data_dir, dev=True, chat_engine="mock", media_engine="comfyui"
    )
    app = create_app(comfyui)
    sent = await _refusing_media_engine(app)

    async with _serving(app) as (client, _session):
        supervisor: ProcessSupervisor = app.state.services.processes
        _fake_workers(supervisor, monkeypatch)
        await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
        await supervisor.stop("media", reason=WorkerStopReason.MANUAL)
        health = (await client.get("/api/health")).json()
        asked = len(sent)
        body = await _status(client)
        after = list(sent)

    assert health["status"] == "degraded"
    assert "/system_stats" in after[:asked], "the health probe asks ComfyUI itself"
    assert after[asked:] == []
    assert _worker(body, "media")["condition"] == "stopped"


async def test_an_imported_workflow_reads_review_required_until_it_is_trusted(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral exported workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "Mock"}},
        },
    )
    assert created.status_code == 201, created.text
    bundle = (await client.get(f"/api/workflows/{created.json()['id']}/export")).json()
    bundle["name"] = "Neutral imported workflow"
    imported = await client.post("/api/workflows/import", json=bundle)
    assert imported.status_code == 201, imported.text
    workflow = imported.json()
    family = await _family_of(client, workflow["id"])
    chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()
    await _select_family(client, chat["id"], family["id"])
    turn = {"text": "Draw a blue cup", "mode": "image"}

    untrusted = await _status(client, chat["id"])
    selections = (await client.get(f"/api/chats/{chat['id']}/workflow-selections")).json()
    refused = await client.post(f"/api/chats/{chat['id']}/turns", json=turn)

    for entry in untrusted["chat"]["workflows"]:
        selection = next(
            item
            for item in selections
            if item["selector_capability"] == entry["selector_capability"]
        )
        assert (entry["mode"], entry["workflow_family_id"]) == (
            selection["mode"],
            selection["workflow_family_id"],
        ), entry["selector_capability"]
    # The mock chat engine loads no model, so chat needs no workflow and nothing installed.
    assert _operations(untrusted, "chat")["text"] == {
        "operation": "text",
        "readiness": "ready",
        "trust": "not_applicable",
        "reason": None,
        "workflow_revision_id": None,
    }
    image = _operations(untrusted, "image")
    assert image["text_to_image"] == {
        "operation": "text_to_image",
        "readiness": "review_required",
        "trust": "untrusted",
        "reason": "revision_untrusted",
        "workflow_revision_id": None,
    }
    # The family makes pictures from words only, so it has no edit to offer.
    assert (image["image_to_image"]["readiness"], image["image_to_image"]["reason"]) == (
        "unavailable",
        "variant_missing",
    )
    assert refused.status_code == 422, refused.text

    seed_workflow_trust(workflow["current_revision_id"])
    trusted = _operations(await _status(client, chat["id"]), "image")
    accepted = await client.post(f"/api/chats/{chat['id']}/turns", json=turn)

    assert trusted["text_to_image"] == {
        "operation": "text_to_image",
        "readiness": "ready",
        "trust": "trusted",
        "reason": None,
        "workflow_revision_id": workflow["current_revision_id"],
    }
    assert accepted.status_code == 202, accepted.text
    assert accepted.json()["run"]["workflow_revision_id"] == workflow["current_revision_id"]


async def test_a_workflow_missing_a_node_package_reads_setup_required_and_its_run_fails(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral packaged workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "Mock"}},
            "dependencies": {"custom_nodes": ["neutral-missing-pack"]},
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    seed_workflow_trust(workflow["current_revision_id"])
    family = await _family_of(client, workflow["id"])
    chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()
    await _select_family(client, chat["id"], family["id"])

    image = _operations(await _status(client, chat["id"]), "image")
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Draw a blue cup", "mode": "image"}
    )

    assert image["text_to_image"] == {
        "operation": "text_to_image",
        "readiness": "setup_required",
        "trust": "trusted",
        "reason": "node_packages_missing",
        "workflow_revision_id": workflow["current_revision_id"],
    }
    # Admission does not look at node packages; the run does as it starts.
    assert accepted.status_code == 202, accepted.text
    run_id = accepted.json()["run"]["id"]

    async def read() -> dict[str, Any]:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, response.text
        run: dict[str, Any] = response.json()
        return run

    run = await wait_for_terminal_status(read, what=f"run {run_id}", expected="failed")
    assert "neutral-missing-pack" in str(run["error"])


async def test_an_unreviewed_comfyui_revision_a_chat_falls_back_to_reads_review_required(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chosen through the profile a chat falls back to, so selection never sees its review.

    A run checks the review before it starts, as `_studio_revision_runs` says,
    and the status read makes the same check of the revision it resolves.
    """

    with monkeypatch.context() as patch:
        # Saving a ComfyUI workflow maps its settings against the running worker.
        _reviewing_runtime(app, patch)
        created = await client.post(
            "/api/workflows",
            json={
                "name": "Neutral unreviewed workflow",
                "operation": "text_to_image",
                "engine": "comfyui",
                "api_graph": {
                    "1": {
                        "class_type": "EmptyLatentImage",
                        "inputs": {"width": 512, "height": 512, "batch_size": 1},
                    }
                },
            },
        )
    assert created.status_code == 201, created.text
    revision_id = created.json()["current_revision_id"]
    with SessionLocal() as session:
        profile = session.scalar(
            select(ModelProfile).where(
                ModelProfile.role == "image", ModelProfile.is_default.is_(True)
            )
        )
        assert profile is not None and profile.model_install_id is None
        # The default picture profile of an instance set up for ComfyUI.
        profile.engine = "comfyui"
        session.commit()
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()

    image = _operations(await _status(client, chat["id"]), "image")

    assert image["text_to_image"] == {
        "operation": "text_to_image",
        "readiness": "review_required",
        "trust": "untrusted",
        "reason": "revision_untrusted",
        "workflow_revision_id": revision_id,
    }
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        assert not app.state.services.orchestrator._studio_revision_runs(session, revision)


async def test_a_variant_whose_review_went_stale_reads_review_required_as_selection_refuses_it(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        _reviewing_runtime(app, patch)
        workflow_id, revision_id, _snapshot = await _approved(client)
    reviewed = (await client.get(f"/api/workflows/{workflow_id}")).json()
    assert next(item for item in reviewed["revisions"] if item["id"] == revision_id)["trusted"]
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        # A change the approved review did not see; the stored flag stays set.
        revision.dependencies_json = {"minimum_vram_bytes": 98765}
        session.commit()
        assert revision.trusted is True
    monkeypatch.setattr(settings, "media_engine", "comfyui")

    family = await _family_of(client, workflow_id)
    variant = next(variant for variant in family["variants"] if variant["id"] == workflow_id)
    detail = (await client.get(f"/api/workflows/{workflow_id}")).json()
    listed = next(
        workflow
        for workflow in (await client.get("/api/workflows")).json()
        if workflow["id"] == workflow_id
    )
    assert (variant["readiness"], variant["readiness_reason"], variant["trusted"]) == (
        "review_required",
        "revision_untrusted",
        False,
    )
    # The Workflows page badges a revision from the workflow it reads.
    for workflow in (detail, listed):
        shown = next(item for item in workflow["revisions"] if item["id"] == revision_id)
        assert shown["trusted"] is False

    chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()
    await _select_family(client, chat["id"], family["id"])
    image = _operations(await _status(client, chat["id"]), "image")
    refused = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Draw a blue cup", "mode": "image"}
    )
    assert (image["text_to_image"]["readiness"], image["text_to_image"]["reason"]) == (
        "review_required",
        "revision_untrusted",
    )
    assert refused.status_code == 422, refused.text


async def test_each_lane_reads_as_its_own_route_reads_it(client: AsyncClient) -> None:
    paused = await client.post(
        "/api/queue/lanes/transfer/pause-after-current",
        json={"expected_revision": 0, "idempotency_key": "neutral-pause"},
    )
    assert paused.status_code == 200, paused.text
    with SessionLocal() as session:
        session.merge(GenerationQueuePolicy(lane="install", dispatch_state="stuck", revision=1))
        # A finished utility job still holding its claim counts as running until
        # the claim is released.
        session.add(
            Job(kind="media_utility", status="complete", phase="complete", claim_owner="neutral")
        )
        session.commit()

    lanes = {lane["lane"]: lane for lane in (await _status(client))["lanes"]}

    assert list(lanes) == ["generation", "transfer", "install", "utility"]
    for name, lane in lanes.items():
        own = await client.get(f"/api/queue/lanes/{name}")
        if own.status_code == 409:
            assert own.json()["code"] == "queue-lane-conflict"
            assert lane == {"lane": name, "dispatch_state": "unreadable", "running_jobs": None}
        else:
            assert own.status_code == 200, own.text
            assert (lane["dispatch_state"], lane["running_jobs"]) == (
                own.json()["dispatch_state"],
                own.json()["running_jobs"],
            ), name
    assert [(lane["dispatch_state"], lane["running_jobs"]) for lane in lanes.values()] == [
        ("open", 0),
        ("paused", 0),
        ("unreadable", None),
        ("open", 1),
    ]


async def test_storage_reads_free_space_against_the_copy_floor_and_the_storage_warning(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import instance_status
    from local_lm.artifacts import COPY_HEADROOM_BYTES

    monkeypatch.setattr(settings, "storage_warning_free_bytes", 2**62)
    low = (await _status(client))["storage"]
    warned = (await client.get("/api/artifacts/storage")).json()["warning"]
    assert low["free_bytes"] >= COPY_HEADROOM_BYTES, "this needs room above the floor to run"
    assert (low["state"], warned) == ("low", True)
    assert (low["floor_bytes"], low["warning_bytes"]) == (COPY_HEADROOM_BYTES, 2**62)

    monkeypatch.setattr(settings, "storage_warning_free_bytes", 0)
    roomy = (await _status(client))["storage"]
    warned = (await client.get("/api/artifacts/storage")).json()["warning"]
    assert (roomy["state"], warned) == ("ok", False)

    monkeypatch.setattr(instance_status, "available_bytes", lambda _root: COPY_HEADROOM_BYTES - 1)
    below = (await _status(client))["storage"]
    assert (below["state"], below["free_bytes"]) == ("below_floor", COPY_HEADROOM_BYTES - 1)

    # Exactly the floor is room enough for a copy, and exactly the warning is no warning.
    monkeypatch.setattr(instance_status, "available_bytes", lambda _root: COPY_HEADROOM_BYTES)
    monkeypatch.setattr(settings, "storage_warning_free_bytes", COPY_HEADROOM_BYTES)
    level = (await _status(client))["storage"]
    assert (level["state"], level["free_bytes"]) == ("ok", COPY_HEADROOM_BYTES)

    def unreadable(_root: object) -> int:
        raise AnchoredDirectoryError("refused")

    monkeypatch.setattr(instance_status, "available_bytes", unreadable)
    unmeasured = (await _status(client))["storage"]
    assert (unmeasured["state"], unmeasured["free_bytes"]) == ("unmeasured", None)


async def test_the_status_read_asks_no_engine_anything(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    comfyui = Settings(
        data_dir=settings.data_dir, dev=True, chat_engine="mock", media_engine="comfyui"
    )
    app = create_app(comfyui)
    sent = await _refusing_media_engine(app)
    services = app.state.services
    probes: list[str] = []

    def refused(name: str) -> Callable[..., Awaitable[Any]]:
        async def probe(*_args: object, **_options: object) -> Any:
            probes.append(name)
            raise AssertionError(f"the status read asked for {name}")

        return probe

    async with _serving(app) as (client, _session):
        chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()
        before = len(sent)
        with monkeypatch.context() as patch:
            for name in (
                "capabilities",
                "object_info",
                "queue_remaining",
                "foreign_prompt_count",
                "_queued_client_ids",
            ):
                patch.setattr(services.engines.media, name, refused(f"media.{name}"))
            for name in ("capabilities", "chat_capabilities", "media_capabilities"):
                patch.setattr(services.engines, name, refused(f"engines.{name}"))
            patch.setattr(services.engines, "settings_for_role", refused("settings_for_role"))
            plain = await client.get(STATUS)
            with_chat = await client.get(STATUS, params={"chat_id": chat["id"]})
        made = sent[before:]

    assert (plain.status_code, with_chat.status_code) == (200, 200), with_chat.text
    assert with_chat.json()["chat"]["chat_id"] == chat["id"]
    assert made == []
    assert probes == []


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


async def test_the_status_read_carries_no_title_draft_family_name_or_path(
    client: AsyncClient, settings: Settings
) -> None:
    chat = (await client.post("/api/chats", json={"title": TITLE})).json()
    drafted = await client.put(
        f"/api/chats/{chat['id']}/composer-draft",
        json={"expected_revision": 0, "draft": {"text": DRAFT}},
    )
    assert drafted.status_code == 200, drafted.text
    created = await client.post(
        "/api/workflows",
        json={
            "name": FAMILY,
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "Mock"}},
        },
    )
    assert created.status_code == 201, created.text
    family = await _family_of(client, created.json()["id"])
    assert family["name"] == FAMILY
    await _select_family(client, chat["id"], family["id"])
    assert (await client.get(f"/api/chats/{chat['id']}")).json()["title"] == TITLE

    response = await client.get(STATUS, params={"chat_id": chat["id"]})

    assert response.status_code == 200, response.text
    for sentinel in (TITLE, DRAFT, FAMILY):
        assert sentinel not in response.text
    data = settings.data_dir
    for spelling in {str(data), data.as_posix(), json.dumps(str(data))[1:-1]}:
        assert spelling not in response.text
    body = response.json()
    assert [text for text in _strings(body) if "/" in text or "\\" in text] == []
    assert set(body) == STATUS_KEYS
    assert set(body["build"]) == {"version", "kind", "commit", "dirty"}
    assert set(body["startup"]) == {"state", "tasks"}
    assert {frozenset(task) for task in body["startup"]["tasks"]} == {frozenset({"task", "state"})}
    assert set(body["workspace_lock"]) == {
        "locked",
        "enabled",
        "require_pin",
        "lock_epoch",
        "idle_lock_seconds",
    }
    assert {frozenset(worker) for worker in body["workers"]} == {
        frozenset({"name", "condition", "restore", "last_stop_reason", "last_end", "failure_code"})
    }
    assert {frozenset(lane) for lane in body["lanes"]} == {
        frozenset({"lane", "dispatch_state", "running_jobs"})
    }
    assert set(body["storage"]) == {"state", "free_bytes", "floor_bytes", "warning_bytes"}
    assert set(body["chat"]) == {"chat_id", "workflows"}
    assert [workflow["selector_capability"] for workflow in body["chat"]["workflows"]] == [
        "chat",
        "image",
        "video",
    ]
    for workflow in body["chat"]["workflows"]:
        assert set(workflow) == {
            "selector_capability",
            "mode",
            "workflow_family_id",
            "operations",
        }
        for operation in workflow["operations"]:
            assert set(operation) == {
                "operation",
                "readiness",
                "trust",
                "reason",
                "workflow_revision_id",
            }
    plain = await _status(client)
    assert plain["chat"] is None


async def test_the_status_read_needs_a_session_and_is_in_neither_public_list(
    settings: Settings,
) -> None:
    assert STATUS not in PUBLIC_API_PATHS
    assert ("GET", STATUS) not in REACHABLE_WHILE_LOCKED
    outside_development = Settings(
        data_dir=settings.data_dir, dev=False, chat_engine="mock", media_engine="mock"
    )
    app = create_app(outside_development)

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as stranger,
    ):
        refused = await stranger.get(STATUS)
        assert (await stranger.post("/api/session")).status_code == 200
        admitted = await stranger.get(STATUS)

    assert refused.status_code == 401
    assert refused.json()["code"] == "session-required"
    assert admitted.status_code == 200, admitted.text


async def test_a_chat_the_status_read_cannot_show_is_not_found(client: AsyncClient) -> None:
    hidden = (await client.post("/api/chats", json={"title": "Neutral"})).json()
    with SessionLocal() as session:
        chat = session.get(Chat, hidden["id"])
        assert chat is not None
        chat.scope = EXPERIMENT_CHAT_SCOPE
        session.commit()

    for chat_id in ("chat_missing", hidden["id"]):
        refused = await client.get(STATUS, params={"chat_id": chat_id})
        selections = await client.get(f"/api/chats/{chat_id}/workflow-selections")
        assert (refused.status_code, refused.json()["code"]) == (404, "chat-not-found"), chat_id
        assert selections.status_code == 404, chat_id
    overlong = await client.get(STATUS, params={"chat_id": "c" * 41})
    assert overlong.status_code == 422


def _link_folder(link: Path, target: Path) -> bool:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        return False
    return True


def _bundle(root: Path, manifest: object) -> Path:
    root.mkdir(parents=True)
    content = manifest if isinstance(manifest, bytes) else json.dumps(manifest).encode()
    (root / "release-manifest.json").write_bytes(content)
    return root


def _release(commit: object = COMMIT, dirty: object = False, **more: object) -> dict[str, Any]:
    return {"version": __version__, "source": {"commit": commit, "dirty": dirty}, **more}


def _bundle_for(case: str, tmp_path: Path) -> Path | None:
    """The folder a frozen release would run from for each case, or None from source."""

    if case == "source":
        return None
    if case == "release":
        return _bundle(tmp_path / "bundle", _release(dirty=True))
    if case == "short commit":
        return _bundle(tmp_path / "bundle", _release(commit=COMMIT[:-1]))
    if case == "uppercase commit":
        return _bundle(tmp_path / "bundle", _release(commit=COMMIT.upper()))
    if case == "dirty is not a flag":
        return _bundle(tmp_path / "bundle", _release(dirty="yes"))
    if case == "not json":
        return _bundle(tmp_path / "bundle", b"\xff\xfe not a manifest")
    if case == "oversized":
        from local_lm.instance_status import MAX_RELEASE_MANIFEST_BYTES

        # Still a valid document as far as the bound, so the size alone refuses it.
        manifest = json.dumps(_release()).encode() + b" " * MAX_RELEASE_MANIFEST_BYTES
        return _bundle(tmp_path / "bundle", manifest)
    if case == "no manifest":
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        return bundle
    if case == "linked manifest":
        real = _bundle(tmp_path / "real", _release())
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        try:
            (bundle / "release-manifest.json").symlink_to(real / "release-manifest.json")
        except OSError:
            pytest.skip("file symlinks are unavailable")
        return bundle
    assert case == "linked bundle"
    real = _bundle(tmp_path / "real", _release())
    if not _link_folder(tmp_path / "bundle", real):
        pytest.skip("folder links are unavailable")
    return tmp_path / "bundle"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("source", ("source", None, None)),
        ("release", ("release", COMMIT, True)),
        ("dirty is not a flag", ("release", COMMIT, None)),
        ("short commit", ("release", None, None)),
        ("uppercase commit", ("release", None, None)),
        ("not json", ("release", None, None)),
        ("oversized", ("release", None, None)),
        ("no manifest", ("release", None, None)),
        ("linked manifest", ("release", None, None)),
        ("linked bundle", ("release", None, None)),
    ],
)
async def test_the_build_names_a_commit_only_from_a_readable_bundled_manifest(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected: tuple[str, str | None, bool | None],
) -> None:
    bundle = _bundle_for(case, tmp_path)
    if bundle is None:
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    else:
        monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    app = create_app(settings)

    async with _serving(app) as (client, _session):
        build = (await _status(client))["build"]
        ready = (await client.get("/api/ready")).json()

    assert (build["kind"], build["commit"], build["dirty"]) == expected
    assert build["version"] == ready["version"] == __version__


def test_the_readiness_reasons_are_the_refusals_a_plain_turn_can_meet() -> None:
    """Every reason selection, a saved choice, a project pin or a recipe refuses with, and no other.

    Read from the source that raises them, so a refusal added there without a
    member here fails this rather than being reported as another reason.
    """

    from local_lm.schemas import RecipeReadinessReason, WorkflowReadinessReason
    from local_lm.workflow_use_case_errors import _MESSAGES
    from local_lm.workflow_use_case_preset_admission import AdmissionRefusal
    from local_lm.workflow_use_case_preset_resolution import ResolutionRefusal
    from local_lm.workflow_use_cases_v1 import UseCaseRefusal

    package = Path(local_lm.__file__).resolve().parent
    # The codes a turn names a recipe refusal with: every one it has words for,
    # and the one it answers when no candidate satisfied the recipe.
    named = set(_MESSAGES)
    for node in ast.walk(ast.parse((package / "workflow_use_case_errors.py").read_text("utf-8"))):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            named |= _constants(node.value.elts[0])
    refused = {
        member
        for refusal in (UseCaseRefusal, ResolutionRefusal, AdmissionRefusal)
        for member in _literal_members(refusal)
    }
    assert "workflow-use-case-no-compatible-workflow" in named and refused <= named
    recipe = set(get_args(RecipeReadinessReason))
    assert recipe == named
    selection: set[str] = set()
    for node in ast.walk(ast.parse((package / "workflow_selection.py").read_text("utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_error" and len(node.args) >= 3:
                selection |= _constants(node.args[2])
            if node.func.id == "WorkflowFamilySelectionError":
                selection |= {
                    text
                    for item in node.keywords
                    if item.arg == "reason"
                    for text in _constants(item.value)
                }
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "failure_reasons"
        ):
            selection |= {text for argument in node.args for text in _constants(argument)}
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "reason" for target in node.targets
        ):
            selection |= _constants(node.value)
    saved_choices: set[str] = set()
    for module in ("workflow_compatibility.py", "orchestrator.py"):
        for node in ast.walk(ast.parse((package / module).read_text("utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "WorkflowSelectionInvalid"
            ):
                saved_choices |= {
                    text
                    for item in node.keywords
                    if item.arg == "reason"
                    for text in _constants(item.value)
                }
    pins = {f"project_pin_{problem}" for problem in ConversationOrchestrator._PIN_PROBLEM_MESSAGES}

    assert "revision_untrusted" in selection and "family_unavailable" in saved_choices
    assert _literal_members(WorkflowReadinessReason) == (
        selection | saved_choices | pins | {"node_packages_missing"} | recipe
    )


def _literal_members(vocabulary: object) -> set[str]:
    """Every string a Literal admits, through any unions of Literals it is made of."""

    members: set[str] = set()
    for part in get_args(vocabulary):
        members |= _literal_members(part) if get_args(part) else {part}
    return members


def _constants(node: ast.AST) -> set[str]:
    """Each string a reason expression can take: a constant, or either arm of a choice."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _constants(node.body) | _constants(node.orelse)
    return set()


async def test_a_pin_a_turn_carries_forward_is_read_without_writing_or_waiting_on_a_writer(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id, project_id, pinned, successor = await _pinned_chat(client)

    with _statements_the_read_writes(monkeypatch) as written:
        read = await client.get(STATUS, params={"chat_id": chat_id})
    holder = sqlite3.connect(settings.state_dir / "local-lm.sqlite3", isolation_level=None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        held = await client.get(STATUS, params={"chat_id": chat_id})
        waited = time.monotonic() - started
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    with SessionLocal() as session:
        project = session.get(Project, project_id)
        assert project is not None
        still_pinned = project.image_workflow_revision_id
    turn = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": "Draw a blue cup", "mode": "image"}
    )

    assert read.status_code == 200, read.text
    assert written == []
    # A write would have waited out the writer's busy timeout and then failed.
    assert held.status_code == 200, held.text
    assert waited < SQLITE_BUSY_TIMEOUT_MS / 1000
    assert held.json()["chat"] == read.json()["chat"]
    assert still_pinned == pinned
    assert _operations(read.json(), "image")["text_to_image"] == {
        "operation": "text_to_image",
        "readiness": "ready",
        "trust": "trusted",
        "reason": None,
        "workflow_revision_id": successor,
    }
    assert turn.status_code == 202, turn.text
    assert turn.json()["run"]["workflow_revision_id"] == successor


@pytest.mark.parametrize(
    ("use_case", "mode", "capability", "operation", "code"),
    [
        (
            "image_generation",
            "image",
            "image",
            "text_to_image",
            "workflow-use-case-preset-setting-unsupported",
        ),
        ("chat", "text", "chat", "text", "workflow-use-case-preset-revision-required"),
    ],
)
async def test_a_saved_recipe_a_plain_turn_refuses_reads_as_that_refusal(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    use_case: str,
    mode: str,
    capability: str,
    operation: str,
    code: str,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()
    before = _operations(await _status(client, chat["id"]), capability)[operation]
    recipe = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Neutral recipe",
            "use_case": use_case,
            "settings_json": {"unsupported_setting": 1},
            "is_default": True,
        },
    )
    assert recipe.status_code == 201, recipe.text

    with _statements_the_read_writes(monkeypatch) as written:
        after = _operations(await _status(client, chat["id"]), capability)[operation]
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "A blue square", "mode": mode}
    )

    assert before["readiness"] == "ready"
    assert (after["readiness"], after["reason"]) == ("setup_required", code)
    assert (turn.status_code, turn.json()["code"]) == (422, code)
    assert written == []
    # The read checks a recipe's settings against these instead of asking the
    # engine, so they must be what the engine answers.
    for role in ("chat", "image", "video"):
        advertised = await app.state.services.engines.settings_for_role(role)
        assert advertised == builtin_settings_for_role(role), role


async def test_a_setup_verification_chat_reads_without_the_recipe_its_turn_leaves_out(
    client: AsyncClient, settings: Settings
) -> None:
    """Setup verification runs the role's own workflow, so no saved recipe applies to it."""

    recipe = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Neutral recipe",
            "use_case": "image_generation",
            "settings_json": {"unsupported_setting": 1},
            "is_default": True,
        },
    )
    assert recipe.status_code == 201, recipe.text
    install, profile, workflow = seed_ready_role(settings, "image", operation="text_to_image")
    assert workflow is not None
    with SessionLocal() as session:
        chat = Chat(
            title="Setup verification",
            archived=True,
            scope=SETUP_VERIFICATION_SCOPE,
            routing_mode="image",
        )
        session.add(chat)
        session.flush()
        session.add(
            SetupVerification(
                role="image",
                evidence_key="r" * 64,
                state="running",
                model_install_id=install.id,
                profile_id=profile.id,
                workflow_revision_id=workflow.id,
                chat_id=chat.id,
            )
        )
        session.commit()
        verifying = chat.id
    plain = (await client.post("/api/chats", json={"title": "Neutral"})).json()["id"]

    plain_read = _operations(await _status(client, plain), "image")["text_to_image"]
    verifying_read = _operations(await _status(client, verifying), "image")["text_to_image"]

    refusal = "workflow-use-case-preset-setting-unsupported"
    assert (plain_read["readiness"], plain_read["reason"]) == ("setup_required", refusal)
    assert verifying_read["reason"] != refusal, verifying_read


async def test_text_reads_setup_required_until_the_profile_a_chat_turn_loads_has_a_model(
    settings: Settings,
) -> None:
    managed = Settings(
        data_dir=settings.data_dir, dev=True, chat_engine="llama.cpp", media_engine="mock"
    )
    app = create_app(managed)

    async with _serving(app) as (client, _session):
        chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()
        missing = _operations(await _status(client, chat["id"]), "chat")["text"]
        turn = await client.post(
            f"/api/chats/{chat['id']}/turns", json={"text": "Say hello", "mode": "text"}
        )
        assert turn.status_code == 202, turn.text
        run = await _ended_run(client, turn.json()["run"]["id"], "failed")
        model = managed.model_dir / "neutral-chat" / "model.gguf"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"neutral model bytes")
        with SessionLocal() as session:
            profile = session.get(ModelProfile, turn.json()["run"]["profile_id"])
            assert profile is not None and profile.model_install_id is None
            install = ModelInstall(
                name="Neutral chat model",
                role="chat",
                engine="llama.cpp",
                local_path=str(model),
                active=True,
            )
            session.add(install)
            session.flush()
            profile.engine = "llama.cpp"
            profile.model_install_id = install.id
            session.commit()
        installed = _operations(await _status(client, chat["id"]), "chat")["text"]

    assert missing == {
        "operation": "text",
        "readiness": "setup_required",
        "trust": "not_applicable",
        "reason": "model_unavailable",
        "workflow_revision_id": None,
    }
    assert "installed model" in str(run["error"])
    assert (installed["readiness"], installed["reason"]) == ("ready", None)


async def test_a_retention_sweep_that_fails_reads_as_given_up_and_stalls_startup(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fails(*_args: object, **_options: object) -> object:
        raise RuntimeError("The neutral sweep failed.")

    monkeypatch.setattr(app.state.services.artifacts, "cleanup_retention", fails)

    async with _serving(app) as (client, _session):
        await _restore_ended(app)
        body = await _status(client)

    assert app.state.retention_sweep.result() is False
    assert body["startup"]["state"] == "stalled"
    assert _tasks(body) == {
        "worker_restore": "done",
        "runtime_verification": "done",
        "reference_index": "done",
        "retention_sweep": "gave_up",
    }


@pytest.mark.parametrize("held_back_by", ["the-writer", "the-reference-index"])
async def test_a_retention_sweep_held_back_past_its_tries_reads_as_given_up(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, held_back_by: str
) -> None:
    """Another writer holds the database, or the reference index is behind, at every try."""

    def held_back(*_args: object, **_options: object) -> object:
        if held_back_by == "the-writer":
            busy = sqlite3.OperationalError("database is locked")
            raise OperationalError("DELETE FROM artifacts", {}, busy)
        raise ArtifactReferenceIndexStale()

    monkeypatch.setattr(app.state.services.artifacts, "cleanup_retention", held_back)
    monkeypatch.setattr(application, "RETENTION_LOCK_RETRIES", 0)
    monkeypatch.setattr(application, "RETENTION_INDEX_RETRIES", 0)

    async with _serving(app) as (client, _session):
        await _restore_ended(app)
        body = await _status(client)

    assert app.state.retention_sweep.result() is False
    assert body["startup"]["state"] == "stalled"
    assert _tasks(body)["retention_sweep"] == "gave_up"


async def test_startup_work_that_raised_reads_as_given_up_and_its_workers_as_failed(
    app: FastAPI,
) -> None:
    async def raises() -> Any:
        raise RuntimeError("The neutral startup work failed.")

    async with _serving(app) as (client, _session):
        await _restore_ended(app)
        raised = asyncio.ensure_future(raises())
        with suppress(RuntimeError):
            await raised
        # Raised where a restore or a runtime check would, and left for the read.
        app.state.startup = dataclasses.replace(
            app.state.startup, worker_restore=raised, runtime_verification=raised
        )
        body = await _status(client)

    assert body["startup"]["state"] == "stalled"
    assert (_tasks(body)["worker_restore"], _tasks(body)["runtime_verification"]) == (
        "gave_up",
        "gave_up",
    )
    for name in ("chat", "media"):
        assert (_worker(body, name)["restore"], _worker(body, name)["condition"]) == (
            "failed",
            "failing",
        ), name


async def test_each_worker_reads_its_own_restore_outcome(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    restore = _StartupRestore(app, monkeypatch)
    restore.start_media.side_effect = RuntimeError("The neutral worker could not start.")

    async with _serving(app) as (client, _session):
        monkeypatch.setattr(settings, "media_engine", "comfyui")
        monkeypatch.setattr(worker_startup, "_media_worker_should_restore", lambda _services: True)
        restore.go.set()
        await _restore_ended(app)
        body = await _status(client)

    assert (_worker(body, "media")["restore"], _worker(body, "media")["condition"]) == (
        "failed",
        "failing",
    )
    assert (_worker(body, "chat")["restore"], _worker(body, "chat")["condition"]) == (
        "not_configured",
        "not_started",
    )


async def test_a_failure_the_worker_status_names_reads_failing_whatever_state_it_shows(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor: ProcessSupervisor = app.state.services.processes
    with monkeypatch.context() as patch:
        # A process this application started earlier, still holding the port.
        patch.setattr(
            supervisor,
            "_orphaned_worker_description",
            lambda name: "process 4242" if name == "media" else None,
        )
        orphan = _worker(await _status(client), "media")
        orphan_state = _worker({"workers": (await client.get("/api/workers")).json()}, "media")
    monkeypatch.setattr(settings, "worker_shutdown_seconds", 1)
    spawned = _fake_workers(supervisor, monkeypatch, _StubbornProcess)
    await supervisor._replace("media", LAUNCH, HEALTH, reason=WorkerStartReason.RUN)
    try:
        with pytest.raises(WorkerStopIncomplete):
            await supervisor.stop("media", reason=WorkerStopReason.MANUAL)
        unfinished = _worker(await _status(client), "media")
        unfinished_state = _worker({"workers": (await client.get("/api/workers")).json()}, "media")
    finally:
        spawned[-1].end(0)

    assert orphan_state["state"] == "stopped"
    assert (orphan["condition"], orphan["failure_code"]) == ("failing", "port_in_use")
    assert unfinished_state["state"] == "stopping"
    assert unfinished["condition"] == "failing"


async def test_a_pinned_revision_whose_dependencies_are_not_activated_reads_setup_required(
    client: AsyncClient,
) -> None:
    workflow = await _created_workflow(
        client, "Neutral contract workflow", dependencies={"version": 1, "slots": []}
    )
    revision_id = workflow["current_revision_id"]
    seed_workflow_trust(revision_id)
    chat_id, _project_id = await _pinning_chat(client, revision_id)

    image = _operations(await _status(client, chat_id), "image")["text_to_image"]
    turn = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": "Draw a blue cup", "mode": "image"}
    )

    assert image == {
        "operation": "text_to_image",
        "readiness": "setup_required",
        "trust": "trusted",
        "reason": "activation_not_ready",
        "workflow_revision_id": revision_id,
    }
    assert turn.status_code == 422, turn.text
    assert "dependencies are not ready" in turn.json()["detail"]


async def test_a_pin_to_an_unreviewed_revision_reads_review_required_as_the_turn_refuses_it(
    client: AsyncClient,
) -> None:
    workflow = await _created_workflow(client, "Neutral unreviewed pinned workflow")
    revision_id = workflow["current_revision_id"]
    chat_id, _project_id = await _pinning_chat(client, revision_id)

    image = _operations(await _status(client, chat_id), "image")["text_to_image"]
    turn = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": "Draw a blue cup", "mode": "image"}
    )

    assert image == {
        "operation": "text_to_image",
        "readiness": "review_required",
        "trust": "untrusted",
        "reason": "project_pin_untrusted",
        "workflow_revision_id": revision_id,
    }
    assert turn.status_code == 409, turn.text
    assert (turn.json()["detail"]["code"], turn.json()["detail"]["reason"]) == (
        "project_workflow_pin_invalid",
        "untrusted",
    )


async def test_an_unreviewed_revision_an_engine_runs_anyway_reads_ready_and_untrusted(
    client: AsyncClient,
) -> None:
    """Only a ComfyUI run refuses an unreviewed revision, so the mock engine runs one."""

    workflow = await _created_workflow(client, "Neutral unreviewed mock workflow")
    revision_id = workflow["current_revision_id"]
    chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()

    image = _operations(await _status(client, chat["id"]), "image")["text_to_image"]
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Draw a blue cup", "mode": "image"}
    )

    assert image == {
        "operation": "text_to_image",
        "readiness": "ready",
        "trust": "untrusted",
        "reason": None,
        "workflow_revision_id": revision_id,
    }
    assert turn.status_code == 202, turn.text
    assert turn.json()["run"]["workflow_revision_id"] == revision_id
    await _ended_run(client, turn.json()["run"]["id"], "complete")


async def test_a_plain_edit_reads_the_workflow_a_whole_picture_edit_chooses(
    app: FastAPI, client: AsyncClient
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()
    await _automatic(client, chat["id"], "image")
    masked_schema = {
        "type": "object",
        "properties": {"mask": {"type": "object", "x-lm-atelier-kind": "mask"}},
    }
    with SessionLocal() as session:
        _only_new_families_offered(session, "image")
        # Offered first, but it waits for a selection a plain edit does not bring.
        masked = _edit_family(session, "Neutral masked", masked_schema, is_default=True)
        whole = _edit_family(
            session, "Neutral whole", {"type": "object", "properties": {}}, is_default=False
        )
        session.commit()
        masked_id, whole_id = masked.id, whole.id

    edit = _operations(await _status(client, chat["id"]), "image")["image_to_image"]
    with SessionLocal() as session:
        stored = session.get(Chat, chat["id"])
        assert stored is not None
        _profile, _selection, chosen = app.state.services.orchestrator._execution_for_turn(
            session,
            stored,
            Operation.IMAGE_TO_IMAGE,
            "",
            TurnRequest(text="Neutral", mode="image"),
        )

    assert chosen is not None and chosen.id == whole_id != masked_id
    assert (edit["readiness"], edit["workflow_revision_id"]) == ("ready", whole_id)


async def test_a_plain_animation_reads_the_workflow_a_turn_without_a_guide_video_chooses(
    app: FastAPI, client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    chat = (await client.post("/api/chats", json={"title": "Neutral"})).json()
    await _automatic(client, chat["id"], "video")
    with SessionLocal() as session:
        _only_new_families_offered(session, "video")
        # Offered first, but it waits for a guide video a plain animation does not bring.
        guided = _video_family(
            session, "Neutral guided", _video_graph("input_video"), is_default=True
        )
        plain = _video_family(session, "Neutral plain", _video_graph(), is_default=False)
        session.commit()
        guided_id, plain_id = guided.id, plain.id

    animation = _operations(await _status(client, chat["id"]), "video")["image_to_video"]
    with SessionLocal() as session:
        stored = session.get(Chat, chat["id"])
        assert stored is not None
        # A ComfyUI turn that brings one picture and no video says so.
        _profile, _selection, chosen = app.state.services.orchestrator._execution_for_turn(
            session,
            stored,
            Operation.IMAGE_TO_VIDEO,
            "",
            TurnRequest(text="Neutral", mode="video"),
            guide_videos=0,
        )

    assert chosen is not None and chosen.id == plain_id != guided_id
    assert (animation["readiness"], animation["workflow_revision_id"]) == ("ready", plain_id)
