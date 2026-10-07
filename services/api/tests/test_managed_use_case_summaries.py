"""Suggestions require a current, matching, managed local chat worker."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import cast

import pytest
from fastapi import FastAPI
from run_waits import PATIENCE_SECONDS
from sqlalchemy.orm import Session

from local_lm import managed_use_case_summaries as summaries
from local_lm.db import SessionLocal
from local_lm.main import Services
from local_lm.models import ModelInstall, ModelProfile
from local_lm.schemas import WorkerStatus
from local_lm.use_case_summaries import UseCaseSummaryError


@pytest.fixture
def loaded_worker(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Services, Session, WorkerStatus, ModelProfile, ModelInstall]]:
    services = cast(Services, app.state.services)
    services.settings.chat_engine = "llama.cpp"
    install = ModelInstall(
        name="Constructed chat model",
        role="chat",
        engine="llama.cpp",
        local_path="unused",
        active=True,
    )
    with SessionLocal() as session:
        session.add(install)
        session.flush()
        profile = ModelProfile(
            name="Constructed chat profile",
            model_install_id=install.id,
            role="chat",
            engine="llama.cpp",
        )
        session.add(profile)
        session.commit()
        worker = WorkerStatus(
            name="chat",
            managed=True,
            running=True,
            state="ready",
            pid=123,
            profile_id=profile.id,
            command=["runtime", "--host", "127.0.0.1", "--port", "12341"],
        )
        monkeypatch.setattr(services.processes, "statuses", lambda: [worker])
        yield services, session, worker, profile, install


@pytest.mark.parametrize("engine", ["llama.cpp", "vllm"])
async def test_ready_managed_model_holds_the_worker_lease_until_the_suggestion_finishes(
    loaded_worker: tuple[Services, Session, WorkerStatus, ModelProfile, ModelInstall],
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
) -> None:
    services, session, _worker, profile, install = loaded_worker
    services.settings.chat_engine = profile.engine = install.engine = engine
    session.commit()
    entered = asyncio.Event()
    finish = asyncio.Event()
    other_acquired = asyncio.Event()

    async def invoke(origin: str, description: str) -> str:
        assert origin == "http://127.0.0.1:12341"
        assert description == "Watercolor landscapes"
        entered.set()
        await finish.wait()
        return "Watercolor scenes."

    async def other() -> None:
        async with services.scheduler.lease("primary"):
            other_acquired.set()

    monkeypatch.setattr(summaries, "suggest_use_case_summary", invoke)
    task = asyncio.create_task(
        summaries.suggest_managed_use_case_summary(services, session, "Watercolor landscapes")
    )
    await asyncio.wait_for(entered.wait(), timeout=PATIENCE_SECONDS)
    competing = asyncio.create_task(other())
    await asyncio.sleep(0)
    assert not other_acquired.is_set()
    finish.set()
    assert await asyncio.wait_for(task, timeout=PATIENCE_SECONDS) == "Watercolor scenes."
    await asyncio.wait_for(competing, timeout=PATIENCE_SECONDS)
    assert other_acquired.is_set()


@pytest.mark.parametrize(
    "invalid",
    [
        "unmanaged",
        "stopped",
        "starting",
        "pid",
        "profile_missing",
        "profile_role",
        "external_engine",
        "engine_mismatch",
        "install_missing",
        "install_inactive",
        "install_role",
        "install_engine",
        "remote_origin",
        "other_host",
        "other_port",
        "missing_host",
        "duplicate_port",
    ],
)
async def test_unproven_workers_never_receive_a_description(
    loaded_worker: tuple[Services, Session, WorkerStatus, ModelProfile, ModelInstall],
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    services, session, worker, profile, install = loaded_worker
    if invalid == "unmanaged":
        worker.managed = False
    elif invalid == "stopped":
        worker.running = False
    elif invalid == "starting":
        worker.state = "starting"
    elif invalid == "pid":
        worker.pid = None
    elif invalid == "profile_missing":
        worker.profile_id = "missing"
    elif invalid == "profile_role":
        profile.role = "image"
    elif invalid == "external_engine":
        services.settings.chat_engine = profile.engine = install.engine = "external"
    elif invalid == "engine_mismatch":
        services.settings.chat_engine = "mock"
    elif invalid == "install_missing":
        profile.model_install_id = None
    elif invalid == "install_inactive":
        install.active = False
    elif invalid == "install_role":
        install.role = "image"
    elif invalid == "install_engine":
        install.engine = "vllm"
    elif invalid == "remote_origin":
        services.settings.llama_url = "https://models.example"
    elif invalid == "other_host":
        worker.command[2] = "localhost"
    elif invalid == "other_port":
        worker.command[4] = "12342"
    elif invalid == "missing_host":
        worker.command = ["runtime", "--port", "12341"]
    elif invalid == "duplicate_port":
        worker.command.extend(["--port", "12341"])
    session.commit()

    async def forbidden(origin: str, description: str) -> str:
        pytest.fail("An unproven worker received a description")

    monkeypatch.setattr(summaries, "suggest_use_case_summary", forbidden)
    with pytest.raises(UseCaseSummaryError, match="^Use-case suggestion is unavailable\\.$"):
        await summaries.suggest_managed_use_case_summary(services, session, "Watercolor scenes")


async def test_worker_and_install_are_rechecked_after_waiting_for_the_lease(
    loaded_worker: tuple[Services, Session, WorkerStatus, ModelProfile, ModelInstall],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services, session, _worker, _profile, install = loaded_worker

    async def forbidden(origin: str, description: str) -> str:
        pytest.fail("A stale installation received a description")

    monkeypatch.setattr(summaries, "suggest_use_case_summary", forbidden)
    async with services.scheduler.lease("primary"):
        task = asyncio.create_task(
            summaries.suggest_managed_use_case_summary(services, session, "Watercolor scenes")
        )
        await asyncio.sleep(0)
        with SessionLocal() as changed:
            current = changed.get(ModelInstall, install.id)
            assert current is not None
            current.active = False
            changed.commit()
        assert install.active
    with pytest.raises(UseCaseSummaryError):
        await asyncio.wait_for(task, timeout=5)


@pytest.mark.parametrize("ending", ["cancel", "timeout"])
async def test_abandoned_managed_suggestions_release_the_worker_lease(
    loaded_worker: tuple[Services, Session, WorkerStatus, ModelProfile, ModelInstall],
    monkeypatch: pytest.MonkeyPatch,
    ending: str,
) -> None:
    services, session, _worker, _profile, _install = loaded_worker
    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def invoke(origin: str, description: str) -> str:
        entered.set()
        await blocked.wait()
        raise AssertionError("Unexpected resume")

    monkeypatch.setattr(summaries, "suggest_use_case_summary", invoke)
    if ending == "timeout":
        monkeypatch.setattr(summaries, "MANAGED_SUMMARY_TIMEOUT_SECONDS", 0.05)
    task = asyncio.create_task(
        summaries.suggest_managed_use_case_summary(services, session, "Watercolor scenes")
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    if ending == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(UseCaseSummaryError):
            await asyncio.wait_for(task, timeout=PATIENCE_SECONDS)
    async with asyncio.timeout(5):
        async with services.scheduler.lease("primary"):
            pass
