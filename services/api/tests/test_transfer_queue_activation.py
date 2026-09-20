"""Download ownership spans transfer, verification and nested activation."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from test_downloads import gguf_bytes
from test_transfer_queue_downloads import control, read, wait_queued

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.model_manifests import inspect_repository_metadata
from local_lm.model_planner import persist_install_plan, resolve_install_plan
from local_lm.models import Job, ModelCapabilityEvidence, ModelInstall, ModelProfile
from local_lm.queue_lane_policy import change_lane_policy, read_lane_policy
from local_lm.schemas import DownloadRequest, QueueControlCommand


async def test_two_downloads_drain_through_complete_transfer_and_activation_lifetimes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.prepare()
    settings.max_concurrent_downloads = 2
    contents = {name: gguf_bytes(name + "_architecture") for name in ("first", "second", "new")}
    activation_entered = [asyncio.Event(), asyncio.Event()]
    activation_release = [asyncio.Event(), asyncio.Event()]
    second_transfer_entered = asyncio.Event()
    second_transfer_release = asyncio.Event()
    downloads: list[str] = []
    probes = 0

    class ChatAdapter:
        async def capabilities(self) -> SimpleNamespace:
            return SimpleNamespace(
                healthy=True, version="constructed-runtime", input_modalities=["text"]
            )

        async def count_tokens(self, _messages: list[dict[str, Any]]) -> int:
            return 3

        async def stream(self, _request: ChatRequest) -> AsyncIterator[ChatEvent]:
            nonlocal probes
            index = probes
            probes += 1
            activation_entered[index].set()
            await activation_release[index].wait()
            yield ChatEvent(type="token", text="OK")
            yield ChatEvent(type="complete")

    class Processes:
        def statuses(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="chat", running=False, profile_id=None)]

        async def load_chat(self, _profile: ModelProfile, _install: ModelInstall) -> None:
            return None

        async def stop(self, _name: str) -> None:
            return None

    adapter = Mock(spec_set=["capabilities", "count_tokens", "stream"], wraps=ChatAdapter())
    processes = Mock(spec_set=["runtimes", "statuses", "load_chat", "stop"], wraps=Processes())
    processes.runtimes = None
    manager = DownloadManager(
        settings, EventBroker(), chat_adapter=lambda: adapter, processes=processes
    )

    def model_info(remote_id: str, **_kwargs: object) -> SimpleNamespace:
        content = contents[remote_id.split("/")[-1]]
        return SimpleNamespace(
            siblings=[
                SimpleNamespace(
                    rfilename="model.gguf",
                    size=len(content),
                    lfs={"sha256": hashlib.sha256(content).hexdigest()},
                )
            ],
            sha="c" * 40,
            pipeline_tag="text-generation",
            tags=["gguf"],
            gated=False,
        )

    monkeypatch.setattr(manager, "_api", SimpleNamespace(model_info=model_info))

    async def download_file(**kwargs: Any) -> str:
        name = str(kwargs["remote_id"]).split("/")[-1]
        downloads.append(name)
        if name == "second":
            second_transfer_entered.set()
            await second_transfer_release.wait()
        target = kwargs["staging"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents[name])
        return str(target)

    monkeypatch.setattr(manager, "_download_file", download_file)

    def create(name: str) -> str:
        content = contents[name]
        digest = hashlib.sha256(content).hexdigest()
        resolved = resolve_install_plan(
            remote_id="example/" + name,
            revision="c" * 40,
            role="chat",
            engine="llama.cpp",
            selected_files=[{"filename": "model.gguf", "size": len(content), "sha256": digest}],
            inspection=inspect_repository_metadata(
                {"model.gguf": content}, ["model.gguf"], role="chat"
            ),
        )
        with SessionLocal() as session:
            plan = persist_install_plan(session, resolved)
            session.commit()
            return manager.create(
                session,
                DownloadRequest(
                    install_plan_id=plan.id,
                    remote_id=plan.remote_id,
                    revision=plan.revision,
                    role=plan.role,
                    engine=plan.engine,
                    allow_patterns=["model.gguf"],
                    expected_sha256={"model.gguf": digest},
                ),
            ).id

    async def wait_entered(event: asyncio.Event, job_id: str) -> None:
        async with asyncio.timeout(10):
            while not event.is_set():
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    assert job is not None and job.status in {"queued", "running"}, (
                        job.error if job else "Download disappeared"
                    )
                await asyncio.sleep(0.01)

    with SessionLocal() as session:
        change_lane_policy(
            session,
            "generation",
            "pause_after_current",
            QueueControlCommand(expected_revision=0, idempotency_key="generation"),
        )
    try:
        first = create("first")
        first_task = manager._tasks[first]
        await wait_entered(activation_entered[0], first)
        second = create("second")
        second_task = manager._tasks[second]
        await wait_entered(second_transfer_entered, second)
        policy = control("pause_after_current", 0)
        assert policy.dispatch_state == "draining" and policy.running_jobs == 2
        new = create("new")
        await wait_queued(new)
        activation_release[0].set()
        await asyncio.wait_for(first_task, timeout=10)
        assert read().dispatch_state == "draining" and read().running_jobs == 1
        second_transfer_release.set()
        await wait_entered(activation_entered[1], second)
        assert read().dispatch_state == "draining" and read().running_jobs == 1
        await wait_queued(new)
        activation_release[1].set()
        await asyncio.wait_for(second_task, timeout=10)
        assert read().dispatch_state == "paused" and read().running_jobs == 0
        assert downloads == ["first", "second"]
        with SessionLocal() as session:
            for job_id in (first, second):
                job = session.get(Job, job_id)
                assert job is not None and job.status == "complete", job.error if job else None
                assert job.claim_owner is None
                install = session.get(ModelInstall, job.result_json["model_install_id"])
                assert install is not None and install.active
            assert session.query(ModelCapabilityEvidence).count() == 2
            assert read_lane_policy(session, "generation").dispatch_state == "paused"
        await wait_queued(new)
    finally:
        for release in activation_release:
            release.set()
        second_transfer_release.set()
        await manager.close()
