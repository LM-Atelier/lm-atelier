"""An activation that loses its claim during measurement cannot replace the model identity."""

from __future__ import annotations

import asyncio
import copy
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS
from test_activation_claim_ownership import _move_claim, _state
from test_activation_probe_ownership import _Processes

from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobKind, JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job, ModelInstall
from local_lm.scheduler import ResourceScheduler


@pytest.mark.parametrize("disposition", ["cleared", "replaced", "removed", "owned"])
async def test_activation_identity_measurement_preserves_a_replacement_claim(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    disposition: str,
) -> None:
    destination = settings.model_dir / "activation-fixture"
    destination.mkdir(parents=True)
    body = b"neutral model bytes for an owning measurement"
    (destination / "fixture.gguf").write_bytes(body)
    initial_manifest = {
        "files": ["fixture.gguf"],
        "expected_sha256": {"fixture.gguf": "00" * 32},
        "file_signatures": {"fixture.gguf": [0, 0]},
    }

    class Chat:
        async def capabilities(self) -> SimpleNamespace:
            return SimpleNamespace(
                healthy=True, version="neutral-runtime", input_modalities=["text"]
            )

        async def count_tokens(self, _messages: list[dict[str, object]]) -> int:
            return 4

        async def stream(self, _request: ChatRequest) -> AsyncIterator[ChatEvent]:
            yield ChatEvent(type="token", text="OK")
            yield ChatEvent(type="complete")

    manager = DownloadManager(
        settings,
        EventBroker(),
        chat_adapter=lambda: Mock(spec_set=Chat, wraps=Chat()),
        processes=Mock(spec_set=_Processes, wraps=_Processes()),
        scheduler=ResourceScheduler(),
    )
    with SessionLocal() as session:
        session.add_all(
            [
                ModelInstall(
                    id="activation-model",
                    name="Neutral measured activation",
                    role="chat",
                    engine="llama.cpp",
                    local_path=str(destination),
                    manifest_json=copy.deepcopy(initial_manifest),
                    active=False,
                ),
                Job(
                    id="activation-claim",
                    kind=JobKind.ACTIVATE.value,
                    status=JobStatus.QUEUED.value,
                    payload_json={"install_id": "activation-model"},
                    queue_resource="primary_compute",
                    queue_group="primary",
                    enqueued_at=utcnow(),
                ),
            ]
        )
        session.commit()
    original_measure = manager.measured_install_identity
    measured: list[tuple[dict[str, str], dict[str, list[int]]]] = []
    before: list[tuple[object, ...]] = []

    def measure(path: Path) -> tuple[dict[str, str], dict[str, list[int]]]:
        result = original_measure(path)
        measured.append(result)
        if disposition != "owned":
            _move_claim(disposition)
        if disposition != "removed":
            before.append(_state())
        return result

    monkeypatch.setattr(manager, "measured_install_identity", measure)
    activation = asyncio.create_task(manager._reactivate("activation-claim"))
    try:
        await asyncio.wait_for(
            asyncio.gather(activation, return_exceptions=True), timeout=PATIENCE_SECONDS
        )
    finally:
        activation.cancel()
        await asyncio.gather(activation, return_exceptions=True)
    assert len(measured) == 1
    assert measured[0][0] == {"fixture.gguf": hashlib.sha256(body).hexdigest()}
    with SessionLocal() as session:
        install = session.get(ModelInstall, "activation-model")
        assert install is not None
        if disposition == "owned":
            assert install.active
            assert install.manifest_json["expected_sha256"] == measured[0][0]
            assert install.manifest_json["file_signatures"] == measured[0][1]
        else:
            assert install.manifest_json == initial_manifest
            assert not install.active
        if disposition == "removed":
            assert session.get(Job, "activation-claim") is None
    if disposition == "owned":
        assert _state()[0] == JobStatus.COMPLETE.value
    elif disposition != "removed":
        assert _state() == before[0]
