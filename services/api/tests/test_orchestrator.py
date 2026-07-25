from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from local_lm.adapters.base import MediaEvent
from local_lm.models import Message, MessagePart, ModelInstall, ModelProfile, Run
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.schemas import WorkerStatus


async def test_managed_chat_worker_is_aligned_to_the_run_profile(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    run = SimpleNamespace(profile_id="profile-selected")
    profile = SimpleNamespace(id="profile-selected", model_install_id="install-selected")
    install = SimpleNamespace(id="install-selected")

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            return None

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (Run, "run-1"): run,
                (ModelProfile, "profile-selected"): profile,
                (ModelInstall, "install-selected"): install,
            }.get((model, identity))

        def expunge(self, _value) -> None:  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr("local_lm.orchestrator.SessionLocal", FakeSession)
    previous = WorkerStatus(
        name="chat",
        state="ready",
        managed=True,
        running=True,
        pid=11,
        profile_id="profile-previous",
    )
    aligned = previous.model_copy(update={"pid": 12, "profile_id": "profile-selected"})
    processes = SimpleNamespace(
        settings=SimpleNamespace(llama_executable=Path("llama-server")),
        statuses=Mock(return_value=[previous]),
        load_chat=AsyncMock(return_value=aligned),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(chat_engine="llama.cpp")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )

    result = await orchestrator._ensure_chat_worker("run-1")

    assert result == aligned
    processes.load_chat.assert_awaited_once_with(profile, install)


async def test_chat_planner_falls_back_during_media_handoff() -> None:
    ready = WorkerStatus(
        name="chat",
        state="ready",
        managed=True,
        running=True,
        pid=12,
        profile_id="profile-selected",
    )
    processes = SimpleNamespace(
        settings=SimpleNamespace(
            llama_executable=Path("llama-server"),
            worker_startup_seconds=60,
        ),
        statuses=Mock(return_value=[ready]),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(chat_engine="llama.cpp")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    orchestrator._chat_planner_ready.clear()

    assert await orchestrator._chat_planner_available() is False
    orchestrator._chat_planner_ready.set()
    assert await orchestrator._chat_planner_available() is True


def test_media_progress_preserves_the_latest_preview() -> None:
    message = Message(chat_id="chat-1", role="assistant", status="pending")
    message.parts = [
        MessagePart(
            position=0,
            type="progress",
            text="Preview",
            metadata_json={"progress": 0.4, "phase": "preview"},
        ),
        MessagePart(
            position=1,
            type="image",
            artifact_id="sha256:preview",
            metadata_json={"preview": True},
        ),
    ]

    parts = ConversationOrchestrator._media_progress_parts(
        message,
        MediaEvent(type="progress", progress=0.5, phase="sampling"),
    )

    assert [(part.position, part.type) for part in parts] == [(0, "progress"), (1, "image")]
    assert parts[0].text == "Sampling"
    assert parts[0].metadata_json == {"progress": 0.5, "phase": "sampling"}
    assert parts[1].artifact_id == "sha256:preview"
    assert parts[1].metadata_json == {"preview": True}


def test_chat_progress_is_removed_without_discarding_text() -> None:
    text = MessagePart(position=0, type="text", text="Partial response")
    progress = MessagePart(
        position=1,
        type="progress",
        text="Waiting for first token",
        metadata_json={"activity": "chat", "phase": "waiting for first token"},
    )
    message = Message(
        chat_id="chat-1",
        role="assistant",
        status="pending",
        parts=[text, progress],
    )

    ConversationOrchestrator._remove_chat_progress(message)

    assert message.parts == [text]
