from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import local_lm.orchestrator as orchestrator_module
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


def test_visual_context_enforces_image_count_and_total_byte_bounds(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    paths = []
    candidates = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        path.write_bytes(b"123456")
        paths.append(path)
        candidates.append(
            SimpleNamespace(
                id=f"artifact-{index}",
                size_bytes=6,
                sha256=hashlib.sha256(b"123456").hexdigest(),
            )
        )
    store = SimpleNamespace(
        delivery_metadata=lambda artifact: (
            paths[int(artifact.id.rsplit("-", 1)[1])],
            "image/png",
            "inline",
        )
    )
    orchestrator = ConversationOrchestrator(
        engines=Mock(),
        artifacts=store,
        events=Mock(),
        scheduler=Mock(),
        processes=Mock(),
    )
    monkeypatch.setattr(orchestrator_module, "MAX_VISION_IMAGES", 2)
    monkeypatch.setattr(orchestrator_module, "MAX_VISION_IMAGE_BYTES", 8)
    monkeypatch.setattr(orchestrator_module, "MAX_VISION_TOTAL_BYTES", 10)
    monkeypatch.setattr(
        orchestrator,
        "_visual_context_artifacts",
        Mock(return_value=candidates),
    )

    messages, metadata = orchestrator._attach_visual_context(
        Mock(),
        SimpleNamespace(),
        [{"role": "user", "content": "Inspect these"}],
    )

    assert metadata == {
        "available": True,
        "images_included": 1,
        "artifact_ids": ["artifact-0"],
        "bytes_included": 6,
        "images_skipped": 2,
    }
    assert len(messages[0]["content"]) == 2


def test_visual_context_requires_a_user_target_before_reading_artifacts() -> None:
    candidate = SimpleNamespace(id="artifact-1")
    store = SimpleNamespace(delivery_metadata=Mock(side_effect=AssertionError))
    orchestrator = ConversationOrchestrator(
        engines=Mock(),
        artifacts=store,
        events=Mock(),
        scheduler=Mock(),
        processes=Mock(),
    )
    orchestrator._visual_context_artifacts = Mock(return_value=[candidate])  # type: ignore[method-assign]
    original = [{"role": "system", "content": "System only"}]

    messages, metadata = orchestrator._attach_visual_context(
        Mock(),
        SimpleNamespace(),
        original,
    )

    assert messages == original
    assert metadata == {
        "available": True,
        "images_included": 0,
        "artifact_ids": [],
        "bytes_included": 0,
        "images_skipped": 1,
    }
    store.delivery_metadata.assert_not_called()


def test_visual_context_rejects_non_raster_and_unverified_bytes(tmp_path: Path) -> None:
    svg_path = tmp_path / "private-diagram.svg"
    svg_path.write_text("<svg/>", encoding="utf-8")
    tampered_path = tmp_path / "tampered.png"
    tampered_path.write_bytes(b"tampered")
    candidates = [
        SimpleNamespace(
            id="svg-artifact",
            size_bytes=svg_path.stat().st_size,
            sha256=hashlib.sha256(svg_path.read_bytes()).hexdigest(),
        ),
        SimpleNamespace(
            id="tampered-artifact",
            size_bytes=tampered_path.stat().st_size,
            sha256="0" * 64,
        ),
    ]
    paths = {
        "svg-artifact": (svg_path, "image/svg+xml", "inline"),
        "tampered-artifact": (tampered_path, "image/png", "inline"),
    }
    store = SimpleNamespace(delivery_metadata=lambda artifact: paths[artifact.id])
    orchestrator = ConversationOrchestrator(
        engines=Mock(),
        artifacts=store,
        events=Mock(),
        scheduler=Mock(),
        processes=Mock(),
    )
    orchestrator._visual_context_artifacts = Mock(  # type: ignore[method-assign]
        return_value=candidates
    )

    messages, metadata = orchestrator._attach_visual_context(
        Mock(),
        SimpleNamespace(),
        [{"role": "user", "content": "Inspect safely"}],
    )

    assert messages == [{"role": "user", "content": "Inspect safely"}]
    assert metadata == {
        "available": True,
        "images_included": 0,
        "artifact_ids": [],
        "bytes_included": 0,
        "images_skipped": 2,
    }
