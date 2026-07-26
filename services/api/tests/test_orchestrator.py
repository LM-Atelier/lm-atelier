from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from local_lm.adapters.base import ChatEvent, MediaEvent
from local_lm.models import Job, Message, MessagePart, ModelInstall, ModelProfile, Run
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.schemas import WorkerStatus


async def test_managed_chat_worker_is_aligned_to_the_run_profile() -> None:
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
        session_factory=FakeSession,
    )

    result = await orchestrator._ensure_chat_worker("run-1")

    assert result == aligned
    processes.load_chat.assert_awaited_once_with(profile, install)


async def test_engine_cancel_runs_after_the_database_session_closes() -> None:
    job = SimpleNamespace(id="job-cancel", status="running", run_id="run-cancel")
    run = SimpleNamespace(id="run-cancel", operation="text")
    session_closed = False

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            nonlocal session_closed
            session_closed = True

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (Job, "job-cancel"): job,
                (Run, "run-cancel"): run,
            }.get((model, identity))

        def commit(self) -> None:
            return None

    async def cancel(run_id: str) -> None:
        assert run_id == "run-cancel"
        assert session_closed is True

    engines = SimpleNamespace(
        settings=SimpleNamespace(),
        chat=SimpleNamespace(cancel=cancel),
        media=SimpleNamespace(cancel=AsyncMock()),
    )
    orchestrator = ConversationOrchestrator(
        engines=engines,
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(publish_job=AsyncMock()),
        processes=Mock(),
        session_factory=FakeSession,
    )
    orchestrator._mark_cancelled = Mock()  # type: ignore[method-assign]

    assert await orchestrator.cancel("job-cancel") is True
    orchestrator._mark_cancelled.assert_called_once()  # type: ignore[attr-defined]


async def test_chat_worker_resume_runs_after_the_database_session_closes() -> None:
    profile = SimpleNamespace(id="profile-resume", model_install_id="install-resume")
    install = SimpleNamespace(id="install-resume")
    session_closed = False

    class FakeSession:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            nonlocal session_closed
            session_closed = True

        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (ModelProfile, "profile-resume"): profile,
                (ModelInstall, "install-resume"): install,
            }.get((model, identity))

        def expunge(self, _value) -> None:  # type: ignore[no-untyped-def]
            return None

    async def load_chat(selected_profile, selected_install):  # type: ignore[no-untyped-def]
        assert session_closed is True
        assert selected_profile is profile
        assert selected_install is install

    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace()),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=SimpleNamespace(load_chat=load_chat),
        session_factory=FakeSession,
    )

    await orchestrator._resume_chat_worker("profile-resume")


async def test_media_handoff_recycles_managed_comfy_before_chat_resume() -> None:
    order: list[str] = []
    media = WorkerStatus(
        name="media",
        state="ready",
        managed=True,
        running=True,
        pid=22,
    )

    async def stop(name: str) -> None:
        assert name == "media"
        order.append("stop media")

    async def start_media() -> None:
        order.append("start media")

    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=stop,
        start_media=start_media,
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )

    async def resume(profile_id: str) -> None:
        assert profile_id == "profile-chat"
        order.append("resume chat")

    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    assert order == ["stop media", "start media", "resume chat"]


async def test_external_media_handoff_only_resumes_chat() -> None:
    media = WorkerStatus(
        name="media",
        state="ready",
        managed=False,
        running=True,
        pid=22,
    )
    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=AsyncMock(),
        start_media=AsyncMock(),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    processes.stop.assert_not_awaited()
    processes.start_media.assert_not_awaited()
    resume.assert_awaited_once_with("profile-chat")


async def test_media_recycle_failure_still_restores_chat() -> None:
    media = WorkerStatus(
        name="media",
        state="ready",
        managed=True,
        running=True,
        pid=22,
    )
    processes = SimpleNamespace(
        statuses=Mock(return_value=[media]),
        stop=AsyncMock(side_effect=RuntimeError("recycle failed")),
        start_media=AsyncMock(),
    )
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=SimpleNamespace(media_engine="comfyui")),
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    resume = AsyncMock()
    orchestrator._resume_chat_worker = resume  # type: ignore[method-assign]

    await orchestrator._complete_media_handoff("profile-chat")

    processes.start_media.assert_not_awaited()
    resume.assert_awaited_once_with("profile-chat")


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


@pytest.mark.parametrize("cancelled", [False, True])
async def test_vision_bridge_restores_the_text_profile_after_completion_or_cancellation(
    cancelled: bool,
) -> None:
    run = SimpleNamespace(
        id="run-vision",
        chat_id="chat-vision",
        user_message_id="message-vision",
        profile_id="profile-text",
        vision_profile_id="profile-vision",
        standalone_prompt="What is visible?",
    )
    text_profile = SimpleNamespace(id="profile-text", model_install_id="install-text")
    vision_profile = SimpleNamespace(id="profile-vision", model_install_id="install-vision")
    text_install = SimpleNamespace(id="install-text")
    vision_install = SimpleNamespace(id="install-vision")

    class FakeSession:
        def get(self, model, identity):  # type: ignore[no-untyped-def]
            return {
                (ModelProfile, "profile-text"): text_profile,
                (ModelProfile, "profile-vision"): vision_profile,
                (ModelInstall, "install-text"): text_install,
                (ModelInstall, "install-vision"): vision_install,
            }.get((model, identity))

        def expunge(self, _value) -> None:  # type: ignore[no-untyped-def]
            return None

        def scalar(self, _statement):  # type: ignore[no-untyped-def]
            return "job-vision"

        def in_transaction(self) -> bool:
            return False

    async def stream(_request):  # type: ignore[no-untyped-def]
        if cancelled:
            raise asyncio.CancelledError
        yield ChatEvent(type="delta", text="A green apple.")
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    processes = SimpleNamespace(
        settings=SimpleNamespace(),
        runtimes=None,
        load_chat=AsyncMock(),
        stop=AsyncMock(),
    )
    engines = SimpleNamespace(
        settings=SimpleNamespace(vision_bridge_max_tokens=128),
        chat_capabilities=AsyncMock(
            return_value=SimpleNamespace(input_modalities=["text", "image"])
        ),
        chat=SimpleNamespace(stream=stream),
    )
    orchestrator = ConversationOrchestrator(
        engines=engines,
        artifacts=Mock(),
        events=Mock(),
        scheduler=Mock(),
        processes=processes,
    )
    orchestrator._set_chat_phase = AsyncMock()  # type: ignore[method-assign]
    orchestrator._attach_visual_context = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            [{"role": "user", "content": [{"type": "text", "text": "Question"}]}],
            {
                "available": True,
                "images_included": 1,
                "artifact_ids": ["sha256:image"],
                "visual_contents_inspected": True,
            },
        )
    )

    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await orchestrator._bridge_visual_context(
                FakeSession(),  # type: ignore[arg-type]
                run,  # type: ignore[arg-type]
                [SimpleNamespace(id="sha256:image")],  # type: ignore[list-item]
            )
    else:
        observation, metadata = await orchestrator._bridge_visual_context(
            FakeSession(),  # type: ignore[arg-type]
            run,  # type: ignore[arg-type]
            [SimpleNamespace(id="sha256:image")],  # type: ignore[list-item]
        )
        assert observation == "A green apple."
        assert metadata["completion"]["finish_reason"] == "stop"

    assert [call.args[0].id for call in processes.load_chat.await_args_list] == [
        "profile-vision",
        "profile-text",
    ]


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
    assert parts[0].metadata_json == {
        "progress": 0.5,
        "phase": "sampling",
        "indeterminate": False,
    }
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
