from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from local_lm.adapters.base import ChatEvent, ChatRequest, estimate_chat_tokens
from local_lm.adapters.contracts import GuardedChatAdapter
from local_lm.adapters.llama_cpp import LlamaCppAdapter
from local_lm.adapters.message_projection import (
    ADAPTER_MESSAGE_PROJECTION_VERSION,
    ADAPTER_MESSAGE_SEPARATOR,
    MAX_ADAPTER_PROJECTION_SOURCE_IDS,
    project_chat_messages,
)
from local_lm.adapters.mock import MockChatAdapter
from local_lm.adapters.vllm import VllmAdapter
from local_lm.db import SessionLocal
from local_lm.domain import MessageRole, MessageStatus, Operation, utcnow
from local_lm.models import Chat, Message, MessagePart, Run
from local_lm.orchestrator import ConversationOrchestrator


def test_projection_coalesces_in_order_with_bounded_source_attribution() -> None:
    messages = [
        {"role": "system", "content": "First instruction."},
        {"role": "system", "content": "Second instruction."},
        {"role": "user", "content": "Earlier surviving request."},
        {"role": "user", "content": "Later surviving request."},
        {"role": "assistant", "content": "Earlier surviving reply."},
        {"role": "assistant", "content": "Later surviving reply."},
    ]
    source_ids = [None, None, "user-1", "user-3", "assistant-4", "assistant-6"]
    original = copy.deepcopy(messages)

    projected = project_chat_messages(messages, source_ids)

    assert messages == original
    assert projected.messages == [
        {
            "role": "system",
            "content": f"First instruction.{ADAPTER_MESSAGE_SEPARATOR}Second instruction.",
        },
        {
            "role": "user",
            "content": (
                f"Earlier surviving request.{ADAPTER_MESSAGE_SEPARATOR}Later surviving request."
            ),
        },
        {
            "role": "assistant",
            "content": (
                f"Earlier surviving reply.{ADAPTER_MESSAGE_SEPARATOR}Later surviving reply."
            ),
        },
    ]
    assert projected.source_message_ids == [
        (),
        ("user-1", "user-3"),
        ("assistant-4", "assistant-6"),
    ]
    metadata = projected.metadata(original_message_count=len(messages))
    assert metadata == {
        "version": ADAPTER_MESSAGE_PROJECTION_VERSION,
        "separator": ADAPTER_MESSAGE_SEPARATOR,
        "original_message_count": 6,
        "projected_message_count": 3,
        "coalesced_message_count": 3,
        "source_message_count": 4,
        "source_message_ids": [
            [],
            ["user-1", "user-3"],
            ["assistant-4", "assistant-6"],
        ],
        "source_ids_truncated": False,
    }
    assert "Message removed" not in json.dumps(projected.messages)


def test_projection_preserves_multimodal_order_and_nonordinary_message_keys() -> None:
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    projected = project_chat_messages(
        [
            {"role": "user", "content": "Earlier text."},
            {"role": "user", "content": [{"type": "text", "text": "Later text."}, image]},
            {"role": "assistant", "content": "Tool request", "tool_calls": []},
            {"role": "assistant", "content": "Ordinary reply"},
        ]
    )

    assert projected.messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Earlier text."},
                {"type": "text", "text": ADAPTER_MESSAGE_SEPARATOR},
                {"type": "text", "text": "Later text."},
                image,
            ],
        },
        {"role": "assistant", "content": "Tool request", "tool_calls": []},
        {"role": "assistant", "content": "Ordinary reply"},
    ]


def test_projection_metadata_caps_source_ids_without_reordering() -> None:
    count = MAX_ADAPTER_PROJECTION_SOURCE_IDS + 3
    projected = project_chat_messages(
        [{"role": "user", "content": str(index)} for index in range(count)],
        [f"message-{index:03}" for index in range(count)],
    )

    metadata = projected.metadata(original_message_count=count)
    assert metadata["source_message_count"] == count
    assert metadata["source_ids_truncated"] is True
    assert metadata["source_message_ids"] == [
        [f"message-{index:03}" for index in range(MAX_ADAPTER_PROJECTION_SOURCE_IDS)]
    ]


async def test_removed_item_creates_one_attributed_adapter_safe_message(
    settings: object,
) -> None:
    del settings
    with SessionLocal() as session:
        chat = Chat(id="chat-adjacency", title="Constructed adjacency")
        first_user = Message(
            id="message-user-1",
            chat=chat,
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
        )
        removed_assistant = Message(
            id="message-assistant-2",
            chat=chat,
            parent_id=first_user.id,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.COMPLETE.value,
            content_removed_at=utcnow(),
        )
        second_user = Message(
            id="message-user-3",
            chat=chat,
            parent_id=removed_assistant.id,
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
        )
        surviving_assistant = Message(
            id="message-assistant-4",
            chat=chat,
            parent_id=second_user.id,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.COMPLETE.value,
        )
        removed_user = Message(
            id="message-user-5",
            chat=chat,
            parent_id=surviving_assistant.id,
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
            content_removed_at=utcnow(),
        )
        second_assistant = Message(
            id="message-assistant-6",
            chat=chat,
            parent_id=removed_user.id,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.COMPLETE.value,
        )
        current_user = Message(
            id="message-user-7",
            chat=chat,
            parent_id=second_assistant.id,
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
        )
        pending_assistant = Message(
            id="message-assistant-8",
            chat=chat,
            parent_id=current_user.id,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.PENDING.value,
        )
        session.add_all(
            [
                chat,
                first_user,
                removed_assistant,
                second_user,
                surviving_assistant,
                removed_user,
                second_assistant,
                current_user,
                pending_assistant,
                MessagePart(
                    message=first_user,
                    position=0,
                    type="text",
                    text="Earlier surviving request.",
                    metadata_json={},
                ),
                MessagePart(
                    message=second_user,
                    position=0,
                    type="text",
                    text="Later surviving request.",
                    metadata_json={},
                ),
                MessagePart(
                    message=surviving_assistant,
                    position=0,
                    type="text",
                    text="Surviving reply.",
                    metadata_json={},
                ),
                MessagePart(
                    message=second_assistant,
                    position=0,
                    type="text",
                    text="Later surviving reply.",
                    metadata_json={},
                ),
                MessagePart(
                    message=current_user,
                    position=0,
                    type="text",
                    text="Current request.",
                    metadata_json={},
                ),
            ]
        )
        session.flush()
        run = Run(
            id="run-adjacency",
            chat_id=chat.id,
            user_message_id=current_user.id,
            assistant_message_id=pending_assistant.id,
            operation=Operation.TEXT.value,
            status="queued",
            standalone_prompt="Current request.",
            provenance_json={},
            settings_json={"max_tokens": 64},
        )
        session.add(run)
        session.commit()

        engines = SimpleNamespace(
            chat=SimpleNamespace(
                count_tokens=AsyncMock(side_effect=lambda value: estimate_chat_tokens(value))
            ),
            chat_capabilities=AsyncMock(
                return_value=SimpleNamespace(input_modalities=["text"], tool_calling=False)
            ),
            settings=SimpleNamespace(vision_prior_visual_lookback=0, chat_engine="mock"),
        )
        orchestrator = ConversationOrchestrator(
            engines=engines,
            artifacts=Mock(),
            events=Mock(),
            scheduler=Mock(),
            processes=Mock(),
        )

        (
            messages,
            _request_settings,
            metadata,
            _tool_calling,
        ) = await orchestrator._prepare_chat_context(session, run)

    assert messages == [
        {
            "role": "user",
            "content": (
                f"Earlier surviving request.{ADAPTER_MESSAGE_SEPARATOR}Later surviving request."
            ),
        },
        {
            "role": "assistant",
            "content": f"Surviving reply.{ADAPTER_MESSAGE_SEPARATOR}Later surviving reply.",
        },
        {"role": "user", "content": "Current request."},
    ]
    assert metadata["adapter_projection"] == {
        "version": ADAPTER_MESSAGE_PROJECTION_VERSION,
        "separator": ADAPTER_MESSAGE_SEPARATOR,
        "original_message_count": 5,
        "projected_message_count": 3,
        "coalesced_message_count": 2,
        "source_message_count": 5,
        "source_message_ids": [
            ["message-user-1", "message-user-3"],
            ["message-assistant-4", "message-assistant-6"],
            ["message-user-7"],
        ],
        "source_ids_truncated": False,
    }
    assert "Message removed" not in json.dumps(messages)


@pytest.mark.parametrize("adapter_type", [LlamaCppAdapter, VllmAdapter])
async def test_openai_compatible_adapters_project_before_count_and_stream(
    adapter_type: type[LlamaCppAdapter],
) -> None:
    observed: list[list[dict[str, Any]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        observed.append(payload["messages"])
        if request.url.path == "/v1/chat/completions/input_tokens":
            return httpx.Response(200, json={"input_tokens": 7})
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            text="data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    adapter = adapter_type("http://adapter.test")
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(
        base_url="http://adapter.test",
        transport=httpx.MockTransport(handler),
    )
    messages = [
        {"role": "user", "content": "Earlier surviving request."},
        {"role": "user", "content": "Later surviving request."},
    ]
    try:
        assert await adapter.count_tokens(messages) == 7
        events = [
            event
            async for event in adapter.stream(ChatRequest(run_id="projection", messages=messages))
        ]
    finally:
        await adapter.close()

    expected = [
        {
            "role": "user",
            "content": (
                f"Earlier surviving request.{ADAPTER_MESSAGE_SEPARATOR}Later surviving request."
            ),
        }
    ]
    assert observed == [expected, expected]
    assert [event.type for event in events] == ["complete"]


async def test_mock_adapter_keeps_both_trailing_user_messages() -> None:
    adapter = MockChatAdapter()
    events = [
        event
        async for event in adapter.stream(
            ChatRequest(
                run_id="mock-projection",
                messages=[
                    {"role": "user", "content": "Earlier surviving request."},
                    {"role": "user", "content": "Later surviving request."},
                ],
            )
        )
    ]
    response = "".join(event.text for event in events)

    assert (
        f"Earlier surviving request.{ADAPTER_MESSAGE_SEPARATOR}Later surviving request." in response
    )
    assert events[-1].type == "complete"


class _RecordingExternalAdapter:
    def __init__(self) -> None:
        self.counted: list[list[dict[str, Any]]] = []
        self.streamed: list[list[dict[str, Any]]] = []

    async def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        self.counted.append(messages)
        return 5

    async def stream(self, request: ChatRequest):  # type: ignore[no-untyped-def]
        self.streamed.append(request.messages)
        yield ChatEvent(type="complete")


async def test_guarded_external_adapter_projects_before_both_entry_points() -> None:
    external = _RecordingExternalAdapter()
    adapter = GuardedChatAdapter(external, "constructed-adapter")  # type: ignore[arg-type]
    messages = [
        {"role": "assistant", "content": "Earlier surviving reply."},
        {"role": "assistant", "content": "Later surviving reply."},
    ]

    assert await adapter.count_tokens(messages) == 5
    events = [
        event async for event in adapter.stream(ChatRequest(run_id="external", messages=messages))
    ]

    expected = [
        {
            "role": "assistant",
            "content": (
                f"Earlier surviving reply.{ADAPTER_MESSAGE_SEPARATOR}Later surviving reply."
            ),
        }
    ]
    assert external.counted == [expected]
    assert external.streamed == [expected]
    assert [event.type for event in events] == ["complete"]
