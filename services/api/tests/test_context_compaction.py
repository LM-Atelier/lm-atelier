from __future__ import annotations

import json

from local_lm.context_compaction import (
    CONTEXT_COMPACTION_VERSION,
    MAX_PROVENANCE_SOURCE_IDS,
    compact_context_messages,
)


def test_context_compaction_is_deterministic_bounded_and_prompt_free() -> None:
    messages = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"Private synthetic detail {index} " * 20,
        }
        for index in range(40)
    ]
    source_ids = [f"message-{index}" for index in range(40)]

    first = compact_context_messages(messages, source_ids, max_characters=480)
    second = compact_context_messages(messages, source_ids, max_characters=480)

    assert first == second
    assert len(first.message["content"]) <= 480
    assert first.message["role"] == "assistant"
    assert first.provenance["version"] == CONTEXT_COMPACTION_VERSION
    assert first.provenance["source_message_count"] == 40
    assert len(first.provenance["source_message_ids"]) == MAX_PROVENANCE_SOURCE_IDS
    assert first.provenance["source_ids_truncated"] is True
    assert first.provenance["transcript_preserved"] is True
    assert "Private synthetic detail" not in json.dumps(first.provenance)


def test_context_compaction_uses_only_text_from_multimodal_content() -> None:
    compacted = compact_context_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the attached image"},
                    {"type": "image_url", "image_url": {"url": "private://image"}},
                ],
            }
        ],
        ["message-visual"],
        max_characters=300,
    )

    assert "Describe the attached image" in compacted.message["content"]
    assert "private://image" not in compacted.message["content"]
    assert compacted.provenance["source_message_ids"] == ["message-visual"]
