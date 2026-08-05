"""Reading a linked page inside a real turn, and the gates that stop it.

The answering pass is built with no tools at all. That is what makes quoting a
page safe rather than merely marked: a page telling the model to do something
has nothing to do it with.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.web_lookup import source_message
from local_lm.web_retrieval import RetrievedSource

pytestmark = pytest.mark.asyncio


def _source() -> RetrievedSource:
    return RetrievedSource(
        url="https://example.com/a",
        final_url="https://example.com/a",
        title="Release notes",
        text="Version 4 adds the thing.",
        byte_count=64,
        truncated=False,
    )


async def test_the_page_joins_the_conversation_as_a_user_message() -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "what changed?"}]
    messages.append(source_message(_source()))

    assert messages[-1]["role"] == "user"
    assert "Version 4 adds the thing." in messages[-1]["content"]
    # The system prompt is untouched: a page cannot become an operator.
    assert not any(message["role"] == "system" for message in messages)


async def test_the_answering_request_carries_no_tools() -> None:
    """The property the whole boundary rests on, asserted rather than assumed.

    `ChatRequest` for the answering pass is constructed without `tools`, so a
    retrieved page has nothing to reach even if it asks.
    """
    from local_lm.adapters.base import ChatRequest

    request = ChatRequest(
        run_id="run-1",
        messages=[{"role": "user", "content": "hi"}, source_message(_source())],
        settings={},
    )

    assert not getattr(request, "tools", None)


@pytest.mark.parametrize(
    ("installation", "chat_settings"),
    [
        (False, {"allow_url_fetch": True}),
        (True, {"allow_url_fetch": False}),
        (True, {}),
        (True, None),
    ],
)
async def test_a_closed_gate_reads_nothing(
    installation: bool, chat_settings: dict[str, Any] | None
) -> None:
    from local_lm.web_access import may_fetch_urls

    assert may_fetch_urls(installation_enabled=installation, chat_settings=chat_settings) is False
