"""Who decided this conversation may reach the internet.

Two gates that both have to be open: the installation's, which a deployment
pins, and the chat's, which a person sets deliberately. Neither is inherited
and neither is on by default, because permission that arrives without anyone
granting it is not permission.
"""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient

from local_lm.config import Settings
from local_lm.web_access import may_fetch_urls

pytestmark = pytest.mark.asyncio


async def test_a_new_chat_cannot_reach_the_internet(client: AsyncClient) -> None:
    created = await client.post("/api/chats", json={"title": "Fresh"})

    assert created.status_code in {200, 201}
    assert created.json()["web_settings_json"] == {"allow_url_fetch": False}


async def test_the_choice_is_per_chat_and_does_not_spread(client: AsyncClient) -> None:
    opened = await client.post("/api/chats", json={"title": "Opened"})
    chat_id = opened.json()["id"]
    updated = await client.patch(
        f"/api/chats/{chat_id}",
        json={"web_settings_json": {"allow_url_fetch": True}},
    )

    assert updated.status_code == 200
    assert updated.json()["web_settings_json"]["allow_url_fetch"] is True

    # A chat made afterwards starts closed. Permission does not spread by
    # being adjacent to permission.
    later = await client.post("/api/chats", json={"title": "Later"})
    assert later.json()["web_settings_json"]["allow_url_fetch"] is False


async def test_the_installation_gate_is_reported_so_the_ui_can_say_why(
    client: AsyncClient, settings: Settings
) -> None:
    """A switch that cannot work should not be offered as if it could."""
    settings.web_access_enabled = False
    shut = await client.get("/api/about")
    assert shut.json()["web_access_enabled"] is False

    settings.web_access_enabled = True
    open_gate = await client.get("/api/about")
    assert open_gate.json()["web_access_enabled"] is True


@pytest.mark.parametrize(
    ("installation", "chat", "expected"),
    [
        (True, {"allow_url_fetch": True}, True),
        (False, {"allow_url_fetch": True}, False),
        (True, {"allow_url_fetch": False}, False),
        (True, {}, False),
        (True, None, False),
        (True, {"allow_url_fetch": "yes"}, False),
        (True, {"allow_url_fetch": 1}, False),
    ],
)
async def test_both_gates_must_be_open_and_only_an_exact_true_opens_one(
    installation: bool, chat: dict[str, object] | None, expected: bool
) -> None:
    """A truthy string is a malformed setting, and a malformed permission is none."""
    assert may_fetch_urls(installation_enabled=installation, chat_settings=chat) is expected
