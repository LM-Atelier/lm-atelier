"""The read surface for empty-chat maintenance.

The interesting assertions are about what the response refuses to contain. The
feature this serves ends in a delete button over chats somebody might not want,
which makes its own payload exactly the wrong place to reproduce what they wrote.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from httpx2 import AsyncClient
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.models import Chat, Message, MessageRole

ROUTE = "/api/maintenance/empty-chats"


def _chat(session: Session, **values: object) -> str:
    row = Chat(**values)
    session.add(row)
    session.flush()
    return row.id


def _two_days_ago() -> datetime:
    return datetime.now(UTC) - timedelta(hours=48)


async def test_the_page_describes_decisions_and_never_content(client: AsyncClient) -> None:
    old = _two_days_ago()
    with SessionLocal() as session:
        blank = _chat(session, created_at=old)
        titled = _chat(session, title="A title I would not want listed", created_at=old)
        drafted = _chat(session, draft_prompt="a draft nobody else should read", created_at=old)
        busy = _chat(session, created_at=old)
        session.add(Message(chat_id=busy, role=MessageRole.USER.value))
        session.commit()

    response = await client.get(ROUTE, params={"min_age_hours": 0})
    assert response.status_code == 200, response.json()
    body = response.json()
    shown = {entry["id"] for entry in body["entries"]}

    # A chat with a message is not empty, whatever it looks like.
    assert busy not in shown
    # Configured chats are excluded by default; the blank one is offered.
    assert shown == {blank}
    assert body["counts"] == {"strict_blank": 1}
    assert body["evaluated_at"].endswith("Z")

    # Nothing anybody wrote appears anywhere in the payload, asserted over the
    # whole response text rather than field by field.
    serialized = response.text
    assert "A title I would not want listed" not in serialized
    assert "a draft nobody else should read" not in serialized
    assert titled not in serialized
    assert drafted not in serialized


async def test_including_configured_reports_why_without_quoting_it(client: AsyncClient) -> None:
    with SessionLocal() as session:
        titled = _chat(
            session, title="Keep this one", draft_prompt="unsent", created_at=_two_days_ago()
        )
        session.commit()

    response = await client.get(ROUTE, params={"min_age_hours": 0, "include_configured": True})
    assert response.status_code == 200, response.json()
    entry = next(item for item in response.json()["entries"] if item["id"] == titled)

    assert entry["classification"] == "configured_blank"
    assert set(entry["reasons"]) == {"custom_title", "has_draft"}
    assert entry["deletable"] is True
    assert "Keep this one" not in response.text
    assert "unsent" not in response.text


async def test_an_inconsistent_chat_is_listed_as_not_deletable(client: AsyncClient) -> None:
    with SessionLocal() as session:
        broken = _chat(session, active_head_message_id="msg_missing", created_at=_two_days_ago())
        session.commit()

    response = await client.get(ROUTE, params={"min_age_hours": 0})
    entry = next(item for item in response.json()["entries"] if item["id"] == broken)

    assert entry["classification"] == "inconsistent"
    assert entry["reasons"] == ["stale_head"]
    assert entry["deletable"] is False


async def test_the_age_floor_defaults_to_a_day(client: AsyncClient) -> None:
    """A chat opened and abandoned minutes ago is not offered by default.

    The floor lives at this surface rather than in the reader, so the default
    has to be asserted where it is applied.
    """

    with SessionLocal() as session:
        _chat(session)
        session.commit()

    default = await client.get(ROUTE)
    assert default.status_code == 200, default.json()
    assert default.json()["entries"] == []

    asked = await client.get(ROUTE, params={"min_age_hours": 0})
    assert len(asked.json()["entries"]) == 1
    assert asked.json()["entries"][0]["age_hours"] >= 0


async def test_nonsense_bounds_are_refused_rather_than_guessed(client: AsyncClient) -> None:
    for params in (
        {"min_age_hours": -1},
        {"limit": 0},
        {"limit": 201},
        {"cursor": "c" * 41},
    ):
        response = await client.get(ROUTE, params=params)
        assert response.status_code == 422, (params, response.json())


async def test_a_configured_first_window_does_not_hide_the_eligible_chat(
    client: AsyncClient,
) -> None:
    """Pins the reader's filling rule at the surface that shows the result.

    If configured chats consumed the page before the filter applied, a small page
    whose first window happened to be configured would come back empty with a
    cursor, and an empty maintenance page reads as nothing to maintain.
    """

    old = _two_days_ago()
    with SessionLocal() as session:
        _chat(session, id="chat_001", title="Configured one", created_at=old)
        _chat(session, id="chat_002", title="Configured two", created_at=old)
        _chat(session, id="chat_003", created_at=old)
        session.commit()

    response = await client.get(ROUTE, params={"min_age_hours": 0, "limit": 2})
    assert response.status_code == 200, response.json()
    body = response.json()

    assert [entry["id"] for entry in body["entries"]] == ["chat_003"]
    assert body["counts"] == {"strict_blank": 1}


async def test_a_chat_created_through_the_api_is_offered_as_untouched(
    client: AsyncClient,
) -> None:
    """The case every unit fixture misses, because fixtures build rows directly.

    Creating a chat through the product writes more than a bare `Chat()` does:
    automatic profile pointers, seeded vision settings and four automatic
    workflow selection rows. Reading any of those as a decision, or as work,
    would leave the default list empty of exactly the chats it exists to offer.
    So this creates the chat the way the product does.
    """

    created = await client.post("/api/chats", json={})
    assert created.status_code == 201, created.json()
    chat_id = created.json()["id"]

    with SessionLocal() as session:
        row = session.get(Chat, chat_id)
        assert row is not None
        # Aged past the default floor without touching anything a person would.
        row.created_at = _two_days_ago()
        session.commit()

    listed = await client.get(ROUTE)
    assert listed.status_code == 200, listed.json()
    entries = {entry["id"]: entry for entry in listed.json()["entries"]}
    assert chat_id in entries, "a freshly created chat was not offered at all"
    assert entries[chat_id]["classification"] == "strict_blank", entries[chat_id]
    assert entries[chat_id]["reasons"] == [], entries[chat_id]


async def test_choosing_a_profile_or_vision_setting_still_counts_as_configured(
    client: AsyncClient,
) -> None:
    """The other side of the same boundary, so the defaults rule cannot be a blanket pass.

    Reading server defaults as decisions hides chats; reading real decisions as
    defaults would offer somebody's chat as untouched.
    """

    old = _two_days_ago()
    with SessionLocal() as session:
        picked = _chat(session, created_at=old, active_chat_profile_id="profile_real")
        tuned = _chat(session, created_at=old, vision_settings_json={"verify_image_edits": False})
        session.commit()

    listed = await client.get(ROUTE, params={"include_configured": True})
    entries = {entry["id"]: entry for entry in listed.json()["entries"]}
    assert entries[picked]["classification"] == "configured_blank"
    assert "profile_chosen" in entries[picked]["reasons"]
    assert entries[tuned]["classification"] == "configured_blank"
    assert "settings_overridden" in entries[tuned]["reasons"]


async def test_reading_the_page_changes_nothing(client: AsyncClient) -> None:
    """Read-only is a property to check, not a docstring to trust."""

    with SessionLocal() as session:
        blank = _chat(session, created_at=_two_days_ago())
        session.commit()
        before = session.get(Chat, blank)
        assert before is not None
        stamp = before.updated_at

    for _ in range(2):
        response = await client.get(ROUTE, params={"min_age_hours": 0, "include_configured": True})
        assert response.status_code == 200, response.json()

    with SessionLocal() as session:
        after = session.get(Chat, blank)
        assert after is not None
        assert after.updated_at == stamp
