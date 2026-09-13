"""A chat's unsent composer draft stays in the database and holds the files it names."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.empty_chats import chat_state_fingerprint, configured_reasons
from local_lm.models import Artifact, Chat, ChatComposerDraft, ChatComposerDraftAttachment
from local_lm.prompt_helpers import PROMPT_HELPER_SCOPE

SOURCE = {
    "version": 1,
    "batch_id": "batch-neutral",
    "expected_plan_version": 2,
    "expected_plan_sha256": "a" * 64,
    "item_id": "item-neutral",
    "expected_review_version": 1,
    "expected_reviewed_sha256": "b" * 64,
    "prompt_template_id": "ptdef-neutral",
    "prompt_template_revision_id": "ptrev-neutral",
    "contract_sha256": "c" * 64,
}


async def _chat(client: AsyncClient) -> str:
    response = await client.post("/api/chats", json={})
    assert response.status_code == 201
    return str(response.json()["id"])


def _aged_preview(settings: Settings, label: str) -> str:
    """A picture nothing else uses, old enough that retention would clear it."""

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        artifact = store.ingest_bytes(
            session,
            f"neutral picture {label}".encode(),
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
            metadata={"temporary_preview": True},
        )
        artifact.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=48)
        session.commit()
        return artifact.id


def _draft(**overrides: Any) -> dict[str, Any]:
    return {
        "text": "A wider path between the beds",
        "mode": "image",
        "output_count": 3,
        "attachments": [],
        "mentions": [{"reference_subject_id": "refsub-garden", "mention_slug": "garden"}],
        "template_settings": {"name": "Soft light", "settings": {"steps": 20}},
        "prompt_source": None,
        **overrides,
    }


async def _put(client: AsyncClient, chat_id: str, revision: int, **overrides: Any) -> Any:
    return await client.put(
        f"/api/chats/{chat_id}/composer-draft",
        json={"expected_revision": revision, "draft": _draft(**overrides)},
    )


def _exists(artifact_id: str) -> bool:
    with SessionLocal() as session:
        return session.get(Artifact, artifact_id) is not None


async def _clear_now(client: AsyncClient) -> None:
    response = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert response.status_code == 200


async def test_a_chat_without_a_draft_reads_as_revision_zero(client: AsyncClient) -> None:
    chat_id = await _chat(client)

    response = await client.get(f"/api/chats/{chat_id}/composer-draft")

    assert response.status_code == 200
    body = response.json()
    assert body["revision"] == 0
    assert body["text"] == "" and body["attachments"] == [] and body["mode"] == "auto"


async def test_a_saved_draft_comes_back_whole(client: AsyncClient, settings: Settings) -> None:
    chat_id = await _chat(client)
    picture = _aged_preview(settings, "whole")

    saved = await _put(
        client,
        chat_id,
        0,
        attachments=[{"artifact_id": picture, "kind": "image", "origin": "generated"}],
        prompt_source=SOURCE,
    )
    assert saved.status_code == 200, saved.text
    read = (await client.get(f"/api/chats/{chat_id}/composer-draft")).json()

    assert read["revision"] == 1
    assert read["text"] == "A wider path between the beds"
    assert read["mode"] == "image" and read["output_count"] == 3
    assert read["attachments"] == [{"artifact_id": picture, "kind": "image", "origin": "generated"}]
    assert read["mentions"] == [{"reference_subject_id": "refsub-garden", "mention_slug": "garden"}]
    assert read["template_settings"] == {"name": "Soft light", "settings": {"steps": 20}}
    assert read["prompt_source"] == SOURCE


async def test_a_write_based_on_an_older_revision_is_refused_and_names_the_current_one(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    assert (await _put(client, chat_id, 0)).status_code == 200
    assert (await _put(client, chat_id, 1, text="From the first window")).status_code == 200

    # A second window that also read revision 1 must not overwrite the first.
    stale = await _put(client, chat_id, 1, text="From the second window")
    fresh_create = await _put(client, chat_id, 0, text="A new draft over an existing one")

    for response in (stale, fresh_create):
        assert response.status_code == 409
        assert response.json()["code"] == "chat-draft-revision-stale"
        assert response.json()["current_revision"] == 2
    read = (await client.get(f"/api/chats/{chat_id}/composer-draft")).json()
    assert read["text"] == "From the first window"


async def test_discarding_needs_the_current_revision_and_leaves_an_empty_draft_at_the_next_one(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    assert (await _put(client, chat_id, 0)).status_code == 200

    stale = await client.delete(
        f"/api/chats/{chat_id}/composer-draft", params={"expected_revision": 5}
    )
    assert stale.status_code == 409 and stale.json()["current_revision"] == 1
    discarded = await client.delete(
        f"/api/chats/{chat_id}/composer-draft", params={"expected_revision": 1}
    )

    assert discarded.status_code == 204
    read = (await client.get(f"/api/chats/{chat_id}/composer-draft")).json()
    assert read["revision"] == 2
    assert read["text"] == "" and read["mentions"] == [] and read["template_settings"] is None


@pytest.mark.parametrize(
    ("attachments", "reason"),
    [
        ([{"artifact_id": "sha256:not-stored", "kind": "image", "origin": "uploaded"}], "missing"),
        ("duplicate", "the same file twice"),
        ("model", "not a picture or video"),
    ],
)
async def test_an_attachment_that_cannot_be_held_is_refused(
    client: AsyncClient, settings: Settings, attachments: Any, reason: str
) -> None:
    chat_id = await _chat(client)
    if attachments == "duplicate":
        picture = _aged_preview(settings, "duplicate")
        attachments = [{"artifact_id": picture, "kind": "image", "origin": "generated"}] * 2
    elif attachments == "model":
        store = ArtifactStore(settings)
        with SessionLocal() as session:
            weights = store.ingest_bytes(
                session,
                b"neutral weights",
                kind=ArtifactKind.MODEL,
                media_type="application/octet-stream",
            )
            session.commit()
            attachments = [{"artifact_id": weights.id, "kind": "image", "origin": "uploaded"}]

    response = await _put(client, chat_id, 0, attachments=attachments)

    assert response.status_code == 422, reason
    assert response.json()["code"] == "chat-draft-attachment-unavailable"
    assert (await client.get(f"/api/chats/{chat_id}/composer-draft")).json()["revision"] == 0


async def test_template_settings_that_name_a_mask_are_refused(client: AsyncClient) -> None:
    chat_id = await _chat(client)

    response = await _put(
        client,
        chat_id,
        0,
        template_settings={"name": "Masked", "settings": {"mask": {"artifact_id": "sha256:x"}}},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "chat-draft-settings-unsupported"


async def test_only_standard_chats_have_drafts(client: AsyncClient) -> None:
    missing = await client.get("/api/chats/chat-missing/composer-draft")
    assert missing.status_code == 404 and missing.json()["code"] == "chat-not-found"

    with SessionLocal() as session:
        studio = Chat(scope=PROMPT_HELPER_SCOPE, title="Prompt helper")
        session.add(studio)
        session.commit()
        studio_id = studio.id
    refused = await _put(client, studio_id, 0)
    assert refused.status_code == 404


async def test_retention_and_clear_now_keep_a_file_only_a_draft_holds(
    client: AsyncClient, settings: Settings
) -> None:
    chat_id = await _chat(client)
    held = _aged_preview(settings, "held")
    loose = _aged_preview(settings, "loose")
    saved = await _put(
        client,
        chat_id,
        0,
        attachments=[{"artifact_id": held, "kind": "image", "origin": "generated"}],
    )
    assert saved.status_code == 200

    preview = (await client.post("/api/artifacts/cleanup", json={"dry_run": True})).json()
    await _clear_now(client)

    # The dry run counts only the loose picture, and a real run removes only it.
    assert preview["removed_count"] == 1
    assert _exists(held)
    assert not _exists(loose)


async def test_the_database_itself_refuses_to_delete_a_file_a_draft_holds(
    client: AsyncClient, settings: Settings
) -> None:
    chat_id = await _chat(client)
    held = _aged_preview(settings, "database")
    assert (
        await _put(
            client,
            chat_id,
            0,
            attachments=[{"artifact_id": held, "kind": "image", "origin": "generated"}],
        )
    ).status_code == 200

    loose = _aged_preview(settings, "database-loose")
    remove = text("DELETE FROM artifacts WHERE id = :artifact_id")

    with SessionLocal() as session:
        with pytest.raises(IntegrityError):
            session.execute(remove, {"artifact_id": held})
        session.rollback()
        # The same statement removes a file no draft holds, so the refusal above
        # is the draft's foreign key and not something about the statement.
        session.execute(remove, {"artifact_id": loose})
        session.commit()
    assert _exists(held)
    assert not _exists(loose)


@pytest.mark.parametrize("release", ["replaced", "discarded", "chat deleted"])
async def test_a_file_is_released_when_the_draft_lets_go(
    client: AsyncClient, settings: Settings, release: str
) -> None:
    chat_id = await _chat(client)
    held = _aged_preview(settings, release)
    assert (
        await _put(
            client,
            chat_id,
            0,
            attachments=[{"artifact_id": held, "kind": "image", "origin": "generated"}],
        )
    ).status_code == 200

    if release == "replaced":
        assert (await _put(client, chat_id, 1, attachments=[])).status_code == 200
    elif release == "discarded":
        response = await client.delete(
            f"/api/chats/{chat_id}/composer-draft", params={"expected_revision": 1}
        )
        assert response.status_code == 204
    else:
        assert (await client.delete(f"/api/chats/{chat_id}")).status_code == 204
    await _clear_now(client)

    assert not _exists(held)
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(ChatComposerDraftAttachment)) == 0
        if release == "chat deleted":
            assert session.scalar(select(func.count()).select_from(ChatComposerDraft)) == 0


async def test_a_draft_makes_an_empty_chat_somebodys_and_changes_its_fingerprint(
    client: AsyncClient, settings: Settings
) -> None:
    chat_id = await _chat(client)
    picture = _aged_preview(settings, "fingerprint")

    def observe() -> tuple[tuple[str, ...], str]:
        with SessionLocal() as session:
            chat = session.get(Chat, chat_id)
            assert chat is not None
            return configured_reasons(session, chat), chat_state_fingerprint(session, chat)

    reasons, untouched = observe()
    assert "has_draft" not in reasons

    # An attachment alone, with no words, is still something somebody put there.
    assert (
        await _put(
            client,
            chat_id,
            0,
            text="",
            mentions=[],
            template_settings=None,
            attachments=[{"artifact_id": picture, "kind": "image", "origin": "generated"}],
        )
    ).status_code == 200
    reasons, drafted = observe()
    assert "has_draft" in reasons
    assert drafted != untouched

    # Discarded and written again with only the words different: the fingerprint
    # sees the change.
    assert (
        await client.delete(f"/api/chats/{chat_id}/composer-draft", params={"expected_revision": 1})
    ).status_code == 204
    # Only the words differ: same revision, same attachment, same everything else.
    assert (
        await _put(
            client,
            chat_id,
            2,
            text="Something else",
            mentions=[],
            template_settings=None,
            attachments=[{"artifact_id": picture, "kind": "image", "origin": "generated"}],
        )
    ).status_code == 200
    _reasons, rewritten = observe()
    assert rewritten not in {untouched, drafted}
