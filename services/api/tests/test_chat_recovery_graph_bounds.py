"""A conversation's recovery graph refuses by size separately from damage, and keeps room."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import text, update
from sqlalchemy.orm import Session
from test_chat_recovery import _chat, _command, _impact
from test_chat_recovery_graph import _artifact_id, _graph
from test_chat_recovery_graph import graph_session as graph_session

from local_lm import chat_recovery_graph
from local_lm.chat_media_purge import generated_media_ids
from local_lm.chat_recovery import (
    chat_recovery_page,
    preview_chat_recovery,
    preview_chat_trash,
    purge_chat,
    trash_chat,
)
from local_lm.models import Artifact, Chat, Job, MessagePart, Run
from local_lm.recovery_v1 import PurgeRecoveryV1, RecoveryImpactV1, TrashChatV1

NOW = datetime(2026, 10, 2, tzinfo=UTC)


def _fields(impact: RecoveryImpactV1, key: str) -> dict[str, str]:
    return {
        "expected_revision": impact.revision,
        "impact_sha256": impact.impact_sha256,
        "operation_key": key,
    }


def _rows(session: Session, monkeypatch: pytest.MonkeyPatch) -> int:
    readers: list[chat_recovery_graph._GraphReader] = []

    class Recording(chat_recovery_graph._GraphReader):
        def __init__(self, current: Session) -> None:
            super().__init__(current)
            readers.append(self)

    with monkeypatch.context() as patch:
        patch.setattr(chat_recovery_graph, "_GraphReader", Recording)
        _graph(session)
    session.rollback()
    return readers[-1].row_count


@pytest.mark.parametrize("bound", ["MAX_GRAPH_ROWS", "MAX_GRAPH_BYTES", "MAX_GRAPH_READ_BYTES"])
def test_a_size_refusal_is_its_own_kind_of_graph_refusal(
    graph_session: Session, monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    monkeypatch.setattr(chat_recovery_graph, bound, 1)
    with pytest.raises(chat_recovery_graph.ChatRecoveryGraphError) as refused:
        _graph(graph_session)
    assert type(refused.value) is chat_recovery_graph.ChatRecoveryGraphTooLarge
    assert str(refused.value) == "chat-recovery-graph-too-large"


def _drop_guard(session: Session, table: str) -> None:
    session.execute(text(f"DROP TRIGGER IF EXISTS {table}_artifact_reference_update_guard"))


def _run_inputs(session: Session) -> None:
    _drop_guard(session, "runs")
    session.execute(
        update(Run)
        .where(Run.id == "run-garden")
        .values(provenance_json={"input_artifact_ids": [123]})
    )


def _job_payload_list(session: Session) -> None:
    _drop_guard(session, "jobs")
    session.execute(update(Job).where(Job.id == "job-garden").values(payload_json=[]))


def _job_chat_number(session: Session) -> None:
    session.execute(update(Job).where(Job.id == "job-garden").values(payload_json={"chat_id": 5}))


def _artifact_metadata_list(session: Session) -> None:
    _drop_guard(session, "artifacts")
    session.execute(update(Artifact).where(Artifact.id == _artifact_id(1)).values(metadata_json=[]))


@pytest.mark.parametrize(
    "damage", [_run_inputs, _job_payload_list, _job_chat_number, _artifact_metadata_list]
)
def test_damaged_stored_references_still_refuse_as_invalid_rather_than_too_large(
    graph_session: Session, damage: Callable[[Session], None]
) -> None:
    damage(graph_session)
    graph_session.commit()
    with pytest.raises(chat_recovery_graph.ChatRecoveryGraphError) as refused:
        _graph(graph_session)
    assert type(refused.value) is chat_recovery_graph.ChatRecoveryGraphError
    assert str(refused.value) == "chat-recovery-graph-invalid"


def test_a_parent_from_another_conversation_is_hashed_whole(graph_session: Session) -> None:
    graph_session.add(
        Job(
            id="job-shared",
            run_id="run-other",
            status="complete",
            payload_json={"chat_id": "chat-garden"},
        )
    )
    graph_session.commit()
    before = _graph(graph_session)
    graph_session.commit()
    assert before.owned_job_ids == ("job-garden",)
    graph_session.execute(
        update(Run).where(Run.id == "run-other").values(duration_ms=7, updated_at=Run.updated_at)
    )
    graph_session.commit()
    after = _graph(graph_session)
    assert after.snapshot.counts == before.snapshot.counts
    assert after.snapshot.fingerprint != before.snapshot.fingerprint


def test_permanent_media_release_selects_only_assistant_outputs(graph_session: Session) -> None:
    graph_session.add(
        MessagePart(
            id="part-user-garden",
            message_id="user-garden",
            position=0,
            type="image",
            artifact_id=_artifact_id(4),
        )
    )
    graph_session.commit()
    assert generated_media_ids(_graph(graph_session)) == frozenset({_artifact_id(1)})


def test_a_conversation_is_admitted_only_while_recently_deleted_can_still_list_and_purge_it(
    graph_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _rows(graph_session, monkeypatch)
    # Exactly its own size leaves no room for the deletion record that trash adds.
    monkeypatch.setattr(chat_recovery_graph, "MAX_GRAPH_ROWS", rows)
    with pytest.raises(chat_recovery_graph.ChatRecoveryGraphError) as refused:
        preview_chat_trash(graph_session, "chat-garden", NOW)
    assert str(refused.value) == "chat-recovery-graph-too-large"
    graph_session.rollback()
    monkeypatch.setattr(chat_recovery_graph, "MAX_GRAPH_ROWS", -(-16 * rows // 15))
    preview = preview_chat_trash(graph_session, "chat-garden", NOW)
    graph_session.commit()
    item = trash_chat(
        graph_session, "chat-garden", TrashChatV1.model_validate(_fields(preview, "trash")), NOW
    )
    graph_session.commit()
    page = chat_recovery_page(graph_session, NOW)
    graph_session.commit()
    assert [listed.deletion_id for listed in page.items] == [item.deletion_id]
    assert page.items[0].counts == item.counts
    preview = preview_chat_recovery(graph_session, item.deletion_id, NOW)
    graph_session.commit()
    command = PurgeRecoveryV1.model_validate(
        {**_fields(preview, "purge"), "acknowledgement": "permanently-delete"}
    )
    purge_chat(graph_session, item.deletion_id, command, NOW)
    graph_session.commit()
    assert graph_session.get(Chat, "chat-garden") is None


async def test_a_conversation_over_the_size_bound_says_it_is_too_long(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id = await _chat(client)
    monkeypatch.setattr(chat_recovery_graph, "MAX_GRAPH_ROWS", 1)
    refused = await client.get(f"/api/chats/{chat_id}/deletion-impact")
    assert refused.status_code == 409
    assert refused.json()["code"] == "chat-recovery-graph-too-large"
    assert (
        refused.json()["detail"] == "This conversation is too long to check its deletion details."
    )


async def test_a_batch_with_a_conversation_over_the_size_bound_says_it_is_too_long(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id = await _chat(client)
    preview = await _impact(client, f"/api/chats/{chat_id}/deletion-impact")
    trashed = await client.post(f"/api/chats/{chat_id}/trash", json=_command(preview, "trash"))
    assert trashed.status_code == 200, trashed.text
    monkeypatch.setattr(chat_recovery_graph, "MAX_GRAPH_ROWS", 1)
    refused = await client.post(
        "/api/recovery-items/batches",
        json={"deletion_ids": [trashed.json()["deletion_id"]], "action": "purge"},
    )
    assert refused.status_code == 409
    assert refused.json()["code"] == "chat-recovery-graph-too-large"
