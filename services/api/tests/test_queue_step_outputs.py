from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event
from sqlalchemy.orm import Session
from test_user_queue_activity import client as client
from test_user_queue_activity import plan
from test_user_queue_activity import session as session

from local_lm.models import (
    Message,
    MessagePart,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    WorkStep,
)
from local_lm.orchestrator import ConversationOrchestrator


def completed(
    session: Session, kinds: list[str] | None = None, *, replacement: bool = False
) -> tuple[Run, WorkStep, ResponseRevision, Message]:
    plan(session, "selected")
    request = Message(id="request", chat_id="chat-selected", role="user")
    staged = Message(id="staged", chat_id="chat-selected", role="assistant", status="complete")
    step = WorkStep(
        id="step",
        plan_id="selected",
        ordinal=1,
        operation="image",
        status="complete",
        output_contract_json=[{"count": 99}],
    )
    session.add_all([request, staged, step])
    session.flush()
    run = Run(
        id="run",
        chat_id="chat-selected",
        user_message_id=request.id,
        assistant_message_id=staged.id,
        work_plan_id="selected",
        work_step_id=step.id,
        operation="text_to_image",
        status="complete",
    )
    session.add(run)
    session.flush()
    step.run_id = run.id
    for position, kind in enumerate(kinds if kinds is not None else ["image", "video"]):
        staged.parts.append(
            MessagePart(
                position=position,
                type=kind,
                artifact_id="same-media",
                metadata_json={
                    "poster_artifact_id": "neutral-private-poster",
                    "browser_proxy_artifact_id": "neutral-private-proxy",
                },
            )
        )
    staged.parts.append(
        MessagePart(position=len(staged.parts), type="text", text="neutral-private-response")
    )
    staged.parts.append(
        MessagePart(
            position=len(staged.parts),
            type="generation_metadata",
            metadata_json={"prompt": "neutral-private-prompt"},
        )
    )
    shown = staged
    if replacement:
        shown = Message(id="shown", chat_id="chat-selected", role="assistant", status="complete")
        session.add(shown)
        session.flush()
        revision = ResponseRevision(
            id="replacement", message_id=shown.id, run_id=run.id, sequence=1, status="pending"
        )
        session.add(revision)
        session.flush()
        run.provenance_json = {
            "response_replacement": {"message_id": shown.id, "revision_id": revision.id}
        }
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    assert orchestrator._finalize_response_revision(session, run, staged, promote=True) == shown.id
    session.commit()
    revision = next(item for item in shown.response_revisions if item.run_id == run.id)
    return run, step, revision, shown


@pytest.mark.parametrize(
    "kinds, replacement",
    [
        (["image", "video"], False),
        (["image", "image"], False),
        ([], False),
        (["image", "video"], True),
    ],
)
async def test_recorded_outputs_follow_the_real_revision_writer_without_hydrating_content(
    session: Session,
    client: AsyncClient,
    kinds: list[str],
    replacement: bool,
) -> None:
    completed(session, kinds, replacement=replacement)
    session.expunge_all()
    statements: list[str] = []
    hydrated: list[object] = []

    def query(_c: object, _u: object, sql: str, _p: object, _x: object, _m: bool) -> None:
        if sql.lstrip().startswith("SELECT"):
            statements.append(sql)

    def load(_s: Session, obj: object) -> None:
        hydrated.append(obj)

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", query)
    event.listen(session, "loaded_as_persistent", load)
    try:
        response = await client.get("/api/queue/plans/selected/steps")
    finally:
        event.remove(engine, "before_cursor_execute", query)
        event.remove(session, "loaded_as_persistent", load)
    assert response.status_code == 200
    assert response.json()["items"][0]["recorded_media_outputs"] == len(kinds)
    assert len(statements) == 3 and hydrated == []
    assert "neutral-private" not in response.text


@pytest.mark.parametrize(
    "case",
    [
        "removed-content",
        "missing-revision",
        "missing-run-link",
        "wrong-step",
        "wrong-plan",
        "wrong-chat",
        "unfinished-run",
        "unfinished-revision",
        "unfinished-step",
    ],
)
async def test_output_count_omits_unavailable_or_mismatched_completion_evidence(
    session: Session,
    client: AsyncClient,
    case: str,
) -> None:
    run, step, revision, message = completed(session)
    if case == "removed-content":
        revision.parts.clear()
        message.parts.clear()
        session.flush()
        message.content_removed_at = datetime.now(UTC)
    elif case == "missing-revision":
        session.delete(revision)
    elif case == "missing-run-link":
        step.run_id = None
    elif case == "wrong-step":
        run.work_step_id = None
    elif case == "wrong-plan":
        plan(session, "other")
        run.work_plan_id = "other"
    elif case == "wrong-chat":
        plan(session, "other")
        message.chat_id = "chat-other"
    elif case == "unfinished-run":
        run.status = "running"
    elif case == "unfinished-revision":
        revision.status = "pending"
    elif case == "unfinished-step":
        step.status = "running"
    session.commit()
    response = await client.get("/api/queue/plans/selected/steps")
    assert response.status_code == 200
    assert response.json()["items"][0]["recorded_media_outputs"] is None


async def test_output_count_stays_with_its_run_when_another_revision_is_selected(
    session: Session,
    client: AsyncClient,
) -> None:
    _run, _step, _revision, message = completed(session)
    alternate = ResponseRevision(
        message_id=message.id,
        sequence=2,
        status="complete",
        parts=[ResponseRevisionPart(position=i, type="image") for i in range(5)],
    )
    session.add(alternate)
    session.commit()
    orchestrator = ConversationOrchestrator.__new__(ConversationOrchestrator)
    selected = orchestrator.select_response_revision(session, message.id, alternate.id)
    assert len(selected.parts) == 5
    response = await client.get("/api/queue/plans/selected/steps")
    assert response.status_code == 200
    assert response.json()["items"][0]["recorded_media_outputs"] == 2


async def test_null_artifact_references_do_not_erase_recorded_output_occurrences(
    session: Session,
    client: AsyncClient,
) -> None:
    _run, _step, revision, _message = completed(session)
    for part in revision.parts:
        part.artifact_id = None
    session.commit()
    response = await client.get("/api/queue/plans/selected/steps")
    assert response.status_code == 200
    assert response.json()["items"][0]["recorded_media_outputs"] == 2
