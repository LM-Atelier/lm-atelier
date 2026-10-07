"""Recovery impact reads history, queue ownership and media links from canonical rows."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session

from local_lm.chat_recovery_graph import ChatRecoveryGraph
from local_lm.config import Settings
from local_lm.db import Base, create_database_engine
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    ChatComposerDraft,
    Job,
    Message,
    MessagePart,
    MessageReference,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    RunContextArtifact,
    RunContextSnapshot,
    TurnCreationClaim,
    WorkPlan,
    WorkStep,
)
from local_lm.recovery_v1 import RecoveryConflict


def _artifact_id(index: int) -> str:
    return "sha256:" + f"{index:064x}"


@pytest.fixture
def graph_session(tmp_path: Path) -> Iterator[Session]:
    settings = Settings(data_dir=tmp_path / "data", chat_engine="mock", media_engine="mock")
    settings.prepare()
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                Chat(id="chat-garden", title="Garden notes"),
                Chat(id="chat-other", title="Other notes"),
            ]
        )
        session.flush()
        session.add_all(
            [
                Message(id="user-garden", chat_id="chat-garden"),
                Message(id="assistant-garden", chat_id="chat-garden", role="assistant"),
                Message(id="user-other", chat_id="chat-other"),
                Message(id="assistant-other", chat_id="chat-other", role="assistant"),
                ChatComposerDraft(chat_id="chat-garden", text="Keep the garden path wide"),
                WorkPlan(
                    id="plan-garden",
                    chat_id="chat-garden",
                    transcript_sequence=1,
                    status="complete",
                ),
            ]
        )
        session.flush()
        session.add(
            WorkStep(
                id="step-garden",
                plan_id="plan-garden",
                ordinal=0,
                operation="text_to_image",
                status="complete",
            )
        )
        for index in range(1, 6):
            session.add(
                Artifact(
                    id=_artifact_id(index),
                    sha256=f"{index:064x}",
                    kind="image",
                    media_type="image/png",
                    size_bytes=index * 10,
                    relative_path=f"pictures/{index}.png",
                )
            )
        session.flush()
        picture = session.get(Artifact, _artifact_id(1))
        assert picture is not None
        picture.metadata_json = {
            "poster_artifact_id": _artifact_id(2),
            "browser_proxy_artifact_id": _artifact_id(3),
        }
        session.add_all(
            [
                Run(
                    id="run-garden",
                    chat_id="chat-garden",
                    user_message_id="user-garden",
                    assistant_message_id="assistant-garden",
                    work_plan_id="plan-garden",
                    work_step_id="step-garden",
                    status="complete",
                    provenance_json={
                        "outputs": [{"artifact_id": _artifact_id(1)}],
                        "input_artifact_ids": [_artifact_id(4)],
                    },
                    settings_json={"mask": {"artifact_id": _artifact_id(5)}},
                ),
                Run(
                    id="run-other",
                    chat_id="chat-other",
                    user_message_id="user-other",
                    assistant_message_id="assistant-other",
                    status="complete",
                ),
                MessagePart(
                    id="part-garden",
                    message_id="assistant-garden",
                    position=0,
                    type="image",
                    artifact_id=_artifact_id(1),
                ),
                ArtifactLibraryEntry(
                    id="libentry:sha256:" + f"{1:064x}",
                    artifact_id=_artifact_id(1),
                    display_name="Garden picture",
                    favorite=True,
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                Job(
                    id="job-garden",
                    run_id="run-garden",
                    work_plan_id="plan-garden",
                    work_step_id="step-garden",
                    status="complete",
                    payload_json={"chat_id": "chat-garden", "source_artifact_id": _artifact_id(1)},
                ),
                RunContextSnapshot(run_id="run-garden", payload_json={}, sha256="a" * 64),
                ResponseRevision(
                    id="revision-garden",
                    message_id="assistant-garden",
                    run_id="run-garden",
                    sequence=1,
                    status="complete",
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                RunContextArtifact(run_id="run-garden", artifact_id=_artifact_id(4)),
                ResponseRevisionPart(
                    id="revision-part-garden",
                    response_revision_id="revision-garden",
                    position=0,
                    type="image",
                    artifact_id=_artifact_id(1),
                ),
            ]
        )
        session.commit()
        yield session
    engine.dispose()


def _graph(session: Session) -> ChatRecoveryGraph:
    from local_lm.chat_recovery_graph import inspect_chat_recovery_graph
    from local_lm.recovery_previews import reserve_recovery_write

    reserve_recovery_write(session)
    return inspect_chat_recovery_graph(session, "chat-garden")


def test_history_and_linked_media_are_counted_once_without_physical_reclaim_claims(
    graph_session: Session,
) -> None:
    graph = _graph(graph_session)
    counts = graph.snapshot.counts
    assert counts.messages == 2 and counts.message_parts == 1 and counts.response_revisions == 1
    assert counts.runs == counts.jobs == counts.work_plans == 1
    assert counts.artifacts == 5 and counts.retained_bytes == 150 and counts.reclaimable_bytes == 0
    assert counts.active_work == 0 and graph.snapshot.conflicts == ()
    assert graph.owned_job_ids == ("job-garden",)
    assert graph.artifact_ids == frozenset(_artifact_id(index) for index in range(1, 6))
    assert "Garden notes" not in repr(graph) and "pictures/1.png" not in repr(graph)


@pytest.mark.parametrize(
    ("status", "active_work"),
    [("queued", 1), ("running", 1), ("paused", 1), ("interrupted", 0), ("unknown", 1)],
)
def test_runless_source_verification_blocks_recovery_until_it_is_terminal(
    graph_session: Session, status: str, active_work: int
) -> None:
    graph_session.add(
        Job(
            id="verification-garden",
            kind="edit_verify",
            status=status,
            payload_json={"source_run_id": "run-garden"},
        )
    )
    graph_session.commit()
    graph = _graph(graph_session)
    assert graph.snapshot.counts.jobs == 2 and graph.snapshot.counts.active_work == active_work
    assert graph.snapshot.conflicts == ((RecoveryConflict.ACTIVE_WORK,) if active_work else ())
    assert set(graph.owned_job_ids) == {"job-garden", "verification-garden"}


def test_a_source_link_does_not_transfer_another_conversations_job(graph_session: Session) -> None:
    graph_session.add(
        Job(
            id="job-other",
            run_id="run-other",
            status="complete",
            payload_json={"chat_id": "chat-other", "source_run_id": "run-garden"},
        )
    )
    graph_session.commit()
    graph = _graph(graph_session)
    assert graph.snapshot.counts.jobs == 2
    assert graph.owned_job_ids == ("job-garden",)


def test_a_historical_reference_survives_the_original_asset_being_removed(
    graph_session: Session,
) -> None:
    graph_session.add(
        MessageReference(
            message_id="assistant-garden",
            reference_subject_id="subject-garden",
            mention_slug="garden",
            subject_name="Garden bed",
            subject_kind="place",
            reference_asset_ids_json=["asset-garden-removed"],
            artifact_ids_json=[_artifact_id(5)],
        )
    )
    graph_session.commit()
    graph = _graph(graph_session)
    assert graph.snapshot.counts.references == 1
    assert graph.snapshot.counts.artifacts == 5
    assert _artifact_id(5) in graph.artifact_ids


def test_an_owned_job_keeps_its_identity_after_its_historical_source_run_is_removed(
    graph_session: Session,
) -> None:
    graph_session.add(
        Job(
            id="job-garden-history",
            run_id="run-garden",
            status="complete",
            payload_json={"chat_id": "chat-garden", "source_run_id": "run-source-removed"},
        )
    )
    graph_session.commit()
    graph = _graph(graph_session)
    assert graph.snapshot.counts.jobs == 2
    assert set(graph.owned_job_ids) == {"job-garden", "job-garden-history"}


def test_a_live_claim_keeps_even_a_terminal_job_active(graph_session: Session) -> None:
    graph_session.execute(
        update(Job).where(Job.id == "job-garden").values(claim_owner="active-garden-worker")
    )
    graph_session.commit()
    assert _graph(graph_session).snapshot.counts.active_work == 1


def test_an_accepted_turn_claim_blocks_deletion_before_any_run_exists(
    graph_session: Session,
) -> None:
    graph_session.add(
        TurnCreationClaim(
            chat_id="chat-garden", idempotency_key="garden-turn", owner_token="garden-claim"
        )
    )
    graph_session.commit()
    assert _graph(graph_session).snapshot.counts.active_work == 1


@pytest.mark.parametrize("changed", ["composer", "library", "fork"])
def test_independent_content_and_location_changes_change_the_private_revision(
    graph_session: Session, changed: str
) -> None:
    before = _graph(graph_session)
    graph_session.commit()
    if changed == "composer":
        graph_session.execute(
            update(ChatComposerDraft).values(text="Keep the garden path narrower")
        )
    elif changed == "library":
        graph_session.execute(
            update(ArtifactLibraryEntry).values(display_name="Garden preview", version=2)
        )
    else:
        graph_session.add(
            Chat(id="chat-fork", title="Garden fork", origin_json={"source_chat_id": "chat-garden"})
        )
    graph_session.commit()
    after = _graph(graph_session)
    assert before.snapshot.counts == after.snapshot.counts
    assert before.snapshot.fingerprint != after.snapshot.fingerprint


@pytest.mark.parametrize("bound", ["MAX_GRAPH_ROWS", "MAX_GRAPH_BYTES"])
def test_a_graph_that_exceeds_the_bound_refuses_instead_of_reporting_partial_counts(
    graph_session: Session, monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    from local_lm import chat_recovery_graph

    monkeypatch.setattr(chat_recovery_graph, bound, 1)
    with pytest.raises(
        chat_recovery_graph.ChatRecoveryGraphError, match="^chat-recovery-graph-invalid$"
    ):
        _graph(graph_session)
