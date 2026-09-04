from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm import artifact_library, artifact_library_schema, exports
from local_lm import artifacts as artifacts_module
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base, SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.main import create_app
from local_lm.models import (
    Artifact,
    Chat,
    Job,
    Message,
    MessageReference,
    Run,
    WorkPlan,
    WorkStep,
)


@pytest.fixture
def sweepable(tmp_path: Path) -> tuple[ArtifactStore, Session]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{settings.data_dir / 'retention.sqlite3'}")
    Base.metadata.create_all(engine)
    store = ArtifactStore(settings)
    with Session(engine) as session:
        yield store, session


def test_the_sweep_walks_the_reference_graph_once_not_once_per_deletion(
    sweepable: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep must pay for the reference graph ONCE, through every path.

    cleanup_retention computes the graph once before its loop. Passing that
    snapshot into _delete_artifact removes one recompute per deletion, but it is
    not sufficient on its own: _delete_artifact flushes for every artifact, and
    the registered before_flush listener independently walks the same graph each
    time. A test that patches only the ArtifactStore wrapper cannot see that,
    because the listener resolves the module-level function in artifact_library
    and each module binds its own reference at import.

    So this counts BOTH bindings. Each module resolves its own global, so
    patching one proves nothing about the other, and counting only the wrapper is
    what made the previous instrument false.
    """

    store, session = sweepable
    for index in range(8):
        store.ingest_bytes(
            session,
            f"disposable {index}".encode(),
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
        )
    session.commit()

    # first pass only marks them unreferenced; nothing is eligible yet
    store.cleanup_retention(session, retention_days=30, temporary_hours=24, dry_run=False)
    session.commit()

    calls = 0
    real = artifact_library.referenced_artifact_ids

    def counted(*args: Any, **kwargs: Any) -> set[str]:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(artifact_library, "referenced_artifact_ids", counted)
    monkeypatch.setattr(artifacts_module, "referenced_artifact_ids", counted)

    summary = store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)
    session.commit()

    assert summary.removed_count == 8, "the sweep must actually delete, or nothing is measured"
    assert calls == 1, (
        f"the reference graph was walked {calls} times to delete 8 artifacts; it must "
        f"be walked once. A count near 9 means the flush listener is still walking it "
        f"per deletion behind a snapshot that only the wrapper honours."
    )


async def test_repeated_cleanup_distinguishes_newly_marked_from_total_pending(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("orphan.bin", b"recoverable", "application/octet-stream")},
    )
    assert uploaded.status_code == 201

    first = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert first.status_code == 200
    assert first.json()["marked_count"] == 1
    assert first.json()["retention_pending_count"] == 1
    assert first.json()["removed_count"] == 0

    repeated = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert repeated.status_code == 200
    assert repeated.json()["marked_count"] == 0
    assert repeated.json()["retention_pending_count"] == 1
    assert repeated.json()["removed_count"] == 0

    storage = await client.get("/api/artifacts/storage")
    assert storage.status_code == 200
    assert storage.json()["retention_pending_count"] == 1
    assert storage.json()["eligible_count"] == 0


def test_a_poster_bearing_artifact_does_not_buy_a_second_graph_walk(
    sweepable: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep's own final flush must lend the snapshot too.

    Marking `unreferenced_at` makes an artifact dirty. If its metadata names a
    poster then `_pending_json_reference_ids` is non-empty at the sweep's final
    flush, so the listener fires there as well as on the deletion flushes. That
    is a constant extra walk rather than the per-deletion N+1 that was rejected,
    but "the graph is computed once per sweep" is not true while it happens, and
    a claim that is nearly true is the kind that gets believed.
    """

    store, session = sweepable
    poster = store.ingest_bytes(
        session,
        b"poster target",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
    )
    artifact = store.ingest_bytes(
        session,
        b"poster bearing",
        kind=ArtifactKind.VIDEO,
        media_type="video/mp4",
    )
    # The poster must EXIST. Naming an absent one is a genuinely dangling
    # reference and the listener is right to refuse it, which is a different
    # test from this one.
    artifact.metadata_json = {**artifact.metadata_json, "poster_artifact_id": poster.id}
    session.commit()

    calls = 0
    real = artifact_library.referenced_artifact_ids

    def counted(*args: Any, **kwargs: Any) -> set[str]:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(artifact_library, "referenced_artifact_ids", counted)
    monkeypatch.setattr(artifacts_module, "referenced_artifact_ids", counted)

    store.cleanup_retention(session, retention_days=30, temporary_hours=24, dry_run=False)
    session.commit()

    assert calls == 1, (
        f"one sweep walked the reference graph {calls} times. A count of 2 means the "
        f"sweep's own final flush is outside the lend, so a dirty poster-bearing "
        f"artifact buys an extra whole-graph walk."
    )


def _aged_video_and_poster(store: ArtifactStore, session: Session) -> tuple[str, str]:
    """A video naming a poster, both sweep-eligible, poster ordered first.

    created_at is pinned rather than left to ingest order: the sweep walks in
    created_at order, and the defect only appears when the poster is reached
    before the artifact naming it. A tie would let the test pass by luck.
    """

    poster = store.ingest_bytes(
        session, b"poster bytes", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    video = store.ingest_bytes(
        session, b"video bytes", kind=ArtifactKind.VIDEO, media_type="video/mp4"
    )
    session.flush()
    poster.created_at = datetime(2020, 1, 1)
    video.created_at = datetime(2020, 1, 2)
    poster.metadata_json = {**dict(poster.metadata_json), "temporary_preview": True}
    video.metadata_json = {
        **dict(video.metadata_json),
        "temporary_preview": True,
        "poster_artifact_id": poster.id,
    }
    session.commit()
    return poster.id, video.id


def test_the_sweep_survives_a_poster_reached_before_the_artifact_naming_it(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """The sweep must skip a poster it cannot yet delete, not abort the pass.

    referenced_artifact_ids propagates artifact-metadata links only from
    artifacts that are themselves retained, so a poster whose only referrer is
    also garbage reads as unreferenced. The database disagrees: the delete
    trigger refuses while ANY surviving artifact names it. Deleting the poster
    first therefore raised IntegrityError and took the whole sweep down,
    including the artifacts it had not reached yet.
    """

    store, session = sweepable
    poster_id, video_id = _aged_video_and_poster(store, session)

    summary = store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)

    assert session.get(Artifact, video_id) is None, "the referrer was not collected"
    assert session.get(Artifact, poster_id) is not None, (
        "the poster was deleted while an artifact still named it"
    )
    assert summary.removed_count == 1


def test_a_skipped_poster_is_collected_once_nothing_names_it(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """Skipping defers collection by one pass; it must not abandon the bytes."""

    store, session = sweepable
    poster_id, _video_id = _aged_video_and_poster(store, session)

    store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)
    store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)

    assert session.get(Artifact, poster_id) is None, (
        "the poster survived a pass in which nothing named it"
    )


def test_the_preview_does_not_promise_a_removal_the_database_refuses(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """A dry run reports what a real run would do, or it is not a preview."""

    store, session = sweepable
    _aged_video_and_poster(store, session)

    preview = store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=True)
    real = store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)

    assert preview.removed_count == real.removed_count
    assert preview.reclaimed_bytes == real.reclaimed_bytes


def test_one_key_tuple_describes_the_artifact_metadata_edge(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """The trigger, the walk and linked deletion must not drift apart.

    These keys decide which artifacts retain which other artifacts, and four
    sites depend on them: the reference walk, the SQL the delete trigger is
    generated from, library deletion's linked-artifact set, and export
    bundling. Equal literals in separate places can drift apart while still
    comparing equal, so this asserts the sites hold one object.
    """

    keys = artifact_library_schema.ARTIFACT_METADATA_REFERENCE_KEYS
    assert artifact_library.ARTIFACT_METADATA_REFERENCE_KEYS is keys
    assert artifacts_module.ARTIFACT_METADATA_REFERENCE_KEYS is keys
    assert exports.ARTIFACT_METADATA_REFERENCE_KEYS is keys
    for key in keys:
        assert f"'$.{key}'" in artifact_library_schema._reference_values("artifacts", "OLD")


def _temporary(artifact: Artifact, **extra: object) -> None:
    artifact.metadata_json = {
        **dict(artifact.metadata_json),
        "temporary_preview": True,
        **extra,
    }


def test_a_chain_of_named_artifacts_drains_one_link_per_pass(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """Skipping defers by a pass, so a chain must still drain completely.

    Deferring is only acceptable if it terminates. Each pass removes whatever
    nothing names any more, which frees the next link, so a chain of length
    three needs three passes and no pass may abort.
    """

    store, session = sweepable
    made = [
        store.ingest_bytes(
            session, f"chain {index}".encode(), kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        for index in range(3)
    ]
    session.flush()
    deepest, middle, head = made
    for offset, artifact in enumerate((deepest, middle, head)):
        artifact.created_at = datetime(2020, 1, 1 + offset)
    _temporary(deepest)
    _temporary(middle, browser_proxy_artifact_id=deepest.id)
    _temporary(head, poster_artifact_id=middle.id)
    session.commit()
    ids = [deepest.id, middle.id, head.id]

    survivors = []
    for _pass in range(3):
        store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)
        survivors.append(sum(1 for value in ids if session.get(Artifact, value) is not None))

    assert survivors == [2, 1, 0], (
        f"a chain of three drained as {survivors}; expected one link per pass"
    )


def test_a_metadata_cycle_is_skipped_rather_than_aborting_the_sweep(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """Two artifacts naming each other are unreclaimable. That must not abort.

    The delete trigger refuses both members of a cycle whatever the order, so
    no sweep can collect them and neither can explicit deletion. This records
    the resulting limit deliberately: the bytes are not reclaimed, but the pass
    completes and everything else in it is still collected.
    """

    store, session = sweepable
    left = store.ingest_bytes(
        session, b"left bytes", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    right = store.ingest_bytes(
        session, b"right bytes", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    unrelated = store.ingest_bytes(
        session, b"unrelated bytes", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    session.flush()
    _temporary(left)
    session.flush()
    _temporary(right, poster_artifact_id=left.id)
    session.flush()
    _temporary(left, poster_artifact_id=right.id)
    _temporary(unrelated)
    session.commit()
    cycle = [left.id, right.id]
    other = unrelated.id

    summary = store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)

    assert session.get(Artifact, other) is None, "the cycle blocked an unrelated artifact"
    assert all(session.get(Artifact, value) is not None for value in cycle)
    assert summary.removed_count == 1


async def test_startup_completes_when_a_video_and_its_poster_have_both_aged_out(
    settings: Settings,
) -> None:
    """The sweep runs inside lifespan, so aborting it stops the application.

    main.py calls cleanup_retention in the artifact-retention-cleanup startup
    stage with dry_run=False, and _startup_stage wraps its body in try/finally
    with no except clause. An IntegrityError from the sweep therefore
    propagates out of lifespan and the app never finishes starting. This drives
    the real create_app lifespan rather than the sweep directly.

    Nothing here is dated to the test run: the pair is stamped in 2020, so it
    ages out under the shipped retention settings rather than settings the test
    chose to make it eligible.
    """

    settings.prepare()
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        poster_id, video_id = _aged_video_and_poster(store, session)

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        pass

    with SessionLocal() as session:
        assert session.get(Artifact, video_id) is None, "startup did not sweep the referrer"
        assert session.get(Artifact, poster_id) is not None


def _chat(session: Session) -> Chat:
    chat = Chat(title="holder")
    session.add(chat)
    session.flush()
    return chat


def _message(session: Session) -> Message:
    chat = _chat(session)
    message = Message(chat_id=chat.id)
    session.add(message)
    session.flush()
    return message


def _via_jobs(store: ArtifactStore, session: Session, target: str) -> None:
    session.add(Job(payload_json={"artifact_id": target}))


def _via_chats(store: ArtifactStore, session: Session, target: str) -> None:
    session.add(Chat(scope="studio", origin_json={"source_artifact_id": target}))


def _via_message_references(store: ArtifactStore, session: Session, target: str) -> None:
    message = _message(session)
    session.add(
        MessageReference(
            message_id=message.id,
            reference_subject_id="subj",
            mention_slug="slug",
            subject_name="name",
            subject_kind="kind",
            artifact_ids_json=[target],
        )
    )


def _via_work_steps(store: ArtifactStore, session: Session, target: str) -> None:
    chat = _chat(session)
    plan = WorkPlan(chat_id=chat.id, transcript_sequence=0)
    session.add(plan)
    session.flush()
    session.add(
        WorkStep(
            plan_id=plan.id,
            ordinal=0,
            operation="text",
            input_bindings_json=[{"artifact_id": target}],
        )
    )


def _via_runs(store: ArtifactStore, session: Session, target: str) -> None:
    chat = _chat(session)
    user = Message(chat_id=chat.id)
    assistant = Message(chat_id=chat.id)
    session.add_all([user, assistant])
    session.flush()
    session.add(
        Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            provenance_json={"input_artifact_ids": [target]},
        )
    )


def _via_artifacts(store: ArtifactStore, session: Session, target: str) -> None:
    referrer = store.ingest_bytes(
        session, b"live referrer", kind=ArtifactKind.VIDEO, media_type="video/mp4"
    )
    session.flush()
    referrer.metadata_json = {
        **dict(referrer.metadata_json),
        "poster_artifact_id": target,
    }


RETAINING_EDGES = {
    "jobs": _via_jobs,
    "chats": _via_chats,
    "message_references": _via_message_references,
    "work_steps": _via_work_steps,
    "runs": _via_runs,
    "artifacts": _via_artifacts,
}


@pytest.mark.parametrize("table", sorted(RETAINING_EDGES))
def test_every_guarded_table_makes_the_sweep_skip(
    sweepable: tuple[ArtifactStore, Session], table: str
) -> None:
    """Every retaining edge must make the sweep skip rather than abort.

    For every table the delete trigger guards, an artifact that table still
    names survives the sweep and the pass completes.

    The [artifacts] case is stated apart because the uniform claim would be
    false for it: its referrer is itself unreferenced garbage, so the walk
    answers nothing for it either way. That case is guarded by the sweep's
    referrer map rather than by the reference walk.
    """

    store, session = sweepable
    target = store.ingest_bytes(
        session, f"target for {table}".encode(), kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    session.flush()
    target.created_at = datetime(2020, 1, 1)
    target.metadata_json = {**dict(target.metadata_json), "temporary_preview": True}
    session.flush()
    RETAINING_EDGES[table](store, session, target.id)
    session.commit()
    held = target.id

    summary = store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)

    assert session.get(Artifact, held) is not None, f"{table}: retained artifact was deleted"
    assert summary.removed_count == 0


def test_the_edge_controls_cover_every_guarded_table() -> None:
    """A table added to the guard without a control here fails this test."""

    assert set(RETAINING_EDGES) == set(artifact_library_schema._TABLE_COLUMNS)


def test_a_video_and_its_linked_artifacts_are_collected_in_one_pass(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """The production ingest order must not cost a pass per link.

    Generation ingests the video first, then its browser proxy, then its poster,
    so created_at runs video < proxy < poster and the sweep reaches the referrer
    BEFORE the artifacts it names. Once the video is gone nothing names them any
    more, so deferring them would be a regression against collecting all three
    together - which is what a skip set frozen before the loop does, because it
    still names a poster whose only referrer this pass already removed.
    """

    store, session = sweepable
    video = store.ingest_bytes(
        session, b"video bytes", kind=ArtifactKind.VIDEO, media_type="video/mp4"
    )
    proxy = store.ingest_bytes(
        session, b"proxy bytes", kind=ArtifactKind.VIDEO, media_type="video/webm"
    )
    poster = store.ingest_bytes(
        session, b"poster bytes", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    session.flush()
    for offset, artifact in enumerate((video, proxy, poster)):
        artifact.created_at = datetime(2020, 1, 1 + offset)
        artifact.metadata_json = {**dict(artifact.metadata_json), "temporary_preview": True}
    video.metadata_json = {
        **dict(video.metadata_json),
        "browser_proxy_artifact_id": proxy.id,
        "poster_artifact_id": poster.id,
    }
    session.commit()
    ids = [video.id, proxy.id, poster.id]

    summary = store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)

    survivors = [value for value in ids if session.get(Artifact, value) is not None]
    assert survivors == [], f"one pass left {len(survivors)} of 3 behind"
    assert summary.removed_count == 3


def test_library_deletion_declines_a_poster_another_video_still_names(
    sweepable: tuple[ArtifactStore, Session],
) -> None:
    """Explicit deletion has the same divergence as the sweep did.

    Ingest deduplicates on sha256, so two videos whose extracted frames are
    identical share ONE poster row. Deleting the first video also deletes its
    linked poster, and referenced_artifact_ids does not see the second video
    naming that poster while the second video is itself unreferenced. The delete
    trigger does see it and refuses, which reached the caller as IntegrityError
    rather than as a declined link.
    """

    store, session = sweepable
    first = store.ingest_bytes(
        session, b"first video", kind=ArtifactKind.VIDEO, media_type="video/mp4"
    )
    second = store.ingest_bytes(
        session, b"second video", kind=ArtifactKind.VIDEO, media_type="video/mp4"
    )
    poster = store.ingest_bytes(
        session, b"shared poster", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    session.flush()
    for video in (first, second):
        video.metadata_json = {**dict(video.metadata_json), "poster_artifact_id": poster.id}
    session.commit()
    first_id, poster_id, second_id = first.id, poster.id, second.id

    _references, removed, _bytes = store.delete_library_artifact(session, first)
    session.commit()

    assert session.get(Artifact, first_id) is None, "the requested video was not deleted"
    assert session.get(Artifact, poster_id) is not None, (
        "a poster the second video still names was deleted"
    )
    assert session.get(Artifact, second_id) is not None
    assert removed == 1, f"removed {removed}; the shared poster must not be counted"
