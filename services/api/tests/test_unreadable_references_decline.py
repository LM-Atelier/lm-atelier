"""Unreadable stored references are declined where they are met, never crashed on.

A deletion needs a complete, valid reference graph. When stored data cannot
provide one, the operation keeps its bytes and says so: startup recovery keeps
an interrupted preview and the app still starts, a deletion preview answers
with its own conflict, an HTTP route answers with a conflict instead of a
server error, and a setup verification still finishes. Two refusals come from
different checks - the Python walk of the references and the database check of
every stored document - and each has to be declined the same way. Setup cleanup
also keeps what it may not delete instead of raising where nothing can answer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import pytest
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.orm import Session
from test_setup_verification import seed_ready_role, wait_for_role
from test_startup_recovery_preview import _pin_a_preview

from local_lm import db
from local_lm.artifact_deletion_authority import ArtifactDeletionProofError
from local_lm.artifact_library import (
    ArtifactReferenceDataError,
    begin_artifact_write_fence,
    ensure_library_entry,
    referenced_artifact_ids,
)
from local_lm.artifact_library_schema import MAX_JSON_DEPTH
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.domain import ArtifactKind, MessageRole
from local_lm.main import create_app
from local_lm.models import Artifact, Chat, Job, Message, SetupVerification
from local_lm.setup_verification import (
    SETUP_VERIFICATION_SCOPE,
    ingest_synthetic_setup_image,
)

#: Makes the input a running verification names, given its id.
SetupInput = Callable[[Session, ArtifactStore, str], Artifact]


def _nested(levels: int) -> object:
    value: object = None
    for _ in range(levels):
        value = [value]
    return value


def _without_insert_guard(
    session: Session, table: str, statement: str, values: dict[str, object]
) -> None:
    """Store a row the insert guard would refuse, as older or damaged stores hold."""

    guard = f"{table}_artifact_reference_insert_guard"
    created = session.execute(
        text("SELECT sql FROM sqlite_master WHERE name = :name"), {"name": guard}
    ).scalar_one()
    session.execute(text(f"DROP TRIGGER {guard}"))
    session.execute(text(statement), values)
    session.execute(text(created))
    session.commit()


def _plant_a_document_only_the_database_refuses(session: Session) -> str:
    digest = "e" * 64
    _without_insert_guard(
        session,
        "artifacts",
        "INSERT INTO artifacts (id, sha256, kind, media_type, size_bytes, relative_path, "
        "metadata_json, favorite, created_at, updated_at) VALUES "
        "(:id, :sha256, 'image', 'image/png', 1, :path, :metadata, 0, "
        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
        {
            "id": f"sha256:{digest}",
            "sha256": digest,
            "path": f"{digest[:2]}/{digest[2:4]}/{digest}",
            "metadata": json.dumps({"nested": _nested(MAX_JSON_DEPTH)}),
        },
    )
    # The walk reads only reference keys, so it accepts this document; the
    # database check of every stored document is what refuses it.
    assert referenced_artifact_ids(session) is not None
    begin_artifact_write_fence(session)
    with pytest.raises((ArtifactDeletionProofError, ArtifactReferenceDataError)):
        referenced_artifact_ids(session, for_deletion=True)
    session.rollback()
    return f"sha256:{digest}"


@asynccontextmanager
async def _started(settings: Settings) -> AsyncIterator[ArtifactStore]:
    """Start the app as a restart would, and keep it up while the test looks."""

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            assert (await client.get("/api/ready")).status_code == 200
        store: ArtifactStore = app.state.services.artifacts
        yield store


def _running_verification(settings: Settings, setup_input: SetupInput) -> tuple[str, str]:
    """A picture verification a restart interrupted, and the input it names."""

    install, profile, _workflow = seed_ready_role(settings, "image", operation="text_to_image")
    store = ArtifactStore(settings)
    with db.SessionLocal() as session:
        chat = Chat(title="Setup verification", archived=True, scope=SETUP_VERIFICATION_SCOPE)
        session.add(chat)
        session.flush()
        verification = SetupVerification(
            role="image",
            evidence_key="s" * 64,
            state="running",
            model_install_id=install.id,
            profile_id=profile.id,
            workflow_revision_id="revision_image",
            chat_id=chat.id,
        )
        session.add(verification)
        session.flush()
        made = setup_input(session, store, verification.id)
        verification.input_artifact_id = made.id
        identifiers = (verification.id, made.id)
        session.commit()
    return identifiers


def _input_bytes(settings: Settings, input_id: str) -> bytes:
    with db.SessionLocal() as session:
        made = session.get(Artifact, input_id)
        assert made is not None
        return ArtifactStore(settings).verified_path(made).read_bytes()


async def test_startup_keeps_a_preview_when_only_the_database_refuses_a_document(
    settings: Settings,
) -> None:
    preview_id, _job_id = _pin_a_preview(settings.data_dir)
    with db.SessionLocal() as session:
        _plant_a_document_only_the_database_refuses(session)

    async with _started(settings) as store:
        with db.SessionLocal() as session:
            preview = session.get(Artifact, preview_id)
            assert preview is not None
            assert store.verified_path(preview).read_bytes() == b"interrupted preview bytes"


async def test_a_media_deletion_preview_answers_its_conflict_when_a_document_is_refused(
    client: AsyncClient, settings: Settings
) -> None:
    store = ArtifactStore(settings)
    with db.SessionLocal() as session:
        picture = store.ingest_bytes(
            session, b"a kept picture", kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        entry = ensure_library_entry(session, picture)
        assert entry is not None
        entry_id = entry.id
        session.commit()
        _plant_a_document_only_the_database_refuses(session)

    response = await client.get(f"/api/artifact-library/{entry_id}/deletion-impact")

    assert response.status_code == 409
    assert response.json()["code"] == "media-recovery-graph-invalid"


async def test_an_unreadable_stored_reference_is_a_conflict_not_a_server_error(
    client: AsyncClient, settings: Settings
) -> None:
    store = ArtifactStore(settings)
    with db.SessionLocal() as session:
        picture = store.ingest_bytes(
            session, b"a picture to delete", kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        chat = Chat(title="neutral references")
        session.add(chat)
        session.flush()
        message = Message(chat_id=chat.id, role=MessageRole.USER.value, status="complete")
        session.add(message)
        session.commit()
        picture_id = picture.id
        # An object where a list of ids belongs: the walk cannot read it.
        _without_insert_guard(
            session,
            "message_references",
            "INSERT INTO message_references (id, message_id, position, reference_subject_id, "
            "mention_slug, subject_name, subject_kind, source, reference_asset_ids_json, "
            "artifact_ids_json, created_at, updated_at) VALUES (:id, :message_id, 0, "
            "'subject_neutral', 'neutral', 'Neutral subject', 'object', 'mention', '[]', "
            ":ids, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            {"id": "msgref_neutral", "message_id": message.id, "ids": "{}"},
        )

    storage = await client.get("/api/artifacts/storage")
    deletion = await client.delete(f"/api/artifacts/{picture_id}")

    assert storage.status_code == 409
    assert storage.json()["code"] == "artifact-references-unavailable"
    assert deletion.status_code == 409
    assert deletion.json()["code"] == "artifact-references-unavailable"
    with db.SessionLocal() as session:
        kept = session.get(Artifact, picture_id)
        assert kept is not None
        assert store.verified_path(kept).read_bytes() == b"a picture to delete"


async def test_deleting_media_answers_the_same_conflict_when_only_the_database_refuses(
    client: AsyncClient, settings: Settings
) -> None:
    # This refusal is also a ValueError, which the route otherwise reports as
    # the artifact being in use.
    store = ArtifactStore(settings)
    with db.SessionLocal() as session:
        picture = store.ingest_bytes(
            session, b"a picture the check keeps", kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        session.commit()
        picture_id = picture.id
        _plant_a_document_only_the_database_refuses(session)

    deletion = await client.delete(f"/api/artifacts/{picture_id}")

    assert deletion.status_code == 409
    assert deletion.json()["code"] == "artifact-references-unavailable"
    with db.SessionLocal() as session:
        kept = session.get(Artifact, picture_id)
        assert kept is not None
        assert store.verified_path(kept).read_bytes() == b"a picture the check keeps"


def _named_by_another_job(job_id: str) -> SetupInput:
    def make(session: Session, store: ArtifactStore, verification_id: str) -> Artifact:
        made = ingest_synthetic_setup_image(session, store, verification_id)
        job = session.get(Job, job_id)
        assert job is not None
        job.payload_json = {**(job.payload_json or {}), "artifact_ids": [made.id]}
        return made

    return make


def _not_a_picture_or_video(session: Session, store: ArtifactStore, _verification: str) -> Artifact:
    return store.ingest_bytes(
        session,
        b"setup bytes that are not media",
        kind=ArtifactKind.OTHER,
        media_type="application/octet-stream",
    )


def _a_poster_another_video_names(
    session: Session, store: ArtifactStore, verification_id: str
) -> Artifact:
    made = ingest_synthetic_setup_image(session, store, verification_id)
    video = store.ingest_bytes(
        session, b"a video naming the poster", kind=ArtifactKind.VIDEO, media_type="video/mp4"
    )
    video.metadata_json = {**video.metadata_json, "poster_artifact_id": made.id}
    return made


async def test_setup_recovery_keeps_an_input_another_job_names_and_the_app_starts(
    settings: Settings,
) -> None:
    _preview_id, job_id = _pin_a_preview(settings.data_dir)
    verification_id, input_id = _running_verification(settings, _named_by_another_job(job_id))
    expected = _input_bytes(settings, input_id)

    async with _started(settings) as store:
        with db.SessionLocal() as session:
            recovered = session.get(SetupVerification, verification_id)
            assert recovered is not None
            assert recovered.state == "failed"
            assert recovered.failure_code == "application_restarted"
            kept = session.get(Artifact, input_id)
            assert kept is not None
            assert store.verified_path(kept).read_bytes() == expected


@pytest.mark.parametrize(
    "setup_input",
    [_not_a_picture_or_video, _a_poster_another_video_names],
    ids=["not-a-picture-or-video", "a-poster-another-video-names"],
)
async def test_setup_recovery_keeps_an_input_it_may_not_delete(
    settings: Settings, setup_input: SetupInput
) -> None:
    verification_id, input_id = _running_verification(settings, setup_input)
    expected = _input_bytes(settings, input_id)

    async with _started(settings) as store:
        with db.SessionLocal() as session:
            recovered = session.get(SetupVerification, verification_id)
            assert recovered is not None
            assert recovered.state == "failed"
            kept = session.get(Artifact, input_id)
            assert kept is not None
            assert store.verified_path(kept).read_bytes() == expected


async def test_setup_recovery_still_deletes_an_input_nothing_holds(settings: Settings) -> None:
    verification_id, input_id = _running_verification(settings, ingest_synthetic_setup_image)
    with db.SessionLocal() as session:
        made = session.get(Artifact, input_id)
        assert made is not None
        path = ArtifactStore(settings).resolve(made)
    assert path.exists()

    async with _started(settings):
        with db.SessionLocal() as session:
            recovered = session.get(SetupVerification, verification_id)
            assert recovered is not None
            assert recovered.state == "failed"
            assert session.get(Artifact, input_id) is None
    assert not path.exists()


async def test_a_verification_still_finishes_when_references_cannot_be_read(
    client: AsyncClient, settings: Settings
) -> None:
    seed_ready_role(settings, "image", operation="text_to_image")
    with db.SessionLocal() as session:
        planted = _plant_a_document_only_the_database_refuses(session)

    assert (await client.post("/api/setup/verify/image")).status_code == 202
    ready = await wait_for_role(client, "image", "ready")

    assert ready["state"] == "ready"
    with db.SessionLocal() as session:
        verification = session.scalar(
            select(SetupVerification).where(SetupVerification.role == "image")
        )
        assert verification is not None
        assert verification.state == "ready"
        assert verification.chat_id is None
        # Its output cannot be proven unreferenced, so it is kept.
        outputs = session.scalars(
            select(Artifact).where(
                Artifact.kind == ArtifactKind.IMAGE.value, Artifact.id != planted
            )
        ).all()
        assert outputs
