"""A refused preview deletion keeps its bytes while interrupted recovery finishes."""

import asyncio

import pytest
from httpx2 import ASGITransport, AsyncClient
from test_setup_verification import seed_ready_role
from test_startup_recovery_preview import _pin_a_preview

from local_lm import artifact_library, db
from local_lm.artifact_library import ArtifactReferenceDataError
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.domain import ArtifactKind, JobStatus
from local_lm.main import create_app
from local_lm.models import Artifact, Chat, Job, Message, Run, SetupVerification
from local_lm.setup_verification import (
    SETUP_VERIFICATION_SCOPE,
    ingest_synthetic_setup_image,
)


async def test_a_reference_row_limit_keeps_previews_and_allows_restart(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview_id, job_id = _pin_a_preview(settings.data_dir)
    monkeypatch.setattr(artifact_library, "MAX_REFERENCE_ROWS", 0)

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            ready = await client.get("/api/ready")
            assert ready.status_code == 200
        with db.SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.status == JobStatus.INTERRUPTED.value
            run = session.get(Run, job.run_id)
            assert run is not None and run.status == "failed"
            preview = session.get(Artifact, preview_id)
            assert preview is not None
            store: ArtifactStore = app.state.services.artifacts
            assert store.verified_path(preview).read_bytes() == b"interrupted preview bytes"
            with pytest.raises(ArtifactReferenceDataError):
                store.referenced_artifact_ids(session, for_deletion=True)


def test_refused_reference_authority_declines_preview_cleanup_without_deleting(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview_id, _job_id = _pin_a_preview(settings.data_dir)
    store = ArtifactStore(settings)
    with db.SessionLocal() as session:
        job = session.get(Job, _job_id)
        assert job is not None
        run = session.get(Run, job.run_id)
        assert run is not None
        assistant = session.get(Message, run.assistant_message_id)
        assert assistant is not None
        assistant.parts.clear()
        session.commit()
        monkeypatch.setattr(artifact_library, "MAX_REFERENCE_ROWS", 0)

        assert store.delete_temporary_preview(session, preview_id) is False
        session.commit()
        preview = session.get(Artifact, preview_id)
        assert preview is not None
        assert store.verified_path(preview).read_bytes() == b"interrupted preview bytes"
        with pytest.raises(ArtifactReferenceDataError):
            store.referenced_artifact_ids(session, for_deletion=True)


def test_valid_unreferenced_previews_are_still_removed(settings: Settings) -> None:
    store = ArtifactStore(settings)
    with db.SessionLocal() as session:
        preview = store.ingest_bytes(
            session,
            b"unreferenced preview bytes",
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
            metadata={"temporary_preview": True},
        )
        session.commit()
        preview_id = preview.id
        path = store.verified_path(preview)
        assert store.delete_temporary_preview(session, preview_id) is True
        session.commit()
        assert session.get(Artifact, preview_id) is None
        assert not path.exists()


@pytest.mark.parametrize("with_job", [False, True])
async def test_restart_keeps_setup_and_preview_bytes_without_reference_authority(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, with_job: bool
) -> None:
    preview_id, job_id = _pin_a_preview(settings.data_dir)
    install, profile, _workflow = seed_ready_role(settings, "image", operation="text_to_image")
    store = ArtifactStore(settings)
    with db.SessionLocal() as session:
        verification_job_id = None
        verification_run_id = None
        if with_job:
            _setup_preview_id, verification_job_id = _pin_a_preview(settings.data_dir)
            job = session.get(Job, verification_job_id)
            assert job is not None
            run = session.get(Run, job.run_id)
            assert run is not None
            verification_run_id = run.id
            chat = session.get(Chat, run.chat_id)
            assert chat is not None
            chat.scope = SETUP_VERIFICATION_SCOPE
            chat.archived = True
        else:
            chat = Chat(title="Setup verification", archived=True, scope=SETUP_VERIFICATION_SCOPE)
            session.add(chat)
            session.flush()
        chat_id = chat.id
        verification = SetupVerification(
            role="image",
            evidence_key="r" * 64,
            state="running",
            model_install_id=install.id,
            profile_id=profile.id,
            workflow_revision_id="revision_image",
            chat_id=chat_id,
            run_id=verification_run_id,
            job_id=verification_job_id,
        )
        session.add(verification)
        session.flush()
        artifact = ingest_synthetic_setup_image(session, store, verification.id)
        verification.input_artifact_id = artifact.id
        artifact_id, verification_id = artifact.id, verification.id
        expected_bytes = store.verified_path(artifact).read_bytes()
        session.commit()
    monkeypatch.setattr(artifact_library, "MAX_REFERENCE_ROWS", 0)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app.state.retention_sweep, timeout=30)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            assert (await client.get("/api/ready")).status_code == 200
        with db.SessionLocal() as session:
            recovered = session.get(SetupVerification, verification_id)
            assert recovered is not None
            assert recovered.state == "failed" and recovered.failure_code == "application_restarted"
            assert recovered.chat_id is None and recovered.input_artifact_id is None
            assert session.get(Chat, chat_id) is None
            if verification_job_id:
                assert session.get(Job, verification_job_id) is None
            job = session.get(Job, job_id)
            assert job is not None and job.status == JobStatus.INTERRUPTED.value
            preview = session.get(Artifact, preview_id)
            assert preview is not None
            assert store.verified_path(preview).read_bytes() == b"interrupted preview bytes"
            kept = session.get(Artifact, artifact_id)
            assert kept is not None and store.verified_path(kept).read_bytes() == expected_bytes
            with pytest.raises(ArtifactReferenceDataError):
                store.delete_library_artifact(session, kept)
            session.commit()
