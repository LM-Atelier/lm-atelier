"""The application starts even when an interrupted job's preview cannot be deleted.

The store's refusals are covered directly in test_artifact_safety.py. This drives
the whole path instead: a real interrupted job, a real preview the store will
refuse to delete, and the real FastAPI lifespan - because the reason that refusal
matters is not that it is a refusal, it is that `recover_interrupted` runs inside
`lifespan` at main.py:669 under `_startup_stage`, which wraps its body in
try/finally with no except. Anything raised there propagates out of lifespan and
the application does not open its port.

Nothing here needs the network, an engine, or any media: the preview is a few
bytes ingested through the store, and the artifact that pins it is another.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_lm import db
from local_lm.config import Settings
from local_lm.domain import ArtifactKind, JobKind, JobStatus, MessageRole, PartType, RunStatus
from local_lm.main import create_app
from local_lm.models import Artifact, Chat, Job, Message, MessagePart, Run


def _pin_a_preview(store_root: Path) -> tuple[str, str]:
    """Seed one interrupted job whose preview the store must refuse to delete.

    The refusal is the same one the store already declines rather than raises:
    a surviving artifact's metadata names the preview, so the delete trigger and
    the deletion proof both refuse it, while the reference walk does not answer
    for it because the naming artifact is itself unreferenced.

    Returns the preview's id and the interrupted job's id.
    """

    from local_lm.artifacts import ArtifactStore

    settings = Settings(data_dir=store_root, dev=True, chat_engine="mock", media_engine="mock")
    store = ArtifactStore(settings)

    with db.SessionLocal() as session:
        preview = store.ingest_bytes(
            session,
            b"interrupted preview bytes",
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
            metadata={"temporary_preview": True},
        )
        namer = store.ingest_bytes(
            session,
            b"an unreferenced video that names the preview",
            kind=ArtifactKind.VIDEO,
            media_type="video/mp4",
        )
        namer.metadata_json = {**namer.metadata_json, "browser_proxy_artifact_id": preview.id}

        chat = Chat(title="interrupted by a restart")
        session.add(chat)
        session.flush()

        user = Message(chat_id=chat.id, role=MessageRole.USER.value, status="complete")
        assistant = Message(chat_id=chat.id, role=MessageRole.ASSISTANT.value, status="pending")
        session.add_all([user, assistant])
        session.flush()

        # The part `_temporary_preview_ids` looks for: an artifact id carrying a
        # `preview` marker on the assistant message.
        session.add(
            MessagePart(
                message_id=assistant.id,
                position=0,
                type=PartType.IMAGE.value,
                artifact_id=preview.id,
                metadata_json={"preview": True},
            )
        )

        run = Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            operation="text_to_image",
            status=RunStatus.RUNNING.value,
        )
        session.add(run)
        session.flush()

        job = Job(
            kind=JobKind.IMAGE.value,
            status=JobStatus.RUNNING.value,
            run_id=run.id,
        )
        session.add(job)
        session.flush()
        preview_id, job_id = preview.id, job.id
        session.commit()

    return preview_id, job_id


@pytest.mark.anyio
async def test_the_application_starts_when_an_interrupted_preview_cannot_be_deleted(
    settings: Settings,
) -> None:
    """The startup recovery meets a preview it may not delete, and starts anyway."""

    preview_id, job_id = _pin_a_preview(settings.data_dir)

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        # Reaching here at all is the assertion: a raise inside
        # `orchestrator-recovery` would have propagated out of lifespan.
        with db.SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            assert job.status == JobStatus.INTERRUPTED.value, (
                "recovery did not reach the interrupted job it was meant to settle"
            )
            preview = session.get(Artifact, preview_id)
            assert preview is not None, "the preview was deleted even though the store refuses it"
