from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact, Chat, Message, MessagePart, Run
from local_lm.schemas import TurnRequest
from local_lm.vision import PreparedVisualContext, VisualFrame


@pytest.mark.parametrize(
    "later_change", ["replace", "add", "delete", "inherit", "inherit_absent", "export"]
)
async def test_accepted_video_poster_cannot_be_retargeted_or_lost(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, later_change: str
) -> None:
    orchestrator = app.state.services.orchestrator
    project = (await client.post("/api/projects", json={"name": "Accepted poster"})).json()
    chat = (
        await client.post(
            "/api/chats", json={"title": "Accepted video poster", "project_id": project["id"]}
        )
    ).json()
    absent = later_change in {"add", "inherit_absent"}
    seen: list[list[str]] = []

    async def prepare(artifacts: list[Artifact], **kwargs: Any) -> PreparedVisualContext:
        seen.append([artifact.id for artifact in artifacts])
        frames = tuple(
            VisualFrame(
                artifact_id=artifact.id,
                artifact_sha256=artifact.sha256,
                media_type="image/png",
                content=b"constructed-frame",
            )
            for artifact in artifacts
            if artifact.media_type == "image/png"
        )
        return PreparedVisualContext(frames=frames, skipped_artifact_ids=())

    monkeypatch.setattr(orchestrator.vision, "prepare", prepare)
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            posters = [
                orchestrator.artifacts.ingest_bytes(
                    session,
                    payload,
                    kind=ArtifactKind.IMAGE,
                    media_type="image/png",
                    original_name="poster.png",
                )
                for payload in (b"constructed-original-poster", b"constructed-later-poster")
            ]
            original_id, later_id = [poster.id for poster in posters]
            video = orchestrator.artifacts.ingest_bytes(
                session,
                b"constructed-video",
                kind=ArtifactKind.VIDEO,
                media_type="video/mp4",
                original_name="clip.mp4",
                metadata={} if absent else {"poster_artifact_id": original_id},
            )
            source = Message(
                chat_id=chat["id"],
                role="user",
                status="complete",
                parts=[
                    MessagePart(position=0, type="video", artifact_id=video.id, metadata_json={})
                ],
            )
            session.add(source)
            session.commit()
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(
                    text="Describe the earlier clip", mode="text", parent_message_id=source.id
                ),
                use_explicit_parent=True,
                freeze_context=True,
                activate_branch=False,
            )
            run = session.get(Run, accepted.run.id)
            assert run is not None
            video.metadata_json = {"poster_artifact_id": later_id}
            session.commit()
            if later_change.startswith("inherit"):
                loaded = await client.get(f"/api/messages/{run.user_message_id}/edit-source")
                assert loaded.status_code == 200, loaded.text
                edited = await client.post(
                    f"/api/messages/{run.user_message_id}/edits",
                    json={
                        "text": "Describe the earlier clip more briefly",
                        "source_run_id": loaded.json()["source_run_id"],
                        "source_snapshot_sha256": loaded.json()["source_snapshot_sha256"],
                        "idempotency_key": "preserve-accepted-poster",
                    },
                )
                assert edited.status_code == 202, edited.text
                run = session.get(Run, edited.json()["run"]["id"])
                assert run is not None
            elif later_change == "export":
                exported = await client.post(f"/api/projects/{project['id']}/export")
                assert exported.status_code == 201, exported.text
                content = await client.get(f"/api/artifacts/{exported.json()['id']}/content")
                imported = await client.post(
                    "/api/projects/import",
                    files={"archive": ("poster.zip", content.content, "application/zip")},
                )
                assert imported.status_code == 201, imported.text
                imported_chat = session.scalar(
                    select(Chat).where(Chat.project_id == imported.json()["id"])
                )
                assert imported_chat is not None
                run = session.scalar(select(Run).where(Run.chat_id == imported_chat.id))
                assert run is not None
            if later_change == "delete":
                with pytest.raises(IntegrityError):
                    session.execute(delete(Artifact).where(Artifact.id == original_id))
                session.rollback()
            else:
                await orchestrator._attach_visual_context(
                    session, run, [{"role": "user", "content": "Describe the earlier clip"}]
                )
                expected = [[video.id]] if absent else [[video.id], [original_id]]
                assert seen == expected


@pytest.mark.parametrize(
    "later_change", ["metadata", "selected_revision", "absent", "missing", "delete", "export"]
)
async def test_pending_video_uses_the_exact_producer_revision_poster(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, later_change: str
) -> None:
    from local_lm.artifact_library import referenced_artifact_ids
    from local_lm.models import Job, ResponseRevision, ResponseRevisionPart, WorkStep

    orchestrator = app.state.services.orchestrator
    project = (await client.post("/api/projects", json={"name": "Pending poster transfer"})).json()
    chat = (
        await client.post(
            "/api/chats", json={"title": "Pending video poster", "project_id": project["id"]}
        )
    ).json()
    seen: list[list[str]] = []

    async def prepare(artifacts: list[Artifact], **kwargs: Any) -> PreparedVisualContext:
        seen.append([artifact.id for artifact in artifacts])
        return PreparedVisualContext(frames=(), skipped_artifact_ids=())

    monkeypatch.setattr(orchestrator.vision, "prepare", prepare)
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(
                    text="Write a story about a paper boat, then create an image based on it, "
                    "then animate the image into a video, then summarize the video",
                    mode="auto",
                    confirm_media=True,
                ),
                freeze_context=True,
                activate_branch=False,
            )
            first = session.get(Run, accepted.run.id)
            assert first is not None
            steps = list(
                session.scalars(
                    select(WorkStep)
                    .where(WorkStep.plan_id == first.work_plan_id)
                    .order_by(WorkStep.ordinal)
                )
            )
            assert len(steps) == 4
            producer = session.get(Run, steps[2].run_id)
            consumer = session.get(Run, steps[3].run_id)
            assert producer is not None and consumer is not None
            posters = [
                orchestrator.artifacts.ingest_bytes(
                    session,
                    value,
                    kind=ArtifactKind.IMAGE,
                    media_type="image/png",
                    original_name="poster.png",
                )
                for value in (b"accepted-producer-poster", b"unrelated-later-poster")
            ]
            video = orchestrator.artifacts.ingest_bytes(
                session,
                b"constructed-producer-video",
                kind=ArtifactKind.VIDEO,
                media_type="video/mp4",
                original_name="clip.mp4",
                metadata={"poster_artifact_id": posters[1].id},
            )
            original_id = None if later_change == "absent" else posters[0].id
            # Construct the durable producer result; this is not a decoder or
            # media-engine execution test. The consumer uses its real resolver.
            producer.status = "complete"
            steps[2].status = "complete"
            producer_job = session.scalar(select(Job).where(Job.run_id == producer.id))
            assert producer_job is not None
            producer_job.status = "complete"
            producer.provenance_json = {
                **producer.provenance_json,
                "outputs": [{"artifact_id": video.id, "poster_artifact_id": original_id}],
            }
            message = session.get(Message, producer.assistant_message_id)
            assert message is not None
            revision = ResponseRevision(
                run_id=producer.id, message_id=message.id, sequence=1, status="complete"
            )
            session.add(revision)
            revision.parts.append(
                ResponseRevisionPart(
                    position=len(revision.parts),
                    type="video",
                    artifact_id=video.id,
                    metadata_json={"poster_artifact_id": original_id},
                )
            )
            message.status = "complete"
            if later_change == "selected_revision":
                other = ResponseRevision(
                    message_id=message.id,
                    sequence=revision.sequence + 1,
                    status="complete",
                    parts=[
                        ResponseRevisionPart(
                            position=0,
                            type="video",
                            artifact_id=video.id,
                            metadata_json={"poster_artifact_id": posters[1].id},
                        )
                    ],
                )
                session.add(other)
                session.flush()
                orchestrator.select_response_revision(session, message.id, other.id)
            session.commit()
            if later_change == "delete":
                with pytest.raises(IntegrityError, match="artifact is retained by JSON reference"):
                    session.execute(delete(Artifact).where(Artifact.id == original_id))
                session.rollback()
                return
            if later_change == "export":
                from local_lm.accepted_turn_context import accepted_context

                exported = await client.post(f"/api/projects/{project['id']}/export")
                assert exported.status_code == 201, exported.text
                archive = await client.get(f"/api/artifacts/{exported.json()['id']}/content")
                imported = await client.post(
                    "/api/projects/import",
                    files={"archive": ("pending-poster.zip", archive.content, "application/zip")},
                )
                assert imported.status_code == 201, imported.text
                imported_chat = session.scalar(
                    select(Chat).where(Chat.project_id == imported.json()["id"])
                )
                assert imported_chat is not None
                imported_consumer = session.scalar(
                    select(Run)
                    .join(WorkStep, WorkStep.run_id == Run.id)
                    .where(Run.chat_id == imported_chat.id, WorkStep.ordinal == steps[3].ordinal)
                )
                assert imported_consumer is not None and imported_consumer.id != consumer.id
                consumer = imported_consumer
                snapshot = accepted_context(session, consumer)
                assert snapshot is not None and len(snapshot.dependencies) == 1
                dependency = snapshot.dependencies[0]
                assert dependency.run_id != producer.id
                imported_producer = session.get(Run, dependency.run_id)
                assert (
                    imported_producer is not None and imported_producer.chat_id == consumer.chat_id
                )
            if later_change == "missing":
                revision.parts[-1].metadata_json = {"poster_artifact_id": "missing-poster"}
                session.commit()
                with pytest.raises(RuntimeError, match="Accepted dependency poster is unavailable"):
                    orchestrator._resolve_step_inputs(session, consumer)
                return
            orchestrator._resolve_step_inputs(session, consumer)
            await orchestrator._attach_visual_context(
                session, consumer, [{"role": "user", "content": "Summarize the video"}]
            )
            assert seen == ([[video.id]] if original_id is None else [[video.id], [original_id]])
            if original_id is not None:
                assert original_id in referenced_artifact_ids(session)
