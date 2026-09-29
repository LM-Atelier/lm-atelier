"""Real media dispatch with a constructed managed runtime and retained pixels."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from test_accepted_source_fit import make_run, save
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_geometry import composited_graph, object_info

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.db import SessionLocal
from local_lm.models import (
    Artifact,
    Job,
    WorkflowDefinition,
    WorkflowRevision,
    WorkflowRevisionReview,
)
from local_lm.scheduler import JobClaim
from local_lm.source_fit_image import replay_source_fit_image
from local_lm.workflow_revision_reviews import build_review_snapshot, record_review


@pytest.mark.parametrize(
    "later_change", ["none", "original_bytes", "prepared_bytes", "review_revoked"]
)
async def test_dispatch_uses_retained_source_fit_after_actual_context_and_review_checks(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    later_change: str,
) -> None:
    services = app.state.services
    orchestrator = services.orchestrator
    seen: list[MediaRequest] = []

    async def describe() -> dict[str, Any]:
        return object_info()

    async def generate(request: MediaRequest) -> AsyncIterator[MediaEvent]:
        seen.append(request)
        yield MediaEvent(type="cancelled")

    monkeypatch.setattr(services.engines.settings, "media_engine", "comfyui")
    monkeypatch.setattr(services.engines.media, "object_info", describe, raising=False)
    monkeypatch.setattr(services.engines.media, "generate", generate)
    monkeypatch.setattr(orchestrator, "_ensure_media_worker", AsyncMock())

    async with services.scheduler.lease("primary"):
        with SessionLocal() as session:
            run, recipe = make_run((orchestrator.artifacts, session))
            revision = session.get(WorkflowRevision, run.workflow_revision_id)
            assert revision is not None
            # Every compiler input is declared; the margins are written into the
            # padding step. No real stored workflow or runtime is read by this fixture.
            revision.input_schema_json = {
                "type": "object",
                "properties": {
                    "input_image": {"type": "string"},
                    "checkpoint": {"type": "string", "default": "fixture.safetensors"},
                },
            }
            definition = session.get(WorkflowDefinition, revision.workflow_id)
            assert definition is not None
            record_review(
                session,
                revision,
                build_review_snapshot(session, definition, revision, object_info=object_info()),
                approved=True,
            )
            save(session, run, recipe)
            expected = replay_source_fit_image(
                session,
                orchestrator.artifacts,
                recipe.image,
                selected_source_id=recipe.image.source_artifact_id,
            ).content
            claim = JobClaim(token="source-fit-dispatch", attempt=1)
            job = Job(
                kind="image",
                status="running",
                run_id=run.id,
                claim_owner=claim.token,
                attempt=claim.attempt,
                queue_group="primary",
                queue_resource="media",
            )
            session.add(job)
            session.flush()
            run_id, job_id = run.id, job.id
            if later_change in {"original_bytes", "prepared_bytes"}:
                artifact = session.get(
                    Artifact,
                    recipe.image.source_artifact_id
                    if later_change == "original_bytes"
                    else recipe.image.prepared_artifact_id,
                )
                assert artifact is not None
                orchestrator.artifacts.resolve(artifact).write_bytes(b"replaced fixture bytes")
            elif later_change == "review_revoked":
                review = session.get(WorkflowRevisionReview, revision.id)
                assert review is not None
                review.state = "revoked"
            session.commit()

        if later_change in {"prepared_bytes", "review_revoked"}:
            with pytest.raises((ValueError, RuntimeError)):
                await orchestrator._execute_media(job_id, run_id, claim)
            assert seen == []
            return
        await orchestrator._execute_media(job_id, run_id, claim)

    assert len(seen) == 1
    request = seen[0]
    assert request.input_contents == (expected,)
    compiled = ComfyUIAdapter._compile(request.workflow, request.parameters)
    assert tuple(compiled["pad"]["inputs"][key] for key in ("left", "top", "right", "bottom")) == (
        1,
        1,
        1,
        1,
    )
    # Only the source padding changed: the accepted margins are written into it.
    bound = composited_graph()
    bound["pad"]["inputs"].update(left=1, top=1, right=1, bottom=1)
    assert request.workflow == bound
