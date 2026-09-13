"""A workflow brought in with a project must be the same executable thing it was.

Moving a project between machines carries its workflows with it. Trust stays
behind, deliberately, so an imported workflow is reviewed again before it runs.
What must NOT stay behind is the workflow's identity: activation binds a
reviewed revision by the digest of what it executes, and a revision that arrives
without one can be reviewed and then never activated, however many times the
review is repeated.

These tests go through the real export, import, review and activation routes,
and compare the imported identity with the one the creation route wrote for the
same content on the machine it came from.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_revision_review import _GRAPH
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition, WorkflowRevision

pytestmark = pytest.mark.asyncio

NAME = "Portable neutral workflow"


async def _exported_project(
    client: AsyncClient,
    dependencies: dict[str, object],
    *,
    operation: str = "text_to_image",
    engine: str = "comfyui",
    api_graph: dict[str, object] = _GRAPH,
    input_schema: dict[str, object] | None = None,
) -> tuple[str, str, bytes]:
    created = await client.post(
        "/api/workflows",
        json={
            "name": NAME,
            "operation": operation,
            "engine": engine,
            "api_graph": api_graph,
            "input_schema": input_schema or {},
            "dependencies": dependencies,
        },
    )
    assert created.status_code == 201, created.text
    workflow_id = created.json()["id"]
    revision_id = created.json()["current_revision_id"]
    pin = "video_workflow_revision_id" if "video" in operation else "image_workflow_revision_id"
    project = await client.post(
        "/api/projects",
        json={"name": "Portable neutral project", pin: revision_id},
    )
    assert project.status_code == 201, project.text
    exported = await client.post(f"/api/projects/{project.json()['id']}/export")
    assert exported.status_code == 201, exported.text
    archive = await client.get(exported.json()["url"])
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        assert any(
            NAME in bundle.read(name).decode("utf-8", "replace") for name in bundle.namelist()
        )
    # Renamed so the import cannot settle for the workflow already on this
    # machine, and has to create the revision the way another machine would.
    renamed = await client.patch(f"/api/workflows/{workflow_id}", json={"name": "Source copy"})
    assert renamed.status_code == 200, renamed.text
    return workflow_id, revision_id, archive.content


async def _imported_revision(client: AsyncClient, archive: bytes) -> WorkflowRevision:
    imported = await client.post(
        "/api/projects/import",
        files={"archive": ("portable.lm-atelier.zip", archive, "application/zip")},
    )
    assert imported.status_code == 201, imported.text
    with SessionLocal() as session:
        definition = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.name == NAME)
        )
        assert definition is not None and definition.current_revision_id
        revision = session.get(WorkflowRevision, definition.current_revision_id)
        assert revision is not None
        session.expunge(revision)
        return revision


def _stored(revision_id: str) -> WorkflowRevision:
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        session.expunge(revision)
        return revision


async def test_an_imported_workflow_keeps_the_identity_its_content_had(
    client: AsyncClient,
) -> None:
    _, source_id, archive = await _exported_project(client, {"version": 1, "slots": []})
    source = _stored(source_id)
    assert source.artifact_sha256 is not None

    imported = await _imported_revision(client, archive)

    assert imported.id != source_id
    assert imported.trusted is False
    assert imported.artifact_sha256 == source.artifact_sha256
    assert imported.dependency_contract_sha256 == source.dependency_contract_sha256


async def test_an_imported_workflow_can_be_reviewed_and_then_activated(
    client: AsyncClient,
) -> None:
    """The chain an imported workflow has to complete before it can run."""

    _, source_id, archive = await _exported_project(client, {"version": 1, "slots": []})
    source = _stored(source_id)
    imported = await _imported_revision(client, archive)
    review_url = f"/api/workflows/{imported.workflow_id}/revisions/{imported.id}/review"
    review = await client.get(review_url)
    assert review.status_code == 200, review.text
    approved = await client.post(
        review_url,
        json={"action": "approve", "subject_sha256": review.json()["subject_sha256"]},
    )
    assert approved.status_code == 200 and approved.json()["trusted"], approved.text

    subject = await client.get(
        f"/api/workflows/{imported.workflow_id}/revisions/{imported.id}/activation"
    )

    assert subject.status_code == 200, subject.text
    # Compared with what creation wrote on the source, not with the imported
    # row the activation subject is itself read from.
    assert subject.json()["workflow_artifact_sha256"] == source.artifact_sha256


async def test_a_legacy_workflow_keeps_its_identity_too(client: AsyncClient) -> None:
    """A workflow declaring no contract is still one executable thing, and says which.

    A different engine, operation, graph and input schema from the other cases,
    and an execution dependency of its own, so an identity built from the wrong
    one of them could not agree with the source by coincidence.
    """

    _, source_id, archive = await _exported_project(
        client,
        {"custom_nodes": [{"name": "portable-node-pack"}]},
        input_schema={
            "type": "object",
            "properties": {"steps": {"type": "integer", "default": 12}},
        },
        operation="text_to_video",
        engine="mock",
        api_graph={"loader": {"class_type": "PortableLoader"}},
    )
    source = _stored(source_id)

    imported = await _imported_revision(client, archive)

    assert imported.dependency_contract_sha256 is None
    assert imported.artifact_sha256 == source.artifact_sha256
