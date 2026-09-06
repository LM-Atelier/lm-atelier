from __future__ import annotations

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import WorkflowRevision

_GRAPH = {
    "1": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    }
}


@pytest.fixture(autouse=True)
def reviewed_runtime(app, monkeypatch):
    services = app.state.services
    original_statuses = services.processes.statuses
    monkeypatch.setattr(
        services.processes,
        "statuses",
        lambda: [
            worker.model_copy(
                update={"managed": True, "running": True, "state": "ready", "pid": 43210}
            )
            if worker.name == "media"
            else worker
            for worker in original_statuses()
        ],
    )

    async def object_info():
        return {
            "EmptyLatentImage": {
                "python_module": "nodes",
                "input": {
                    "required": {
                        "width": ["INT", {"default": 512}],
                        "height": ["INT", {"default": 512}],
                        "batch_size": ["INT", {"default": 1}],
                    }
                },
                "output": ["LATENT"],
            }
        }

    monkeypatch.setattr(services.engines.media, "object_info", object_info, raising=False)


async def _created(client: AsyncClient) -> tuple[str, str]:
    response = await client.post(
        "/api/workflows",
        json={
            "name": "Constructed editable workflow",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": _GRAPH,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"], response.json()["current_revision_id"]


def _url(workflow: str, revision: str) -> str:
    return f"/api/workflows/{workflow}/revisions/{revision}/review"


async def _approved(client: AsyncClient) -> tuple[str, str, dict]:
    workflow, revision = await _created(client)
    preview = await client.get(_url(workflow, revision))
    assert preview.status_code == 200, preview.text
    snapshot = preview.json()
    assert snapshot["can_approve"] is True
    approved = await client.post(
        _url(workflow, revision),
        json={"action": "approve", "subject_sha256": snapshot["subject_sha256"]},
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["trusted"] is True
    return workflow, revision, snapshot


async def test_preview_does_not_mint_trust(client: AsyncClient) -> None:
    workflow, revision = await _created(client)
    response = await client.get(_url(workflow, revision))
    assert response.status_code == 200, response.text
    assert response.json()["revision_id"] == revision
    assert response.json()["trusted"] is False
    assert len(response.json()["subject_sha256"]) == 64
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None and row.trusted is False


async def test_explicit_review_persists_and_revokes_exact_revision(client: AsyncClient) -> None:
    workflow, revision, snapshot = await _approved(client)
    refreshed = await client.get(_url(workflow, revision))
    assert refreshed.status_code == 200
    assert refreshed.json()["trusted"] is True
    assert refreshed.json()["subject_sha256"] == snapshot["subject_sha256"]
    revoked = await client.post(
        _url(workflow, revision),
        json={"action": "revoke", "subject_sha256": snapshot["subject_sha256"]},
    )
    assert revoked.status_code == 200 and revoked.json()["trusted"] is False
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None and row.trusted is False
        assert row.api_graph_json == _GRAPH


async def test_changed_preview_cannot_approve_revised_bytes(client: AsyncClient) -> None:
    workflow, revision = await _created(client)
    preview = await client.get(_url(workflow, revision))
    assert preview.status_code == 200
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None
        row.api_graph_json = {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 768, "height": 512, "batch_size": 1},
            }
        }
        session.commit()
    response = await client.post(
        _url(workflow, revision),
        json={"action": "approve", "subject_sha256": preview.json()["subject_sha256"]},
    )
    assert response.status_code == 409
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None and not row.trusted


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "api_graph_json",
            {
                "1": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {"width": 768, "height": 512, "batch_size": 1},
                }
            },
        ),
        ("input_schema_json", {"type": "object", "properties": {"seed": {"type": "integer"}}}),
        ("dependencies_json", {"minimum_vram_bytes": 12345}),
    ],
)
async def test_review_revalidates_stored_execution_identity(
    client: AsyncClient, field: str, replacement: dict
) -> None:
    workflow, revision, snapshot = await _approved(client)
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None
        setattr(row, field, replacement)
        session.commit()
    response = await client.get(_url(workflow, revision))
    assert response.status_code == 200
    assert response.json()["trusted"] is False
    assert response.json()["subject_sha256"] != snapshot["subject_sha256"]


async def test_review_rejects_a_revision_from_another_definition(client: AsyncClient) -> None:
    workflow, revision = await _created(client)
    other_workflow, _ = await _created(client)
    response = await client.get(_url(workflow, revision))
    assert response.status_code == 200
    wrong = await client.post(
        _url(other_workflow, revision),
        json={"action": "approve", "subject_sha256": response.json()["subject_sha256"]},
    )
    assert wrong.status_code == 404
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None and row.trusted is False


async def test_clone_carries_review_evidence_and_revalidates_it(client: AsyncClient) -> None:
    workflow, revision, _ = await _approved(client)
    response = await client.post(f"/api/workflows/{workflow}/clone", json={})
    assert response.status_code == 201
    clone = response.json()
    target = clone["current_revision_id"]
    preview = await client.get(_url(clone["id"], target))
    assert preview.status_code == 200
    assert preview.json()["state"] == "approved"
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, target)
        assert row is not None
        row.dependencies_json = {"minimum_vram_bytes": 98765}
        session.commit()
    stale = await client.get(_url(clone["id"], target))
    assert stale.status_code == 200 and stale.json()["trusted"] is False


async def test_restore_carries_review_evidence(client: AsyncClient) -> None:
    workflow, revision, _ = await _approved(client)
    created = await client.post(
        f"/api/workflows/{workflow}/revisions",
        json={"api_graph": _GRAPH, "input_schema": {"seed": {"type": "integer"}}},
    )
    assert created.status_code == 201
    restored = await client.post(f"/api/workflows/{workflow}/revisions/{revision}/restore")
    assert restored.status_code == 201
    preview = await client.get(_url(workflow, restored.json()["id"]))
    assert preview.status_code == 200 and preview.json()["state"] == "approved"


async def test_selection_rejects_a_review_whose_revision_changed(client: AsyncClient) -> None:
    from local_lm.domain import Operation
    from local_lm.models import WorkflowDefinition
    from local_lm.workflow_selection import WorkflowFamilySelectionError, _validate_revision

    workflow, revision, _ = await _approved(client)
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        definition = session.get(WorkflowDefinition, workflow)
        assert row is not None and definition is not None and definition.family_id is not None
        row.dependencies_json = {"minimum_vram_bytes": 98765}
        session.commit()
        with pytest.raises(WorkflowFamilySelectionError):
            _validate_revision(
                session,
                row,
                capability="image",
                operation=Operation.TEXT_TO_IMAGE,
                engine="comfyui",
                workflow_family_id=definition.family_id,
                required_capabilities=frozenset(),
            )


async def test_downgrade_drops_review_authority_without_retaining_trusted_cache(
    client: AsyncClient, settings
) -> None:
    from alembic import command
    from sqlalchemy import text

    from local_lm import db
    from local_lm.database_migrations import alembic_config

    workflow, revision, _ = await _approved(client)
    config = alembic_config(settings)
    command.downgrade(config, "a2c7f9d31b60")
    with db.engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT trusted FROM workflow_revisions WHERE id=:id"), {"id": revision}
            ).scalar()
            == 0
        )
    command.upgrade(config, "head")
    preview = await client.get(_url(workflow, revision))
    assert preview.status_code == 200 and preview.json()["trusted"] is False


@pytest.mark.parametrize("changed", ["definition", "node_contract", "worker"])
async def test_execution_rechecks_changes_during_package_verification(
    client: AsyncClient, app, settings, monkeypatch, changed: str
) -> None:
    from local_lm import workflow_review_runtime
    from local_lm.models import WorkflowDefinition
    from local_lm.workflow_revision_reviews import WorkflowReviewError

    workflow, revision, _ = await _approved(client)
    services = app.state.services

    async def verify_packages(*args, **kwargs):
        if changed == "definition":
            with SessionLocal() as writer:
                definition = writer.get(WorkflowDefinition, workflow)
                assert definition is not None
                definition.operation = "image_to_image"
                writer.commit()
        elif changed == "node_contract":
            original_info = services.engines.media.object_info

            async def changed_info():
                info = await original_info()
                info["EmptyLatentImage"]["output"] = ["IMAGE"]
                return info

            monkeypatch.setattr(services.engines.media, "object_info", changed_info)
        else:
            statuses = services.processes.statuses
            monkeypatch.setattr(
                services.processes,
                "statuses",
                lambda: [
                    worker.model_copy(update={"pid": 43211}) if worker.name == "media" else worker
                    for worker in statuses()
                ],
            )

    monkeypatch.setattr(workflow_review_runtime, "verify_reviewed_packages", verify_packages)
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None
        with pytest.raises(WorkflowReviewError):
            await workflow_review_runtime.verify_workflow_review_runtime(
                settings, services.processes, services.engines.media, session, row
            )


async def test_execution_accepts_current_reviewed_runtime(
    client: AsyncClient, app, settings
) -> None:
    from local_lm.workflow_review_runtime import verify_workflow_review_runtime

    _, revision, _ = await _approved(client)
    services = app.state.services
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None
        await verify_workflow_review_runtime(
            settings, services.processes, services.engines.media, session, row
        )


@pytest.mark.parametrize(
    "graph",
    [
        {"bad id": {"class_type": "EmptyLatentImage", "inputs": {}}},
        {"1": {"class_type": "EmptyLatentImage", "inputs": []}},
        {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": ["missing", 0], "height": 512, "batch_size": 1},
            }
        },
        {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": ["2", 1], "height": 512, "batch_size": 1},
            },
            "2": _GRAPH["1"],
        },
        {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": ["1", 0], "height": 512, "batch_size": 1},
            }
        },
        {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 512}}},
        {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": "${undeclared}", "height": 512, "batch_size": 1},
            }
        },
        {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": "/outside/model.bin", "height": 512, "batch_size": 1},
            }
        },
        {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": "../outside.bin", "height": 512, "batch_size": 1},
            }
        },
        {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": "https://example.com/model.bin",
                    "height": 512,
                    "batch_size": 1,
                },
            }
        },
    ],
    ids=[
        "node-id",
        "input-container",
        "missing-link",
        "output-index",
        "cycle",
        "required-input",
        "undeclared-placeholder",
        "absolute-path",
        "parent-path",
        "remote-url",
    ],
)
async def test_review_refuses_invalid_graph_before_approval(
    client: AsyncClient, graph: dict
) -> None:
    workflow, revision = await _created(client)
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None
        row.api_graph_json = graph
        session.commit()
    response = await client.get(_url(workflow, revision))
    assert response.status_code == 422
    assert response.json()["code"] == "workflow-review-invalid"
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None and row.trusted is False


async def test_approved_review_can_be_revoked_while_runtime_is_offline(
    client: AsyncClient, app, monkeypatch
) -> None:
    workflow, revision, _ = await _approved(client)
    services = app.state.services
    statuses = services.processes.statuses
    monkeypatch.setattr(
        services.processes,
        "statuses",
        lambda: [
            worker.model_copy(update={"running": False, "state": "stopped", "pid": None})
            if worker.name == "media"
            else worker
            for worker in statuses()
        ],
    )
    preview = await client.get(_url(workflow, revision))
    assert preview.status_code == 200
    snapshot = preview.json()
    assert snapshot["state"] == "approved" and snapshot["can_approve"] is False
    refused = await client.post(
        _url(workflow, revision),
        json={"action": "approve", "subject_sha256": snapshot["subject_sha256"]},
    )
    assert refused.status_code == 409
    revoked = await client.post(
        _url(workflow, revision),
        json={"action": "revoke", "subject_sha256": snapshot["subject_sha256"]},
    )
    assert revoked.status_code == 200
    assert revoked.json()["state"] == "revoked" and revoked.json()["trusted"] is False


async def test_runtime_contract_refresh_preserves_unchanged_workflow_approval(
    client: AsyncClient, app, settings, monkeypatch
) -> None:
    from local_lm.workflow_review_runtime import verify_workflow_review_runtime

    workflow, revision, approved = await _approved(client)
    services = app.state.services
    original_info = services.engines.media.object_info

    async def refreshed_info():
        info = await original_info()
        info["EmptyLatentImage"]["input"]["optional"] = {
            "new_optional_setting": ["INT", {"default": 0}]
        }
        return info

    monkeypatch.setattr(services.engines.media, "object_info", refreshed_info)
    preview = await client.get(_url(workflow, revision))
    assert preview.status_code == 200
    assert preview.json()["trusted"] is True
    assert preview.json()["subject_sha256"] == approved["subject_sha256"]
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None
        await verify_workflow_review_runtime(
            settings, services.processes, services.engines.media, session, row
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("api_graph_json", {str(index): _GRAPH["1"] for index in range(4097)}),
        (
            "api_graph_json",
            {
                "1": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {"width": "x" * 65537, "height": 512, "batch_size": 1},
                }
            },
        ),
        ("input_schema_json", {"properties": {f"slot{index}": {} for index in range(257)}}),
        ("dependencies_json", {"models": [{} for _ in range(257)]}),
        ("dependencies_json", {"custom_nodes": [{} for _ in range(65)]}),
        ("dependencies_json", {"registry_packages": [{} for _ in range(65)]}),
    ],
    ids=[
        "nodes",
        "literal-length",
        "schema-properties",
        "models",
        "git-packages",
        "registry-packages",
    ],
)
async def test_review_enforces_declared_structure_limits(
    client: AsyncClient, field: str, replacement: dict
) -> None:
    workflow, revision = await _created(client)
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None
        setattr(row, field, replacement)
        session.commit()
    response = await client.get(_url(workflow, revision))
    assert response.status_code == 422
    assert response.json()["code"] == "workflow-review-invalid"
    with SessionLocal() as session:
        row = session.get(WorkflowRevision, revision)
        assert row is not None and row.trusted is False


@pytest.mark.parametrize("value", ["x" * 1000, "\u00e9\n" * 500], ids=["ascii", "escaped-unicode"])
def test_canonical_encoding_stops_at_byte_budget(value: str) -> None:
    import tracemalloc

    from local_lm.workflow_revision_reviews import WorkflowReviewError, _canonical

    # The input shares one short string. Encoding it in full would allocate
    # megabytes even though the caller accepts only sixteen KiB.
    payload = [value] * 1000
    tracemalloc.start()
    try:
        with pytest.raises(WorkflowReviewError):
            _canonical(payload, max_bytes=16 * 1024)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 256 * 1024
