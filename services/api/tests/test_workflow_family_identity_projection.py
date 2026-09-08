from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition


async def test_workflow_responses_keep_the_family_identity_after_archiving(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Neutral grouped workflow",
            "operation": "text_to_image",
            "engine": "comfyui",
            "api_graph": {},
            "input_schema": {},
        },
    )
    assert created.status_code == 201
    workflow_id = created.json()["id"]
    with SessionLocal() as session:
        definition = session.get(WorkflowDefinition, workflow_id)
        assert definition is not None
        family_id = definition.family_id
    assert family_id is not None
    assert created.json()["family_id"] == family_id
    archived = await client.patch(f"/api/workflow-families/{family_id}", json={"archived": True})
    assert archived.status_code == 200
    families = await client.get("/api/workflow-families")
    assert family_id not in {row["id"] for row in families.json()}
    listed = await client.get("/api/workflows")
    retained = next(row for row in listed.json() if row["id"] == workflow_id)
    assert retained["family_id"] == family_id


async def test_a_legacy_ungrouped_definition_is_explicitly_null(client: AsyncClient) -> None:
    with SessionLocal() as session:
        definition = WorkflowDefinition(
            name="Neutral ungrouped definition", operation="text_to_image", description=""
        )
        session.add(definition)
        session.commit()
        workflow_id = definition.id
    listed = await client.get("/api/workflows")
    retained = next(row for row in listed.json() if row["id"] == workflow_id)
    assert "family_id" in retained
    assert retained["family_id"] is None
