"""By-id editing and prompt drafts remain hidden while their chat is recoverable."""

from httpx2 import AsyncClient
from test_chat_deletion import _text_exchange
from test_prompt_library_api import _create_payload, _wait_for_run
from test_recovery_lifecycle_boundaries import _restore, _trash


async def test_a_recoverable_chat_cannot_activate_an_edited_branch(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Alternate garden notes"})).json()
    source = await _text_exchange(client, chat["id"], "A blue paper boat")
    response = await client.post(
        f"/api/messages/{source['user_message']['id']}/edits",
        json={"text": "A green paper boat", "idempotency_key": "recovery-edit"},
    )
    assert response.status_code == 202, response.text
    edited = response.json()
    await _wait_for_run(client, edited["run"]["id"])
    item = await _trash(client, f"/api/chats/{chat['id']}", "trash-edited-branch")
    payload = {"expected_active_head_message_id": source["assistant_message"]["id"]}
    response = await client.post(
        f"/api/chats/{chat['id']}/edited-branches/{edited['work_plan_id']}/activate",
        json=payload,
    )
    assert response.status_code == 404, response.text
    await _restore(client, item["deletion_id"])
    assert (
        await client.get(f"/api/messages/{source['user_message']['id']}/edit-source")
    ).status_code == 200
    assert (await client.get(f"/api/chats/{chat['id']}/edited-branches")).status_code == 200


async def test_prompt_draft_routes_and_helper_creation_refuse_a_hidden_source(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Shape drafts"})).json()
    template = await client.post(
        "/api/prompt-templates", json=_create_payload(name="Shape variations")
    )
    assert template.status_code == 201, template.text
    revision = template.json()["revision"]
    response = await client.post(
        f"/api/chats/{chat['id']}/prompt-batches",
        json={
            "idempotency_key": "shape-batch",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 1,
            "selection_seed": 7,
            "inputs": {"subject": ["blue circle"]},
        },
    )
    assert response.status_code == 201, response.text
    batch = response.json()
    item = await _trash(client, f"/api/chats/{chat['id']}", "trash-prompt-drafts")
    path = f"/api/prompt-batches/{batch['id']}"
    assert (await client.get(path)).status_code == 404
    patched = await client.patch(
        path + "/items/1",
        json={
            "expected_review_version": batch["items"][0]["review_version"],
            "expected_plan_version": batch["plan_version"],
            "reviewed_prompt": "A green circle",
            "selected": True,
        },
    )
    assert patched.status_code == 404, patched.text
    queued = await client.post(
        path + "/queue",
        json={
            "idempotency_key": "queue-hidden-draft",
            "expected_plan_version": batch["plan_version"],
            "expected_plan_sha256": batch["plan_sha256"],
        },
    )
    assert queued.status_code == 404, queued.text
    helper = await client.post(
        "/api/prompt-helpers", json={"source_chat_id": chat["id"], "draft_prompt": "A blue circle"}
    )
    assert helper.status_code == 404, helper.text
    await _restore(client, item["deletion_id"])
    restored = await client.get(path)
    assert restored.status_code == 200, restored.text
    assert restored.json() == batch | {"replayed": True}
