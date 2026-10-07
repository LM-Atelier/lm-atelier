"""Exercise permanent deletion through its public confirmation flow."""

from httpx2 import AsyncClient


async def permanently_delete_chat(
    client: AsyncClient, chat_id: str, *, delete_generated_media: bool = False
) -> None:
    preview = await client.get(
        f"/api/chats/{chat_id}/deletion-impact",
        params={"delete_generated_media": delete_generated_media},
    )
    assert preview.status_code == 200, preview.text
    impact = preview.json()
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash",
        json={
            "expected_revision": impact["revision"],
            "impact_sha256": impact["impact_sha256"],
            "operation_key": f"trash-{chat_id}",
            "delete_generated_media": delete_generated_media,
        },
    )
    assert trashed.status_code == 200, trashed.text
    deletion_id = trashed.json()["deletion_id"]
    preview = await client.get(f"/api/recovery-items/{deletion_id}/impact")
    assert preview.status_code == 200, preview.text
    impact = preview.json()
    purged = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={
            "expected_revision": impact["revision"],
            "impact_sha256": impact["impact_sha256"],
            "operation_key": f"purge-{chat_id}",
            "acknowledgement": "permanently-delete",
        },
    )
    assert purged.status_code == 200, purged.text
    assert purged.json()["reclaimed_bytes"] == 0


async def permanently_delete_project(client: AsyncClient, project_id: str) -> None:
    preview = await client.get(f"/api/projects/{project_id}/deletion-impact")
    assert preview.status_code == 200, preview.text
    impact = preview.json()
    trashed = await client.post(
        f"/api/projects/{project_id}/trash",
        json={
            "expected_revision": impact["revision"],
            "impact_sha256": impact["impact_sha256"],
            "operation_key": f"trash-{project_id}",
        },
    )
    assert trashed.status_code == 200, trashed.text
    deletion_id = trashed.json()["deletion_id"]
    preview = await client.get(f"/api/recovery-items/{deletion_id}/impact")
    assert preview.status_code == 200, preview.text
    impact = preview.json()
    purged = await client.post(
        f"/api/recovery-items/{deletion_id}/purge",
        json={
            "expected_revision": impact["revision"],
            "impact_sha256": impact["impact_sha256"],
            "operation_key": f"purge-{project_id}",
            "acknowledgement": "permanently-delete",
        },
    )
    assert purged.status_code == 200, purged.text
