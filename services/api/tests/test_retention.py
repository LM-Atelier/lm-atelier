from __future__ import annotations

from httpx2 import AsyncClient


async def test_repeated_cleanup_distinguishes_newly_marked_from_total_pending(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("orphan.bin", b"recoverable", "application/octet-stream")},
    )
    assert uploaded.status_code == 201

    first = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert first.status_code == 200
    assert first.json()["marked_count"] == 1
    assert first.json()["retention_pending_count"] == 1
    assert first.json()["removed_count"] == 0

    repeated = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert repeated.status_code == 200
    assert repeated.json()["marked_count"] == 0
    assert repeated.json()["retention_pending_count"] == 1
    assert repeated.json()["removed_count"] == 0

    storage = await client.get("/api/artifacts/storage")
    assert storage.status_code == 200
    assert storage.json()["retention_pending_count"] == 1
    assert storage.json()["eligible_count"] == 0
