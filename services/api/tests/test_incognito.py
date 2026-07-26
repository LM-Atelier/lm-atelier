from __future__ import annotations

import asyncio
import hashlib

from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select

from local_lm.adapters.base import ChatEvent
from local_lm.db import SessionLocal
from local_lm.incognito import INCOGNITO_HEADER
from local_lm.models import Artifact, Chat, Message, WorkPlan


async def _wait_for_terminal_run(client: AsyncClient, run_id: str) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            assert run["status"] == "complete", run
            return run
        await asyncio.sleep(0.03)
    raise AssertionError("incognito run did not complete")


async def _wait_for_run_status(
    client: AsyncClient,
    run_id: str,
    expected: str,
) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] == expected:
            return run
        if run["status"] in {"complete", "failed", "cancelled"}:
            raise AssertionError(f"expected {expected}, received {run['status']}")
        await asyncio.sleep(0.03)
    raise AssertionError(f"incognito run did not reach {expected}")


async def test_incognito_content_never_enters_durable_storage(
    app: FastAPI,
    client: AsyncClient,
    settings,
) -> None:  # type: ignore[no-untyped-def]
    durable_chat = (
        await client.post("/api/chats", json={"title": "Durable control conversation"})
    ).json()
    marker = "INCOGNITO-MARKER-7f14044c56"
    marker_bytes = b"INCOGNITO-BYTES-6df5529380"

    started = await client.post("/api/incognito/session")
    assert started.status_code == 201
    token = started.json()["token"]
    assert marker not in started.text
    assert app.state.services.processes.private_output_suppressed is True
    client.headers[INCOGNITO_HEADER] = token

    assert (await client.get("/api/chats")).json() == []
    private_chat = (await client.post("/api/chats", json={"title": f"Private {marker}"})).json()
    accepted = await client.post(
        f"/api/chats/{private_chat['id']}/turns",
        json={"text": f"Repeat this synthetic marker: {marker}", "mode": "text"},
    )
    assert accepted.status_code == 202
    await _wait_for_terminal_run(client, accepted.json()["run"]["id"])

    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("private.bin", marker_bytes, "application/octet-stream")},
    )
    assert uploaded.status_code == 201
    delivered = await client.get(uploaded.json()["url"])
    assert delivered.status_code == 200
    assert delivered.content == marker_bytes
    assert delivered.headers["cache-control"] == "no-store"

    plans = (await client.get("/api/work-plans", params={"chat_id": private_chat["id"]})).json()
    assert len(plans) == 1
    assert plans[0]["persistence_scope"] == "incognito"

    with SessionLocal() as durable:
        assert durable.get(Chat, durable_chat["id"])
        assert not durable.get(Chat, private_chat["id"])
        assert (
            durable.scalar(
                select(func.count(Message.id)).where(Message.chat_id == private_chat["id"])
            )
            == 0
        )
        assert (
            durable.scalar(
                select(func.count(WorkPlan.id)).where(WorkPlan.chat_id == private_chat["id"])
            )
            == 0
        )
        assert not durable.get(Artifact, uploaded.json()["id"])

    marker_digest = hashlib.sha256(marker_bytes).hexdigest()
    durable_marker_path = (
        settings.artifact_dir / marker_digest[:2] / marker_digest[2:4] / marker_digest
    )
    assert not durable_marker_path.exists()

    backup = await client.post("/api/backups")
    diagnostics = await client.post("/api/diagnostics")
    projects = await client.get("/api/projects")
    assert backup.status_code == 409
    assert diagnostics.status_code == 409
    assert projects.status_code == 409
    for path in settings.data_dir.rglob("*"):
        if path.is_file() and "incognito" not in path.parts:
            assert marker.encode() not in path.read_bytes()
            assert marker_bytes not in path.read_bytes()

    scope_root = app.state.services.incognito.get(token).root
    assert scope_root.is_dir()
    ended = await client.delete("/api/incognito/session")
    assert ended.status_code == 204
    assert app.state.services.processes.private_output_suppressed is False
    assert not scope_root.exists()
    assert (await client.get("/api/chats")).status_code == 404

    del client.headers[INCOGNITO_HEADER]
    durable_chats = (await client.get("/api/chats")).json()
    assert [chat["id"] for chat in durable_chats] == [durable_chat["id"]]


async def test_incognito_rejects_an_adapter_without_a_purge_contract(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(app.state.services.engines.chat, "supports_incognito", False)
    response = await client.post("/api/incognito/session")
    assert response.status_code == 409
    assert "cannot purge run state" in response.json()["detail"]


async def test_startup_sweeps_only_managed_stale_incognito_roots(
    app: FastAPI,
    settings,
) -> None:  # type: ignore[no-untyped-def]
    manager = app.state.services.incognito
    stale = manager.root / f"scope_{'c' * 48}"
    stale.mkdir(parents=True)
    (stale / "marker.bin").write_bytes(b"synthetic stale marker")
    unrelated = manager.root / "keep-me"
    unrelated.mkdir()

    await manager.sweep_stale_roots()

    assert not stale.exists()
    assert unrelated.is_dir()
    unrelated.rmdir()


async def test_incognito_failure_retry_cancel_and_shutdown_cleanup(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    started = await client.post("/api/incognito/session")
    token = started.json()["token"]
    client.headers[INCOGNITO_HEADER] = token
    scope = app.state.services.incognito.get(token)
    scope_root = scope.root
    chat = (await client.post("/api/chats", json={"title": "Lifecycle"})).json()

    adapter = app.state.services.engines.chat
    original_stream = adapter.stream

    async def fail_after_marker(request):  # type: ignore[no-untyped-def]
        yield ChatEvent(type="delta", text="synthetic private failure marker")
        yield ChatEvent(type="error", data={"error": ""})

    monkeypatch.setattr(adapter, "stream", fail_after_marker)
    failed = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Fail synthetically", "mode": "text"},
    )
    failed_run = await _wait_for_run_status(client, failed.json()["run"]["id"], "failed")
    jobs = (await client.get("/api/jobs")).json()
    failed_job = next(job for job in jobs if job["run_id"] == failed_run["id"])

    monkeypatch.setattr(adapter, "stream", original_stream)
    retried = await client.post(f"/api/jobs/{failed_job['id']}/retry")
    assert retried.status_code == 200
    await _wait_for_terminal_run(client, failed_run["id"])

    media = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a cancellable private image", "mode": "image"},
    )
    cancelled = await client.post(f"/api/chats/{chat['id']}/cancel")
    assert cancelled.status_code == 200
    await _wait_for_run_status(client, media.json()["run"]["id"], "cancelled")

    await app.state.services.incognito.close()
    assert not scope_root.exists()
    assert app.state.services.processes.private_output_suppressed is False
    assert (await client.get("/api/chats")).status_code == 404


async def test_durable_and_incognito_work_are_isolated_while_coexisting(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    durable_chat = (await client.post("/api/chats", json={"title": "Durable"})).json()
    durable_turn = await client.post(
        f"/api/chats/{durable_chat['id']}/turns",
        json={
            "text": "Durable synthetic request",
            "mode": "text",
            "idempotency_key": "same-client-key",
        },
    )
    await _wait_for_terminal_run(client, durable_turn.json()["run"]["id"])

    started = await client.post("/api/incognito/session")
    token = started.json()["token"]
    client.headers[INCOGNITO_HEADER] = token
    private_chat = (await client.post("/api/chats", json={"title": "Private"})).json()
    private_turn = await client.post(
        f"/api/chats/{private_chat['id']}/turns",
        json={
            "text": "Private synthetic request",
            "mode": "text",
            "idempotency_key": "same-client-key",
        },
    )
    assert private_turn.status_code == 202
    await _wait_for_terminal_run(client, private_turn.json()["run"]["id"])
    assert private_chat["id"] != durable_chat["id"]
    assert private_turn.json()["run"]["id"] != durable_turn.json()["run"]["id"]

    del client.headers[INCOGNITO_HEADER]
    durable_chats = (await client.get("/api/chats")).json()
    assert [item["id"] for item in durable_chats] == [durable_chat["id"]]

    client.headers[INCOGNITO_HEADER] = token
    private_chats = (await client.get("/api/chats")).json()
    assert [item["id"] for item in private_chats] == [private_chat["id"]]
    await client.delete("/api/incognito/session")


async def test_durable_profile_created_during_incognito_is_available_to_private_chat(
    client: AsyncClient,
) -> None:
    started = await client.post("/api/incognito/session")
    token = started.json()["token"]
    client.headers[INCOGNITO_HEADER] = token
    chat = (await client.post("/api/chats", json={"title": "Private configuration"})).json()

    profile = await client.post(
        "/api/profiles",
        json={
            "name": "New durable profile during Incognito",
            "role": "chat",
            "engine": "mock",
            "model_install_id": None,
            "load_settings": {},
            "request_settings": {},
            "is_default": False,
        },
    )
    assert profile.status_code == 201
    updated = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_chat_profile_id": profile.json()["id"]},
    )
    assert updated.status_code == 200
    assert updated.json()["active_chat_profile_id"] == profile.json()["id"]

    await client.delete("/api/incognito/session")
    del client.headers[INCOGNITO_HEADER]
    durable_profiles = (await client.get("/api/profiles", params={"role": "chat"})).json()
    assert profile.json()["id"] in {item["id"] for item in durable_profiles}


async def test_incognito_accepts_concurrent_turns_with_isolated_database_sessions(
    client: AsyncClient,
) -> None:
    started = await client.post("/api/incognito/session")
    client.headers[INCOGNITO_HEADER] = started.json()["token"]
    chat = (await client.post("/api/chats", json={"title": "Concurrent private work"})).json()

    responses = await asyncio.gather(
        *(
            client.post(
                f"/api/chats/{chat['id']}/turns",
                json={
                    "text": f"Synthetic private request {index}",
                    "mode": "text",
                    "idempotency_key": f"private-concurrent-{index}",
                },
            )
            for index in range(3)
        )
    )

    assert all(response.status_code == 202 for response in responses)
    run_ids = [response.json()["run"]["id"] for response in responses]
    assert len(set(run_ids)) == 3
    await asyncio.gather(*(_wait_for_terminal_run(client, run_id) for run_id in run_ids))
    await client.delete("/api/incognito/session")


async def test_incognito_purge_failure_is_generic_and_retryable(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    started = await client.post("/api/incognito/session")
    token = started.json()["token"]
    client.headers[INCOGNITO_HEADER] = token
    chat = (await client.post("/api/chats", json={"title": "Private purge retry"})).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a run to purge", "mode": "text"},
    )
    await _wait_for_terminal_run(client, accepted.json()["run"]["id"])

    adapter = app.state.services.engines.chat
    original = adapter.purge_run
    attempts = 0

    async def flaky_purge(run_id: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("PRIVATE-PURGE-ERROR-MARKER")
        await original(run_id)

    monkeypatch.setattr(adapter, "purge_run", flaky_purge)
    scope_root = app.state.services.incognito.get(token).root
    first = await client.delete("/api/incognito/session")
    assert first.status_code == 503
    assert "PRIVATE-PURGE-ERROR-MARKER" not in first.text
    assert scope_root.exists()
    assert app.state.services.processes.private_output_suppressed is True

    manager = app.state.services.incognito
    original_remove = manager._remove_scope_root
    remove_attempts = 0

    def flaky_remove(path):  # type: ignore[no-untyped-def]
        nonlocal remove_attempts
        remove_attempts += 1
        if remove_attempts == 1:
            raise OSError("PRIVATE-FILESYSTEM-ERROR-MARKER")
        original_remove(path)

    monkeypatch.setattr(manager, "_remove_scope_root", flaky_remove)
    second = await client.delete("/api/incognito/session")
    assert second.status_code == 503
    assert "PRIVATE-FILESYSTEM-ERROR-MARKER" not in second.text
    assert attempts == 2
    assert scope_root.exists()
    assert app.state.services.processes.private_output_suppressed is True

    third = await client.delete("/api/incognito/session")
    assert third.status_code == 204
    assert attempts == 2
    assert remove_attempts == 2
    assert not scope_root.exists()
    assert app.state.services.processes.private_output_suppressed is False
