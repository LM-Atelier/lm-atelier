"""Project trash refuses accepted work that cannot be safely preserved."""

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import select
from test_chat_recovery import _command, _impact
from test_project_recovery_api import _project
from test_recovery_expiry import manual_expiry as manual_expiry

from local_lm import orchestrator
from local_lm.db import SessionLocal
from local_lm.models import RecoveryItem, RecoveryOperation, Run, RunContextSnapshot


@pytest.mark.parametrize("fault", ["unknown-status", "invalid-snapshot", "row-limit"])
async def test_project_trash_refuses_unpreservable_pending_work_atomically(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    project_id, chat_id = await _project(client)
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={"text": "Keep the garden paths clear", "mode": "text"},
        )
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run"]["id"]
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None and session.get(RunContextSnapshot, run_id) is None
            status = run.status
            if fault == "unknown-status":
                run.status = "unrecognized-state"
            elif fault == "invalid-snapshot":
                session.add(RunContextSnapshot(run_id=run_id, payload_json={}, sha256="a" * 64))
            session.commit()
            provenance = dict(run.provenance_json)
        if fault == "row-limit":
            monkeypatch.setattr(orchestrator, "MAX_GRAPH_ROWS", 0)

        try:
            preview = await _impact(client, f"/api/projects/{project_id}/deletion-impact")
            sequence = app.state.services.events.sequence
            refused = await client.post(
                f"/api/projects/{project_id}/trash",
                json=_command(preview, "trash-unpreservable-project"),
            )
            assert refused.status_code == 409, refused.text
            assert refused.json()["code"] == "project-recovery-accepted-work-invalid"
            assert app.state.services.events.sequence == sequence
            assert (await client.get(f"/api/projects/{project_id}")).status_code == 200
            with SessionLocal() as session:
                assert session.scalar(select(RecoveryItem)) is None
                assert session.scalar(select(RecoveryOperation)) is None
                run = session.get(Run, run_id)
                assert run is not None and run.provenance_json == provenance
                assert run.status == ("unrecognized-state" if fault == "unknown-status" else status)
                snapshot = session.get(RunContextSnapshot, run_id)
                if fault == "invalid-snapshot":
                    assert snapshot is not None
                    assert snapshot.payload_json == {} and snapshot.sha256 == "a" * 64
                else:
                    assert snapshot is None
        finally:
            with SessionLocal() as session:
                run = session.get(Run, run_id)
                assert run is not None
                run.status = status
                snapshot = session.get(RunContextSnapshot, run_id)
                if fault == "invalid-snapshot" and snapshot is not None:
                    session.delete(snapshot)
                session.commit()

    async def read():
        return (await client.get(f"/api/runs/{run_id}")).json()

    assert (await wait_for_terminal_status(read, what="preserved garden work"))[
        "status"
    ] == "complete"
