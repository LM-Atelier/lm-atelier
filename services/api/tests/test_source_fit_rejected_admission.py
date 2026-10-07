"""Rejected source canvases must not create files even when database rows roll back."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import event, func, select
from sqlalchemy.engine import Engine
from test_source_fit_acceptance import prepared_turn
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.db import SessionLocal
from local_lm.models import Artifact, Job, Message, Run, RunContextSnapshot, WorkPlan, WorkStep


def artifact_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def durable_counts() -> tuple[int | None, ...]:
    with SessionLocal() as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (Artifact, Job, Message, Run, RunContextSnapshot, WorkPlan, WorkStep)
        )


@pytest.mark.parametrize("refusal", ["graph", "canvas", "accepted"])
async def test_source_canvas_is_validated_before_admission_writes(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    refusal: str,
) -> None:
    chat_id, request = await prepared_turn(
        app, client, monkeypatch, broken_graph=refusal == "graph"
    )
    services = app.state.services
    async with services.scheduler.lease("primary"):
        before_files = artifact_files(services.artifacts.root)
        before_counts = durable_counts()
        assert before_files
        writes: list[str] = []

        def record_write(
            _connection: Any, _cursor: Any, _sql: str, _parameters: Any, context: Any, _many: Any
        ) -> None:
            if context.isinsert or context.isupdate or context.isdelete:
                table = context.compiled.statement.table.name
                if table in {
                    "messages",
                    "message_parts",
                    "work_plans",
                    "work_steps",
                    "runs",
                    "jobs",
                    "artifacts",
                    "run_context_snapshots",
                    "response_revisions",
                }:
                    writes.append(table)

        event.listen(Engine, "before_cursor_execute", record_write)
        try:
            response = await client.post(
                f"/api/chats/{chat_id}/turns",
                json={
                    **request,
                    "source_fit": {
                        "mode": "extend",
                        "width": 1 if refusal == "canvas" else 4,
                        "height": 5,
                    },
                },
            )
        finally:
            event.remove(Engine, "before_cursor_execute", record_write)
        if refusal == "accepted":
            assert response.status_code == 202, response.text
            assert {"messages", "work_plans", "runs", "jobs", "run_context_snapshots"} <= set(
                writes
            )
            assert durable_counts() != before_counts
        else:
            assert response.status_code == 422, response.text
            assert durable_counts() == before_counts
            assert artifact_files(services.artifacts.root) == before_files
            assert writes == []
