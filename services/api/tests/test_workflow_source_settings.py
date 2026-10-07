from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime
from test_workflow_source_completion import (
    test_source_completion_requires_its_exact_downloaded_declared_resource as install_source,
)

from local_lm import models
from local_lm.db import SessionLocal


async def test_source_installation_binds_the_same_native_controls_as_package_import(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
) -> None:
    await install_source(client, app, monkeypatch, source_runtime, "none")
    with SessionLocal() as session:
        offer = session.scalar(select(models.WorkflowInstallOffer))
        assert offer is not None and offer.status == "completed"
        revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
        assert revision is not None
        assert revision.api_graph_json["1"]["inputs"]["seed"] == "${seed}", (
            "Source installation discarded its native setting binding"
        )
        assert revision.input_schema_json["properties"]["seed"]["default"] == 42
        assert revision.api_graph_json["1"]["inputs"]["label"] == "camera"
        assert revision.api_graph_json["3"]["inputs"]["lora_name"] == "detail.safetensors"
