from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import patch

from fastapi import FastAPI
from local_lm.main import create_app

from .runner_readiness import install_readiness

RUNTIME_IDENTITY = "workflow-editor-synthetic-browser-protocol-fixture"

app = create_app()
install_readiness(app)
_product_lifespan = app.router.lifespan_context


@asynccontextmanager
async def _fixture_lifespan(active_app: FastAPI) -> AsyncIterator[None]:
    async with _product_lifespan(active_app):
        services = active_app.state.services
        with patch.object(
            services.processes,
            "workflow_editor_runtime_identity",
            return_value=RUNTIME_IDENTITY,
        ):
            yield


app.router.lifespan_context = _fixture_lifespan
