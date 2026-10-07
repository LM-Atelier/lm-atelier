"""Compare packaged node names with an isolated, physically verified runtime."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy import select
from test_workflow_review_live_comfy import settings as settings

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall, CustomNodeInstall
from local_lm.runtime_provisioning import RuntimeProvisioner
from local_lm.workflow_review_runtime import review_runtime_object_info
from local_lm.workflow_revision_reviews import _CORE_MODULE


@pytest.mark.asyncio
async def test_verified_runtime_matches_its_packaged_node_inventory(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = os.environ.get("LM_ATELIER_TEST_COMFY_DATA_DIR")
    if not data:
        pytest.skip("The previously verified disposable runtime data directory is required")
    services = app.state.services
    configured = services.settings
    verification = RuntimeProvisioner(
        Settings(
            data_dir=Path(data),
            chat_engine="mock",
            media_engine="mock",
            comfy_executable=configured.comfy_executable,
            comfy_directory=configured.comfy_directory,
        ),
        environment={},
        platform_key="windows-x86_64-nvidia-cu13",
    )
    try:
        before = await asyncio.to_thread(verification.verify_status, "comfyui")
        assert before.state == "ready" and before.managed
        original = services.processes._replace

        async def cpu_replace(name: str, command: list[str], *args: Any, **kwargs: Any) -> None:
            await original(
                name,
                [*command, "--cpu", "--disable-all-custom-nodes"] if name == "media" else command,
                *args,
                **kwargs,
            )

        monkeypatch.setattr(services.processes, "_replace", cpu_replace)
        async with services.scheduler.lease("primary"):
            await services.processes.start_media()
            with SessionLocal() as session:
                assert list(session.scalars(select(ComfyRegistryInstall))) == []
                assert list(session.scalars(select(CustomNodeInstall))) == []
            info = await review_runtime_object_info(services.processes, services.engines.media)
            assert info is not None
            nodes = {
                name: value["python_module"]
                for name, value in sorted(info.items())
                if isinstance(value, dict)
                and isinstance(value.get("python_module"), str)
                and (
                    value["python_module"] == "nodes"
                    or _CORE_MODULE.fullmatch(value["python_module"]) is not None
                )
            }
            assert {"EmptyImage", "SaveImage", "LoraLoader"} <= nodes.keys()
            assert not {"ExampleNode", "ExampleLoraImage"} & nodes.keys()
            definition = verification._definition("comfyui")
            asset = definition["runtime_assets"]["windows-x86_64-nvidia-cu13"]
            measured = {
                "version": 1,
                "engine": "comfyui",
                "release": definition["pinned_release"],
                "asset_key": "windows-x86_64-nvidia-cu13",
                "archive_sha256": asset["sha256"],
                "runtime_contract_sha256": verification._runtime_contract_sha256(asset),
                "nodes": nodes,
            }
            after = await asyncio.to_thread(verification.verify_status, "comfyui")
            assert after.state == "ready" and after.managed
        recorded = tmp_path / "runtime-nodes.json"
        recorded.write_bytes((json.dumps(measured, indent=2) + "\n").encode("utf-8"))
        expected = (
            Path(__file__).resolve().parents[3]
            / "packaging/runtime-reviews/comfyui-v0.28.0-nodes.json"
        )
        assert json.loads(expected.read_text(encoding="utf-8")) == measured
    finally:
        try:
            await services.processes.stop("media")
        finally:
            await verification.close()
