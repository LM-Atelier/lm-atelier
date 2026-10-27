"""An imported workflow's latent upscaler installs where ComfyUI loads it.

`LatentUpscaleModelLoader` reads its weights from `latent_upscale_models`. The
analyzer reported the file as an ordinary upscaler, so the asset installer asked
the catalog for an upscaler, the install check refused the file as the wrong
kind, and the workflow could never be completed. Had it been bound, it would have
landed in `upscale_models`, where that loader never looks.

These cases follow the browser's own path: analyze the graph, check the file the
workflow names under the kind the analysis reported, bind it, queue it, and then
hold the staged download to the plan exactly as the download manager does.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.catalog import HuggingFaceCatalog
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.models import InstallPlan
from local_lm.schemas import DownloadRequest

REPOSITORY = "studio/motion-upscaler"
REVISION = "b" * 40
FILENAME = "motion-spatial-upscaler-x2.safetensors"


def _safetensors(tensor_names: list[str]) -> bytes:
    header = json.dumps(
        {
            name: {"dtype": "F32", "shape": [1], "data_offsets": [index * 4, index * 4 + 4]}
            for index, name in enumerate(tensor_names)
        }
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"\0" * (4 * len(tensor_names))


# Tensor names no inspector rule recognises, as in a real spatial upsampler.
CONTENT = _safetensors(["initial_conv.weight", "upsampler.0.weight"])
DIGEST = hashlib.sha256(CONTENT).hexdigest()


def _workflow() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "LatentUpscaleModelLoader",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [FILENAME],
                "properties": {"cnr_id": "comfy-core", "version": "0.28.0"},
            }
        ],
        "links": [],
    }


@pytest.mark.parametrize("header_readable", [True, False], ids=["header", "no-header"])
async def test_an_imported_latent_upscaler_installs_into_its_own_folder(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    header_readable: bool,
) -> None:
    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict[str, Any]:
        del revision, requested_role
        assert remote_id == REPOSITORY
        return {
            "model": {
                "remote_id": remote_id,
                "name": "motion-upscaler",
                "license_id": "apache-2.0",
                "pipeline_tag": None,
                "formats": ["safetensors"],
                "compatibility": "likely",
                "compatibility_reasons": [],
            },
            "revision": REVISION,
            "files": [{"filename": FILENAME, "size": len(CONTENT), "sha256": DIGEST}],
        }

    async def inspect_file_prefix(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str,
        filename: str,
        *,
        max_bytes: int,
    ) -> bytes:
        assert (remote_id, revision, filename) == (REPOSITORY, REVISION, FILENAME)
        if not header_readable:
            raise ValueError("this provider serves no header for the file")
        return CONTENT[:max_bytes]

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    monkeypatch.setattr(HuggingFaceCatalog, "inspect_file_prefix", inspect_file_prefix)
    monkeypatch.setattr(app.state.services.downloads, "start", lambda _job_id: None)

    analyzed = await client.post("/api/workflows/packages/analyze", json={"ui_graph": _workflow()})
    assert analyzed.status_code == 200, analyzed.text
    (reference,) = analyzed.json()["asset_references"]
    assert (reference["filename"], reference["present_locally"]) == (FILENAME, False)

    # The installer checks the named file under whatever kind the analysis reported.
    checked = await client.post(
        f"/api/catalog/{REPOSITORY}/preflight",
        json={
            "revision": "main",
            "role": "image",
            "engine": "comfyui",
            "selected_files": [FILENAME],
            "workflow_reference_kind": reference["kind"],
        },
    )
    assert checked.status_code == 200, checked.text
    plan = checked.json()["install_plan"]
    assert (plan["compatibility"], plan["failure_code"]) == ("supported", None)
    assert reference["kind"] == "latent_upscaler"
    (artifact,) = plan["artifacts_json"]
    assert (artifact["kind"], artifact["target_folder"]) == (
        "latent_upscaler",
        "latent_upscale_models",
    )
    assert plan["runtime_contract_json"]["comfy_paths"] == {"latent_upscale_models": "."}

    selections = [
        {
            "reference_filename": FILENAME,
            "install_plan_id": plan["id"],
            "artifact_path": FILENAME,
        }
    ]
    reviewed = await client.post(
        "/api/workflows/packages/assets/review",
        json={"ui_graph": _workflow(), "selections": selections},
    )
    assert reviewed.status_code == 200, reviewed.text
    (bound,) = reviewed.json()["assets"]
    assert (bound["kind"], bound["artifact_kind"], bound["target_folder"]) == (
        "latent_upscaler",
        "latent_upscaler",
        "latent_upscale_models",
    )

    queued = await client.post(
        "/api/workflows/packages/assets/install",
        json={
            "ui_graph": _workflow(),
            "selections": selections,
            "binding_plan_hash": reviewed.json()["binding_plan_hash"],
        },
    )
    assert queued.status_code == 202, queued.text
    (job,) = queued.json()
    request = DownloadRequest.model_validate(job["payload_json"])
    assert request.workflow_asset_kind == "latent_upscaler"
    assert request.comfy_paths == {"latent_upscale_models": "."}

    # The worker inspects the staged bytes against the plan before keeping them.
    staging = tmp_path / "staged-upscaler"
    staging.mkdir()
    (staging / FILENAME).write_bytes(CONTENT)
    with SessionLocal() as session:
        stored = session.get(InstallPlan, request.install_plan_id)
        assert stored is not None
        inspection = app.state.services.downloads._inspect_staged_component_manifest(
            staging, [FILENAME], request, None, stored
        )
        assert inspection is not None
        assert [(item.kind, item.target_folder) for item in inspection.components] == [
            ("latent_upscaler", "latent_upscale_models")
        ]
        DownloadManager._validate_staged_plan(stored, inspection, {FILENAME: DIGEST})
