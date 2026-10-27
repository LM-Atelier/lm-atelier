"""Official video workflows that upscale in latent space install as one set.

ComfyUI's latent upscale loader reads its weights from a folder of its own,
`latent_upscale_models`. The install check accepted an official bundle that
needed one, and the download it had just planned was then refused for naming a
folder the installer did not know. These cases hold the whole path: the check
and the download agree, a downloaded upscaler keeps the kind it was planned as,
and the runtime is told where to load it from.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.downloads import DownloadManager
from local_lm.model_manifests import inspect_repository_metadata
from local_lm.models import InstallPlan, ModelInstall
from local_lm.processes import ProcessSupervisor

REPOSITORY = "studio/motion"
REVISION = "a" * 40
TEMPLATE = "video_motion_spatial_upscale_t2v"
MODEL = "motion.safetensors"
UPSCALER = "latent_upscale_models/motion-spatial-upscaler-x2.safetensors"
BUNDLE = [
    ("UNETLoader", "diffusion_models", MODEL, 4_096, "1" * 64),
    ("LatentUpscaleModelLoader", "latent_upscale_models", UPSCALER, 1_024, "2" * 64),
    ("CLIPLoader", "text_encoders", "encoders/motion-text.safetensors", 2_048, "3" * 64),
    ("VAELoader", "vae", "motion-vae.safetensors", 512, "4" * 64),
]


def _install_official_template(settings: Settings, tmp_path: Path) -> None:
    settings.media_engine = "comfyui"
    runtime = tmp_path / "external-comfy"
    runtime.mkdir()
    settings.comfy_executable = runtime / "python.exe"
    settings.comfy_executable.write_bytes(b"external runtime")
    settings.comfy_directory = runtime / "ComfyUI"
    templates = (
        settings.comfy_directory
        / ".venv"
        / "Lib"
        / "site-packages"
        / "comfyui_workflow_templates_json"
        / "templates"
    )
    templates.mkdir(parents=True)
    (settings.comfy_directory / "main.py").write_text("", encoding="utf-8")
    nodes: list[dict[str, Any]] = [
        {
            "id": index,
            "type": loader,
            "inputs": [],
            "outputs": [],
            "properties": {
                "cnr_id": "comfy-core",
                "models": [
                    {
                        "directory": directory,
                        "name": Path(path).name,
                        "url": f"https://huggingface.co/{REPOSITORY}/resolve/main/{path}",
                    }
                ],
            },
            "widgets_values": [Path(path).name],
        }
        for index, (loader, directory, path, _size, _digest) in enumerate(BUNDLE, start=1)
    ]
    nodes.append(
        {
            "id": 10,
            "type": "SaveVideo",
            "inputs": [],
            "outputs": [],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": ["motion"],
        }
    )
    (templates / f"{TEMPLATE}.json").write_text(
        json.dumps({"nodes": nodes, "links": []}), encoding="utf-8"
    )
    (templates / "index.json").write_text(
        json.dumps([{"name": TEMPLATE, "date": "2026-07-10"}]), encoding="utf-8"
    )


async def test_the_install_check_and_the_download_agree_on_a_latent_upscaler(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_official_template(settings, tmp_path)

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
                "name": "motion",
                "license_id": "apache-2.0",
                "pipeline_tag": "text-to-video",
                "formats": ["safetensors"],
                "compatibility": "likely",
                "compatibility_reasons": ["video pipeline metadata detected"],
            },
            "revision": REVISION,
            "files": [
                {"filename": path, "size": size, "sha256": digest}
                for _loader, _directory, path, size, digest in BUNDLE
            ],
        }

    async def inspect_file_prefix(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise ValueError("headers are not read for this case")

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    monkeypatch.setattr(HuggingFaceCatalog, "inspect_file_prefix", inspect_file_prefix)
    monkeypatch.setattr(app.state.services.downloads, "start", lambda _job_id: None)

    checked = await client.post(
        f"/api/catalog/{REPOSITORY}/preflight",
        json={
            "revision": REVISION,
            "role": "video",
            "engine": "comfyui",
            "workflow_template_id": TEMPLATE,
        },
    )
    assert checked.status_code == 200, checked.text
    preflight = checked.json()
    assert preflight["can_install"] is True
    assert preflight["comfy_paths"] == {
        "diffusion_models": ".",
        "latent_upscale_models": "latent_upscale_models",
        "text_encoders": "encoders",
        "vae": ".",
    }
    assert UPSCALER in preflight["selected_files"]

    # The request the browser builds from that check, field for field.
    downloaded = await client.post(
        "/api/downloads",
        json={
            "remote_id": preflight["remote_id"],
            "source_remote_id": preflight["source_remote_id"],
            "revision": preflight["revision"],
            "role": "video",
            "engine": "comfyui",
            "allow_patterns": preflight["selected_files"],
            "expected_sha256": preflight["expected_sha256"],
            "file_sources": preflight["file_sources"] or {},
            "comfy_paths": preflight["comfy_paths"],
            "workflow_template_id": preflight["workflow_template_id"],
            "workflow_template_sha256": preflight["workflow_template_sha256"],
            "install_plan_id": preflight["install_plan"]["id"],
            "auxiliary_kind": None,
            "content_rating": preflight.get("content_rating") or "unknown",
        },
    )
    assert downloaded.status_code == 202, downloaded.text
    assert downloaded.json()["payload_json"]["comfy_paths"] == preflight["comfy_paths"]
    # The plan files the upscaler where the workflow loads it from, under its own kind.
    artifacts = {item["path"]: item for item in preflight["install_plan"]["artifacts_json"]}
    assert (artifacts[UPSCALER]["kind"], artifacts[UPSCALER]["target_folder"]) == (
        "latent_upscaler",
        "latent_upscale_models",
    )


def _safetensors_header(tensor_names: list[str]) -> bytes:
    header = json.dumps(
        {
            name: {"dtype": "F32", "shape": [1], "data_offsets": [index * 4, index * 4 + 4]}
            for index, name in enumerate(tensor_names)
        }
    ).encode("utf-8")
    return struct.pack("<Q", len(header)) + header


def test_a_downloaded_latent_upscaler_keeps_the_kind_it_was_planned_as() -> None:
    planned = {
        UPSCALER: ("latent_upscaler", "latent_upscale_models"),
        MODEL: ("diffusion_model", "diffusion_models"),
    }
    inspection = inspect_repository_metadata(
        {
            UPSCALER: _safetensors_header(["initial_conv.weight", "upsampler.0.weight"]),
            MODEL: _safetensors_header(["model.diffusion_model.block.weight"]),
        },
        list(planned),
        role="video",
        component_folders={path: folder for path, (_kind, folder) in planned.items()},
    )
    assert {item.path: (item.kind, item.target_folder) for item in inspection.components} == (
        planned
    )

    plan = InstallPlan(
        family=None,
        role="video",
        artifacts_json=[
            {"path": path, "kind": kind, "target_folder": folder, "required": True}
            for path, (kind, folder) in planned.items()
        ],
        runtime_contract_json={
            "auxiliary_kind": None,
            "workflow_asset_kind": None,
            "workflow_template_id": TEMPLATE,
        },
    )
    DownloadManager._validate_staged_plan(plan, inspection, {UPSCALER: "2" * 64, MODEL: "1" * 64})


def test_the_runtime_is_told_where_an_installed_latent_upscaler_lives(
    app: FastAPI, settings: Settings, tmp_path: Path
) -> None:
    del app
    installed = tmp_path / "installed-motion"
    (installed / "latent_upscale_models").mkdir(parents=True)
    paths = {"diffusion_models": ".", "latent_upscale_models": "latent_upscale_models"}
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="install_motion",
                name=REPOSITORY,
                role="video",
                engine="comfyui",
                local_path=str(installed),
                size_bytes=1,
                manifest_json={"comfy_paths": paths},
                active=True,
            )
        )
        session.commit()

    settings.prepare()
    published = json.loads(
        ProcessSupervisor(settings)._write_comfy_model_paths().read_text(encoding="utf-8")
    )

    assert {"base_path": str(installed.resolve()), **paths} in published.values()
