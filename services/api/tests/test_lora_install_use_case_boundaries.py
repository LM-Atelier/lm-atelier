from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_downloads import safetensors_bytes

from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.domain import JobKind, JobStatus
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.model_manifests import inspect_repository_metadata
from local_lm.model_planner import (
    INSTALL_RESOLVER_VERSION,
    persist_install_plan,
    resolve_install_plan,
)
from local_lm.models import InstallPlan, Job, ModelAssetInstall, ModelInstall, ModelSource
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import DownloadRequest


async def test_an_accepted_plan_without_use_case_metadata_ignores_later_provider_tags(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    content = safetensors_bytes(
        ["lora_unet_block.lora_down.weight"],
        {
            "ss_network_module": "networks.lora",
            "ss_network_dim": "8",
            "modelspec.trigger_phrase": "atelier ink",
        },
    )
    digest = hashlib.sha256(content).hexdigest()
    inspection = inspect_repository_metadata(
        {"adapter.safetensors": content},
        ["adapter.safetensors"],
        role="image",
    )
    resolved = resolve_install_plan(
        remote_id="synthetic/atelier-lora",
        revision="d" * 40,
        role="image",
        engine="comfyui",
        selected_files=[
            {
                "filename": "adapter.safetensors",
                "size": len(content),
                "sha256": digest,
                "metadata": {},
            }
        ],
        inspection=inspection,
        comfy_paths={"loras": "."},
        auxiliary_kind="lora",
    )
    with SessionLocal() as session:
        plan = persist_install_plan(session, resolved)
        session.commit()
        request = DownloadRequest(
            install_plan_id=plan.id,
            remote_id=plan.remote_id,
            revision=plan.revision,
            role="image",
            engine=plan.engine,
            allow_patterns=["adapter.safetensors"],
            expected_sha256={"adapter.safetensors": digest},
            comfy_paths={"loras": "."},
            auxiliary_kind="lora",
        )
        session.add(
            Job(
                id="job_atelier_lora",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.QUEUED.value,
                payload_json=request.model_dump(mode="json"),
            )
        )
        session.commit()

    class Processes:
        def __init__(self) -> None:
            self.started: list[tuple[Path, dict[str, str]]] = []
            self.stopped: list[str] = []

        def statuses(self) -> list[object]:
            return [SimpleNamespace(name="media", running=False, profile_id=None)]

        async def start_media(
            self, model_root: tuple[Path, dict[str, str]], **_kwargs: object
        ) -> None:
            self.started.append(model_root)

        async def stop(self, name: str, **_kwargs: object) -> None:
            self.stopped.append(name)

    class MediaAdapter:
        def invalidate_object_info_cache(self) -> None:
            return None

        async def object_info(self) -> dict[str, object]:
            return {"LoraLoader": {}}

    processes = Processes()
    manager = DownloadManager(
        settings,
        EventBroker(),
        scheduler=ResourceScheduler(),
        media_adapter=cast(Any, MediaAdapter()),
        processes=cast(Any, processes),
    )
    cast(Any, manager)._api = SimpleNamespace(
        model_info=lambda *_args, **_kwargs: SimpleNamespace(
            siblings=[
                SimpleNamespace(
                    rfilename="adapter.safetensors",
                    size=len(content),
                    lfs={"sha256": digest},
                )
            ],
            sha="d" * 40,
            pipeline_tag=None,
            tags=["watercolor landscapes"],
            gated=False,
        )
    )

    async def download_file(**kwargs: Any) -> str:
        target = kwargs["staging"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    monkeypatch.setattr(manager, "_download_file", download_file)
    await manager._download("job_atelier_lora")

    with SessionLocal() as session:
        job = session.get(Job, "job_atelier_lora")
        asset = session.query(ModelAssetInstall).one()
        stored_plan = session.get(type(plan), plan.id)
        assert job and job.status == JobStatus.COMPLETE.value
        assert asset.active is True
        assert asset.verified_at is not None
        assert asset.kind == "lora"
        assert asset.use_case == ""
        assert asset.use_case_derived is False
        assert asset.auto_apply is False
        assert stored_plan is not None
        assert "use_case_metadata" not in stored_plan.runtime_contract_json
        assert asset.manifest_json["use_case_metadata"] == {}
        source = session.get(ModelSource, asset.source_id)
        assert source and source.metadata_json["tags"] == ["watercolor landscapes"]
        assert asset.manifest_json["sha256"] == digest
        assert asset.manifest_json["comfy_name"] == "adapter.safetensors"
        assert asset.manifest_json["metadata"]["trigger_words"] == [
            "atelier ink",
        ]
        # A request that declares no rating stores the explicit unknown.
        assert asset.manifest_json["content_rating"] == "unknown"
        assert stored_plan and stored_plan.status == "activated"
    assert processes.started[0][1] == {"loras": "."}
    assert processes.stopped == ["media"]


async def test_checkpoint_installation_does_not_derive_a_lora_use_case(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    filename = "workflow-checkpoint.safetensors"
    content = safetensors_bytes(
        [
            "model.diffusion_model.input_blocks.0.weight",
            "first_stage_model.encoder.weight",
        ]
    )
    digest = hashlib.sha256(content).hexdigest()
    plan_hash = "9" * 64
    with SessionLocal() as session:
        plan = InstallPlan(
            id="plan_workflow_checkpoint",
            provider="civitai",
            remote_id="101",
            revision="202",
            role="image",
            engine="comfyui",
            plan_hash=plan_hash,
            resolver_version=INSTALL_RESOLVER_VERSION,
            compatibility="supported",
            artifacts_json=[
                {
                    "path": filename,
                    "kind": "checkpoint",
                    "target_folder": "checkpoints",
                    "size_bytes": len(content),
                    "sha256": digest,
                    "required": True,
                    "reuse": "download",
                    "source_version_id": "202",
                    "source_file_id": "301",
                }
            ],
            runtime_contract_json={
                "use_case_metadata": {"tags": ["watercolor landscapes"]},
                "auxiliary_kind": None,
                "workflow_asset_kind": "checkpoint",
                "comfy_paths": {"checkpoints": "."},
                "workflow_component_folders": {filename: "checkpoints"},
            },
            activation_probe_json={"kind": "workflow_asset", "required": False},
            status="planned",
        )
        session.add(plan)
        request = DownloadRequest(
            install_plan_id=plan.id,
            remote_id=plan.remote_id,
            revision=plan.revision,
            role="image",
            engine="comfyui",
            allow_patterns=[filename],
            expected_sha256={filename: digest},
            comfy_paths={"checkpoints": "."},
            workflow_asset_kind="checkpoint",
        )
        session.add(
            Job(
                id="job_workflow_checkpoint",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.QUEUED.value,
                payload_json=request.model_dump(mode="json"),
            )
        )
        session.commit()

    class Processes:
        def __init__(self) -> None:
            self.started: list[tuple[Path, dict[str, str]]] = []
            self.stopped: list[str] = []

        def statuses(self) -> list[object]:
            return [SimpleNamespace(name="media", running=False, profile_id=None)]

        async def start_media(
            self, model_root: tuple[Path, dict[str, str]], **_kwargs: object
        ) -> None:
            self.started.append(model_root)

        async def stop(self, name: str, **_kwargs: object) -> None:
            self.stopped.append(name)

    class MediaAdapter:
        def invalidate_object_info_cache(self) -> None:
            return None

        async def object_info(self) -> dict[str, object]:
            return {"CheckpointLoaderSimple": {}}

    processes = Processes()
    manager = DownloadManager(
        settings,
        EventBroker(),
        scheduler=ResourceScheduler(),
        media_adapter=cast(Any, MediaAdapter()),
        processes=cast(Any, processes),
    )

    async def download_file(**kwargs: Any) -> str:
        target = kwargs["staging"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    monkeypatch.setattr(manager, "_download_file", download_file)
    await manager._download("job_workflow_checkpoint")

    with SessionLocal() as session:
        job = session.get(Job, "job_workflow_checkpoint")
        asset = session.query(ModelAssetInstall).one()
        stored_plan = session.get(InstallPlan, "plan_workflow_checkpoint")
        assert job and job.status == JobStatus.COMPLETE.value
        assert asset.active is True
        assert asset.verified_at is not None
        assert asset.kind == "checkpoint"
        assert asset.use_case == ""
        assert asset.use_case_derived is False
        assert stored_plan is not None
        assert stored_plan.runtime_contract_json["use_case_metadata"] == {
            "tags": ["watercolor landscapes"]
        }
        assert asset.manifest_json["comfy_name"] == filename
        assert asset.manifest_json["workflow_asset_kind"] == "checkpoint"
        assert session.query(ModelInstall).count() == 0
        assert stored_plan and stored_plan.status == "activated"
    assert processes.started[0][0].name.endswith(f"-asset-{plan_hash[:12]}")
    assert processes.started[0][1] == {"checkpoints": "."}
    assert processes.stopped == ["media"]
