from __future__ import annotations

import asyncio
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select

from local_lm.adapters.base import ChatEvent, ChatRequest, MediaRequest
from local_lm.auxiliary_assets import workflow_lora_extension
from local_lm.comfy_templates import (
    ComfyModelDependency,
    ComfyTemplate,
    CompiledComfyTemplate,
)
from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.domain import JobKind, JobStatus
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.model_manifests import (
    InspectedComponent,
    ModelManifestInspection,
    inspect_repository_metadata,
)
from local_lm.model_planner import (
    INSTALL_RESOLVER_VERSION,
    persist_install_plan,
    resolve_install_plan,
)
from local_lm.models import (
    InstallPlan,
    Job,
    ModelAssetInstall,
    ModelCapabilityEvidence,
    ModelComponentManifest,
    ModelInstall,
    ModelProfile,
    WorkflowDefinition,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import CatalogFileSource, DownloadRequest


class FakeWorker:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.pid = 123

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 1

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> int:
        return self.returncode or 0


class FakeCompletedWorker(FakeWorker):
    def __init__(self, command: list[str]) -> None:
        super().__init__()
        self.command = command
        self.payload = b""

    def communicate(self, payload: bytes) -> tuple[bytes, bytes]:
        self.payload = payload
        self.returncode = 0
        return json.dumps({"path": "C:/models/model.gguf"}).encode(), b""


class FakeBatchCompletedWorker(FakeCompletedWorker):
    def communicate(self, payload: bytes) -> tuple[bytes, bytes]:
        self.payload = payload
        self.returncode = 0
        request = json.loads(payload)
        return (
            json.dumps(
                {"paths": {filename: f"C:/models/{filename}" for filename in request["files"]}}
            ).encode(),
            b"",
        )


class FakeProbeAdapter:
    def __init__(self) -> None:
        self.request: MediaRequest | None = None
        self.timeout_seconds: float | None = None
        self.input_contents: list[bytes] = []

    async def probe_workflow(
        self,
        request: MediaRequest,
        *,
        timeout_seconds: float,
    ) -> None:
        self.request = request
        self.timeout_seconds = timeout_seconds
        self.input_contents = [path.read_bytes() for path in request.input_paths]


def gguf_bytes(architecture: str) -> bytes:
    key = b"general.architecture"
    value = architecture.encode()
    return (
        b"GGUF"
        + struct.pack("<IQQ", 3, 1, 1)
        + struct.pack("<Q", len(key))
        + key
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(value))
        + value
    )


def safetensors_bytes(
    tensor_names: list[str],
    metadata: dict[str, str] | None = None,
) -> bytes:
    header = {
        **{
            name: {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [index * 2, index * 2 + 2],
            }
            for index, name in enumerate(tensor_names)
        },
        "__metadata__": metadata or {},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return len(encoded).to_bytes(8, "little") + encoded


def test_official_workflow_staging_honors_declared_safe_component_contracts() -> None:
    contracts = [
        ("model.safetensors", "diffusion_model", "diffusion_models"),
        ("encoder.safetensors", "text_encoder", "text_encoders"),
        ("vae.safetensors", "vae", "vae"),
        ("lightning.safetensors", "lora", "loras"),
    ]
    plan = SimpleNamespace(
        family=None,
        role="image",
        artifacts_json=[
            {
                "path": path,
                "kind": kind,
                "target_folder": target,
                "required": True,
            }
            for path, kind, target in contracts
        ],
        runtime_contract_json={
            "auxiliary_kind": None,
            "workflow_template_id": "official-edit-template",
        },
    )
    inspection = ModelManifestInspection(
        architecture=None,
        family=None,
        components=tuple(
            InspectedComponent(
                path=path,
                kind="lora" if kind == "lora" else "unknown_safetensors",
                target_folder="loras" if kind == "lora" else "checkpoints",
            )
            for path, kind, _target in contracts
        ),
        metadata_files=(),
    )
    hashes = {path: str(index) * 64 for index, (path, _kind, _target) in enumerate(contracts, 1)}

    DownloadManager._validate_staged_plan(plan, inspection, hashes)  # type: ignore[arg-type]

    mismatched = ModelManifestInspection(
        architecture=None,
        family=None,
        components=(
            InspectedComponent(
                path="model.safetensors",
                kind="lora",
                target_folder="loras",
            ),
            *inspection.components[1:],
        ),
        metadata_files=(),
    )
    with pytest.raises(ValueError, match="contract changed"):
        DownloadManager._validate_staged_plan(plan, mismatched, hashes)  # type: ignore[arg-type]

    standalone = SimpleNamespace(
        family=None,
        role="image",
        artifacts_json=[
            {
                "path": item.path,
                "kind": item.kind,
                "target_folder": item.target_folder,
                "required": True,
            }
            for item in inspection.components
        ],
        runtime_contract_json={
            "auxiliary_kind": None,
            "workflow_template_id": None,
        },
    )
    with pytest.raises(ValueError, match="unsupported"):
        DownloadManager._validate_staged_plan(  # type: ignore[arg-type]
            standalone,
            inspection,
            hashes,
        )


async def test_planned_chat_activation_requires_completion_and_records_evidence(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()

    class FakeChatAdapter:
        async def capabilities(self) -> object:
            return SimpleNamespace(
                healthy=True,
                version="llama-test",
                input_modalities=["text", "image"],
            )

        async def count_tokens(self, _messages: list[dict[str, Any]]) -> int:
            return 4

        async def stream(self, request: ChatRequest):  # type: ignore[no-untyped-def]
            assert request.settings["max_tokens"] == 8
            content = request.messages[0]["content"]
            assert isinstance(content, list)
            assert content[1]["type"] == "image_url"
            yield ChatEvent(type="token", text="OK")
            yield ChatEvent(type="complete")

    class FakeProcesses:
        def __init__(self) -> None:
            self.loaded: list[str] = []
            self.stopped: list[str] = []

        def statuses(self) -> list[object]:
            return [
                SimpleNamespace(
                    name="chat",
                    running=False,
                    profile_id=None,
                )
            ]

        async def load_chat(
            self,
            profile: ModelProfile,
            _install: ModelInstall,
        ) -> None:
            self.loaded.append(profile.id)

        async def stop(self, name: str) -> None:
            self.stopped.append(name)

    processes = FakeProcesses()

    def active_adapter() -> FakeChatAdapter:
        assert processes.loaded
        return FakeChatAdapter()

    manager = DownloadManager(
        settings,
        EventBroker(),
        chat_adapter=active_adapter,
        processes=processes,  # type: ignore[arg-type]
    )
    with SessionLocal() as session:
        install = ModelInstall(
            id="model_planned_chat",
            name="Planned chat",
            role="chat",
            engine="llama.cpp",
            local_path=str(settings.model_dir / "planned-chat"),
            manifest_json={"files": ["model.gguf", "mmproj-model.gguf"]},
            active=False,
        )
        profile = ModelProfile(
            id="profile_planned_chat",
            model_install_id=install.id,
            name="Planned chat",
            role="chat",
            engine="llama.cpp",
        )
        session.add_all(
            [
                install,
                profile,
                Job(
                    id="job_planned_chat",
                    kind=JobKind.DOWNLOAD.value,
                    status=JobStatus.RUNNING.value,
                ),
            ]
        )
        session.commit()

    result = await manager._activate_chat_install(
        job_id="job_planned_chat",
        install_id="model_planned_chat",
        default_settings={},
        component_hashes={"model.gguf": "a" * 64},
    )

    assert result == "profile_planned_chat"
    assert len(processes.loaded) == 1
    assert processes.stopped == ["chat"]
    with SessionLocal() as session:
        assert session.get(ModelInstall, "model_planned_chat").active is True  # type: ignore[union-attr]
        evidence = session.query(ModelCapabilityEvidence).one()
        assert evidence.result == "ready"
        assert evidence.runtime_build == "llama-test"
        assert evidence.details_json["input_modalities"] == ["text", "image"]
        assert evidence.details_json["projector_expected"] is True
        assert evidence.details_json["vision_probe"] == "bounded_image_completion"


async def test_failed_chat_probe_restores_the_previous_profile(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()

    class FailingChatAdapter:
        async def capabilities(self) -> object:
            return SimpleNamespace(
                healthy=True,
                version="llama-test",
                input_modalities=["text"],
            )

        async def count_tokens(self, _messages: list[dict[str, Any]]) -> int:
            return 4

        async def stream(self, _request: ChatRequest):  # type: ignore[no-untyped-def]
            raise RuntimeError("probe completion failed")
            yield ChatEvent(type="complete")

    class RestoringProcesses:
        def __init__(self) -> None:
            self.loaded: list[str] = []

        def statuses(self) -> list[object]:
            return [
                SimpleNamespace(
                    name="chat",
                    running=True,
                    profile_id="profile_previous",
                )
            ]

        async def load_chat(
            self,
            profile: ModelProfile,
            _install: ModelInstall,
        ) -> None:
            self.loaded.append(profile.id)

        async def stop(self, _name: str) -> None:
            raise AssertionError("a working prior profile should be restored")

    processes = RestoringProcesses()
    manager = DownloadManager(
        settings,
        EventBroker(),
        chat_adapter=FailingChatAdapter(),  # type: ignore[arg-type]
        processes=processes,  # type: ignore[arg-type]
    )
    with SessionLocal() as session:
        previous = ModelInstall(
            id="model_previous",
            name="Previous",
            role="chat",
            engine="llama.cpp",
            local_path=str(settings.model_dir / "previous"),
            manifest_json={"files": ["previous.gguf"]},
            active=True,
        )
        candidate = ModelInstall(
            id="model_candidate",
            name="Candidate",
            role="chat",
            engine="llama.cpp",
            local_path=str(settings.model_dir / "candidate"),
            manifest_json={"files": ["candidate.gguf"]},
            active=False,
        )
        session.add_all(
            [
                previous,
                candidate,
                ModelProfile(
                    id="profile_previous",
                    model_install_id=previous.id,
                    name="Previous",
                    role="chat",
                    engine="llama.cpp",
                ),
                Job(
                    id="job_candidate",
                    kind=JobKind.DOWNLOAD.value,
                    status=JobStatus.RUNNING.value,
                ),
            ]
        )
        session.commit()

    with pytest.raises(RuntimeError, match="probe completion failed"):
        await manager._activate_chat_install(
            job_id="job_candidate",
            install_id="model_candidate",
            default_settings={},
            component_hashes={"candidate.gguf": "b" * 64},
        )

    assert len(processes.loaded) == 2
    assert processes.loaded[-1] == "profile_previous"
    with SessionLocal() as session:
        assert session.get(ModelInstall, "model_candidate").active is False  # type: ignore[union-attr]
        assert session.query(ModelCapabilityEvidence).count() == 0


async def test_unknown_gguf_plan_installs_and_activates_with_one_request(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    content = gguf_bytes("future_architecture")
    digest = hashlib.sha256(content).hexdigest()
    inspection = inspect_repository_metadata(
        {"weights.bin.gguf": content},
        ["weights.bin.gguf"],
        role="chat",
    )
    resolved = resolve_install_plan(
        remote_id="synthetic/future-chat",
        revision="c" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": "weights.bin.gguf",
                "size": len(content),
                "sha256": digest,
            }
        ],
        inspection=inspection,
    )
    with SessionLocal() as session:
        plan = persist_install_plan(session, resolved)
        session.commit()
        request = DownloadRequest(
            install_plan_id=plan.id,
            remote_id=plan.remote_id,
            revision=plan.revision,
            role=plan.role,  # type: ignore[arg-type]
            engine=plan.engine,
            allow_patterns=["weights.bin.gguf"],
            expected_sha256={"weights.bin.gguf": digest},
        )
        job = Job(
            id="job_future_chat",
            kind=JobKind.DOWNLOAD.value,
            status=JobStatus.QUEUED.value,
            payload_json=request.model_dump(mode="json"),
        )
        session.add(job)
        session.commit()

    class ChatAdapter:
        async def capabilities(self) -> object:
            return SimpleNamespace(
                healthy=True,
                version="llama-future",
                input_modalities=["text"],
            )

        async def count_tokens(self, _messages: list[dict[str, Any]]) -> int:
            return 3

        async def stream(self, _request: ChatRequest):  # type: ignore[no-untyped-def]
            yield ChatEvent(type="token", text="OK")
            yield ChatEvent(type="complete")

    class Processes:
        runtimes = None

        def statuses(self) -> list[object]:
            return [SimpleNamespace(name="chat", running=False, profile_id=None)]

        async def load_chat(
            self,
            _profile: ModelProfile,
            _install: ModelInstall,
        ) -> None:
            return None

        async def stop(self, _name: str) -> None:
            return None

    manager = DownloadManager(
        settings,
        EventBroker(),
        chat_adapter=ChatAdapter(),  # type: ignore[arg-type]
        processes=Processes(),  # type: ignore[arg-type]
    )
    manager._api = SimpleNamespace(
        model_info=lambda *_args, **_kwargs: SimpleNamespace(
            siblings=[
                SimpleNamespace(
                    rfilename="weights.bin.gguf",
                    size=len(content),
                    lfs={"sha256": digest},
                )
            ],
            sha="c" * 40,
            pipeline_tag="text-generation",
            tags=["gguf"],
            gated=False,
        )
    )  # type: ignore[assignment]

    async def download_file(**kwargs: Any) -> str:
        target = kwargs["staging"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    monkeypatch.setattr(manager, "_download_file", download_file)
    await manager._download("job_future_chat")

    with SessionLocal() as session:
        completed = session.get(Job, "job_future_chat")
        installs = session.query(ModelInstall).all()
        assert completed
        assert completed.status == JobStatus.COMPLETE.value, completed.error
        assert len(installs) == 1
        assert installs[0].active is True
        assert session.query(ModelProfile).count() == 1
        assert session.query(ModelComponentManifest).count() == 1
        assert session.query(ModelCapabilityEvidence).count() == 1


async def test_chat_plan_downloads_a_pinned_projector_from_a_companion_repo(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    model_content = gguf_bytes("qwen")
    projector_content = gguf_bytes("clip")
    model_digest = hashlib.sha256(model_content).hexdigest()
    projector_digest = hashlib.sha256(projector_content).hexdigest()
    model_name = "Descriptive-Qwen-27B-Q4_K_M.gguf"
    source_projector_name = "mmproj-Descriptive-Qwen-27B-f16.gguf"
    projector_name = f"companions/author/model/{source_projector_name}"
    inspection = inspect_repository_metadata(
        {
            model_name: model_content,
            projector_name: projector_content,
        },
        [model_name, projector_name],
        role="chat",
    )
    resolved = resolve_install_plan(
        remote_id="converter/model",
        revision="a" * 40,
        role="chat",
        engine="llama.cpp",
        selected_files=[
            {
                "filename": model_name,
                "size": len(model_content),
                "sha256": model_digest,
                "metadata": {"trained_words": ["visionary-style"]},
            },
            {
                "filename": projector_name,
                "size": len(projector_content),
                "sha256": projector_digest,
                "source_remote_id": "author/model",
                "source_revision": "b" * 40,
                "source_filename": source_projector_name,
            },
        ],
        inspection=inspection,
    )
    with SessionLocal() as session:
        plan = persist_install_plan(session, resolved)
        session.commit()
        request = DownloadRequest(
            install_plan_id=plan.id,
            remote_id=plan.remote_id,
            revision=plan.revision,
            role="chat",
            engine=plan.engine,
            content_rating="general",
            allow_patterns=[model_name, projector_name],
            expected_sha256={
                model_name: model_digest,
                projector_name: projector_digest,
            },
            file_sources={
                projector_name: CatalogFileSource(
                    remote_id="author/model",
                    revision="b" * 40,
                    filename=source_projector_name,
                    size_bytes=len(projector_content),
                    sha256=projector_digest,
                )
            },
        )
        session.add(
            Job(
                id="job_companion_projector",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.QUEUED.value,
                payload_json=request.model_dump(mode="json"),
            )
        )
        session.commit()

    class ChatAdapter:
        async def capabilities(self) -> object:
            return SimpleNamespace(
                healthy=True,
                version="llama-vision",
                input_modalities=["text", "image"],
            )

        async def count_tokens(self, _messages: list[dict[str, Any]]) -> int:
            return 3

        async def stream(self, _request: ChatRequest):  # type: ignore[no-untyped-def]
            yield ChatEvent(type="token", text="OK")
            yield ChatEvent(type="complete")

    class Processes:
        runtimes = None

        def statuses(self) -> list[object]:
            return [SimpleNamespace(name="chat", running=False, profile_id=None)]

        async def load_chat(
            self,
            _profile: ModelProfile,
            _install: ModelInstall,
        ) -> None:
            return None

        async def stop(self, _name: str) -> None:
            return None

    def model_info(remote_id: str, **_kwargs: object) -> object:
        if remote_id == "converter/model":
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(
                        rfilename=model_name,
                        size=len(model_content),
                        lfs={"sha256": model_digest},
                    )
                ],
                sha="a" * 40,
                pipeline_tag="text-generation",
                tags=["gguf"],
                gated=False,
            )
        assert remote_id == "author/model"
        return SimpleNamespace(
            siblings=[
                SimpleNamespace(
                    rfilename=source_projector_name,
                    size=len(projector_content),
                    lfs={"sha256": projector_digest},
                )
            ],
            sha="b" * 40,
            pipeline_tag="image-text-to-text",
            tags=["vision"],
            gated=False,
        )

    manager = DownloadManager(
        settings,
        EventBroker(),
        chat_adapter=ChatAdapter(),  # type: ignore[arg-type]
        processes=Processes(),  # type: ignore[arg-type]
    )
    manager._api = SimpleNamespace(model_info=model_info)  # type: ignore[assignment]
    transfers: list[tuple[str, str, str]] = []

    async def download_file(**kwargs: Any) -> str:
        transfers.append((kwargs["remote_id"], kwargs["revision"], kwargs["filename"]))
        content = projector_content if kwargs["remote_id"] == "author/model" else model_content
        target = kwargs["staging"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    monkeypatch.setattr(manager, "_download_file", download_file)
    await manager._download("job_companion_projector")

    assert transfers == [
        ("converter/model", "a" * 40, model_name),
        ("author/model", "b" * 40, source_projector_name),
    ]
    with SessionLocal() as session:
        job = session.get(Job, "job_companion_projector")
        install = session.query(ModelInstall).one()
        assert job and job.status == JobStatus.COMPLETE.value, job.error if job else None
        assert install.active is True
        assert install.manifest_json["file_sources"][projector_name]["remote_id"] == "author/model"
        assert install.manifest_json["trigger_words"] == ["visionary-style"]
        # The declared rating travels into the manifest so provenance is honest.
        assert install.manifest_json["content_rating"] == "general"
        components = {
            component.relative_path for component in session.query(ModelComponentManifest)
        }
        assert components == {model_name, projector_name}
        assert session.query(ModelCapabilityEvidence).one().result == "ready"


def test_companion_relocation_rejects_a_worker_path_outside_staging(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    expected = staging / "mmproj-model.gguf"
    expected.write_bytes(b"expected")
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="unexpected local path"):
        DownloadManager._relocate_companion_download(
            staging=staging,
            source_filename="mmproj-model.gguf",
            destination_filename="companions/author/model/mmproj-model.gguf",
            downloaded_path=str(outside),
        )

    relocated = DownloadManager._relocate_companion_download(
        staging=staging,
        source_filename="mmproj-model.gguf",
        destination_filename="companions/author/model/mmproj-model.gguf",
        downloaded_path=str(expected),
    )
    assert Path(relocated).read_bytes() == b"expected"
    assert not expected.exists()


async def test_lora_plan_installs_as_a_verified_auxiliary_asset(
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
                "metadata": {"trained_words": ["provider ink"]},
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

        async def start_media(self, model_root: tuple[Path, dict[str, str]]) -> None:
            self.started.append(model_root)

        async def stop(self, name: str) -> None:
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
        media_adapter=MediaAdapter(),  # type: ignore[arg-type]
        processes=processes,  # type: ignore[arg-type]
    )
    manager._api = SimpleNamespace(
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
            tags=["lora"],
            gated=False,
        )
    )  # type: ignore[assignment]

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
        assert asset.manifest_json["sha256"] == digest
        assert asset.manifest_json["comfy_name"] == "adapter.safetensors"
        assert asset.manifest_json["metadata"]["trigger_words"] == [
            "atelier ink",
            "provider ink",
        ]
        # A request that declares no rating stores the explicit unknown.
        assert asset.manifest_json["content_rating"] == "unknown"
        assert stored_plan and stored_plan.status == "activated"
    assert processes.started[0][1] == {"loras": "."}
    assert processes.stopped == ["media"]


async def test_workflow_checkpoint_installs_as_an_inert_verified_asset(
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

        async def start_media(self, model_root: tuple[Path, dict[str, str]]) -> None:
            self.started.append(model_root)

        async def stop(self, name: str) -> None:
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
        media_adapter=MediaAdapter(),  # type: ignore[arg-type]
        processes=processes,  # type: ignore[arg-type]
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
        assert asset.manifest_json["comfy_name"] == filename
        assert asset.manifest_json["workflow_asset_kind"] == "checkpoint"
        assert session.query(ModelInstall).count() == 0
        assert stored_plan and stored_plan.status == "activated"
    assert processes.started[0][0].name.endswith(f"-asset-{plan_hash[:12]}")
    assert processes.started[0][1] == {"checkpoints": "."}
    assert processes.stopped == ["media"]


def test_duplicate_active_download_requests_reuse_the_existing_job(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    manager = DownloadManager(settings, EventBroker())
    monkeypatch.setattr(manager, "start", lambda _job_id: None)
    request = DownloadRequest(
        remote_id="owner/model",
        revision="abc123",
        role="chat",
        engine="llama.cpp",
        allow_patterns=["model-Q4_K_M.gguf"],
    )
    starts: list[str] = []
    with SessionLocal() as session:
        monkeypatch.setattr(manager, "start", starts.append)
        first = manager.create(session, request)
        second = manager.create(session, request)
        first.status = JobStatus.PAUSED.value
        session.commit()
        starts.clear()
        paused = manager.create(session, request)
        jobs = list(session.query(Job).all())
    assert second.id == first.id
    assert paused.id == first.id
    assert starts == []
    assert len(jobs) == 1


def test_managed_model_directory_is_short_and_stable() -> None:
    remote_id = "owner/" + ("very-descriptive-model-name-" * 8)
    revision = "a" * 40

    first = DownloadManager._install_directory_name(remote_id, revision)
    second = DownloadManager._install_directory_name(remote_id, revision)
    different = DownloadManager._install_directory_name(remote_id, "b" * 40)

    assert first == second
    assert first != different
    assert len(first) == 24


@pytest.mark.parametrize(
    "filename",
    ["../model.gguf", "/model.gguf", "C:/model.gguf", "folder\\model.gguf", ""],
)
def test_download_staging_rejects_unsafe_relative_filenames(filename: str) -> None:
    assert DownloadManager._safe_relative_filename(filename) is False


def test_staged_model_family_must_match_the_immutable_plan() -> None:
    inspection = inspect_repository_metadata(
        {"model.gguf": gguf_bytes("llama")},
        ["model.gguf"],
        role="chat",
    )
    plan = SimpleNamespace(
        family="qwen",
        artifacts_json=[
            {
                "path": "model.gguf",
                "kind": "gguf_model",
                "target_folder": "models",
                "required": True,
            }
        ],
    )

    with pytest.raises(ValueError, match="family"):
        DownloadManager._validate_staged_plan(  # type: ignore[arg-type]
            plan,
            inspection,
            {"model.gguf": "a" * 64},
        )


async def test_stop_task_terminates_transfer_process_before_controller(
    settings: Settings,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    worker = FakeWorker()
    controller_started = asyncio.Event()

    async def controller() -> None:
        controller_started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(controller())
    await controller_started.wait()
    manager._tasks["job_test"] = task
    manager._workers["job_test"] = worker  # type: ignore[assignment]

    await manager._stop_task("job_test")

    assert worker.terminated is True
    assert task.cancelled() is True


async def test_download_worker_receives_token_over_stdin_not_command_line(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    manager.settings.hf_token = "secret-token"
    workers: list[FakeCompletedWorker] = []
    environments: list[dict[str, str]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> FakeCompletedWorker:
        worker = FakeCompletedWorker(command)
        workers.append(worker)
        environments.append(kwargs["env"])
        return worker

    monkeypatch.setenv("LOCAL_LM_HF_TOKEN", "inherited-token")
    monkeypatch.setenv("GITHUB_TOKEN", "inherited-github-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "inherited-cloud-token")
    monkeypatch.setattr("local_lm.downloads.subprocess.Popen", fake_popen)
    path = await manager._download_file(
        job_id="job_test",
        remote_id="owner/model",
        filename="model.gguf",
        revision="a" * 40,
        staging=Path("C:/staging"),
    )

    assert path == "C:/models/model.gguf"
    assert workers[0].command[-2:] == ["-m", "local_lm.download_worker"]
    assert "secret-token" not in " ".join(workers[0].command)
    assert json.loads(workers[0].payload)["kind"] == "huggingface"
    assert json.loads(workers[0].payload)["token"] == "secret-token"
    assert environments[0]["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert "LOCAL_LM_HF_TOKEN" not in environments[0]
    assert "GITHUB_TOKEN" not in environments[0]
    assert "AWS_SECRET_ACCESS_KEY" not in environments[0]
    assert "job_test" not in manager._workers


def _civitai_plan(*, artifact: dict[str, Any]) -> InstallPlan:
    return InstallPlan(
        id="plan_civitai",
        provider="civitai",
        remote_id="202",
        revision="202",
        role="image",
        engine="comfyui",
        plan_hash="f" * 64,
        resolver_version="test",
        compatibility="ready",
        artifacts_json=[artifact],
        runtime_contract_json={},
        activation_probe_json={},
    )


def _civitai_request(*, digest: str) -> DownloadRequest:
    return DownloadRequest(
        install_plan_id="plan_civitai",
        remote_id="202",
        revision="202",
        role="image",
        engine="comfyui",
        allow_patterns=["model.safetensors"],
        expected_sha256={"model.safetensors": digest},
        comfy_paths={"checkpoints": "."},
    )


async def test_civitai_sources_come_only_from_the_immutable_plan(
    settings: Settings,
) -> None:
    digest = "a" * 64
    plan = _civitai_plan(
        artifact={
            "path": "model.safetensors",
            "required": True,
            "size_bytes": 17,
            "sha256": digest,
            "source_version_id": "202",
            "source_file_id": "303",
        }
    )
    manager = DownloadManager(settings, EventBroker())
    manager._api = SimpleNamespace(
        model_info=lambda *_args, **_kwargs: pytest.fail(
            "CivitAI provenance must not be re-resolved through Hugging Face"
        )
    )  # type: ignore[assignment]

    siblings, sources, revision, metadata = await manager._download_sources(
        _civitai_request(digest=digest),
        plan,
    )

    assert revision == "202"
    assert metadata == {"source_version_id": "202"}
    assert [(item.rfilename, item.size, item.lfs) for item in siblings] == [
        ("model.safetensors", 17, {"sha256": digest})
    ]
    source = sources["model.safetensors"]
    assert source.provider == "civitai"
    assert source.remote_id == "202"
    assert source.revision == "202"
    assert source.filename == "model.safetensors"
    assert source.source_file_id == "303"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("size_bytes", None),
        ("size_bytes", True),
        ("sha256", "A" * 64),
        ("source_version_id", "latest"),
        ("source_file_id", ""),
    ],
)
def test_civitai_sources_reject_incomplete_provenance(
    settings: Settings,
    field: str,
    value: object,
) -> None:
    digest = "a" * 64
    artifact: dict[str, Any] = {
        "path": "model.safetensors",
        "required": True,
        "size_bytes": 17,
        "sha256": digest,
        "source_version_id": "202",
        "source_file_id": "303",
    }
    artifact[field] = value
    request_digest = str(artifact["sha256"])
    manager = DownloadManager(settings, EventBroker())

    with pytest.raises(ValueError, match="provenance is incomplete"):
        manager._civitai_download_sources(
            _civitai_request(digest=request_digest),
            _civitai_plan(artifact=artifact),
        )


async def test_civitai_download_worker_receives_a_verified_redacted_envelope(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    manager.settings.civitai_token = "secret-civitai-token"
    workers: list[FakeCompletedWorker] = []
    environments: list[dict[str, str]] = []

    def fake_popen(command: list[str], **kwargs: Any) -> FakeCompletedWorker:
        worker = FakeCompletedWorker(command)
        workers.append(worker)
        environments.append(kwargs["env"])
        return worker

    monkeypatch.setenv("CIVITAI_TOKEN", "inherited-civitai-token")
    monkeypatch.setenv("GITHUB_TOKEN", "inherited-github-token")
    monkeypatch.setattr("local_lm.downloads.subprocess.Popen", fake_popen)
    path = await manager._download_file(
        job_id="job_civitai",
        provider="civitai",
        remote_id="202",
        revision="202",
        source_file_id="303",
        filename="model.safetensors",
        expected_sha256="a" * 64,
        file_size=17,
        staging=Path("C:/staging"),
    )

    assert path == "C:/models/model.gguf"
    assert "secret-civitai-token" not in " ".join(workers[0].command)
    assert json.loads(workers[0].payload) == {
        "kind": "https",
        "url": "https://civitai.com/api/download/models/202?fileId=303",
        "filename": "model.safetensors",
        "local_dir": str(Path("C:/staging")),
        "expected_sha256": "a" * 64,
        "expected_size": 17,
        # civitai.com redirects, b2 serves the small files, and anything large
        # comes from their Cloudflare R2 delivery domain.
        "allowed_hosts": ["civitai.com", "b2.civitai.com", ".r2.cloudflarestorage.com"],
        "bearer_token": "secret-civitai-token",
    }
    assert "CIVITAI_TOKEN" not in environments[0]
    assert "GITHUB_TOKEN" not in environments[0]
    assert "job_civitai" not in manager._workers


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_file_id", "latest", "exact immutable file provenance"),
        ("file_size", 0, "exact immutable file provenance"),
        ("expected_sha256", "A" * 64, "exact immutable file provenance"),
    ],
)
async def test_civitai_worker_rejects_unverified_inputs_before_spawn(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    manager.settings.civitai_token = "secret-civitai-token"
    values: dict[str, Any] = {
        "job_id": "job_civitai",
        "provider": "civitai",
        "remote_id": "202",
        "revision": "202",
        "source_file_id": "303",
        "filename": "model.safetensors",
        "expected_sha256": "a" * 64,
        "file_size": 17,
        "staging": Path("C:/staging"),
    }
    values[field] = value
    monkeypatch.setattr(
        "local_lm.downloads.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not spawn a worker"),
    )

    with pytest.raises(ValueError, match=message):
        await manager._download_file(**values)


async def test_civitai_worker_requires_a_configured_credential(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    manager.settings.civitai_token = None
    monkeypatch.setattr(
        "local_lm.downloads.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("missing credentials must not spawn a worker"),
    )

    with pytest.raises(ValueError, match="credential is not configured"):
        await manager._download_file(
            job_id="job_civitai",
            provider="civitai",
            remote_id="202",
            revision="202",
            source_file_id="303",
            filename="model.safetensors",
            expected_sha256="a" * 64,
            file_size=17,
            staging=Path("C:/staging"),
        )


async def test_planned_components_use_one_bounded_parallel_worker(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    workers: list[FakeBatchCompletedWorker] = []

    def fake_popen(command: list[str], **_kwargs: Any) -> FakeBatchCompletedWorker:
        worker = FakeBatchCompletedWorker(command)
        workers.append(worker)
        return worker

    monkeypatch.setattr("local_lm.downloads.subprocess.Popen", fake_popen)
    paths = await manager._download_files_parallel(
        job_id="job_batch",
        remote_id="owner/model",
        filenames=["model.gguf", "mmproj.gguf"],
        revision="a" * 40,
        staging=Path("C:/staging"),
        completed_bytes=0,
        total_size=None,
        batch_size=20,
        bytes_reused=0,
    )

    payload = json.loads(workers[0].payload)
    assert payload["kind"] == "huggingface"
    assert payload["files"] == ["model.gguf", "mmproj.gguf"]
    assert payload["max_workers"] == 2
    assert paths == {
        "model.gguf": "C:/models/model.gguf",
        "mmproj.gguf": "C:/models/mmproj.gguf",
    }
    assert "job_batch" not in manager._workers


async def test_download_file_retries_transient_worker_failures(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DownloadManager(settings, EventBroker())
    attempts = 0
    sleeps: list[int] = []

    async def download_once(**_kwargs: Any) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("incomplete HTTP read")
        return "C:/models/model.gguf"

    async def sleep(seconds: int) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(manager, "_download_file_once", download_once)
    monkeypatch.setattr(asyncio, "sleep", sleep)

    path = await manager._download_file(
        job_id="job_test",
        remote_id="owner/model",
        filename="model.gguf",
        revision="a" * 40,
        staging=Path("C:/staging"),
    )

    assert path == "C:/models/model.gguf"
    assert attempts == 3
    assert sleeps == [1, 2]


async def test_verified_installed_component_is_reused_without_network_transfer(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    manager = DownloadManager(settings, EventBroker())
    content = b"verified reusable model bytes"
    digest = hashlib.sha256(content).hexdigest()
    installed = settings.model_dir / "existing"
    installed.mkdir(parents=True)
    (installed / "model.gguf").write_bytes(content)
    staging = settings.download_dir / f"plan-{'a' * 64}.partial"
    staging.mkdir(parents=True)
    (staging / "model.gguf").write_bytes(b"corrupt")
    with SessionLocal() as session:
        install = ModelInstall(
            id="model_reuse",
            name="Reusable",
            role="chat",
            engine="llama.cpp",
            local_path=str(installed),
            manifest_json={"files": ["model.gguf"]},
            active=True,
        )
        session.add(install)
        session.flush()
        session.add(
            ModelComponentManifest(
                model_install_id=install.id,
                kind="gguf_model",
                relative_path="model.gguf",
                target_folder="models",
                sha256=digest,
                size_bytes=len(content),
                required=True,
            )
        )
        session.commit()

        candidates = manager._verified_reuse_candidates(
            session,
            staging=staging,
            filename="model.gguf",
            expected_sha256=digest,
        )

    reused = await manager._reuse_verified_file(
        candidates=candidates,
        staging=staging,
        filename="model.gguf",
        expected_sha256=digest,
        expected_size=len(content),
    )

    assert reused == (staging / "model.gguf", len(content))
    assert (staging / "model.gguf").read_bytes() == content


async def test_verified_failed_plan_component_is_reused_by_a_new_plan(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    manager = DownloadManager(settings, EventBroker())
    content = b"verified failed plan bytes"
    digest = hashlib.sha256(content).hexdigest()
    previous = settings.download_dir / f"plan-{'b' * 64}.partial"
    previous.mkdir(parents=True)
    (previous / "weights").mkdir()
    source = previous / "weights" / "model.safetensors"
    source.write_bytes(content)
    staging = settings.download_dir / f"plan-{'a' * 64}.partial"
    staging.mkdir(parents=True)

    with SessionLocal() as session:
        candidates = manager._verified_reuse_candidates(
            session,
            staging=staging,
            filename="weights/model.safetensors",
            expected_sha256=digest,
        )

    assert source in candidates
    reused = await manager._reuse_verified_file(
        candidates=candidates,
        staging=staging,
        filename="weights/model.safetensors",
        expected_sha256=digest,
        expected_size=len(content),
    )

    target = staging / "weights" / "model.safetensors"
    assert reused == (target, len(content))
    assert target.read_bytes() == content
    assert source.read_bytes() == content


async def test_transfer_monitor_reports_process_tree_bytes(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    broker = EventBroker()
    manager = DownloadManager(settings, broker)
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_progress",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.RUNNING.value,
                progress=0,
                phase="inspecting",
            )
        )
        session.commit()

    staging = settings.download_dir / "job_progress.partial"
    staging.mkdir(parents=True)
    write_samples = iter([0, 50])
    monkeypatch.setattr(
        manager,
        "_process_tree_write_bytes",
        lambda _pid: next(write_samples, 50),
    )
    stop = asyncio.Event()

    async def stop_after_sample() -> None:
        await asyncio.sleep(0.01)
        stop.set()

    stopper = asyncio.create_task(stop_after_sample())
    await manager._monitor_transfer(
        job_id="job_progress",
        filename="model.safetensors",
        staging=staging,
        process=SimpleNamespace(pid=123),  # type: ignore[arg-type]
        file_size=100,
        completed_bytes=0,
        total_size=100,
        stop=stop,
    )
    await stopper

    with SessionLocal() as session:
        job = session.get(Job, "job_progress")
        assert job
        assert job.phase == "downloading model.safetensors"
        assert job.progress == pytest.approx(0.5)
        assert job.progress_json["completed_units"] == 50
        assert job.progress_json["total_units"] == 100
        assert job.progress_json["unit"] == "bytes"
    event = next(event for event in broker.since(0) if event.type == "download.progress")
    assert event.payload["downloaded_bytes"] == 50
    assert event.payload["file_size_bytes"] == 100
    assert event.payload["total_bytes"] == 100


async def test_adaptive_checkpoint_activation_runs_a_small_bounded_generation(
    settings: Settings,
) -> None:
    adapter = FakeProbeAdapter()
    manager = DownloadManager(
        settings,
        EventBroker(),
        media_adapter=adapter,  # type: ignore[arg-type]
    )
    graph = {"loader": {"class_type": "CheckpointLoaderSimple", "inputs": {}}}

    await manager._probe_adaptive_checkpoint(  # type: ignore[arg-type]
        SimpleNamespace(api_graph=graph)
    )

    assert adapter.request
    assert adapter.request.workflow == graph
    assert adapter.request.operation == "text_to_image"
    assert adapter.request.parameters == {
        "width": 256,
        "height": 256,
        "batch_size": 1,
        "seed": 0,
        "steps": 1,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "normal",
        "denoise": 1.0,
    }
    assert adapter.timeout_seconds == 300


async def test_native_edit_activation_uses_ephemeral_inputs_for_each_loader(
    settings: Settings,
) -> None:
    adapter = FakeProbeAdapter()
    manager = DownloadManager(
        settings,
        EventBroker(),
        media_adapter=adapter,  # type: ignore[arg-type]
    )
    graph = {
        "first": {"class_type": "LoadImage", "inputs": {"image": "${input_image_0}"}},
        "second": {"class_type": "LoadImage", "inputs": {"image": "${input_image_1}"}},
    }
    compiled = SimpleNamespace(
        api_graph=graph,
        input_schema={
            "properties": {
                "input_image_0": {"type": "string"},
                "input_image_1": {"type": "string"},
            }
        },
        template=SimpleNamespace(operation="image_to_image"),
    )

    await manager._probe_adaptive_checkpoint(compiled)  # type: ignore[arg-type]

    assert adapter.request
    assert adapter.request.operation == "image_to_image"
    assert len(adapter.request.input_paths) == 2
    assert adapter.input_contents == [adapter.input_contents[0]] * 2
    assert adapter.input_contents[0].startswith(b"\x89PNG\r\n\x1a\n")
    assert not adapter.request.input_paths[0].exists()


async def test_workflow_refresh_adds_an_image_edit_contract_for_existing_installs(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    object_info = {"LoadImage": {"input": {}}, "VAEEncode": {"input": {}}}

    class MediaAdapter:
        async def object_info(self) -> dict[str, object]:
            return object_info

    dependency = ComfyModelDependency(
        remote_id="Comfy-Org/z_image_turbo",
        revision="main",
        path="z_image.safetensors",
        directory="diffusion_models",
        name="z_image.safetensors",
        url="",
    )
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="image_z_image_turbo",
            path=settings.data_dir / "template.json",
            role="image",
            operation="text_to_image",
            score=1_000,
            sha256="a" * 64,
            dependencies=(dependency,),
        ),
        ui_graph={"nodes": []},
        api_graph={
            "empty": {"class_type": "EmptySD3LatentImage", "inputs": {}},
            "vae": {"class_type": "VAELoader", "inputs": {}},
            "sampler": {
                "class_type": "KSampler",
                "inputs": {"latent_image": ["empty", 0], "denoise": "${denoise}"},
            },
            "decode": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]},
            },
        },
        input_schema={"type": "object", "properties": {"denoise": {"default": 1.0}}},
    )
    manager = DownloadManager(
        settings,
        EventBroker(),
        media_adapter=MediaAdapter(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(manager.comfy_templates, "compile", lambda *_args, **_kwargs: compiled)
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="model_z_image_existing",
                name="Z-Image Turbo",
                role="image",
                engine="comfyui",
                local_path=str(settings.model_dir / "z-image"),
                manifest_json={
                    "workflow_template_id": "image_z_image_turbo",
                    "workflow_template_sha256": "a" * 64,
                    "remote_id": "Comfy-Org/z_image_turbo",
                    "revision": "main",
                    "files": ["z_image.safetensors"],
                    "comfy_paths": {"diffusion_models": "."},
                },
                active=True,
            )
        )
        legacy_definition = WorkflowDefinition(
            name="ComfyUI template · image_z_image_turbo_image_to_image",
            operation="image_to_image",
            description="Existing generated workflow",
        )
        session.add(legacy_definition)
        session.flush()
        legacy_revision = WorkflowRevision(
            workflow_id=legacy_definition.id,
            version=1,
            engine="comfyui",
            ui_graph_json={"legacy": True},
            api_graph_json={"legacy": {"class_type": "Legacy"}},
            input_schema_json={
                "type": "object",
                "properties": {"denoise": {"type": "number", "default": 0.9}},
            },
            dependencies_json={
                "model_install_ids": ["model_z_image_existing"],
                "compiler_version": "legacy",
                "template_sha256": "b" * 64,
            },
            trusted=True,
        )
        session.add(legacy_revision)
        session.flush()
        legacy_definition.current_revision_id = legacy_revision.id
        legacy_revision_id = legacy_revision.id
        session.commit()
    assert await manager.refresh_installed_media_workflows() == 2
    with SessionLocal() as session:
        definitions = session.query(WorkflowDefinition).all()
        assert {definition.operation for definition in definitions} == {
            "text_to_image",
            "image_to_image",
        }
        edit = next(item for item in definitions if item.operation == "image_to_image")
        revision = session.get(WorkflowRevision, edit.current_revision_id)
        assert revision is not None
        revisions = sorted(edit.revisions, key=lambda item: item.version)
        assert [item.version for item in revisions] == [1, 2]
        assert revisions[0].id == legacy_revision_id
        assert revisions[0].api_graph_json == {"legacy": {"class_type": "Legacy"}}
        assert "x-lm-atelier-edit-calibration" not in revisions[0].input_schema_json
        assert revision.dependencies_json["model_install_ids"] == ["model_z_image_existing"]
        assert revision.api_graph_json["lma-load-image"]["inputs"]["image"] == ("${input_image}")
        assert revision.input_schema_json["x-lm-atelier-edit-calibration"]["version"] == 1


async def test_media_activation_waits_for_the_shared_compute_lease(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    scheduler = ResourceScheduler()
    started = asyncio.Event()

    class FakeProcesses:
        async def start_media(self, _model_paths: object = None) -> None:
            started.set()

    class FakeMediaAdapter:
        async def object_info(self) -> dict[str, object]:
            return {}

        async def validate_workflow(self, _graph: dict[str, object]) -> list[str]:
            return []

    compiled = SimpleNamespace(
        template=SimpleNamespace(
            id="lease-image",
            operation="text_to_image",
            runtime_adaptive=False,
            selected_files=["model.safetensors"],
            component_folders={"model.safetensors": "checkpoints"},
            sha256="a" * 64,
        ),
        ui_graph={},
        api_graph={"loader": {"class_type": "CheckpointLoaderSimple", "inputs": {}}},
        input_schema={},
    )
    manager = DownloadManager(
        settings,
        EventBroker(),
        media_adapter=FakeMediaAdapter(),  # type: ignore[arg-type]
        processes=FakeProcesses(),  # type: ignore[arg-type]
        scheduler=scheduler,
    )
    monkeypatch.setattr(manager.comfy_templates, "compile", lambda *_args, **_kwargs: compiled)
    destination = settings.model_dir / "lease-image"
    destination.mkdir()
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="model_lease_image",
                name="Lease image",
                role="image",
                engine="comfyui",
                local_path=str(destination),
                manifest_json={"files": ["model.safetensors"]},
                active=False,
            )
        )
        session.add(
            Job(
                id="job_lease_image",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.RUNNING.value,
            )
        )
        session.commit()
    request = DownloadRequest(
        remote_id="owner/lease-image",
        revision="main",
        role="image",
        engine="comfyui",
        allow_patterns=["model.safetensors"],
        comfy_paths={"checkpoints": "."},
        workflow_template_id="lease-image",
        workflow_template_sha256="a" * 64,
    )

    async with scheduler.lease("primary"):
        activation = asyncio.create_task(
            manager._activate_comfy_install(  # type: ignore[arg-type]
                job_id="job_lease_image",
                install_id="model_lease_image",
                destination=destination,
                request=request,
                compiled=compiled,
                default_settings={},
            )
        )
        await asyncio.sleep(0.03)
        assert started.is_set() is False

    result = await asyncio.wait_for(activation, timeout=2)
    assert result
    assert started.is_set() is True
    with SessionLocal() as session:
        assert session.get(ModelInstall, "model_lease_image").active is True  # type: ignore[union-attr]


async def test_planned_media_activation_requires_output_and_records_evidence(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()

    class Processes:
        runtimes = None

        async def start_media(self, _model_paths: object = None) -> None:
            return None

    class MediaAdapter:
        def __init__(self) -> None:
            self.probes = 0

        async def object_info(self) -> dict[str, object]:
            return {}

        async def validate_workflow(self, _graph: dict[str, object]) -> list[str]:
            return []

        async def probe_workflow(
            self,
            _request: MediaRequest,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds == 300
            self.probes += 1

        async def capabilities(self) -> object:
            return SimpleNamespace(healthy=True, version="comfy-test")

    performance = {
        "version": 1,
        "signals": [{"kind": "native-low-step", "steps": 4}],
        "native_optimized": True,
    }
    compiled = SimpleNamespace(
        template=SimpleNamespace(
            id="planned-image",
            operation="text_to_image",
            runtime_adaptive=False,
            selected_files=["model.safetensors"],
            component_folders={"model.safetensors": "checkpoints"},
            sha256="a" * 64,
        ),
        ui_graph={},
        api_graph={"loader": {"class_type": "CheckpointLoaderSimple", "inputs": {}}},
        input_schema={"x-lm-atelier-workflow-performance": performance},
    )
    adapter = MediaAdapter()
    manager = DownloadManager(
        settings,
        EventBroker(),
        media_adapter=adapter,  # type: ignore[arg-type]
        processes=Processes(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(manager.comfy_templates, "compile", lambda *_args, **_kwargs: compiled)
    destination = settings.model_dir / "planned-image"
    destination.mkdir()
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="model_planned_image",
                name="Planned image",
                role="image",
                engine="comfyui",
                local_path=str(destination),
                manifest_json={
                    "files": ["model.safetensors"],
                    "expected_sha256": {"model.safetensors": "b" * 64},
                },
                active=False,
            )
        )
        session.add(
            Job(
                id="job_planned_image",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.RUNNING.value,
            )
        )
        session.commit()
    request = DownloadRequest(
        install_plan_id="plan_planned_image",
        remote_id="owner/planned-image",
        revision="main",
        role="image",
        engine="comfyui",
        allow_patterns=["model.safetensors"],
        comfy_paths={"checkpoints": "."},
        workflow_template_id="planned-image",
        workflow_template_sha256="a" * 64,
    )

    result = await manager._activate_comfy_install(  # type: ignore[arg-type]
        job_id="job_planned_image",
        install_id="model_planned_image",
        destination=destination,
        request=request,
        compiled=compiled,
        default_settings={},
    )

    assert result
    assert adapter.probes == 1
    with SessionLocal() as session:
        install = session.get(ModelInstall, "model_planned_image")
        evidence = session.query(ModelCapabilityEvidence).one()
        assert install and install.active is True
        assert evidence.result == "ready"
        assert evidence.runtime_build == "comfy-test"
        # Evidence is keyed on what the workflow executes, not the compiler
        # version, so a compiler change that alters nothing leaves it valid.
        revision = session.query(WorkflowRevision).one()
        assert revision.artifact_sha256
        assert evidence.workflow_contract_version == revision.artifact_sha256
        assert install.manifest_json["activation_artifact_sha256"] == revision.artifact_sha256
        assert evidence.details_json["workflow_performance"] == performance


async def test_adaptive_activation_failure_is_removed_before_retry(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    compiled = SimpleNamespace(
        template=SimpleNamespace(
            id="adaptive-image",
            operation="text_to_image",
            runtime_adaptive=True,
            selected_files=["model.safetensors"],
            component_folders={"model.safetensors": "checkpoints"},
            sha256="a" * 64,
        ),
        ui_graph={},
        api_graph={"loader": {"class_type": "CheckpointLoaderSimple", "inputs": {}}},
        input_schema={},
    )

    class FakeProcesses:
        async def start_media(self, _model_paths: object = None) -> None:
            return None

    class FakeMediaAdapter:
        async def object_info(self) -> dict[str, object]:
            return {}

        async def validate_workflow(self, _graph: dict[str, object]) -> list[str]:
            return []

    manager = DownloadManager(
        settings,
        EventBroker(),
        media_adapter=FakeMediaAdapter(),  # type: ignore[arg-type]
        processes=FakeProcesses(),  # type: ignore[arg-type]
    )
    info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename="model.safetensors", size=4, lfs=None)],
        sha="b" * 40,
        pipeline_tag="text-to-image",
        tags=[],
        gated=False,
    )
    manager._api = SimpleNamespace(model_info=lambda *_args, **_kwargs: info)  # type: ignore[assignment]

    async def prepare(_request: DownloadRequest) -> object:
        return compiled

    async def download_file(**kwargs: Any) -> str:
        target = kwargs["staging"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"safe")
        return str(target)

    probe_attempts = 0

    async def probe(_compiled: object) -> None:
        nonlocal probe_attempts
        probe_attempts += 1
        if probe_attempts == 1:
            raise RuntimeError("activation probe failed")

    monkeypatch.setattr(manager, "_prepare_comfy_template", prepare)
    monkeypatch.setattr(manager, "_download_file", download_file)
    monkeypatch.setattr(manager, "_probe_adaptive_checkpoint", probe)
    monkeypatch.setattr(manager, "_validate_standard_checkpoint_safetensors", lambda _path: None)
    monkeypatch.setattr(manager.comfy_templates, "compile", lambda *_args, **_kwargs: compiled)
    request = DownloadRequest(
        remote_id="owner/adaptive",
        revision="main",
        role="image",
        engine="comfyui",
        allow_patterns=["model.safetensors"],
        comfy_paths={"checkpoints": "."},
        workflow_template_id="adaptive-image",
        workflow_template_sha256="a" * 64,
    )
    with SessionLocal() as session:
        job = Job(
            id="job_adaptive_retry",
            kind=JobKind.DOWNLOAD.value,
            status=JobStatus.QUEUED.value,
            payload_json=request.model_dump(mode="json"),
        )
        session.add(job)
        session.commit()

    await manager._download("job_adaptive_retry")

    destination = settings.model_dir / manager._install_directory_name(
        request.remote_id,
        info.sha,
    )
    with SessionLocal() as session:
        failed_job = session.get(Job, "job_adaptive_retry")
        assert failed_job
        assert failed_job.status == JobStatus.FAILED.value
        assert failed_job.result_json == {}
        assert session.query(ModelInstall).count() == 0
        assert session.query(ModelProfile).count() == 0
    assert not destination.exists()

    starts: list[str] = []
    monkeypatch.setattr(manager, "start", starts.append)
    assert manager.resume("job_adaptive_retry") is True
    assert starts == ["job_adaptive_retry"]
    await manager._download("job_adaptive_retry")

    with SessionLocal() as session:
        completed_job = session.get(Job, "job_adaptive_retry")
        installs = session.query(ModelInstall).all()
        profiles = session.query(ModelProfile).all()
        assert completed_job
        assert completed_job.status == JobStatus.COMPLETE.value
        assert len(installs) == 1
        assert installs[0].active is True
        assert len(profiles) == 1
        assert profiles[0].model_install_id == installs[0].id
    assert (destination / "model.safetensors").read_bytes() == b"safe"


async def test_cancel_removes_provisional_install_and_abandoned_partial(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    scheduler = ResourceScheduler()
    manager = DownloadManager(settings, EventBroker(), scheduler=scheduler)
    destination = settings.model_dir / "provisional"
    destination.mkdir(parents=True)
    (destination / "model.safetensors").write_bytes(b"provisional")
    partial = settings.download_dir / "job_cancel_provisional.partial"
    partial.mkdir(parents=True)
    (partial / "chunk").write_bytes(b"partial")
    with SessionLocal() as session:
        install = ModelInstall(
            id="model_provisional",
            name="Provisional",
            role="image",
            engine="comfyui",
            local_path=str(destination),
            manifest_json={"files": ["model.safetensors"]},
            active=False,
        )
        session.add(install)
        session.add(
            Job(
                id="job_cancel_provisional",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.RUNNING.value,
                result_json={
                    "_provisional_install": {"model_install_id": install.id},
                },
            )
        )
        session.commit()

    async with scheduler.lease("primary"):
        cancellation = asyncio.create_task(manager.cancel("job_cancel_provisional"))
        await asyncio.sleep(0.03)
        assert cancellation.done() is False
        assert destination.exists()

    assert await asyncio.wait_for(cancellation, timeout=2) is True

    with SessionLocal() as session:
        job = session.get(Job, "job_cancel_provisional")
        assert job
        assert job.status == JobStatus.CANCELLED.value
        assert job.result_json == {}
        assert session.get(ModelInstall, "model_provisional") is None
    assert not destination.exists()
    assert not partial.exists()


def test_provisional_cleanup_preserves_files_used_by_an_active_selection(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    manager = DownloadManager(settings, EventBroker())
    destination = settings.model_dir / "shared"
    destination.mkdir(parents=True)
    (destination / "shared.safetensors").write_bytes(b"shared")
    (destination / "failed.safetensors").write_bytes(b"failed")
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id="model_active",
                name="Active",
                role="image",
                engine="comfyui",
                local_path=str(destination),
                manifest_json={"files": ["shared.safetensors"]},
                active=True,
            )
        )
        session.add(
            ModelInstall(
                id="model_failed",
                name="Failed",
                role="image",
                engine="comfyui",
                local_path=str(destination),
                manifest_json={"files": ["shared.safetensors", "failed.safetensors"]},
                active=False,
            )
        )
        session.add(
            Job(
                id="job_failed_shared",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.FAILED.value,
                result_json={
                    "_provisional_install": {"model_install_id": "model_failed"},
                },
            )
        )
        session.commit()

    assert manager._cleanup_provisional_install("job_failed_shared") is True

    with SessionLocal() as session:
        assert session.get(ModelInstall, "model_active")
        assert session.get(ModelInstall, "model_failed") is None
    assert (destination / "shared.safetensors").read_bytes() == b"shared"
    assert not (destination / "failed.safetensors").exists()


def test_provisional_cleanup_removes_files_even_if_the_row_never_committed(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    manager = DownloadManager(settings, EventBroker())
    destination = settings.model_dir / "uncommitted"
    destination.mkdir(parents=True)
    (destination / "model.safetensors").write_bytes(b"uncommitted")
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_uncommitted",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.FAILED.value,
            )
        )
        session.commit()

    assert (
        manager._cleanup_provisional_install(
            "job_uncommitted",
            provisional_path=destination,
            provisional_files=["model.safetensors"],
        )
        is True
    )
    assert not destination.exists()


def test_partial_cleanup_reclaims_terminal_quarantine_but_skips_active_job(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    manager = DownloadManager(settings, EventBroker())
    quarantine = settings.download_dir / ".discarded-installs"
    terminal = quarantine / "job_terminal-discard_dead"
    active = quarantine / "job_active-discard_live"
    terminal.mkdir(parents=True)
    active.mkdir()
    (terminal / "model").write_bytes(b"terminal")
    (active / "model").write_bytes(b"active")
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_active",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.RUNNING.value,
            )
        )
        session.commit()
        removed_count, reclaimed_bytes = manager.cleanup_partials(session)

    assert removed_count == 1
    assert reclaimed_bytes == len(b"terminal")
    assert not terminal.exists()
    assert (active / "model").read_bytes() == b"active"


async def test_template_workflow_exposes_model_only_loras(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    graph = {
        "model": {"class_type": "UNETLoader", "inputs": {}},
        "sampling": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["model", 0]},
        },
        "switch": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "on_true": ["sampling", 0],
                "on_false": ["model", 0],
            },
        },
        "sampler": {"class_type": "KSampler", "inputs": {"model": ["switch", 0]}},
    }
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="image_split_model_lora_test",
            path=settings.data_dir / "split-model-template.json",
            role="image",
            operation="image_to_image",
            score=1_000,
            sha256="9" * 64,
            dependencies=(),
        ),
        ui_graph={"nodes": []},
        api_graph=graph,
        input_schema={"type": "object", "properties": {}},
    )
    with SessionLocal() as session:
        install = ModelInstall(
            name="Split model editor",
            role="image",
            engine="comfyui",
            local_path=str(settings.model_dir / "split-model-editor"),
            manifest_json={"family": "split-model-test"},
            active=True,
        )
        session.add(install)
        session.flush()

        revision = DownloadManager._ensure_template_workflow(session, compiled, install)

        assert revision.definition.family_id is not None
        preference = session.scalar(
            select(WorkflowPreference).where(
                WorkflowPreference.workflow_family_id == revision.definition.family_id,
                WorkflowPreference.selector_capability == "image",
            )
        )
        assert preference is not None and preference.enabled
        assert workflow_lora_extension(revision) == {
            "mode": "model_only",
            "model": ["switch", 0],
        }
        assert revision.input_schema_json["properties"]["loras"] == {
            "type": "array",
            "title": "LoRAs",
            "description": "Optional verified LoRAs applied in order.",
            "default": [],
            "maxItems": 8,
        }
        assert revision.dependencies_json["extensions"]["lora"] == {
            "mode": "model_only",
            "model": ["switch", 0],
        }

        without_sampler = CompiledComfyTemplate(
            template=compiled.template,
            ui_graph=compiled.ui_graph,
            api_graph={"model": graph["model"]},
            input_schema={"type": "object", "properties": {}},
        )
        refreshed = DownloadManager._ensure_template_workflow(
            session,
            without_sampler,
            install,
        )
        assert refreshed.id != revision.id
        assert refreshed.dependencies_json["extensions"] == {}
        assert "loras" not in refreshed.input_schema_json["properties"]


async def test_a_slow_large_transfer_still_reports_its_speed(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Transfer rate needs byte samples closer together than five seconds.

    The monitor used to write one only when overall progress advanced a tenth
    of a percent. On a 40 GB install that is 40 MB of movement, which on a
    normal connection takes far longer than the rate window - so the speed the
    user was promised never appeared.
    """
    settings.prepare()
    configure_database(settings)
    init_db()
    manager = DownloadManager(settings, EventBroker())
    monkeypatch.setattr("local_lm.downloads._TRANSFER_SAMPLE_SECONDS", 0.01)

    total_size = 40 * 1024**3
    staging = tmp_path / "staging"
    staging.mkdir()
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_slow_transfer",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.RUNNING.value,
                phase="downloading",
                payload_json={},
            )
        )
        session.commit()

    # One megabyte of movement per sample: real progress, but far below the
    # old 0.1%-of-total threshold that used to gate a write.
    transferred = iter(range(0, 6 * 1024**2, 1024**2))

    def written_bytes(_pid: int) -> int:
        return next(transferred, 5 * 1024**2)

    monkeypatch.setattr(DownloadManager, "_process_tree_write_bytes", staticmethod(written_bytes))
    stop = asyncio.Event()

    async def run_monitor() -> None:
        await manager._monitor_transfer(
            job_id="job_slow_transfer",
            filename="model.safetensors",
            staging=staging,
            process=SimpleNamespace(pid=1234),  # type: ignore[arg-type]
            file_size=total_size,
            completed_bytes=0,
            total_size=total_size,
            stop=stop,
        )

    task = asyncio.create_task(run_monitor())
    # Poll for the rate instead of granting one fixed slice of wall clock: a
    # loaded runner can starve the monitor past any single sleep, and the
    # regression being pinned is "the speed appears", not "it appears fast".
    progress: dict[str, Any] = {}
    try:
        for _ in range(100):
            await asyncio.sleep(0.05)
            with SessionLocal() as session:
                job = session.get(Job, "job_slow_transfer")
                progress = job.progress_json if job else {}
            if progress.get("rate_bytes_per_second"):
                break
    finally:
        stop.set()
        await task

    assert progress["unit"] == "bytes"
    assert progress["completed_units"] > 0
    assert progress["rate_bytes_per_second"], "a moving transfer must report its speed"
