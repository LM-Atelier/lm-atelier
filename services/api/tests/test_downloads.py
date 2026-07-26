from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_lm.adapters.base import MediaRequest
from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.domain import JobKind, JobStatus
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.models import Job, ModelInstall, ModelProfile
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import DownloadRequest


class FakeWorker:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False

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


class FakeProbeAdapter:
    def __init__(self) -> None:
        self.request: MediaRequest | None = None
        self.timeout_seconds: float | None = None

    async def probe_workflow(
        self,
        request: MediaRequest,
        *,
        timeout_seconds: float,
    ) -> None:
        self.request = request
        self.timeout_seconds = timeout_seconds


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
    assert json.loads(workers[0].payload)["token"] == "secret-token"
    assert environments[0]["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert "LOCAL_LM_HF_TOKEN" not in environments[0]
    assert "GITHUB_TOKEN" not in environments[0]
    assert "AWS_SECRET_ACCESS_KEY" not in environments[0]
    assert "job_test" not in manager._workers


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
        assert job.progress == pytest.approx(0.45)
    event = broker.since(0)[0]
    assert event.type == "download.progress"
    assert event.payload["downloaded_bytes"] == 50
    assert event.payload["file_size_bytes"] == 100


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
