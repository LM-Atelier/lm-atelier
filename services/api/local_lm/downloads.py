from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import psutil
from huggingface_hub import HfApi
from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapters.base import MediaRequest
from .comfy_templates import (
    COMFY_TEMPLATE_COMPILER_VERSION,
    ComfyTemplateRegistry,
    CompiledComfyTemplate,
)
from .config import Settings
from .domain import CompatibilityLevel, JobKind, JobStatus, new_id, utcnow
from .events import EventBroker
from .gguf import (
    GGUFSelectionError,
    automatic_gguf_selection,
    automatic_mmproj_selection,
    validate_gguf_selection,
)
from .models import Job, ModelInstall, ModelSource, WorkflowDefinition, WorkflowRevision
from .profile_service import ensure_profile_for_install, retire_profiles_for_installs
from .progress import completed_progress, update_job_progress
from .scheduler import ResourceScheduler
from .schemas import DownloadRequest
from .subprocess_env import subprocess_environment

if TYPE_CHECKING:
    from .adapters.comfyui import ComfyUIAdapter
    from .processes import ProcessSupervisor

_REMOTE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TRANSFER_ATTEMPTS = 3
_PROVISIONAL_INSTALL_KEY = "_provisional_install"
logger = logging.getLogger(__name__)


def _template_workflow_name(template_id: str) -> str:
    return f"ComfyUI template \u00b7 {template_id}"


def download_worker_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--download-worker"]
    return [sys.executable, "-m", "local_lm.download_worker"]


class DownloadManager:
    def __init__(
        self,
        settings: Settings,
        events: EventBroker,
        *,
        media_adapter: ComfyUIAdapter | None = None,
        processes: ProcessSupervisor | None = None,
        scheduler: ResourceScheduler | None = None,
    ) -> None:
        self.settings = settings
        self.events = events
        self.media_adapter = media_adapter
        self.processes = processes
        self.scheduler = scheduler or ResourceScheduler()
        self.comfy_templates = ComfyTemplateRegistry(settings)
        self._api = HfApi(token=settings.hf_token)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._workers: dict[str, subprocess.Popen[bytes]] = {}

    def set_token(self, token: str | None) -> None:
        self.settings.hf_token = token
        self._api = HfApi(token=token)

    def create(self, session: Session, request: DownloadRequest) -> Job:
        if not _REMOTE_ID.fullmatch(request.remote_id):
            raise ValueError("remote_id must be in owner/model form")
        if request.source_remote_id and not _REMOTE_ID.fullmatch(request.source_remote_id):
            raise ValueError("source_remote_id must be in owner/model form")
        if request.role != "chat" and not request.comfy_paths:
            request = request.model_copy(
                update={"comfy_paths": self._automatic_comfy_paths(request.allow_patterns)}
            )
        if request.workflow_template_id:
            template = self.comfy_templates.validate_download(
                request.workflow_template_id,
                request.role,
                request.remote_id,
                request.allow_patterns,
                request.comfy_paths,
                revision=request.revision,
            )
            if request.workflow_template_sha256 != template.sha256:
                raise ValueError("ComfyUI template changed; run the install check again")
        elif request.workflow_template_sha256:
            raise ValueError("workflow template hash requires a template identifier")
        for filename, digest in request.expected_sha256.items():
            path = PurePosixPath(filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("expected hash paths must be safe relative paths")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("expected SHA-256 values must be lowercase hexadecimal")
        allowed_comfy_folders = {
            "checkpoints",
            "diffusion_models",
            "text_encoders",
            "vae",
            "clip_vision",
        }
        for folder, relative_path in request.comfy_paths.items():
            path = PurePosixPath(relative_path)
            if folder not in allowed_comfy_folders:
                raise ValueError(f"unsupported ComfyUI model folder: {folder}")
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("ComfyUI model paths must be safe relative paths")
        serialized_request = request.model_dump(mode="json")
        active_statuses = {
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            JobStatus.PAUSED.value,
        }
        for existing in session.scalars(
            select(Job)
            .where(
                Job.kind == JobKind.DOWNLOAD.value,
                Job.status.in_(active_statuses),
            )
            .order_by(Job.created_at)
        ).all():
            if existing.payload_json == serialized_request:
                if existing.status != JobStatus.PAUSED.value:
                    self.start(existing.id)
                return existing
        job = Job(
            kind=JobKind.DOWNLOAD.value,
            status=JobStatus.QUEUED.value,
            queue_resource="network_transfer",
            queue_group="network",
            queue_priority=-10,
            queue_ticket=new_id("ticket"),
            enqueued_at=utcnow(),
            payload_json=serialized_request,
        )
        update_job_progress(
            job,
            stage="queued",
            queue_resource="network_transfer",
            indeterminate=True,
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        self.start(job.id)
        return job

    @staticmethod
    def _automatic_comfy_paths(filenames: list[str]) -> dict[str, str]:
        known_folders = {
            "checkpoints",
            "diffusion_models",
            "text_encoders",
            "vae",
            "clip_vision",
        }
        paths: dict[str, str] = {}
        for filename in filenames:
            parts = PurePosixPath(filename).parts
            for index, part in enumerate(parts[:-1]):
                if part in known_folders:
                    paths[part] = str(PurePosixPath(*parts[: index + 1]))
        if paths:
            return paths
        primary_name = " ".join(filenames).lower()
        if any(marker in primary_name for marker in ("diffusion", "unet", "t2v", "i2v")):
            return {"diffusion_models": "."}
        return {"checkpoints": "."}

    def start(self, job_id: str) -> None:
        if job_id in self._tasks and not self._tasks[job_id].done():
            return
        task = asyncio.create_task(self._download(job_id), name=f"download-{job_id}")
        self._tasks[job_id] = task

        def discard(completed: asyncio.Task[None]) -> None:
            if self._tasks.get(job_id) is completed:
                self._tasks.pop(job_id, None)

        task.add_done_callback(discard)

    def recover_interrupted(self) -> None:
        from .db import SessionLocal

        with SessionLocal() as session:
            jobs = session.scalars(
                select(Job).where(
                    Job.kind == JobKind.DOWNLOAD.value,
                    Job.status.in_(
                        [
                            JobStatus.INTERRUPTED.value,
                            JobStatus.RUNNING.value,
                            JobStatus.QUEUED.value,
                        ]
                    ),
                )
            ).all()
            for job in jobs:
                job.status = JobStatus.QUEUED.value
                job.error = None
                job.completed_at = None
                job.claim_owner = None
                job.claim_expires_at = None
                job.heartbeat_at = None
                job.enqueued_at = utcnow()
                update_job_progress(
                    job,
                    stage="resuming",
                    queue_resource=job.queue_resource or "network_transfer",
                    indeterminate=True,
                )
            session.commit()
            job_ids = [job.id for job in jobs]
        for job_id in job_ids:
            self.start(job_id)

    async def cancel(self, job_id: str) -> bool:
        from .db import SessionLocal

        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if not job or job.kind != JobKind.DOWNLOAD.value:
                return False
            if job.status in {
                JobStatus.COMPLETE.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            }:
                return False
            job.status = JobStatus.CANCELLED.value
            job.completed_at = utcnow()
            update_job_progress(
                job,
                stage="cancelled",
                queue_resource=job.queue_resource,
                indeterminate=True,
            )
            session.commit()
        await self._stop_task(job_id)
        await self._cleanup_provisional_install_serialized(job_id)
        self._discard_partial(job_id)
        await self.events.publish("download.cancelled", job_id)
        return True

    async def pause(self, job_id: str) -> bool:
        from .db import SessionLocal

        await self._stop_task(job_id)
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if (
                not job
                or job.kind != JobKind.DOWNLOAD.value
                or job.status not in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}
            ):
                return False
            job.status = JobStatus.PAUSED.value
            update_job_progress(
                job,
                stage="paused",
                queue_resource=job.queue_resource,
                indeterminate=True,
            )
            session.commit()
        await self.events.publish("download.paused", job_id)
        return True

    def resume(self, job_id: str) -> bool:
        from .db import SessionLocal

        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if (
                not job
                or job.kind != JobKind.DOWNLOAD.value
                or job.status
                not in {
                    JobStatus.PAUSED.value,
                    JobStatus.FAILED.value,
                    JobStatus.INTERRUPTED.value,
                }
            ):
                return False
            retrying = job.status != JobStatus.PAUSED.value
            job.status = JobStatus.QUEUED.value
            job.error = None
            job.started_at = None
            job.completed_at = None
            job.claim_owner = None
            job.claim_expires_at = None
            job.heartbeat_at = None
            job.enqueued_at = utcnow()
            update_job_progress(
                job,
                stage="retry queued" if retrying else "resume queued",
                queue_resource=job.queue_resource or "network_transfer",
                indeterminate=True,
            )
            session.commit()
        self.start(job_id)
        return True

    async def close(self) -> None:
        from .db import SessionLocal

        await asyncio.gather(
            *(self._stop_task(job_id) for job_id in list(self._tasks)),
            return_exceptions=True,
        )
        with SessionLocal() as session:
            jobs = session.scalars(
                select(Job).where(
                    Job.kind == JobKind.DOWNLOAD.value,
                    Job.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
                )
            ).all()
            for job in jobs:
                job.status = JobStatus.INTERRUPTED.value
                job.completed_at = utcnow()
                update_job_progress(
                    job,
                    stage="interrupted by shutdown",
                    queue_resource=job.queue_resource,
                    indeterminate=True,
                )
            session.commit()

    def cleanup_partials(self, session: Session) -> tuple[int, int]:
        active_ids = set(
            session.scalars(
                select(Job.id).where(
                    Job.kind == JobKind.DOWNLOAD.value,
                    Job.status.in_(
                        [
                            JobStatus.QUEUED.value,
                            JobStatus.RUNNING.value,
                            JobStatus.PAUSED.value,
                        ]
                    ),
                )
            ).all()
        )
        removed_count = 0
        reclaimed_bytes = 0
        for candidate in self.settings.download_dir.glob("*.partial"):
            job_id = candidate.name.removesuffix(".partial")
            if job_id in active_ids:
                continue
            reclaimed_bytes += self._path_size(candidate)
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate)
            else:
                candidate.unlink(missing_ok=True)
            removed_count += 1
        quarantine_parent = self.settings.download_dir / ".discarded-installs"
        if quarantine_parent.is_dir() and not quarantine_parent.is_symlink():
            for candidate in quarantine_parent.iterdir():
                if any(candidate.name.startswith(f"{job_id}-") for job_id in active_ids):
                    continue
                if not candidate.is_symlink():
                    reclaimed_bytes += self._path_size(candidate)
                if candidate.is_dir() and not candidate.is_symlink():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink(missing_ok=True)
                removed_count += 1
            with suppress(OSError):
                quarantine_parent.rmdir()
        return removed_count, reclaimed_bytes

    async def _download(self, job_id: str) -> None:
        from .db import SessionLocal

        provisional_install_id: str | None = None
        provisional_path: Path | None = None
        provisional_files: list[str] = []
        try:
            async with self.scheduler.job_lease(
                job_id,
                resource="network_transfer",
                group="network",
                priority=-10,
                capacity=self.settings.max_concurrent_downloads,
            ):
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    if not job:
                        return
                    request = DownloadRequest.model_validate(job.payload_json)
                    job.completed_at = None
                    job.error = None
                    update_job_progress(
                        job,
                        stage="inspecting",
                        queue_resource="network_transfer",
                        indeterminate=True,
                    )
                    session.commit()
                await self.scheduler.publish_job(job_id)
                await self.events.publish(
                    "download.started", job_id, {"remote_id": request.remote_id}
                )
                if request.workflow_template_id:
                    await self._cleanup_provisional_install_serialized(job_id)
                    with SessionLocal() as session:
                        job = session.get(Job, job_id)
                        if job:
                            update_job_progress(
                                job,
                                stage="preparing media runtime",
                                queue_resource=job.queue_resource,
                                indeterminate=True,
                            )
                            session.commit()
                    await self.scheduler.publish_job(job_id)
                    async with self.scheduler.lease("primary"):
                        compiled_template = await self._prepare_comfy_template(request)
                else:
                    compiled_template = None

                info = await asyncio.to_thread(
                    self._api.model_info,
                    request.remote_id,
                    revision=request.revision,
                    files_metadata=True,
                )
                siblings = list(info.siblings or [])
                filenames = self._select_files(request, siblings)
                if not filenames:
                    raise ValueError("no files matched the requested model selection")
                resolved_sha256 = self._resolved_sha256(request, siblings, filenames)
                total_size = sum(
                    int(getattr(sibling, "size", 0) or 0)
                    for sibling in siblings
                    if sibling.rfilename in filenames
                )
                file_sizes = {
                    str(sibling.rfilename): int(getattr(sibling, "size", 0) or 0)
                    for sibling in siblings
                    if sibling.rfilename in filenames
                }
                free_bytes = shutil.disk_usage(self.settings.model_dir).free
                if total_size and free_bytes < int(total_size * 1.1):
                    raise OSError(
                        f"insufficient disk space: need about {total_size:,} bytes, "
                        f"have {free_bytes:,}"
                    )

                revision = str(info.sha or request.revision)
                staging = self.settings.download_dir / f"{job_id}.partial"
                destination = self.settings.model_dir / self._install_directory_name(
                    request.remote_id,
                    revision,
                )
                staging.mkdir(parents=True, exist_ok=True)
                completed_bytes = 0
                for index, filename in enumerate(filenames):
                    with SessionLocal() as session:
                        job = session.get(Job, job_id)
                        if not job or job.status == JobStatus.CANCELLED.value:
                            return
                        update_job_progress(
                            job,
                            stage=f"downloading {filename}",
                            completed_units=completed_bytes,
                            total_units=total_size or None,
                            unit="bytes" if total_size else None,
                            file_index=index + 1,
                            file_count=len(filenames),
                            queue_resource=job.queue_resource,
                            indeterminate=not bool(total_size),
                        )
                        session.commit()
                    await self.scheduler.publish_job(job_id)
                    downloaded_path = await self._download_file(
                        job_id=job_id,
                        remote_id=request.remote_id,
                        filename=filename,
                        revision=revision,
                        staging=staging,
                        file_size=file_sizes.get(filename) or None,
                        completed_bytes=completed_bytes,
                        total_size=total_size or None,
                    )
                    expected_hash = resolved_sha256.get(filename)
                    if expected_hash:
                        with SessionLocal() as session:
                            job = session.get(Job, job_id)
                            if job:
                                update_job_progress(
                                    job,
                                    stage=f"verifying {filename}",
                                    completed_units=completed_bytes + file_sizes.get(filename, 0),
                                    total_units=total_size or None,
                                    unit="bytes" if total_size else None,
                                    file_index=index + 1,
                                    file_count=len(filenames),
                                    queue_resource="disk",
                                    indeterminate=True,
                                )
                                session.commit()
                        await self.scheduler.publish_job(job_id)
                        async with self.scheduler.lease("disk"):
                            actual_hash = await asyncio.to_thread(
                                self._sha256_file, Path(downloaded_path)
                            )
                        if actual_hash != expected_hash:
                            raise ValueError(f"SHA-256 mismatch for {filename}")
                    with SessionLocal() as session:
                        job = session.get(Job, job_id)
                        if not job or job.status == JobStatus.CANCELLED.value:
                            return
                        completed_bytes += file_sizes.get(filename, 0)
                        update_job_progress(
                            job,
                            stage=f"downloaded {filename}",
                            completed_units=completed_bytes if total_size else index + 1,
                            total_units=total_size if total_size else len(filenames),
                            unit="bytes" if total_size else "files",
                            file_index=index + 1,
                            file_count=len(filenames),
                            queue_resource=job.queue_resource,
                        )
                        session.commit()
                    await self.scheduler.publish_job(job_id)
                    await self.events.publish(
                        "download.progress",
                        job_id,
                        {
                            "progress": (
                                completed_bytes / total_size
                                if total_size
                                else (index + 1) / len(filenames)
                            ),
                            "downloaded_bytes": completed_bytes if total_size else None,
                            "total_bytes": total_size or None,
                            "filename": filename,
                        },
                    )

                if compiled_template and compiled_template.template.runtime_adaptive:
                    with SessionLocal() as session:
                        job = session.get(Job, job_id)
                        if job:
                            update_job_progress(
                                job,
                                stage="validating checkpoint structure",
                                completed_units=completed_bytes if total_size else None,
                                total_units=total_size or None,
                                unit="bytes" if total_size else None,
                                queue_resource="disk",
                                indeterminate=True,
                            )
                            session.commit()
                    await self.scheduler.publish_job(job_id)
                    selected = PurePosixPath(compiled_template.template.selected_files[0])
                    checkpoint_path = staging.joinpath(*selected.parts)
                    async with self.scheduler.lease("disk"):
                        await asyncio.to_thread(
                            self._validate_standard_checkpoint_safetensors,
                            checkpoint_path,
                        )

                destination.parent.mkdir(parents=True, exist_ok=True)
                if compiled_template:
                    provisional_path = destination
                    provisional_files = list(filenames)
                async with self.scheduler.lease("disk"):
                    self._activate_staging(staging, destination)
                    installed_size = sum(
                        path.stat().st_size for path in destination.rglob("*") if path.is_file()
                    )
                template_defaults = (
                    self._template_defaults(compiled_template) if compiled_template else {}
                )
                default_settings = {**template_defaults, **request.default_settings}

                with SessionLocal() as session:
                    source = session.scalar(
                        select(ModelSource).where(
                            ModelSource.provider == "huggingface",
                            ModelSource.remote_id == request.remote_id,
                            ModelSource.revision == revision,
                        )
                    )
                    if not source:
                        source = ModelSource(
                            provider="huggingface",
                            remote_id=request.remote_id,
                            revision=revision,
                            metadata_json={
                                "pipeline_tag": info.pipeline_tag,
                                "tags": info.tags or [],
                                "gated": info.gated,
                            },
                        )
                        session.add(source)
                        session.flush()
                    install = ModelInstall(
                        id=new_id("model"),
                        source_id=source.id,
                        name=request.remote_id.rsplit("/", 1)[-1],
                        role=request.role,
                        engine=request.engine,
                        local_path=str(destination),
                        size_bytes=installed_size,
                        compatibility=CompatibilityLevel.LIKELY.value,
                        manifest_json={
                            "remote_id": request.remote_id,
                            "source_remote_id": request.source_remote_id,
                            "revision": revision,
                            "files": filenames,
                            "expected_sha256": resolved_sha256,
                            "recipe_id": request.recipe_id,
                            "recipe_version": request.recipe_version,
                            "comfy_paths": request.comfy_paths,
                            "workflow_path": request.workflow_path,
                            "workflow_template_id": request.workflow_template_id,
                            "workflow_template_sha256": request.workflow_template_sha256,
                            "default_settings": default_settings,
                        },
                        active=compiled_template is None,
                    )
                    session.add(install)
                    session.flush()
                    provisional_install_id = install.id if compiled_template else None
                    profile = None
                    workflow_revision_id = None
                    superseded_install_ids: list[str] = []
                    if not compiled_template:
                        profile = ensure_profile_for_install(
                            session,
                            install,
                            default_settings=default_settings,
                        )
                    job = session.get(Job, job_id)
                    if not job:
                        return
                    update_job_progress(
                        job,
                        stage="validating runtime" if compiled_template else "activating",
                        completed_units=completed_bytes if total_size else None,
                        total_units=total_size or None,
                        unit="bytes" if total_size else None,
                        queue_resource="primary_compute" if compiled_template else "disk",
                        indeterminate=True,
                    )
                    if profile:
                        job.result_json = {
                            "model_install_id": install.id,
                            "profile_id": profile.id,
                            "workflow_revision_id": None,
                            "superseded_model_install_ids": [],
                        }
                    elif provisional_install_id:
                        job.result_json = {
                            _PROVISIONAL_INSTALL_KEY: {
                                "model_install_id": provisional_install_id,
                                "local_path": str(destination),
                                "files": filenames,
                            }
                        }
                    session.commit()
                    install_id = install.id
                    profile_id = profile.id if profile else None

                if compiled_template:
                    activation = await self._activate_comfy_install(
                        job_id=job_id,
                        install_id=install_id,
                        destination=destination,
                        request=request,
                        compiled=compiled_template,
                        default_settings=default_settings,
                    )
                    if not activation:
                        return
                    compiled_template, profile_id, workflow_revision_id, superseded_install_ids = (
                        activation
                    )
                    provisional_install_id = None
                    provisional_path = None
                    provisional_files = []

                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    if not job:
                        return
                    job.status = JobStatus.COMPLETE.value
                    job.completed_at = utcnow()
                    completed_progress(job)
                    session.commit()
                await self.scheduler.publish_job(job_id)
                await self.events.publish(
                    "download.completed",
                    job_id,
                    {"model_install_id": install_id, "profile_id": profile_id},
                )
        except asyncio.CancelledError:
            await self._cleanup_provisional_install_serialized(
                job_id,
                provisional_install_id=provisional_install_id,
                provisional_path=provisional_path,
                provisional_files=provisional_files,
            )
            raise
        except Exception as exc:
            try:
                await self._cleanup_provisional_install_serialized(
                    job_id,
                    provisional_install_id=provisional_install_id,
                    provisional_path=provisional_path,
                    provisional_files=provisional_files,
                )
            except Exception:
                logger.exception("Could not safely clean failed model install %s", job_id)
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                if job:
                    job.status = JobStatus.FAILED.value
                    job.error = str(exc)
                    job.completed_at = utcnow()
                    update_job_progress(
                        job,
                        stage="failed",
                        queue_resource=job.queue_resource,
                        indeterminate=True,
                    )
                    session.commit()
            await self.scheduler.publish_job(job_id)
            await self.events.publish("download.failed", job_id, {"error": str(exc)})

    async def _prepare_comfy_template(
        self, request: DownloadRequest
    ) -> CompiledComfyTemplate | None:
        if not request.workflow_template_id:
            return None
        if not self.media_adapter:
            raise RuntimeError("automatic ComfyUI workflow setup is unavailable")
        try:
            object_info = await self.media_adapter.object_info()
        except Exception:
            if not self.processes:
                raise
            await self.processes.start_media()
            object_info = await self.media_adapter.object_info()
        compiled = self.comfy_templates.compile(
            request.workflow_template_id,
            request.role,
            object_info,
            validate_model_choices=False,
            remote_id=request.remote_id,
            revision=request.revision,
            selected_files=request.allow_patterns,
            comfy_paths=request.comfy_paths,
        )
        if compiled.template.sha256 != request.workflow_template_sha256:
            raise ValueError("ComfyUI template changed; run the install check again")
        errors = await self.media_adapter.validate_workflow(compiled.api_graph)
        if errors:
            raise ValueError("; ".join(errors))
        return compiled

    async def _probe_adaptive_checkpoint(
        self,
        compiled: CompiledComfyTemplate,
    ) -> None:
        if not self.media_adapter:
            raise RuntimeError("automatic ComfyUI activation is unavailable")
        await self.media_adapter.probe_workflow(
            MediaRequest(
                run_id=new_id("activation-probe"),
                operation="text_to_image",
                prompt="simple geometric shape",
                negative_prompt="",
                input_paths=[],
                workflow=compiled.api_graph,
                parameters={
                    "width": 256,
                    "height": 256,
                    "batch_size": 1,
                    "seed": 0,
                    "steps": 1,
                    "cfg": 1.0,
                    "sampler": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            ),
            timeout_seconds=300,
        )

    async def _activate_comfy_install(
        self,
        *,
        job_id: str,
        install_id: str,
        destination: Path,
        request: DownloadRequest,
        compiled: CompiledComfyTemplate,
        default_settings: dict[str, Any],
    ) -> tuple[CompiledComfyTemplate, str, str, list[str]] | None:
        """Restart, probe, and commit one media install under the compute lease."""

        if not self.processes or not self.media_adapter:
            raise RuntimeError("automatic ComfyUI activation is unavailable")
        from .db import SessionLocal

        async with self.scheduler.lease("primary"):
            await self.processes.start_media((destination, request.comfy_paths))
            refreshed_object_info = await self.media_adapter.object_info()
            compiled = self.comfy_templates.compile(
                request.workflow_template_id or "",
                request.role,
                refreshed_object_info,
                remote_id=request.remote_id,
                revision=request.revision,
                selected_files=request.allow_patterns,
                comfy_paths=request.comfy_paths,
            )
            if compiled.template.sha256 != request.workflow_template_sha256:
                raise ValueError("ComfyUI template changed during installation")
            validation_errors = await self.media_adapter.validate_workflow(compiled.api_graph)
            if validation_errors:
                raise ValueError("; ".join(validation_errors))
            if compiled.template.runtime_adaptive:
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    if job:
                        update_job_progress(
                            job,
                            stage="probing checkpoint runtime",
                            queue_resource="primary_compute",
                            indeterminate=True,
                        )
                        session.commit()
                await self.scheduler.publish_job(job_id)
                await self._probe_adaptive_checkpoint(compiled)
            with SessionLocal() as session:
                activated_install = session.get(ModelInstall, install_id)
                job = session.get(Job, job_id)
                if not activated_install or not job:
                    return None
                activated_install.active = True
                superseded_install_ids = self._deactivate_superseded_media_installs(
                    session,
                    activated_install,
                    request.source_remote_id,
                )
                profile = ensure_profile_for_install(
                    session,
                    activated_install,
                    default_settings=default_settings,
                )
                workflow_revision = self._ensure_template_workflow(
                    session,
                    compiled,
                    activated_install,
                )
                update_job_progress(
                    job,
                    stage="activating",
                    queue_resource="primary_compute",
                    indeterminate=True,
                )
                job.result_json = {
                    "model_install_id": activated_install.id,
                    "profile_id": profile.id,
                    "workflow_revision_id": workflow_revision.id,
                    "superseded_model_install_ids": superseded_install_ids,
                }
                session.commit()
                await self.scheduler.publish_job(job_id)
                return (
                    compiled,
                    profile.id,
                    workflow_revision.id,
                    superseded_install_ids,
                )

    async def refresh_installed_media_workflows(self) -> int:
        """Recompile installed catalog workflows after compiler improvements."""
        if not self.media_adapter:
            return 0
        from .db import SessionLocal

        object_info = await self.media_adapter.object_info()
        refreshed = 0
        with SessionLocal() as session:
            installs = session.scalars(
                select(ModelInstall).where(
                    ModelInstall.active.is_(True),
                    ModelInstall.engine == "comfyui",
                    ModelInstall.role.in_(["image", "video"]),
                )
            ).all()
            for install in installs:
                template_id = install.manifest_json.get("workflow_template_id")
                template_sha256 = install.manifest_json.get("workflow_template_sha256")
                if not isinstance(template_id, str) or not template_id:
                    continue
                try:
                    compiled = self.comfy_templates.compile(
                        template_id,
                        install.role,
                        object_info,
                        remote_id=str(install.manifest_json.get("remote_id") or ""),
                        revision=str(install.manifest_json.get("revision") or ""),
                        selected_files=[
                            str(item)
                            for item in install.manifest_json.get("files", [])
                            if isinstance(item, str)
                        ],
                        comfy_paths={
                            str(key): str(value)
                            for key, value in (
                                install.manifest_json.get("comfy_paths") or {}
                            ).items()
                            if isinstance(key, str) and isinstance(value, str)
                        },
                    )
                    if template_sha256 and compiled.template.sha256 != template_sha256:
                        logger.warning(
                            "Skipped changed ComfyUI template %s for model %s",
                            template_id,
                            install.id,
                        )
                        continue
                    before = session.scalar(
                        select(WorkflowDefinition.current_revision_id).where(
                            WorkflowDefinition.name
                            == _template_workflow_name(compiled.template.id),
                            WorkflowDefinition.operation == compiled.template.operation,
                        )
                    )
                    revision = self._ensure_template_workflow(session, compiled, install)
                    if revision.id != before:
                        refreshed += 1
                except (KeyError, TypeError, ValueError):
                    logger.exception(
                        "Could not refresh ComfyUI workflow for model %s",
                        install.id,
                    )
            session.commit()
        return refreshed

    @staticmethod
    def _deactivate_superseded_media_installs(
        session: Session,
        current: ModelInstall,
        source_remote_id: str | None,
    ) -> list[str]:
        if not source_remote_id:
            return []
        source_key = source_remote_id.casefold()
        superseded: list[str] = []
        candidates = session.scalars(
            select(ModelInstall).where(
                ModelInstall.id != current.id,
                ModelInstall.role == current.role,
                ModelInstall.engine == current.engine,
                ModelInstall.active.is_(True),
            )
        ).all()
        for candidate in candidates:
            identities = {
                str(candidate.manifest_json.get("remote_id") or "").casefold(),
                str(candidate.manifest_json.get("source_remote_id") or "").casefold(),
            }
            if source_key not in identities:
                continue
            candidate.active = False
            superseded.append(candidate.id)
        retire_profiles_for_installs(session, superseded)
        return superseded

    @staticmethod
    def _template_defaults(compiled: CompiledComfyTemplate) -> dict[str, Any]:
        properties = compiled.input_schema.get("properties")
        if not isinstance(properties, dict):
            return {}
        return {
            key: value["default"]
            for key, value in properties.items()
            if isinstance(value, dict)
            and "default" in value
            and not (isinstance(value["default"], str) and value["default"].startswith("${"))
        }

    @staticmethod
    def _ensure_template_workflow(
        session: Session,
        compiled: CompiledComfyTemplate,
        install: ModelInstall,
    ) -> WorkflowRevision:
        name = _template_workflow_name(compiled.template.id)
        definition = session.scalar(
            select(WorkflowDefinition).where(
                WorkflowDefinition.name == name,
                WorkflowDefinition.operation == compiled.template.operation,
            )
        )
        if not definition:
            definition = WorkflowDefinition(
                name=name,
                operation=compiled.template.operation,
                description=(
                    "Automatically configured from the workflow catalog shipped with ComfyUI."
                ),
            )
            session.add(definition)
            session.flush()
        current = (
            session.get(WorkflowRevision, definition.current_revision_id)
            if definition.current_revision_id
            else None
        )
        declared_installs = (
            {str(item) for item in current.dependencies_json.get("model_install_ids", [])}
            if current and isinstance(current.dependencies_json.get("model_install_ids"), list)
            else set()
        )
        if (
            current
            and current.dependencies_json.get("template_sha256") == compiled.template.sha256
            and current.dependencies_json.get("compiler_version") == COMFY_TEMPLATE_COMPILER_VERSION
            and install.id in declared_installs
        ):
            return current
        version = max((item.version for item in definition.revisions), default=0) + 1
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=version,
            engine="comfyui",
            ui_graph_json=compiled.ui_graph,
            api_graph_json=compiled.api_graph,
            input_schema_json=compiled.input_schema,
            dependencies_json={
                "model_install_ids": sorted({install.id, *declared_installs}),
                "compiler_version": COMFY_TEMPLATE_COMPILER_VERSION,
                "template_id": compiled.template.id,
                "template_sha256": compiled.template.sha256,
                "model_files": compiled.template.selected_files,
                "custom_nodes": [],
            },
            trusted=True,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        return revision

    async def _cleanup_provisional_install_serialized(
        self,
        job_id: str,
        *,
        provisional_install_id: str | None = None,
        provisional_path: Path | None = None,
        provisional_files: list[str] | None = None,
    ) -> bool:
        """Retire provisional media files without racing an active generation."""

        async with self.scheduler.lease("primary"):
            cleaned = self._cleanup_provisional_install(
                job_id,
                provisional_install_id=provisional_install_id,
                provisional_path=provisional_path,
                provisional_files=provisional_files,
            )
            if cleaned and self.processes:
                with suppress(Exception):
                    await self.processes.start_media()
            return cleaned

    def _cleanup_provisional_install(
        self,
        job_id: str,
        *,
        provisional_install_id: str | None = None,
        provisional_path: Path | None = None,
        provisional_files: list[str] | None = None,
    ) -> bool:
        """Atomically retire an inactive install after quarantining its managed files."""

        from .db import SessionLocal

        quarantined_root: Path | None = None
        moves: list[tuple[Path, Path]] = []
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            marker = (
                job.result_json.get(_PROVISIONAL_INSTALL_KEY)
                if job and isinstance(job.result_json, dict)
                else None
            )
            marker_install_id = marker.get("model_install_id") if isinstance(marker, dict) else None
            install_id = provisional_install_id or (
                marker_install_id if isinstance(marker_install_id, str) else None
            )
            marker_path = marker.get("local_path") if isinstance(marker, dict) else None
            cleanup_path = provisional_path or (
                Path(marker_path) if isinstance(marker_path, str) and marker_path else None
            )
            marker_files = marker.get("files") if isinstance(marker, dict) else None
            cleanup_files = provisional_files or (
                [item for item in marker_files if isinstance(item, str)]
                if isinstance(marker_files, list)
                else []
            )
            install = session.get(ModelInstall, install_id) if install_id else None
            if not install:
                if not cleanup_path:
                    if job and marker is not None:
                        result = dict(job.result_json)
                        result.pop(_PROVISIONAL_INSTALL_KEY, None)
                        job.result_json = result
                        session.commit()
                    return False
                install = ModelInstall(
                    id=install_id or new_id("discard"),
                    name="Incomplete install",
                    role="image",
                    engine="comfyui",
                    local_path=str(cleanup_path),
                    manifest_json={"files": cleanup_files},
                    active=False,
                )
            elif install.active:
                return False
            persisted_install = session.get(ModelInstall, install.id)
            try:
                moves, quarantined_root = self._quarantine_provisional_files(
                    session,
                    install,
                    job_id,
                )
                if persisted_install:
                    retire_profiles_for_installs(session, [persisted_install.id])
                    session.delete(persisted_install)
                if job and marker is not None:
                    result = dict(job.result_json)
                    result.pop(_PROVISIONAL_INSTALL_KEY, None)
                    job.result_json = result
                session.commit()
            except Exception:
                session.rollback()
                self._restore_quarantined_files(moves)
                raise
        if quarantined_root:
            try:
                if quarantined_root.is_dir() and not quarantined_root.is_symlink():
                    shutil.rmtree(quarantined_root)
                else:
                    quarantined_root.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Provisional install files remain safely quarantined at %s",
                    quarantined_root,
                    exc_info=True,
                )
        return True

    def _quarantine_provisional_files(
        self,
        session: Session,
        install: ModelInstall,
        job_id: str,
    ) -> tuple[list[tuple[Path, Path]], Path | None]:
        model_root = self.settings.model_dir.resolve()
        install_path = Path(install.local_path).resolve()
        if install_path == model_root or model_root not in install_path.parents:
            raise RuntimeError("refusing to clean a provisional install outside managed storage")
        if not install_path.exists():
            return [], None

        quarantine_parent = self.settings.download_dir / ".discarded-installs"
        quarantine_parent.mkdir(parents=True, exist_ok=True)
        quarantine_root = quarantine_parent / f"{job_id}-{new_id('discard')}"
        active_siblings = [
            candidate
            for candidate in session.scalars(
                select(ModelInstall).where(
                    ModelInstall.id != install.id,
                    ModelInstall.active.is_(True),
                )
            ).all()
            if Path(candidate.local_path).resolve() == install_path
        ]
        moves: list[tuple[Path, Path]] = []
        try:
            if not active_siblings:
                os.replace(install_path, quarantine_root)
                moves.append((quarantine_root, install_path))
                return moves, quarantine_root

            retained_files = {
                str(filename)
                for sibling in active_siblings
                for filename in sibling.manifest_json.get("files", [])
                if isinstance(filename, str)
            }
            for filename in install.manifest_json.get("files", []):
                if not isinstance(filename, str) or filename in retained_files:
                    continue
                relative = PurePosixPath(filename)
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                candidate = install_path.joinpath(*relative.parts).resolve()
                if install_path not in candidate.parents or not candidate.is_file():
                    continue
                quarantined = quarantine_root.joinpath(*relative.parts)
                quarantined.parent.mkdir(parents=True, exist_ok=True)
                os.replace(candidate, quarantined)
                moves.append((quarantined, candidate))
            for directory in sorted(
                (item for item in install_path.rglob("*") if item.is_dir()),
                reverse=True,
            ):
                with suppress(OSError):
                    directory.rmdir()
            return moves, quarantine_root if moves else None
        except Exception:
            self._restore_quarantined_files(moves)
            raise

    @staticmethod
    def _restore_quarantined_files(moves: list[tuple[Path, Path]]) -> None:
        for quarantined, original in reversed(moves):
            if not quarantined.exists():
                continue
            original.parent.mkdir(parents=True, exist_ok=True)
            with suppress(OSError):
                os.replace(quarantined, original)

    def _discard_partial(self, job_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
            return
        partial = self.settings.download_dir / f"{job_id}.partial"
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        else:
            partial.unlink(missing_ok=True)

    async def _stop_task(self, job_id: str) -> None:
        """Stop the isolated transfer process before cancelling its controller task."""
        worker = self._workers.get(job_id)
        if worker and worker.poll() is None:
            with suppress(ProcessLookupError):
                worker.terminate()
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if worker and worker.poll() is None:
            with suppress(ProcessLookupError):
                await asyncio.to_thread(worker.wait)

    async def _download_file(
        self,
        *,
        job_id: str,
        remote_id: str,
        filename: str,
        revision: str,
        staging: Path,
        file_size: int | None = None,
        completed_bytes: int = 0,
        total_size: int | None = None,
    ) -> str:
        from .db import SessionLocal

        for attempt in range(1, _TRANSFER_ATTEMPTS + 1):
            try:
                return await self._download_file_once(
                    job_id=job_id,
                    remote_id=remote_id,
                    filename=filename,
                    revision=revision,
                    staging=staging,
                    file_size=file_size,
                    completed_bytes=completed_bytes,
                    total_size=total_size,
                )
            except RuntimeError:
                if attempt >= _TRANSFER_ATTEMPTS:
                    raise
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    if job:
                        update_job_progress(
                            job,
                            stage=f"retrying {filename} ({attempt + 1}/{_TRANSFER_ATTEMPTS})",
                            queue_resource=job.queue_resource,
                            indeterminate=True,
                        )
                        session.commit()
                await self.scheduler.publish_job(job_id)
                await self.events.publish(
                    "download.retrying",
                    job_id,
                    {
                        "filename": filename,
                        "attempt": attempt + 1,
                        "maximum_attempts": _TRANSFER_ATTEMPTS,
                    },
                )
                await asyncio.sleep(2 ** (attempt - 1))
        raise RuntimeError("download transfer exhausted its retry budget")

    async def _download_file_once(
        self,
        *,
        job_id: str,
        remote_id: str,
        filename: str,
        revision: str,
        staging: Path,
        file_size: int | None = None,
        completed_bytes: int = 0,
        total_size: int | None = None,
    ) -> str:
        """Run the blocking Hub transfer in a process that pause/cancel can terminate."""
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        environment = subprocess_environment(overrides={"HF_HUB_DISABLE_PROGRESS_BARS": "1"})
        process = subprocess.Popen(
            download_worker_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            creationflags=creationflags,
        )
        self._workers[job_id] = process
        payload = json.dumps(
            {
                "repo_id": remote_id,
                "filename": filename,
                "revision": revision,
                "local_dir": str(staging),
                "token": self.settings.hf_token,
            }
        ).encode()
        monitor_stop: asyncio.Event | None = None
        monitor: asyncio.Task[None] | None = None
        if file_size and total_size:
            monitor_stop = asyncio.Event()
            monitor = asyncio.create_task(
                self._monitor_transfer(
                    job_id=job_id,
                    filename=filename,
                    staging=staging,
                    process=process,
                    file_size=file_size,
                    completed_bytes=completed_bytes,
                    total_size=total_size,
                    stop=monitor_stop,
                )
            )
        try:
            stdout, stderr = await asyncio.to_thread(process.communicate, payload)
        finally:
            if monitor_stop and monitor:
                monitor_stop.set()
                await asyncio.gather(monitor, return_exceptions=True)
            if self._workers.get(job_id) is process:
                self._workers.pop(job_id, None)
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    process.terminate()
                with suppress(ProcessLookupError):
                    await asyncio.to_thread(process.wait)
        if process.returncode:
            detail = stderr.decode(errors="replace").strip()
            raise RuntimeError(detail[-2_000:] or "Hub download worker failed")
        try:
            result = json.loads(stdout)
            path = result["path"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("Hub download worker returned an invalid response") from exc
        if not isinstance(path, str):
            raise RuntimeError("Hub download worker returned an invalid path")
        return path

    async def _monitor_transfer(
        self,
        *,
        job_id: str,
        filename: str,
        staging: Path,
        process: subprocess.Popen[bytes],
        file_size: int | None,
        completed_bytes: int,
        total_size: int | None,
        stop: asyncio.Event,
    ) -> None:
        from .db import SessionLocal

        initial_current_bytes = self._staged_current_file_bytes(staging, completed_bytes, file_size)
        initial_write_bytes = self._process_tree_write_bytes(process.pid)
        maximum_transferred = initial_current_bytes
        last_reported = -1.0
        while not stop.is_set():
            staged_bytes = self._staged_current_file_bytes(staging, completed_bytes, file_size)
            written_bytes = max(
                self._process_tree_write_bytes(process.pid) - initial_write_bytes,
                0,
            )
            maximum_transferred = max(
                maximum_transferred,
                staged_bytes,
                initial_current_bytes + written_bytes,
            )
            if file_size:
                maximum_transferred = min(maximum_transferred, file_size)
            transferred_bytes = completed_bytes + maximum_transferred
            progress = min(transferred_bytes / total_size, 1.0) if total_size else 0.0
            if last_reported < 0 or progress - last_reported >= 0.001:
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    if not job or job.status != JobStatus.RUNNING.value:
                        return
                    update_job_progress(
                        job,
                        stage=f"downloading {filename}",
                        completed_units=transferred_bytes,
                        total_units=total_size,
                        unit="bytes",
                        queue_resource=job.queue_resource,
                    )
                    session.commit()
                await self.scheduler.publish_job(job_id)
                await self.events.publish(
                    "download.progress",
                    job_id,
                    {
                        "progress": progress,
                        "filename": filename,
                        "downloaded_bytes": transferred_bytes,
                        "file_size_bytes": file_size,
                        "total_bytes": total_size,
                    },
                )
                last_reported = progress
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=1)

    @classmethod
    def _staged_current_file_bytes(
        cls,
        staging: Path,
        completed_bytes: int,
        file_size: int | None,
    ) -> int:
        staged = max(cls._path_size(staging) - completed_bytes, 0)
        return min(staged, file_size) if file_size else staged

    @staticmethod
    def _process_tree_write_bytes(pid: int) -> int:
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return 0
        total = 0
        for process in processes:
            try:
                total += int(process.io_counters().write_bytes)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        return total

    @staticmethod
    def _activate_staging(staging: Path, destination: Path) -> None:
        """Atomically add complete files while preserving other selections from a revision."""
        if not destination.exists():
            os.replace(staging, destination)
            return
        for source in sorted(staging.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(staging)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                source.unlink()
            else:
                os.replace(source, target)
        shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _install_directory_name(remote_id: str, revision: str) -> str:
        """Keep managed model paths short while retaining identity in the database."""
        identity = f"{remote_id}\0{revision}".encode()
        return hashlib.sha256(identity).hexdigest()[:24]

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _path_size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    @staticmethod
    def _select_files(request: DownloadRequest, siblings: list[Any]) -> list[str]:
        blocked = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
        available: list[str] = []
        unsafe: list[str] = []
        for sibling in siblings:
            if not sibling.rfilename:
                continue
            filename = str(sibling.rfilename)
            path = PurePosixPath(filename)
            if path.is_absolute() or ".." in path.parts:
                continue
            if path.suffix.lower() in blocked:
                unsafe.append(filename)
                continue
            available.append(filename)
        if request.allow_patterns:
            selected = [
                filename
                for filename in available
                if any(fnmatch.fnmatch(filename, pattern) for pattern in request.allow_patterns)
            ]
            blocked_selected = [
                filename
                for filename in unsafe
                if any(fnmatch.fnmatch(filename, pattern) for pattern in request.allow_patterns)
            ]
            if blocked_selected:
                raise ValueError("pickle-compatible model weights are blocked by default")
            if request.role == "chat" and request.engine == "llama.cpp":
                auxiliary = [
                    filename
                    for filename in selected
                    if filename.lower().endswith(".gguf")
                    and "mmproj" in PurePosixPath(filename).name.lower()
                ]
                if len(auxiliary) > 1:
                    raise ValueError(
                        "llama.cpp chat installs may select only one multimodal projector"
                    )
                unexpected = [
                    filename for filename in selected if not filename.lower().endswith(".gguf")
                ]
                if unexpected:
                    raise ValueError("llama.cpp chat installs may select only GGUF model files")
                records = [
                    DownloadManager._sibling_gguf_metadata(
                        sibling,
                        request.expected_sha256,
                    )
                    for sibling in siblings
                    if str(getattr(sibling, "rfilename", "") or "") in selected
                ]
                try:
                    primary = validate_gguf_selection(
                        records,
                        require_split_metadata=True,
                    )
                except GGUFSelectionError as exc:
                    raise ValueError(str(exc)) from exc
                return [*primary, *auxiliary]
            return selected
        if request.role == "chat" and request.engine == "llama.cpp":
            records = [
                DownloadManager._sibling_gguf_metadata(sibling, request.expected_sha256)
                for sibling in siblings
            ]
            try:
                primary = automatic_gguf_selection(records, psutil.virtual_memory().total)
            except GGUFSelectionError as exc:
                raise ValueError(str(exc)) from exc
            projector = automatic_mmproj_selection(records, primary)
            return [*primary, *([projector] if projector else [])]
        raise ValueError("select explicit files for image and video model downloads")

    @staticmethod
    def _sibling_gguf_metadata(
        sibling: Any,
        expected_sha256: dict[str, str],
    ) -> dict[str, Any]:
        filename = str(getattr(sibling, "rfilename", "") or "")
        return {
            "filename": filename,
            "size": getattr(sibling, "size", None),
            "sha256": expected_sha256.get(filename) or DownloadManager._sibling_sha256(sibling),
        }

    @staticmethod
    def _sibling_sha256(sibling: Any) -> str | None:
        lfs = getattr(sibling, "lfs", None)
        value = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            return None
        return value.lower()

    @staticmethod
    def _resolved_sha256(
        request: DownloadRequest,
        siblings: list[Any],
        filenames: list[str],
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        selected = set(filenames)
        for sibling in siblings:
            filename = str(getattr(sibling, "rfilename", "") or "")
            if filename not in selected:
                continue
            requested = request.expected_sha256.get(filename)
            live = DownloadManager._sibling_sha256(sibling)
            if requested and live and requested != live:
                raise ValueError(
                    f"SHA-256 metadata changed for {filename}; run the install check again"
                )
            digest = requested or live
            if digest:
                resolved[filename] = digest
        return resolved

    @staticmethod
    def _validate_standard_checkpoint_safetensors(path: Path) -> None:
        """Bounded header probe for the adaptive CheckpointLoaderSimple contract."""

        maximum_header_bytes = 64 * 1024**2
        try:
            file_size = path.stat().st_size
            with path.open("rb") as handle:
                encoded_size = handle.read(8)
                if len(encoded_size) != 8:
                    raise ValueError
                header_size = int.from_bytes(encoded_size, "little", signed=False)
                if (
                    header_size < 2
                    or header_size > maximum_header_bytes
                    or header_size > file_size - 8
                ):
                    raise ValueError
                raw_header = handle.read(header_size)
            header = json.loads(raw_header)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(
                "the selected file is not a valid bounded safetensors archive"
            ) from exc
        if not isinstance(header, dict):
            raise ValueError("the selected safetensors header is not an object")
        tensor_names = [
            str(name).casefold()
            for name, value in header.items()
            if name != "__metadata__" and isinstance(value, dict)
        ]
        component_prefixes = (
            (
                "model.diffusion_model.",
                "diffusion_model.",
                "model.model.",
            ),
            (
                "first_stage_model.",
                "vae.",
            ),
            (
                "cond_stage_model.",
                "conditioner.",
                "text_encoder.",
            ),
        )
        if not all(
            any(name.startswith(prefix) for name in tensor_names for prefix in prefixes)
            for prefixes in component_prefixes
        ):
            raise ValueError(
                "the selected safetensors file is not a complete standard checkpoint; "
                "an official model workflow is required"
            )
