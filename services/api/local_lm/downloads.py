from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import HfApi
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .domain import CompatibilityLevel, JobKind, JobStatus, new_id, utcnow
from .events import EventBroker
from .models import Job, ModelInstall, ModelSource
from .schemas import DownloadRequest

_REMOTE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DownloadManager:
    def __init__(self, settings: Settings, events: EventBroker) -> None:
        self.settings = settings
        self.events = events
        self._api = HfApi(token=settings.hf_token)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._workers: dict[str, subprocess.Popen[bytes]] = {}
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_downloads)

    def set_token(self, token: str | None) -> None:
        self.settings.hf_token = token
        self._api = HfApi(token=token)

    def create(self, session: Session, request: DownloadRequest) -> Job:
        if not _REMOTE_ID.fullmatch(request.remote_id):
            raise ValueError("remote_id must be in owner/model form")
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
        job = Job(
            kind=JobKind.DOWNLOAD.value,
            status=JobStatus.QUEUED.value,
            phase="queued",
            payload_json=request.model_dump(mode="json"),
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        self.start(job.id)
        return job

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
                job.phase = "resuming"
            session.commit()
            job_ids = [job.id for job in jobs]
        for job_id in job_ids:
            self.start(job_id)

    async def cancel(self, job_id: str) -> bool:
        from .db import SessionLocal

        await self._stop_task(job_id)
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
            job.phase = "cancelled"
            job.completed_at = utcnow()
            session.commit()
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
            job.phase = "paused"
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
                or job.status != JobStatus.PAUSED.value
            ):
                return False
            job.status = JobStatus.QUEUED.value
            job.phase = "resume queued"
            job.completed_at = None
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
                job.phase = "interrupted by shutdown"
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
        return removed_count, reclaimed_bytes

    async def _download(self, job_id: str) -> None:
        from .db import SessionLocal

        try:
            async with self._semaphore:
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    if not job:
                        return
                    request = DownloadRequest.model_validate(job.payload_json)
                    job.status = JobStatus.RUNNING.value
                    job.phase = "inspecting"
                    job.started_at = utcnow()
                    job.attempt += 1
                    session.commit()
                await self.events.publish(
                    "download.started", job_id, {"remote_id": request.remote_id}
                )

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
                total_size = sum(
                    int(getattr(sibling, "size", 0) or 0)
                    for sibling in siblings
                    if sibling.rfilename in filenames
                )
                free_bytes = shutil.disk_usage(self.settings.model_dir).free
                if total_size and free_bytes < int(total_size * 1.1):
                    raise OSError(
                        f"insufficient disk space: need about {total_size:,} bytes, "
                        f"have {free_bytes:,}"
                    )

                revision = str(info.sha or request.revision)
                safe_name = request.remote_id.replace("/", "--")
                staging = self.settings.download_dir / f"{job_id}.partial"
                destination = self.settings.model_dir / safe_name / revision
                staging.mkdir(parents=True, exist_ok=True)
                for index, filename in enumerate(filenames):
                    downloaded_path = await self._download_file(
                        job_id=job_id,
                        remote_id=request.remote_id,
                        filename=filename,
                        revision=revision,
                        staging=staging,
                    )
                    expected_hash = request.expected_sha256.get(filename)
                    if expected_hash:
                        with SessionLocal() as session:
                            job = session.get(Job, job_id)
                            if job:
                                job.phase = f"verifying {filename}"
                                session.commit()
                        actual_hash = await asyncio.to_thread(
                            self._sha256_file, Path(downloaded_path)
                        )
                        if actual_hash != expected_hash:
                            raise ValueError(f"SHA-256 mismatch for {filename}")
                    with SessionLocal() as session:
                        job = session.get(Job, job_id)
                        if not job or job.status == JobStatus.CANCELLED.value:
                            return
                        job.phase = f"downloaded {filename}"
                        job.progress = (index + 1) / len(filenames) * 0.9
                        session.commit()
                    await self.events.publish(
                        "download.progress",
                        job_id,
                        {
                            "progress": (index + 1) / len(filenames) * 0.9,
                            "filename": filename,
                        },
                    )

                destination.parent.mkdir(parents=True, exist_ok=True)
                self._activate_staging(staging, destination)
                installed_size = sum(
                    path.stat().st_size for path in destination.rglob("*") if path.is_file()
                )

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
                            "revision": revision,
                            "files": filenames,
                            "expected_sha256": request.expected_sha256,
                            "recipe_id": request.recipe_id,
                            "recipe_version": request.recipe_version,
                            "comfy_paths": request.comfy_paths,
                            "workflow_path": request.workflow_path,
                            "default_settings": request.default_settings,
                        },
                    )
                    session.add(install)
                    job = session.get(Job, job_id)
                    if not job:
                        return
                    job.status = JobStatus.COMPLETE.value
                    job.progress = 1
                    job.phase = "complete"
                    job.result_json = {"model_install_id": install.id}
                    job.completed_at = utcnow()
                    session.commit()
                await self.events.publish(
                    "download.completed", job_id, {"model_install_id": install.id}
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                if job:
                    job.status = JobStatus.FAILED.value
                    job.phase = "failed"
                    job.error = str(exc)
                    job.completed_at = utcnow()
                    session.commit()
            await self.events.publish("download.failed", job_id, {"error": str(exc)})

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
    ) -> str:
        """Run the blocking Hub transfer in a process that pause/cancel can terminate."""
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        environment = os.environ.copy()
        environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-m", "local_lm.download_worker"],
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
        try:
            stdout, stderr = await asyncio.to_thread(process.communicate, payload)
        finally:
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
            return selected
        if request.role == "chat" and request.engine == "llama.cpp":
            ggufs = [
                sibling for sibling in siblings if str(sibling.rfilename).lower().endswith(".gguf")
            ]
            if not ggufs:
                return []
            ggufs.sort(key=lambda item: int(getattr(item, "size", 0) or 0))
            return [str(ggufs[0].rfilename)]
        raise ValueError("select explicit files for image and video model downloads")
