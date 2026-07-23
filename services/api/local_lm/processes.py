from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal
from urllib.parse import urlparse

import httpx
import psutil

from .config import Settings
from .models import ModelInstall, ModelProfile
from .schemas import WorkerStatus

logger = logging.getLogger(__name__)


@dataclass
class WorkerRecord:
    name: str
    process: asyncio.subprocess.Process
    command: list[str]
    log_handle: IO[bytes]
    profile_id: str | None = None
    state: Literal["starting", "ready"] = "starting"
    estimated_memory_bytes: int | None = None
    peak_memory_bytes: int = 0


class ProcessSupervisor:
    """Owns engine subprocesses without ever invoking a shell."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._workers: dict[str, WorkerRecord] = {}
        self._locks = {"chat": asyncio.Lock(), "media": asyncio.Lock()}

    def statuses(self) -> list[WorkerStatus]:
        result: list[WorkerStatus] = []
        for name in ("chat", "media"):
            record = self._workers.get(name)
            running = record is not None and record.process.returncode is None
            current_memory = (
                self._process_tree_rss(record.process.pid) if running and record else None
            )
            if record and current_memory is not None:
                record.peak_memory_bytes = max(record.peak_memory_bytes, current_memory)
            result.append(
                WorkerStatus(
                    name=name,
                    state=(
                        record.state if running and record else "exited" if record else "stopped"
                    ),
                    managed=record is not None,
                    running=running,
                    pid=record.process.pid if running and record else None,
                    profile_id=record.profile_id if record else None,
                    command=record.command if record else [],
                    exit_code=record.process.returncode if record else None,
                    estimated_memory_bytes=record.estimated_memory_bytes if record else None,
                    current_memory_bytes=current_memory,
                    peak_memory_bytes=(record.peak_memory_bytes or None) if record else None,
                )
            )
        return result

    async def load_chat(self, profile: ModelProfile, install: ModelInstall) -> WorkerStatus:
        if profile.engine != "llama.cpp":
            raise ValueError("the selected profile is not a llama.cpp profile")
        executable = self.settings.llama_executable
        if not executable:
            raise RuntimeError("LOCAL_LM_LLAMA_EXECUTABLE is not configured")
        model_path = self._gguf_path(Path(install.local_path), install.manifest_json)
        parsed = urlparse(self.settings.llama_url)
        command = [
            str(executable.expanduser().resolve(strict=True)),
            "--model",
            str(model_path),
            "--host",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port or 12341),
            *self._llama_load_arguments(profile.load_settings_json),
        ]
        estimate = self._estimate_chat_memory(model_path.stat().st_size, profile.load_settings_json)
        await self._replace(
            "chat",
            command,
            self.settings.llama_url + "/health",
            profile.id,
            estimated_memory_bytes=estimate,
        )
        return self.statuses()[0]

    async def start_media(self) -> WorkerStatus:
        executable = self.settings.comfy_executable
        directory = self.settings.comfy_directory
        if not executable or not directory:
            raise RuntimeError(
                "LOCAL_LM_COMFY_EXECUTABLE and LOCAL_LM_COMFY_DIRECTORY must be configured"
            )
        directory = directory.expanduser().resolve(strict=True)
        entrypoint = (directory / "main.py").resolve(strict=True)
        if directory not in entrypoint.parents:
            raise ValueError("ComfyUI entrypoint escapes its configured directory")
        parsed = urlparse(self.settings.comfy_url)
        trusted_custom_nodes = await self._trusted_comfy_node_folders()
        output_directory = self.settings.comfy_output_dir.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        command = [
            str(executable.expanduser().resolve(strict=True)),
            str(entrypoint),
            "--listen",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port or 8188),
            "--extra-model-paths-config",
            str(self._write_comfy_model_paths()),
            "--output-directory",
            str(output_directory),
            "--preview-method",
            "latent2rgb",
            "--disable-all-custom-nodes",
        ]
        if trusted_custom_nodes:
            command.extend(["--whitelist-custom-nodes", *trusted_custom_nodes])
        await self._replace("media", command, self.settings.comfy_url + "/system_stats")
        return self.statuses()[1]

    async def _trusted_comfy_node_folders(self) -> list[str]:
        from sqlalchemy import select

        from .custom_nodes import CustomNodeManager
        from .db import SessionLocal
        from .models import CustomNodeInstall

        with SessionLocal() as session:
            installs = list(
                session.scalars(
                    select(CustomNodeInstall)
                    .where(
                        CustomNodeInstall.active.is_(True),
                        CustomNodeInstall.trusted.is_(True),
                    )
                    .order_by(CustomNodeInstall.installed_path)
                ).all()
            )
            manager = CustomNodeManager(self.settings)
            for install in installs:
                await manager.verify(install)
            return [install.installed_path for install in installs]

    def _write_comfy_model_paths(self) -> Path:
        from sqlalchemy import select

        from .db import SessionLocal

        config: dict[str, dict[str, str]] = {}
        seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        with SessionLocal() as session:
            installs = session.scalars(
                select(ModelInstall).where(
                    ModelInstall.engine == "comfyui", ModelInstall.active.is_(True)
                )
            ).all()
            for install in installs:
                raw_paths = install.manifest_json.get("comfy_paths") or {}
                paths = self._validated_comfy_paths(raw_paths)
                if not paths:
                    continue
                base_path = str(Path(install.local_path).resolve())
                signature = (base_path, tuple(sorted(paths.items())))
                if signature in seen:
                    continue
                seen.add(signature)
                config[f"local_lm_{len(config) + 1}"] = {
                    "base_path": base_path,
                    **paths,
                }
        destination = self.settings.state_dir / "comfy-extra-model-paths.yaml"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
        return destination

    @staticmethod
    def _validated_comfy_paths(value: object) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        allowed = {"checkpoints", "diffusion_models", "text_encoders", "vae", "clip_vision"}
        result: dict[str, str] = {}
        for key, item in value.items():
            if key not in allowed or not isinstance(item, str):
                continue
            path = Path(item)
            if path.is_absolute() or ".." in path.parts:
                continue
            result[key] = item
        return result

    async def stop(self, name: str) -> WorkerStatus:
        if name not in self._locks:
            raise ValueError("worker must be chat or media")
        async with self._locks[name]:
            await self._stop_unlocked(name)
        return next(item for item in self.statuses() if item.name == name)

    async def close(self) -> None:
        await asyncio.gather(*(self.stop(name) for name in tuple(self._locks)))

    async def _replace(
        self,
        name: str,
        command: list[str],
        health_url: str,
        profile_id: str | None = None,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        async with self._locks[name]:
            await self._stop_unlocked(name)
            log_path = self.settings.log_dir / f"{name}-worker.log"
            log_handle = log_path.open("ab", buffering=0)
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception:
                log_handle.close()
                raise
            record = WorkerRecord(
                name,
                process,
                command,
                log_handle,
                profile_id,
                estimated_memory_bytes=estimated_memory_bytes,
            )
            self._workers[name] = record
            try:
                await self._wait_healthy(record, health_url)
                record.state = "ready"
            except Exception:
                await self._stop_unlocked(name)
                raise

    async def _wait_healthy(self, record: WorkerRecord, url: str) -> None:
        deadline = asyncio.get_running_loop().time() + self.settings.worker_startup_seconds
        async with httpx.AsyncClient() as client:
            while asyncio.get_running_loop().time() < deadline:
                if record.process.returncode is not None:
                    raise RuntimeError(
                        f"{record.name} worker exited with code {record.process.returncode}"
                    )
                try:
                    response = await client.get(url, timeout=1)
                    if response.is_success:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
        raise TimeoutError(f"{record.name} worker did not become healthy before timeout")

    async def _stop_unlocked(self, name: str) -> None:
        record = self._workers.pop(name, None)
        if not record:
            return
        try:
            if record.process.returncode is None:
                record.process.terminate()
                try:
                    await asyncio.wait_for(
                        record.process.wait(), timeout=self.settings.worker_shutdown_seconds
                    )
                except TimeoutError:
                    record.process.kill()
                    await record.process.wait()
        finally:
            record.log_handle.close()

    @staticmethod
    def _gguf_path(path: Path, manifest: dict[str, Any]) -> Path:
        path = path.expanduser().resolve(strict=True)
        if path.is_file() and path.suffix.lower() == ".gguf":
            return path
        raw_files = manifest.get("files", [])
        filenames = raw_files if isinstance(raw_files, list) else []
        candidates = [
            (path / filename).resolve()
            for filename in filenames
            if isinstance(filename, str) and filename.lower().endswith(".gguf")
        ]
        candidates = [item for item in candidates if item.is_file() and path in item.parents]
        if not candidates:
            candidates = sorted(path.rglob("*.gguf")) if path.is_dir() else []
        if len(candidates) != 1:
            raise ValueError("the model install must resolve to exactly one GGUF file")
        return candidates[0]

    @staticmethod
    def _llama_load_arguments(settings: dict[str, Any]) -> list[str]:
        mapping = {
            "context_length": "--ctx-size",
            "gpu_layers": "--n-gpu-layers",
            "threads": "--threads",
            "batch_size": "--batch-size",
            "ubatch_size": "--ubatch-size",
            "kv_cache_type_k": "--cache-type-k",
            "kv_cache_type_v": "--cache-type-v",
            "split_mode": "--split-mode",
            "tensor_split": "--tensor-split",
            "main_gpu": "--main-gpu",
            "rope_frequency_base": "--rope-freq-base",
            "rope_frequency_scale": "--rope-freq-scale",
        }
        arguments: list[str] = []
        for key, flag in mapping.items():
            if key in settings:
                arguments.extend([flag, str(settings[key])])
        if settings.get("flash_attention") is True:
            arguments.extend(["--flash-attn", "on"])
        elif settings.get("flash_attention") is False:
            arguments.extend(["--flash-attn", "off"])
        if settings.get("mmap") is False:
            arguments.append("--no-mmap")
        if settings.get("mlock") is True:
            arguments.append("--mlock")
        return arguments

    @staticmethod
    def _estimate_chat_memory(model_bytes: int, settings: dict[str, Any]) -> int:
        context_length = int(settings.get("context_length", 8192))
        context_overhead = max(512 * 1024**2, context_length * 128 * 1024)
        return model_bytes + context_overhead

    @staticmethod
    def _process_tree_rss(pid: int) -> int | None:
        try:
            process = psutil.Process(pid)
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            return total
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None
