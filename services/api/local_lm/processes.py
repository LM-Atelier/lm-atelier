from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import httpx
import psutil

from .config import Settings
from .gguf import GGUFSelectionError, automatic_mmproj_selection, validate_gguf_selection
from .models import ModelAssetInstall, ModelInstall, ModelProfile
from .schemas import WorkerStatus
from .subprocess_env import subprocess_environment

if TYPE_CHECKING:
    from .runtime_provisioning import RuntimeProvisioner

logger = logging.getLogger(__name__)

WORKER_LOG_MAX_BYTES = 2 * 1024 * 1024
WORKER_LOG_BACKUP_COUNT = 3
WORKER_STDERR_TAIL_BYTES = 16 * 1024
WORKER_STDERR_DISPLAY_CHARS = 2_000
WORKER_STDERR_DISPLAY_LINES = 12

_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-_][0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_BEARER_SECRET = re.compile(r"(?i)\b(bearer)\s+\S+")
_NAMED_SECRET = re.compile(
    r"""(?ix)
    \b(api[_-]?key|authorization|credential|password|secret|token)
    (\s*[:=]\s*)
    (?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s,;]+)
    """
)
_KNOWN_TOKEN = re.compile(
    r"\b(?:hf_[A-Za-z0-9_-]{6,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"AKIA[A-Z0-9]{16}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."
    r"[A-Za-z0-9_-]{10,})\b"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?<![\w])(?:[a-z]:[\\/])[^\s\"'<>|]+")
_UNIX_PRIVATE_PATH = re.compile(r"(?<![\w:])/(?:home|root|Users|tmp|var/tmp|run/user)/[^\s\"'<>|]+")


class _RotatingWorkerLog:
    """Small binary log writer with fixed per-file and retention bounds."""

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = WORKER_LOG_MAX_BYTES,
        backup_count: int = WORKER_LOG_BACKUP_COUNT,
    ) -> None:
        if max_bytes < 1 or backup_count < 1:
            raise ValueError("worker log bounds must be positive")
        self.path = path
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        path.parent.mkdir(parents=True, exist_ok=True)
        self._remove_expired_backups()
        self._normalize_retained_files()
        if path.is_file() and path.stat().st_size:
            self._rotate_files()
        self._handle = path.open("ab", buffering=0)
        self._size = path.stat().st_size

    def write(self, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            remaining = self.max_bytes - self._size
            if remaining <= 0:
                self._handle.close()
                self._rotate_files()
                self._handle = self.path.open("ab", buffering=0)
                self._size = 0
                remaining = self.max_bytes
            segment = view[:remaining]
            written = self._handle.write(segment)
            if written is None:
                written = len(segment)
            self._size += written
            view = view[written:]

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def _normalize_retained_files(self) -> None:
        for candidate in (
            self.path,
            *(self._backup_path(index) for index in range(1, self.backup_count + 1)),
        ):
            if not candidate.is_file() or candidate.stat().st_size <= self.max_bytes:
                continue
            with candidate.open("rb") as source:
                source.seek(-self.max_bytes, os.SEEK_END)
                tail = source.read(self.max_bytes)
            candidate.write_bytes(tail)

    def _remove_expired_backups(self) -> None:
        prefix = f"{self.path.name}."
        for candidate in self.path.parent.glob(f"{self.path.name}.*"):
            suffix = candidate.name.removeprefix(prefix)
            if suffix.isdigit() and int(suffix) > self.backup_count:
                candidate.unlink(missing_ok=True)

    def _rotate_files(self) -> None:
        self._normalize_retained_files()
        self._backup_path(self.backup_count).unlink(missing_ok=True)
        for index in range(self.backup_count - 1, 0, -1):
            source = self._backup_path(index)
            if source.is_file():
                os.replace(source, self._backup_path(index + 1))
        if self.path.is_file() and self.path.stat().st_size:
            os.replace(self.path, self._backup_path(1))

    def _backup_path(self, index: int) -> Path:
        return self.path.with_name(f"{self.path.name}.{index}")


@dataclass
class WorkerRecord:
    name: str
    process: asyncio.subprocess.Process
    command: list[str]
    log: _RotatingWorkerLog
    profile_id: str | None = None
    state: Literal["starting", "ready"] = "starting"
    estimated_memory_bytes: int | None = None
    startup_duration_ms: int | None = None
    peak_memory_bytes: int = 0
    stderr_tail: bytearray = field(default_factory=bytearray)
    output_task: asyncio.Task[None] | None = None
    failure_detail: str | None = None


class ProcessSupervisor:
    """Owns engine subprocesses without ever invoking a shell."""

    def __init__(
        self,
        settings: Settings,
        runtimes: RuntimeProvisioner | None = None,
    ) -> None:
        self.settings = settings
        self.runtimes = runtimes
        self._workers: dict[str, WorkerRecord] = {}
        self._locks = {"chat": asyncio.Lock(), "media": asyncio.Lock()}
        self._private_output_suppression_depth = 0

    @property
    def private_output_suppressed(self) -> bool:
        return self._private_output_suppression_depth > 0

    def begin_private_session(self) -> None:
        """Keep backend output from entering durable logs during private work."""
        self._private_output_suppression_depth += 1
        for record in self._workers.values():
            record.stderr_tail.clear()

    def end_private_session(self) -> None:
        if self._private_output_suppression_depth > 0:
            self._private_output_suppression_depth -= 1
        for record in self._workers.values():
            record.stderr_tail.clear()

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
            stderr_tail = self._stderr_tail(record) if record and not running else None
            failure_detail = None
            if record and not running:
                exit_code = (
                    record.process.returncode
                    if record.process.returncode is not None
                    else "unknown"
                )
                failure_detail = record.failure_detail or (
                    f"{record.name} worker exited with code {exit_code}."
                )
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
                    command=self._safe_command(record.command) if record else [],
                    exit_code=record.process.returncode if record else None,
                    estimated_memory_bytes=record.estimated_memory_bytes if record else None,
                    startup_duration_ms=record.startup_duration_ms if record else None,
                    current_memory_bytes=current_memory,
                    peak_memory_bytes=(record.peak_memory_bytes or None) if record else None,
                    failure_detail=failure_detail,
                    stderr_tail=stderr_tail,
                    log_path=self._public_log_path(name),
                )
            )
        return result

    async def load_chat(self, profile: ModelProfile, install: ModelInstall) -> WorkerStatus:
        if profile.engine == "vllm":
            return await self._load_vllm_chat(profile, install)
        if profile.engine != "llama.cpp":
            raise ValueError("the selected profile is not a managed chat profile")
        if (
            not self.settings.llama_executable or not self.settings.llama_executable.is_file()
        ) and self.runtimes:
            await self.runtimes.ensure("llama.cpp")
        executable = self.settings.llama_executable
        if not executable:
            raise RuntimeError("The llama.cpp runtime is not installed.")
        model_paths = self._gguf_paths(Path(install.local_path), install.manifest_json)
        model_path = model_paths[0]
        launch_path = self._llama_model_path(model_path)
        projection_path = self._llama_mmproj_path(
            Path(install.local_path),
            install.manifest_json,
            model_paths,
        )
        parsed = urlparse(self.settings.llama_url)
        command = [
            str(executable.expanduser().resolve(strict=True)),
            "--model",
            str(launch_path),
            "--host",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port or 12341),
            *self._llama_load_arguments(profile.load_settings_json),
        ]
        if projection_path is not None:
            command.extend(["--mmproj", str(self._llama_model_path(projection_path))])
        model_size = sum(item.stat().st_size for item in model_paths)
        if projection_path is not None:
            model_size += projection_path.stat().st_size
        estimate = self._estimate_chat_memory(model_size, profile.load_settings_json)
        previous_engine = self.settings.chat_engine
        self.settings.chat_engine = "llama.cpp"
        try:
            await self._replace(
                "chat",
                command,
                self.settings.llama_url + "/health",
                profile.id,
                estimated_memory_bytes=estimate,
            )
        except Exception:
            self.settings.chat_engine = previous_engine
            raise
        return self.statuses()[0]

    async def _load_vllm_chat(
        self,
        profile: ModelProfile,
        install: ModelInstall,
    ) -> WorkerStatus:
        if (
            not self.settings.vllm_executable or not self.settings.vllm_executable.is_file()
        ) and self.runtimes:
            await self.runtimes.ensure("vllm")
        executable = self.settings.vllm_executable
        if not executable:
            raise RuntimeError("The vLLM runtime is not installed.")
        model_root = Path(install.local_path).expanduser().resolve(strict=True)
        if not model_root.is_dir():
            raise ValueError("the vLLM model install must be a snapshot directory")
        if (
            not (model_root / "config.json").is_file()
            or not (model_root / "hf_quant_config.json").is_file()
        ):
            raise ValueError("the vLLM model install is missing ModelOpt metadata")
        if not any(model_root.rglob("*.safetensors")):
            raise ValueError("the vLLM model install has no safetensors weights")
        parsed = urlparse(self.settings.llama_url)
        context_length = int(profile.load_settings_json.get("context_length", 8192))
        command = [
            str(executable.expanduser().resolve(strict=True)),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_root),
            "--served-model-name",
            "local-model",
            "--host",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port or 12341),
            "--quantization",
            "modelopt",
            "--max-model-len",
            str(context_length),
            "--max-num-seqs",
            "1",
            "--gpu-memory-utilization",
            "0.9",
            "--limit-mm-per-prompt",
            json.dumps({"image": self.settings.vision_max_images, "video": 1}),
        ]
        raw_offload = profile.load_settings_json.get("cpu_offload_gb")
        if (
            isinstance(raw_offload, (int, float))
            and not isinstance(raw_offload, bool)
            and raw_offload > 0
        ):
            command.extend(["--cpu-offload-gb", str(float(raw_offload))])
        model_size = sum(
            candidate.stat().st_size for candidate in model_root.rglob("*") if candidate.is_file()
        )
        estimate = self._estimate_chat_memory(model_size, profile.load_settings_json)
        previous_engine = self.settings.chat_engine
        self.settings.chat_engine = "vllm"
        try:
            await self._replace(
                "chat",
                command,
                self.settings.llama_url + "/health",
                profile.id,
                estimated_memory_bytes=estimate,
            )
        except Exception:
            self.settings.chat_engine = previous_engine
            raise
        return self.statuses()[0]

    async def start_media(
        self,
        provisional_model_paths: tuple[Path, dict[str, str]] | None = None,
    ) -> WorkerStatus:
        if (
            not self.settings.comfy_executable
            or not self.settings.comfy_executable.is_file()
            or not self.settings.comfy_directory
            or not (self.settings.comfy_directory / "main.py").is_file()
        ) and self.runtimes:
            await self.runtimes.ensure("comfyui")
        executable = self.settings.comfy_executable
        directory = self.settings.comfy_directory
        if not executable or not directory:
            raise RuntimeError("The ComfyUI runtime is not installed.")
        directory = directory.expanduser().resolve(strict=True)
        entrypoint = (directory / "main.py").resolve(strict=True)
        if directory not in entrypoint.parents:
            raise ValueError("ComfyUI entrypoint escapes its configured directory")
        parsed = urlparse(self.settings.comfy_url)
        trusted_custom_nodes = await self._trusted_comfy_node_folders()
        output_directory = self.settings.comfy_output_dir.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        model_paths_config = (
            self._write_comfy_model_paths(provisional_model_paths)
            if provisional_model_paths
            else self._write_comfy_model_paths()
        )
        command = [
            str(executable.expanduser().resolve(strict=True)),
            str(entrypoint),
            "--listen",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port or 8188),
            "--extra-model-paths-config",
            str(model_paths_config),
            "--output-directory",
            str(output_directory),
            "--preview-method",
            "latent2rgb",
            "--disable-all-custom-nodes",
        ]
        if trusted_custom_nodes:
            command.extend(["--whitelist-custom-nodes", *trusted_custom_nodes])
        await self._replace("media", command, self.settings.comfy_url + "/object_info")
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
                session.expunge(install)
        for install in installs:
            await manager.verify(install)
        return [install.installed_path for install in installs]

    def _write_comfy_model_paths(
        self,
        provisional_model_paths: tuple[Path, dict[str, str]] | None = None,
    ) -> Path:
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
            assets = session.scalars(
                select(ModelAssetInstall).where(ModelAssetInstall.active.is_(True))
            ).all()
            for asset in assets:
                folder = {
                    "lora": "loras",
                    "vae": "vae",
                    "controlnet": "controlnet",
                    "upscaler": "upscale_models",
                    "embedding": "embeddings",
                    "ip_adapter": "ipadapter",
                }.get(asset.kind)
                if not folder:
                    continue
                base_path = str(Path(asset.local_path).resolve())
                signature = (base_path, ((folder, "."),))
                if signature in seen:
                    continue
                seen.add(signature)
                config[f"local_lm_{len(config) + 1}"] = {
                    "base_path": base_path,
                    folder: ".",
                }
        if provisional_model_paths:
            provisional_root, raw_paths = provisional_model_paths
            paths = self._validated_comfy_paths(raw_paths)
            base_path = str(provisional_root.resolve())
            signature = (base_path, tuple(sorted(paths.items())))
            if paths and signature not in seen:
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
        allowed = {
            "checkpoints",
            "diffusion_models",
            "text_encoders",
            "vae",
            "clip_vision",
            "loras",
            "controlnet",
            "upscale_models",
            "embeddings",
            "ipadapter",
        }
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
            startup_started_at = time.perf_counter()
            log_path = self.settings.log_dir / f"{name}-worker.log"
            worker_log = _RotatingWorkerLog(log_path)
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=subprocess_environment(),
                    start_new_session=True,
                )
            except Exception as exc:
                worker_log.close()
                message = self._sanitize_diagnostic(str(exc))
                if isinstance(exc, OSError):
                    raise OSError(message) from exc
                raise RuntimeError(message or f"Could not start the {name} worker.") from exc
            record = WorkerRecord(
                name,
                process,
                command,
                worker_log,
                profile_id,
                estimated_memory_bytes=estimated_memory_bytes,
            )
            record.output_task = asyncio.create_task(self._capture_process_output(record))
            self._workers[name] = record
            try:
                await self._wait_healthy(record, health_url)
                record.startup_duration_ms = round(
                    (time.perf_counter() - startup_started_at) * 1_000
                )
                record.state = "ready"
            except Exception as exc:
                record.startup_duration_ms = round(
                    (time.perf_counter() - startup_started_at) * 1_000
                )
                record.failure_detail = self._sanitize_diagnostic(str(exc)).rstrip(".") + "."
                await self._terminate_record(record)
                message = record.failure_detail
                stderr_tail = self._stderr_tail(record)
                if stderr_tail:
                    message = f"{message} {stderr_tail}"
                if isinstance(exc, TimeoutError):
                    raise TimeoutError(message) from exc
                if isinstance(exc, OSError):
                    raise OSError(message) from exc
                raise RuntimeError(message) from exc

    async def _wait_healthy(self, record: WorkerRecord, url: str) -> None:
        deadline = asyncio.get_running_loop().time() + self.settings.worker_startup_seconds
        async with httpx.AsyncClient(trust_env=False) as client:
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
        await self._terminate_record(record)

    async def _terminate_record(self, record: WorkerRecord) -> None:
        try:
            if record.process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    record.process.terminate()
                try:
                    await asyncio.wait_for(
                        record.process.wait(), timeout=self.settings.worker_shutdown_seconds
                    )
                except TimeoutError:
                    record.process.kill()
                    await record.process.wait()
            if record.output_task:
                await record.output_task
        finally:
            record.log.close()

    async def _capture_process_output(self, record: WorkerRecord) -> None:
        log_failed = False

        async def pump(stream: asyncio.StreamReader | None, *, stderr: bool) -> None:
            nonlocal log_failed
            if stream is None:
                return
            try:
                while chunk := await stream.read(16 * 1024):
                    if self.private_output_suppressed:
                        continue
                    if stderr:
                        record.stderr_tail.extend(chunk)
                        overflow = len(record.stderr_tail) - WORKER_STDERR_TAIL_BYTES
                        if overflow > 0:
                            del record.stderr_tail[:overflow]
                    if not log_failed:
                        try:
                            record.log.write(chunk)
                        except (OSError, ValueError):
                            log_failed = True
                            logger.exception("Could not write the %s worker log", record.name)
            except (OSError, ValueError):
                logger.exception("Could not read %s worker output", record.name)

        try:
            await asyncio.gather(
                pump(record.process.stdout, stderr=False),
                pump(record.process.stderr, stderr=True),
            )
        finally:
            record.log.close()

    def _stderr_tail(self, record: WorkerRecord) -> str | None:
        if not record.stderr_tail:
            return None
        value = record.stderr_tail.decode("utf-8", errors="replace")
        value = self._sanitize_diagnostic(value)
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if not lines:
            return None
        value = "\n".join(lines[-WORKER_STDERR_DISPLAY_LINES:])
        if len(value) > WORKER_STDERR_DISPLAY_CHARS:
            value = "…" + value[-WORKER_STDERR_DISPLAY_CHARS:]
        return value

    def _sanitize_diagnostic(self, value: str) -> str:
        value = _ANSI_ESCAPE.sub("", value)
        if self.settings.hf_token:
            value = value.replace(self.settings.hf_token, "[redacted]")
        for private_root, replacement in (
            (str(self.settings.data_dir.expanduser().resolve()), "[data folder]"),
            (str(Path.home().resolve()), "[home]"),
        ):
            if private_root:
                value = re.sub(re.escape(private_root), replacement, value, flags=re.IGNORECASE)
        value = _BEARER_SECRET.sub(r"\1 [redacted]", value)
        value = _NAMED_SECRET.sub(r"\1\2[redacted]", value)
        value = _KNOWN_TOKEN.sub("[redacted]", value)
        value = _WINDOWS_ABSOLUTE_PATH.sub("[local path]", value)
        value = _UNIX_PRIVATE_PATH.sub("[local path]", value)
        return "".join(
            character for character in value if character in "\r\n\t" or character >= " "
        )

    def _safe_command(self, command: list[str]) -> list[str]:
        return [
            (
                "[local path]"
                if Path(argument).expanduser().is_absolute()
                else self._sanitize_diagnostic(argument)
            )
            for argument in command
        ]

    def _public_log_path(self, name: str) -> str:
        path = self.settings.log_dir / f"{name}-worker.log"
        try:
            return path.resolve().relative_to(self.settings.data_dir.resolve()).as_posix()
        except ValueError:
            return f"logs/{name}-worker.log"

    @staticmethod
    def _gguf_path(path: Path, manifest: dict[str, Any]) -> Path:
        return ProcessSupervisor._gguf_paths(path, manifest)[0]

    @staticmethod
    def _gguf_paths(path: Path, manifest: dict[str, Any]) -> tuple[Path, ...]:
        path = path.expanduser().resolve(strict=True)
        if path.is_file() and path.suffix.lower() == ".gguf":
            return (path,)
        raw_files = manifest.get("files", [])
        filenames = raw_files if isinstance(raw_files, list) else []
        candidate_records: list[dict[str, Any]] = []
        candidate_paths: dict[str, Path] = {}
        for filename in filenames:
            if not isinstance(filename, str) or not filename.lower().endswith(".gguf"):
                continue
            relative = PurePosixPath(filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in filename:
                continue
            candidate = path.joinpath(*relative.parts).resolve()
            if candidate.is_file() and path in candidate.parents:
                candidate_records.append(
                    {
                        "filename": filename,
                        "size": candidate.stat().st_size,
                        "sha256": None,
                    }
                )
                candidate_paths[filename] = candidate
        if not candidate_records and path.is_dir():
            for candidate in sorted(path.rglob("*.gguf")):
                filename = candidate.relative_to(path).as_posix()
                candidate_records.append(
                    {
                        "filename": filename,
                        "size": candidate.stat().st_size,
                        "sha256": None,
                    }
                )
                candidate_paths[filename] = candidate
        try:
            ordered = validate_gguf_selection(
                candidate_records,
                require_split_metadata=False,
            )
        except GGUFSelectionError as exc:
            raise ValueError(f"the model install has an invalid GGUF layout: {exc}") from exc
        return tuple(candidate_paths[filename] for filename in ordered)

    @staticmethod
    def _llama_mmproj_path(
        path: Path,
        manifest: dict[str, Any],
        model_paths: tuple[Path, ...],
    ) -> Path | None:
        path = path.expanduser().resolve(strict=True)
        if path.is_file():
            return None
        raw_files = manifest.get("files", [])
        filenames = raw_files if isinstance(raw_files, list) else []
        candidate_records: list[dict[str, Any]] = []
        candidate_paths: dict[str, Path] = {}
        for filename in filenames:
            if not isinstance(filename, str):
                continue
            relative = PurePosixPath(filename)
            if (
                not filename.casefold().endswith(".gguf")
                or "mmproj" not in relative.name.casefold()
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in filename
            ):
                continue
            candidate = path.joinpath(*relative.parts).resolve()
            if candidate.is_file() and path in candidate.parents:
                candidate_records.append(
                    {
                        "filename": filename,
                        "size": candidate.stat().st_size,
                        "sha256": None,
                    }
                )
                candidate_paths[filename] = candidate
        if not candidate_records:
            for candidate in sorted(path.rglob("*.gguf")):
                if "mmproj" not in candidate.name.casefold():
                    continue
                filename = candidate.relative_to(path).as_posix()
                candidate_records.append(
                    {
                        "filename": filename,
                        "size": candidate.stat().st_size,
                        "sha256": None,
                    }
                )
                candidate_paths[filename] = candidate
        primary_names = [
            model_path.relative_to(path).as_posix()
            for model_path in model_paths
            if path in model_path.parents
        ]
        selected = automatic_mmproj_selection(candidate_records, primary_names)
        return candidate_paths.get(selected) if selected else None

    @staticmethod
    def _llama_model_path(path: Path) -> Path:
        """Use the filesystem's short name when llama.cpp would exceed MAX_PATH."""
        if sys.platform != "win32":
            return path
        if len(str(path)) < 240:
            return path
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:
            raise OSError("Windows path shortening is unavailable in this Python runtime")
        kernel32 = win_dll("kernel32", use_last_error=True)
        get_short_path = kernel32.GetShortPathNameW
        get_short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        get_short_path.restype = ctypes.c_uint32
        buffer = ctypes.create_unicode_buffer(32_768)
        length = get_short_path(str(path), buffer, len(buffer))
        if not length or length >= len(buffer):
            error = ctypes.get_last_error()
            raise OSError(
                error,
                "The model path is too long for llama.cpp and Windows could not create "
                "a short path. Move the LM Atelier data folder closer to the drive root.",
                str(path),
            )
        shortened = Path(buffer.value)
        if len(str(shortened)) >= 260:
            raise OSError(
                "The model path remains too long for llama.cpp after Windows path shortening"
            )
        return shortened

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
