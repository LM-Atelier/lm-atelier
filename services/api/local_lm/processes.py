from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

import httpx
import psutil

from .comfy_editor_bridge import (
    ComfyEditorBridgeError,
    ComfyEditorBridgeSupport,
    prepare_comfy_editor_bridge,
)
from .comfy_registry_paths import registry_wheel_environment_root
from .config import Settings
from .events import EventBroker
from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    create_entry,
    discard_entry,
    open_child_directory,
    read_entry,
    rename_entry,
    sync_directory,
)
from .gguf import GGUFSelectionError, automatic_mmproj_selection, validate_gguf_selection
from .model_manifests import COMFY_MODEL_FOLDERS, comfy_folder_for_kind
from .models import ModelAssetInstall, ModelInstall, ModelProfile
from .network import shared_tls_context
from .schemas import WorkerStatus
from .security import trusted_browser_origins
from .subprocess_env import python_subprocess_environment
from .worker_failures import WorkerFailure, WorkerFailureCode, classify_worker_failure

if TYPE_CHECKING:
    from .comfy_registry_installs import ComfyRegistryLaunchContract
    from .runtime_provisioning import RuntimeProvisioner
    from .workflow_activations import WorkflowActivationLaunchScope


STATE_REFUSED = "LM Atelier's state folder may not be a filesystem link"


class ProcessStateError(RuntimeError):
    """State beneath the data folder could not be reached or published safely."""


@contextlib.contextmanager
def _held_state(state_dir: Path, *children: str) -> Iterator[AnchoredDirectory]:
    """Hold the state folder, and any child, for a whole operation.

    Every missing component is established THROUGH its held parent rather than
    by a pathname mkdir, so a redirection anywhere in the ancestry refuses
    before anything below it is created - creation cannot precede containment.

    The chain is retained as it descends instead of being released: an adopted
    child holds only its own handle, so letting an ancestor go would hand back
    the thing being held.
    """

    root = state_dir.expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    held: list[AnchoredDirectory] = []
    try:
        anchor = AnchoredDirectory(Path(root.parts[0]))
        held.append(anchor)
        for component in (*root.parts[1:], *children):
            anchor = open_child_directory(anchor, component, create=True)
            held.append(anchor)
        yield anchor
    except AnchoredDirectoryError as exc:
        raise ProcessStateError(STATE_REFUSED) from exc
    finally:
        for entry in reversed(held):
            entry.close()


def _publish_bytes(anchor: AnchoredDirectory, name: str, payload: bytes) -> None:
    """Publish `payload` as `name` inside the held directory, atomically.

    Staged under a name that cannot be predicted or reused and created
    EXCLUSIVELY, then moved into place. A fixed ".tmp" sibling can be
    pre-planted as a link, and a plain write follows one - so guarding only the
    published name protects everything except the file the bytes travel
    through. An entry abandoned by an interrupted run is inert here rather than
    blocking a later write.
    """

    staging = ""
    descriptor = -1
    for _attempt in range(5):
        staging = f"{name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = create_entry(anchor, staging)
            break
        except AnchoredEntryExists:
            continue
        except AnchoredDirectoryError as exc:
            raise ProcessStateError(STATE_REFUSED) from exc
    if descriptor < 0:
        raise ProcessStateError(STATE_REFUSED)
    try:
        handle = os.fdopen(descriptor, "wb")
    except BaseException:
        # fdopen takes ownership only once it succeeds, so every failure
        # BEFORE it has to close the descriptor itself.
        with contextlib.suppress(OSError):
            os.close(descriptor)
        with contextlib.suppress(Exception):
            discard_entry(anchor, staging)
        raise
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(Exception):
            discard_entry(anchor, staging)
        raise
    try:
        # Publishing over the previous file is the point, so replacement is
        # requested explicitly rather than left to differ by platform.
        rename_entry(anchor, staging, name, replace=True)
        # The record is durable; the directory ENTRY naming it is not until
        # the directory itself is synced.
        sync_directory(anchor)
    except AnchoredDirectoryError as exc:
        with contextlib.suppress(Exception):
            discard_entry(anchor, staging)
        raise ProcessStateError(STATE_REFUSED) from exc


logger = logging.getLogger(__name__)

WORKER_LOG_MAX_BYTES = 2 * 1024 * 1024
WORKER_LOG_BACKUP_COUNT = 3
WORKER_STDERR_TAIL_BYTES = 16 * 1024
WORKER_STDERR_DISPLAY_CHARS = 2_000
WORKER_STDERR_DISPLAY_LINES = 12
# asyncio's proactor loop tears a lost connection down inside a bare `finally:`,
# after it has already delivered `connection_lost` to the protocol. So a failure
# there - Windows reports WSAEINVAL (10022) or a reset - is logged once the
# connection is fully handled, and nothing was dropped. The block is nine lines,
# which is most of the twelve a user is shown when a worker dies, so leaving it
# in the tail lets routine teardown noise hide the actual cause of a crash. It
# stays in the worker log file; only the failure diagnostic drops it.
WORKER_TEARDOWN_NOISE_MAX_LINES = 10
WORKER_HEALTH_REQUEST_TIMEOUT_SECONDS = 5.0
WORKER_HEALTH_INITIAL_DELAY_SECONDS = 0.25
WORKER_HEALTH_MAX_DELAY_SECONDS = 2.0
WORKER_LIVENESS_INTERVAL_SECONDS = 5.0
WORKER_LIVENESS_FAILURE_THRESHOLD = 3
COMFY_OBJECT_INFO_MAX_BYTES = 64 * 1024 * 1024
WINDOWS_CREATE_NO_WINDOW = 0x08000000
WINDOWS_SO_EXCLUSIVEADDRUSE = int(getattr(socket, "SO_EXCLUSIVEADDRUSE", -5))
#: How many times the bind is retried after reclaiming a port, and the gap
#: between attempts. A terminated listener can leave the address briefly
#: unbindable, and one attempt would report that transient as a refusal.
_PORT_RECLAIM_ATTEMPTS = 10
_PORT_RECLAIM_INTERVAL_SECONDS = 0.2
# What `localhost` stands for. The loopback address of each family and nothing
# else: the rest of 127.0.0.0/8 is loopback but is not localhost, and treating
# it as such would put a descendant on 127.0.0.2 back in range of termination.
_LOCALHOST_ADDRESSES = ("127.0.0.1", "::1")


# The official Windows ComfyUI runtime uses Python's isolated ``._pth`` layout,
# which deliberately ignores PYTHONPATH. Insert only the already verified
# Registry overlays as ordinary sys.path entries, then execute the exact ComfyUI
# entrypoint. Ordinary insertion is important: unlike site.addsitedir, it never
# evaluates executable lines from a wheel's .pth files.
_COMFY_REGISTRY_BOOTSTRAP = (
    "import os,runpy,sys;"
    "count=int(sys.argv[1]);"
    "paths=sys.argv[2:2+count];"
    "sys.argv=sys.argv[2+count:];"
    "sys.path[:0]=paths+[os.path.dirname(sys.argv[0])];"
    "runpy.run_path(sys.argv[0],run_name='__main__')"
)


def _with_comfy_registry_overlays(
    command: list[str],
    site_packages: tuple[Path, ...],
) -> list[str]:
    """Run ComfyUI with verified overlays while leaving every .pth inert."""
    if len(command) < 2 or not site_packages:
        raise ValueError("ComfyUI Registry overlay command is incomplete")
    overlays: list[str] = []
    for path in site_packages:
        resolved = path.resolve(strict=True)
        if resolved.name != "site-packages" or not resolved.is_dir():
            raise ValueError("ComfyUI Registry overlay path is invalid")
        overlays.append(str(resolved))
    if len(overlays) != len(set(overlays)):
        raise ValueError("ComfyUI Registry overlay paths contain duplicates")
    return [
        command[0],
        "-c",
        _COMFY_REGISTRY_BOOTSTRAP,
        str(len(overlays)),
        *overlays,
        *command[1:],
    ]


_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-_][0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
# asyncio composes the first line as "Exception in callback {callback}", but a
# worker rarely writes it bare: captured logs from the field carry a "[ERROR] "
# level tag or a timestamp and logger name in front of it, and a frozen runtime
# renders the callback with empty parentheses rather than "(None)". Anchoring on
# the start of the line matched only the shape reproduced in a bare interpreter
# and would have missed every real occurrence, so the prefix is skipped instead.
_TEARDOWN_PREFIX = rb"^.{0,120}?"
_TEARDOWN_CALLBACK = re.compile(
    _TEARDOWN_PREFIX + rb"Exception in callback _Proactor\w*\._call_connection_lost\b"
)
_TEARDOWN_CONTINUATION = re.compile(
    rb"^(?:\s"
    rb"|.{0,120}?handle: <Handle _Proactor\w*\._call_connection_lost"
    rb"|.{0,120}?Traceback \(most recent call last\):"
    rb"|.{0,120}?(?:OSError|ConnectionResetError|ConnectionAbortedError): \[WinError \d+\])"
)
# The block always ends on its exception line. Stopping there, rather than only
# on the line budget, is what keeps a real traceback that follows immediately
# from being swallowed along with it.
_TEARDOWN_TERMINAL = re.compile(
    _TEARDOWN_PREFIX + rb"(?:OSError|ConnectionResetError|ConnectionAbortedError): \[WinError \d+\]"
)
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


@dataclass(frozen=True)
class _ProcessIdentity:
    pid: int
    create_time: float


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
    stderr_pending: bytearray = field(default_factory=bytearray)
    suppressed_teardown_lines: int = 0
    output_task: asyncio.Task[None] | None = None
    monitor_task: asyncio.Task[None] | None = None
    failure_detail: str | None = None
    launch_scope_sha256: str | None = None
    editor_bridge_launch_id: str | None = None
    editor_bridge_support: ComfyEditorBridgeSupport | None = None


class ProcessSupervisor:
    """Owns engine subprocesses without ever invoking a shell."""

    def __init__(
        self,
        settings: Settings,
        runtimes: RuntimeProvisioner | None = None,
        events: EventBroker | None = None,
        *,
        liveness_interval_seconds: float = WORKER_LIVENESS_INTERVAL_SECONDS,
        liveness_failure_threshold: int = WORKER_LIVENESS_FAILURE_THRESHOLD,
    ) -> None:
        if liveness_interval_seconds <= 0:
            raise ValueError("worker liveness interval must be positive")
        if liveness_failure_threshold < 1:
            raise ValueError("worker liveness failure threshold must be positive")
        self.settings = settings
        self.runtimes = runtimes
        self.events = events
        self._liveness_interval_seconds = liveness_interval_seconds
        self._liveness_failure_threshold = liveness_failure_threshold
        self._workers: dict[str, WorkerRecord] = {}
        self._locks = {"chat": asyncio.Lock(), "media": asyncio.Lock()}
        self._identity_path = self.settings.state_dir / "worker-processes.json"
        self._identity_lock = threading.Lock()
        self._worker_identities = self._load_worker_identities()
        self._reap_persisted_workers()

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
            failure = WorkerFailure(WorkerFailureCode.UNKNOWN, None)
            if record and not running:
                exit_code = (
                    record.process.returncode
                    if record.process.returncode is not None
                    else "unknown"
                )
                failure_detail = record.failure_detail or (
                    f"{record.name} worker exited with code {exit_code}."
                )
                failure = classify_worker_failure(
                    failure_detail=failure_detail,
                    # Classify against everything retained, not the twelve lines
                    # shown. The line that names an out-of-memory failure is
                    # printed when the allocation fails and can be well above the
                    # shutdown chatter that follows it.
                    stderr_tail=self._retained_stderr(record),
                    exit_code=record.process.returncode,
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
                    failure_code=(
                        failure.code if failure.code != WorkerFailureCode.UNKNOWN else None
                    ),
                    failure_remedy=failure.remedy,
                    stderr_tail=stderr_tail,
                    log_path=self._public_log_path(name),
                )
            )
        return result

    def launch_scope_sha256(self, name: str) -> str | None:
        """Return the live ready worker's exact activation scope, if it has one."""

        record = self._workers.get(name)
        if record is None or record.process.returncode is not None or record.state != "ready":
            return None
        return record.launch_scope_sha256

    def workflow_editor_runtime_identity(self) -> str | None:
        """Return authority for the ready launch that whitelisted the verified bridge."""

        record = self._workers.get("media")
        if record is None or record.process.returncode is not None or record.state != "ready":
            return None
        support = record.editor_bridge_support
        if support is None or not support.supported:
            return None
        return record.editor_bridge_launch_id

    def workflow_editor_bridge_support(self) -> ComfyEditorBridgeSupport | None:
        """Return the exact editor support fact bound to the live ready media launch."""

        record = self._workers.get("media")
        if record is None or record.process.returncode is not None or record.state != "ready":
            return None
        support = record.editor_bridge_support
        if support is None:
            return None
        if support.supported != (record.editor_bridge_launch_id is not None):
            return None
        return support

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
        except (Exception, asyncio.CancelledError):
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
        except (Exception, asyncio.CancelledError):
            self.settings.chat_engine = previous_engine
            raise
        return self.statuses()[0]

    async def start_media(
        self,
        provisional_model_paths: tuple[Path, dict[str, str]] | None = None,
        *,
        phase_callback: Callable[[str], Awaitable[None]] | None = None,
        activation_scope: WorkflowActivationLaunchScope | None = None,
    ) -> WorkerStatus:
        async def report_phase(phase: str) -> None:
            if phase_callback is None:
                return
            try:
                await phase_callback(phase)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Could not publish media startup phase", exc_info=True)

        if provisional_model_paths is not None and activation_scope is not None:
            raise ValueError("Provisional model paths cannot broaden an activation-scoped launch")
        if (
            not self.settings.comfy_executable
            or not self.settings.comfy_executable.is_file()
            or not self.settings.comfy_directory
            or not (self.settings.comfy_directory / "main.py").is_file()
        ) and self.runtimes:
            await report_phase("Provisioning media runtime")
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
        editor_bridge_support: ComfyEditorBridgeSupport | None = None
        await report_phase("Validating media dependencies")
        custom_node_types: tuple[str, ...] = ()
        if activation_scope is None:
            trusted_custom_nodes = await self._trusted_comfy_node_folders()
            registry_contract = await asyncio.to_thread(self._trusted_comfy_registry_contract)
        else:
            trusted_custom_nodes, custom_node_types = await self._scoped_comfy_node_folders(
                activation_scope
            )
            registry_contract = await asyncio.to_thread(
                self._scoped_comfy_registry_contract,
                activation_scope,
            )
        if registry_contract.runtime_distributions:
            await report_phase("Verifying media runtime packages")
            from .comfy_registry_interpreter import (
                ComfyRegistryInterpreterError,
                probe_comfy_registry_runtime_target,
            )

            try:
                _, _, current_runtime_distributions = await probe_comfy_registry_runtime_target(
                    executable
                )
            except ComfyRegistryInterpreterError as exc:
                raise RuntimeError(
                    "The managed media runtime package baseline could not be verified."
                ) from exc
            if current_runtime_distributions != registry_contract.runtime_distributions:
                raise RuntimeError(
                    "The managed media runtime changed after workflow dependencies "
                    "were prepared. Prepare the workflow package again."
                )
        try:
            editor_bridge = await asyncio.to_thread(
                prepare_comfy_editor_bridge,
                comfy_executable=executable,
                comfy_directory=directory,
                custom_node_root=self.settings.custom_node_dir,
                coordinator_origins=trusted_browser_origins(self.settings),
            )
        except ComfyEditorBridgeError as exc:
            editor_bridge_support = ComfyEditorBridgeSupport(False, exc.code, str(exc))
            logger.warning("Native workflow editing is unavailable: %s", exc)
        else:
            editor_bridge_support = editor_bridge.support
            if editor_bridge.folder is not None and editor_bridge.support.supported:
                trusted_custom_nodes.append(editor_bridge.folder.name)
            elif editor_bridge.folder is not None or editor_bridge.support.supported:
                editor_bridge_support = ComfyEditorBridgeSupport(
                    False,
                    "workflow-editor-bridge-staging-failed",
                    "The verified workflow editor bridge was not staged for this launch.",
                    editor_bridge.support.comfyui_version,
                    editor_bridge.support.frontend_version,
                )
                logger.warning(
                    "Native workflow editing is unavailable: %s",
                    editor_bridge_support.message,
                )
            elif editor_bridge.support.code != "workflow-editor-runtime-unavailable":
                logger.debug(
                    "Native workflow editing is unavailable: %s",
                    editor_bridge.support.message,
                )
        trusted_custom_nodes = sorted(
            {*trusted_custom_nodes, *registry_contract.custom_node_folders}
        )
        output_directory = self.settings.comfy_output_dir.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        environment_overrides = (
            {"PYTHONDONTWRITEBYTECODE": "1"} if registry_contract.site_packages else None
        )
        if activation_scope is not None:
            model_paths_config = self._write_scoped_comfy_model_paths(activation_scope)
        else:
            model_paths_config = (
                self._write_comfy_model_paths(provisional_model_paths)
                if provisional_model_paths is not None
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
        if registry_contract.site_packages:
            command = _with_comfy_registry_overlays(
                command,
                registry_contract.site_packages,
            )
        if trusted_custom_nodes:
            command.extend(["--whitelist-custom-nodes", *trusted_custom_nodes])
        await report_phase("Starting media runtime")
        if activation_scope is not None:
            expected_node_types = tuple(sorted({*custom_node_types, *registry_contract.node_types}))
            await self._replace(
                "media",
                command,
                self.settings.comfy_url + "/system_stats",
                environment_overrides=environment_overrides,
                ready_check=(
                    (lambda: self._verify_comfy_node_types(expected_node_types))
                    if expected_node_types
                    else None
                ),
                launch_scope_sha256=activation_scope.launch_sha256,
                editor_bridge_support=editor_bridge_support,
            )
        elif registry_contract.site_packages:
            await self._replace(
                "media",
                command,
                self.settings.comfy_url + "/system_stats",
                environment_overrides=environment_overrides,
                ready_check=lambda: self._verify_comfy_node_types(registry_contract.node_types),
                editor_bridge_support=editor_bridge_support,
            )
        else:
            await self._replace(
                "media",
                command,
                self.settings.comfy_url + "/system_stats",
                editor_bridge_support=editor_bridge_support,
            )
        return self.statuses()[1]

    def _scoped_comfy_registry_contract(
        self, scope: WorkflowActivationLaunchScope
    ) -> ComfyRegistryLaunchContract:
        from .comfy_registry_installs import scoped_comfy_registry_launch_contract
        from .db import SessionLocal

        if tuple(item.registry_install_id for item in scope.registry_packages) != (
            scope.registry_install_ids
        ):
            raise ValueError("Workflow activation Registry package scope is inconsistent")
        with SessionLocal() as session:
            return scoped_comfy_registry_launch_contract(
                session,
                scope.registry_packages,
                custom_node_root=self.settings.custom_node_dir,
                environment_root=registry_wheel_environment_root(self.settings.registry_dir),
            )

    def _trusted_comfy_registry_contract(self) -> ComfyRegistryLaunchContract:
        from .comfy_registry_installs import trusted_comfy_registry_launch_contract
        from .db import SessionLocal

        with SessionLocal() as session:
            return trusted_comfy_registry_launch_contract(
                session,
                custom_node_root=self.settings.custom_node_dir,
                environment_root=registry_wheel_environment_root(self.settings.registry_dir),
            )

    def _trusted_comfy_registry_package_node_types(
        self,
    ) -> dict[tuple[str, str], frozenset[str]]:
        from sqlalchemy import select

        from .comfy_registry_installs import trusted_comfy_registry_launch_contract
        from .db import SessionLocal
        from .models import ComfyRegistryInstall

        with SessionLocal() as session:
            contract = trusted_comfy_registry_launch_contract(
                session,
                custom_node_root=self.settings.custom_node_dir,
                environment_root=registry_wheel_environment_root(self.settings.registry_dir),
            )
            installs = session.scalars(
                select(ComfyRegistryInstall).where(
                    ComfyRegistryInstall.trusted.is_(True),
                    ComfyRegistryInstall.active.is_(True),
                )
            ).all()
            result: dict[tuple[str, str], frozenset[str]] = {}
            for install in installs:
                key = (install.package_id, install.package_version)
                if key in result:
                    raise RuntimeError("Registry launch package identity is duplicated")
                result[key] = frozenset(str(value) for value in install.node_types_json)
        declared_node_types = {node_type for values in result.values() for node_type in values}
        if declared_node_types != set(contract.node_types):
            raise RuntimeError("Registry launch package node ownership is inconsistent")
        return result

    async def trusted_comfy_registry_package_node_types(
        self,
    ) -> dict[tuple[str, str], frozenset[str]]:
        """Node ownership an unscoped launch can load from verified installs.

        This deliberately runs the same disk and environment verification used
        by ``start_media``. Database declarations alone are not enough to make
        a package importable, and package/version ownership prevents one
        installed package from laundering another package's node types. The
        returned names are not compiler schemas; callers must still start the
        worker and read its live ``object_info`` before compilation.
        """

        return await asyncio.to_thread(self._trusted_comfy_registry_package_node_types)

    async def trusted_comfy_custom_node_package_node_types(
        self,
    ) -> dict[tuple[str, str], frozenset[str]]:
        """Reviewed ownership from exact trusted manual installs on disk."""

        from sqlalchemy import select

        from .custom_nodes import CustomNodeManager, reviewed_custom_node_types
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
                    .order_by(CustomNodeInstall.name, CustomNodeInstall.revision)
                ).all()
            )
            manager = CustomNodeManager(self.settings)
            for install in installs:
                session.expunge(install)
        result: dict[tuple[str, str], frozenset[str]] = {}
        for install in installs:
            await manager.verify(install)
            node_types = reviewed_custom_node_types(install.security_json)
            if not node_types:
                continue
            key = (install.name, install.revision)
            if key in result:
                raise RuntimeError("Manual custom-node package identity is duplicated")
            result[key] = frozenset(node_types)
        return result

    async def trusted_comfy_custom_node_packages_awaiting_review(
        self,
    ) -> frozenset[tuple[str, str]]:
        """Trusted installs whose reviewed node inventory was never recorded.

        These are indistinguishable from a package nobody installed, as far as
        resolution is concerned: the inventory is the evidence, and without it
        nothing can say this package owns the nodes a workflow attributes to it.
        They are not the same thing to the person looking at them, though. One
        needs installing and the other needs reading, and an installation that
        reports the first when it means the second sends somebody to fetch what
        they already have.

        This exists because the inventory began being recorded after people had
        already trusted packages. Upgrading does not retroactively review
        anything - trust says somebody read that exact revision, and nothing here
        can assert that on their behalf - so the honest answer is to name the
        state and let them re-read it.
        """

        from sqlalchemy import select

        from .custom_nodes import reviewed_custom_node_types
        from .db import SessionLocal
        from .models import CustomNodeInstall

        with SessionLocal() as session:
            installs = list(
                session.scalars(
                    select(CustomNodeInstall).where(
                        CustomNodeInstall.active.is_(True),
                        CustomNodeInstall.trusted.is_(True),
                    )
                ).all()
            )
            return frozenset(
                (install.name, install.revision)
                for install in installs
                if not reviewed_custom_node_types(install.security_json)
            )

    async def comfy_node_inventory(self) -> frozenset[str]:
        """Every node type the running worker loaded.

        One read for two callers. Startup verification and an omission proof
        must never parse different inventories, because the whole force of the
        proof is that it saw what startup saw.
        """
        payload = bytearray()
        async with (
            httpx.AsyncClient(
                trust_env=False,
                verify=shared_tls_context(trust_environment=False),
            ) as client,
            client.stream(
                "GET",
                self.settings.comfy_url + "/object_info",
                timeout=WORKER_HEALTH_REQUEST_TIMEOUT_SECONDS,
            ) as response,
        ):
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                payload.extend(chunk)
                if len(payload) > COMFY_OBJECT_INFO_MAX_BYTES:
                    raise RuntimeError("ComfyUI node inventory exceeds the response limit")
        try:
            inventory = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("ComfyUI returned an invalid node inventory") from exc
        if not isinstance(inventory, dict):
            raise RuntimeError("ComfyUI returned an invalid node inventory")
        return frozenset(str(key) for key in inventory)

    async def _verify_comfy_node_types(self, expected: tuple[str, ...]) -> None:
        if not expected:
            return
        missing = sorted(set(expected) - await self.comfy_node_inventory())
        if missing:
            raise RuntimeError(f"ComfyUI did not load required node type {missing[0]}")

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

    async def _scoped_comfy_node_folders(
        self, scope: WorkflowActivationLaunchScope
    ) -> tuple[list[str], tuple[str, ...]]:
        from .custom_nodes import CustomNodeManager
        from .db import SessionLocal
        from .models import CustomNodeInstall
        from .workflow_bindings import materialize_custom_node

        if tuple(item.custom_node_install_id for item in scope.custom_nodes) != (
            scope.custom_node_install_ids
        ):
            raise ValueError("Workflow activation custom-node scope is inconsistent")
        manager = CustomNodeManager(self.settings)
        installs: list[CustomNodeInstall] = []
        node_types: set[str] = set()
        with SessionLocal() as session:
            for binding in scope.custom_nodes:
                install = session.get(CustomNodeInstall, binding.custom_node_install_id)
                raw_node_types = (
                    install.security_json.get("node_types")
                    if install is not None and isinstance(install.security_json, dict)
                    else None
                )
                declared = (
                    tuple(sorted(raw_node_types, key=lambda item: (item.casefold(), item)))
                    if isinstance(raw_node_types, list)
                    and all(isinstance(item, str) and item for item in raw_node_types)
                    else ()
                )
                expected_path = (
                    (self.settings.custom_node_dir / install.installed_path).resolve()
                    if install is not None
                    else None
                )
                identity = materialize_custom_node(install).identity if install is not None else {}
                if (
                    install is None
                    or not install.active
                    or not install.trusted
                    or expected_path != binding.installed_path
                    or identity.get("source_url") != binding.source_url
                    or identity.get("revision") != binding.revision
                    or identity.get("tree_hash") != binding.tree_hash
                    or declared != binding.node_types
                ):
                    raise ValueError("Workflow activation custom-node identity changed")
                installs.append(install)
                node_types.update(declared)
                session.expunge(install)
        for install in installs:
            await manager.verify(install)
        return [install.installed_path for install in installs], tuple(sorted(node_types))

    def _write_scoped_comfy_model_paths(self, scope: WorkflowActivationLaunchScope) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", scope.launch_sha256):
            raise ValueError("Workflow activation launch identity is invalid")
        if tuple(item.model_install_id for item in scope.models) != scope.model_install_ids:
            raise ValueError("Workflow activation model scope is inconsistent")
        if tuple(item.model_asset_install_id for item in scope.assets) != (
            scope.model_asset_install_ids
        ):
            raise ValueError("Workflow activation asset scope is inconsistent")
        config: dict[str, dict[str, str]] = {}
        for model in scope.models:
            paths = dict(model.comfy_paths)
            if self._validated_comfy_paths(paths) != paths or not model.base_path.is_dir():
                raise ValueError("Workflow activation model path scope is invalid")
            config[f"local_lm_{len(config) + 1}"] = {
                "base_path": str(model.base_path),
                **paths,
            }
        for asset in scope.assets:
            if (
                asset.loader_folder not in self._validated_comfy_paths({asset.loader_folder: "."})
                or not asset.base_path.is_dir()
            ):
                raise ValueError("Workflow activation asset path scope is invalid")
            config[f"local_lm_{len(config) + 1}"] = {
                "base_path": str(asset.base_path),
                asset.loader_folder: ".",
            }
        name = f"{scope.launch_sha256}.yaml"
        payload = json.dumps(config, indent=2).encode("utf-8")
        with _held_state(self.settings.state_dir, "comfy-launch") as anchor:
            _publish_bytes(anchor, name, payload)
        return self.settings.state_dir / "comfy-launch" / name

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
                folder = comfy_folder_for_kind(asset.kind)
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
        name = "comfy-extra-model-paths.yaml"
        payload = json.dumps(config, indent=2).encode("utf-8")
        with _held_state(self.settings.state_dir) as anchor:
            _publish_bytes(anchor, name, payload)
        return self.settings.state_dir / name

    @staticmethod
    def _validated_comfy_paths(value: object) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, str] = {}
        for key, item in value.items():
            if key not in COMFY_MODEL_FOLDERS or not isinstance(item, str):
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
        environment_overrides: Mapping[str, str] | None = None,
        ready_check: Callable[[], Awaitable[None]] | None = None,
        launch_scope_sha256: str | None = None,
        editor_bridge_support: ComfyEditorBridgeSupport | None = None,
    ) -> None:
        if launch_scope_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", launch_scope_sha256
        ):
            raise ValueError("Worker launch scope identity is invalid")
        async with self._locks[name]:
            current = self._workers.get(name)
            if (
                launch_scope_sha256 is not None
                and current is not None
                and current.process.returncode is None
                and current.state == "ready"
                and current.command == command
                and current.launch_scope_sha256 == launch_scope_sha256
                and current.editor_bridge_support == editor_bridge_support
                and (current.editor_bridge_launch_id is not None)
                == bool(editor_bridge_support and editor_bridge_support.supported)
            ):
                return
            await self._stop_unlocked(name)
            await self._reclaim_port_from_our_own_children(name, health_url)
            await self._ensure_port_available(name, health_url)
            startup_started_at = time.perf_counter()
            log_path = self.settings.log_dir / f"{name}-worker.log"
            worker_log = _RotatingWorkerLog(log_path)
            try:
                environment = python_subprocess_environment(overrides=environment_overrides)
                if os.name == "nt":
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=environment,
                        start_new_session=True,
                        creationflags=WINDOWS_CREATE_NO_WINDOW,
                    )
                else:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=environment,
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
                launch_scope_sha256=launch_scope_sha256,
                editor_bridge_launch_id=(
                    secrets.token_urlsafe(24)
                    if editor_bridge_support and editor_bridge_support.supported
                    else None
                ),
                editor_bridge_support=editor_bridge_support,
            )
            record.output_task = asyncio.create_task(self._capture_process_output(record))
            self._workers[name] = record
            try:
                if process.returncode is None:
                    self._record_worker_process_tree(name, process.pid)
                await self._wait_healthy(record, health_url)
                if ready_check is not None:
                    await ready_check()
                record.startup_duration_ms = round(
                    (time.perf_counter() - startup_started_at) * 1_000
                )
                record.state = "ready"
                record.monitor_task = asyncio.create_task(
                    self._monitor_worker(record, health_url),
                    name=f"monitor-{name}-worker",
                )
                await self._publish_worker_event("worker.ready", record, state="ready")
            except asyncio.CancelledError:
                cleanup = asyncio.create_task(self._terminate_record(record))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    with contextlib.suppress(Exception):
                        await cleanup
                except Exception:
                    logger.exception("Could not clean up cancelled %s worker start", name)
                finally:
                    if self._workers.get(name) is record:
                        self._workers.pop(name)
                raise
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
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.worker_startup_seconds
        delay = WORKER_HEALTH_INITIAL_DELAY_SECONDS
        async with httpx.AsyncClient(
            trust_env=False,
            verify=shared_tls_context(trust_environment=False),
        ) as client:
            while loop.time() < deadline:
                if record.process.returncode is not None:
                    raise RuntimeError(
                        f"{record.name} worker exited with code {record.process.returncode}"
                    )
                self._record_worker_process_tree(record.name, record.process.pid)
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    response = await client.get(
                        url,
                        timeout=min(WORKER_HEALTH_REQUEST_TIMEOUT_SECONDS, remaining),
                    )
                    if response.is_success:
                        if self._listener_owned_by_worker(record.process.pid, url):
                            return
                        raise RuntimeError(
                            f"{record.name} worker health endpoint is served by another process"
                        )
                except httpx.HTTPError:
                    pass
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 2, WORKER_HEALTH_MAX_DELAY_SECONDS)
        raise TimeoutError(f"{record.name} worker did not become healthy before timeout")

    async def _probe_worker_health(
        self,
        client: httpx.AsyncClient,
        record: WorkerRecord,
        url: str,
    ) -> bool:
        if record.process.returncode is not None:
            return False
        try:
            response = await client.get(url, timeout=WORKER_HEALTH_REQUEST_TIMEOUT_SECONDS)
        except httpx.HTTPError:
            return False
        if not response.is_success:
            return False
        if not self._listener_owned_by_worker(record.process.pid, url):
            return False
        self._record_worker_process_tree(record.name, record.process.pid)
        return True

    async def _monitor_worker(self, record: WorkerRecord, health_url: str) -> None:
        try:
            await self._monitor_worker_loop(record, health_url)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("%s worker supervision failed", record.name)
            try:
                await self._record_monitor_failure(record, exc)
            except Exception:
                logger.exception("Could not stop %s after supervision failure", record.name)

    async def _monitor_worker_loop(self, record: WorkerRecord, health_url: str) -> None:
        process_wait = asyncio.create_task(
            record.process.wait(),
            name=f"wait-{record.name}-worker",
        )
        failures = 0
        try:
            async with httpx.AsyncClient(
                trust_env=False,
                verify=shared_tls_context(trust_environment=False),
            ) as client:
                while True:
                    done, _ = await asyncio.wait(
                        {process_wait},
                        timeout=self._liveness_interval_seconds,
                    )
                    if process_wait in done:
                        await self._record_unexpected_exit(record, process_wait.result())
                        return
                    if self._workers.get(record.name) is not record:
                        return
                    if await self._probe_worker_health(client, record, health_url):
                        failures = 0
                        continue
                    failures += 1
                    if failures < self._liveness_failure_threshold:
                        continue
                    async with self._locks[record.name]:
                        if self._workers.get(record.name) is not record:
                            return
                        record.failure_detail = (
                            f"{record.name} worker stopped responding to health checks."
                        )
                        await self._terminate_record(record, cancel_monitor=False)
                        await self._publish_worker_event(
                            "worker.unhealthy",
                            record,
                            state="exited",
                            exit_code=record.process.returncode,
                        )
                    return
        finally:
            if not process_wait.done():
                process_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await process_wait

    async def _record_monitor_failure(
        self,
        record: WorkerRecord,
        exc: Exception,
    ) -> None:
        async with self._locks[record.name]:
            if self._workers.get(record.name) is not record:
                return
            detail = self._sanitize_diagnostic(str(exc)).rstrip(".")
            suffix = f": {detail}" if detail else ""
            record.failure_detail = f"{record.name} worker supervision failed{suffix}."
            await self._terminate_record(record, cancel_monitor=False)
            await self._publish_worker_event(
                "worker.unhealthy",
                record,
                state="exited",
                exit_code=record.process.returncode,
            )

    async def _record_unexpected_exit(
        self,
        record: WorkerRecord,
        exit_code: int,
    ) -> None:
        async with self._locks[record.name]:
            if self._workers.get(record.name) is not record:
                return
            record.failure_detail = f"{record.name} worker exited with code {exit_code}."
            await self._publish_worker_event(
                "worker.exited",
                record,
                state="exited",
                exit_code=exit_code,
            )
            await self._terminate_record(record, cancel_monitor=False)

    async def _publish_worker_event(
        self,
        event_type: str,
        record: WorkerRecord,
        *,
        state: str,
        exit_code: int | None = None,
    ) -> None:
        if self.events is None:
            return
        await self.events.publish(
            event_type,
            record.name,
            {
                "name": record.name,
                "state": state,
                "exit_code": exit_code,
            },
        )

    @staticmethod
    def _target_addresses(
        host: str,
    ) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
        """The literal addresses this host names, or empty if it names none we can prove.

        `localhost` is admitted because the settings validator admits it: worker
        URLs must be loopback and it accepts `localhost` alongside literal
        loopback addresses, so it is a supported configuration rather than an
        exotic one. Refusing it here would have left the recovery silently
        inoperative for anyone who wrote their URLs that way, which a mutation
        found before a reader would have.

        It stands for the loopback address of each family and nothing else. The
        rest of 127.0.0.0/8 is loopback but is not localhost, so a descendant on
        127.0.0.2 stays out of range exactly as it is for a literal target.

        Any other name resolves through machinery this code does not control,
        and an endpoint it cannot prove is not a basis for terminating anything.
        """

        if host == "localhost":
            return tuple(ipaddress.ip_address(alias) for alias in _LOCALHOST_ADDRESSES)
        try:
            return (ipaddress.ip_address(host),)
        except ValueError:
            return ()

    @staticmethod
    def _own_descendants() -> list[psutil.Process]:
        """Every process this one parented, at any depth.

        Its own method so the selection above can be exercised without spawning
        a real process tree. The platform decides which listeners can coexist at
        one port, so a test that had to arrange a real exact holder AND a real
        wildcard holder together could only run on the platform where that is
        possible - and the rule it would be checking is the one that must hold
        on both.
        """

        try:
            return list(psutil.Process(os.getpid()).children(recursive=True))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return []

    @staticmethod
    def _listening_match(process: psutil.Process, host: str, port: int) -> str | None:
        """How this process's listeners relate to host:port - exact, wildcard, or not.

        The PORT NUMBER IS NOT THE ENDPOINT, and matching on it alone authorises
        a termination this cannot justify. A sibling of ours listening on
        another loopback address at the same number is not what stands between
        the worker and the address it is about to bind: killing it frees nothing
        and destroys something that was working. An earlier version of this did
        exactly that, because it compared `laddr.port` and discarded the host it
        had already parsed.

        The two kinds are reported separately rather than merged into one
        boolean because whether a wildcard listener BLOCKS a specific bind is a
        property of the platform, not of this application. Measured here rather
        than assumed, on the machine the incident came from:

            holder 0.0.0.0    -> binding 127.0.0.1 at the same port SUCCEEDS
            holder 127.0.0.2  -> binding 127.0.0.1 at the same port SUCCEEDS
            holder 127.0.0.1  -> binding 127.0.0.1 at the same port FAILS

        with and without SO_REUSEADDR on the holder, so it is the wildcard rule
        itself and not a socket option. On Windows a specific address may be
        bound alongside a wildcard; on POSIX it may not. `_own_descendants_
        blocking` uses the distinction rather than guessing which platform it is
        on.

        WILDCARD EVIDENCE IS SAME-FAMILY ONLY, and this fails closed on purpose.
        A socket bound to `::` accepts IPv4 connections wherever the platform
        leaves IPV6_V6ONLY off - but psutil cannot report that flag, so whether a
        particular `::` listener holds an IPv4 endpoint is exactly what this
        cannot observe. An earlier version accepted it anyway and reasoned that
        the endpoint had already been measured busy; that reasoning is wrong,
        because the thing making it busy can be a FOREIGN IPv4 holder while an
        IPv6-only child of ours sits innocently at the same number. It would then
        be selected and killed for a bind it could never have blocked. Unprovable
        evidence must not authorise a termination, so cross-family wildcards
        report nothing.
        """

        # BOUND TO THE FAMILY THE PROBE ACTUALLY USES. `_port_is_free` and
        # `_ensure_port_available` both choose the family from whether the host
        # text contains a colon, so `localhost` is probed as IPv4 only. Selecting
        # against both families while probing one would let an IPv4 busy result
        # authorise terminating an exact ::1 descendant that had nothing to do
        # with it. The evidence and the decision must be about one endpoint.
        probed_version = 6 if ":" in host else 4
        targets = tuple(
            address
            for address in ProcessSupervisor._target_addresses(host)
            if address.version == probed_version
        )
        if not targets:
            return None
        families = {target.version for target in targets}
        try:
            connections = process.net_connections(kind="tcp")
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            return None
        wildcard = False
        for connection in connections:
            if connection.status != psutil.CONN_LISTEN:
                continue
            address = connection.laddr
            listening_host = getattr(address, "ip", None)
            listening_port = getattr(address, "port", None)
            if listening_host is None and len(address) > 1:
                listening_host, listening_port = address[0], address[1]
            if listening_port != port or not listening_host:
                continue
            try:
                # A scope id makes an address unparsable and says nothing about
                # which address it is, so it is dropped rather than guessed at.
                listening = ipaddress.ip_address(str(listening_host).split("%", 1)[0])
            except ValueError:
                continue
            if listening in targets:
                return "exact"
            if listening.is_unspecified and listening.version in families:
                wildcard = True
        return "wildcard" if wildcard else None

    def _pids_of_other_workers(self, name: str) -> set[int]:
        """Every process another managed worker owns, live or persisted.

        Nothing requires two workers to be configured on DIFFERENT endpoints.
        `validate_worker_url` checks each URL is loopback and stops there, so
        `llama_url` and `comfy_url` may name the same host and port. If they do,
        the other worker is a descendant of ours listening on exactly the address
        this one is about to bind - indistinguishable, by endpoint alone, from
        the lost worker being reclaimed.

        Endpoint ownership is therefore not enough to authorise a termination:
        it establishes that a process is in the way, not that it is THIS
        worker's to remove. A live sibling is something the ordinary preflight
        would have refused over; killing it first turns a clean refusal into a
        stopped service.
        """

        pids: set[int] = set()
        for worker, record in self._workers.items():
            if worker == name:
                continue
            pid = getattr(getattr(record, "process", None), "pid", None)
            if isinstance(pid, int):
                pids.add(pid)
        for worker, identities in self._worker_identities.items():
            if worker == name:
                continue
            for identity in identities:
                pid = getattr(identity, "pid", None)
                if isinstance(pid, int):
                    pids.add(pid)
        return pids

    @staticmethod
    def _within_any(process: psutil.Process, pids: set[int]) -> bool:
        """Whether this process IS, or is inside the tree of, one of those pids.

        The subtree matters as much as the process. `_terminate_processes` takes
        the whole tree below whatever it is given, so selecting a CHILD of
        another worker would tear down part of that worker just as surely as
        selecting the worker itself.
        """

        if process.pid in pids:
            return True
        try:
            return any(parent.pid in pids for parent in process.parents())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            # Unreadable ancestry cannot prove the process is ours to take.
            return True

    def _own_descendants_blocking(self, name: str, host: str, port: int) -> list[psutil.Process]:
        """Our own descendants blocking the endpoint, most specific evidence first.

        Asked of OUR process tree rather than of the endpoint. A global
        connection table lookup needs privileges this application does not have
        on Windows, and it would answer a different question anyway: who holds
        the address, rather than whether we own them. Walking down from
        `os.getpid()` answers only the second, which is the only one that
        authorises a termination.

        An exact holder is preferred and a wildcard holder is only a fallback,
        which makes this correct on both platforms without either being named.
        Where a wildcard does not block - Windows - an exact holder is the only
        thing that can be holding the address, so the fallback is never reached
        and a wildcard child of ours is left running. Where a wildcard does
        block - POSIX - the two cannot coexist at one port, so an empty exact
        tier is positive evidence that the wildcard is the blocker. Selecting
        both tiers at once would terminate a working bystander on Windows for
        no gain, which is the same class of harm as matching on the port alone.
        """

        others = self._pids_of_other_workers(name)
        # Skipped entirely when no other worker owns anything. Walking ancestry
        # to answer "is this inside an empty set" is work that cannot change the
        # answer, and asking a process for its parents is the one step here that
        # can refuse.
        candidates = [
            process
            for process in self._own_descendants()
            if not others or not self._within_any(process, others)
        ]
        matched = [(self._listening_match(process, host, port), process) for process in candidates]
        exact = [process for kind, process in matched if kind == "exact"]
        if exact:
            return exact
        return [process for kind, process in matched if kind == "wildcard"]

    async def _reclaim_port_from_our_own_children(self, name: str, url: str) -> None:
        """Free the port when this application is what is holding it.

        The persisted identities are the first way a lost worker is recognised,
        but they are not sufficient and assuming they were is what made an
        earlier version of this fix narrower than it claimed. They can be absent
        entirely - `_load_worker_identities` returns an empty mapping on a read
        error or malformed JSON - and they can be emptied while a process still
        holds the port, because the snapshot only ever covers the tree as it
        stood when it was taken, and `_refresh_worker_identities_after_stop`
        drops every recorded identity once those exact processes are gone. A
        descendant that appeared after the last snapshot is invisible to them.

        So ownership is established a second way, which needs no record at all:
        the process is a descendant of THIS process and is listening on an
        address that blocks the endpoint we are about to bind. That was true of
        the measured incident, where the listener's parent was the application
        itself. Both tests are required together - a descendant that blocks no
        endpoint of ours is left alone, and a listener we did not parent is not
        ours to kill.

        The first of those is narrower than it sounds and the limit is worth
        stating. Once a blocking descendant is selected, `_terminate_processes`
        takes its whole subtree, including members holding nothing. So what is
        left alone is every descendant OUTSIDE the selected process's own tree;
        a child of the blocker goes with it. That is intended - a worker's own
        children are part of the worker - but it is not the same claim.

        Not reached unless the endpoint is actually busy. The ordinary start
        path therefore pays one extra bind probe here, and then the preflight's
        own bind: two attempts on a free port, not one.
        """

        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
        if not host or not port:
            return
        if await asyncio.to_thread(self._port_is_free, host, port):
            return
        holders = await asyncio.to_thread(self._own_descendants_blocking, name, host, port)
        if not holders:
            # Something else holds it. Refusing is correct: we cannot prove we
            # own it, and killing by endpoint is not a recovery.
            return
        logger.info(
            "Reclaiming %s:%s from %s process(es) this application parented",
            host,
            port,
            len(holders),
        )
        await asyncio.to_thread(
            self._terminate_processes, holders, self.settings.worker_shutdown_seconds
        )
        self._refresh_worker_identities_after_stop(name)
        # A terminated listener can leave the address briefly unbindable, so the
        # bind is retried rather than judged on one attempt. Bounded: this only
        # runs when the port was busy and we have just terminated its owner.
        for _attempt in range(_PORT_RECLAIM_ATTEMPTS):
            if await asyncio.to_thread(self._port_is_free, host, port):
                return
            await asyncio.sleep(_PORT_RECLAIM_INTERVAL_SECONDS)

    @staticmethod
    def _port_is_free(host: str, port: int) -> bool:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        address: tuple[str, int] | tuple[str, int, int, int]
        address = (host, port, 0, 0) if family == socket.AF_INET6 else (host, port)
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                if os.name == "nt":
                    probe.setsockopt(socket.SOL_SOCKET, WINDOWS_SO_EXCLUSIVEADDRUSE, 1)
                else:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(address)
        except OSError:
            return False
        return True

    async def _ensure_port_available(self, name: str, url: str) -> None:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port
        if not host or not port:
            raise ValueError(f"{name} worker health URL has no host and port")
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        address: tuple[str, int] | tuple[str, int, int, int]
        address = (host, port, 0, 0) if family == socket.AF_INET6 else (host, port)
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                if os.name == "nt":
                    probe.setsockopt(socket.SOL_SOCKET, WINDOWS_SO_EXCLUSIVEADDRUSE, 1)
                else:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(address)
        except OSError as exc:
            occupant = self._port_listener_description(host, port)
            detail = f"{name} worker cannot start because {host}:{port} is already in use"
            raise OSError(f"{detail} by {occupant}" if occupant else detail) from exc

    def _port_listener_description(self, host: str, port: int) -> str | None:
        """Name the process listening on this endpoint, when the platform will say.

        Only ever called after a bind has already failed, so the ordinary start
        path pays nothing for it. Returns None rather than guessing: an
        unidentifiable occupant leaves the original message exactly as it was,
        because a wrong name in a diagnostic is worse than no name.

        Reports the process name and pid only. A command line can carry a data
        folder or a home directory, and this text reaches the user.
        """

        try:
            wildcards = {"0.0.0.0", "::", ""}
            for connection in psutil.net_connections(kind="tcp"):
                if connection.status != psutil.CONN_LISTEN or connection.pid is None:
                    continue
                address = connection.laddr
                listening_host = getattr(address, "ip", None)
                listening_port = getattr(address, "port", None)
                if listening_port is None and len(address) > 1:
                    listening_host, listening_port = address[0], address[1]
                if listening_port != port:
                    continue
                if listening_host not in wildcards and listening_host != host:
                    # A listener on another interface does not block this bind,
                    # and naming it would send the reader after the wrong process.
                    continue
                try:
                    return self._sanitize_diagnostic(
                        f"{psutil.Process(connection.pid).name()} (pid {connection.pid})"
                    )
                except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                    return f"pid {connection.pid}"
        except (psutil.AccessDenied, psutil.Error, OSError, RuntimeError):
            # Enumerating sockets is a privileged operation on some systems.
            # Failing to identify the occupant must not replace the real error.
            return None
        return None

    @staticmethod
    def _listener_owned_by_worker(pid: int, url: str) -> bool:
        parsed = urlparse(url)
        port = parsed.port
        if not port:
            return False
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return False
        for process in processes:
            try:
                connections = process.net_connections(kind="tcp")
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue
            for connection in connections:
                address = connection.laddr
                connection_port = getattr(address, "port", None)
                if connection_port is None and len(address) > 1:
                    connection_port = address[1]
                if connection.status == psutil.CONN_LISTEN and connection_port == port:
                    return True
        return False

    def _load_worker_identities(self) -> dict[str, list[_ProcessIdentity]]:
        """Read the persisted identities through ONE held state directory.

        Reading by pathname first meant a redirected state folder could SUPPLY
        this record, and __init__ hands it straight to _reap_persisted_workers,
        which terminates the processes it names. Guarding publication and
        discard was not enough while the read that feeds termination was not -
        and of the three, the read is the dangerous one.

        A refusal yields no identities, so nothing is reaped: an unreadable
        record must never be able to cause a termination.
        """

        name = self._identity_path.name
        try:
            with _held_state(self.settings.state_dir) as anchor:
                raw = read_entry(anchor, name)
                if raw is None:
                    return {}
                try:
                    return self._parse_worker_identities(raw)
                except (UnicodeError, ValueError, json.JSONDecodeError):
                    logger.warning("Discarding an invalid persisted worker identity record")
                    with contextlib.suppress(Exception):
                        discard_entry(anchor, name)
                    return {}
        except (OSError, ProcessStateError):
            logger.warning("Could not read the persisted worker identity record")
            return {}

    def _parse_worker_identities(self, raw: bytes) -> dict[str, list[_ProcessIdentity]]:
        """Validate the record's bytes. Raises for anything malformed."""

        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported worker identity record")
        workers = payload.get("workers")
        if not isinstance(workers, dict) or set(workers) - set(self._locks):
            raise ValueError("invalid worker identity record")
        result: dict[str, list[_ProcessIdentity]] = {}
        for name, items in workers.items():
            if not isinstance(items, list) or len(items) > 64:
                raise ValueError("invalid worker identity list")
            identities: list[_ProcessIdentity] = []
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("invalid worker identity")
                pid = item.get("pid")
                create_time = item.get("create_time")
                if (
                    not isinstance(pid, int)
                    or isinstance(pid, bool)
                    or pid <= 0
                    or not isinstance(create_time, (int, float))
                    or isinstance(create_time, bool)
                    or create_time <= 0
                ):
                    raise ValueError("invalid worker identity")
                identities.append(_ProcessIdentity(pid, float(create_time)))
            if identities:
                result[name] = identities
        return result

    def _persist_worker_identities(self) -> None:
        if not self._worker_identities:
            with (
                contextlib.suppress(OSError, ProcessStateError),
                _held_state(self.settings.state_dir) as anchor,
            ):
                discard_entry(anchor, self._identity_path.name)
            return
        payload = {
            "version": 1,
            "workers": {
                name: [
                    {"pid": identity.pid, "create_time": identity.create_time}
                    for identity in identities
                ]
                for name, identities in sorted(self._worker_identities.items())
            },
        }
        try:
            with _held_state(self.settings.state_dir) as anchor:
                _publish_bytes(
                    anchor,
                    self._identity_path.name,
                    json.dumps(payload, sort_keys=True).encode("utf-8"),
                )
        except (OSError, ProcessStateError):
            # Persisting identities is best effort, as it was before, but a
            # containment refusal is logged rather than passed over in silence.
            logger.exception("Could not persist worker process identities")

    @staticmethod
    def _matching_process(identity: _ProcessIdentity) -> psutil.Process | None:
        if identity.pid == os.getpid():
            return None
        try:
            process = psutil.Process(identity.pid)
            if process.create_time() != identity.create_time:
                return None
            return process
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None

    @staticmethod
    def _identity_may_still_exist(identity: _ProcessIdentity) -> bool:
        if identity.pid == os.getpid():
            return False
        try:
            return psutil.Process(identity.pid).create_time() == identity.create_time
        except psutil.AccessDenied:
            return True
        except psutil.NoSuchProcess:
            return False

    def _record_worker_process_tree(self, name: str, pid: int) -> None:
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return
        identities: list[_ProcessIdentity] = []
        for process in processes:
            try:
                identities.append(_ProcessIdentity(process.pid, process.create_time()))
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        identities.sort(key=lambda identity: identity.pid)
        if not identities:
            return
        with self._identity_lock:
            if self._worker_identities.get(name) == identities:
                return
            self._worker_identities[name] = identities
            self._persist_worker_identities()

    def _refresh_worker_identities_after_stop(self, name: str) -> None:
        with self._identity_lock:
            remaining = [
                identity
                for identity in self._worker_identities.get(name, [])
                if self._identity_may_still_exist(identity)
            ]
            if remaining:
                self._worker_identities[name] = remaining
            else:
                self._worker_identities.pop(name, None)
            self._persist_worker_identities()

    def _matching_worker_processes(self, name: str) -> list[psutil.Process]:
        with self._identity_lock:
            identities = tuple(self._worker_identities.get(name, ()))
        matches: list[psutil.Process] = []
        for identity in identities:
            process = self._matching_process(identity)
            if process is not None:
                matches.append(process)
        return matches

    def _reap_persisted_workers(self) -> None:
        matches: list[psutil.Process] = []
        for identities in self._worker_identities.values():
            for identity in identities:
                process = self._matching_process(identity)
                if process is not None:
                    matches.append(process)
        if matches:
            self._terminate_processes(matches, self.settings.worker_shutdown_seconds)
            logger.info("Attempted cleanup of %s persisted worker process(es)", len(matches))
        for name in tuple(self._worker_identities):
            self._refresh_worker_identities_after_stop(name)

    @staticmethod
    def _terminate_processes(
        processes: list[psutil.Process],
        timeout_seconds: float,
    ) -> None:
        by_pid = {process.pid: process for process in processes}
        for process in tuple(by_pid.values()):
            with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess):
                for child in process.children(recursive=True):
                    by_pid.setdefault(child.pid, child)
        live: list[psutil.Process] = []
        for process in by_pid.values():
            try:
                process.terminate()
                live.append(process)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        _gone, remaining = psutil.wait_procs(live, timeout=timeout_seconds)
        for process in remaining:
            with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess):
                process.kill()
        if remaining:
            psutil.wait_procs(remaining, timeout=timeout_seconds)

    @staticmethod
    def _descendant_processes(pid: int) -> list[psutil.Process]:
        try:
            return psutil.Process(pid).children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return []

    @staticmethod
    def _terminate_descendants(
        descendants: list[psutil.Process],
        timeout_seconds: float,
    ) -> None:
        live: list[psutil.Process] = []
        for process in descendants:
            try:
                process.terminate()
                live.append(process)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        _gone, remaining = psutil.wait_procs(live, timeout=timeout_seconds)
        for process in remaining:
            with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess):
                process.kill()
        if remaining:
            psutil.wait_procs(remaining, timeout=timeout_seconds)

    async def _stop_unlocked(self, name: str) -> None:
        record = self._workers.pop(name, None)
        if record:
            await self._terminate_record(record)
            return
        # No in-memory record does NOT mean nothing is running, and treating it
        # that way is what made this unrecoverable from inside the product.
        #
        # Measured on a live install: /api/workers reported the media worker
        # stopped with pid=None while the ComfyUI process the app itself had
        # launched was still alive, still a child of the app, and still holding
        # 127.0.0.1:8289. Returning here made stop a no-op, so the port preflight
        # in _replace then refused every start and restart with "already in use".
        # There was no sequence of supported actions that recovered it; the
        # process had to be killed from outside the app.
        #
        # The identities are persisted precisely so a worker can be recognised
        # after the handle is lost, and _terminate_record already consults them.
        # This reaches the same machinery on the path where the record is gone.
        await self._terminate_persisted_worker(name)

    async def _terminate_persisted_worker(self, name: str) -> None:
        """Stop worker processes we can still identify without a live record.

        Matching is by persisted identity - pid plus creation time - so a pid
        that has since been reused by an unrelated process is not matched and
        not killed. That check lives in `_matching_process`, and it is the reason
        this can be done safely at all.
        """

        persisted = await asyncio.to_thread(self._matching_worker_processes, name)
        if not persisted:
            return
        logger.info(
            "Stopping %s orphaned %s worker process(es) with no live record",
            len(persisted),
            name,
        )
        try:
            await asyncio.to_thread(
                self._terminate_processes,
                persisted,
                self.settings.worker_shutdown_seconds,
            )
        finally:
            self._refresh_worker_identities_after_stop(name)

    async def _terminate_record(
        self,
        record: WorkerRecord,
        *,
        cancel_monitor: bool = True,
    ) -> None:
        current_task = asyncio.current_task()
        if cancel_monitor and record.monitor_task and record.monitor_task is not current_task:
            if not record.monitor_task.done():
                record.monitor_task.cancel()
            await asyncio.gather(record.monitor_task, return_exceptions=True)
        try:
            descendants = (
                await asyncio.to_thread(self._descendant_processes, record.process.pid)
                if record.process.returncode is None
                else []
            )
            persisted = await asyncio.to_thread(self._matching_worker_processes, record.name)
            if record.process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    record.process.terminate()
                try:
                    await asyncio.wait_for(
                        record.process.wait(), timeout=self.settings.worker_shutdown_seconds
                    )
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        record.process.kill()
                    await record.process.wait()
            remaining = {
                process.pid: process
                for process in [*descendants, *persisted]
                if process.pid != record.process.pid
            }
            if remaining:
                await asyncio.to_thread(
                    self._terminate_processes,
                    list(remaining.values()),
                    self.settings.worker_shutdown_seconds,
                )
            if record.output_task:
                await record.output_task
        finally:
            self._refresh_worker_identities_after_stop(record.name)
            record.log.close()

    async def _capture_process_output(self, record: WorkerRecord) -> None:
        log_failed = False

        async def pump(stream: asyncio.StreamReader | None, *, stderr: bool) -> None:
            nonlocal log_failed
            if stream is None:
                return
            try:
                while chunk := await stream.read(16 * 1024):
                    if stderr:
                        self._append_stderr_tail(record, chunk)
                    if not log_failed:
                        try:
                            record.log.write(chunk)
                        except (OSError, ValueError):
                            log_failed = True
                            logger.exception("Could not write the %s worker log", record.name)
            except (OSError, ValueError):
                logger.exception("Could not read %s worker output", record.name)
            finally:
                if stderr:
                    self._append_stderr_tail(record, b"", final=True)

        try:
            await asyncio.gather(
                pump(record.process.stdout, stderr=False),
                pump(record.process.stderr, stderr=True),
            )
        finally:
            record.log.close()

    @staticmethod
    def _is_loading_health_line(line: bytes) -> bool:
        lowered = line.lower()
        return b"/health" in lowered and b"503" in lowered and b"get " in lowered

    @staticmethod
    def _is_connection_teardown_noise(record: WorkerRecord, line: bytes) -> bool:
        """Whether this line belongs to asyncio's connection-teardown traceback.

        `_ProactorBasePipeTransport._call_connection_lost` closes its socket
        inside a bare `finally:`, having already delivered `connection_lost` to
        the protocol. When Windows refuses the shutdown - WSAEINVAL after a
        client disconnects - the error escapes into the callback and asyncio logs
        nine lines. Nothing was lost: the connection had been handled in full.

        Suppression starts only on that callback's own line, ends on the block's
        exception line, and ends immediately on any line that is not traceback
        continuation, so a real failure printed next is never swallowed. The
        worker log file still records every line; this only decides what the
        failure diagnostic shows.
        """
        if _TEARDOWN_CALLBACK.match(line):
            record.suppressed_teardown_lines = WORKER_TEARDOWN_NOISE_MAX_LINES
            return True
        if record.suppressed_teardown_lines and _TEARDOWN_CONTINUATION.match(line):
            record.suppressed_teardown_lines -= 1
            if _TEARDOWN_TERMINAL.match(line):
                record.suppressed_teardown_lines = 0
            return True
        record.suppressed_teardown_lines = 0
        return False

    @classmethod
    def _append_stderr_tail(
        cls,
        record: WorkerRecord,
        chunk: bytes,
        *,
        final: bool = False,
    ) -> None:
        payload = bytes(record.stderr_pending) + chunk
        record.stderr_pending.clear()
        lines = payload.splitlines(keepends=True)
        if not final and lines and not payload.endswith((b"\n", b"\r")):
            record.stderr_pending.extend(lines.pop())
        for line in lines:
            if cls._is_loading_health_line(line):
                continue
            if cls._is_connection_teardown_noise(record, line):
                continue
            record.stderr_tail.extend(line)
            overflow = len(record.stderr_tail) - WORKER_STDERR_TAIL_BYTES
            if overflow > 0:
                del record.stderr_tail[:overflow]
        if len(record.stderr_pending) > WORKER_STDERR_TAIL_BYTES:
            del record.stderr_pending[:-WORKER_STDERR_TAIL_BYTES]

    def _retained_stderr(self, record: WorkerRecord) -> str | None:
        """Everything still buffered, sanitized but not trimmed for display."""
        if not record.stderr_tail:
            return None
        return self._sanitize_diagnostic(record.stderr_tail.decode("utf-8", errors="replace"))

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
        for token in (self.settings.hf_token, self.settings.civitai_token):
            if token:
                value = value.replace(token, "[redacted]")
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
