from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from typing import Any

import psutil

from .config import Settings
from .platforms import assess_platform
from .schemas import DeviceInfo, SystemInfo
from .subprocess_env import subprocess_environment

_CAPABILITY_CLASS_TTL_SECONDS = 60.0
_capability_class_cache: tuple[float, str] | None = None


def _cpu_model() -> str:
    system = platform.system()
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if str(value).strip():
                    return " ".join(str(value).split())
        except (ImportError, OSError):
            pass
    elif system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith(("model name", "hardware")) and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return " ".join(value.split())
        except OSError:
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
                env=subprocess_environment(),
            )
            if result.stdout.strip():
                return " ".join(result.stdout.split())
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "CPU"


def _run_json(command: list[str], timeout: float = 5) -> Any | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=subprocess_environment(),
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def _nvidia_devices() -> list[DeviceInfo]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env=subprocess_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return []
    devices: list[DeviceInfo] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        index, name, total_mib, free_mib, driver = fields
        devices.append(
            DeviceInfo(
                id=f"cuda:{index}",
                name=name,
                kind="gpu",
                total_memory_bytes=int(total_mib) * 1024 * 1024,
                available_memory_bytes=int(free_mib) * 1024 * 1024,
                backend="cuda",
                details={"driver": driver},
            )
        )
    return devices


def _llama_devices() -> list[DeviceInfo]:
    executable = shutil.which("llama-server")
    if not executable:
        return []
    payload = _run_json([executable, "--list-devices", "--json"])
    if not isinstance(payload, list):
        return []
    devices: list[DeviceInfo] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        devices.append(
            DeviceInfo(
                id=str(item.get("id", f"llama:{index}")),
                name=str(item.get("name", f"llama device {index}")),
                kind="accelerator",
                total_memory_bytes=item.get("memory_total"),
                available_memory_bytes=item.get("memory_free"),
                backend=str(item.get("backend", "llama.cpp")),
                details=item,
            )
        )
    return devices


def collect_system_info(settings: Settings) -> SystemInfo:
    platform_name = platform.system()
    platform_release = platform.release()
    distribution = platform_name
    distribution_version = platform_release
    if platform_name == "Windows":
        try:
            windows_build = int(platform.version().split(".")[-1])
        except ValueError:
            windows_build = 0
        if windows_build >= 22_000:
            distribution_version = "11"
    elif platform_name == "Linux":
        try:
            release = platform.freedesktop_os_release()
            distribution = release.get("NAME", release.get("ID", "Linux"))
            distribution_version = release.get("VERSION_ID", platform_release)
        except OSError:
            pass
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path(settings.data_dir).resolve())
    cpu_model = _cpu_model()
    devices = _nvidia_devices()
    known = {device.id for device in devices}
    devices.extend(device for device in _llama_devices() if device.id not in known)
    devices.append(
        DeviceInfo(
            id="cpu:0",
            name=cpu_model,
            kind="cpu",
            total_memory_bytes=memory.total,
            available_memory_bytes=memory.available,
            backend="cpu",
        )
    )
    support = assess_platform(
        platform_name=platform_name,
        architecture=platform.machine(),
        distribution=distribution,
        distribution_version=distribution_version,
        memory_total_bytes=memory.total,
        devices=devices,
    )
    return SystemInfo(
        platform=platform_name,
        platform_release=platform_release,
        distribution=distribution,
        distribution_version=distribution_version,
        architecture=platform.machine(),
        python_version=sys.version.split()[0],
        cpu_model=cpu_model,
        cpu_count=psutil.cpu_count(logical=True) or 1,
        memory_total_bytes=memory.total,
        memory_available_bytes=memory.available,
        disk_total_bytes=disk.total,
        disk_free_bytes=disk.free,
        ffmpeg_available=shutil.which("ffmpeg") is not None,
        devices=devices,
        support=support,
    )


def reset_hardware_capability_class_cache() -> None:
    """Forget the memoized capability class."""

    global _capability_class_cache
    _capability_class_cache = None


def hardware_capability_class(settings: Settings) -> str:
    """Return a stable hardware key without volatile free-memory measurements.

    Enumerating devices shells out to `nvidia-smi` and `llama-server`, and this
    runs once per installed model on every setup readiness poll, so a fresh
    machine polling every few seconds spawned several subprocesses per poll and
    made setup appear to hang. The key deliberately excludes volatile
    measurements, so it is memoized briefly; the lifetime is short enough that a
    genuine hardware or runtime change is picked up without a restart.
    """

    global _capability_class_cache
    now = monotonic()
    cached = _capability_class_cache
    if cached and now - cached[0] < _CAPABILITY_CLASS_TTL_SECONDS:
        return cached[1]
    system = collect_system_info(settings)
    stable_devices = sorted(
        (
            device.name,
            device.kind,
            device.backend,
            device.total_memory_bytes,
        )
        for device in system.devices
    )
    payload = {
        "platform": system.platform,
        "release": system.distribution_version,
        "architecture": system.architecture,
        "cpu": system.cpu_model,
        "memory": system.memory_total_bytes,
        "devices": stable_devices,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    capability_class = f"{system.platform.casefold()}-{system.architecture.casefold()}-{digest}"
    _capability_class_cache = (now, capability_class)
    return capability_class


# What a machine has to offer for a proof recorded elsewhere to still apply.
#
# `hardware_capability_class` is an equality token over the CPU model, total
# memory and the device list - and that device list is only populated when
# `llama-server` and `nvidia-smi` resolve on PATH. So a driver update or a PATH
# change silently invalidated every proof on a machine that had not changed. An
# envelope asks the question that actually matters: can this machine still do
# what the machine that proved it could do.
HARDWARE_ENVELOPE_VERSION = 1


def hardware_envelope(settings: Settings) -> dict[str, Any]:
    """What this machine offers, in comparable terms."""

    system = collect_system_info(settings)
    accelerators = [
        device
        for device in system.devices
        if device.kind.casefold() != "cpu" and device.total_memory_bytes
    ]
    return {
        "version": HARDWARE_ENVELOPE_VERSION,
        "platform": system.platform.casefold(),
        "architecture": system.architecture.casefold(),
        "memory_bytes": system.memory_total_bytes,
        "accelerator_memory_bytes": max(
            (device.total_memory_bytes or 0 for device in accelerators),
            default=0,
        ),
        "backends": sorted(
            {
                device.backend.casefold()
                for device in accelerators
                if isinstance(device.backend, str) and device.backend
            }
        ),
    }


def hardware_envelope_satisfied(
    recorded: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> bool:
    """Whether `current` can still do what the machine that recorded this could.

    Conservative on purpose. The envelope records what the proving machine
    *had*, not what the run *needed*, so "at least as much memory" is an
    approximation - but it errs towards refusing a claim rather than making one
    that might not hold, and it does not throw away a proof because a driver
    updated.
    """
    if not isinstance(recorded, Mapping) or not recorded:
        return False
    if recorded.get("version") != HARDWARE_ENVELOPE_VERSION:
        return False
    # A proof from another operating system or architecture says nothing here.
    if recorded.get("platform") != current.get("platform"):
        return False
    if recorded.get("architecture") != current.get("architecture"):
        return False

    def _amount(source: Mapping[str, Any], key: str) -> int:
        value = source.get(key)
        return value if isinstance(value, int) and value >= 0 else 0

    if _amount(current, "memory_bytes") < _amount(recorded, "memory_bytes"):
        return False
    required_accelerator = _amount(recorded, "accelerator_memory_bytes")
    if required_accelerator and _amount(current, "accelerator_memory_bytes") < required_accelerator:
        return False
    recorded_backends = recorded.get("backends")
    current_backends = current.get("backends")
    if isinstance(recorded_backends, list) and recorded_backends:
        available = set(current_backends) if isinstance(current_backends, list) else set()
        # The proof used some accelerator backend; this machine needs one of them.
        if not available & set(recorded_backends):
            return False
    return True
