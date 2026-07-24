from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

from .config import Settings
from .platforms import assess_platform
from .schemas import DeviceInfo, SystemInfo


def _cpu_model() -> str:
    system = platform.system()
    if system == "Windows":
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
            )
            if result.stdout.strip():
                return " ".join(result.stdout.split())
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "CPU"


def _run_json(command: list[str], timeout: float = 5) -> Any | None:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=timeout
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
