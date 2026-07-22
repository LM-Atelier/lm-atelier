from __future__ import annotations

from typing import Literal

from .schemas import DeviceInfo, PlatformAssessment, PlatformMatrixEntry

GIB = 1024**3


PLATFORM_MATRIX: tuple[PlatformMatrixEntry, ...] = (
    PlatformMatrixEntry(
        id="windows-11-nvidia-cuda",
        name="Windows 11 x64 + NVIDIA CUDA",
        status="target",
        operating_systems=["Windows 11"],
        architectures=["x86_64"],
        accelerator="NVIDIA CUDA",
        workloads=["chat", "image", "video"],
        vram_tiers_gb=[12, 16, 24],
        evidence="Automated Windows tests; physical GPU certification pending",
        notes=["CPU chat remains available when CUDA is absent."],
    ),
    PlatformMatrixEntry(
        id="ubuntu-2404-nvidia-cuda",
        name="Ubuntu 24.04 LTS x64 + NVIDIA CUDA",
        status="target",
        operating_systems=["Ubuntu 24.04 LTS"],
        architectures=["x86_64"],
        accelerator="NVIDIA CUDA",
        workloads=["chat", "image", "video"],
        vram_tiers_gb=[12, 16, 24],
        evidence="Automated Ubuntu tests; physical GPU certification pending",
        notes=["CPU chat remains available when CUDA is absent."],
    ),
    PlatformMatrixEntry(
        id="windows-ubuntu-cpu",
        name="Windows/Ubuntu CPU fallback",
        status="target",
        operating_systems=["Windows 11", "Ubuntu 24.04 LTS"],
        architectures=["x86_64"],
        accelerator="CPU",
        workloads=["chat"],
        evidence="Automated Windows and Ubuntu tests; performance certification pending",
        notes=["The Qwen3 8B Q4_K_M recipe is the baseline CPU chat model."],
    ),
    PlatformMatrixEntry(
        id="macos-apple-metal",
        name="macOS Apple Silicon + Metal",
        status="experimental",
        operating_systems=["macOS"],
        architectures=["arm64"],
        accelerator="Apple Metal",
        workloads=["chat", "image", "video"],
        evidence="No dedicated CI runner or certification machine",
        notes=["Functionality depends on separately installed runtime support."],
    ),
    PlatformMatrixEntry(
        id="linux-amd-rocm",
        name="Linux x64 + AMD ROCm",
        status="experimental",
        operating_systems=["Linux"],
        architectures=["x86_64"],
        accelerator="AMD ROCm",
        workloads=["chat", "image", "video"],
        evidence="No dedicated CI runner or certification machine",
        notes=["GPU and runtime compatibility vary by ROCm release."],
    ),
)


def list_platform_matrix() -> list[PlatformMatrixEntry]:
    return list(PLATFORM_MATRIX)


def _vram_tier(devices: list[DeviceInfo], backend: str) -> int | None:
    totals = [
        device.total_memory_bytes or 0
        for device in devices
        if (device.backend or "").lower() == backend
    ]
    if not totals:
        return None
    total = max(totals)
    for tier in (24, 16, 12):
        if total >= tier * 1_000_000_000:
            return tier
    return 0


def assess_platform(
    *,
    platform_name: str,
    architecture: str,
    distribution: str,
    distribution_version: str,
    memory_total_bytes: int,
    devices: list[DeviceInfo],
) -> PlatformAssessment:
    machine = architecture.lower()
    x64 = machine in {"amd64", "x86_64"}
    system = platform_name.lower()
    distro = distribution.lower()
    version = distribution_version.lower()
    messages: list[str] = []
    platform_status: Literal["target", "experimental", "unsupported"]
    accelerator_status: Literal["primary", "experimental", "cpu-only"]
    certification_status: Literal["hardware-pending", "experimental", "unsupported"]

    if system == "windows" and x64 and version.startswith("11"):
        platform_status = "target"
        platform_label = "Windows 11 x64 target"
        messages.append("Windows automated tests cover the application stack.")
    elif (
        system == "linux"
        and x64
        and distro in {"ubuntu", "ubuntu linux"}
        and version.startswith("24.04")
    ):
        platform_status = "target"
        platform_label = "Ubuntu 24.04 LTS x64 target"
        messages.append("Ubuntu automated tests cover the application stack.")
    elif system == "darwin" and machine in {"arm64", "aarch64"}:
        platform_status = "experimental"
        platform_label = "Apple Silicon experimental"
        messages.append("Apple Metal remains outside the initial certification scope.")
    elif x64:
        platform_status = "experimental"
        platform_label = f"{distribution or platform_name} x64 experimental"
        messages.append("The certification baseline is Windows 11 or Ubuntu 24.04 LTS.")
    else:
        platform_status = "unsupported"
        platform_label = f"{platform_name} {architecture} unsupported"
        messages.append("The initial target architecture is x86_64.")

    cuda_tier = _vram_tier(devices, "cuda")
    backends = {(device.backend or "").lower() for device in devices}
    if cuda_tier is not None:
        accelerator_status = "primary"
        if cuda_tier:
            accelerator_label = f"NVIDIA CUDA {cuda_tier} GB tier"
            messages.append("CUDA meets the minimum reference-media VRAM tier.")
        else:
            accelerator_label = "NVIDIA CUDA below 12 GB tier"
            messages.append("CUDA was detected, but reference media starts at 12 GB VRAM.")
    elif backends & {"metal", "mps", "rocm", "hip"}:
        accelerator_status = "experimental"
        accelerator_label = "Experimental GPU accelerator"
        messages.append("Metal and ROCm require dedicated validation before certification.")
    else:
        accelerator_status = "cpu-only"
        accelerator_label = "CPU fallback"
        messages.append("No primary media accelerator was detected; local chat remains available.")

    chat_ready = memory_total_bytes >= 8 * GIB
    if not chat_ready:
        messages.append("The CPU chat baseline requires at least 8 GB system RAM.")
    reference_media_ready = (
        platform_status == "target" and accelerator_status == "primary" and bool(cuda_tier)
    )
    certification_status = (
        "hardware-pending"
        if platform_status == "target"
        else "experimental"
        if platform_status == "experimental"
        else "unsupported"
    )
    return PlatformAssessment(
        platform_status=platform_status,
        platform_label=platform_label,
        accelerator_status=accelerator_status,
        accelerator_label=accelerator_label,
        certification_status=certification_status,
        chat_ready=chat_ready,
        reference_media_ready=reference_media_ready,
        vram_tier_gb=cuda_tier or None,
        messages=messages,
    )
