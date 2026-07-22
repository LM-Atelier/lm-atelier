from __future__ import annotations

from httpx2 import AsyncClient

from local_lm.platforms import GIB, assess_platform, list_platform_matrix
from local_lm.schemas import DeviceInfo


def cuda_device(vram_gb: int) -> DeviceInfo:
    return DeviceInfo(
        id="cuda:0",
        name="Test NVIDIA GPU",
        kind="gpu",
        total_memory_bytes=vram_gb * GIB,
        available_memory_bytes=vram_gb * GIB,
        backend="cuda",
    )


def test_target_matrix_encodes_approved_scope() -> None:
    matrix = list_platform_matrix()
    targets = [entry for entry in matrix if entry.status == "target"]
    experimental = [entry for entry in matrix if entry.status == "experimental"]
    assert {entry.id for entry in targets} == {
        "windows-11-nvidia-cuda",
        "ubuntu-2404-nvidia-cuda",
        "windows-ubuntu-cpu",
    }
    assert {entry.id for entry in experimental} == {
        "macos-apple-metal",
        "linux-amd-rocm",
    }
    assert all(entry.evidence for entry in matrix)


def test_ubuntu_cuda_machine_is_reference_media_capable_but_pending() -> None:
    assessment = assess_platform(
        platform_name="Linux",
        architecture="x86_64",
        distribution="Ubuntu",
        distribution_version="24.04",
        memory_total_bytes=32 * GIB,
        devices=[cuda_device(16)],
    )
    assert assessment.platform_status == "target"
    assert assessment.accelerator_status == "primary"
    assert assessment.vram_tier_gb == 16
    assert assessment.reference_media_ready is True
    assert assessment.certification_status == "hardware-pending"


def test_cpu_fallback_and_experimental_accelerators_are_distinct() -> None:
    cpu = DeviceInfo(id="cpu:0", name="CPU", kind="cpu", backend="cpu")
    windows = assess_platform(
        platform_name="Windows",
        architecture="AMD64",
        distribution="Windows",
        distribution_version="11",
        memory_total_bytes=16 * GIB,
        devices=[cpu],
    )
    assert windows.platform_status == "target"
    assert windows.accelerator_status == "cpu-only"
    assert windows.chat_ready is True
    assert windows.reference_media_ready is False

    metal = DeviceInfo(id="metal:0", name="Apple GPU", kind="gpu", backend="metal")
    macos = assess_platform(
        platform_name="Darwin",
        architecture="arm64",
        distribution="macOS",
        distribution_version="15",
        memory_total_bytes=32 * GIB,
        devices=[metal],
    )
    assert macos.platform_status == "experimental"
    assert macos.accelerator_status == "experimental"
    assert macos.reference_media_ready is False

    windows_10 = assess_platform(
        platform_name="Windows",
        architecture="AMD64",
        distribution="Windows",
        distribution_version="10",
        memory_total_bytes=16 * GIB,
        devices=[cpu],
    )
    assert windows_10.platform_status == "experimental"


async def test_platform_matrix_and_assessment_are_exposed(client: AsyncClient) -> None:
    matrix = await client.get("/api/platforms")
    assert matrix.status_code == 200
    assert len(matrix.json()) == 5

    system = await client.get("/api/system")
    assert system.status_code == 200
    assert system.json()["support"]["certification_status"] in {
        "hardware-pending",
        "experimental",
        "unsupported",
    }
