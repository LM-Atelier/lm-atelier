from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from local_lm.config import Settings
from local_lm.preflight import assess_catalog_install
from local_lm.schemas import (
    CatalogDetail,
    CatalogModel,
    CatalogPreflightRequest,
    DeviceInfo,
    PlatformAssessment,
    SystemInfo,
)

_GIB = 1024**3


def _system(free: int | None, *, device: bool = True) -> SystemInfo:
    return SystemInfo(
        platform="linux",
        platform_release="fixture",
        distribution="Fixture Linux",
        distribution_version="1",
        architecture="x86_64",
        python_version="3.12.10",
        cpu_model="Fixture CPU",
        cpu_count=8,
        memory_total_bytes=32 * _GIB,
        memory_available_bytes=24 * _GIB,
        disk_total_bytes=100 * _GIB,
        disk_free_bytes=100 * _GIB,
        ffmpeg_available=False,
        devices=[
            DeviceInfo(
                id="cuda:0",
                name="Fixture GPU",
                kind="gpu",
                backend="cuda",
                total_memory_bytes=16 * _GIB,
                available_memory_bytes=free,
            ),
        ]
        if device
        else [],
        support=PlatformAssessment(
            platform_status="target",
            platform_label="Fixture Linux",
            accelerator_status="primary",
            accelerator_label="Fixture GPU",
            certification_status="hardware-pending",
            chat_ready=True,
            reference_media_ready=True,
        ),
    )


def _detail(size: int = 2 * _GIB) -> CatalogDetail:
    return CatalogDetail(
        model=CatalogModel(remote_id="example/model", name="Fixture", compatibility="likely"),
        revision="a" * 40,
        files=[
            {"filename": "selected.safetensors", "size": size, "sha256": "b" * 64},
            {"filename": "other.safetensors", "size": _GIB, "sha256": "c" * 64},
        ],
    )


@pytest.mark.parametrize("role", ["image", "video"])
def test_busy_memory_keeps_general_fit_and_the_explicit_file_selection(
    tmp_path: Path, role: Literal["image", "video"]
) -> None:
    request = CatalogPreflightRequest(
        role=role, engine="comfyui", selected_files=["selected.safetensors"]
    )
    results = [
        assess_catalog_install(
            _detail(), request, Settings(data_dir=tmp_path / str(free)), _system(free)
        )
        for free in (14 * _GIB, _GIB)
    ]
    memory = [next(check for check in result.checks if check.id == "memory") for result in results]

    assert [check.status for check in memory] == ["pass", "pass"]
    assert all(check.detail.startswith("Likely fit.") for check in memory)
    assert memory[0].detail == memory[1].detail
    assert not any(check.id == "memory-availability" for check in results[0].checks)
    assert any(
        check.id == "memory-availability" and check.status == "warn" for check in results[1].checks
    )
    assert all(result.selected_files == ["selected.safetensors"] for result in results)
    assert all(result.expected_sha256 == {"selected.safetensors": "b" * 64} for result in results)
    assert all(result.can_install for result in results)


@pytest.mark.parametrize("device", [False, True])
def test_missing_or_unreported_free_memory_stays_an_advisory_estimate(
    tmp_path: Path, device: bool
) -> None:
    result = assess_catalog_install(
        _detail(),
        CatalogPreflightRequest(
            role="image", engine="comfyui", selected_files=["selected.safetensors"]
        ),
        Settings(data_dir=tmp_path),
        _system(None, device=device),
    )
    memory = next(check for check in result.checks if check.id == "memory")

    assert memory.status == ("pass" if device else "warn")
    assert memory.detail.startswith("Likely fit." if device else "Hardware fit is unknown.")
    assert result.can_install
    assert "tested" not in memory.detail.lower()
    assert "certified" not in memory.detail.lower()
    assert not any(check.id == "memory-availability" for check in result.checks)


def test_an_oversized_estimate_keeps_the_explicit_install_advisory(tmp_path: Path) -> None:
    result = assess_catalog_install(
        _detail(20 * _GIB),
        CatalogPreflightRequest(
            role="image", engine="comfyui", selected_files=["selected.safetensors"]
        ),
        Settings(data_dir=tmp_path),
        _system(14 * _GIB),
    )
    memory = next(check for check in result.checks if check.id == "memory")

    assert memory.status == "warn"
    assert memory.detail.startswith("Tight fit.")
    assert "fits total" not in memory.detail
    assert result.can_install
    assert result.selected_files == ["selected.safetensors"]


def test_temporary_memory_pressure_remains_visible_in_install_warnings(tmp_path: Path) -> None:
    result = assess_catalog_install(
        _detail(),
        CatalogPreflightRequest(
            role="image", engine="comfyui", selected_files=["selected.safetensors"]
        ),
        Settings(data_dir=tmp_path),
        _system(_GIB),
    )

    memory = next(check for check in result.checks if check.id == "memory")
    warnings = [check for check in result.checks if check.status == "warn"]
    assert memory.status == "pass"
    assert any(
        check.id == "memory-availability" and "currently free" in check.detail for check in warnings
    )
    assert any("unload idle models" in check.detail for check in warnings)
    assert result.can_install
