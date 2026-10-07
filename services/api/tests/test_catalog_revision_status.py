from __future__ import annotations

from pathlib import Path

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


@pytest.mark.parametrize(
    ("provider", "revision", "status"),
    [
        ("huggingface", "main", "warn"),
        ("huggingface", "release", "warn"),
        ("huggingface", "v1.0", "warn"),
        ("huggingface", "abcdef1", "warn"),
        ("huggingface", "a" * 39, "warn"),
        ("huggingface", "g" * 40, "warn"),
        ("huggingface", "a" * 41, "warn"),
        ("huggingface", "201", "warn"),
        ("huggingface", "a" * 40, "pass"),
        ("civitai", "201", "pass"),
        ("civitai", "0", "warn"),
        ("civitai", "01", "warn"),
        ("civitai", "1" * 13, "warn"),
        ("civitai", "main", "warn"),
        ("civitai", "a" * 40, "warn"),
        ("other", "a" * 40, "warn"),
    ],
)
def test_revision_advice_uses_the_resolved_provider_identity(
    tmp_path: Path, provider: str, revision: str, status: str
) -> None:
    detail = CatalogDetail(
        model=CatalogModel(
            provider=provider,
            remote_id="201" if provider == "civitai" else "example/model",
            name="Fixture",
            compatibility="likely",
        ),
        revision=revision,
        files=[{"filename": "model.safetensors", "size": _GIB, "sha256": "b" * 64}],
    )
    result = assess_catalog_install(
        detail,
        CatalogPreflightRequest(
            revision="release",
            role="image",
            engine="comfyui",
            selected_files=["model.safetensors"],
        ),
        Settings(data_dir=tmp_path),
        _system(14 * _GIB),
    )
    check = next(check for check in result.checks if check.id == "revision")

    assert check.status == status
    if status == "pass":
        assert check.detail == f"Install is pinned to {revision}."
    else:
        assert "Install is pinned" not in check.detail
    assert result.revision == revision
    assert result.selected_files == ["model.safetensors"]
    assert result.expected_sha256 == {"model.safetensors": "b" * 64}
    if provider == "huggingface":
        assert result.can_install
