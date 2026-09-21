from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from httpx2 import AsyncClient

from local_lm.catalog import HuggingFaceCatalog
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


@pytest.mark.parametrize("free", [14 * _GIB, _GIB, None])
def test_catalog_response_exposes_capacity_and_pressure_separately(
    tmp_path: Path, free: int | None
) -> None:
    result = assess_catalog_install(
        _detail(),
        CatalogPreflightRequest(
            role="image", engine="comfyui", selected_files=["selected.safetensors"]
        ),
        Settings(data_dir=tmp_path),
        _system(free),
    )
    payload = result.model_dump(mode="json")
    assert payload["download_size_complete"] is True
    advice = payload["hardware_fit"]
    assert advice["status"] == "likely"
    assert advice["basis"] == "calculated"
    assert advice["evidence_label"] is None
    accelerator = next(item for item in advice["resources"] if item["kind"] == "accelerator")
    assert accelerator["capacity_bytes"] == 16 * _GIB
    assert accelerator["available_bytes"] == free
    assert accelerator["immediate_pressure"] is (free == _GIB)
    assert any(item["code"] == "accelerator_memory_estimated" for item in advice["reasons"])
    assert bool(advice["alternatives"]) is (free == _GIB)
    assert advice["settings"] == []
    assert payload["can_install"]
    assert payload["selected_files"] == ["selected.safetensors"]
    assert payload["expected_sha256"] == {"selected.safetensors": "b" * 64}
    assert result.model_copy(update={"install_plan": None}).model_dump()["hardware_fit"] == advice


def test_catalog_response_keeps_unknown_hardware_advisory(tmp_path: Path) -> None:
    result = assess_catalog_install(
        _detail(),
        CatalogPreflightRequest(
            role="image", engine="comfyui", selected_files=["selected.safetensors"]
        ),
        Settings(data_dir=tmp_path),
        _system(None, device=False),
    )
    advice = result.model_dump(mode="json")["hardware_fit"]
    assert advice["status"] == "unknown"
    assert advice["evidence_label"] is None
    assert result.can_install


@pytest.mark.parametrize("role", ["image", "video"])
@pytest.mark.parametrize(
    "size_metadata",
    [
        {},
        {"size": None},
        {"size": 0},
        {"size": -1},
        {"size": True},
        {"size": 1.5},
        {"size": "unknown"},
    ],
)
def test_catalog_fit_requires_sizes_for_every_selected_file(
    tmp_path: Path,
    role: Literal["image", "video"],
    size_metadata: dict[str, int | float | str | None],
) -> None:
    detail = _detail()
    detail.files[1] = {
        "filename": "other.safetensors",
        "sha256": "c" * 64,
        **size_metadata,
    }
    result = assess_catalog_install(
        detail,
        CatalogPreflightRequest(
            role=role,
            engine="comfyui",
            selected_files=["selected.safetensors", "other.safetensors"],
        ),
        Settings(data_dir=tmp_path),
        _system(14 * _GIB),
    )
    assert result.estimated_ram_bytes is None
    assert result.estimated_vram_bytes is None
    advice = result.model_dump(mode="json")["hardware_fit"]
    assert advice["status"] == "unknown"
    assert advice["basis"] == "unknown"
    assert advice["resources"] == []
    assert advice["evidence_label"] is None
    assert result.download_bytes == 2 * _GIB
    assert result.model_dump()["download_size_complete"] is False
    assert result.selected_files == ["selected.safetensors", "other.safetensors"]
    assert result.expected_sha256 == {
        "selected.safetensors": "b" * 64,
        "other.safetensors": "c" * 64,
    }
    checks = {check.id: check for check in result.checks}
    assert checks["memory"].status == "warn"
    assert checks["disk"].status == "warn"
    assert result.can_install


def test_unselected_unknown_file_size_does_not_remove_fit_estimate(tmp_path: Path) -> None:
    detail = _detail()
    detail.files[1].pop("size")
    result = assess_catalog_install(
        detail,
        CatalogPreflightRequest(
            role="image", engine="comfyui", selected_files=["selected.safetensors"]
        ),
        Settings(data_dir=tmp_path),
        _system(14 * _GIB),
    )
    assert result.estimated_ram_bytes is not None
    assert result.model_dump()["download_size_complete"] is True
    assert result.estimated_vram_bytes is not None
    assert result.model_dump(mode="json")["hardware_fit"]["status"] == "likely"
    assert result.can_install


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/catalog/example/model/preflight",
        "/api/catalog/preflight?source=huggingface&id=example/model",
    ],
)
@pytest.mark.parametrize(
    "size, status", [(2 * _GIB, "likely"), (40 * _GIB, "tight"), (None, "unknown")]
)
async def test_http_preflight_keeps_advice_and_explicit_model_choice(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    size: int | None,
    status: str,
) -> None:
    detail = _detail()
    detail.files = [
        {"filename": "chosen.gguf", "size": size, "sha256": "b" * 64},
        {"filename": "smaller.gguf", "size": _GIB, "sha256": "c" * 64},
    ]
    inventory = _system(None, device=False)
    inventory.memory_available_bytes = _GIB

    def system_info(_settings: Settings) -> SystemInfo:
        return inventory

    monkeypatch.setattr("local_lm.api.collect_system_info", system_info)
    monkeypatch.setattr(HuggingFaceCatalog, "inspect", AsyncMock(return_value=detail.model_dump()))
    monkeypatch.setattr(
        HuggingFaceCatalog, "discover_vision_projector", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        HuggingFaceCatalog,
        "inspect_file_prefix",
        AsyncMock(side_effect=OSError("Metadata unavailable")),
    )
    response = await client.post(
        endpoint,
        json={
            "revision": "a" * 40,
            "role": "chat",
            "engine": "llama.cpp",
            "selected_files": ["chosen.gguf"],
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["download_size_complete"] is (size is not None)
    assert payload["selected_files"] == ["chosen.gguf"]
    assert payload["expected_sha256"] == {"chosen.gguf": "b" * 64}
    assert payload["revision"] == "a" * 40
    assert payload["can_install"]
    assert payload["install_plan"]["id"]
    assert payload["install_plan"]["compatibility"] == "supported"
    advice = payload["hardware_fit"]
    assert advice["status"] == status
    assert advice["basis"] == ("calculated" if size is not None else "unknown")
    assert advice["evidence_label"] is None
    if size is None:
        assert advice["resources"] == []
    else:
        assert len(advice["resources"]) == 1
        assert advice["resources"][0]["capacity_bytes"] == 32 * _GIB
        assert advice["resources"][0]["available_bytes"] == _GIB
        assert advice["resources"][0]["immediate_pressure"]

    # Hardware advice cannot authorize a different file against the approved plan.
    changed = await client.post(
        "/api/downloads",
        json={
            "install_plan_id": payload["install_plan"]["id"],
            "remote_id": "example/model",
            "revision": "a" * 40,
            "role": "chat",
            "engine": "llama.cpp",
            "allow_patterns": ["smaller.gguf"],
        },
    )
    assert changed.status_code == 422
    assert "immutable plan" in changed.json()["detail"]
