from pathlib import Path
from typing import Any

import pytest
from test_catalog_hardware_advice import _system

from local_lm.config import Settings
from local_lm.preflight import assess_catalog_install
from local_lm.processes import ProcessSupervisor
from local_lm.schemas import CatalogDetail, CatalogModel, CatalogPreflightRequest

GIB = 1024**3


def _detail(size: Any = 6 * GIB) -> CatalogDetail:
    return CatalogDetail(
        model=CatalogModel(remote_id="example/model", name="Fixture", compatibility="likely"),
        revision="a" * 40,
        files=[{"filename": "selected.gguf", "size": size, "sha256": "b" * 64}],
    )


def _request() -> CatalogPreflightRequest:
    return CatalogPreflightRequest(
        role="chat", engine="llama.cpp", selected_files=["selected.gguf"]
    )


@pytest.mark.parametrize("free", [GIB, 8 * GIB])
def test_catalog_context_range_uses_total_capacity_and_the_launch_cost(
    tmp_path: Path, free: int
) -> None:
    system = _system(None, device=False)
    system.memory_total_bytes = 9 * GIB
    system.memory_available_bytes = free
    request = _request()
    before = request.model_dump()
    result = assess_catalog_install(_detail(), request, Settings(data_dir=tmp_path), system)
    assert result.hardware_fit is not None
    assert result.estimated_ram_bytes == ProcessSupervisor._estimate_chat_memory(
        int(6 * GIB * 1.2), {}
    )
    assert result.hardware_fit.basis == "calculated"
    assert result.hardware_fit.evidence_label is None
    assert any(
        reason.code == "chat_context_estimate" and "8192-token" in reason.message
        for reason in result.hardware_fit.reasons
    )
    assert result.hardware_fit.status == "tight"
    assert len(result.hardware_fit.settings) == 1
    context = result.hardware_fit.settings[0]
    assert context.key == "context_length" and context.unit == "tokens"
    assert context.minimum == 512 and context.maximum == 7168
    assert context.advisory_only
    assert result.can_install
    assert request.model_dump() == before


def test_catalog_context_range_does_not_exceed_the_default_context(tmp_path: Path) -> None:
    result = assess_catalog_install(
        _detail(), _request(), Settings(data_dir=tmp_path), _system(None, device=False)
    )
    assert result.hardware_fit is not None
    assert len(result.hardware_fit.settings) == 1
    context = result.hardware_fit.settings[0]
    assert (context.minimum, context.maximum) == (2048, 8192)


@pytest.mark.parametrize("size", [None, 0, -1, True, "unknown"])
def test_unknown_model_sizes_do_not_produce_context_ranges(tmp_path: Path, size: Any) -> None:
    result = assess_catalog_install(
        _detail(size), _request(), Settings(data_dir=tmp_path), _system(None, device=False)
    )
    assert result.hardware_fit is not None
    assert result.hardware_fit.settings == []
    assert result.hardware_fit.basis == "unknown"


@pytest.mark.parametrize("capacity", [0, GIB, 8 * GIB])
def test_insufficient_or_unknown_capacity_does_not_invent_a_context_range(
    tmp_path: Path, capacity: int
) -> None:
    system = _system(None, device=False)
    system.memory_total_bytes = capacity
    result = assess_catalog_install(_detail(), _request(), Settings(data_dir=tmp_path), system)
    assert result.hardware_fit is not None
    assert result.hardware_fit.settings == []
    assert result.can_install


def test_alternative_context_ranges_match_an_explicit_new_preflight(tmp_path: Path) -> None:
    detail = _detail()
    detail.files.append({"filename": "other.gguf", "size": 5 * GIB, "sha256": "c" * 64})
    system = _system(None, device=False)
    system.memory_total_bytes = 9 * GIB
    result = assess_catalog_install(detail, _request(), Settings(data_dir=tmp_path), system)
    alternative = result.hardware_alternatives[0]
    assert len(alternative.hardware_fit.settings) == 1
    fresh = assess_catalog_install(
        detail,
        _request().model_copy(update={"selected_files": alternative.selected_files}),
        Settings(data_dir=tmp_path),
        system,
    )
    assert alternative.hardware_fit == fresh.hardware_fit


@pytest.mark.parametrize("context", [512, 2048, 8192, 16384, 65536])
def test_explicit_launch_context_settings_keep_the_existing_estimate(context: int) -> None:
    settings = {"context_length": context}
    assert ProcessSupervisor._estimate_chat_memory(5 * GIB, settings) == 5 * GIB + max(
        512 * 1024**2, context * 128 * 1024
    )
    assert settings == {"context_length": context}
