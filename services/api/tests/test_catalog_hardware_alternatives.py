from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx2 import AsyncClient
from test_catalog_hardware_advice import _system

from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.preflight import assess_catalog_install
from local_lm.schemas import CatalogDetail, CatalogModel, CatalogPreflightRequest

GIB = 1024**3


def _file(name: str, size: Any = GIB) -> dict[str, Any]:
    return {"filename": name, "size": size, "sha256": "b" * 64}


def _detail() -> CatalogDetail:
    return CatalogDetail(
        model=CatalogModel(remote_id="example/model", name="Fixture", compatibility="likely"),
        revision="a" * 40,
        files=[
            _file("chosen.gguf", 8 * GIB),
            _file("large.gguf", 40 * GIB),
            _file("small.gguf"),
            _file("unknown.gguf", None),
        ],
    )


def test_alternatives_rank_total_capacity_and_preserve_the_selected_model(tmp_path: Path) -> None:
    request = CatalogPreflightRequest(
        role="chat", engine="llama.cpp", selected_files=["chosen.gguf"]
    )
    results = []
    for free in (30 * GIB, 1):
        system = _system(None, device=False)
        system.memory_available_bytes = free
        result = assess_catalog_install(_detail(), request, Settings(data_dir=tmp_path), system)
        results.append(result)
        assert result.selected_files == ["chosen.gguf"]
        assert result.expected_sha256 == {"chosen.gguf": "b" * 64}
        assert [choice.selected_files for choice in result.hardware_alternatives] == [
            ["small.gguf"],
            ["large.gguf"],
            ["unknown.gguf"],
        ]
        assert [choice.hardware_fit.status for choice in result.hardware_alternatives] == [
            "likely",
            "tight",
            "unknown",
        ]
        assert all(
            choice.hardware_fit.evidence_label is None for choice in result.hardware_alternatives
        )
    assert not results[0].hardware_alternatives[0].hardware_fit.resources[0].immediate_pressure
    assert results[1].hardware_alternatives[0].hardware_fit.resources[0].immediate_pressure


def test_alternatives_keep_complete_shards_and_include_projector_cost(tmp_path: Path) -> None:
    detail = _detail()
    detail.files = [
        _file("chosen.gguf"),
        _file("split-00002-of-00002.gguf"),
        _file("split-00001-of-00002.gguf"),
        _file("mmproj-f16.gguf", 2 * GIB),
        _file("missing-00001-of-00002.gguf"),
        _file("../unsafe.gguf"),
        _file("bad-00001-of-00001.gguf", None),
    ]
    result = assess_catalog_install(
        detail,
        CatalogPreflightRequest(
            role="chat",
            engine="llama.cpp",
            selected_files=["chosen.gguf"],
        ),
        Settings(data_dir=tmp_path),
        _system(None, device=False),
    )
    assert len(result.hardware_alternatives) == 1
    option = result.hardware_alternatives[0]
    assert option.selected_files == [
        "split-00001-of-00002.gguf",
        "split-00002-of-00002.gguf",
        "mmproj-f16.gguf",
    ]
    assert option.download_bytes == 4 * GIB
    assert option.download_size_complete
    fresh = assess_catalog_install(
        detail,
        CatalogPreflightRequest(
            role="chat",
            engine="llama.cpp",
            selected_files=option.selected_files,
        ),
        Settings(data_dir=tmp_path),
        _system(None, device=False),
    )
    assert fresh.can_install
    assert fresh.selected_files == option.selected_files
    assert fresh.hardware_fit == option.hardware_fit
    assert set(fresh.expected_sha256) == set(option.selected_files)


@pytest.mark.parametrize("size", [None, 0, -1, True, "unknown"])
def test_unknown_projector_size_keeps_alternative_fit_unknown(tmp_path: Path, size: Any) -> None:
    detail = _detail()
    detail.files.append(_file("mmproj-f16.gguf", size))
    result = assess_catalog_install(
        detail,
        CatalogPreflightRequest(
            role="chat",
            engine="llama.cpp",
            selected_files=["chosen.gguf"],
        ),
        Settings(data_dir=tmp_path),
        _system(None, device=False),
    )
    assert result.hardware_alternatives
    assert all(
        not option.download_size_complete and option.hardware_fit.status == "unknown"
        for option in result.hardware_alternatives
    )


@pytest.mark.parametrize(
    "mutation", ["floating", "duplicate", "case_duplicate", "media", "vllm", "provider"]
)
def test_alternatives_require_unambiguous_pinned_gguf_metadata(
    tmp_path: Path, mutation: str
) -> None:
    detail = _detail()
    request = CatalogPreflightRequest(
        role="chat", engine="llama.cpp", selected_files=["chosen.gguf"]
    )
    if mutation == "floating":
        detail.revision = "main"
    elif mutation == "duplicate":
        detail.files.append(_file("small.gguf", 2 * GIB))
    elif mutation == "case_duplicate":
        detail.files.append(_file("SMALL.gguf"))
    elif mutation == "media":
        request.role, request.engine = "image", "comfyui"
    elif mutation == "vllm":
        request.engine = "vllm"
    else:
        detail.model.provider = "civitai"
    result = assess_catalog_install(
        detail, request, Settings(data_dir=tmp_path), _system(None, device=False)
    )
    assert result.hardware_alternatives == []


def test_alternatives_bound_the_response_after_ranking_all_candidates(tmp_path: Path) -> None:
    detail = _detail()
    detail.files = [
        _file("chosen.gguf"),
        *[_file(f"large-{i}.gguf", 40 * GIB) for i in range(8)],
        _file("small.gguf"),
    ]
    result = assess_catalog_install(
        detail,
        CatalogPreflightRequest(
            role="chat",
            engine="llama.cpp",
            selected_files=["chosen.gguf"],
        ),
        Settings(data_dir=tmp_path),
        _system(None, device=False),
    )
    assert len(result.hardware_alternatives) == 5
    assert result.hardware_alternatives[0].selected_files == ["small.gguf"]


async def test_http_alternative_selection_creates_a_fresh_pinned_install_plan(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    detail = _detail()
    monkeypatch.setattr("local_lm.api.collect_system_info", lambda _: _system(None, device=False))
    inspect = AsyncMock(return_value=detail.model_dump())
    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    monkeypatch.setattr(
        HuggingFaceCatalog, "discover_vision_projector", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        HuggingFaceCatalog,
        "inspect_file_prefix",
        AsyncMock(side_effect=OSError("Metadata unavailable")),
    )
    original = await client.post(
        "/api/catalog/example/model/preflight",
        json={
            "revision": "a" * 40,
            "role": "chat",
            "engine": "llama.cpp",
            "selected_files": ["chosen.gguf"],
        },
    )
    assert original.status_code == 200
    initial = original.json()
    assert initial.get("hardware_alternatives")
    option = initial["hardware_alternatives"][0]
    response = await client.post(
        "/api/catalog/example/model/preflight",
        json={
            "revision": initial["revision"],
            "role": "chat",
            "engine": "llama.cpp",
            "selected_files": option["selected_files"],
        },
    )
    assert response.status_code == 200
    fresh = response.json()
    assert fresh["selected_files"] == ["small.gguf"]
    assert fresh["expected_sha256"] == {"small.gguf": "b" * 64}
    assert fresh["hardware_fit"] == option["hardware_fit"]
    assert fresh["install_plan"]["id"] != initial["install_plan"]["id"]
    assert fresh["revision"] == initial["revision"]
    assert inspect.await_count == 2
