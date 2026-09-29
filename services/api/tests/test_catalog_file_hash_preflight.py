from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest
from httpx2 import AsyncClient

from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.schemas import CatalogDetail, CatalogModel, SystemInfo
from tests.test_catalog_hardware_advice import _system
from tests.test_model_planner import _gguf


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/catalog/example/model/preflight",
        "/api/catalog/preflight?source=huggingface&id=example/model",
    ],
)
@pytest.mark.parametrize("companion", [False, True], ids=["primary", "companion"])
async def test_preflight_records_complete_selected_file_hashes(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, endpoint: str, companion: bool
) -> None:
    content = _gguf({"general.architecture": "llama"})
    expected = hashlib.sha256(content).hexdigest()
    detail = CatalogDetail(
        model=CatalogModel(remote_id="example/model", name="Fixture", compatibility="likely"),
        revision="a" * 40,
        files=[{"filename": "chosen.gguf", "size": len(content), "sha256": None}],
    )
    if companion:
        detail.files[0].update(
            source_remote_id="example/companion",
            source_revision="b" * 40,
            source_filename="original.gguf",
        )

    def system_info(_settings: Settings) -> SystemInfo:
        return _system(None, device=False)

    monkeypatch.setattr("local_lm.api.collect_system_info", system_info)
    monkeypatch.setattr(HuggingFaceCatalog, "inspect", AsyncMock(return_value=detail.model_dump()))
    monkeypatch.setattr(
        HuggingFaceCatalog, "discover_vision_projector", AsyncMock(return_value=None)
    )
    prefix = AsyncMock(return_value=content)
    monkeypatch.setattr(HuggingFaceCatalog, "inspect_file_prefix", prefix)
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
    assert payload["can_install"]
    assert payload["install_plan"]["compatibility"] == "supported"
    assert payload["expected_sha256"] == {"chosen.gguf": expected}
    artifacts = payload["install_plan"]["artifacts_json"]
    assert len(artifacts) == 1
    assert artifacts[0]["sha256"] == expected
    assert artifacts[0]["size_bytes"] == len(content)
    if companion:
        source = payload["file_sources"]["chosen.gguf"]
        assert source["sha256"] == expected
        assert source["remote_id"] == "example/companion"
        assert source["revision"] == "b" * 40
        assert source["filename"] == "original.gguf"
        assert artifacts[0]["source_remote_id"] == source["remote_id"]
        assert artifacts[0]["source_revision"] == source["revision"]
        assert artifacts[0]["source_path"] == source["filename"]
        assert all(
            call.args[:3] == ("example/companion", "b" * 40, "original.gguf")
            for call in prefix.await_args_list
        )
    checksum = next(check for check in payload["checks"] if check["id"] == "checksum")
    assert checksum["status"] == "pass"
    assert detail.files[0]["sha256"] is None
