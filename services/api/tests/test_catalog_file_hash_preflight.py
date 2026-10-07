from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest
from httpx2 import AsyncClient
from test_catalog_hardware_advice import _system
from test_model_planner import _gguf

from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.schemas import CatalogDetail, CatalogModel, SystemInfo


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


@pytest.mark.parametrize("revision,size", [("main", 128), ("a" * 40, 5 * 1024 * 1024)])
async def test_preflight_explains_why_incomplete_evidence_cannot_be_installed(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, revision: str, size: int
) -> None:
    content = _gguf({"general.architecture": "llama"})
    detail = CatalogDetail(
        model=CatalogModel(remote_id="example/model", name="Fixture", compatibility="likely"),
        revision=revision,
        files=[{"filename": "chosen.gguf", "size": size, "sha256": None}],
    )
    monkeypatch.setattr(
        "local_lm.api.collect_system_info", lambda _settings: _system(None, device=False)
    )
    monkeypatch.setattr(HuggingFaceCatalog, "inspect", AsyncMock(return_value=detail.model_dump()))
    monkeypatch.setattr(
        HuggingFaceCatalog, "discover_vision_projector", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(HuggingFaceCatalog, "inspect_file_prefix", AsyncMock(return_value=content))
    response = await client.post(
        "/api/catalog/example/model/preflight",
        json={
            "revision": revision,
            "role": "chat",
            "engine": "llama.cpp",
            "selected_files": ["chosen.gguf"],
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["can_install"] is False
    assert payload["install_plan"]["compatibility"] == "unsupported"
    assert payload["install_plan"]["failure_code"] == "preflight_blocked"
    check = next(check for check in payload["checks"] if check["id"] == "install-evidence")
    assert check["status"] == "block"
    assert "fully verified" in check["detail"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/catalog/example/model/preflight",
        "/api/catalog/preflight?source=huggingface&id=example/model",
    ],
)
async def test_failed_file_verification_can_be_retried(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    header = _gguf({"general.architecture": "llama"})
    size = 5 * 1024 * 1024
    content = header + b"\0" * (size - len(header))
    detail = CatalogDetail(
        model=CatalogModel(remote_id="example/model", name="Fixture", compatibility="likely"),
        revision="a" * 40,
        files=[{"filename": "chosen.gguf", "size": size, "sha256": None}],
    )
    monkeypatch.setattr(
        "local_lm.api.collect_system_info", lambda _settings: _system(None, device=False)
    )
    monkeypatch.setattr(HuggingFaceCatalog, "inspect", AsyncMock(return_value=detail.model_dump()))
    monkeypatch.setattr(
        HuggingFaceCatalog, "discover_vision_projector", AsyncMock(return_value=None)
    )
    complete_reads = 0

    async def read(_remote: str, _revision: str, _filename: str, *, max_bytes: int) -> bytes:
        nonlocal complete_reads
        if max_bytes == size + 1:
            complete_reads += 1
            if complete_reads == 1:
                raise OSError("Temporary constructed read failure")
        return content[:max_bytes]

    monkeypatch.setattr(HuggingFaceCatalog, "inspect_file_prefix", AsyncMock(side_effect=read))
    request = {
        "revision": "a" * 40,
        "role": "chat",
        "engine": "llama.cpp",
        "selected_files": ["chosen.gguf"],
    }
    first = await client.post(endpoint, json=request)
    assert first.status_code == 200, first.text
    blocked = first.json()
    assert blocked["can_install"] is False
    assert blocked["install_plan"]["failure_code"] == "preflight_blocked"
    check = next(item for item in blocked["checks"] if item["id"] == "install-evidence")
    assert check["status"] == "block"
    assert "could not be read" in check["detail"]
    assert "Run the install check again" in check["detail"]
    assert "provider did not supply" not in blocked["install_plan"]["failure_reason"]
    second = await client.post(endpoint, json=request)
    assert second.status_code == 200, second.text
    accepted = second.json()
    assert accepted["can_install"] is True
    assert accepted["install_plan"]["compatibility"] == "supported"
    assert accepted["expected_sha256"] == {"chosen.gguf": hashlib.sha256(content).hexdigest()}
    assert complete_reads == 2
    assert detail.files[0]["sha256"] is None
