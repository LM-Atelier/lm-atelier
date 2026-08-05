from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_lm.catalog_cache import CatalogCachePolicy
from local_lm.civitai_catalog import CivitaiCatalog
from local_lm.config import Settings

SHA256 = "a" * 64


def _file(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": 301,
        "name": "model.safetensors",
        "sizeKB": 1024,
        "hashes": {"SHA256": SHA256},
        "pickleScanResult": "Success",
        "virusScanResult": "Success",
        "type": "Model",
        "metadata": {"format": "SafeTensor", "fp": "fp16"},
    }
    value.update(updates)
    return value


def _version(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": 201,
        "name": "Version 1",
        "publishedAt": "2026-07-31T00:00:00Z",
        "nsfwLevel": 1,
        "baseModel": "SDXL 1.0",
        "trainedWords": ["portrait-style"],
        "stats": {"downloadCount": 25, "thumbsUpCount": 4},
        "files": [_file()],
    }
    value.update(updates)
    return value


def _item(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": 101,
        "name": "Safe model",
        "type": "Checkpoint",
        "nsfw": False,
        "tags": ["photorealistic"],
        "baseModels": ["SDXL 1.0"],
        "creator": {"username": "creator"},
        "stats": {"downloadCount": 50, "thumbsUpCount": 8},
        "availability": "Public",
        "allowCommercialUse": ["Image"],
        "allowDerivatives": True,
        "allowDifferentLicense": False,
        "allowNoCredit": True,
        "modelVersions": [_version()],
    }
    value.update(updates)
    return value


def _version_detail(**updates: Any) -> dict[str, Any]:
    value = _version(
        modelId=101,
        model={"name": "Safe model", "nsfw": False, "type": "Checkpoint"},
    )
    value.update(updates)
    return value


def _catalog(
    tmp_path: Path,
    handler: Any,
    *,
    policy: CatalogCachePolicy | None = None,
    sleep: Any = asyncio.sleep,
) -> CivitaiCatalog:
    return CivitaiCatalog(
        Settings(data_dir=tmp_path),
        transport=httpx.MockTransport(handler),
        cache_policy=policy,
        sleep=sleep,
    )


def test_civitai_item_ids_are_bounded_positive_decimals() -> None:
    assert CivitaiCatalog.validate_item_id("1") is True
    assert CivitaiCatalog.validate_item_id("123456789012") is True
    assert CivitaiCatalog.validate_item_id("0") is False
    assert CivitaiCatalog.validate_item_id("01") is False
    assert CivitaiCatalog.validate_item_id("1234567890123") is False
    assert CivitaiCatalog.validate_item_id("../model") is False


async def test_search_is_general_only_normalized_cached_and_cursor_bounded(
    tmp_path: Path,
) -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.host == "civitai.com"
        assert request.url.params["nsfw"] == "false"
        assert request.url.params["primaryFileOnly"] == "true"
        assert request.url.params["types"] == "Checkpoint"
        return httpx.Response(
            200,
            json={
                "items": [
                    _item(
                        modelVersions=[
                            _version(),
                            _version(
                                id=202,
                                name="Version 2",
                                nsfwLevel=3,
                                baseModel="Flux.1 D",
                            ),
                            _version(id=203, name="Mature version", nsfwLevel=5),
                        ]
                    ),
                    _item(id=102, name="Filtered model", nsfw=True),
                ],
                "metadata": {
                    "nextPage": (
                        "https://civitai.com/api/v1/models?cursor=next"
                        "&nsfw=false&primaryFileOnly=true"
                    ),
                },
            },
        )

    catalog = _catalog(tmp_path, handler)
    try:
        page = await catalog.search(query="portrait", role="image", limit=999)
        cached = await catalog.search(query="portrait", role="image", limit=999)
    finally:
        await catalog.close()

    assert calls == 1
    assert page == cached
    assert page.next_cursor == (
        "https://civitai.com/api/v1/models?cursor=next&nsfw=false&primaryFileOnly=true"
    )
    assert [item.remote_id for item in page.items] == ["201", "202"]
    assert page.items[0].provider == "civitai"
    assert page.items[0].content_rating == "general"
    assert page.items[0].architecture == "SDXL 1.0"
    assert page.items[0].formats == ["safetensors"]
    assert page.items[0].quantizations == ["fp16"]
    assert page.items[0].total_size_bytes == 1024 * 1024
    assert page.items[0].compatibility == "advanced_import"


async def test_search_rejects_untrusted_cursor_before_network(tmp_path: Path) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid cursors must not reach the network")

    catalog = _catalog(tmp_path, handler)
    try:
        with pytest.raises(ValueError, match="cursor is invalid"):
            await catalog.search(cursor="https://attacker.invalid/api/v1/models")
        with pytest.raises(ValueError, match="cursor is invalid"):
            await catalog.search(cursor="http://civitai.com/api/v1/models")
        with pytest.raises(ValueError, match="cursor is invalid"):
            await catalog.search(cursor="https://civitai.com/api/v1/models/101")
        with pytest.raises(ValueError, match="cursor is invalid"):
            await catalog.search(
                cursor=(
                    "https://civitai.com/api/v1/models?cursor=next&nsfw=true&primaryFileOnly=true"
                )
            )
        with pytest.raises(ValueError, match="cursor is invalid"):
            await catalog.search(cursor="https://civitai.com/api/v1/models?cursor=next")
    finally:
        await catalog.close()


async def test_search_drops_untrusted_next_page_and_skips_non_image_roles(
    tmp_path: Path,
) -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "items": [_item()],
                "metadata": {"nextPage": "https://attacker.invalid/api/v1/models?cursor=next"},
            },
        )

    catalog = _catalog(tmp_path, handler)
    try:
        chat = await catalog.search(role="chat")
        video = await catalog.search(role="video")
        image = await catalog.search(role="image")
    finally:
        await catalog.close()

    assert chat.items == []
    assert video.items == []
    assert calls == 1
    assert image.next_cursor is None


async def test_lora_role_requests_and_normalizes_only_lora_assets(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["types"] == "LORA"
        return httpx.Response(
            200,
            json={"items": [_item(type="LORA")], "metadata": {}},
        )

    catalog = _catalog(tmp_path, handler)
    try:
        page = await catalog.search(role="lora")
    finally:
        await catalog.close()

    assert len(page.items) == 1
    assert page.items[0].pipeline_tag == "lora"
    assert page.items[0].compatibility == "advanced_import"
    assert page.items[0].required_runtime == "comfyui"


async def test_rate_limit_retry_is_bounded(tmp_path: Path) -> None:
    calls = 0
    delays: list[float] = []

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "2"})
        return httpx.Response(200, json={"items": [], "metadata": {}})

    async def sleep(delay: float) -> None:
        delays.append(delay)

    catalog = _catalog(tmp_path, handler, sleep=sleep)
    try:
        page = await catalog.search()
    finally:
        await catalog.close()

    assert page.items == []
    assert calls == 2
    assert delays == [2.0]


async def test_long_retry_after_fails_without_sleeping(tmp_path: Path) -> None:
    delays: list[float] = []

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "31"})

    async def sleep(delay: float) -> None:
        delays.append(delay)

    catalog = _catalog(tmp_path, handler, sleep=sleep)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await catalog.search()
    finally:
        await catalog.close()

    assert delays == []


async def test_response_body_size_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{" + b" " * 32 + b"}")

    monkeypatch.setattr("local_lm.civitai_catalog._MAX_RESPONSE_BYTES", 16)
    catalog = _catalog(tmp_path, handler)
    try:
        with pytest.raises(ValueError, match="response exceeds the size limit"):
            await catalog.search()
    finally:
        await catalog.close()


async def test_transient_failure_uses_stale_cache(tmp_path: Path) -> None:
    failing = False

    async def handler(request: httpx.Request) -> httpx.Response:
        if failing:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json={"items": [_item()], "metadata": {}})

    catalog = _catalog(
        tmp_path,
        handler,
        policy=CatalogCachePolicy(fresh_seconds=-1, stale_seconds=60),
    )
    try:
        first = await catalog.search(query="cached")
        failing = True
        stale = await catalog.search(query="cached")
    finally:
        await catalog.close()

    assert first.stale is False
    assert stale.stale is True
    assert stale.items[0].remote_id == "201"


async def test_inspect_pins_version_and_preserves_recommendation_metadata(
    tmp_path: Path,
) -> None:
    requested_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/api/v1/model-versions/202":
            return httpx.Response(
                200,
                json=_version_detail(
                    id=202,
                    name="Instruction Edit",
                    publishedAt="2026-07-31T00:00:00Z",
                    trainedWords=["change-outfit", "portrait-style"],
                ),
            )
        return httpx.Response(
            200,
            json=_item(
                tags=["portrait"],
                modelVersions=[
                    _version(id=201, publishedAt="2026-07-30T00:00:00Z"),
                    _version(
                        id=202,
                        publishedAt="2026-07-31T00:00:00Z",
                        trainedWords=["change-outfit", "portrait-style"],
                    ),
                ],
            ),
        )

    catalog = _catalog(tmp_path, handler)
    try:
        detail = await catalog.inspect("202", requested_role="image")
        with pytest.raises(NotImplementedError, match="verified transfer support"):
            await catalog.inspect_file_prefix(
                "202",
                "202",
                "model.safetensors",
                max_bytes=1024,
            )
    finally:
        await catalog.close()

    assert requested_paths == [
        "/api/v1/model-versions/202",
        "/api/v1/models/101",
    ]
    assert detail["revision"] == "202"
    assert detail["model"]["provider"] == "civitai"
    assert detail["files"][0]["sha256"] == SHA256
    assert detail["files"][0]["source_file_type"] == "Model"
    assert detail["files"][0]["source_file_precision"] == "fp16"
    metadata = detail["files"][0]["metadata"]
    assert metadata["content_rating"] == "general"
    assert metadata["source_model_id"] == "101"
    assert metadata["source_version_id"] == "202"
    assert metadata["edit_tailored"] == "declared"
    assert metadata["trained_words"] == ["change-outfit", "portrait-style"]
    assert metadata["permissions"]["allow_derivatives"] is True


async def test_inspect_rejects_mature_version_before_parent_lookup(tmp_path: Path) -> None:
    requested_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(200, json=_version_detail(id=202, nsfwLevel=5))

    catalog = _catalog(tmp_path, handler)
    try:
        with pytest.raises(ValueError, match="not available in the general catalog"):
            await catalog.inspect("202", requested_role="image")
    finally:
        await catalog.close()

    assert requested_paths == ["/api/v1/model-versions/202"]


async def test_metadata_arrays_are_bounded(tmp_path: Path) -> None:
    values = [f"value-{index}" for index in range(200)]

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/model-versions/201":
            return httpx.Response(200, json=_version_detail(trainedWords=values))
        return httpx.Response(200, json=_item(tags=values))

    catalog = _catalog(tmp_path, handler)
    try:
        detail = await catalog.inspect("201", requested_role="image")
    finally:
        await catalog.close()

    metadata = detail["files"][0]["metadata"]
    assert len(metadata["trained_words"]) == 128
    assert len(metadata["tags"]) == 128


async def test_unsafe_or_unhashed_weights_are_not_install_candidates(tmp_path: Path) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [
                    _item(
                        modelVersions=[
                            _version(
                                files=[
                                    _file(
                                        hashes={},
                                        pickleScanResult="Danger",
                                    )
                                ]
                            )
                        ]
                    )
                ],
                "metadata": {},
            },
        )

    catalog = _catalog(tmp_path, handler)
    try:
        page = await catalog.search(role="image")
    finally:
        await catalog.close()

    assert page.items[0].compatibility == "unsupported"
    assert page.items[0].compatibility_reasons == ["no scan-cleared, hash-pinned safetensors file"]


async def test_requests_are_serialized(tmp_path: Path) -> None:
    active = 0
    max_active = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json={"items": [], "metadata": {}})

    catalog = _catalog(tmp_path, handler)
    try:
        await asyncio.gather(
            catalog.search(query="first"),
            catalog.search(query="second"),
        )
    finally:
        await catalog.close()

    assert max_active == 1


async def test_versions_lists_general_summaries_in_provider_order_and_caches(
    tmp_path: Path,
) -> None:
    requested_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(
            200,
            json=_item(
                modelVersions=[
                    _version(
                        id=203,
                        name="Version 3",
                        publishedAt="2026-07-31T00:00:00Z",
                        description="Sharper hands",
                    ),
                    _version(id=202, name="Mature only", nsfwLevel=5),
                    _version(id=201, publishedAt="2026-07-30T00:00:00Z"),
                ],
            ),
        )

    catalog = _catalog(tmp_path, handler)
    try:
        summary = await catalog.versions("101")
        again = await catalog.versions("101")
    finally:
        await catalog.close()

    # One request: the second call answered from the fresh cache.
    assert requested_paths == ["/api/v1/models/101"]
    assert summary == again
    assert summary["model_id"] == "101"
    assert summary["model_name"] == "Safe model"
    # The mature version is absent, not marked, and provider order holds.
    assert [entry["version_id"] for entry in summary["versions"]] == ["203", "201"]
    assert summary["versions"][0]["changelog"] == "Sharper hands"
    assert summary["versions"][0]["published_at"] == "2026-07-31T00:00:00Z"


async def test_versions_refuses_a_mature_parent_model(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_item(nsfw=True))

    catalog = _catalog(tmp_path, handler)
    try:
        with pytest.raises(ValueError, match="not available in the general catalog"):
            await catalog.versions("101")
    finally:
        await catalog.close()


def test_a_card_names_the_model_it_is_a_version_of() -> None:
    """A CivitAI card is one version, because a version is what installs.

    The library wants to list versions under one parent without giving up the
    version identity the download path is bound to, so the card carries both.
    """
    version = {"id": 9001, "name": "v3.0", "baseModel": "SDXL 1.0", "files": [_file()]}
    card = CivitaiCatalog._normalize(
        {"id": 4201, "name": "Lustify", "type": "Checkpoint", "modelVersions": [version]},
        "image",
        version=version,
    )

    assert card.remote_id == "9001"
    assert card.parent_model_id == "4201"
    assert card.parent_model_name == "Lustify"
    # The version identity is untouched: install still binds to the version.
    assert card.parent_model_id != card.remote_id
