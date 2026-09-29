"""Optional model cards follow the inspected revision into accepted metadata."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings


@pytest.mark.parametrize(
    "card,status,expected",
    [
        (b"Watercolor landscapes", 200, "Watercolor landscapes"),
        (b"x" * 32_000, 206, "x" * 8_000),
        (b"x" * 32_001, 200, ""),
        (b"\xff", 200, ""),
        (b" \n", 200, ""),
        (b"", 404, ""),
        (b"", 403, ""),
        (b"", 503, ""),
    ],
    ids=["text", "bounded", "oversized", "invalid-utf8", "empty", "missing", "gated", "offline"],
)
async def test_model_card_is_bounded_optional_and_pinned(
    tmp_path: Path, card: bytes, status: int, expected: str
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(
                200,
                json={
                    "id": "sample/model",
                    "sha": "a" * 40,
                    "siblings": [
                        {"rfilename": "README.md"},
                        {"rfilename": "model.gguf", "size": 100},
                    ],
                },
            )
        assert request.url.path == "/sample/model/resolve/" + "a" * 40 + "/README.md"
        assert request.headers["range"] == "bytes=0-31999"
        return httpx.Response(status, content=card)

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co", transport=httpx.MockTransport(handler)
    )
    try:
        detail = await catalog.inspect("sample/model", "main", "chat")
        cached = await catalog.inspect("sample/model", "main", "chat")
    finally:
        await catalog.close()
    assert len(requests) == 2
    assert cached == detail
    assert detail["revision"] == "a" * 40
    assert detail["files"][1]["filename"] == "model.gguf"
    assert detail["files"][1].get("metadata", {}).get("description", "") == expected


@pytest.mark.parametrize("revision,has_card", [("main", True), ("a" * 40, False)])
async def test_missing_card_or_unresolved_revision_does_not_fetch_prose(
    tmp_path: Path, revision: str, has_card: bool
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "sample/model",
                "sha": revision,
                "siblings": [{"rfilename": "README.md"}] if has_card else [],
            },
        )

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co", transport=httpx.MockTransport(handler)
    )
    try:
        detail = await catalog.inspect("sample/model")
    finally:
        await catalog.close()
    assert len(requests) == 1
    assert all("metadata" not in item for item in detail["files"])


async def test_slow_card_fetch_is_cancelled_without_losing_file_inspection(tmp_path: Path) -> None:
    cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(
                200,
                json={
                    "id": "sample/model",
                    "sha": "a" * 40,
                    "siblings": [{"rfilename": "README.md"}, {"rfilename": "model.gguf"}],
                },
            )
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("The stalled request unexpectedly completed")

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co", transport=httpx.MockTransport(handler)
    )
    try:
        async with asyncio.timeout(10):
            detail = await catalog.inspect("sample/model")
    finally:
        await catalog.close()
    assert cancelled.is_set()
    assert detail["files"][1]["filename"] == "model.gguf"
    assert "metadata" not in detail["files"][1]
