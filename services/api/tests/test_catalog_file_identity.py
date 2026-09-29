from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from local_lm import catalog_file_identity as module


async def _hash(
    files: list[dict[str, Any]], read: AsyncMock, *, provider: str = "huggingface"
) -> list[dict[str, Any]]:
    return await module.hash_selected_catalog_files(
        files, provider=provider, remote_id="example/model", revision="a" * 40, read=read
    )


async def test_hashes_complete_config_without_mutating_catalog_metadata() -> None:
    content = b'{"model_type":"fixture"}'
    original = {"filename": "config.json", "size": len(content), "sha256": None}
    reader = AsyncMock(return_value=content)
    result = await _hash([original], reader)
    assert result == [{**original, "sha256": hashlib.sha256(content).hexdigest()}]
    assert original["sha256"] is None
    reader.assert_awaited_once_with(
        "example/model", "a" * 40, "config.json", max_bytes=len(content) + 1
    )


async def test_hashes_companion_bytes_at_their_own_source_identity() -> None:
    reader = AsyncMock(return_value=b"abc")
    source = {
        "filename": "nested/companion.json",
        "size": 3,
        "source_remote_id": "example/companion",
        "source_revision": "b" * 40,
        "source_filename": "original.json",
    }
    result = await _hash([source], reader)
    assert result == [{**source, "sha256": hashlib.sha256(b"abc").hexdigest()}]
    reader.assert_awaited_once_with("example/companion", "b" * 40, "original.json", max_bytes=4)


@pytest.mark.parametrize("content", [b"ab", b"abcd", b"abcde"])
async def test_incomplete_or_longer_files_do_not_supply_a_hash(content: bytes) -> None:
    original = {"filename": "config.json", "size": 3}
    assert await _hash([original], AsyncMock(return_value=content)) == [original]


@pytest.mark.parametrize("digest", ["a" * 64, "A" * 64, "invalid", ""])
async def test_explicit_hash_metadata_is_never_replaced(digest: str) -> None:
    original = {"filename": "config.json", "size": 3, "sha256": digest}
    reader = AsyncMock()
    assert await _hash([original], reader) == [original]
    reader.assert_not_awaited()


@pytest.mark.parametrize("size", [None, True, False, 0, -1, "3", 3.5, 4 * 1024 * 1024 + 1])
async def test_unknown_or_unbounded_sizes_are_not_read(size: object) -> None:
    original = {"filename": "config.json", "size": size}
    reader = AsyncMock()
    assert await _hash([original], reader) == [original]
    reader.assert_not_awaited()


@pytest.mark.parametrize("revision", ["main", "abc1234", "g" * 40, "123"])
async def test_mutable_or_invalid_revisions_are_not_read(revision: str) -> None:
    original = {"filename": "config.json", "size": 3}
    reader = AsyncMock()
    result = await module.hash_selected_catalog_files(
        [original],
        provider="huggingface",
        remote_id="example/model",
        revision=revision,
        read=reader,
    )
    assert result == [original]
    reader.assert_not_awaited()


@pytest.mark.parametrize(
    "source",
    [
        {"source_remote_id": "example/other"},
        {"source_revision": "b" * 40},
        {"source_filename": "config.json"},
        {
            "source_remote_id": "example/other",
            "source_revision": "main",
            "source_filename": "config.json",
        },
        {
            "source_remote_id": "https://example.invalid/file",
            "source_revision": "b" * 40,
            "source_filename": "config.json",
        },
    ],
)
async def test_companions_never_fall_back_to_the_primary_identity(source: dict[str, str]) -> None:
    original = {"filename": "config.json", "size": 3, **source}
    reader = AsyncMock()
    assert await _hash([original], reader) == [original]
    reader.assert_not_awaited()


@pytest.mark.parametrize("provider", ["civitai", "other"])
async def test_other_providers_keep_their_own_hash_authority(provider: str) -> None:
    original = {"filename": "config.json", "size": 3}
    reader = AsyncMock()
    assert await _hash([original], reader, provider=provider) == [original]
    reader.assert_not_awaited()


async def test_read_count_is_bounded_even_when_every_read_fails() -> None:
    originals = [{"filename": f"{index}.json", "size": 3} for index in range(20)]
    reader = AsyncMock(side_effect=OSError("Unavailable"))
    assert await _hash(originals, reader) == originals
    assert reader.await_count == 16


async def test_total_read_bytes_are_reserved_before_each_attempt() -> None:
    originals = [{"filename": f"{index}.json", "size": 4 * 1024 * 1024} for index in range(8)]
    reader = AsyncMock(side_effect=OSError("Unavailable"))
    assert await _hash(originals, reader) == originals
    assert reader.await_count == 3
    assert sum(call.kwargs["max_bytes"] for call in reader.await_args_list) <= 16 * 1024 * 1024


async def test_read_deadline_preserves_completed_hashes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_HASH_TIMEOUT_SECONDS", 0.05)
    originals = [{"filename": name, "size": 3} for name in ["first.json", "held.json"]]
    calls = 0

    async def read(*_args: object, **_kwargs: object) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            await asyncio.Event().wait()
        return b"abc"

    result = await _hash(originals, AsyncMock(side_effect=read))
    assert result == [
        {**originals[0], "sha256": hashlib.sha256(b"abc").hexdigest()},
        originals[1],
    ]


async def test_caller_cancellation_is_not_treated_as_missing_metadata() -> None:
    with pytest.raises(asyncio.CancelledError):
        await _hash(
            [{"filename": "config.json", "size": 3}],
            AsyncMock(side_effect=asyncio.CancelledError),
        )
