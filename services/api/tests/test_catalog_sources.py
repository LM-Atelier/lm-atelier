from __future__ import annotations

from typing import Any

import pytest

from local_lm.catalog_sources import CatalogSourceNotFound, CatalogSources
from local_lm.schemas import CatalogPage


class FakeCatalogSource:
    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self.display_name = source_id.title()
        self.closed = False

    def validate_item_id(self, item_id: str) -> bool:
        return bool(item_id)

    async def search(self, **_parameters: Any) -> CatalogPage:
        return CatalogPage(items=[])

    async def inspect(
        self,
        item_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict[str, Any]:
        return {
            "model": {
                "remote_id": item_id,
                "name": item_id,
                "compatibility": "likely",
            },
            "revision": revision,
            "files": [],
        }

    async def inspect_file_prefix(
        self,
        item_id: str,
        revision: str,
        filename: str,
        *,
        max_bytes: int,
    ) -> bytes:
        return b""

    async def close(self) -> None:
        self.closed = True


async def test_catalog_sources_resolve_default_and_close_once() -> None:
    source = FakeCatalogSource("huggingface")
    sources = CatalogSources([source])

    assert sources.default() is source
    assert sources.get("huggingface") is source

    await sources.close()

    assert source.closed is True


def test_catalog_sources_reject_unknown_and_duplicate_ids() -> None:
    source = FakeCatalogSource("huggingface")
    sources = CatalogSources([source])

    with pytest.raises(CatalogSourceNotFound, match="unknown catalog source"):
        sources.get("missing")
    with pytest.raises(ValueError, match="duplicate catalog source"):
        CatalogSources([source, FakeCatalogSource("huggingface")])
    with pytest.raises(ValueError, match="default catalog source"):
        CatalogSources([source], default_source_id="missing")
