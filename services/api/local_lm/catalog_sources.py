from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from .schemas import CatalogPage


class CatalogSource(Protocol):
    source_id: str
    display_name: str

    def validate_item_id(self, item_id: str) -> bool: ...

    async def search(
        self,
        *,
        query: str = "",
        role: str | None = None,
        sort: str = "trending",
        limit: int = 30,
        cursor: str | None = None,
        compatibility: str | None = None,
        file_format: str | None = None,
        quantization: str | None = None,
        license_id: str | None = None,
        gated: str | None = None,
        architecture: str | None = None,
        min_parameters: int | None = None,
        max_parameters: int | None = None,
        max_size_bytes: int | None = None,
        updated_within_days: int | None = None,
    ) -> CatalogPage: ...

    async def inspect(
        self,
        item_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict[str, Any]: ...

    async def inspect_file_prefix(
        self,
        item_id: str,
        revision: str,
        filename: str,
        *,
        max_bytes: int,
    ) -> bytes: ...

    async def close(self) -> None: ...


class CatalogSourceNotFound(ValueError):
    pass


class CatalogSources:
    def __init__(
        self,
        sources: Iterable[CatalogSource],
        *,
        default_source_id: str = "huggingface",
    ) -> None:
        indexed: dict[str, CatalogSource] = {}
        for source in sources:
            if source.source_id in indexed:
                raise ValueError(f"duplicate catalog source: {source.source_id}")
            indexed[source.source_id] = source
        if default_source_id not in indexed:
            raise ValueError("default catalog source is not registered")
        self._sources = indexed
        self.default_source_id = default_source_id

    def get(self, source_id: str) -> CatalogSource:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise CatalogSourceNotFound(f"unknown catalog source: {source_id}") from exc

    def default(self) -> CatalogSource:
        return self.get(self.default_source_id)

    async def close(self) -> None:
        for source in self._sources.values():
            await source.close()
