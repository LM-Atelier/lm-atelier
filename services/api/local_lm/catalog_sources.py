from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from .schemas import CatalogPage
from .workflow_source_candidates import catalog_host_map


class WorkflowCatalogSource(Protocol):
    """A source that can be asked for workflows rather than models.

    Deliberately a SECOND protocol rather than another parameter on
    `CatalogSource.search`. That signature carries fifteen model-shaped
    filters - quantization, architecture, parameter counts - which no workflow
    source can answer, and it is repeated byte-identically in three files with
    37 call sites between them. Widening it would make every one of those
    places carry a question none of them asks.

    The cost is that serving workflows becomes a runtime capability rather than
    a typed one: a source that does not implement this is discovered when it is
    asked. A caller narrows to it the same way `versions()` is already reached,
    and answers a 404 shaped like the existing catalog-source-not-found when
    the narrowing fails.
    """

    source_id: str
    display_name: str
    web_origin: str

    async def search_workflows(
        self,
        *,
        query: str = "",
        sort: str = "trending",
        limit: int = 30,
        cursor: str | None = None,
    ) -> CatalogPage: ...


class CatalogSource(Protocol):
    source_id: str
    display_name: str
    # The site this source serves, so a link found in a workflow can be
    # attributed to it. A deployment registering a different source brings its
    # own origin with it; nothing elsewhere hardcodes which hosts exist.
    web_origin: str

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

    def host_map(self) -> dict[str, str]:
        """Hostname -> source id for every source registered here.

        This is the only answer to "may we parse a link pointing there": a
        host is acceptable because a registered source serves it, not because
        it appears in a list somewhere.
        """
        return catalog_host_map(
            (source.source_id, getattr(source, "web_origin", ""))
            for source in self._sources.values()
        )

    async def close(self) -> None:
        for source in self._sources.values():
            await source.close()
