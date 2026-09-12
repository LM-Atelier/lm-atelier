"""The Discover route: what it asks, and what it says when it cannot.

Workflow discovery is a separate route from model discovery because the two ask
different questions. The tests that matter here are therefore about the SEAMS
rather than about search itself, which the source's own tests already pin:

- a source that does not serve workflows must say so, and must not answer with
  an empty page, because an empty page is indistinguishable from a source that
  serves workflows and has none today;
- an unreachable provider must not take the library down with it;
- a bad continuation must be refused rather than followed.

The registered source is reached through `app.state.services.catalog_sources`
and its method replaced with monkeypatch, rather than by writing into the
registry's internals. The first case needs no substitution at all: Hugging Face
genuinely has no workflow capability, so it is the real thing rather than a
stand-in for it.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.schemas import CatalogPage


def _civitai(app: FastAPI) -> Any:
    return app.state.services.catalog_sources.get("civitai")


def _answers(
    monkeypatch: pytest.MonkeyPatch,
    app: FastAPI,
    *,
    page: CatalogPage | None = None,
    error: Exception | None = None,
) -> list[dict[str, Any]]:
    """Make the registered source answer a fixed way, and record what it was asked."""

    calls: list[dict[str, Any]] = []
    answer = page if page is not None else CatalogPage(items=[])

    async def search_workflows(**kwargs: Any) -> CatalogPage:
        calls.append(kwargs)
        if error is not None:
            raise error
        return answer

    monkeypatch.setattr(_civitai(app), "search_workflows", search_workflows)
    return calls


async def test_a_source_without_workflows_says_so_rather_than_answering_empty(
    client: AsyncClient,
) -> None:
    """The distinction this route exists to preserve.

    An empty page would read as "this source has no workflows today", which is a
    claim about the library. The truth is "this source does not do workflows at
    all", which is a claim about the source, and only one of them is actionable.

    Hugging Face is the real case rather than a fixture: it is registered, it
    serves models, and it has no workflow capability.
    """

    response = await client.get("/api/workflow-catalog", params={"source": "huggingface"})

    assert response.status_code == 404
    assert response.json()["code"] == "catalog-source-serves-no-workflows"


async def test_an_unknown_source_is_refused_with_the_existing_shape(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/workflow-catalog", params={"source": "nowhere"})

    assert response.status_code == 404
    assert response.json()["code"] == "catalog-source-not-found"


async def test_a_search_reaches_the_source_with_only_neutral_arguments(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No model filter can leak in, because the route does not accept one."""

    calls = _answers(monkeypatch, app)

    response = await client.get(
        "/api/workflow-catalog",
        params={"source": "civitai", "query": "portrait", "sort": "newest", "limit": 5},
    )

    assert response.status_code == 200
    assert calls == [{"query": "portrait", "sort": "newest", "limit": 5, "cursor": None}]
    # The model vocabulary is absent by construction, not by filtering.
    assert not {"role", "quantization", "architecture"} & set(calls[0])


async def test_an_unreachable_provider_does_not_take_the_library_down(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """503 and a sentence a person can act on, not a stack trace."""

    _answers(monkeypatch, app, error=httpx.ConnectError("unreachable"))

    response = await client.get("/api/workflow-catalog", params={"source": "civitai"})

    assert response.status_code == 503
    assert response.json()["code"] == "catalog-unavailable"


async def test_a_refused_continuation_is_a_bad_request_not_an_outage(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cursor the source refuses is the caller's problem, so 422 not 503."""

    _answers(monkeypatch, app, error=ValueError("CivitAI catalog cursor is invalid"))

    response = await client.get(
        "/api/workflow-catalog",
        params={"source": "civitai", "cursor": "https://elsewhere.test/api/v1/models"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "catalog-request-invalid"


async def test_a_stale_page_is_still_served_and_still_says_it_is_stale(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider being down must not re-render an installed library as unready."""

    _answers(monkeypatch, app, page=CatalogPage(items=[], stale=True))

    response = await client.get("/api/workflow-catalog", params={"source": "civitai"})

    assert response.status_code == 200
    assert response.json()["stale"] is True


@pytest.mark.parametrize("limit", [0, 101])
async def test_an_out_of_range_limit_is_refused_before_the_provider_is_called(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    calls = _answers(monkeypatch, app)

    response = await client.get(
        "/api/workflow-catalog", params={"source": "civitai", "limit": limit}
    )

    assert response.status_code == 422
    assert calls == []
