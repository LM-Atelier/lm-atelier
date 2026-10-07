"""What a workflow search must ask CivitAI for, pinned against the wire.

Three things here are not defensive detail; each is a way the request can look
successful and be wrong.

A workflow is not a resource on CivitAI - there is no workflows endpoint. It is
a model whose type is a workflow type, and there are TWO of them. The published
enums page lists only `Workflows`; `ComfyWorkflows` is live and undocumented,
and the two sets are disjoint, so asking for the documented one returns a
quietly partial library rather than an error.

Multi-value `types` must be REPEATED parameters. The comma-joined form is
refused outright with HTTP 400, which is survivable because it is loud. The
bracket form is the dangerous one: it is accepted, ignored, and answers with
checkpoints and LoRAs as though no filter had been applied. Unknown parameters
are likewise ignored rather than rejected, so a 200 never establishes that a
parameter did anything. That is why these assertions read the encoded query
string rather than trusting the call to have worked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_lm.catalog_cache import CatalogCachePolicy
from local_lm.catalog_sources import WorkflowCatalogSource
from local_lm.civitai_catalog import CivitaiCatalog
from local_lm.config import Settings

SHA256 = "b" * 64


def _graph_file(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": 901,
        "name": "portrait-workflow.json",
        "sizeKB": 12,
        "hashes": {"SHA256": SHA256},
        "pickleScanResult": "Success",
        "virusScanResult": "Success",
        "type": "Workflow",
    }
    value.update(updates)
    return value


def _version(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": 801,
        "name": "v1",
        "publishedAt": "2026-08-01T00:00:00Z",
        "nsfwLevel": 1,
        "baseModel": "Flux.1 D",
        "stats": {"downloadCount": 40, "thumbsUpCount": 9},
        "files": [_graph_file()],
    }
    value.update(updates)
    return value


def _item(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": 701,
        "name": "A portrait workflow",
        "type": "Workflows",
        "nsfw": False,
        "tags": ["comfyui"],
        "baseModels": ["Flux.1 D"],
        "creator": {"username": "someone"},
        "stats": {"downloadCount": 40, "thumbsUpCount": 9},
        "availability": "Public",
        "allowCommercialUse": ["Image"],
        "allowDerivatives": True,
        "allowDifferentLicense": False,
        "allowNoCredit": True,
        "modelVersions": [_version()],
    }
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


async def test_both_workflow_types_are_asked_for_as_repeated_parameters(
    tmp_path: Path,
) -> None:
    """The one assertion that stops a silently partial or unfiltered library."""

    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.query.decode())
        return httpx.Response(200, json={"items": [_item()], "metadata": {}})

    catalog = _catalog(tmp_path, handler)
    page = await catalog.search_workflows(query="portrait")
    await catalog.close()

    assert len(page.items) == 1
    query = seen[0]
    # Repeated, because that is the only form CivitAI honours.
    assert query.count("types=") == 2
    assert "types=Workflows" in query and "types=ComfyWorkflows" in query
    # Not the 400 form, and not the accepted-then-ignored form.
    assert "types%5B%5D" not in query
    assert "Workflows%2CComfyWorkflows" not in query
    # The graph is often not the primary file, so one-file-per-version hides it.
    assert "primaryFileOnly" not in query


async def test_a_workflow_search_does_not_read_a_model_search_cached_page(
    tmp_path: Path,
) -> None:
    """Same words, different question: the two must not share a cache entry."""

    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.query.decode())
        return httpx.Response(200, json={"items": [_item()], "metadata": {}})

    catalog = _catalog(tmp_path, handler)
    await catalog.search(query="portrait")
    await catalog.search_workflows(query="portrait")
    await catalog.close()

    assert len(requests) == 2, "the workflow search reused the model search cache"
    assert "types=" not in requests[0]
    assert requests[1].count("types=") == 2


async def test_a_repeated_workflow_search_is_served_from_cache(tmp_path: Path) -> None:
    """And the cache still works for the question it does own."""

    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"items": [_item()], "metadata": {}})

    catalog = _catalog(tmp_path, handler)
    first = await catalog.search_workflows(query="portrait")
    second = await catalog.search_workflows(query="portrait")
    await catalog.close()

    assert calls == 1
    assert [item.remote_id for item in first.items] == [item.remote_id for item in second.items]
    assert second.stale is False


async def test_an_unreachable_source_serves_the_stale_page_rather_than_failing(
    tmp_path: Path,
) -> None:
    """An installed workflow must stay usable when the provider is down.

    The page is marked stale rather than presented as current, so a caller can
    say so instead of implying the library is up to date.
    """

    failing = False

    async def handler(request: httpx.Request) -> httpx.Response:
        if failing:
            raise httpx.ConnectError("provider unreachable")
        return httpx.Response(200, json={"items": [_item()], "metadata": {}})

    policy = CatalogCachePolicy(fresh_seconds=0, stale_seconds=3600)
    catalog = _catalog(tmp_path, handler, policy=policy)
    await catalog.search_workflows(query="portrait")
    failing = True
    page = await catalog.search_workflows(query="portrait")
    await catalog.close()

    assert page.stale is True
    assert len(page.items) == 1


def test_the_civitai_source_satisfies_the_workflow_protocol(tmp_path: Path) -> None:
    """Capability is a runtime question here, so something must actually check it.

    The second protocol buys a signature nothing else has to carry; the price is
    that a source which does not serve workflows is only discovered when asked.
    This is where that is established for the one source that does.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [], "metadata": {}})

    catalog = _catalog(tmp_path, handler)
    source: WorkflowCatalogSource = catalog

    assert callable(source.search_workflows)
    assert source.source_id
    assert source.web_origin


def _workflow_next_page(cursor: str = "next") -> str:
    return (
        "https://civitai.com/api/v1/models?nsfw=false"
        "&types=Workflows&types=ComfyWorkflows&cursor=" + cursor
    )


async def test_a_workflow_continuation_is_followed_rather_than_silently_dropped(
    tmp_path: Path,
) -> None:
    """Page two must be reachable, and reusing the model policy loses it.

    The model cursor policy requires `primaryFileOnly=true`. A workflow response
    carries no such parameter, because the workflow search deliberately does not
    send one, so under that policy the continuation fails validation, is swallowed
    as None, and the library silently ends after its first page.
    """

    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if len(seen) == 1:
            return httpx.Response(
                200,
                json={
                    "items": [_item()],
                    "metadata": {"nextPage": _workflow_next_page()},
                },
            )
        # A distinct VERSION, because the card id is the version id rather
        # than the model id - varying the model alone proves nothing here.
        return httpx.Response(
            200,
            json={"items": [_item(id=702, modelVersions=[_version(id=802)])], "metadata": {}},
        )

    catalog = _catalog(tmp_path, handler)
    first = await catalog.search_workflows(query="portrait")
    assert first.next_cursor, "the workflow continuation was dropped"
    second = await catalog.search_workflows(query="portrait", cursor=first.next_cursor)
    await catalog.close()

    assert len(seen) == 2
    assert "cursor=next" in seen[1]
    assert [item.remote_id for item in second.items] != [item.remote_id for item in first.items]


async def test_a_model_cursor_cannot_be_spent_on_a_workflow_search(
    tmp_path: Path,
) -> None:
    """The other half of the same defect, and the more dangerous half.

    A model continuation satisfies the model policy, so reusing that policy here
    would accept it and answer a workflow search with checkpoints - a wrong
    answer rather than a failure.
    """

    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"items": [_item()], "metadata": {}})

    model_cursor = (
        "https://civitai.com/api/v1/models?nsfw=false"
        "&primaryFileOnly=true&types=Checkpoint&cursor=next"
    )
    catalog = _catalog(tmp_path, handler)
    with pytest.raises(ValueError):
        await catalog.search_workflows(query="portrait", cursor=model_cursor)
    await catalog.close()

    assert seen == [], "a model continuation reached the network as a workflow search"


async def test_a_workflow_cursor_cannot_be_spent_on_a_model_search(
    tmp_path: Path,
) -> None:
    """And the model search keeps the constraint it already had."""

    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"items": [_item()], "metadata": {}})

    catalog = _catalog(tmp_path, handler)
    with pytest.raises(ValueError):
        await catalog.search(query="portrait", cursor=_workflow_next_page())
    await catalog.close()

    assert seen == []


async def test_a_continuation_off_the_endpoint_is_refused_for_both_searches(
    tmp_path: Path,
) -> None:
    """Splitting the policy must not weaken what both halves still share."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [], "metadata": {}})

    elsewhere = (
        "https://example.test/api/v1/models?nsfw=false"
        "&types=Workflows&types=ComfyWorkflows&cursor=next"
    )
    unrated = "https://civitai.com/api/v1/models?types=Workflows&types=ComfyWorkflows"
    catalog = _catalog(tmp_path, handler)
    for bad in (elsewhere, unrated):
        with pytest.raises(ValueError):
            await catalog.search_workflows(query="portrait", cursor=bad)
    await catalog.close()
