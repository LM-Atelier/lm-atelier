"""Fetching one workflow graph: what is refused, and what identity it carries.

Two rules are doing the work here and both are deliberate refusals rather than
conveniences.

A version can carry several files - the graph, a bundle, sample images - and the
file list is the only thing that says which is which. Exactly one graph
candidate is required. Zero and several are both refused, naming what was
found, because the install this feeds is meant to pause at a genuine ambiguity
rather than guess which file somebody meant.

And the digest is taken over the bytes that arrived, not over a re-serialization
of the parsed graph. Those differ for identical content - key order, spacing -
so hashing the round trip would quietly break the claim that what gets installed
is what was inspected.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_lm.civitai_catalog import CivitaiCatalog
from local_lm.config import Settings

GRAPH = {"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}
GRAPH_BYTES = json.dumps(GRAPH, indent=2).encode("utf-8")
DOWNLOAD = "https://civitai.com/api/download/models/801"


def _file(**updates: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": 901,
        "name": "portrait-workflow.json",
        "type": "Workflow",
        "sizeKB": 1,
        "downloadUrl": DOWNLOAD,
    }
    value.update(updates)
    return value


def _version(*files: dict[str, Any]) -> dict[str, Any]:
    return {"id": 801, "name": "v1", "files": list(files) or [_file()]}


def _catalog(tmp_path: Path, handler: Any) -> CivitaiCatalog:
    return CivitaiCatalog(
        Settings(data_dir=tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=asyncio.sleep,
    )


def _serving(version: dict[str, Any], body: bytes = GRAPH_BYTES) -> tuple[Any, list[str]]:
    """A transport handler, and the list of URLs it was actually asked for.

    The URLs come back beside the handler rather than hanging off it, so a test
    can assert that something was never requested without reaching for an
    attribute the type checker has to be told to ignore.
    """

    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "model-versions" in str(request.url):
            return httpx.Response(200, json=version)
        return httpx.Response(200, content=body)

    return handler, seen


async def test_a_version_with_one_graph_returns_it_under_the_bytes_it_arrived_as(
    tmp_path: Path,
) -> None:
    handler, _ = _serving(_version())
    catalog = _catalog(tmp_path, handler)
    artifact = await catalog.fetch_workflow_graph("801")
    await catalog.close()

    assert artifact.graph == GRAPH
    assert artifact.raw == GRAPH_BYTES
    assert artifact.sha256 == hashlib.sha256(GRAPH_BYTES).hexdigest()
    assert artifact.provider_filename == "portrait-workflow.json"
    # The digest is over what arrived. A re-serialization of the same graph has
    # a different digest, which is exactly the mistake this guards.
    assert artifact.sha256 != hashlib.sha256(json.dumps(GRAPH).encode()).hexdigest()


async def test_a_version_with_no_graph_says_what_it_does_carry(tmp_path: Path) -> None:
    version = _version(_file(name="bundle.zip", type="Archive"))
    handler, _ = _serving(version)
    catalog = _catalog(tmp_path, handler)

    with pytest.raises(ValueError) as error:
        await catalog.fetch_workflow_graph("801")
    await catalog.close()

    assert "no workflow graph" in str(error.value)
    assert "bundle.zip" in str(error.value)


async def test_two_graphs_are_an_ambiguity_that_names_both(tmp_path: Path) -> None:
    """The pause the one-click install is allowed to make."""

    version = _version(_file(name="simple.json"), _file(id=902, name="detailed.json"))
    handler, _ = _serving(version)
    catalog = _catalog(tmp_path, handler)

    with pytest.raises(ValueError) as error:
        await catalog.fetch_workflow_graph("801")
    await catalog.close()

    assert "ambiguous" in str(error.value)
    assert "simple.json" in str(error.value)
    assert "detailed.json" in str(error.value)


async def test_a_graph_is_recognised_by_suffix_when_the_type_is_absent(
    tmp_path: Path,
) -> None:
    """The declared type is better evidence, but it is not always present."""

    version = _version(_file(type=None), _file(id=903, name="preview.png", type="Image"))
    handler, _ = _serving(version)
    catalog = _catalog(tmp_path, handler)
    artifact = await catalog.fetch_workflow_graph("801")
    await catalog.close()

    assert artifact.provider_filename == "portrait-workflow.json"


async def test_a_download_address_off_the_provider_is_refused_before_it_is_followed(
    tmp_path: Path,
) -> None:
    """The URL arrives inside a response body, so it is not trusted on sight."""

    handler, seen = _serving(_version(_file(downloadUrl="https://elsewhere.test/graph.json")))
    catalog = _catalog(tmp_path, handler)

    with pytest.raises(ValueError, match="not an allowed delivery target"):
        await catalog.fetch_workflow_graph("801")
    await catalog.close()

    assert not any("elsewhere.test" in url for url in seen)


async def test_a_graph_past_the_ceiling_is_refused_rather_than_truncated(
    tmp_path: Path,
) -> None:
    """Truncating would fail later at the parse and report the wrong cause."""

    oversized = b'{"padding": "' + b"x" * (3 * 1024 * 1024) + b'"}'
    handler, _ = _serving(_version(), body=oversized)
    catalog = _catalog(tmp_path, handler)

    with pytest.raises(ValueError, match="larger than this reader accepts"):
        await catalog.fetch_workflow_graph("801")
    await catalog.close()


async def test_a_body_that_is_not_a_json_object_is_refused(tmp_path: Path) -> None:
    handler, _ = _serving(_version(), body=b"[1, 2, 3]")
    catalog = _catalog(tmp_path, handler)

    with pytest.raises(ValueError, match="not a JSON object"):
        await catalog.fetch_workflow_graph("801")
    await catalog.close()


async def test_a_body_that_is_not_json_at_all_says_so(tmp_path: Path) -> None:
    handler, _ = _serving(_version(), body=b"<html>nope</html>")
    catalog = _catalog(tmp_path, handler)

    with pytest.raises(ValueError, match="not valid JSON"):
        await catalog.fetch_workflow_graph("801")
    await catalog.close()


async def test_a_malformed_version_id_never_reaches_the_network(tmp_path: Path) -> None:
    handler, seen = _serving(_version())
    catalog = _catalog(tmp_path, handler)

    with pytest.raises(ValueError, match="positive decimal integer"):
        await catalog.fetch_workflow_graph("../secrets")
    await catalog.close()

    assert seen == []


def _redirecting(
    status: int, location: str, body: bytes = GRAPH_BYTES
) -> tuple[Any, list[httpx.Request]]:
    """CivitAI answers a download with a redirect to wherever the bytes live."""

    seen: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url)
        if "model-versions" in url:
            return httpx.Response(200, json=_version())
        if url == DOWNLOAD:
            return httpx.Response(status, headers={"location": location})
        return httpx.Response(200, content=body)

    return handler, seen


@pytest.mark.parametrize("status", [302, 307])
async def test_the_provider_delivery_hop_is_followed(tmp_path: Path, status: int) -> None:
    """The normal path, not an edge case.

    CivitAI does not serve the file from the address you ask for. A reader that
    refuses this redirect reaches no file at all, so the hop has to be followed
    for any graph to arrive.
    """

    delivery = "https://b2.civitai.com/neutral/workflow.json"
    handler, seen = _redirecting(status, delivery)
    catalog = _catalog(tmp_path, handler)
    artifact = await catalog.fetch_workflow_graph("801")
    await catalog.close()

    assert artifact.graph == GRAPH
    assert artifact.sha256 == hashlib.sha256(GRAPH_BYTES).hexdigest()
    assert str(seen[-1].url) == delivery


async def test_the_catalog_token_does_not_travel_to_the_delivery_host(
    tmp_path: Path,
) -> None:
    """The signed address already authorizes the file, so the token buys nothing.

    Sending it anyway would disclose our credential to a host that never needed
    it, which is a worse outcome than a failed download.
    """

    delivery = "https://b2.civitai.com/neutral/workflow.json"
    handler, seen = _redirecting(302, delivery)
    catalog = CivitaiCatalog(
        Settings(data_dir=tmp_path),
        transport=httpx.MockTransport(handler),
        token="SECRET-CATALOG-TOKEN",
        sleep=asyncio.sleep,
    )
    await catalog.fetch_workflow_graph("801")
    await catalog.close()

    by_host = {request.url.host: request.headers for request in seen}
    assert "authorization" in by_host["civitai.com"]
    assert "authorization" not in by_host["b2.civitai.com"]
    assert not any(
        "SECRET-CATALOG-TOKEN" in str(value)
        for request in seen
        if request.url.host != "civitai.com"
        for value in request.headers.values()
    )


async def test_a_redirect_off_the_allowed_delivery_hosts_is_refused(
    tmp_path: Path,
) -> None:
    """The far side chooses the destination, so every hop is checked, not just the first."""

    handler, seen = _redirecting(302, "https://elsewhere.test/workflow.json")
    catalog = _catalog(tmp_path, handler)

    with pytest.raises(ValueError, match="not an allowed delivery target"):
        await catalog.fetch_workflow_graph("801")
    await catalog.close()

    assert not any(request.url.host == "elsewhere.test" for request in seen)


async def test_a_redirect_that_keeps_going_is_refused_rather_than_walked(
    tmp_path: Path,
) -> None:
    """A chain that does not end is a loop or a mistake; either way, say so."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if "model-versions" in str(request.url):
            return httpx.Response(200, json=_version())
        return httpx.Response(302, headers={"location": "https://b2.civitai.com/again.json"})

    catalog = _catalog(tmp_path, handler)

    with pytest.raises(ValueError, match="too many times"):
        await catalog.fetch_workflow_graph("801")
    await catalog.close()
