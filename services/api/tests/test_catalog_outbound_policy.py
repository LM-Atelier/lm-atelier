"""The Hugging Face catalog asks the outbound policy before every request and redirect."""

from __future__ import annotations

from pathlib import Path

import httpcore
import httpx
import pytest

from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.network import OutboundPolicy, OutboundRequestRefused

PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
REMOTE_ID = "owner/Model-8B-GGUF"


@pytest.fixture(autouse=True)
def no_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


def network(monkeypatch: pytest.MonkeyPatch, answer: dict[str, httpx.Response]) -> list[str]:
    """Answer each request the real client sends by its host, recording the hosts reached."""

    reached: list[str] = []

    async def send(transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        del transport
        reached.append(request.url.host)
        return answer[request.url.host]

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    return reached


def moved(location: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": location})


async def checked(host: str, *, domains: frozenset[str]) -> list[str]:
    reached: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        reached.append(request.url.host)
        return httpx.Response(200)

    policy = OutboundPolicy(lambda: True)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [policy.request_check("model-catalog", domains=domains)]},
    ) as client:
        await client.get(f"https://{host}/")
    return reached


@pytest.mark.parametrize(
    "host",
    ["huggingface.co", "HuggingFace.co.", "cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co"],
)
async def test_a_domain_admits_itself_and_the_names_below_it(host: str) -> None:
    assert len(await checked(host, domains=frozenset({"huggingface.co", "hf.co"}))) == 1


@pytest.mark.parametrize(
    "host",
    ["example.com", "evilhf.co", "huggingface.co.example.com", "hf.co.example.com", "203.0.113.7"],
)
async def test_a_host_outside_the_domains_is_refused_before_it_is_reached(host: str) -> None:
    with pytest.raises(OutboundRequestRefused) as refused:
        await checked(host, domains=frozenset({"huggingface.co", "hf.co"}))

    assert refused.value.code == "network-host-refused"


async def test_an_address_is_never_a_name_below_a_domain() -> None:
    # An address written as a domain admits that address and no name that
    # merely ends in the same digits, and an address is never below a name.
    assert await checked("192.0.2.1", domains=frozenset({"192.0.2.1"})) == ["192.0.2.1"]
    with pytest.raises(OutboundRequestRefused):
        await checked("10.192.0.2.1", domains=frozenset({"192.0.2.1"}))
    with pytest.raises(OutboundRequestRefused):
        await checked("203.0.113.7", domains=frozenset({"113.7"}))


async def test_the_policy_is_asked_again_at_every_request() -> None:
    allowed = [True]
    reached: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        reached.append(request.url.host)
        return httpx.Response(200)

    policy = OutboundPolicy(lambda: allowed[0])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        event_hooks={"request": [policy.request_check("web-page")]},
    ) as client:
        await client.get("https://example.com/")
        allowed[0] = False
        with pytest.raises(OutboundRequestRefused) as refused:
            await client.get("https://example.com/")

    assert refused.value.code == "network-refused"
    assert reached == ["example.com"]


async def test_a_client_can_be_made_before_its_purpose_is_allowed() -> None:
    # A client is made at startup; only a request may be refused, never the
    # making of the client.
    allowed = [False]
    policy = OutboundPolicy(lambda: allowed[0])
    check = policy.request_check("web-page")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
        event_hooks={"request": [check]},
    ) as client:
        with pytest.raises(OutboundRequestRefused):
            await client.get("https://example.com/")
        allowed[0] = True
        assert (await client.get("https://example.com/")).status_code == 200


async def test_the_catalog_follows_a_redirect_to_its_own_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reached = network(
        monkeypatch,
        {
            "huggingface.co": moved("https://cas-bridge.xethub.hf.co/object?signature=s"),
            "cas-bridge.xethub.hf.co": httpx.Response(206, content=b"GGUF-prefix"),
        },
    )
    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    try:
        prefix = await catalog.inspect_file_prefix(REMOTE_ID, "a" * 40, "model.gguf", max_bytes=64)
    finally:
        await catalog.close()

    assert prefix == b"GGUF-prefix"
    assert reached == ["huggingface.co", "cas-bridge.xethub.hf.co"]


async def test_the_catalog_refuses_a_redirect_elsewhere_before_reaching_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reached = network(
        monkeypatch, {"huggingface.co": moved("https://files.example.com/model.gguf")}
    )
    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    try:
        with pytest.raises(OutboundRequestRefused) as refused:
            await catalog.inspect_file_prefix(REMOTE_ID, "a" * 40, "model.gguf", max_bytes=64)
    finally:
        await catalog.close()

    assert refused.value.code == "network-host-refused"
    assert reached == ["huggingface.co"]


async def test_a_refused_redirect_falls_back_to_saved_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listing = [
        {"id": REMOTE_ID, "pipeline_tag": "text-generation", "tags": ["gguf", "license:mit"]}
    ]
    answer = {"huggingface.co": httpx.Response(200, json=listing)}
    reached = network(monkeypatch, answer)
    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    try:
        live = await catalog.search(role="chat", sort="trending")
        answer["huggingface.co"] = moved("http://127.0.0.1:8080/api/models")
        # Another filter, so the answer is not the fresh saved page itself.
        saved = await catalog.search(role="chat", sort="trending", license_id="mit")
    finally:
        await catalog.close()

    assert live.stale is False
    assert [item.remote_id for item in saved.items] == [REMOTE_ID]
    assert saved.stale is True
    assert reached == ["huggingface.co", "huggingface.co"]


async def test_the_catalog_still_reaches_hugging_face_through_an_environment_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:3128")
    through: list[bool] = []
    reached: list[str] = []

    async def send(transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        through.append(isinstance(transport._pool, httpcore.AsyncHTTPProxy))
        reached.append(request.url.host)
        if request.url.host == "huggingface.co":
            return moved("https://files.example.com/model.gguf")
        return httpx.Response(206, content=b"elsewhere")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    try:
        with pytest.raises(OutboundRequestRefused):
            await catalog.inspect_file_prefix(REMOTE_ID, "a" * 40, "model.gguf", max_bytes=64)
    finally:
        await catalog.close()

    # The request went through the proxy, and the refused hop never reached
    # the proxy at all.
    assert through == [True]
    assert reached == ["huggingface.co"]
