from __future__ import annotations

import socket
import ssl
from unittest.mock import Mock, call

import httpx
import pytest

from local_lm import network
from local_lm.web_retrieval import fetch_source


def test_tls_context_is_reused_until_trust_environment_changes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    first = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    second = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    isolated = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    factory = Mock(side_effect=[first, second, isolated])
    monkeypatch.setattr(httpx, "create_ssl_context", factory)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    network._tls_context_for_environment.cache_clear()

    try:
        assert network.shared_tls_context() is first
        assert network.shared_tls_context() is first

        monkeypatch.setenv("SSL_CERT_FILE", "alternate-trust-roots.pem")
        assert network.shared_tls_context() is second
        assert network.shared_tls_context(trust_environment=False) is isolated
        monkeypatch.setenv("SSL_CERT_FILE", "another-trust-root.pem")
        assert network.shared_tls_context(trust_environment=False) is isolated
        assert factory.call_args_list == [
            call(trust_env=True),
            call(trust_env=True),
            call(trust_env=False),
        ]
    finally:
        network._tls_context_for_environment.cache_clear()


@pytest.fixture
def reached(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every name lookup tried, each refused, so no request can connect anywhere.

    A connection to a named host starts with its lookup, so a request that
    was never looked up was never connected either.
    """

    seen: list[str] = []

    def lookup(host: object, *args: object, **kwargs: object) -> object:
        seen.append(host.decode() if isinstance(host, bytes) else str(host))
        raise OSError("this test looks nothing up")

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    return seen


def _policy(allowed: list[bool]) -> network.OutboundPolicy:
    # A list, so a test can turn the switch off after the lease is taken.
    return network.OutboundPolicy(lambda: allowed[0])


def test_a_purpose_the_policy_refuses_gets_no_lease() -> None:
    with pytest.raises(network.OutboundRefused) as refused:
        _policy([False]).lease("web-page")

    assert refused.value.code == "network-refused"


@pytest.mark.parametrize(
    ("url", "revoked", "code"),
    [
        ("https://other.test/page", False, "network-host-refused"),
        ("https://allowed.test/page", True, "network-refused"),
    ],
    ids=["host-outside-the-lease", "switch-turned-off-after-the-lease"],
)
async def test_a_refused_request_is_never_looked_up_or_connected(
    reached: list[str], url: str, revoked: bool, code: str
) -> None:
    allowed = [True]
    lease = _policy(allowed).lease("web-page", hosts=frozenset({"Allowed.test."}))
    allowed[0] = not revoked

    async with network.outbound_client(lease, timeout=5) as client:
        with pytest.raises(network.OutboundRefused) as refused:
            await client.get(url)

    assert refused.value.code == code
    assert reached == []


async def test_an_allowed_request_is_looked_up_as_usual(reached: list[str]) -> None:
    lease = _policy([True]).lease("web-page", hosts=frozenset({"allowed.test"}))

    async with network.outbound_client(lease, timeout=5) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("https://allowed.test/page")

    # The same instrument as above, seeing the lookup it is there to see.
    assert reached == ["allowed.test"]


async def test_a_page_whose_host_is_refused_is_never_looked_up() -> None:
    looked_up: list[str] = []
    requested: list[str] = []

    def lookup(host: str, port: object) -> object:
        looked_up.append(host)
        raise AssertionError("looked up")

    async def request(url: str) -> object:
        requested.append(url)
        raise AssertionError("requested")

    lease = _policy([True]).lease("web-page", hosts=frozenset({"allowed.test"}))
    with pytest.raises(network.OutboundRefused):
        await fetch_source(
            "https://other.test/page", request=request, resolve=lease.resolver(lookup)
        )

    assert (looked_up, requested) == ([], [])
