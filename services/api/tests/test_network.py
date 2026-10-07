from __future__ import annotations

import asyncio
import ipaddress
import socket
import ssl
from collections.abc import AsyncIterator
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


class _Loopback:
    """A plain HTTP server on this machine that records every connection and request it gets."""

    def __init__(self) -> None:
        self.connections = 0
        self.requests: list[bytes] = []
        self.port = 0
        self.handlers: set[asyncio.Future[object]] = set()

    async def answer(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        handler = asyncio.current_task()
        if handler is not None:
            self.handlers.add(handler)
        try:
            # Kept open for further requests until the client closes it, which
            # also keeps Windows from resetting an answer not yet read.
            while True:
                self.requests.append(await reader.readuntil(b"\r\n\r\n"))
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
                await writer.drain()
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


@pytest.fixture
async def loopback() -> AsyncIterator[_Loopback]:
    server = _Loopback()
    listening = await asyncio.start_server(server.answer, "127.0.0.1", 0)
    server.port = listening.sockets[0].getsockname()[1]
    try:
        yield server
    finally:
        listening.close()
        # Each connection's own handler, not wait_closed(): on Windows a
        # connection the client closed first can fail to report that it was
        # lost, and wait_closed() then waits for it forever.
        if server.handlers:
            await asyncio.wait(server.handlers, timeout=5)


def _answering(monkeypatch: pytest.MonkeyPatch, addresses: list[str]) -> list[str]:
    """Answer every name lookup with ``addresses``, recording each name asked for."""

    seen: list[str] = []

    def lookup(host: object, port: object, *args: object, **kwargs: object) -> object:
        seen.append(host.decode() if isinstance(host, bytes) else str(host))
        family = {4: socket.AF_INET, 6: socket.AF_INET6}
        return [
            (family[ipaddress.ip_address(a).version], socket.SOCK_STREAM, 6, "", (a, 0))
            for a in addresses
        ]

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    return seen


@pytest.mark.parametrize(
    "addresses",
    [["127.0.0.1"], ["::1", "127.0.0.1"]],
    ids=["a-refused-address", "a-refused-address-beside-an-admitted-one"],
)
async def test_a_request_is_connected_only_to_addresses_the_caller_admits(
    monkeypatch: pytest.MonkeyPatch, loopback: _Loopback, addresses: list[str]
) -> None:
    looked_up = _answering(monkeypatch, addresses)
    lease = _policy([True]).lease("web-page")

    # A caller that admits only IPv6 addresses, so nothing here leaves this machine.
    async with network.outbound_client(
        lease, timeout=5, admit=lambda address: address.version == 6
    ) as client:
        with pytest.raises(network.OutboundRefused) as refused:
            await client.get(f"http://allowed.test:{loopback.port}/page")

    assert refused.value.code == "network-address-refused"
    assert looked_up == ["allowed.test"]
    assert loopback.connections == 0


async def test_an_admitted_address_is_connected_to_under_the_name_it_was_asked_for(
    monkeypatch: pytest.MonkeyPatch, loopback: _Loopback
) -> None:
    # Nothing listens on the first address, so the second is tried after it.
    looked_up = _answering(monkeypatch, ["::1", "127.0.0.1"])
    lease = _policy([True]).lease("web-page")

    async with network.outbound_client(
        lease, timeout=5, admit=lambda address: address.is_loopback
    ) as client:
        response = await client.get(f"http://allowed.test:{loopback.port}/page")

    assert response.status_code == 200 and response.text == "ok"
    # Looked up once, by the client, and not again by the connection.
    assert looked_up == ["allowed.test"]
    assert loopback.connections == 1
    assert f"\r\nHost: allowed.test:{loopback.port}\r\n".encode() in loopback.requests[0]


async def test_a_connection_is_not_kept_for_a_request_under_another_name(
    monkeypatch: pytest.MonkeyPatch, loopback: _Loopback
) -> None:
    # Both names answer with the same address.
    _answering(monkeypatch, ["127.0.0.1"])
    lease = _policy([True]).lease("web-page")

    async with network.outbound_client(
        lease, timeout=5, admit=lambda address: address.is_loopback
    ) as client:
        for name in ("first.test", "second.test"):
            response = await client.get(f"http://{name}:{loopback.port}/page")
            assert response.status_code == 200

    assert loopback.connections == 2
    assert [b"\r\nHost: second.test:" in request for request in loopback.requests] == [False, True]


async def test_a_request_sent_to_its_checked_address_keeps_its_name_for_the_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answering(monkeypatch, ["93.184.216.34"])
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    lease = _policy([True]).lease("web-page")
    async with network.outbound_client(
        lease,
        timeout=5,
        transport=httpx.MockTransport(answer),
        admit=lambda address: address.is_global,
    ) as client:
        await client.get("https://allowed.test/page")

    assert [request.url.host for request in sent] == ["93.184.216.34"]
    assert sent[0].headers["host"] == "allowed.test"
    assert sent[0].extensions["sni_hostname"] == "allowed.test"
