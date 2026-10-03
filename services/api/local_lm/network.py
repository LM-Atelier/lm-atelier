from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import ssl
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import httpx


@lru_cache(maxsize=8)
def _tls_context_for_environment(
    trust_environment: bool,
    certificate_file: str | None,
    certificate_directory: str | None,
) -> ssl.SSLContext:
    # httpx honors these environment variables while constructing its default
    # context. Including them in the key prevents a later configuration change
    # from silently reusing the wrong trust roots.
    _ = certificate_file, certificate_directory
    return httpx.create_ssl_context(trust_env=trust_environment)


def shared_tls_context(*, trust_environment: bool = True) -> ssl.SSLContext:
    """Reuse immutable trust roots across independent outbound client pools."""

    return _tls_context_for_environment(
        trust_environment,
        os.environ.get("SSL_CERT_FILE") if trust_environment else None,
        os.environ.get("SSL_CERT_DIR") if trust_environment else None,
    )


OutboundPurpose = Literal["web-page", "web-search"]


class OutboundRefused(Exception):
    """An outbound request this installation does not allow, refused before any lookup."""

    def __init__(
        self,
        code: Literal["network-refused", "network-host-refused", "network-address-refused"],
    ) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class OutboundPolicy:
    """Which outbound purposes this installation allows, read again at every use.

    Only the web purposes exist so far, and they follow the installation's web
    access switch. Reading it at every use, not once, is what lets a switch
    turned off in the middle of a request stop that request's next hop.
    """

    web_access_enabled: Callable[[], bool]

    @classmethod
    def from_settings(cls, settings: Any) -> OutboundPolicy:
        return cls(lambda: settings.web_access_enabled is True)

    def allows(self, purpose: OutboundPurpose) -> bool:
        return self.web_access_enabled()

    def lease(
        self, purpose: OutboundPurpose, *, hosts: frozenset[str] | None = None
    ) -> OutboundLease:
        """Permission for one purpose, to the given hosts or to any host its caller admits."""

        if not self.allows(purpose):
            raise OutboundRefused("network-refused")
        return OutboundLease(
            self, purpose, None if hosts is None else frozenset(map(_host_key, hosts))
        )


@dataclass(frozen=True)
class OutboundLease:
    """What one outbound use may reach. It is checked again before every lookup and request."""

    policy: OutboundPolicy
    purpose: OutboundPurpose
    hosts: frozenset[str] | None

    def check(self, host: str) -> None:
        if not self.policy.allows(self.purpose):
            raise OutboundRefused("network-refused")
        if self.hosts is not None and _host_key(host) not in self.hosts:
            raise OutboundRefused("network-host-refused")

    def resolver(self, base: Callable[..., Any] | None = None) -> Callable[..., Any]:
        """A name lookup that asks this lease first, so a refused host is never looked up.

        Without ``base`` it uses the system lookup as it stands when called,
        not as it stood when this was made.
        """

        def resolve(host: str, port: str | int | bytes | None, *args: Any, **kwargs: Any) -> Any:
            self.check(host)
            return (base or socket.getaddrinfo)(host, port, *args, **kwargs)

        return resolve


Address = ipaddress.IPv4Address | ipaddress.IPv6Address


class _LeasedTransport(httpx.AsyncBaseTransport):
    # The lease is asked before the request reaches the transport underneath,
    # which is where the host is looked up and connected to.
    def __init__(
        self,
        lease: OutboundLease,
        inner: httpx.AsyncBaseTransport,
        admit: Callable[[Address], bool] | None,
    ) -> None:
        self._lease = lease
        self._inner = inner
        self._admit = admit

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._lease.check(request.url.host)
        if self._admit is None:
            return await self._inner.handle_async_request(request)
        *earlier, last = await asyncio.to_thread(_admitted, self._lease, self._admit, request)
        for address in earlier:
            try:
                return await self._inner.handle_async_request(_sent_to(request, address))
            except httpx.ConnectError:
                continue
        return await self._inner.handle_async_request(_sent_to(request, last))

    async def aclose(self) -> None:
        await self._inner.aclose()


def _admitted(
    lease: OutboundLease, admit: Callable[[Address], bool], request: httpx.Request
) -> list[Address]:
    """Every address the request's host has now, refused unless ``admit`` takes each one."""

    try:
        found = lease.resolver()(request.url.host, None, 0, socket.SOCK_STREAM)
    except OSError as error:
        raise httpx.ConnectError("That host could not be found.", request=request) from error
    addresses: list[Address] = []
    for entry in found:
        try:
            address = ipaddress.ip_address(entry[4][0])
        except (ValueError, IndexError):
            raise OutboundRefused("network-address-refused") from None
        if not admit(address):
            raise OutboundRefused("network-address-refused")
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise httpx.ConnectError("That host could not be found.", request=request)
    return addresses


def _sent_to(request: httpx.Request, address: Address) -> httpx.Request:
    # The same request for the address that was checked, so the transport
    # looks nothing up again. The name still goes in the Host header, and the
    # certificate is still checked against it.
    return httpx.Request(
        request.method,
        request.url.copy_with(host=address.compressed),
        headers=request.headers,
        stream=request.stream,
        extensions={**request.extensions, "sni_hostname": request.url.host},
    )


def outbound_client(
    lease: OutboundLease,
    *,
    timeout: httpx.Timeout | float,
    headers: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    admit: Callable[[Address], bool] | None = None,
) -> httpx.AsyncClient:
    """A client whose every request is asked of ``lease`` before its host is looked up.

    It follows no redirect, since each caller checks every hop itself, and
    takes no proxy or trust roots from the environment: a proxy would make the
    connection somewhere the caller's own checks never see.

    With ``admit``, a request is connected only to an address it admits. The
    host is looked up once, through the lease, as the request is sent, and the
    connection goes to that answer, so a name that resolved to an admitted
    address when the caller checked it cannot answer with another one by the
    time it is connected to. Every address in the answer must be admitted.
    """

    if transport is not None:
        inner = transport
    elif admit is None:
        inner = httpx.AsyncHTTPTransport(verify=shared_tls_context(trust_environment=False))
    else:
        # A connection to a checked address is known by that address alone, so
        # one kept open could carry a later request under another name, whose
        # certificate it was never checked against. None is kept open.
        inner = httpx.AsyncHTTPTransport(
            verify=shared_tls_context(trust_environment=False),
            limits=httpx.Limits(max_keepalive_connections=0),
        )
    return httpx.AsyncClient(
        transport=_LeasedTransport(lease, inner, admit),
        follow_redirects=False,
        trust_env=False,
        timeout=timeout,
        headers=headers,
    )


def _host_key(host: str) -> str:
    """One spelling of a host, so a name approved in one form matches a request in another.

    A request carries a name decoded (an internationalized name in its own
    letters) while an approval may hold it as written (``xn--`` punycode), and
    an address may be written expanded or compressed. Names become their
    lowercase ASCII form and addresses their compressed form.
    """

    name = host.strip().rstrip(".")
    try:
        return ipaddress.ip_address(name.strip("[]")).compressed
    except ValueError:
        pass
    if any(character in name for character in "/\\@?#:"):
        # Not a host name at all, so it can match no host a request names.
        return name.casefold()
    try:
        return httpx.URL(f"https://{name}/").raw_host.decode("ascii").casefold()
    except (httpx.InvalidURL, UnicodeError):
        return name.casefold()
