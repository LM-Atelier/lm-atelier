from __future__ import annotations

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

    def __init__(self, code: Literal["network-refused", "network-host-refused"]) -> None:
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


class _LeasedTransport(httpx.AsyncBaseTransport):
    # The lease is asked before the request reaches the transport underneath,
    # which is where the host is looked up and connected to.
    def __init__(self, lease: OutboundLease, inner: httpx.AsyncBaseTransport) -> None:
        self._lease = lease
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._lease.check(request.url.host)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


def outbound_client(
    lease: OutboundLease,
    *,
    timeout: httpx.Timeout | float,
    headers: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """A client whose every request is asked of ``lease`` before its host is looked up.

    It follows no redirect, since each caller checks every hop itself, and
    takes no proxy or trust roots from the environment: a proxy would make the
    connection somewhere the caller's own checks never see.
    """

    inner = transport or httpx.AsyncHTTPTransport(
        verify=shared_tls_context(trust_environment=False)
    )
    return httpx.AsyncClient(
        transport=_LeasedTransport(lease, inner),
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
