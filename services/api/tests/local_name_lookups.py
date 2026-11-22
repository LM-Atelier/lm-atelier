"""While tests run, only the names this computer answers itself can be looked up.

A client that reaches a service by name, as the model catalog and model hub
clients do, looks the name up first, so refusing the lookup stops it before
anything is sent. A refused lookup raises the resolver's own "no such name"
error, which the code under test already handles as a host it could not reach.
Because that code often carries on as if nothing happened, each refusal is
also charged to the test that made it, and that test fails.

Only lookups made through this process's resolver functions are refused. The
guard does not see a connection made straight to an address, which looks
nothing up (the event loop and httpx connect to an IP literal that way), a
name a socket's own methods resolve inside the C library, a request a proxy
carries for its client, or anything a child process does. So a test that
answers a lookup with an address on another computer must also give the code
under test a constructed transport.

A refusal is charged by the thread it happened in: in the thread running the
tests, to the test running at that moment; in any other thread, to the test
that started that thread, directly or through threads of its own, however long
the thread outlives it. A thread one test leaves running therefore never fails
a later test. Refusals that no running test can answer for, because the test
that started their thread had already finished or no test started it, are
reported when the session ends instead.
"""

from __future__ import annotations

import _socket
import ipaddress
import socket
import threading
import weakref
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import pytest

#: What to do instead of reaching a service on another computer. A constructed
#: resolver alone is not enough: the real transport would still connect to
#: whatever address it answers.
INSTEAD = "Give the code under test a constructed transport or answer instead."

REFUSED = f"While testing, only names this computer answers itself may be looked up. {INSTEAD}"

HostOf = Callable[[tuple[Any, ...], Mapping[str, Any]], object]


def answered_here(host: object) -> bool:
    """Whether looking up this host asks nothing of another computer.

    That is no host at all, ``localhost``, and a loopback or unspecified
    address written as an IPv4 address, or as an IPv6 one plain, in brackets,
    with a scope, or carrying an IPv4 address. Every other name is a question
    some resolvers pass on to the network, a name below ``localhost`` and an
    IPv4 address in brackets included.
    """

    if host is None:
        # No host is this computer's own addresses, to listen on.
        return True
    if isinstance(host, bytes | bytearray):
        host = bytes(host).decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    if host == "" or host.casefold() == "localhost":
        return True
    if host.startswith("[") and host.endswith("]") and ":" in host:
        host = host[1:-1]
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped.is_loopback or address.ipv4_mapped.is_unspecified
    return address.is_loopback or address.is_unspecified


def _argument(keyword: str) -> HostOf:
    def host_of(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> object:
        return args[0] if args else kwargs.get(keyword)

    return host_of


def _address_named(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> object:
    address = args[0] if args else kwargs.get("sockaddr")
    flags = args[1] if len(args) > 1 else kwargs.get("flags", 0)
    if isinstance(flags, int) and flags & socket.NI_NUMERICHOST:
        # Asked only to write the address out, which looks nothing up.
        return None
    return address[0] if isinstance(address, tuple) and address else address


#: Each resolver, where its host is, and the error it raises for a name it
#: cannot find.
_RESOLVERS: dict[str, tuple[HostOf, type[OSError], int]] = {
    "getaddrinfo": (_argument("host"), socket.gaierror, socket.EAI_NONAME),
    "gethostbyname": (_argument("hostname"), socket.gaierror, socket.EAI_NONAME),
    "gethostbyname_ex": (_argument("hostname"), socket.gaierror, socket.EAI_NONAME),
    # A reverse lookup reports a host it cannot find as HOST_NOT_FOUND.
    "gethostbyaddr": (_argument("ip_address"), socket.herror, 1),
    "getnameinfo": (_address_named, socket.gaierror, socket.EAI_NONAME),
}


class NameLookupGuard:
    """Refuses lookups beyond this computer and charges each one to the test that made it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runner: threading.Thread | None = None
        self._running: str | None = None
        self._started_by: weakref.WeakKeyDictionary[threading.Thread, str | None] = (
            weakref.WeakKeyDictionary()
        )
        self._refused: Counter[str | None] = Counter()

    def _answerable(self) -> str | None:
        # The caller holds the lock.
        thread = threading.current_thread()
        if thread is self._runner:
            return self._running
        return self._started_by.get(thread)

    def _thread_starting(self, thread: threading.Thread) -> None:
        with self._lock:
            self._started_by[thread] = self._answerable()

    def guarded(
        self, lookup: Callable[..., Any], host_of: HostOf, error: type[OSError], code: int
    ) -> Callable[..., Any]:
        """``lookup``, refusing every host it is asked about but this computer's own."""

        def guarded_lookup(*args: Any, **kwargs: Any) -> Any:
            if not answered_here(host_of(args, kwargs)):
                with self._lock:
                    self._refused[self._answerable()] += 1
                raise error(code, REFUSED)
            return lookup(*args, **kwargs)

        return guarded_lookup

    def install(
        self,
        patch: pytest.MonkeyPatch,
        modules: tuple[object, ...] = (_socket, socket),
    ) -> None:
        """Guard every resolver these modules offer, and note who starts each thread.

        ``socket.getaddrinfo`` is left to call the guarded resolver beneath it,
        so code that kept a reference to it from before this ran is refused too.
        """

        for module in modules:
            for name, (host_of, error, code) in _RESOLVERS.items():
                if module is socket and name == "getaddrinfo" and _socket in modules:
                    continue
                if hasattr(module, name):
                    lookup = getattr(module, name)
                    patch.setattr(module, name, self.guarded(lookup, host_of, error, code))
        start = threading.Thread.start
        guard = self

        def start_noting_the_test(thread: threading.Thread) -> None:
            guard._thread_starting(thread)
            start(thread)

        patch.setattr(threading.Thread, "start", start_noting_the_test)

    def begin(self, test: str) -> None:
        """Charge to ``test`` what this thread looks up, and what the threads it starts do."""

        with self._lock:
            self._runner = threading.current_thread()
            self._running = test

    def end(self, test: str) -> int:
        """Stop charging ``test`` and return how many lookups it was refused."""

        with self._lock:
            if self._running == test:
                self._running = None
            return self._refused.pop(test, 0)

    def unanswered(self) -> dict[str | None, int]:
        """Refusals no test answered for, by the test that started their thread, if any."""

        with self._lock:
            return {test: count for test, count in self._refused.items() if count}


def refused_during_test(refused: int) -> str:
    """Why a test fails for the lookups it was refused."""

    lookups = (
        "lookup of a name beyond this one was"
        if refused == 1
        else "lookups of names beyond this one were"
    )
    return f"This test tried to reach another computer: {refused} {lookups} refused. {INSTEAD}"


def refused_unanswered(unanswered: Mapping[str | None, int]) -> str:
    """Why the session fails for refusals no running test could answer for."""

    lines = [
        "Lookups of names beyond this computer were refused where no running test "
        "could answer for them:"
    ]
    for test, count in sorted(unanswered.items(), key=lambda item: item[0] or ""):
        threads = "a thread" if count == 1 else "threads"
        where = (
            "outside any test, between tests or in a thread no test started"
            if test is None
            else f"in {threads} started by {test}, after that test had finished"
        )
        lines.append(f"  {count} {where}")
    lines.append(INSTEAD)
    return "\n".join(lines)
