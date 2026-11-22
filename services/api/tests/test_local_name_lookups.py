"""While tests run, only names this computer answers itself can be looked up.

A lookup of any other name is refused before it leaves this computer, and the
refusal is charged to the test that made it: in the test's own thread, or in a
thread that test started, never in a thread an earlier test left running.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from local_name_lookups import (
    REFUSED,
    NameLookupGuard,
    answered_here,
    refused_during_test,
    refused_unanswered,
)

#: Kept before any test ran, the way code keeps a resolver as a default argument.
_LOOKUP_KEPT_AT_IMPORT = socket.getaddrinfo

#: A name that exists nowhere, so even a lookup that escaped would find nothing.
ELSEWHERE = "catalog.example"


@pytest.mark.parametrize(
    "host",
    [
        None,
        "",
        "localhost",
        "LocalHost",
        b"localhost",
        "127.0.0.1",
        "127.20.30.40",
        "::1",
        "[::1]",
        "::1%1",
        "0:0:0:0:0:0:0:1",
        "::ffff:127.0.0.1",
        "[::ffff:127.0.0.1]",
        "0.0.0.0",
        "::",
        "[::]",
    ],
)
def test_this_computers_own_names_are_answered_here(host: object) -> None:
    assert answered_here(host)


@pytest.mark.parametrize(
    "host",
    [
        ELSEWHERE,
        b"catalog.example",
        "constructed-host",
        "service.localhost",
        "localhost.example",
        "localhost.",
        "[127.0.0.1]",
        "192.0.2.10",
        "[192.0.2.10]",
        "2001:db8::10",
        "[2001:db8::10]",
        "fe80::1%2",
        "::ffff:192.0.2.10",
        "10.0.0.1",
        443,
    ],
)
def test_every_other_name_is_refused(host: object) -> None:
    assert not answered_here(host)


def _constructed_resolver() -> tuple[Any, list[tuple[Any, ...]]]:
    """A resolver of its own, which answers whatever it is asked and notes the question."""

    asked: list[tuple[Any, ...]] = []

    def getaddrinfo(*args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        asked.append((*args, *sorted(kwargs.items())))
        return [("answered", args[0] if args else kwargs.get("host"))]

    return SimpleNamespace(getaddrinfo=getaddrinfo), asked


def _refusals_in(lookup: Callable[[], object], refusals: list[str]) -> Callable[[], None]:
    def look_up() -> None:
        try:
            lookup()
        except socket.gaierror:
            refusals.append("refused")

    return look_up


def test_an_answered_lookup_reaches_the_resolver_unchanged() -> None:
    guard = NameLookupGuard()
    resolver, asked = _constructed_resolver()
    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch, modules=(resolver,))
        guard.begin("this test")

        answer = resolver.getaddrinfo("localhost", 8080, type=socket.SOCK_STREAM)

        assert guard.end("this test") == 0
    assert answer == [("answered", "localhost")]
    assert asked == [("localhost", 8080, ("type", socket.SOCK_STREAM))]


def test_a_refusal_names_neither_the_host_nor_what_was_asked() -> None:
    guard = NameLookupGuard()
    resolver, asked = _constructed_resolver()
    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch, modules=(resolver,))
        guard.begin("this test")

        with pytest.raises(socket.gaierror) as refused:
            resolver.getaddrinfo("private-catalog.example", 4433)

        assert guard.end("this test") == 1
    assert asked == []
    assert refused.value.errno == socket.EAI_NONAME
    assert refused.value.strerror == REFUSED
    assert "private-catalog" not in str(refused.value)
    assert "4433" not in str(refused.value)


def test_a_refusal_is_charged_only_to_the_test_running_when_it_was_made() -> None:
    guard = NameLookupGuard()
    resolver, _ = _constructed_resolver()
    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch, modules=(resolver,))
        guard.begin("first test")
        with pytest.raises(socket.gaierror):
            resolver.getaddrinfo(ELSEWHERE, 443)
        assert guard.end("first test") == 1

        guard.begin("second test")
        assert guard.end("second test") == 0
    assert guard.unanswered() == {}


def test_a_thread_a_test_starts_is_charged_to_that_test() -> None:
    guard = NameLookupGuard()
    resolver, _ = _constructed_resolver()
    refusals: list[str] = []
    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch, modules=(resolver,))
        guard.begin("this test")
        worker = threading.Thread(
            target=_refusals_in(lambda: resolver.getaddrinfo(ELSEWHERE, 443), refusals)
        )
        worker.start()
        worker.join(timeout=10)

        assert guard.end("this test") == 1
    assert refusals == ["refused"]
    assert guard.unanswered() == {}


def test_a_thread_an_earlier_test_left_running_never_fails_a_later_test() -> None:
    guard = NameLookupGuard()
    resolver, _ = _constructed_resolver()
    released = threading.Event()
    refusals: list[str] = []

    def look_up_once_released() -> None:
        released.wait(timeout=10)
        resolver.getaddrinfo(ELSEWHERE, 443)

    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch, modules=(resolver,))
        guard.begin("earlier test")
        left_running = threading.Thread(target=_refusals_in(look_up_once_released, refusals))
        left_running.start()
        assert guard.end("earlier test") == 0

        guard.begin("later test")
        released.set()
        left_running.join(timeout=10)
        assert guard.end("later test") == 0

    assert refusals == ["refused"]
    unanswered = guard.unanswered()
    assert unanswered == {"earlier test": 1}
    assert "1 in a thread started by earlier test, after that test had finished" in (
        refused_unanswered(unanswered)
    )


def test_a_thread_no_test_started_is_charged_to_no_test() -> None:
    guard = NameLookupGuard()
    resolver, _ = _constructed_resolver()
    released = threading.Event()
    refusals: list[str] = []

    def look_up_once_released() -> None:
        released.wait(timeout=10)
        resolver.getaddrinfo(ELSEWHERE, 443)

    with pytest.MonkeyPatch.context() as patch:
        guard.install(patch, modules=(resolver,))
        between_tests = threading.Thread(target=_refusals_in(look_up_once_released, refusals))
        between_tests.start()

        guard.begin("this test")
        released.set()
        between_tests.join(timeout=10)
        assert guard.end("this test") == 0

    assert refusals == ["refused"]
    unanswered = guard.unanswered()
    assert unanswered == {None: 1}
    assert "1 outside any test" in refused_unanswered(unanswered)


async def test_a_lookup_the_event_loop_makes_in_its_own_thread_is_charged_to_the_test() -> None:
    guard = NameLookupGuard()
    with pytest.MonkeyPatch.context() as patch:
        # Over the session's own guard, which then never sees this refusal.
        guard.install(patch)
        guard.begin("this test")

        with pytest.raises(socket.gaierror):
            await asyncio.get_running_loop().getaddrinfo(ELSEWHERE, 443)

        assert guard.end("this test") == 1
    assert guard.unanswered() == {}


def test_each_resolver_and_a_connection_by_name_refuse_names_beyond_this_computer(
    request: pytest.FixtureRequest, names_resolve_only_locally: NameLookupGuard
) -> None:
    lookups: list[Callable[[], object]] = [
        lambda: socket.getaddrinfo(ELSEWHERE, 443),
        lambda: _LOOKUP_KEPT_AT_IMPORT(ELSEWHERE, 443),
        lambda: socket.gethostbyname(ELSEWHERE),
        lambda: socket.gethostbyname_ex(ELSEWHERE),
        lambda: socket.gethostbyaddr("192.0.2.10"),
        lambda: socket.getnameinfo(("192.0.2.10", 443), 0),
        lambda: socket.create_connection((ELSEWHERE, 443), timeout=1),
    ]
    for lookup in lookups:
        with pytest.raises(OSError) as refused:
            lookup()
        assert refused.value.strerror == REFUSED

    # This test settles its own refusals rather than failing for them.
    test = request.node.nodeid
    assert names_resolve_only_locally.end(test) == len(lookups)
    names_resolve_only_locally.begin(test)


def _listening_on_localhost() -> socket.socket:
    for family, _, _, _, address in socket.getaddrinfo("localhost", 0, type=socket.SOCK_STREAM):
        try:
            return socket.create_server(address[:2], family=family)
        except OSError:
            continue
    raise AssertionError("localhost names no address this computer can listen on")


def test_a_server_on_this_computer_answers_by_name_and_by_address() -> None:
    with _listening_on_localhost() as server:
        server.settimeout(5)
        address, port = server.getsockname()[:2]
        for host in ("localhost", address):
            with socket.create_connection((host, port), timeout=5) as connection:
                accepted, _ = server.accept()
                with accepted:
                    connection.sendall(b"ping")
                    assert accepted.recv(4) == b"ping"


async def test_a_server_on_this_computer_answers_the_event_loop_by_name() -> None:
    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(await reader.readexactly(4))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(answer, host="localhost", port=0)
    port = server.sockets[0].getsockname()[1]
    async with server, asyncio.timeout(10):
        reader, writer = await asyncio.open_connection("localhost", port)
        writer.write(b"ping")
        await writer.drain()
        assert await reader.readexactly(4) == b"ping"
        writer.close()
        await writer.wait_closed()


def test_a_test_that_was_refused_is_told_how_many_times_and_nothing_more() -> None:
    assert refused_during_test(1).startswith(
        "This test tried to reach another computer: 1 lookup of a name beyond this one was"
    )
    assert "2 lookups of names beyond this one were refused" in refused_during_test(2)
