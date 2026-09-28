"""A failed connection must not take the listening socket, or a clean stop, with it.

The proactor accept loop in the standard library closes the LISTENING socket
whenever a pending accept completes with any OSError, and the error is usually
about the incoming connection instead: a peer that resets between AcceptEx
completing and its result being read produces ERROR_NETNAME_DELETED for that
connection. The process then keeps running, keeps serving what it already holds,
and can never be reached again.

These cases pin the two halves that keep that from happening: telling a
connection's error apart from the listener's, and retrying rather than
surfacing. They also pin the close: a connection whose socket cannot be shut
down is still closed and released from its server, or stopping the application
waits for it for ever. The whole-application evidence is a separate matter and
was taken by driving the real entry point; what is here is what a suite can hold.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import socket
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_lm.windows_listener import belongs_to_the_connection, listener_preserving_loop

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="the proactor loop this repairs is the Windows one"
)


def _genuine_reset() -> OSError:
    """A real ConnectionResetError from the operating system, not a constructed one.

    Constructing one and asserting the predicate accepts it would only show that
    the test and the predicate agree about a value the test chose.
    """

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname())
    server, _ = listener.accept()
    try:
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
        client.close()
        for _ in range(64):
            try:
                server.sendall(b"x" * 4096)
            except ConnectionResetError as error:
                return error
            except OSError as error:
                return error
        raise AssertionError("the peer's reset did not reach this socket")
    finally:
        server.close()
        listener.close()


def test_a_connection_error_is_told_apart_from_a_listener_error() -> None:
    assert belongs_to_the_connection(_genuine_reset()) is True

    # A listener's own failure: the address is already taken. Nothing about it
    # belongs to an incoming connection, and waiting on it would hide it.
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    held.listen(1)
    second = socket.socket()
    try:
        second.bind(held.getsockname())
        pytest.skip("this host allows the address to be bound twice")
    except OSError as error:
        assert belongs_to_the_connection(error) is False
    finally:
        second.close()
        held.close()


@windows_only
def test_the_network_name_error_the_standard_loop_dies_on_is_recognised() -> None:
    """ERROR_NETNAME_DELETED reaches us as a plain OSError with a winerror.

    Only Windows keeps the winerror an OSError is constructed with, which is
    also the only place this error occurs, so elsewhere there is nothing to
    recognise.
    """

    error = OSError(errno.EINVAL, "The specified network name is no longer available", None, 64)
    assert belongs_to_the_connection(error) is True
    assert belongs_to_the_connection(OSError(errno.EACCES, "denied")) is False


def _serving(loop: asyncio.AbstractEventLoop, accept: Any) -> tuple[bool, int]:
    """Serve on `loop` with `accept` standing in, and report what survived.

    Returns whether the listening socket was still open at the end, and the port
    it listened on. Everything is bounded by a timeout, because the failure this
    guards against is a retry that never stops, and a test that hangs reports
    nothing.
    """

    async def scenario() -> tuple[bool, int]:
        running = asyncio.get_running_loop()
        server = await running.create_server(asyncio.Protocol, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            with contextlib.suppress(TimeoutError, OSError):
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", port), timeout=5
                )
                writer.close()
                with contextlib.suppress(OSError):
                    await writer.wait_closed()
            # Let the accept loop settle whichever way it is going to settle.
            for _ in range(20):
                await asyncio.sleep(0.01)
            return server.sockets[0].fileno() != -1, port
        finally:
            server.close()
            # On the standard transport a connection whose close meets WinError
            # 10022 never detaches from the server, and wait_closed() then waits
            # for it for ever. The loop under test repairs that, and a case
            # below holds it; these cases are about the listener, so they bound
            # the wait rather than depend on that repair.
            with contextlib.suppress(OSError, TimeoutError):
                await asyncio.wait_for(server.wait_closed(), timeout=2)

    original = asyncio.IocpProactor.accept
    asyncio.IocpProactor.accept = accept  # type: ignore[method-assign]
    try:
        return loop.run_until_complete(asyncio.wait_for(scenario(), timeout=30))
    finally:
        asyncio.IocpProactor.accept = original  # type: ignore[method-assign]
        loop.close()


def _network_name_deleted() -> OSError:
    return OSError(errno.EINVAL, "The specified network name is no longer available", None, 64)


@windows_only
def test_an_accept_that_fails_for_the_connection_leaves_the_listener_serving() -> None:
    """One connection's error must cost the listener nothing."""

    attempts: list[int] = []
    original = asyncio.IocpProactor.accept

    def failing_once(self: Any, listener: Any) -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            failed: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            failed.set_exception(_network_name_deleted())
            return failed
        return original(self, listener)

    still_listening, port = _serving(listener_preserving_loop(), failing_once)

    # Both halves: the failure really happened, and it cost nothing.
    assert len(attempts) >= 2, "the injected failure has to have been retried"
    assert port
    assert still_listening, "a connection's error closed the listening socket"


@windows_only
def test_an_error_that_is_not_the_connection_is_not_waited_out() -> None:
    """Waiting is right for a peer's reset and wrong for anything else.

    Without this the repair could retry every accept failure forever, and a
    listener broken for a reason nobody reports would look like a quiet one.
    The observable difference is the listening socket: a refusal that reaches
    the standard accept loop closes it, and a refusal swallowed by a retry
    would leave it open and the accept pending for ever.
    """

    def always_denied(self: Any, listener: Any) -> Any:
        denied: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        denied.set_exception(OSError(errno.EACCES, "permission denied"))
        return denied

    still_listening, _ = _serving(listener_preserving_loop(), always_denied)

    assert not still_listening, "an error that is not a connection's was waited out"


@windows_only
def test_a_retry_that_cannot_rearm_hands_its_error_to_the_caller() -> None:
    """A re-arm that fails outright must not leave the listener waiting forever.

    The first attempt runs inside accept() and raises to the caller directly. A
    retry runs from the loop, where a raise would reach nobody, and the listener
    would stay bound with no accept armed: taking connections into its backlog
    and never answering them, which looks alive and is not.
    """

    calls: list[int] = []

    def fails_then_cannot_rearm(self: Any, listener: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            failed: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            failed.set_exception(_network_name_deleted())
            return failed
        raise OSError(errno.EMFILE, "too many open files")

    still_listening, _ = _serving(listener_preserving_loop(), fails_then_cannot_rearm)

    assert len(calls) == 2, "the re-arm has to have been attempted"
    assert not still_listening, "a re-arm that failed left the listener waiting"


@windows_only
def test_a_connection_whose_shutdown_fails_is_still_closed_and_released() -> None:
    """A failed shutdown must not leave the server waiting for the connection.

    Once the peer has gone, shutting the socket down can fail with WinError
    10022. The standard transport then skips closing the socket and releasing
    it from its server, and the server waits for it for ever, which is what
    keeps the application from stopping.
    """

    class ShutdownFails(socket.socket):
        def shutdown(self, how: int) -> None:
            raise OSError(errno.EINVAL, "An invalid argument was supplied", None, 10022)

    released: list[bool] = []
    server = SimpleNamespace(
        _attach=lambda *_: None,
        _detach=lambda *_: released.append(True),
    )
    left, right = socket.socketpair()
    connection = ShutdownFails(fileno=left.detach())
    loop = listener_preserving_loop()

    async def scenario() -> None:
        running = cast(Any, asyncio.get_running_loop())
        transport = running._make_socket_transport(connection, asyncio.Protocol(), server=server)
        transport.close()
        for _ in range(10):
            await asyncio.sleep(0)

    try:
        loop.run_until_complete(asyncio.wait_for(scenario(), timeout=10))
    finally:
        right.close()
        loop.close()

    assert connection.fileno() == -1, "the socket was left open"
    assert released == [True], "the server was never told the connection had ended"


@windows_only
def test_the_loop_the_server_is_given_still_runs_subprocesses() -> None:
    """The engines are subprocesses, so a loop that cannot spawn is no repair.

    The selector loop would fix the accept defect outright and is unusable for
    exactly this reason, which is why the repair stays on the proactor.
    """

    async def spawn() -> bytes:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "print('spawned')",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await process.communicate()
        return out.strip()

    loop = listener_preserving_loop()
    try:
        assert loop.run_until_complete(spawn()) == b"spawned"
    finally:
        loop.close()


def test_the_factory_hands_back_a_usable_loop_on_every_platform() -> None:
    loop = listener_preserving_loop()
    try:
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert loop.run_until_complete(asyncio.sleep(0)) is None
        if sys.platform == "win32":
            # Still a proactor loop, the one that can run the engines' subprocesses.
            assert isinstance(loop, asyncio.ProactorEventLoop)
    finally:
        loop.close()
    assert os.name  # the module imported and ran on this platform at all
