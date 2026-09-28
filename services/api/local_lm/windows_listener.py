"""Keep a failed connection from taking down the Windows listener or its shutdown.

On Windows the server runs on asyncio's proactor loop, whose accept loop treats
ANY OSError from a pending accept as fatal to the LISTENING socket: it reports
"Accept failed on a socket", closes that socket, and never re-arms. The error is
usually not about the listener at all. A peer that resets between AcceptEx
completing and its result being read produces ERROR_NETNAME_DELETED for the
INCOMING connection, and the listener is closed for it.

What that costs is worse than a dropped request. The process keeps running, goes
on serving the connections it already holds and goes on generating, and can
never be reached again; the first sign of it is that the application stops
answering while its window says it is busy. Any client can cause it, and nothing
about doing so has to be deliberate.

So the accept is retried when the completion's error belongs to the connection
rather than to the listener, and everything else is raised as before. The loop
below is otherwise the standard proactor loop, and it is still a PROACTOR loop:
the selector loop would fix this and cannot be used, because it does not support
subprocesses on Windows and the engines are subprocesses.

The same loop closes connections the standard one can leave half-closed. When a
connection ends, the standard transport shuts its socket down inside a finally,
and if that raises, as WinError 10022 does once the peer has already gone, the
socket is never closed and never detached from its server. The server then
counts it as open for ever, and stopping the application, which waits for every
connection to finish, never finishes. A browser that opens a connection and
closes it unused is enough.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import socket
import sys
from typing import Any

#: Completion errors that describe the incoming connection rather than the
#: listening socket. ERROR_NETNAME_DELETED is what a peer's reset produces here;
#: the others are the same event under names Python maps for us.
_CONNECTION_ERRNOS = frozenset({errno.ENOTCONN, errno.ECONNRESET, errno.ECONNABORTED})
_ERROR_NETNAME_DELETED = 64


def belongs_to_the_connection(error: OSError) -> bool:
    """Whether this accept failure is about the peer, not about the listener."""

    if isinstance(error, ConnectionResetError | ConnectionAbortedError):
        return True
    if getattr(error, "winerror", None) == _ERROR_NETNAME_DELETED:
        return True
    return error.errno in _CONNECTION_ERRNOS


if sys.platform == "win32":
    from asyncio import proactor_events

    class ListenerPreservingProactor(asyncio.IocpProactor):
        """Retry an accept whose completion failed for the connection."""

        def accept(self, listener: Any) -> Any:
            # The serving loop schedules its accept through call_soon, so this
            # always runs inside the running loop; asking the loop for itself
            # avoids reaching into the proactor's private attribute for it.
            loop = asyncio.get_running_loop()
            outer: asyncio.Future[Any] = loop.create_future()
            inner_holder: list[Any] = [None]

            def attempt() -> None:
                if outer.cancelled():
                    return
                inner = asyncio.IocpProactor.accept(self, listener)
                inner_holder[0] = inner
                inner.add_done_callback(settled)

            def retry() -> None:
                # The first attempt runs inside accept() and raises to the
                # caller. A retry runs from the loop, where a raise would reach
                # only the loop's exception handler and leave the caller waiting
                # on an accept that is never armed: a listener still bound and
                # answering nobody. So a failed re-arm is handed over instead.
                try:
                    attempt()
                except Exception as error:
                    if not outer.done():
                        outer.set_exception(error)

            def settled(inner: asyncio.Future[Any]) -> None:
                if outer.cancelled():
                    return
                try:
                    result = inner.result()
                except asyncio.CancelledError:
                    if not outer.done():
                        outer.cancel()
                    return
                except OSError as error:
                    # A listener that has already been closed has no descriptor
                    # to re-arm, so the error is the caller's answer rather than
                    # something to retry behind their back.
                    if belongs_to_the_connection(error) and listener.fileno() != -1:
                        loop.call_soon(retry)
                        return
                    if not outer.done():
                        outer.set_exception(error)
                    return
                except BaseException as error:
                    # Anything else is mirrored to the caller unchanged.
                    if not outer.done():
                        outer.set_exception(error)
                    return
                if not outer.done():
                    outer.set_result(result)

            def cancel_inner(_: asyncio.Future[Any]) -> None:
                # The serving loop cancels the accept future to stop serving, so
                # the pending overlapped accept has to be cancelled with it.
                inner = inner_holder[0]
                if inner is not None and not inner.done():
                    inner.cancel()

            outer.add_done_callback(cancel_inner)
            attempt()
            return outer

    class ShutdownTolerantTransport(proactor_events._ProactorSocketTransport):
        """Close a connection and detach it from its server even when shutdown fails.

        The standard method with one change: the socket's shutdown may fail and
        the rest still runs, so the socket is closed and the server stops
        counting it. Shutdown is only a courtesy to a peer that may already be
        gone; closing and detaching are what everything after depends on.
        """

        def _call_connection_lost(self, exc: BaseException | None) -> None:
            # These are the standard transport's own fields, which the stubs do
            # not declare; this method is the standard one with shutdown contained.
            state: Any = self
            if state._called_connection_lost:
                return
            try:
                state._protocol.connection_lost(exc)
            finally:
                sock = state._sock
                if hasattr(sock, "shutdown") and sock.fileno() != -1:
                    with contextlib.suppress(OSError):
                        sock.shutdown(socket.SHUT_RDWR)
                sock.close()
                state._sock = None
                server = state._server
                if server is not None:
                    server._detach()
                    state._server = None
                state._called_connection_lost = True

    class ListenerPreservingLoop(asyncio.ProactorEventLoop):
        """The proactor loop, making its socket transports the tolerant kind."""

        def _make_socket_transport(
            self,
            sock: Any,
            protocol: Any,
            waiter: Any = None,
            extra: Any = None,
            server: Any = None,
        ) -> ShutdownTolerantTransport:
            return ShutdownTolerantTransport(self, sock, protocol, waiter, extra, server)

    def listener_preserving_loop() -> asyncio.AbstractEventLoop:
        """The event loop factory the server is configured with on Windows."""

        return ListenerPreservingLoop(ListenerPreservingProactor())

else:

    def listener_preserving_loop() -> asyncio.AbstractEventLoop:
        """Nothing to repair away from Windows, where neither problem occurs."""

        return asyncio.new_event_loop()
