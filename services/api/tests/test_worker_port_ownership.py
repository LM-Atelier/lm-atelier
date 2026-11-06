"""A worker's port is recognised from its socket, not from the state Windows reports.

Recent Windows 11 builds report listening sockets with arbitrary TCP states:
closed, a closing state, or a value with no name, while each row's addresses and
owning process stay right. Every check that waited for LISTEN then failed. A
worker that was serving its health endpoint was refused as "served by another
process", the reclaim of a lost worker found nothing to reclaim, and a busy port
had no occupant to name.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import psutil
import pytest

from local_lm.config import Settings
from local_lm.processes import ProcessSupervisor

PORT = 51247
# Taken before any case replaces psutil.Process, so every stand-in keeps its shape.
PROCESS = psutil.Process


def _row(
    status: str, port: int = PORT, *, peer: tuple[str, int] | None = None, pid: int | None = None
) -> Any:
    return SimpleNamespace(
        status=status,
        laddr=SimpleNamespace(ip="127.0.0.1", port=port),
        raddr=SimpleNamespace(ip=peer[0], port=peer[1]) if peer else (),
        pid=pid,
    )


def _worker(rows: list[Any]) -> Mock:
    process = Mock(spec=PROCESS, net_connections=lambda kind="tcp": rows)
    process.children.return_value = []
    return process


@pytest.mark.parametrize("reported", [psutil.CONN_NONE, psutil.CONN_CLOSE, psutil.CONN_FIN_WAIT1])
def test_a_serving_worker_is_recognised_whatever_state_its_listener_reports(
    monkeypatch: pytest.MonkeyPatch, reported: str
) -> None:
    worker = _worker([_row(reported)])
    monkeypatch.setattr(psutil, "Process", lambda _pid: worker)

    assert ProcessSupervisor._listener_owned_by_worker(
        4321, f"http://127.0.0.1:{PORT}/system_stats"
    )


def test_a_connection_on_the_port_is_not_a_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    # The worker's end of a connection to its own port proves nothing about who
    # answered the health request, whatever state the row carries.
    for reported in (psutil.CONN_ESTABLISHED, psutil.CONN_NONE):
        worker = _worker([_row(reported, peer=("127.0.0.1", 60001))])
        monkeypatch.setattr(psutil, "Process", lambda _pid, worker=worker: worker)

        assert not ProcessSupervisor._listener_owned_by_worker(
            4321, f"http://127.0.0.1:{PORT}/system_stats"
        ), reported


def test_a_listener_on_another_port_does_not_count(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _worker([_row(psutil.CONN_NONE, port=PORT + 1)])
    monkeypatch.setattr(psutil, "Process", lambda _pid: worker)

    assert not ProcessSupervisor._listener_owned_by_worker(
        4321, f"http://127.0.0.1:{PORT}/system_stats"
    )


def test_a_real_listening_socket_is_recognised_as_its_owners() -> None:
    # Read from this platform's own socket table, so a platform that misreports
    # states is held to the same answer as one that does not.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        assert ProcessSupervisor._listener_owned_by_worker(
            os.getpid(), f"http://127.0.0.1:{port}/health"
        )


def test_the_reclaim_finds_a_lost_worker_whose_listener_reports_no_state() -> None:
    worker = _worker([_row(psutil.CONN_NONE)])

    assert ProcessSupervisor._listening_match(worker, "127.0.0.1", PORT) == "exact"
    # A connection still proves nothing, so it still authorises nothing.
    connected = _worker([_row(psutil.CONN_NONE, peer=("127.0.0.1", 60002))])
    assert ProcessSupervisor._listening_match(connected, "127.0.0.1", PORT) is None


def test_a_busy_port_names_its_occupant_whatever_state_it_reports(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(
        psutil,
        "net_connections",
        lambda kind="tcp": [
            _row(psutil.CONN_ESTABLISHED, peer=("127.0.0.1", 60003), pid=1),
            _row(psutil.CONN_CLOSE, pid=os.getpid()),
        ],
    )

    described = supervisor._port_listener_description("127.0.0.1", PORT)

    assert described is not None and f"(pid {os.getpid()})" in described


def test_a_busy_port_is_never_blamed_on_pid_0(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A server that has just closed connections leaves rows filed under pid 0,
    # and Windows sometimes reports one as LISTEN with its peer dropped. Ahead
    # of the real occupant it named "System Idle Process (pid 0)".
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(
        psutil,
        "net_connections",
        lambda kind="tcp": [
            _row(psutil.CONN_LISTEN, pid=0),
            _row(psutil.CONN_NONE, pid=os.getpid()),
        ],
    )

    described = supervisor._port_listener_description("127.0.0.1", PORT)

    assert described is not None and f"(pid {os.getpid()})" in described
    monkeypatch.setattr(
        psutil, "net_connections", lambda kind="tcp": [_row(psutil.CONN_LISTEN, pid=0)]
    )
    assert supervisor._port_listener_description("127.0.0.1", PORT) is None
