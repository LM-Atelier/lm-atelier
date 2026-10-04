"""A linked page is read only while the installation allows it, and only from a public address."""

from __future__ import annotations

import asyncio
import socket

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_api import _claimed_text_job

from local_lm.db import SessionLocal
from local_lm.models import Job
from local_lm.web_lookup import LookupRequest

PAGE = "https://page.example.test/notes"


@pytest.mark.parametrize(
    ("web_access", "refused", "lookups"),
    [(False, "network-refused", []), (True, "web-host-unresolved", ["page.example.test"])],
    ids=["switched-off", "allowed"],
)
async def test_a_linked_page_is_never_looked_up_once_web_access_is_off(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    web_access: bool,
    refused: str,
    lookups: list[str],
) -> None:
    orch = app.state.services.orchestrator
    monkeypatch.setattr(type(orch), "start", lambda self, *args: None)
    _chat_id, job_id, claim = await _claimed_text_job(client, monkeypatch)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.run_id is not None
        run_id = job.run_id
    looked_up: list[str] = []

    def lookup(host: object, *args: object, **kwargs: object) -> object:
        looked_up.append(host.decode() if isinstance(host, bytes) else str(host))
        raise OSError("this test looks nothing up")

    async def choose(*args: object, **kwargs: object) -> LookupRequest:
        return LookupRequest(url=PAGE, reason="the person linked it")

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    monkeypatch.setitem(orch._read_linked_page.__globals__, "choose_from_conversation", choose)
    # Switched after the turn's own check, as when it is turned off mid-turn.
    monkeypatch.setattr(orch.engines.settings, "web_access_enabled", web_access)

    provenance = await orch._read_linked_page(
        [{"role": "user", "content": f"Read {PAGE}"}], run_id, job_id, claim
    )

    assert provenance == {"url": PAGE, "reason": "the person linked it", "refused": refused}
    assert looked_up == lookups


async def test_a_linked_page_whose_host_turns_private_after_its_check_is_never_connected_to(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    orch = app.state.services.orchestrator
    monkeypatch.setattr(type(orch), "start", lambda self, *args: None)
    _chat_id, job_id, claim = await _claimed_text_job(client, monkeypatch)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.run_id is not None
        run_id = job.run_id
    connections: list[object] = []

    async def record(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connections.append(writer.get_extra_info("peername"))
        writer.close()

    # A service on this machine, which the name will answer with once checked.
    listening = await asyncio.start_server(record, "127.0.0.1", 0)
    page = f"https://page.example.test:{listening.sockets[0].getsockname()[1]}/notes"
    looked_up: list[str] = []

    def lookup(host: object, *args: object, **kwargs: object) -> object:
        looked_up.append(host.decode() if isinstance(host, bytes) else str(host))
        address = "93.184.216.34" if len(looked_up) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

    async def choose(*args: object, **kwargs: object) -> LookupRequest:
        return LookupRequest(url=page, reason="the person linked it")

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    monkeypatch.setitem(orch._read_linked_page.__globals__, "choose_from_conversation", choose)
    monkeypatch.setattr(orch.engines.settings, "web_access_enabled", True)
    try:
        provenance = await orch._read_linked_page(
            [{"role": "user", "content": f"Read {page}"}], run_id, job_id, claim
        )
    finally:
        # Not wait_closed(): on Windows a connection the other side closed
        # first can fail to report that it was lost, and that waits forever.
        listening.close()

    refused = {"url": page, "reason": "the person linked it", "refused": "network-address-refused"}
    assert provenance == refused
    assert looked_up == ["page.example.test", "page.example.test"]
    assert connections == []
