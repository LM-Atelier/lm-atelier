"""A request's database connection goes back to the pool before its response is sent.

An endpoint's session used to close only once its response had been sent, so
every response on its way out held a pooled connection. A burst of reads from
open windows could hold all fifteen, and a write on the event loop then waited
thirty seconds for a connection that only the waiting loop could hand back:
the whole service stopped answering until the wait gave up. Each session now
closes as soon as its endpoint has built the response.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.pool import QueuePool

from local_lm import db as database_module
from local_lm.api import get_conversation_session
from local_lm.db import get_session


def _api_routes(routes: Iterable[Any]) -> Iterator[APIRoute | APIWebSocketRoute]:
    """Every endpoint, including those of routers the application includes."""

    for route in routes:
        if isinstance(route, APIRoute | APIWebSocketRoute):
            yield route
        included = getattr(route, "original_router", None)
        if included is not None:
            yield from _api_routes(included.routes)


def _session_dependencies(dependant: Any) -> list[Any]:
    found: list[Any] = []
    for dependency in dependant.dependencies:
        if dependency.call in (get_session, get_conversation_session):
            found.append(dependency)
        found.extend(_session_dependencies(dependency))
    return found


def test_every_route_closes_its_session_before_it_responds(app: FastAPI) -> None:
    sessions = 0
    for route in _api_routes(app.routes):
        for dependency in _session_dependencies(route.dependant):
            sessions += 1
            assert dependency.scope == "function", route.path
    # Most of the API reads or writes through one of the two dependencies.
    assert sessions > 200


def _checked_out() -> int:
    pool = database_module.engine.pool
    assert isinstance(pool, QueuePool)
    return pool.checkedout()


def _watching(app: FastAPI, client: AsyncClient, seen: list[int]) -> AsyncClient:
    """A client whose responses note how many pooled connections are out as each one starts."""

    async def watched(scope: Any, receive: Any, send: Any) -> None:
        async def watch(message: Any) -> None:
            if message["type"] == "http.response.start":
                seen.append(_checked_out())
            await send(message)

        await app(scope, receive, watch)

    return AsyncClient(
        transport=ASGITransport(app=watched),
        base_url="http://testserver",
        cookies=client.cookies,
        headers=client.headers,
    )


async def _chat(client: AsyncClient) -> str:
    created = await client.post("/api/chats", json={"title": "Pooled connections"})
    assert created.status_code == 201
    return str(created.json()["id"])


@pytest.mark.parametrize(
    "path",
    [
        "/api/chats/{chat}/context",
        "/api/jobs/activity?active_limit=100",
        "/api/projects?limit=5",
        "/api/workflow-use-case-presets",
    ],
)
async def test_a_read_hands_its_connection_back_before_its_response_starts(
    app: FastAPI, client: AsyncClient, path: str
) -> None:
    chat = await _chat(client)
    seen: list[int] = []
    async with _watching(app, client, seen) as watching:
        response = await watching.get(path.format(chat=chat))

    assert response.status_code == 200, response.text
    assert seen == [0]


async def test_a_burst_of_reads_never_holds_more_than_one_connection(
    app: FastAPI, client: AsyncClient
) -> None:
    """Twelve windows refreshing at once, as the event stream makes them do."""

    chat = await _chat(client)
    engine = database_module.engine
    most = 0

    def checked_out(*_args: Any) -> None:
        nonlocal most
        most = max(most, _checked_out())

    event.listen(engine, "checkout", checked_out)
    try:
        async with _watching(app, client, []) as watching:
            responses = await asyncio.wait_for(
                asyncio.gather(
                    *(watching.get(f"/api/chats/{chat}/context") for _ in range(6)),
                    *(watching.get("/api/jobs/activity?active_limit=100") for _ in range(6)),
                ),
                timeout=60,
            )
    finally:
        event.remove(engine, "checkout", checked_out)

    assert [response.status_code for response in responses] == [200] * 12
    assert most == 1
