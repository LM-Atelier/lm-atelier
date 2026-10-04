from __future__ import annotations

import asyncio
from typing import Any, cast

from fastapi import WebSocket

from local_lm.events import EventBroker
from local_lm.main import _stream_events


class FakeWebSocket:
    def __init__(self, *, block_sends: bool = False) -> None:
        self.received: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.send_started = asyncio.Event()
        self.send_cancelled = asyncio.Event()
        self.closed: list[tuple[int, str | None]] = []
        self._block_sends = block_sends

    async def receive(self) -> dict[str, Any]:
        return await self.received.get()

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed.append((code, reason))

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.send_started.set()
        if self._block_sends:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.send_cancelled.set()
                raise
        self.sent.append(payload)


def websocket(fake: FakeWebSocket) -> WebSocket:
    return cast(WebSocket, fake)


async def wait_for_subscriber(broker: EventBroker) -> None:
    for _ in range(100):
        if broker._subscribers:
            return
        await asyncio.sleep(0)
    raise AssertionError("event stream did not subscribe")


async def test_idle_disconnect_unsubscribes_without_waiting_for_an_event() -> None:
    broker = EventBroker()
    fake = FakeWebSocket()
    stream = asyncio.create_task(_stream_events(websocket(fake), broker, after=0))
    await wait_for_subscriber(broker)

    fake.received.put_nowait({"type": "websocket.disconnect", "code": 1000})
    await asyncio.wait_for(stream, timeout=1)

    assert not broker._subscribers
    assert fake.sent == []


async def test_disconnect_cancels_an_in_flight_send_and_unsubscribes() -> None:
    broker = EventBroker()
    fake = FakeWebSocket(block_sends=True)
    stream = asyncio.create_task(_stream_events(websocket(fake), broker, after=0))
    await wait_for_subscriber(broker)

    await broker.publish("jobs.changed", "job-1")
    await asyncio.wait_for(fake.send_started.wait(), timeout=1)
    fake.received.put_nowait({"type": "websocket.disconnect", "code": 1000})
    await asyncio.wait_for(stream, timeout=1)

    assert fake.send_cancelled.is_set()
    assert fake.sent == []
    assert not broker._subscribers


async def test_disconnect_wins_when_a_replayed_event_is_already_ready() -> None:
    broker = EventBroker()
    await broker.publish("jobs.changed", "job-1")
    fake = FakeWebSocket()
    fake.received.put_nowait({"type": "websocket.disconnect", "code": 1000})

    await asyncio.wait_for(_stream_events(websocket(fake), broker, after=0), timeout=1)

    assert fake.sent == []
    assert not broker._subscribers


async def test_a_lock_closes_an_idle_stream_with_the_lock_code() -> None:
    broker = EventBroker()
    fake = FakeWebSocket()
    closing = asyncio.Event()
    stream = asyncio.create_task(_stream_events(websocket(fake), broker, after=0, closing=closing))
    await wait_for_subscriber(broker)

    closing.set()
    await asyncio.wait_for(stream, timeout=1)

    assert fake.closed == [(4423, "workspace-locked")]
    assert fake.sent == []
    assert not broker._subscribers


async def test_a_lock_cancels_an_in_flight_send_and_closes() -> None:
    broker = EventBroker()
    fake = FakeWebSocket(block_sends=True)
    closing = asyncio.Event()
    stream = asyncio.create_task(_stream_events(websocket(fake), broker, after=0, closing=closing))
    await wait_for_subscriber(broker)

    await broker.publish("jobs.changed", "job-1")
    await asyncio.wait_for(fake.send_started.wait(), timeout=1)
    closing.set()
    await asyncio.wait_for(stream, timeout=1)

    assert fake.send_cancelled.is_set()
    assert fake.sent == []
    assert fake.closed == [(4423, "workspace-locked")]
    assert not broker._subscribers


async def test_a_lock_wins_over_a_replayed_event_that_is_already_ready() -> None:
    broker = EventBroker()
    await broker.publish("jobs.changed", "job-1")
    fake = FakeWebSocket()
    closing = asyncio.Event()
    closing.set()

    await asyncio.wait_for(
        _stream_events(websocket(fake), broker, after=0, closing=closing), timeout=1
    )

    assert fake.sent == []
    assert fake.closed == [(4423, "workspace-locked")]
    assert not broker._subscribers
