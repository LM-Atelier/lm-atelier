from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

from .domain import utcnow
from .schemas import EventOut


@dataclass(eq=False)
class Subscription:
    queue: asyncio.Queue[EventOut]


class EventBroker:
    def __init__(self, history_size: int = 2_000) -> None:
        self._history: deque[EventOut] = deque(maxlen=history_size)
        self._subscribers: set[Subscription] = set()
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def publish(
        self, event_type: str, entity_id: str | None = None, payload: dict[str, Any] | None = None
    ) -> EventOut:
        async with self._lock:
            self._sequence += 1
            event = EventOut(
                sequence=self._sequence,
                type=event_type,
                entity_id=entity_id,
                payload=payload or {},
                created_at=utcnow(),
            )
            self._history.append(event)
            subscribers = tuple(self._subscribers)

        for subscription in subscribers:
            try:
                subscription.queue.put_nowait(event)
            except asyncio.QueueFull:
                with suppress(asyncio.QueueEmpty):
                    subscription.queue.get_nowait()
                subscription.queue.put_nowait(event)
        return event

    def since(self, sequence: int) -> list[EventOut]:
        return [event for event in self._history if event.sequence > sequence]

    @asynccontextmanager
    async def subscribe(self, after: int = 0) -> AsyncIterator[asyncio.Queue[EventOut]]:
        subscription = Subscription(queue=asyncio.Queue(maxsize=500))
        for event in self.since(after):
            subscription.queue.put_nowait(event)
        self._subscribers.add(subscription)
        try:
            yield subscription.queue
        finally:
            self._subscribers.discard(subscription)
