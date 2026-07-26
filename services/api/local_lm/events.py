from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from .domain import utcnow
from .schemas import EventOut


@dataclass(eq=False)
class Subscription:
    queue: asyncio.Queue[EventOut]


class EventBroker:
    def __init__(self, history_size: int = 2_000, *, epoch: str | None = None) -> None:
        self._history: deque[EventOut] = deque(maxlen=history_size)
        self._subscribers: set[Subscription] = set()
        self._sequence = 0
        self._epoch = epoch or uuid.uuid4().hex
        self._lock = asyncio.Lock()

    @property
    def epoch(self) -> str:
        return self._epoch

    @property
    def sequence(self) -> int:
        return self._sequence

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
                # A slow browser must never receive a plausible-looking partial
                # sequence after authoritative state transitions were dropped.
                # Discard the stale backlog and make the loss explicit so the
                # client can reconcile from durable state before continuing.
                while True:
                    try:
                        subscription.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                subscription.queue.put_nowait(
                    self._gap_event(
                        sequence=event.sequence - 1,
                        reason="subscriber_overflow",
                        oldest_available_sequence=event.sequence,
                    )
                )
                subscription.queue.put_nowait(event)
        return event

    def since(self, sequence: int) -> list[EventOut]:
        return [event for event in self._history if event.sequence > sequence]

    async def clear(self) -> None:
        async with self._lock:
            self._history.clear()
            self._subscribers.clear()
            self._sequence = 0
            self._epoch = uuid.uuid4().hex

    @asynccontextmanager
    async def subscribe(self, after: int = 0) -> AsyncIterator[asyncio.Queue[EventOut]]:
        async with self._lock:
            replay = self.since(after)
            oldest_available = self._history[0].sequence if self._history else self._sequence + 1
            replay_gap = after < oldest_available - 1
            subscription = Subscription(
                queue=asyncio.Queue(maxsize=max(500, len(replay) + 500 + (1 if replay_gap else 0)))
            )
            if replay_gap:
                subscription.queue.put_nowait(
                    self._gap_event(
                        sequence=oldest_available - 1,
                        reason="history_expired",
                        oldest_available_sequence=oldest_available,
                    )
                )
            for event in replay:
                subscription.queue.put_nowait(event)
            self._subscribers.add(subscription)
        try:
            yield subscription.queue
        finally:
            async with self._lock:
                self._subscribers.discard(subscription)

    def _gap_event(
        self,
        *,
        sequence: int,
        reason: str,
        oldest_available_sequence: int,
    ) -> EventOut:
        return EventOut(
            sequence=max(0, sequence),
            type="events.replay_gap",
            payload={
                "reason": reason,
                "oldest_available_sequence": oldest_available_sequence,
                "latest_sequence": self._sequence,
            },
            created_at=utcnow(),
        )
