from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class ResourceScheduler:
    """Small-machine resource leases with deterministic release on cancellation."""

    def __init__(self) -> None:
        self._locks: defaultdict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(1))

    @asynccontextmanager
    async def lease(self, device_id: str = "primary") -> AsyncIterator[None]:
        lock = self._locks[device_id]
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()
