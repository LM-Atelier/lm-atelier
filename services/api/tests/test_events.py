from __future__ import annotations

from local_lm.events import EventBroker


async def test_event_replay_can_exceed_the_live_queue_capacity() -> None:
    broker = EventBroker(history_size=600)
    for index in range(600):
        await broker.publish("text.delta", payload={"text": str(index)})

    async with broker.subscribe(after=0) as queue:
        assert queue.qsize() == 600
        replay = [queue.get_nowait() for _ in range(600)]

    assert [event.sequence for event in replay] == list(range(1, 601))


async def test_event_subscription_keeps_replay_and_new_events_in_order() -> None:
    broker = EventBroker(history_size=600)
    for index in range(550):
        await broker.publish("text.delta", payload={"text": str(index)})

    async with broker.subscribe(after=50) as queue:
        await broker.publish("run.completed")
        replay = [queue.get_nowait() for _ in range(501)]

    assert [event.sequence for event in replay] == list(range(51, 552))
