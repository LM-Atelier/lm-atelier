from __future__ import annotations

from local_lm.events import EventBroker


def test_event_broker_exposes_a_stable_process_epoch() -> None:
    first = EventBroker(epoch="process-one")
    second = EventBroker(epoch="process-two")

    assert first.epoch == "process-one"
    assert first.epoch == "process-one"
    assert second.epoch == "process-two"


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


async def test_event_subscription_signals_when_requested_history_expired() -> None:
    broker = EventBroker(history_size=3)
    for index in range(5):
        await broker.publish("text.delta", payload={"text": str(index)})

    async with broker.subscribe(after=1) as queue:
        replay = [queue.get_nowait() for _ in range(4)]

    assert replay[0].type == "events.replay_gap"
    assert replay[0].sequence == 2
    assert replay[0].payload == {
        "reason": "history_expired",
        "oldest_available_sequence": 3,
        "latest_sequence": 5,
    }
    assert [event.sequence for event in replay[1:]] == [3, 4, 5]


async def test_event_subscription_does_not_signal_for_complete_replay() -> None:
    broker = EventBroker(history_size=3)
    for index in range(3):
        await broker.publish("text.delta", payload={"text": str(index)})

    async with broker.subscribe(after=0) as queue:
        replay = [queue.get_nowait() for _ in range(3)]

    assert [event.type for event in replay] == ["text.delta"] * 3
    assert [event.sequence for event in replay] == [1, 2, 3]


async def test_slow_subscriber_receives_gap_before_latest_event() -> None:
    broker = EventBroker(history_size=1_000)

    async with broker.subscribe(after=0) as queue:
        for index in range(501):
            await broker.publish("text.delta", payload={"text": str(index)})
        replay = [queue.get_nowait(), queue.get_nowait()]

    assert replay[0].type == "events.replay_gap"
    assert replay[0].sequence == 500
    assert replay[0].payload == {
        "reason": "subscriber_overflow",
        "oldest_available_sequence": 501,
        "latest_sequence": 501,
    }
    assert replay[1].sequence == 501
    assert replay[1].payload == {"text": "500"}
