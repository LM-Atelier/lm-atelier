"""Generation policy entry points over the shared logical-lane contract."""

from __future__ import annotations

from dataclasses import asdict

from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .queue_lane_policy import GENERATION_KINDS as GENERATION_KINDS
from .queue_lane_policy import Action as Action
from .queue_lane_policy import LaneDispatch as GenerationDispatch
from .queue_lane_policy import (
    QueueLaneConflict,
    change_lane_policy,
    lane_claim_predicate,
    lane_dispatch,
    read_lane_policy,
    reconcile_queue_lanes,
    recover_queue_lanes,
)
from .schemas import GenerationQueuePolicyOut, QueueControlCommand

GenerationQueueConflict = QueueLaneConflict


def generation_dispatch(session: Session) -> GenerationDispatch:
    return lane_dispatch(session, "generation")


def generation_claim_predicate(snapshot: GenerationDispatch) -> ColumnElement[bool]:
    return lane_claim_predicate(snapshot)


def reconcile_generation_queue(session: Session) -> None:
    reconcile_queue_lanes(session, ("generation",))


def read_generation_queue(session: Session) -> GenerationQueuePolicyOut:
    return GenerationQueuePolicyOut.model_validate(asdict(read_lane_policy(session, "generation")))


def recover_generation_queue(session: Session) -> None:
    recover_queue_lanes(session, ("generation",))


def change_generation_queue(
    session: Session, action: Action, command: QueueControlCommand
) -> GenerationQueuePolicyOut:
    result = change_lane_policy(session, "generation", action, command)
    return GenerationQueuePolicyOut.model_validate(asdict(result))
