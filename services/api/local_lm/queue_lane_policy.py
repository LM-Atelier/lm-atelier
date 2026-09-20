"""Durable dispatch policies for logical lanes, independent of resource groups."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Annotated, Literal, cast

from pydantic import Field, TypeAdapter, ValidationError
from sqlalchemy import and_, exists, func, literal, or_, select, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .domain import JobKind, JobStatus
from .models import GenerationQueuePolicy, GenerationQueueReceipt, Job
from .schemas import QueueControlCommand

Lane = Literal["generation", "transfer"]
Action = Literal["pause_after_current", "resume"]
LANES: tuple[Lane, ...] = ("generation", "transfer")
GENERATION_KINDS = (
    JobKind.CHAT.value,
    JobKind.IMAGE.value,
    JobKind.VIDEO.value,
    JobKind.EDIT_VERIFY.value,
)
TRANSFER_KINDS = (JobKind.DOWNLOAD.value, JobKind.EXPORT.value)
_KINDS: dict[Lane, tuple[str, ...]] = {
    "generation": GENERATION_KINDS,
    "transfer": TRANSFER_KINDS,
}
_MAX_REVISION = 9_223_372_036_854_775_807


class QueueLaneConflict(Exception):
    """The lane or command no longer matches its observed durable state."""


@dataclass(frozen=True)
class LaneDispatch:
    state: str
    revision: int
    valid: bool
    lane: Lane = "generation"

    @property
    def open(self) -> bool:
        return self.valid and self.state == "open"


@dataclass(frozen=True)
class LanePolicy:
    lane: Lane
    dispatch_state: Literal["open", "draining", "paused"]
    revision: Annotated[int, Field(ge=0)]
    running_jobs: Annotated[int, Field(ge=0)]
    allowed_actions: list[Action]


_POLICY = TypeAdapter(LanePolicy)


def lane_for_kind(kind: str) -> Lane | None:
    return next((lane for lane in LANES if kind in _KINDS[lane]), None)


def lane_dispatch(session: Session, lane: Lane) -> LaneDispatch:
    row = session.execute(
        select(GenerationQueuePolicy.dispatch_state, GenerationQueuePolicy.revision).where(
            GenerationQueuePolicy.lane == lane
        )
    ).one_or_none()
    if row is None:
        return LaneDispatch("open", 0, True, lane)
    state, revision = row
    return LaneDispatch(
        state,
        revision,
        state in ("open", "draining", "paused")
        and isinstance(revision, int)
        and 0 <= revision <= _MAX_REVISION,
        lane,
    )


def queue_dispatches(session: Session) -> dict[Lane, LaneDispatch]:
    return {lane: lane_dispatch(session, lane) for lane in LANES}


def dispatch_allowed(kind: str, snapshots: dict[Lane, LaneDispatch]) -> bool:
    lane = lane_for_kind(kind)
    return lane is None or snapshots[lane].open


def lane_claim_predicate(snapshot: LaneDispatch) -> ColumnElement[bool]:
    """Bind a final claim to the observed open lane revision, including ABA."""
    unchanged = (
        ~exists(
            select(GenerationQueuePolicy.lane).where(GenerationQueuePolicy.lane == snapshot.lane)
        )
        if snapshot.revision == 0
        else exists(
            select(GenerationQueuePolicy.lane).where(
                GenerationQueuePolicy.lane == snapshot.lane,
                GenerationQueuePolicy.dispatch_state == "open",
                GenerationQueuePolicy.revision == snapshot.revision,
            )
        )
    )
    return or_(
        Job.kind.not_in(_KINDS[snapshot.lane]),
        and_(literal(snapshot.open), unchanged),
    )


def _claims(lane: Lane) -> ColumnElement[bool]:
    # A terminal result still drains until its execution lease releases ownership.
    return and_(Job.kind.in_(_KINDS[lane]), Job.claim_owner.is_not(None))


def reconcile_queue_lanes(session: Session, lanes: tuple[Lane, ...] = LANES) -> None:
    """Join claim release or recovery without committing the caller's transaction."""
    for lane in lanes:
        session.execute(
            update(GenerationQueuePolicy)
            .where(
                GenerationQueuePolicy.lane == lane,
                GenerationQueuePolicy.dispatch_state == "draining",
                GenerationQueuePolicy.revision < _MAX_REVISION,
                ~exists(select(Job.id).where(_claims(lane))),
            )
            .values(dispatch_state="paused", revision=GenerationQueuePolicy.revision + 1)
            .execution_options(synchronize_session=False)
        )


def read_lane_policy(session: Session, lane: Lane) -> LanePolicy:
    """Read without joining the writer queue; recovery and release reconcile."""
    snapshot = lane_dispatch(session, lane)
    if not snapshot.valid:
        raise QueueLaneConflict
    running = session.scalar(select(func.count(Job.id)).where(_claims(lane))) or 0
    state = cast(Literal["open", "draining", "paused"], snapshot.state)
    return LanePolicy(
        lane=lane,
        dispatch_state=state,
        revision=snapshot.revision,
        running_jobs=running,
        allowed_actions=["pause_after_current"] if state == "open" else ["resume"],
    )


@contextmanager
def _transaction(session: Session) -> Iterator[None]:
    if session.in_transaction():
        raise QueueLaneConflict
    try:
        session.execute(text("BEGIN IMMEDIATE"))
        yield
        session.commit()
    except OperationalError as exc:
        session.rollback()
        code = getattr(exc.orig, "sqlite_errorcode", None)
        if isinstance(code, int) and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise QueueLaneConflict from exc
        raise
    except BaseException:
        session.rollback()
        raise


def recover_queue_lanes(session: Session, lanes: tuple[Lane, ...] = LANES) -> None:
    """Clear terminal claims left by the departed process before dispatch starts."""
    kinds = tuple(kind for lane in lanes for kind in _KINDS[lane])
    session.execute(
        update(Job)
        .where(
            Job.kind.in_(kinds),
            Job.status.in_(
                (
                    JobStatus.COMPLETE.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                    JobStatus.INTERRUPTED.value,
                )
            ),
            Job.claim_owner.is_not(None),
        )
        .values(claim_owner=None, claim_expires_at=None, heartbeat_at=None)
    )
    reconcile_queue_lanes(session, lanes)


def change_lane_policy(
    session: Session, lane: Lane, action: Action, command: QueueControlCommand
) -> LanePolicy:
    with _transaction(session):
        receipt = session.get(GenerationQueueReceipt, (lane, command.idempotency_key))
        if receipt is not None:
            if receipt.action != action or receipt.expected_revision != command.expected_revision:
                raise QueueLaneConflict
            try:
                result = _POLICY.validate_python(receipt.response_json)
            except ValidationError as exc:
                raise QueueLaneConflict from exc
            if result.lane != lane:
                raise QueueLaneConflict
            return result
        reconcile_queue_lanes(session, (lane,))
        current = read_lane_policy(session, lane)
        if (
            current.revision != command.expected_revision
            or current.revision >= _MAX_REVISION
            or action not in current.allowed_actions
        ):
            raise QueueLaneConflict
        policy = session.get(GenerationQueuePolicy, lane)
        if policy is None:
            policy = GenerationQueuePolicy(lane=lane)
            session.add(policy)
        policy.dispatch_state = (
            "open" if action == "resume" else "draining" if current.running_jobs else "paused"
        )
        policy.revision = current.revision + 1
        session.flush()
        result = read_lane_policy(session, lane)
        session.add(
            GenerationQueueReceipt(
                lane=lane,
                command_key=command.idempotency_key,
                action=action,
                expected_revision=command.expected_revision,
                response_json=asdict(result),
            )
        )
    return result
