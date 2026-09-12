"""Durable generation dispatch policy, independent of physical resource groups."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import ValidationError
from sqlalchemy import and_, exists, func, literal, or_, select, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .domain import JobKind, JobStatus
from .models import GenerationQueuePolicy, GenerationQueueReceipt, Job
from .schemas import GenerationQueuePolicyOut, QueueControlCommand

GENERATION_KINDS = (
    JobKind.CHAT.value,
    JobKind.IMAGE.value,
    JobKind.VIDEO.value,
    JobKind.EDIT_VERIFY.value,
)
_MAX_REVISION = 9_223_372_036_854_775_807
Action = Literal["pause_after_current", "resume"]


class GenerationQueueConflict(Exception):
    """The lane or command no longer matches its observed durable state."""


@dataclass(frozen=True)
class GenerationDispatch:
    state: str
    revision: int
    valid: bool

    @property
    def open(self) -> bool:
        return self.valid and self.state == "open"


def generation_dispatch(session: Session) -> GenerationDispatch:
    row = session.execute(
        select(GenerationQueuePolicy.dispatch_state, GenerationQueuePolicy.revision).where(
            GenerationQueuePolicy.lane == "generation"
        )
    ).one_or_none()
    if row is None:
        return GenerationDispatch("open", 0, True)
    state, revision = row
    return GenerationDispatch(
        state,
        revision,
        state in ("open", "draining", "paused")
        and isinstance(revision, int)
        and 0 <= revision <= _MAX_REVISION,
    )


def generation_claim_predicate(snapshot: GenerationDispatch) -> ColumnElement[bool]:
    """Require the observed open revision in the final claim UPDATE, including ABA."""
    unchanged = (
        ~exists(
            select(GenerationQueuePolicy.lane).where(GenerationQueuePolicy.lane == "generation")
        )
        if snapshot.revision == 0
        else exists(
            select(GenerationQueuePolicy.lane).where(
                GenerationQueuePolicy.lane == "generation",
                GenerationQueuePolicy.dispatch_state == "open",
                GenerationQueuePolicy.revision == snapshot.revision,
            )
        )
    )
    return or_(
        Job.kind.not_in(GENERATION_KINDS),
        and_(literal(snapshot.open), unchanged),
    )


def _claims() -> ColumnElement[bool]:
    # Keep a terminal job counted until its claim is actually released.
    return and_(Job.kind.in_(GENERATION_KINDS), Job.claim_owner.is_not(None))


def reconcile_generation_queue(session: Session) -> None:
    """Join claim release/recovery's transaction; never commit the caller's work."""
    session.execute(
        update(GenerationQueuePolicy)
        .where(
            GenerationQueuePolicy.lane == "generation",
            GenerationQueuePolicy.dispatch_state == "draining",
            GenerationQueuePolicy.revision < _MAX_REVISION,
            ~exists(select(Job.id).where(_claims())),
        )
        .values(
            dispatch_state="paused",
            revision=GenerationQueuePolicy.revision + 1,
        )
        .execution_options(synchronize_session=False)
    )


def _view(session: Session) -> GenerationQueuePolicyOut:
    snapshot = generation_dispatch(session)
    if not snapshot.valid:
        raise GenerationQueueConflict
    running = session.scalar(select(func.count(Job.id)).where(_claims())) or 0
    state = cast(Literal["open", "draining", "paused"], snapshot.state)
    return GenerationQueuePolicyOut(
        lane="generation",
        dispatch_state=state,
        revision=snapshot.revision,
        running_jobs=running,
        allowed_actions=["pause_after_current"] if state == "open" else ["resume"],
    )


@contextmanager
def _transaction(session: Session) -> Iterator[None]:
    if session.in_transaction():
        raise GenerationQueueConflict
    try:
        session.execute(text("BEGIN IMMEDIATE"))
        yield
        session.commit()
    except OperationalError as exc:
        session.rollback()
        code = getattr(exc.orig, "sqlite_errorcode", None)
        if isinstance(code, int) and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise GenerationQueueConflict from exc
        raise
    except BaseException:
        session.rollback()
        raise


def read_generation_queue(session: Session) -> GenerationQueuePolicyOut:
    """Read without joining the writer queue; recovery and release reconcile."""
    return _view(session)


def recover_generation_queue(session: Session) -> None:
    """Release completed handoffs from the departed process during startup.

    Normal recovery has already interrupted running jobs. Terminal results stay
    intact; only ownership left between completion and lease release is removed.
    This runs before background work can acquire a new claim.
    """
    session.execute(
        update(Job)
        .where(
            Job.kind.in_(GENERATION_KINDS),
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
    reconcile_generation_queue(session)


def change_generation_queue(
    session: Session, action: Action, command: QueueControlCommand
) -> GenerationQueuePolicyOut:
    with _transaction(session):
        receipt = session.get(GenerationQueueReceipt, ("generation", command.idempotency_key))
        if receipt is not None:
            if receipt.action != action or receipt.expected_revision != command.expected_revision:
                raise GenerationQueueConflict
            try:
                return GenerationQueuePolicyOut.model_validate(receipt.response_json)
            except ValidationError as exc:
                raise GenerationQueueConflict from exc
        reconcile_generation_queue(session)
        current = _view(session)
        if (
            current.revision != command.expected_revision
            or current.revision >= _MAX_REVISION
            or action not in current.allowed_actions
        ):
            raise GenerationQueueConflict
        policy = session.get(GenerationQueuePolicy, "generation")
        if policy is None:
            policy = GenerationQueuePolicy(lane="generation")
            session.add(policy)
        policy.dispatch_state = (
            "open" if action == "resume" else "draining" if current.running_jobs else "paused"
        )
        policy.revision = current.revision + 1
        session.flush()
        result = _view(session)
        session.add(
            GenerationQueueReceipt(
                lane="generation",
                command_key=command.idempotency_key,
                action=action,
                expected_revision=command.expected_revision,
                response_json=result.model_dump(mode="json"),
            )
        )
    return result
