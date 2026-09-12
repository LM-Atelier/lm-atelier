"""Durable approval and once-only dispatch for one exact proposed search.

Every transition uses a fresh SQLite writer transaction. No transaction spans
provider I/O or a wait for a person. Application entry points own notification,
execution resumption, and the separate tool-free answering pass.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, get_args

from sqlalchemy import and_, or_, select, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from .domain import utcnow
from .models import Chat, Job, Run, WebSearchProposal, WorkPlan, WorkStep
from .scheduler import JobClaim
from .web_access import may_search, must_confirm_each_query
from .web_search import (
    CrwSearchProvider,
    SearchErrorCode,
    SearchResults,
    WebSearchError,
    search_results_from_record,
    validate_search_query,
)
from .work_plans import plan_status_summary, refresh_plan_status

SearchDecision = Literal["approve", "decline", "cancel"]
MAX_PROPOSAL_REVISION = 2**63 - 1
AUTOMATIC_SEARCH_DELAY_SECONDS = 5


class SearchConsentConflict(Exception):
    def __init__(self) -> None:
        super().__init__("search_consent_conflict")


@dataclass(frozen=True)
class SearchProposalView:
    job_id: str
    run_id: str
    revision: int
    state: str
    query: str = field(repr=False)
    provider_endpoint: str = field(repr=False)
    provider_revision: str = field(repr=False)
    dispatch_after: datetime | None = None


def _view(proposal: WebSearchProposal) -> SearchProposalView:
    deadline = proposal.dispatch_after
    if deadline is not None and deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return SearchProposalView(
        proposal.job_id,
        proposal.run_id,
        proposal.revision,
        proposal.state,
        proposal.query,
        proposal.provider_endpoint,
        proposal.provider_revision,
        deadline,
    )


@contextmanager
def _writer(session: Session) -> Iterator[None]:
    if session.in_transaction():
        raise SearchConsentConflict
    try:
        session.execute(text("BEGIN IMMEDIATE"))
        yield
        session.commit()
    except OperationalError as exc:
        session.rollback()
        code = getattr(exc.orig, "sqlite_errorcode", None)
        if isinstance(code, int) and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise SearchConsentConflict from None
        raise
    except BaseException:
        session.rollback()
        raise


def _owner(session: Session, job_id: str) -> tuple[Job, Run, Chat]:
    row = session.execute(
        select(Job, Run, Chat)
        .join(Run, Job.run_id == Run.id)
        .join(Chat, Run.chat_id == Chat.id)
        .where(
            Job.id == job_id,
            Job.kind == "chat",
            Job.status.in_(("running", "queued", "paused")),
            Run.operation == "text",
            Run.status.in_(("pending", "queued", "running")),
            Chat.scope == "standard",
        )
    ).one_or_none()
    if row is None:
        raise SearchConsentConflict
    return row[0], row[1], row[2]


def _assert_claim(session: Session, job_id: str, claim: JobClaim) -> None:
    owned = session.scalar(
        update(Job)
        .where(
            Job.id == job_id,
            Job.status == "running",
            Job.claim_owner == claim.token,
            Job.attempt == claim.attempt,
            Job.claim_expires_at > utcnow(),
        )
        .values(claim_owner=claim.token)
        .returning(Job.id)
    )
    if owned is None:
        raise SearchConsentConflict


def _permission(chat: Chat, installation_enabled: bool) -> None:
    if not may_search(
        installation_enabled=installation_enabled,
        chat_settings=chat.web_settings_json,
    ):
        raise SearchConsentConflict


def _work_status(session: Session, run: Run, status: str) -> None:
    if run.work_step_id:
        step = session.get(WorkStep, run.work_step_id)
        if step:
            step.status = status
    if run.work_plan_id:
        plan = session.get(WorkPlan, run.work_plan_id)
        if plan:
            session.flush()
            refresh_plan_status(session, plan.id)
            plan.summary_json = {
                **plan.summary_json,
                "status_counts": plan_status_summary(session, plan.id),
            }


def _proposal(session: Session, job: Job, run: Run, revision: int) -> WebSearchProposal:
    if type(revision) is not int or not 1 <= revision <= MAX_PROPOSAL_REVISION:
        raise SearchConsentConflict
    proposal = session.scalar(
        select(WebSearchProposal).where(
            WebSearchProposal.job_id == job.id,
            WebSearchProposal.run_id == run.id,
            WebSearchProposal.revision == revision,
        )
    )
    if proposal is None:
        raise SearchConsentConflict
    return proposal


def pause_for_search(
    session: Session,
    job_id: str,
    claim: JobClaim,
    *,
    query: str,
    provider_endpoint: str,
    provider_revision: str,
    installation_enabled: bool,
) -> SearchProposalView:
    validate_search_query(query)
    CrwSearchProvider(provider_endpoint)
    if not provider_revision or len(provider_revision) > 80 or not provider_revision.isascii():
        raise SearchConsentConflict
    with _writer(session):
        _assert_claim(session, job_id, claim)
        job, run, chat = _owner(session, job_id)
        _permission(chat, installation_enabled)
        if session.scalar(select(WebSearchProposal.id).where(WebSearchProposal.run_id == run.id)):
            raise SearchConsentConflict
        automatic = not must_confirm_each_query(chat.web_settings_json)
        proposal = WebSearchProposal(
            job_id=job.id,
            run_id=run.id,
            revision=1,
            state="scheduled" if automatic else "awaiting_approval",
            query=query,
            provider_endpoint=provider_endpoint,
            provider_revision=provider_revision,
            dispatch_after=utcnow() + timedelta(seconds=AUTOMATIC_SEARCH_DELAY_SECONDS)
            if automatic
            else None,
        )
        session.add(proposal)
        job.status = "paused"
        job.phase = "Search starts shortly" if automatic else "Waiting for search approval"
        run.status = "queued"
        _work_status(session, run, "paused")
        # The execution scope releases its own claim and compute slot on return.
        # Clearing the token here would prevent that scope's normal release.
        _record_search(run, proposal)
        result = _view(proposal)
    return result


def decide_search(
    session: Session,
    job_id: str,
    revision: int,
    action: SearchDecision,
) -> SearchProposalView:
    targets = {"approve": "approved", "decline": "declined", "cancel": "cancelled"}
    if action not in targets:
        raise SearchConsentConflict
    with _writer(session):
        job, run, _ = _owner(session, job_id)
        proposal = _proposal(session, job, run, revision)
        target = targets[action]
        if proposal.state == target:
            return _view(proposal)
        allowed = (
            ("awaiting_approval", "approved", "scheduled")
            if action == "cancel"
            else ("awaiting_approval",)
        )
        if proposal.state not in allowed:
            raise SearchConsentConflict
        proposal.state = target
        proposal.approved_automatically = False
        proposal.dispatch_after = None
        if job.status == "paused":
            job.status = "queued"
            job.phase = "Queued"
            run.status = "queued"
            _work_status(session, run, "queued")
        _record_search(run, proposal)
        result = _view(proposal)
    return result


def claim_search_dispatch(
    session: Session,
    job_id: str,
    revision: int,
    claim: JobClaim,
    *,
    provider_endpoint: str,
    provider_revision: str,
    installation_enabled: bool,
) -> SearchProposalView:
    """Commit the one outbound attempt before any provider I/O can begin.

    The caller uses the returned immutable query and endpoint, not mutable
    settings read after this transition. A failed/uncertain request never rolls
    this operation back to approved.
    """
    with _writer(session):
        _assert_claim(session, job_id, claim)
        job, run, chat = _owner(session, job_id)
        _permission(chat, installation_enabled)
        proposal = _proposal(session, job, run, revision)
        if (
            proposal.state != "approved"
            or (proposal.approved_automatically and must_confirm_each_query(chat.web_settings_json))
            or proposal.provider_endpoint != provider_endpoint
            or proposal.provider_revision != provider_revision
        ):
            raise SearchConsentConflict
        result = _begin_search_dispatch(proposal, job, run, claim)
    return result


def _begin_search_dispatch(
    proposal: WebSearchProposal,
    job: Job,
    run: Run,
    claim: JobClaim,
) -> SearchProposalView:
    proposal.state = "dispatching"
    proposal.dispatch_owner = claim.token
    proposal.dispatch_attempt = claim.attempt
    proposal.error_code = None
    job.phase = "Searching the web"
    _record_search(run, proposal)
    return _view(proposal)


def prepare_search_dispatch(
    session: Session,
    job_id: str,
    revision: int,
    claim: JobClaim,
    *,
    provider_endpoint: str | None,
    provider_revision: str | None,
    installation_enabled: bool,
) -> SearchProposalView:
    """Reconcile current permission and commit one outbound attempt together.

    A new destination or account requires a fresh visible approval. Lost
    permission or an unavailable provider lets this owned turn answer locally.
    The returned dispatching state is available exactly once; a repeated call
    cannot authorize a second request.
    """
    with _writer(session):
        _assert_claim(session, job_id, claim)
        job, run, chat = _owner(session, job_id)
        proposal = _proposal(session, job, run, revision)
        if proposal.state in ("declined", "cancelled"):
            return _view(proposal)
        if proposal.state != "approved":
            raise SearchConsentConflict
        provider_valid = bool(
            provider_endpoint
            and provider_revision
            and len(provider_revision) <= 80
            and provider_revision.isascii()
        )
        if provider_valid:
            try:
                CrwSearchProvider(provider_endpoint or "")
            except WebSearchError:
                provider_valid = False
        changed = (
            proposal.provider_endpoint != provider_endpoint
            or proposal.provider_revision != provider_revision
        )
        needs_confirmation = proposal.approved_automatically and must_confirm_each_query(
            chat.web_settings_json
        )
        if not may_search(
            installation_enabled=installation_enabled,
            chat_settings=chat.web_settings_json,
        ):
            proposal.state = "cancelled"
            proposal.error_code = "search_permission_revoked"
        elif not provider_valid:
            proposal.state = "cancelled"
            proposal.error_code = "search_provider_invalid"
        elif changed or needs_confirmation:
            if proposal.revision >= MAX_PROPOSAL_REVISION:
                proposal.state = "cancelled"
                proposal.error_code = "search_provider_changed"
            else:
                proposal.revision += 1
                proposal.provider_endpoint = provider_endpoint or ""
                proposal.provider_revision = provider_revision or ""
                proposal.state = "awaiting_approval"
                proposal.error_code = "search_provider_changed" if changed else None
                job.status = "paused"
                job.phase = "Waiting for search approval"
                run.status = "queued"
                _work_status(session, run, "paused")
        else:
            return _begin_search_dispatch(proposal, job, run, claim)
        proposal.approved_automatically = False
        proposal.dispatch_after = None
        _record_search(run, proposal)
        return _view(proposal)


def cancel_pending_search(session: Session, run: Run) -> None:
    """Revoke consent in the caller's terminal-job or retry transaction."""
    # The conditional write serializes with dispatch. An already dispatched
    # request cannot be reported as never sent, even when its job is stopped.
    proposals = session.scalars(
        update(WebSearchProposal)
        .where(
            WebSearchProposal.run_id == run.id,
            WebSearchProposal.state.in_(("awaiting_approval", "approved", "scheduled")),
        )
        .values(
            state="cancelled",
            approved_automatically=False,
            dispatch_after=None,
            error_code="search_work_unavailable",
        )
        .returning(WebSearchProposal),
        execution_options={"populate_existing": True},
    ).all()
    for proposal in proposals:
        _record_search(run, proposal)


def recover_search_dispatches(session: Session) -> int:
    """Run only at startup, before this process starts any execution tasks."""
    with _writer(session):
        interrupted = list(
            session.scalars(
                select(WebSearchProposal).where(WebSearchProposal.state == "dispatching")
            )
        )
        for proposal in interrupted:
            proposal.state = "uncertain"
            proposal.error_code = "search_dispatch_uncertain"
            run = session.get(Run, proposal.run_id)
            if run is not None:
                _record_search(run, proposal)
        # Time while the process was absent does not consume the person's
        # visible cancellation window after the application returns.
        scheduled = session.scalars(
            select(WebSearchProposal)
            .join(Job, WebSearchProposal.job_id == Job.id)
            .where(WebSearchProposal.state == "scheduled", Job.status == "paused")
        ).all()
        for proposal in scheduled:
            proposal.dispatch_after = utcnow() + timedelta(seconds=AUTOMATIC_SEARCH_DELAY_SECONDS)
            run = session.get(Run, proposal.run_id)
            if run is not None:
                _record_search(run, proposal)
        waiting = select(WebSearchProposal.job_id).where(
            WebSearchProposal.state.in_(("awaiting_approval", "scheduled"))
        )
        session.execute(
            update(Job)
            .where(Job.status == "paused", Job.id.in_(waiting))
            .values(claim_owner=None, claim_expires_at=None, heartbeat_at=None)
        )
    return len(interrupted)


def finish_search(
    session: Session,
    job_id: str,
    revision: int,
    claim: JobClaim,
    *,
    results: SearchResults | None = None,
    error_code: SearchErrorCode | None = None,
) -> SearchProposalView:
    """Store a bounded transport result only for the dispatching execution."""
    if (results is None) == (error_code is None):
        raise SearchConsentConflict
    if error_code is not None and error_code not in get_args(SearchErrorCode):
        raise SearchConsentConflict
    with _writer(session):
        _assert_claim(session, job_id, claim)
        job, run, _ = _owner(session, job_id)
        proposal = _proposal(session, job, run, revision)
        if (
            proposal.state != "dispatching"
            or proposal.dispatch_owner != claim.token
            or proposal.dispatch_attempt != claim.attempt
        ):
            raise SearchConsentConflict
        proposal.state = "complete" if results is not None else "failed"
        proposal.result_json = asdict(results) if results is not None else {}
        proposal.error_code = error_code
        _record_search(run, proposal)
        result = _view(proposal)
    return result


def resumable_search_run(session: Session, job_id: str) -> str | None:
    return session.scalar(
        select(Run.id)
        .join(Job, Job.run_id == Run.id)
        .join(WebSearchProposal, WebSearchProposal.job_id == Job.id)
        .where(
            Job.id == job_id,
            WebSearchProposal.run_id == Run.id,
            or_(
                and_(
                    Job.status == "queued",
                    WebSearchProposal.state.in_(("approved", "declined", "cancelled")),
                ),
                and_(Job.status == "paused", WebSearchProposal.state == "scheduled"),
            ),
            Run.status == "queued",
        )
    )


def replace_search_proposal(
    session: Session,
    job_id: str,
    revision: int,
    *,
    query: str,
    provider_endpoint: str,
    provider_revision: str,
    installation_enabled: bool,
) -> SearchProposalView:
    """Editing a still-pending proposal invalidates its earlier approval UI."""
    validate_search_query(query)
    CrwSearchProvider(provider_endpoint)
    if not provider_revision or len(provider_revision) > 80 or not provider_revision.isascii():
        raise SearchConsentConflict
    with _writer(session):
        job, run, chat = _owner(session, job_id)
        _permission(chat, installation_enabled)
        proposal = _proposal(session, job, run, revision)
        if (
            job.status != "paused"
            or proposal.state != "awaiting_approval"
            or revision >= MAX_PROPOSAL_REVISION
        ):
            raise SearchConsentConflict
        proposal.query = query
        proposal.provider_endpoint = provider_endpoint
        proposal.provider_revision = provider_revision
        proposal.revision += 1
        proposal.approved_automatically = False
        _record_search(run, proposal)
        result = _view(proposal)
    return result


def pending_search(session: Session, job_id: str) -> SearchProposalView | None:
    try:
        job, run, _ = _owner(session, job_id)
    except SearchConsentConflict:
        return None
    proposal = session.scalar(
        select(WebSearchProposal).where(
            WebSearchProposal.job_id == job.id,
            WebSearchProposal.run_id == run.id,
        )
    )
    return _view(proposal) if proposal is not None else None


def scheduled_search_jobs(session: Session) -> list[str]:
    return list(
        session.scalars(
            select(Job.id)
            .join(WebSearchProposal, WebSearchProposal.job_id == Job.id)
            .where(Job.status == "paused", WebSearchProposal.state == "scheduled")
        )
    )


def approve_scheduled_search(
    session: Session,
    job_id: str,
    revision: int,
    *,
    provider_endpoint: str | None,
    provider_revision: str | None,
    installation_enabled: bool,
) -> SearchProposalView:
    """Recheck automatic permission after the persisted cancellation window."""
    with _writer(session):
        job, run, chat = _owner(session, job_id)
        proposal = _proposal(session, job, run, revision)
        if proposal.state != "scheduled" or job.status != "paused":
            raise SearchConsentConflict
        deadline = _view(proposal).dispatch_after
        if deadline is None:
            raise SearchConsentConflict
        if utcnow() < deadline:
            return _view(proposal)
        if not may_search(
            installation_enabled=installation_enabled,
            chat_settings=chat.web_settings_json,
        ):
            proposal.state = "cancelled"
            proposal.error_code = "search_permission_revoked"
        elif must_confirm_each_query(chat.web_settings_json):
            proposal.state = "awaiting_approval"
            proposal.error_code = None
        elif (
            proposal.provider_endpoint != provider_endpoint
            or proposal.provider_revision != provider_revision
        ):
            proposal.state = "awaiting_approval"
            proposal.error_code = "search_provider_changed"
        else:
            proposal.state = "approved"
            proposal.error_code = None
        proposal.dispatch_after = None
        proposal.approved_automatically = proposal.state == "approved"
        if proposal.state in ("approved", "cancelled"):
            job.status = "queued"
            job.phase = "Queued"
            run.status = "queued"
            _work_status(session, run, "queued")
        else:
            job.phase = "Waiting for search approval"
        _record_search(run, proposal)
        result = _view(proposal)
    return result


def _record_search(run: Run, proposal: WebSearchProposal) -> None:
    # This is history for the person reading the response, never dispatch
    # authority. The separate proposal row owns exact approval and state.
    results = proposal.result_json or {}
    rows = results.get("results", [])
    deadline = _view(proposal).dispatch_after
    run.provenance_json = {
        **(run.provenance_json or {}),
        "web_search": {
            "query": proposal.query,
            "provider": "CRW",
            "provider_endpoint": proposal.provider_endpoint,
            "state": proposal.state,
            "dispatch_after": deadline.isoformat() if deadline else None,
            "result_count": len(rows) if isinstance(rows, (list, tuple)) else 0,
            "results": results,
            "error_code": proposal.error_code,
        },
    }


def retain_search_wait(
    session: Session,
    job_id: str,
    revision: int,
    claim: JobClaim,
) -> SearchProposalView:
    with _writer(session):
        _assert_claim(session, job_id, claim)
        job, run, _ = _owner(session, job_id)
        proposal = _proposal(session, job, run, revision)
        if proposal.state not in ("awaiting_approval", "scheduled"):
            raise SearchConsentConflict
        job.status = "paused"
        job.phase = (
            "Search starts shortly"
            if proposal.state == "scheduled"
            else "Waiting for search approval"
        )
        run.status = "queued"
        _work_status(session, run, "paused")
        return _view(proposal)


def search_result_for_answer(
    session: Session,
    job_id: str,
    revision: int,
    claim: JobClaim,
) -> SearchResults | None:
    """Read completed evidence; an unfinished outbound attempt is never replayed."""
    with _writer(session):
        _assert_claim(session, job_id, claim)
        job, run, _ = _owner(session, job_id)
        proposal = _proposal(session, job, run, revision)
        if proposal.state == "dispatching":
            proposal.state = "uncertain"
            proposal.error_code = "search_dispatch_uncertain"
            _record_search(run, proposal)
        if proposal.state != "complete":
            return None
        try:
            return search_results_from_record(proposal.result_json)
        except WebSearchError:
            proposal.state = "failed"
            proposal.error_code = "search_response_invalid"
            _record_search(run, proposal)
            return None
