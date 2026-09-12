"""Show durable search consent and bounded history without granting authority."""

from __future__ import annotations

from typing import get_args

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Job, Run, WebSearchProposal
from .schemas import WebSearchOut, WebSearchResultOut
from .web_search import (
    CrwSearchProvider,
    SearchErrorCode,
    WebSearchError,
    _result_url,
    _text,
    validate_search_query,
)

_PENDING = {"awaiting_approval", "scheduled", "approved"}
_ERROR_CODES = {
    *get_args(SearchErrorCode),
    "search_dispatch_uncertain",
    "search_permission_revoked",
    "search_provider_changed",
    "search_work_unavailable",
}


def _project(
    run: Run,
    proposal: WebSearchProposal | None,
    job_status: str | None,
) -> WebSearchOut | None:
    provenance = run.provenance_json
    history = provenance.get("web_search") if isinstance(provenance, dict) else None
    if proposal is not None:
        values = {
            "query": proposal.query,
            "provider_endpoint": proposal.provider_endpoint,
            "state": proposal.state,
            "dispatch_after": proposal.dispatch_after,
            "results": proposal.result_json,
            "error_code": proposal.error_code,
        }
    elif isinstance(history, dict):
        values = history
    else:
        return None
    query, endpoint = values.get("query"), values.get("provider_endpoint")
    if not isinstance(query, str) or not isinstance(endpoint, str):
        return None
    try:
        validate_search_query(query)
        CrwSearchProvider(endpoint)
    except WebSearchError:
        return None
    state = values.get("state")
    if not isinstance(state, str):
        return None
    active = job_status in {"running", "queued", "paused"}
    error = values.get("error_code")
    if state in _PENDING and not active:
        state = "cancelled"
        error = "search_work_unavailable"
    if state == "dispatching" and not active:
        state = "uncertain"
        error = "search_dispatch_uncertain"
    can_act = active and proposal is not None and state in _PENDING
    envelope = values.get("results")
    rows = envelope.get("results") if isinstance(envelope, dict) else None
    results: list[WebSearchResultOut] = []
    if isinstance(rows, (list, tuple)):
        for row in rows[:5]:
            if not isinstance(row, dict):
                continue
            url = _result_url(row.get("url"))
            title, _ = _text(row.get("title"), 200)
            snippet, _ = _text(row.get("snippet"), 2_000)
            if url is not None and title:
                results.append(WebSearchResultOut(url=url, title=title, snippet=snippet))
    try:
        return WebSearchOut.model_validate(
            {
                "run_id": run.id,
                "assistant_message_id": run.assistant_message_id,
                "job_id": proposal.job_id if can_act and proposal else None,
                "revision": proposal.revision if can_act and proposal else None,
                "state": state,
                "query": query,
                "provider_endpoint": endpoint,
                "dispatch_after": values.get("dispatch_after") if state == "scheduled" else None,
                "results": results,
                "result_count": len(results),
                "truncated": isinstance(envelope, dict) and envelope.get("truncated") is True,
                "error_code": error if isinstance(error, str) and error in _ERROR_CODES else None,
            }
        )
    except ValidationError:
        # Imported history is display data. A malformed historical entry may
        # not invent controls or prevent the rest of the chat from opening.
        return None


def chat_searches(session: Session, chat_id: str) -> list[WebSearchOut]:
    rows = session.execute(
        select(Run, WebSearchProposal, Job.status)
        .outerjoin(WebSearchProposal, WebSearchProposal.run_id == Run.id)
        .outerjoin(Job, Job.id == WebSearchProposal.job_id)
        .where(Run.chat_id == chat_id, Run.operation == "text")
        .order_by(Run.created_at, Run.id)
    )
    result = []
    for run, proposal, status in rows:
        item = _project(run, proposal, status)
        if item is not None:
            result.append(item)
    return result


def search_for_run(session: Session, run_id: str) -> WebSearchOut | None:
    row = session.execute(
        select(Run, WebSearchProposal, Job.status)
        .outerjoin(WebSearchProposal, WebSearchProposal.run_id == Run.id)
        .outerjoin(Job, Job.id == WebSearchProposal.job_id)
        .where(Run.id == run_id, Run.operation == "text")
    ).one_or_none()
    return _project(row[0], row[1], row[2]) if row is not None else None
