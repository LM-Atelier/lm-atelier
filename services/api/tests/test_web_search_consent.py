"""Search approval is durable and bound to the exact proposed operation."""

from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Event
from typing import Any

import pytest
from sqlalchemy import event, select

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import Chat, Job, Message, Run, WorkPlan, WorkStep
from local_lm.scheduler import JobClaim

QUERY = "Compare brass and aluminum"
ENDPOINT = "https://search.example.test"
CLAIM = JobClaim(token="constructed-search-claim", attempt=1)


def _seed(settings: Settings) -> tuple[Any, str, str, str]:
    consent = importlib.import_module("local_lm.web_search_consent")
    with SessionLocal() as session:
        chat = Chat(title="Search consent", web_settings_json={"allow_search": True})
        session.add(chat)
        session.flush()
        user = Message(chat_id=chat.id, role="user", status="complete")
        assistant = Message(chat_id=chat.id, role="assistant", status="pending")
        plan = WorkPlan(chat_id=chat.id, transcript_sequence=1, status="running")
        session.add_all([user, assistant, plan])
        session.flush()
        step = WorkStep(plan_id=plan.id, ordinal=0, operation="text", status="running")
        session.add(step)
        session.flush()
        run = Run(
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            work_plan_id=plan.id,
            work_step_id=step.id,
            status="running",
        )
        session.add(run)
        session.flush()
        step.run_id = run.id
        job = Job(
            run_id=run.id,
            work_plan_id=plan.id,
            work_step_id=step.id,
            kind="chat",
            status="running",
            attempt=CLAIM.attempt,
            claim_owner=CLAIM.token,
            claim_expires_at=utcnow() + timedelta(minutes=5),
        )
        session.add(job)
        session.commit()
        return consent, job.id, run.id, chat.id


def _pause(consent: Any, job_id: str, *, query: str = QUERY) -> Any:
    with SessionLocal() as session:
        return consent.pause_for_search(
            session,
            job_id,
            CLAIM,
            query=query,
            provider_endpoint=ENDPOINT,
            provider_revision="provider-one",
            installation_enabled=True,
        )


def _decide(consent: Any, job_id: str, revision: int, action: str) -> Any:
    with SessionLocal() as session:
        return consent.decide_search(session, job_id, revision, action)


def _resume(job_id: str) -> JobClaim:
    next_claim = JobClaim("constructed-resumed-claim", 2)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None and job.status == "queued"
        job.status = "running"
        job.claim_owner = next_claim.token
        job.attempt = next_claim.attempt
        job.claim_expires_at = utcnow() + timedelta(minutes=5)
        session.commit()
    return next_claim


def _dispatch(consent: Any, job_id: str, revision: int, claim: JobClaim, **overrides: Any) -> Any:
    values = {
        "provider_endpoint": ENDPOINT,
        "provider_revision": "provider-one",
        "installation_enabled": True,
    }
    values.update(overrides)
    with SessionLocal() as session:
        return consent.claim_search_dispatch(session, job_id, revision, claim, **values)


def test_wait_persists_exact_proposal_and_pauses_owned_work(settings: Settings) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    proposed = _pause(consent, job_id, query="  " + QUERY + "  ")
    assert proposed.query == "  " + QUERY + "  "
    assert proposed.provider_endpoint == ENDPOINT
    assert proposed.revision == 1 and proposed.state == "awaiting_approval"
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        run = session.get(Run, run_id)
        assert job is not None and run is not None
        assert job.status == "paused" and job.claim_owner == CLAIM.token
        assert run.status == "queued" and run.completed_at is None
        assert session.get(WorkStep, run.work_step_id).status == "paused"
        assert session.get(WorkPlan, run.work_plan_id).status == "paused"
        persisted = session.scalar(select(consent.WebSearchProposal))
        assert persisted.query == proposed.query
        assert persisted.provider_revision == "provider-one"


@pytest.mark.parametrize("failure", ["token", "attempt", "expired", "cancelled", "permission"])
def test_stale_or_disallowed_producer_cannot_publish_query(
    settings: Settings, failure: str
) -> None:
    consent, job_id, _, chat_id = _seed(settings)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if failure == "token":
            job.claim_owner = "another-claim"
        elif failure == "attempt":
            job.attempt += 1
        elif failure == "expired":
            job.claim_expires_at = utcnow() - timedelta(seconds=1)
        elif failure == "cancelled":
            job.status = "cancelled"
        else:
            session.get(Chat, chat_id).web_settings_json = {}
        session.commit()
    with pytest.raises(consent.SearchConsentConflict):
        _pause(consent, job_id)
    with SessionLocal() as session:
        assert session.scalar(select(consent.WebSearchProposal)) is None


def test_approval_retry_is_stable_but_stale_revision_is_refused(settings: Settings) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    first = _decide(consent, job_id, proposed.revision, "approve")
    assert first == _decide(consent, job_id, proposed.revision, "approve")
    assert first.state == "approved" and first.query == QUERY
    with pytest.raises(consent.SearchConsentConflict):
        _decide(consent, job_id, proposed.revision + 1, "approve")
    with SessionLocal() as session:
        assert session.get(Job, job_id).status == "queued"


@pytest.mark.parametrize("action", ["decline", "cancel"])
def test_refusal_requeues_answer_without_authorizing_search(
    settings: Settings, action: str
) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    refused = _decide(consent, job_id, proposed.revision, action)
    assert refused.state == ("declined" if action == "decline" else "cancelled")
    assert refused == _decide(consent, job_id, proposed.revision, action)
    claim = _resume(job_id)
    with pytest.raises(consent.SearchConsentConflict):
        _dispatch(consent, job_id, proposed.revision, claim)


@pytest.mark.parametrize("change", ["global", "chat", "endpoint", "provider_revision"])
def test_dispatch_rechecks_current_authority(settings: Settings, change: str) -> None:
    consent, job_id, _, chat_id = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    overrides: dict[str, Any] = {}
    if change == "global":
        overrides["installation_enabled"] = False
    elif change == "chat":
        with SessionLocal() as session:
            session.get(Chat, chat_id).web_settings_json = {}
            session.commit()
    elif change == "endpoint":
        overrides["provider_endpoint"] = "https://other.example.test"
    else:
        overrides["provider_revision"] = "provider-two"
    with pytest.raises(consent.SearchConsentConflict):
        _dispatch(consent, job_id, proposed.revision, claim, **overrides)
    with SessionLocal() as session:
        assert session.scalar(select(consent.WebSearchProposal.state)) == "approved"


def test_dispatch_is_once_only_and_owned_by_resumed_claim(settings: Settings) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    with pytest.raises(consent.SearchConsentConflict):
        _dispatch(consent, job_id, proposed.revision, CLAIM)
    dispatched = _dispatch(consent, job_id, proposed.revision, claim)
    assert dispatched.state == "dispatching" and dispatched.query == QUERY
    with pytest.raises(consent.SearchConsentConflict):
        _dispatch(consent, job_id, proposed.revision, claim)
    with pytest.raises(consent.SearchConsentConflict):
        _decide(consent, job_id, proposed.revision, "cancel")


def test_cancel_committed_before_dispatch_prevents_dispatch(settings: Settings) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    _decide(consent, job_id, proposed.revision, "cancel")
    with pytest.raises(consent.SearchConsentConflict):
        _dispatch(consent, job_id, proposed.revision, claim)


def test_uncertain_dispatch_is_not_reapproved_or_reissued_after_restart(settings: Settings) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    _dispatch(consent, job_id, proposed.revision, claim)
    with SessionLocal() as session:
        assert consent.recover_search_dispatches(session) == 1
    with SessionLocal() as session:
        persisted = session.scalar(select(consent.WebSearchProposal))
        assert persisted.state == "uncertain"
        assert persisted.error_code == "search_dispatch_uncertain"
    with SessionLocal() as repeated:
        assert consent.recover_search_dispatches(repeated) == 0
    with pytest.raises(consent.SearchConsentConflict):
        _decide(consent, job_id, proposed.revision, "approve")
    with pytest.raises(consent.SearchConsentConflict):
        _dispatch(consent, job_id, proposed.revision, claim)


def test_restart_leaves_an_unsent_pending_query_waiting(settings: Settings) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    with SessionLocal() as session:
        assert consent.recover_search_dispatches(session) == 0
    assert _decide(consent, job_id, proposed.revision, "approve").state == "approved"


def test_run_deletion_removes_pending_consent(settings: Settings) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    _pause(consent, job_id)
    with SessionLocal() as session:
        session.delete(session.get(Run, run_id))
        session.commit()
    with SessionLocal() as session:
        assert session.scalar(select(consent.WebSearchProposal)) is None
    with pytest.raises(consent.SearchConsentConflict):
        _decide(consent, job_id, 1, "approve")


@pytest.mark.parametrize("first_action", ["cancel", "dispatch"])
def test_competing_commands_observe_the_first_committed_decision(
    settings: Settings,
    first_action: str,
) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    writer_ready, release_writer, contender_started = Event(), Event(), Event()

    def command(action: str, first: bool) -> Any:
        with SessionLocal() as session:
            if first:

                def hold_before_commit(active: Any) -> None:
                    active.flush()
                    writer_ready.set()
                    assert release_writer.wait(30)

                event.listen(session, "before_commit", hold_before_commit)
            else:
                contender_started.set()
            if action == "cancel":
                return consent.decide_search(session, job_id, proposed.revision, "cancel")
            return consent.claim_search_dispatch(
                session,
                job_id,
                proposed.revision,
                claim,
                provider_endpoint=ENDPOINT,
                provider_revision="provider-one",
                installation_enabled=True,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(command, first_action, True)
        try:
            assert writer_ready.wait(30)
            other = executor.submit(
                command, "dispatch" if first_action == "cancel" else "cancel", False
            )
            assert contender_started.wait(30)
        finally:
            release_writer.set()
        assert first.result(timeout=30).state == (
            "cancelled" if first_action == "cancel" else "dispatching"
        )
        with pytest.raises(consent.SearchConsentConflict):
            other.result(timeout=30)


@pytest.mark.parametrize("lost", [False, True])
def test_only_the_dispatching_execution_can_store_its_result(
    settings: Settings, lost: bool
) -> None:
    from local_lm.web_search import SearchResult, SearchResults

    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    _dispatch(consent, job_id, proposed.revision, claim)
    results = SearchResults(
        (SearchResult("https://example.test/item", "Material", "A summary"),), False, 0
    )
    if lost:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            job.claim_owner = "replacement-claim"
            job.attempt += 1
            session.commit()
    with SessionLocal() as session:
        if lost:
            with pytest.raises(consent.SearchConsentConflict):
                consent.finish_search(session, job_id, proposed.revision, claim, results=results)
        else:
            assert (
                consent.finish_search(
                    session,
                    job_id,
                    proposed.revision,
                    claim,
                    results=results,
                ).state
                == "complete"
            )
    with SessionLocal() as session:
        row = session.scalar(select(consent.WebSearchProposal))
        assert row.state == ("dispatching" if lost else "complete")
        assert row.result_json == (
            {}
            if lost
            else {
                "results": [
                    {
                        "url": "https://example.test/item",
                        "title": "Material",
                        "snippet": "A summary",
                    }
                ],
                "truncated": False,
                "discarded_results": 0,
            }
        )


def test_failed_search_cannot_be_dispatched_again(settings: Settings) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    _dispatch(consent, job_id, proposed.revision, claim)
    with SessionLocal() as session:
        assert (
            consent.finish_search(
                session,
                job_id,
                proposed.revision,
                claim,
                error_code="search_timeout",
            ).state
            == "failed"
        )
    with pytest.raises(consent.SearchConsentConflict):
        _dispatch(consent, job_id, proposed.revision, claim)


@pytest.mark.parametrize("stage", ["awaiting_approval", "approved", "dispatching"])
async def test_application_restart_preserves_consent_without_replaying_uncertain_search(
    settings: Settings,
    app: Any,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    if stage != "awaiting_approval":
        _decide(consent, job_id, proposed.revision, "approve")
    if stage == "dispatching":
        _dispatch(consent, job_id, proposed.revision, _resume(job_id))
    orchestrator = app.state.services.orchestrator
    starts: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        type(orchestrator),
        "start",
        lambda self, job, run: starts.append((job, run)),
    )
    orchestrator.recover_interrupted()
    with SessionLocal() as session:
        proposal = session.scalar(select(consent.WebSearchProposal))
        job = session.get(Job, job_id)
        run = session.get(Run, run_id)
        assert job.claim_owner is None
        assert job.claim_expires_at is None
        assert job.heartbeat_at is None
        assert proposal.state == ("uncertain" if stage == "dispatching" else stage)
        assert starts == ([(job_id, run_id)] if stage == "approved" else [])
        if stage == "dispatching":
            assert job.status == "interrupted" and run.status == "failed"
            assert proposal.error_code == "search_dispatch_uncertain"
        elif stage == "awaiting_approval":
            assert job.status == "paused" and run.status == "queued"
            assert session.get(WorkStep, run.work_step_id).status == "paused"
        else:
            assert job.status == "queued" and run.status == "queued"


@pytest.mark.parametrize("edit", ["query", "endpoint", "provider_revision"])
def test_replacing_a_proposal_requires_approval_of_its_new_revision(
    settings: Settings,
    edit: str,
) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    values = {
        "query": QUERY,
        "provider_endpoint": ENDPOINT,
        "provider_revision": "provider-one",
        "installation_enabled": True,
    }
    if edit == "query":
        values["query"] = "Compare copper and steel"
    elif edit == "endpoint":
        values["provider_endpoint"] = "https://second.example.test"
    else:
        values["provider_revision"] = "provider-two"
    with SessionLocal() as session:
        replacement = consent.replace_search_proposal(
            session,
            job_id,
            proposed.revision,
            **values,
        )
    assert replacement.revision == proposed.revision + 1
    assert replacement.state == "awaiting_approval"
    with pytest.raises(consent.SearchConsentConflict):
        _decide(consent, job_id, proposed.revision, "approve")
    _decide(consent, job_id, replacement.revision, "approve")
    dispatched = _dispatch(
        consent,
        job_id,
        replacement.revision,
        _resume(job_id),
        provider_endpoint=values["provider_endpoint"],
        provider_revision=values["provider_revision"],
    )
    assert dispatched.query == values["query"]
    assert dispatched.provider_endpoint == values["provider_endpoint"]
    assert dispatched.provider_revision == values["provider_revision"]


def test_an_approval_that_won_first_cannot_be_retargeted(settings: Settings) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    with SessionLocal() as session, pytest.raises(consent.SearchConsentConflict):
        consent.replace_search_proposal(
            session,
            job_id,
            proposed.revision,
            query="A different query",
            provider_endpoint=ENDPOINT,
            provider_revision="provider-one",
            installation_enabled=True,
        )
    sent = _dispatch(consent, job_id, proposed.revision, _resume(job_id))
    assert sent.query == QUERY


@pytest.mark.parametrize("revision", [True, "1", 0, -1, 2**63])
def test_malformed_revision_is_refused_before_database_comparison(
    settings: Settings,
    revision: Any,
) -> None:
    consent, job_id, _, _ = _seed(settings)
    _pause(consent, job_id)
    with pytest.raises(consent.SearchConsentConflict):
        _decide(consent, job_id, revision, "approve")


def test_an_old_edit_cannot_overwrite_a_newer_query(settings: Settings) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    with SessionLocal() as session:
        newer = consent.replace_search_proposal(
            session,
            job_id,
            proposed.revision,
            query="Compare iron and nickel",
            provider_endpoint=ENDPOINT,
            provider_revision="provider-one",
            installation_enabled=True,
        )
    with SessionLocal() as session, pytest.raises(consent.SearchConsentConflict):
        consent.replace_search_proposal(
            session,
            job_id,
            proposed.revision,
            query="An obsolete edit",
            provider_endpoint=ENDPOINT,
            provider_revision="provider-one",
            installation_enabled=True,
        )
    with SessionLocal() as session:
        persisted = session.scalar(select(consent.WebSearchProposal))
        assert persisted.query == newer.query
        assert persisted.revision == newer.revision


def _schedule(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, str, Any, Any]:
    consent, job_id, _, chat_id = _seed(settings)
    moment = utcnow()
    monkeypatch.setattr(consent, "utcnow", lambda: moment)
    with SessionLocal() as session:
        session.get(Chat, chat_id).web_settings_json = {
            "allow_search": True,
            "allow_search_without_asking": True,
        }
        session.commit()
    return consent, job_id, _pause(consent, job_id), moment


def _approve_automatic(consent: Any, job_id: str, revision: int, **overrides: Any) -> Any:
    values = {
        "provider_endpoint": ENDPOINT,
        "provider_revision": "provider-one",
        "installation_enabled": True,
    }
    values.update(overrides)
    with SessionLocal() as session:
        return consent.approve_scheduled_search(session, job_id, revision, **values)


def test_automatic_search_has_a_persisted_five_second_cancellation_window(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consent, job_id, proposal, moment = _schedule(settings, monkeypatch)
    assert proposal.state == "scheduled"
    assert proposal.dispatch_after == moment + timedelta(seconds=5)
    assert _approve_automatic(consent, job_id, proposal.revision).state == "scheduled"
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job.status == "paused"
        run = session.get(Run, job.run_id)
        assert run.provenance_json["web_search"]["state"] == "scheduled"
        assert run.provenance_json["web_search"]["query"] == QUERY
    monkeypatch.setattr(consent, "utcnow", lambda: moment + timedelta(seconds=5))
    assert _approve_automatic(consent, job_id, proposal.revision).state == "approved"
    assert _dispatch(consent, job_id, proposal.revision, _resume(job_id)).state == "dispatching"


@pytest.mark.parametrize("change", ["global", "chat", "confirmation", "provider"])
def test_automatic_search_rechecks_permission_and_provider_when_the_window_ends(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    consent, job_id, proposal, moment = _schedule(settings, monkeypatch)
    monkeypatch.setattr(consent, "utcnow", lambda: moment + timedelta(seconds=5))
    overrides: dict[str, Any] = {}
    if change == "global":
        overrides["installation_enabled"] = False
    elif change in ("chat", "confirmation"):
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, job.run_id)
            chat = session.get(Chat, run.chat_id)
            chat.web_settings_json = {} if change == "chat" else {"allow_search": True}
            session.commit()
    else:
        overrides["provider_revision"] = "provider-two"
    result = _approve_automatic(consent, job_id, proposal.revision, **overrides)
    assert result.state == ("cancelled" if change in ("global", "chat") else "awaiting_approval")
    assert result.dispatch_after is None
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job.status == ("queued" if result.state == "cancelled" else "paused")


def test_cancelling_the_automatic_window_cannot_be_undone_by_its_timer(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consent, job_id, proposal, moment = _schedule(settings, monkeypatch)
    _decide(consent, job_id, proposal.revision, "cancel")
    monkeypatch.setattr(consent, "utcnow", lambda: moment + timedelta(seconds=5))
    with pytest.raises(consent.SearchConsentConflict):
        _approve_automatic(consent, job_id, proposal.revision)
    with SessionLocal() as session:
        assert session.scalar(select(consent.WebSearchProposal.state)) == "cancelled"
        assert session.get(Job, job_id).status == "queued"


def test_search_history_survives_removal_of_a_completed_execution_job(settings: Settings) -> None:
    from local_lm.web_search import SearchResult, SearchResults

    consent, job_id, run_id, _ = _seed(settings)
    proposal = _pause(consent, job_id)
    _decide(consent, job_id, proposal.revision, "approve")
    claim = _resume(job_id)
    _dispatch(consent, job_id, proposal.revision, claim)
    with SessionLocal() as session:
        consent.finish_search(
            session,
            job_id,
            proposal.revision,
            claim,
            results=SearchResults(
                (SearchResult("https://example.test/source", "Material", "A summary"),), False, 0
            ),
        )
    with SessionLocal() as session:
        session.get(Run, run_id).status = "complete"
        session.delete(session.get(Job, job_id))
        session.commit()
    with SessionLocal() as session:
        assert session.scalar(select(consent.WebSearchProposal)) is None
        history = session.get(Run, run_id).provenance_json["web_search"]
        assert history["query"] == QUERY
        assert history["provider_endpoint"] == ENDPOINT
        assert history["state"] == "complete"
        assert history["result_count"] == 1
        assert history["results"]["results"][0]["url"] == "https://example.test/source"


@pytest.mark.parametrize("automatic", [True, False])
def test_dispatch_rechecks_automatic_permission_without_revoking_manual_approval(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    automatic: bool,
) -> None:
    if automatic:
        consent, job_id, proposal, moment = _schedule(settings, monkeypatch)
        monkeypatch.setattr(consent, "utcnow", lambda: moment + timedelta(seconds=5))
        assert _approve_automatic(consent, job_id, proposal.revision).state == "approved"
    else:
        consent, job_id, _, _ = _seed(settings)
        proposal = _pause(consent, job_id)
        _decide(consent, job_id, proposal.revision, "approve")
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        run = session.get(Run, job.run_id)
        session.get(Chat, run.chat_id).web_settings_json = {"allow_search": True}
        session.commit()
    claim = _resume(job_id)
    if automatic:
        with pytest.raises(consent.SearchConsentConflict):
            _dispatch(consent, job_id, proposal.revision, claim)
        with SessionLocal() as session:
            row = session.scalar(select(consent.WebSearchProposal))
            assert row.state == "approved"
            assert row.dispatch_owner is None
    else:
        assert _dispatch(consent, job_id, proposal.revision, claim).state == "dispatching"


async def test_restart_reoffers_the_full_automatic_window_and_resumes_its_timer(
    settings: Settings,
    app: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import UTC

    consent, job_id, proposal, moment = _schedule(settings, monkeypatch)
    restarted = moment + timedelta(minutes=1)
    monkeypatch.setattr(consent, "utcnow", lambda: restarted)
    orchestrator = app.state.services.orchestrator
    resumed: list[str] = []
    monkeypatch.setattr(type(orchestrator), "resume_search", lambda self, job: resumed.append(job))
    orchestrator.recover_interrupted()
    assert resumed == [job_id]
    with SessionLocal() as session:
        row = session.scalar(select(consent.WebSearchProposal))
        job = session.get(Job, job_id)
        assert row.state == "scheduled"
        assert row.dispatch_after.replace(tzinfo=UTC) == restarted + timedelta(seconds=5)
        assert row.dispatch_after.replace(tzinfo=UTC) != proposal.dispatch_after
        assert job.status == "paused" and job.claim_owner is None
        assert job.claim_expires_at is None and job.heartbeat_at is None
        run = session.get(Run, row.run_id)
        assert run.status == "queued"
        assert (
            run.provenance_json["web_search"]["dispatch_after"]
            == (restarted + timedelta(seconds=5)).isoformat()
        )


def _prepare_dispatch(
    consent: Any, job_id: str, revision: int, claim: JobClaim, **changes: Any
) -> Any:
    values = {
        "provider_endpoint": ENDPOINT,
        "provider_revision": "provider-one",
        "installation_enabled": True,
    }
    values.update(changes)
    with SessionLocal() as session:
        return consent.prepare_search_dispatch(session, job_id, revision, claim, **values)


@pytest.mark.parametrize("change", ["installation", "chat", "unconfigured"])
def test_dispatch_preparation_continues_locally_after_permission_or_provider_loss(
    settings: Settings,
    change: str,
) -> None:
    consent, job_id, run_id, chat_id = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    values: dict[str, Any] = {}
    if change == "installation":
        values["installation_enabled"] = False
    elif change == "chat":
        with SessionLocal() as session:
            session.get(Chat, chat_id).web_settings_json = {}
            session.commit()
    else:
        values.update(provider_endpoint=None, provider_revision=None)
    result = _prepare_dispatch(consent, job_id, proposed.revision, claim, **values)
    assert result.state == "cancelled"
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        run = session.get(Run, run_id)
        stored = session.scalar(select(consent.WebSearchProposal))
        assert job.status == "running" and job.claim_owner == claim.token
        assert stored.dispatch_owner is None and stored.dispatch_attempt is None
        assert stored.error_code == (
            "search_provider_invalid" if change == "unconfigured" else "search_permission_revoked"
        )
        assert run.provenance_json["web_search"]["state"] == "cancelled"


@pytest.mark.parametrize("change", ["destination", "account"])
def test_dispatch_preparation_requires_new_approval_for_changed_provider(
    settings: Settings,
    change: str,
) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    endpoint = "https://another-search.example.test" if change == "destination" else ENDPOINT
    result = _prepare_dispatch(
        consent,
        job_id,
        proposed.revision,
        claim,
        provider_endpoint=endpoint,
        provider_revision="provider-two",
    )
    assert result.state == "awaiting_approval" and result.revision == proposed.revision + 1
    assert result.query == QUERY and result.provider_endpoint == endpoint
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        run = session.get(Run, run_id)
        stored = session.scalar(select(consent.WebSearchProposal))
        assert job.status == "paused" and job.claim_owner == claim.token
        assert run.status == "queued" and session.get(WorkStep, run.work_step_id).status == "paused"
        assert stored.provider_revision == "provider-two"
        assert stored.dispatch_owner is None and stored.dispatch_attempt is None
        assert stored.approved_automatically is False
    with pytest.raises(consent.SearchConsentConflict):
        _decide(consent, job_id, proposed.revision, "approve")
    _decide(consent, job_id, result.revision, "approve")
    current_claim = _resume(job_id)
    started = _prepare_dispatch(
        consent,
        job_id,
        result.revision,
        current_claim,
        provider_endpoint=endpoint,
        provider_revision="provider-two",
    )
    assert started.state == "dispatching"


@pytest.mark.parametrize("automatic", [False, True])
def test_dispatch_preparation_respects_revoked_automatic_approval(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    automatic: bool,
) -> None:
    if automatic:
        consent, job_id, proposed, moment = _schedule(settings, monkeypatch)
        monkeypatch.setattr(consent, "utcnow", lambda: moment + timedelta(seconds=5))
        assert _approve_automatic(consent, job_id, proposed.revision).state == "approved"
    else:
        consent, job_id, _, _ = _seed(settings)
        proposed = _pause(consent, job_id)
        _decide(consent, job_id, proposed.revision, "approve")
    with SessionLocal() as session:
        run_id = session.get(Job, job_id).run_id
        chat_id = session.get(Run, run_id).chat_id
        session.get(Chat, chat_id).web_settings_json = {"allow_search": True}
        session.commit()
    claim = _resume(job_id)
    result = _prepare_dispatch(consent, job_id, proposed.revision, claim)
    assert result.state == ("awaiting_approval" if automatic else "dispatching")
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        run = session.get(Run, run_id)
        proposal = session.scalar(select(consent.WebSearchProposal))
        assert job.status == ("paused" if automatic else "running")
        if automatic:
            assert result.revision == proposed.revision + 1
            assert run.status == "queued"
            assert proposal.dispatch_owner is None and proposal.approved_automatically is False
        else:
            assert (
                proposal.dispatch_owner == claim.token
                and proposal.dispatch_attempt == claim.attempt
            )


@pytest.mark.parametrize("lost", ["token", "attempt", "expired"])
def test_dispatch_preparation_cannot_reconcile_for_an_execution_that_lost_ownership(
    settings: Settings,
    lost: str,
) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if lost == "token":
            job.claim_owner = "another-execution"
        elif lost == "attempt":
            job.attempt += 1
        else:
            job.claim_expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
    with pytest.raises(consent.SearchConsentConflict):
        _prepare_dispatch(consent, job_id, proposed.revision, claim, installation_enabled=False)
    with SessionLocal() as session:
        proposal = session.scalar(select(consent.WebSearchProposal))
        assert proposal.state == "approved" and proposal.revision == proposed.revision
        assert proposal.dispatch_owner is None


def test_dispatch_preparation_cannot_start_the_same_provider_request_twice(
    settings: Settings,
) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    assert _prepare_dispatch(consent, job_id, proposed.revision, claim).state == "dispatching"
    with pytest.raises(consent.SearchConsentConflict):
        _prepare_dispatch(consent, job_id, proposed.revision, claim)


def test_dispatch_preparation_keeps_a_cancellation_that_committed_first(settings: Settings) -> None:
    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    _decide(consent, job_id, proposed.revision, "cancel")
    result = _prepare_dispatch(consent, job_id, proposed.revision, claim)
    assert result.state == "cancelled"
    with SessionLocal() as session:
        proposal = session.scalar(select(consent.WebSearchProposal))
        assert proposal.dispatch_owner is None and session.get(Job, job_id).status == "running"


def test_saved_search_evidence_is_restored_without_authorizing_another_dispatch(
    settings: Settings,
) -> None:
    from local_lm.web_search import SearchResult, SearchResults

    consent, job_id, _, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    _dispatch(consent, job_id, proposed.revision, claim)
    expected = SearchResults(
        (SearchResult("https://example.test/materials", "Material data", "A neutral summary"),),
        truncated=True,
        discarded_results=3,
    )
    with SessionLocal() as session:
        consent.finish_search(session, job_id, proposed.revision, claim, results=expected)
    with SessionLocal() as session:
        restored = consent.search_result_for_answer(session, job_id, proposed.revision, claim)
    assert restored == expected
    with pytest.raises(consent.SearchConsentConflict):
        _prepare_dispatch(consent, job_id, proposed.revision, claim)


def test_an_unfinished_saved_dispatch_becomes_uncertain_instead_of_repeating(
    settings: Settings,
) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    _dispatch(consent, job_id, proposed.revision, claim)
    with SessionLocal() as session:
        assert consent.search_result_for_answer(session, job_id, proposed.revision, claim) is None
    with SessionLocal() as session:
        stored = session.scalar(select(consent.WebSearchProposal))
        assert stored.state == "uncertain" and stored.error_code == "search_dispatch_uncertain"
        assert session.get(Run, run_id).provenance_json["web_search"]["state"] == "uncertain"
        assert session.get(Job, job_id).status == "running"
    with SessionLocal() as session:
        assert consent.search_result_for_answer(session, job_id, proposed.revision, claim) is None
    with pytest.raises(consent.SearchConsentConflict):
        _prepare_dispatch(consent, job_id, proposed.revision, claim)


@pytest.mark.parametrize("lost", ["token", "attempt", "expired"])
def test_saved_search_outcome_cannot_be_changed_after_execution_ownership_is_lost(
    settings: Settings,
    lost: str,
) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    claim = _resume(job_id)
    _dispatch(consent, job_id, proposed.revision, claim)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if lost == "token":
            job.claim_owner = "new-owner"
        elif lost == "attempt":
            job.attempt += 1
        else:
            job.claim_expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
    with SessionLocal() as session, pytest.raises(consent.SearchConsentConflict):
        consent.search_result_for_answer(session, job_id, proposed.revision, claim)
    with SessionLocal() as session:
        stored = session.scalar(select(consent.WebSearchProposal))
        assert stored.state == "dispatching" and stored.error_code is None
        assert session.get(Run, run_id).provenance_json["web_search"]["state"] == "dispatching"


def test_revocation_commits_and_rolls_back_with_its_owning_job(settings: Settings) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    proposal = _pause(consent, job_id)
    _decide(consent, job_id, proposal.revision, "approve")
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        job = session.get(Job, job_id)
        job.status = "cancelled"
        consent.cancel_pending_search(session, run)
        session.rollback()
    with SessionLocal() as session:
        assert session.get(Job, job_id).status == "queued"
        assert session.scalar(select(consent.WebSearchProposal)).state == "approved"
        run = session.get(Run, run_id)
        session.get(Job, job_id).status = "cancelled"
        consent.cancel_pending_search(session, run)
        session.commit()
    with SessionLocal() as session:
        assert session.get(Job, job_id).status == "cancelled"
        assert session.scalar(select(consent.WebSearchProposal)).state == "cancelled"
        assert session.get(Run, run_id).provenance_json["web_search"]["state"] == "cancelled"


def test_revocation_cannot_claim_an_already_started_dispatch_was_cancelled(
    settings: Settings,
) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    proposal = _pause(consent, job_id)
    _decide(consent, job_id, proposal.revision, "approve")
    claim = _resume(job_id)
    with SessionLocal() as session:
        dispatched = consent.prepare_search_dispatch(
            session,
            job_id,
            proposal.revision,
            claim,
            provider_endpoint=ENDPOINT,
            provider_revision="provider-one",
            installation_enabled=True,
        )
    assert dispatched.state == "dispatching"
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        consent.cancel_pending_search(session, run)
        session.commit()
    with SessionLocal() as session:
        assert session.scalar(select(consent.WebSearchProposal)).state == "dispatching"
        assert session.get(Run, run_id).provenance_json["web_search"]["state"] == "dispatching"


@pytest.mark.parametrize("status", ["running", "failed", "cancelled", "interrupted"])
async def test_retry_after_restart_revokes_legacy_pending_approval(
    settings: Settings,
    app: Any,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    proposed = _pause(consent, job_id)
    _decide(consent, job_id, proposed.revision, "approve")
    _resume(job_id)
    with SessionLocal() as session:
        session.get(Job, job_id).status = status
        if status != "running":
            session.get(Run, run_id).status = "failed"
        session.commit()
    orch = app.state.services.orchestrator
    starts = []
    monkeypatch.setattr(type(orch), "start", lambda self, *args: starts.append(args))
    orch.recover_interrupted()
    with SessionLocal() as session:
        # Existing recovery never replays a dispatch. An older pending
        # approval is instead revoked by the explicit retry transaction.
        assert session.scalar(select(consent.WebSearchProposal)).state == "approved"
        orch.prepare_retry(session, session.get(Run, run_id))
        session.commit()
    with SessionLocal() as session:
        row = session.scalar(select(consent.WebSearchProposal))
        assert row.state == "cancelled"
        assert row.dispatch_after is None and not row.approved_automatically
        assert session.get(Run, run_id).provenance_json["web_search"]["state"] == "cancelled"
    assert starts == []


@pytest.mark.parametrize("first_action", ["stop", "dispatch"])
def test_stopping_and_preparing_dispatch_serialize_the_actual_writes(
    settings: Settings,
    first_action: str,
) -> None:
    consent, job_id, run_id, _ = _seed(settings)
    proposal = _pause(consent, job_id)
    _decide(consent, job_id, proposal.revision, "approve")
    claim = _resume(job_id)
    writer_ready, release_writer, contender_started = Event(), Event(), Event()

    def command(action: str, first: bool) -> str:
        with SessionLocal() as session:
            if first:

                def hold_before_commit(active: Any) -> None:
                    active.flush()
                    writer_ready.set()
                    assert release_writer.wait(30)

                event.listen(session, "before_commit", hold_before_commit)
            else:
                contender_started.set()
            if action == "dispatch":
                return consent.prepare_search_dispatch(
                    session,
                    job_id,
                    proposal.revision,
                    claim,
                    provider_endpoint=ENDPOINT,
                    provider_revision="provider-one",
                    installation_enabled=True,
                ).state
            run = session.get(Run, run_id)
            session.get(Job, job_id).status = "cancelled"
            run.status = "cancelled"
            consent.cancel_pending_search(session, run)
            session.commit()
            return "cancelled"

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(command, first_action, True)
        try:
            assert writer_ready.wait(30)
            second = executor.submit(
                command,
                "dispatch" if first_action == "stop" else "stop",
                False,
            )
            assert contender_started.wait(30)
        finally:
            release_writer.set()
        assert first.result(timeout=30) == (
            "cancelled" if first_action == "stop" else "dispatching"
        )
        if first_action == "stop":
            with pytest.raises(consent.SearchConsentConflict):
                second.result(timeout=30)
        else:
            assert second.result(timeout=30) == "cancelled"
    with SessionLocal() as session:
        row = session.scalar(select(consent.WebSearchProposal))
        assert row.state == ("cancelled" if first_action == "stop" else "dispatching")
        assert session.get(Job, job_id).status == "cancelled"
