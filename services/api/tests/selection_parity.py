"""Resolve a turn's model and workflow the workflow way and the legacy way, and say how they differ.

A turn chooses its model profile and workflow revision in
`ConversationOrchestrator._profile_and_workflow_for_operation`. It asks the
workflow path first (`_workflow_family_for_operation`). When that path has no
answer, the turn falls back to the legacy path: the profile the chat names, or
its role's default or best-matching profile, and then that profile's workflow.

Retiring the legacy path changes exactly the turns that fall back today, so
this reports, for one chat and one operation:

- what the workflow path answers, or why it has no answer;
- what the legacy path alone answers, run as the entry point runs it when the
  workflow path has no answer: the same entry point with only that first step
  switched off, so neither side is a copy of the application's code.

Each side runs in a session of its own that is rolled back, because the legacy
path may move a project's pinned revision onto an identical successor and
write that down.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.orm import Session

from local_lm import orchestrator as orchestrator_module
from local_lm.domain import Operation
from local_lm.models import Chat, ModelProfile, WorkflowRevision
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.workflow_selection import WorkflowFamilySelectionError

#: What a turn does today.
WORKFLOW = "workflow"
FALLS_BACK = "falls back"
REFUSED = "refused"

#: Why the workflow path has no answer before it resolves anything.
CHAT_PROFILE = "chat profile"
PROJECT_REVISION = "project revision"

#: The ways either path refuses a turn.
REFUSALS = (ValueError, LookupError, RuntimeError)


@dataclass(frozen=True)
class Choice:
    """The model profile and workflow revision one path chose."""

    profile_id: str | None
    revision_id: str | None


@dataclass(frozen=True)
class TurnSelection:
    """How one chat's turn of one operation is resolved each way."""

    chat_id: str
    operation: Operation
    #: WORKFLOW when the workflow path answers, so retiring the legacy path
    #: changes nothing for this turn; FALLS_BACK when the legacy path answers
    #: today; REFUSED when today's turn is refused.
    outcome: str
    #: Why the turn falls back or is refused, empty for WORKFLOW.
    cause: str
    workflow: Choice | None
    legacy: Choice | None
    #: The workflow path answered through a profile's compatibility family,
    #: the one kind of answer the legacy path can express too.
    compatibility: bool = False

    @property
    def agrees(self) -> bool | None:
        """Whether both paths chose the same profile and revision.

        Asked only of a compatibility answer: a workflow family has no legacy
        counterpart, so the profile the legacy path would fall back to says
        nothing about it.
        """
        if not self.compatibility or self.workflow is None or self.legacy is None:
            return None
        return self.workflow == self.legacy


def _choice(profile: ModelProfile | None, revision: WorkflowRevision | None) -> Choice:
    return Choice(profile.id if profile else None, revision.id if revision else None)


def _refusal(error: BaseException) -> str:
    reason = getattr(error, "reason", None) or getattr(error, "code", None)
    return f"{type(error).__name__}: {reason}" if reason else type(error).__name__


def compare_turn_selection(
    orchestrator: ConversationOrchestrator,
    session_factory: Callable[[], Session],
    chat_id: str,
    operation: Operation,
    prompt: str = "",
) -> TurnSelection:
    """Resolve one chat's turn both ways and classify the result."""

    asked: list[tuple[str | None, str | None]] = []
    resolve = orchestrator_module.resolve_workflow_family

    def recording(*args: Any, **kwargs: Any) -> Any:
        try:
            resolved = resolve(*args, **kwargs)
        except WorkflowFamilySelectionError as error:
            asked.append((kwargs.get("mode"), error.reason))
            raise
        asked.append((kwargs.get("mode"), None))
        return resolved

    workflow: Choice | None = None
    compatibility = False
    workflow_refusal: BaseException | None = None
    chat_names_a_profile = False
    with session_factory() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None, chat_id
        try:
            with patch.object(orchestrator_module, "resolve_workflow_family", recording):
                answered = orchestrator._workflow_family_for_operation(
                    session, chat, operation, prompt, preferred_revision_id=None
                )
        except REFUSALS as error:
            answered, workflow_refusal = None, error
        if answered is not None:
            workflow = _choice(answered[0], answered[2])
            compatibility = bool(answered[1].get("workflow_compatibility"))
        elif workflow_refusal is None and not asked:
            selection = orchestrator._chat_workflow_for_turn(session, chat, operation, None)
            chat_names_a_profile = selection.mode == "legacy" and selection.profile_id is not None
        session.rollback()

    legacy: Choice | None = None
    legacy_refusal: BaseException | None = None
    with session_factory() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None, chat_id
        try:
            with patch.object(orchestrator, "_workflow_family_for_operation", return_value=None):
                profile, _selection, revision = orchestrator._profile_and_workflow_for_operation(
                    session, chat, operation, prompt
                )
            legacy = _choice(profile, revision)
        except REFUSALS as error:
            legacy_refusal = error
        session.rollback()

    if workflow_refusal is not None:
        outcome, cause = REFUSED, f"workflow: {_refusal(workflow_refusal)}"
    elif workflow is not None:
        outcome, cause = WORKFLOW, ""
    elif legacy_refusal is not None:
        outcome, cause = REFUSED, f"legacy: {_refusal(legacy_refusal)}"
    elif not asked:
        outcome, cause = FALLS_BACK, CHAT_PROFILE if chat_names_a_profile else PROJECT_REVISION
    else:
        mode, reason = asked[-1]
        outcome, cause = FALLS_BACK, f"{mode}: {reason}"
    return TurnSelection(
        chat_id=chat_id,
        operation=operation,
        outcome=outcome,
        cause=cause,
        workflow=workflow,
        legacy=legacy,
        compatibility=compatibility,
    )


def compare_every_chat(
    orchestrator: ConversationOrchestrator,
    session_factory: Callable[[], Session],
    operations: Iterable[Operation] = tuple(Operation),
    prompt: str = "",
) -> list[TurnSelection]:
    """Every chat's turn of every operation, resolved both ways, in chat id order."""

    with session_factory() as session:
        chat_ids = list(session.scalars(select(Chat.id).order_by(Chat.id)).all())
    wanted = tuple(operations)
    return [
        compare_turn_selection(orchestrator, session_factory, chat_id, operation, prompt)
        for chat_id in chat_ids
        for operation in wanted
    ]


def summarize(results: Iterable[TurnSelection]) -> Counter[str]:
    """How many turns do what today, and why the rest fall back or are refused.

    The keys are the outcome, then the cause where there is one, and a
    compatibility answer the legacy path would not have chosen is counted as a
    disagreement of its own: over a real database these are the turns that
    retiring the legacy path would change.
    """

    counts: Counter[str] = Counter()
    for item in results:
        counts[f"{item.outcome}: {item.cause}" if item.cause else item.outcome] += 1
        if item.agrees is False:
            counts["workflow: disagrees with the legacy path"] += 1
    return counts
