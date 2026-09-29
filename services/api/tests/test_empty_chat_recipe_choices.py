"""Require deliberate cleanup of chats that retain workflow recipe choices."""

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session
from test_empty_chat_deletion import _filters, _issued, _spend, chat
from test_empty_chat_deletion import session as session

from local_lm.empty_chats import (
    EmptyChatClass,
    EmptyChatExecuteError,
    EmptyChatExecuteRefusal,
    classify,
    empty_chat_page,
    execute_deletion,
)
from local_lm.models import Chat, ChatWorkflowUseCaseSelection, WorkflowUseCasePreset


def _choice(session: Session, chat_id: str, explicit: bool) -> ChatWorkflowUseCaseSelection:
    preset = WorkflowUseCasePreset(name="Example recipe", use_case="image_edit")
    session.add(preset)
    session.flush()
    choice = ChatWorkflowUseCaseSelection(
        chat_id=chat_id, use_case="image_edit", preset_id=preset.id if explicit else None
    )
    session.add(choice)
    session.flush()
    return choice


@pytest.mark.parametrize("explicit", [False, True], ids=["automatic", "recipe"])
def test_a_recipe_choice_makes_an_otherwise_blank_chat_configured(
    session: Session, explicit: bool
) -> None:
    configured = chat(session)
    _choice(session, configured.id, explicit)
    entry = classify(session, configured)
    assert entry.classification is EmptyChatClass.CONFIGURED_BLANK
    assert entry.reasons == ("settings_overridden",)
    assert empty_chat_page(session).entries == ()
    assert [row.chat_id for row in empty_chat_page(session, include_configured=True).entries] == [
        configured.id
    ]


@pytest.mark.parametrize("explicit", [False, True], ids=["automatic", "recipe"])
def test_recipe_choices_require_acknowledgement_before_chat_deletion(
    session: Session, explicit: bool
) -> None:
    configured = chat(session)
    choice = _choice(session, configured.id, explicit)
    with pytest.raises(EmptyChatExecuteError) as refused:
        _spend(session, [configured], filters=_filters(0, include_configured=True))
    assert refused.value.refusal is EmptyChatExecuteRefusal.CONFIGURED_NOT_ACKNOWLEDGED
    assert session.get(Chat, configured.id) is not None
    assert session.get(ChatWorkflowUseCaseSelection, (configured.id, "image_edit")) is choice


@pytest.mark.parametrize("change", ["add", "replace", "remove", "stored_replace"])
def test_recipe_choice_changes_invalidate_a_deletion_preview(session: Session, change: str) -> None:
    configured = chat(session, title="Example chat")
    choice = _choice(session, configured.id, False) if change != "add" else None
    filters = _filters(0, include_configured=True)
    preview_id, digest, count = _issued(session, [configured], filters)
    if change == "add":
        _choice(session, configured.id, True)
    elif change in {"replace", "stored_replace"}:
        assert choice is not None
        preset = WorkflowUseCasePreset(name="Another recipe", use_case="image_edit")
        session.add(preset)
        session.flush()
        if change == "replace":
            choice.preset_id = preset.id
        else:
            session.execute(
                update(ChatWorkflowUseCaseSelection)
                .where(ChatWorkflowUseCaseSelection.chat_id == configured.id)
                .values(preset_id=preset.id),
                execution_options={"synchronize_session": False},
            )
            assert choice.preset_id is None
    else:
        assert choice is not None
        session.delete(choice)
    session.flush()
    with pytest.raises(EmptyChatExecuteError) as refused:
        execute_deletion(
            session,
            operation_id="changed-recipe",
            preview_id=preview_id,
            digest=digest,
            acknowledged_count=count,
            acknowledged_configured=True,
            chat_ids=[configured.id],
            filters=filters,
        )
    assert refused.value.refusal is EmptyChatExecuteRefusal.SELECTION_DRIFTED
    assert refused.value.chat_ids == (configured.id,)
    assert session.get(Chat, configured.id) is not None
