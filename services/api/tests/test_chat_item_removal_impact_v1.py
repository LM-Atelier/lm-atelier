from __future__ import annotations

import pytest

from local_lm.chat_item_removal_impact_v1 import (
    INVALID_IMPACT,
    MAX_ID,
    MAX_REF_COUNT,
    ChatItemRemovalImpactError,
    ChatItemRemovalImpactV1,
    declare_chat_item_removal_impact,
)


def test_declared_preview_is_not_repository_evidence() -> None:
    impact = declare_chat_item_removal_impact(
        message_id="m1",
        declared_has_replies=True,
        declared_source_backs_regeneration=False,
        proposed_detached_ref_ids=["r1", "r2"],
    )
    assert impact.declared_has_replies is True
    assert impact.proposed_detached_ref_ids == ("r1", "r2")
    assert impact.repository_snapshot_verified is False
    assert impact.message_id_bound is False
    assert impact.impact_verified is False
    assert impact.execute_authorized is False
    assert not hasattr(impact, "has_replies")
    assert not hasattr(impact, "detached_ref_ids")


def test_refuses_invalid_and_unbounded_refs() -> None:
    with pytest.raises(ChatItemRemovalImpactError, match=INVALID_IMPACT):
        declare_chat_item_removal_impact(
            message_id="m1",
            declared_has_replies=True,
            declared_source_backs_regeneration=False,
            proposed_detached_ref_ids=["r1"] * (MAX_REF_COUNT + 1),
        )
    with pytest.raises(ChatItemRemovalImpactError, match=INVALID_IMPACT):
        declare_chat_item_removal_impact(
            message_id="m" * (MAX_ID + 1),
            declared_has_replies=False,
            declared_source_backs_regeneration=False,
            proposed_detached_ref_ids=(),
        )
    with pytest.raises(ChatItemRemovalImpactError, match=INVALID_IMPACT):
        declare_chat_item_removal_impact(
            message_id="m1",
            declared_has_replies=1,
            declared_source_backs_regeneration=False,
            proposed_detached_ref_ids=(),
        )
    with pytest.raises(ChatItemRemovalImpactError, match=INVALID_IMPACT):
        declare_chat_item_removal_impact(
            message_id="m1",
            declared_has_replies=False,
            declared_source_backs_regeneration=False,
            proposed_detached_ref_ids=["bad\ud800"],
        )
    with pytest.raises(ChatItemRemovalImpactError, match=INVALID_IMPACT):
        declare_chat_item_removal_impact(
            message_id="m1",
            declared_has_replies=False,
            declared_source_backs_regeneration=False,
            proposed_detached_ref_ids=["r1", "r1"],
        )


def test_public_constructor_cannot_verify_impact() -> None:
    with pytest.raises(ChatItemRemovalImpactError, match=INVALID_IMPACT):
        ChatItemRemovalImpactV1()
    with pytest.raises(TypeError):
        ChatItemRemovalImpactV1(
            schema="lm-atelier-chat-item-removal-impact-v1",
            schema_version=1,
            repository_snapshot_verified=True,
            message_id_bound=True,
            impact_verified=True,
        )
