from __future__ import annotations

import pytest

from local_lm.search_visibility_v1 import (
    INVALID_VISIBILITY,
    SearchVisibilityError,
    SearchVisibilityV1,
    evaluate_search_visibility,
    filter_indexable_bodies,
)


def _eval(**over):
    facts = {
        "message_id": "m1",
        "transcript_visible": True,
        "content_removed": False,
        "private_session": False,
        "helper_session": False,
        "secret_payload": False,
    }
    facts.update(over)
    return evaluate_search_visibility(**facts)


def test_visibility_public_constructor_cannot_mint_eligible() -> None:
    with pytest.raises(TypeError):
        SearchVisibilityV1(
            schema="lm-atelier-search-visibility-v1",
            schema_version=1,
            message_id_bound=False,
            eligible=True,
            code="deny_secret_payload",
        )
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        SearchVisibilityV1(
            schema="lm-atelier-search-visibility-v1",
            schema_version=1,
        )


def test_visibility_positive_facts_are_witness_owned() -> None:
    denied = _eval(content_removed=True)
    assert denied.eligible is False
    assert denied.code == "deny_content_removed"
    assert denied.may_index_body is False
    assert denied.may_emit_snippet is False
    assert denied.may_rank is False
    assert denied.index_rebuild_authorized is False
    eligible = _eval()
    assert eligible.eligible is True
    assert eligible.may_index_body is True
    assert eligible.may_emit_snippet is True
    assert eligible.may_rank is True
    with pytest.raises(TypeError):
        SearchVisibilityV1(
            schema="lm-atelier-search-visibility-v1",
            schema_version=1,
            message_id_bound=True,
            eligible=False,
            code="deny_content_removed",
            may_index_body=True,
            may_emit_snippet=True,
            may_rank=True,
        )
    with pytest.raises(TypeError):
        SearchVisibilityV1(
            schema="lm-atelier-search-visibility-v1",
            schema_version=1,
            message_id_bound=True,
            eligible=False,
            code="deny_content_removed",
            query_execution_authorized=True,
        )


def test_visibility_denies_private_helper_secret_and_hidden() -> None:
    assert _eval(transcript_visible=False).code == "deny_not_transcript_visible"
    assert _eval(private_session=True).code == "deny_private_session"
    assert _eval(helper_session=True).code == "deny_helper_session"
    assert _eval(secret_payload=True).code == "deny_secret_payload"
    for denied in (
        _eval(transcript_visible=False),
        _eval(private_session=True),
        _eval(helper_session=True),
        _eval(secret_payload=True),
    ):
        assert denied.eligible is False
        assert denied.may_rank is False


def test_visibility_refuses_hostile_facts() -> None:
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        _eval(transcript_visible="yes")
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        _eval(content_removed=1)
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        _eval(message_id="bad id")
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        _eval(message_id="")
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        _eval(message_id="x" * 129)
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        filter_indexable_bodies("nope")
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        filter_indexable_bodies([{"message_id": "m1"}])
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        filter_indexable_bodies(
            [
                {
                    "message_id": "m1",
                    "body": None,
                    "transcript_visible": True,
                    "content_removed": False,
                    "private_session": False,
                    "helper_session": False,
                    "secret_payload": False,
                }
            ]
        )


def test_filter_refuses_hostile_str_keys() -> None:
    class HostileKey(str):
        def __eq__(self, other):
            raise RuntimeError("private attacker detail")

        def __hash__(self):
            return str.__hash__(self)

    row = {
        HostileKey("message_id"): "keep",
        HostileKey("body"): "hello",
        HostileKey("transcript_visible"): True,
        HostileKey("content_removed"): False,
        HostileKey("private_session"): False,
        HostileKey("helper_session"): False,
        HostileKey("secret_payload"): False,
    }
    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY) as caught:
        filter_indexable_bodies([row])
    assert "private attacker detail" not in str(caught.value)


def test_filter_drops_ineligible_and_keeps_visible_bodies() -> None:
    kept = filter_indexable_bodies(
        [
            {
                "message_id": "keep",
                "body": "hello",
                "transcript_visible": True,
                "content_removed": False,
                "private_session": False,
                "helper_session": False,
                "secret_payload": False,
            },
            {
                "message_id": "gone",
                "body": "secret hello",
                "transcript_visible": True,
                "content_removed": False,
                "private_session": False,
                "helper_session": False,
                "secret_payload": True,
            },
        ]
    )
    assert kept == (("keep", "hello"),)


def test_a_verdict_cannot_be_minted_without_the_evaluator_witness() -> None:
    """The authority guard: only the evaluator may declare a visibility verdict.

    SearchVisibilityV1 is a decision about whether a message body may appear in
    search, so a caller that can construct one directly can assert eligibility
    the evaluator never granted. The witness is a module-private sentinel and is
    the only thing standing between those two situations.

    It was unbound: deleting the check left the entire API suite green, because
    every legitimate path goes through evaluate_search_visibility and therefore
    passes the right witness. Only a forgery attempt can tell the difference.
    """
    from local_lm import search_visibility_v1 as module

    with pytest.raises(SearchVisibilityError, match=INVALID_VISIBILITY):
        module._visibility_from_evaluator(witness=object(), code="eligible")

    # And the real evaluator still works, so the guard refuses forgery rather
    # than refusing everything.
    verdict = evaluate_search_visibility(
        message_id="m" * 8,
        transcript_visible=True,
        content_removed=False,
        private_session=False,
        helper_session=False,
        secret_payload=False,
    )
    assert verdict.eligible is True
