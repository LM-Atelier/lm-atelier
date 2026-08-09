from __future__ import annotations

import pytest

from local_lm.references import (
    MAX_REFERENCES_PER_TURN,
    MentionSource,
    ReferenceError,
    ReferenceKind,
    parse_kind,
    parse_reference_requests,
    slugify_mention,
    valid_mention_slug,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Ada Lovelace", "ada-lovelace"),
        ("  Ada   Lovelace  ", "ada-lovelace"),
        ("ADA_LOVELACE", "ada-lovelace"),
        ("Ada.Lovelace", "ada-lovelace"),
        ("Café", "cafe"),
        ("--Ada--", "ada"),
    ],
)
def test_a_display_name_becomes_one_predictable_mention(name: str, expected: str) -> None:
    assert slugify_mention(name) == expected
    assert valid_mention_slug(slugify_mention(name))


def test_every_separator_reaches_the_same_canonical_mention() -> None:
    """If "ada_lovelace" and "ada-lovelace" were both valid, two subjects could
    occupy what a person reads as one mention."""

    forms = ["Ada Lovelace", "ada_lovelace", "ada-lovelace", "Ada.Lovelace", "ada--lovelace"]
    assert len({slugify_mention(item) for item in forms}) == 1


def test_visually_identical_names_do_not_become_two_subjects() -> None:
    """Composed and decomposed forms of the same name look identical on screen.
    If they slugged differently, two subjects could occupy what a person reads
    as one mention, and picking between them would be a coin toss."""

    composed = "René"  # single code point
    decomposed = "René"  # e + combining acute
    assert composed != decomposed
    assert slugify_mention(composed) == slugify_mention(decomposed)


def test_case_is_folded_rather_than_lowered() -> None:
    """Lowercasing is not the same operation in every script, and a mention that
    resolves differently depending on how it was typed is a way to reach the
    wrong subject."""

    assert slugify_mention("STRASSE") == slugify_mention("strasse")


def test_a_name_with_no_usable_characters_is_refused() -> None:
    """Silently inventing a mention for it would create a subject nobody can
    address, which reads as the feature being broken rather than the name."""

    for unusable in ("", "   ", "!!!", "中文"):
        with pytest.raises(ReferenceError):
            slugify_mention(unusable)


def test_the_kind_vocabulary_is_closed() -> None:
    """A workflow declares which kinds it can condition on, so an open
    vocabulary would mean a subject whose compatibility can never be decided."""

    assert parse_kind("person") is ReferenceKind.PERSON
    assert parse_kind(" Person ") is ReferenceKind.PERSON
    assert parse_kind(ReferenceKind.STYLE) is ReferenceKind.STYLE

    with pytest.raises(ReferenceError) as caught:
        parse_kind("spaceship")
    # The refusal names what is permitted, so a caller is not left guessing.
    assert "person" in str(caught.value)


def test_a_turn_carries_references_as_data_not_as_text() -> None:
    requests = parse_reference_requests(
        [
            {"reference_subject_id": "ref_1", "role": "subject"},
            {
                "reference_subject_id": "ref_2",
                "role": "wardrobe",
                "selected_asset_ids": ["a", "b", "a"],
                "strength": 0.8,
                "source": "picker",
            },
        ]
    )
    assert len(requests) == 2
    assert requests[0].source is MentionSource.MENTION, "mention is the default"
    assert requests[1].selected_asset_ids == ("a", "b"), "duplicates collapse, order kept"
    assert requests[1].strength == 0.8


def test_the_same_subject_may_fill_two_roles_but_not_one_twice() -> None:
    twice_over = [
        {"reference_subject_id": "ref_1", "role": "subject"},
        {"reference_subject_id": "ref_1", "role": "wardrobe"},
    ]
    assert len(parse_reference_requests(twice_over)) == 2

    with pytest.raises(ReferenceError):
        parse_reference_requests(
            [
                {"reference_subject_id": "ref_1", "role": "subject"},
                {"reference_subject_id": "ref_1", "role": "subject"},
            ]
        )


def test_how_a_reference_arrived_is_recorded() -> None:
    """A typed mention and something inherited from context differ in how much
    the user actually asserted, and a later question about why an image contains
    someone has to be able to tell them apart."""

    for value, expected in (
        ("mention", MentionSource.MENTION),
        ("picker", MentionSource.PICKER),
        ("inherited_context", MentionSource.INHERITED_CONTEXT),
    ):
        parsed = parse_reference_requests([{"reference_subject_id": "ref_1", "source": value}])
        assert parsed[0].source is expected

    with pytest.raises(ReferenceError):
        parse_reference_requests([{"reference_subject_id": "ref_1", "source": "overheard"}])


@pytest.mark.parametrize(
    "payload",
    [
        "ref_1",
        [{"role": "subject"}],
        [{"reference_subject_id": ""}],
        [{"reference_subject_id": "ref_1", "selected_asset_ids": "a"}],
        [{"reference_subject_id": "ref_1", "strength": "0.8"}],
        [{"reference_subject_id": "ref_1", "strength": True}],
        [{"reference_subject_id": "ref_1", "strength": 5}],
        [{"reference_subject_id": "ref_1", "strength": -1}],
        ["ref_1"],
    ],
)
def test_an_unreadable_request_is_refused_rather_than_guessed_at(payload: object) -> None:
    """The fallback if this is unreliable is recovering References by reading
    the prompt for @name, and that path silently binds whoever the text most
    resembles."""

    with pytest.raises(ReferenceError):
        parse_reference_requests(payload)


def test_no_references_is_not_an_error() -> None:
    assert parse_reference_requests(None) == ()
    assert parse_reference_requests([]) == ()


def test_a_turn_cannot_carry_unbounded_references() -> None:
    too_many = [
        {"reference_subject_id": f"ref_{index}"} for index in range(MAX_REFERENCES_PER_TURN + 1)
    ]
    with pytest.raises(ReferenceError):
        parse_reference_requests(too_many)
