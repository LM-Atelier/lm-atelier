from __future__ import annotations

import pytest

from local_lm.adapter_grammars import (
    COMPOSITION_UNDEFINED,
    GrammarOutcome,
    ReviewedGrammar,
    resolve_one,
    resolve_stack,
)
from local_lm.prompt_grammar import canonical_grammar_digest, normalize_grammar

ASSET = "a" * 64
OTHER_ASSET = "b" * 64
COMPILER = "visual-prompt-compiler-v1"
CEILING = 900

GRAMMAR = normalize_grammar(
    {
        "trigger": "TRIGGERWORD",
        "template": "TRIGGERWORD <shape>, <description>",
        "slots": [{"name": "shape", "required": True, "values": ["circle", "square"]}],
    },
    verified_values={"shape": frozenset({"circle"})},
)
DIGEST = canonical_grammar_digest(GRAMMAR)


def review(**changes: object) -> ReviewedGrammar:
    values: dict[str, object] = {
        "install_id": "install-1",
        "asset_sha256": ASSET,
        "grammar_sha256": DIGEST,
        "schema_version": 1,
        "compiler_version": COMPILER,
        "compiler_ceiling": CEILING,
        "fits": True,
        "reviewed": True,
        "grammar": GRAMMAR,
    }
    values.update(changes)
    return ReviewedGrammar(**values)  # type: ignore[arg-type]


def resolve(item: ReviewedGrammar | None, **changes: object):  # type: ignore[no-untyped-def]
    values: dict[str, object] = {
        "install_id": "install-1",
        "running_asset_sha256": ASSET,
        "compiler_version": COMPILER,
        "compiler_ceiling": CEILING,
        "recomputed_grammar_sha256": DIGEST,
    }
    values.update(changes)
    return resolve_one(item, **values)  # type: ignore[arg-type]


def test_no_grammar_is_the_ordinary_case_and_bars_nothing() -> None:
    """Most adapters have no grammar. They must behave exactly as before, and
    must not be excluded from automatic choice for lacking one."""

    resolution = resolve(None)
    assert resolution.outcome is GrammarOutcome.ABSENT
    assert not resolution.bars_automatic_choice
    assert resolution.grammar is None


def test_a_current_review_is_usable() -> None:
    resolution = resolve(review())
    assert resolution.outcome is GrammarOutcome.CURRENT
    assert resolution.grammar is GRAMMAR
    assert resolution.provenance["applied"] is True


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"running_asset_sha256": OTHER_ASSET}, "adapter file changed"),
        ({"recomputed_grammar_sha256": "c" * 64}, "no longer matches what was reviewed"),
        ({"compiler_version": "visual-prompt-compiler-v2"}, "prompt compiler changed"),
        ({"compiler_ceiling": 1200}, "prompt length limit changed"),
    ],
)
def test_anything_the_review_depended_on_moving_makes_it_stale(
    changes: dict[str, object], expected: str
) -> None:
    """Each of these lives in a subsystem with no reason to update this row,
    which is why the verdict is recomputed rather than stored."""

    resolution = resolve(review(), **changes)
    assert resolution.outcome is GrammarOutcome.STALE
    assert any(expected in reason for reason in resolution.reasons), resolution.reasons
    assert resolution.bars_automatic_choice
    assert resolution.provenance["applied"] is False


def test_an_unreviewed_or_reformatted_grammar_is_stale() -> None:
    assert resolve(review(reviewed=False)).outcome is GrammarOutcome.STALE
    assert resolve(review(schema_version=99)).outcome is GrammarOutcome.STALE


def test_incompatible_is_not_the_same_answer_as_stale() -> None:
    """One says the evidence expired, the other says it could never be used.
    Collapsing them would hide which, and they need different repairs."""

    resolution = resolve(review(fits=False))
    assert resolution.outcome is GrammarOutcome.INCOMPATIBLE
    assert resolution.bars_automatic_choice
    assert "cannot fit" in resolution.reasons[0]


def test_provenance_carries_digests_and_reasons_but_never_text() -> None:
    provenance = resolve(review(fits=False)).provenance
    assert provenance["grammar_sha256"] == DIGEST
    serialized = repr(provenance)
    assert "TRIGGERWORD" not in serialized
    assert "circle" not in serialized


def test_two_grammars_in_one_stack_have_no_defined_composition() -> None:
    """Two templates have no agreed ordering, no rule for conflicting slots and
    no shared length accounting, so merging them would produce a prompt neither
    grammar describes."""

    both = [resolve(review()), resolve(review(install_id="install-2"), install_id="install-2")]
    automatic = resolve_stack(both, automatic=True)
    assert automatic.grammar is None
    assert automatic.refuse_automatic
    assert COMPOSITION_UNDEFINED in automatic.warnings

    explicit = resolve_stack(both, automatic=False)
    assert explicit.grammar is None
    assert COMPOSITION_UNDEFINED in explicit.warnings


def test_an_explicit_choice_still_runs_but_is_told_the_truth() -> None:
    """Refusing to choose an adapter is a different act from overriding a choice
    already made. An explicit stack runs; it simply gets no grammar and a
    reason, rather than a silent claim that one was applied."""

    stale = [resolve(review(), running_asset_sha256=OTHER_ASSET)]

    automatic = resolve_stack(stale, automatic=True)
    assert automatic.refuse_automatic

    explicit = resolve_stack(stale, automatic=False)
    assert not explicit.refuse_automatic
    assert explicit.grammar is None
    assert explicit.warnings
    assert explicit.provenance[0]["applied"] is False


def test_one_current_grammar_in_a_stack_is_passed_through() -> None:
    mixed = [resolve(review()), resolve(None, install_id="install-2")]
    assert resolve_stack(mixed, automatic=True).grammar is GRAMMAR


def test_a_review_whose_grammar_cannot_be_rebuilt_is_stale() -> None:
    """Without this it passed as current carrying no grammar, which reads
    downstream as "there was nothing to apply" rather than as a failure."""

    resolution = resolve(review(grammar=None))
    assert resolution.outcome is GrammarOutcome.STALE
    assert any("cannot read" in reason for reason in resolution.reasons)
    assert resolution.bars_automatic_choice


def test_composition_does_not_block_a_choice_already_made() -> None:
    """An explicit stack has no automatic selection to refuse, so refusing one
    would read as blocking the user's own choice."""

    both = [resolve(review()), resolve(review(install_id="install-2"), install_id="install-2")]
    assert resolve_stack(both, automatic=True).refuse_automatic
    assert not resolve_stack(both, automatic=False).refuse_automatic
