"""Whether a recorded grammar still describes the adapter that is about to run.

A grammar is reviewed once and consulted every turn, and everything it depends
on lives somewhere else: the adapter file, the grammar schema, the compiler's
own limits, and the reviewer's approvals. None of those subsystems has any
reason to come back and update a row here, so no verdict is stored. What is
stored is what was reviewed and what it was reviewed against, and the answer is
recomputed against the world on read - the same rule `workflow_attestation`
follows, for the same reason.

Two failures are kept apart throughout. *Stale* means the evidence no longer
describes the world. *Incompatible* means it never could be used, because a
complete prompt in that shape cannot fit what the consumer emits. Both bar an
adapter from being chosen automatically, and neither may silently claim a
grammar was applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .prompt_grammar import SCHEMA_VERSION, PromptGrammar

# The identity of an adapter as it will actually load. Taken from the frozen
# selection, never from a chat default, and never from a mutable manifest when
# the run recorded a file digest of its own.
COMPOSITION_UNDEFINED = "prompt_grammar_composition_undefined"


class GrammarOutcome(StrEnum):
    CURRENT = "current"
    ABSENT = "absent"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class ReviewedGrammar:
    """One recorded review, and the world it was recorded against."""

    install_id: str
    asset_sha256: str
    grammar_sha256: str
    schema_version: int
    compiler_version: str
    compiler_ceiling: int
    fits: bool
    reviewed: bool
    grammar: PromptGrammar | None = None


@dataclass(frozen=True)
class GrammarResolution:
    outcome: GrammarOutcome
    install_id: str | None = None
    grammar: PromptGrammar | None = None
    reasons: tuple[str, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.outcome is GrammarOutcome.CURRENT

    @property
    def bars_automatic_choice(self) -> bool:
        """Whether this adapter may still be chosen automatically.

        Absent is not a bar: an adapter with no grammar is the ordinary case and
        behaves as it always has. Recorded-but-unusable is a bar, because
        something was reviewed and no longer holds, and applying an adapter
        without the grammar it needs is worse than leaving it off.
        """

        return self.outcome in (GrammarOutcome.STALE, GrammarOutcome.INCOMPATIBLE)


def staleness_reasons(
    review: ReviewedGrammar,
    *,
    running_asset_sha256: str,
    compiler_version: str,
    compiler_ceiling: int,
    recomputed_grammar_sha256: str | None = None,
) -> tuple[str, ...]:
    """Why a recorded review no longer describes what is about to run."""

    reasons: list[str] = []
    if not review.reviewed:
        reasons.append("this grammar has not been reviewed on this machine")
    # A review whose grammar this build can no longer rebuild is not a review of
    # anything. Without this it passed as current carrying no grammar, which
    # reads downstream as "there was nothing to apply" rather than as a failure.
    if review.grammar is None:
        reasons.append("this build cannot read the reviewed grammar")
    if review.asset_sha256 != running_asset_sha256:
        reasons.append("the adapter file changed since the grammar was reviewed")
    if review.schema_version != SCHEMA_VERSION:
        reasons.append("the grammar format changed since it was reviewed")
    if recomputed_grammar_sha256 is not None and recomputed_grammar_sha256 != review.grammar_sha256:
        reasons.append("the grammar no longer matches what was reviewed")
    # A compiler change moves the ceiling the fit was judged against. Reporting
    # it as stale is what keeps the failure at review time instead of letting a
    # turn overflow.
    if review.compiler_version != compiler_version:
        reasons.append("the prompt compiler changed since the grammar was reviewed")
    elif review.compiler_ceiling != compiler_ceiling:
        reasons.append("the prompt length limit changed since the grammar was reviewed")
    return tuple(reasons)


def resolve_one(
    review: ReviewedGrammar | None,
    *,
    install_id: str,
    running_asset_sha256: str,
    compiler_version: str,
    compiler_ceiling: int,
    recomputed_grammar_sha256: str | None = None,
) -> GrammarResolution:
    """Resolve one adapter's grammar against the world as it is now."""

    if review is None:
        return GrammarResolution(GrammarOutcome.ABSENT, install_id=install_id)

    def provenance(applied: bool, reasons: tuple[str, ...]) -> dict[str, object]:
        # Digests and named reasons only. The grammar's text never travels.
        return {
            "install_id": install_id,
            "asset_sha256": review.asset_sha256,
            "grammar_sha256": review.grammar_sha256,
            "schema_version": review.schema_version,
            "applied": applied,
            "reasons": list(reasons),
        }

    reasons = staleness_reasons(
        review,
        running_asset_sha256=running_asset_sha256,
        compiler_version=compiler_version,
        compiler_ceiling=compiler_ceiling,
        recomputed_grammar_sha256=recomputed_grammar_sha256,
    )
    if reasons:
        return GrammarResolution(
            GrammarOutcome.STALE, install_id, None, reasons, provenance(False, reasons)
        )
    if not review.fits:
        why = ("a complete prompt in this adapter's shape cannot fit the prompt length limit",)
        return GrammarResolution(
            GrammarOutcome.INCOMPATIBLE, install_id, None, why, provenance(False, why)
        )
    return GrammarResolution(
        GrammarOutcome.CURRENT, install_id, review.grammar, (), provenance(True, ())
    )


@dataclass(frozen=True)
class StackGrammar:
    """What the compiler should be told about a whole selected stack."""

    grammar: PromptGrammar | None
    refuse_automatic: bool
    warnings: tuple[str, ...]
    provenance: tuple[dict[str, object], ...]


def resolve_stack(resolutions: list[GrammarResolution], *, automatic: bool) -> StackGrammar:
    """Combine per-adapter answers into one decision for the run.

    Composition is deliberately undefined. Two adapters each carrying a template
    have no agreed ordering, no rule for conflicting slots, and no shared
    accounting against the length limit, so inventing a merge here would produce
    a prompt neither grammar describes. An automatic stack is refused outright;
    an explicit one is honoured and simply told nothing, because refusing to
    choose an adapter is very different from overriding a choice already made.
    """

    warnings = tuple(reason for item in resolutions for reason in item.reasons)
    provenance = tuple(item.provenance for item in resolutions if item.provenance)
    usable = [item for item in resolutions if item.usable]

    if len(usable) > 1:
        # An explicit stack is honoured and simply told nothing. There is no
        # automatic selection to refuse, and refusing one anyway would read as
        # blocking a choice the user already made.
        return StackGrammar(
            None,
            refuse_automatic=automatic,
            warnings=warnings + (COMPOSITION_UNDEFINED,),
            provenance=provenance,
        )
    if automatic and any(item.bars_automatic_choice for item in resolutions):
        return StackGrammar(None, True, warnings, provenance)
    return StackGrammar(
        usable[0].grammar if usable else None,
        refuse_automatic=False,
        warnings=warnings,
        provenance=provenance,
    )
