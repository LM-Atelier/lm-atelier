"""Answer, for one run's adapter stack, which grammar may be used.

This is the only place that joins the three records the decision needs: what
this machine reviewed, which file the activation actually bound, and what the
compiler can currently emit. Keeping the join here leaves `adapter_grammars`
free of a session, which is what stops a caller from reaching a chat default or
a mutable manifest by accident rather than by discipline.

Every path fails closed. An adapter whose bound digest cannot be established
does not fall back to the manifest and does not get its grammar applied.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapter_frozen_identity import frozen_adapter_digest
from .adapter_grammars import (
    GrammarOutcome,
    GrammarResolution,
    ReviewedGrammar,
    StackGrammar,
    resolve_one,
    resolve_stack,
)
from .models import AdapterPromptGrammar
from .prompt_grammar import PromptGrammarError, canonical_grammar_digest, normalize_grammar

# An adapter the activation never bound. Named separately from staleness because
# it is not a review that expired - there is nothing to check the review against.
UNBOUND = "the run did not bind a verified file for this adapter"


def _reviewed(record: AdapterPromptGrammar) -> tuple[ReviewedGrammar, str | None]:
    """Rebuild a review from its row, and recompute what it should digest to.

    The stored grammar is re-normalized under the overlays that were reviewed
    with it, so a change to the schema, the approved prose or the verified
    values shows up as a digest that no longer matches. Refusing to normalize at
    all is treated the same way, because a grammar this build cannot read is one
    it certainly must not act on.
    """

    verified = {
        slot: frozenset(values)
        for slot, values in (record.verified_values_json or {}).items()
        if isinstance(values, list)
    }
    try:
        grammar = normalize_grammar(
            record.grammar_json,
            approved_prose=frozenset(record.approved_prose_json or []),
            verified_values=verified,
        )
    except PromptGrammarError:
        grammar = None
    review = ReviewedGrammar(
        install_id=record.model_asset_install_id,
        asset_sha256=record.asset_sha256,
        grammar_sha256=record.grammar_sha256,
        schema_version=record.schema_version,
        compiler_version=record.compiler_version,
        compiler_ceiling=record.compiler_ceiling,
        fits=record.fits,
        reviewed=record.reviewed_at is not None,
        grammar=grammar,
    )
    return review, (canonical_grammar_digest(grammar) if grammar else None)


def stack_grammar(
    session: Session,
    *,
    workflow_activation_id: str | None,
    lora_settings: list[dict[str, Any]],
    compiler_version: str,
    compiler_ceiling: int,
    automatic: bool,
) -> StackGrammar:
    """The grammar decision for a whole resolved stack."""

    install_ids = [
        str(item["asset_id"])
        for item in lora_settings
        if isinstance(item, dict) and item.get("asset_id") and item.get("enabled", True)
    ]
    if not install_ids:
        return StackGrammar(None, refuse_automatic=False, warnings=(), provenance=())

    records = {
        record.model_asset_install_id: record
        for record in session.scalars(
            select(AdapterPromptGrammar).where(
                AdapterPromptGrammar.model_asset_install_id.in_(install_ids)
            )
        ).all()
    }

    resolutions: list[GrammarResolution] = []
    for install_id in install_ids:
        record = records.get(install_id)
        if record is None:
            # No grammar was ever recorded. The ordinary case, and it bars
            # nothing - this adapter behaves exactly as it always has.
            resolutions.append(GrammarResolution(GrammarOutcome.ABSENT, install_id=install_id))
            continue
        bound = (
            frozen_adapter_digest(
                session,
                workflow_activation_id=workflow_activation_id,
                model_asset_install_id=install_id,
            )
            if workflow_activation_id
            else None
        )
        if bound is None:
            # A review exists but nothing here can say which file it describes.
            # Reading the manifest would answer, and would be the one answer a
            # review must never be checked against.
            resolutions.append(
                GrammarResolution(
                    GrammarOutcome.STALE,
                    install_id=install_id,
                    reasons=(UNBOUND,),
                    provenance={
                        "install_id": install_id,
                        "asset_sha256": record.asset_sha256,
                        "grammar_sha256": record.grammar_sha256,
                        "schema_version": record.schema_version,
                        "applied": False,
                        "reasons": [UNBOUND],
                    },
                )
            )
            continue
        review, recomputed = _reviewed(record)
        resolutions.append(
            resolve_one(
                review,
                install_id=install_id,
                running_asset_sha256=bound,
                compiler_version=compiler_version,
                compiler_ceiling=compiler_ceiling,
                recomputed_grammar_sha256=recomputed,
            )
        )
    return resolve_stack(resolutions, automatic=automatic)
