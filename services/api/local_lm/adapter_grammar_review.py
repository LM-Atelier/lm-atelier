"""Turn a published grammar and a reviewer's approvals into a stored review.

This is the only place that writes `AdapterPromptGrammar`. Everywhere else in
the subsystem reads what was written here and recomputes whether it still applies;
nothing downstream may create or widen a review, which is what keeps a
generation outcome from ever promoting itself into evidence.

Approval is content-bound rather than a flag, and that discipline starts here:
a reviewer approves prose by supplying the exact text, and it is accepted only
if that text is actually present in the source document being reviewed. There
is no way to approve text that was never read.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import AdapterPromptGrammar
from .prompt_grammar import (
    canonical_grammar_digest,
    grammar_fits,
    normalize_grammar,
    prose_digest,
)
from .visual_prompt_compiler import (
    COMPILER_VERSION,
    MAX_COMPILED_PROMPT_CHARS,
    MIN_COMPILED_PROMPT_CHARS,
)


def _source_digest(source_text: str) -> str:
    return hashlib.sha256(source_text.strip().encode("utf-8")).hexdigest()


def review_adapter_grammar(
    session: Session,
    *,
    model_asset_install_id: str,
    asset_sha256: str,
    source_identity: str,
    source_text: str,
    payload: object,
    approve_prose: tuple[str, ...] = (),
    verified_values: Mapping[str, frozenset[str]] | None = None,
) -> AdapterPromptGrammar:
    """Record one review, replacing any earlier review of the same install.

    Raises `PromptGrammarError` for anything the grammar itself gets wrong, and
    raises before touching the database, so a bad re-review cannot clobber a
    good earlier one. The compiler's own current version and floor are read
    here rather than accepted as arguments, because the fit check exists to
    protect that specific consumer - anchoring it to a caller-supplied number
    would let review and consumer drift apart.
    """

    # Approving text is only meaningful if that exact text was in the document
    # being reviewed. Otherwise "approved" would mean nothing more than
    # "someone typed something".
    approved = frozenset(
        prose_digest(text) for text in approve_prose if text.strip() and text.strip() in source_text
    )

    grammar = normalize_grammar(
        payload, approved_prose=approved, verified_values=verified_values or {}
    )
    grammar_sha256 = canonical_grammar_digest(grammar)
    fits = grammar_fits(
        grammar, MAX_COMPILED_PROMPT_CHARS, minimum_description=MIN_COMPILED_PROMPT_CHARS
    )

    existing = session.scalar(
        select(AdapterPromptGrammar).where(
            AdapterPromptGrammar.model_asset_install_id == model_asset_install_id
        )
    )
    row = existing or AdapterPromptGrammar(model_asset_install_id=model_asset_install_id)
    row.asset_sha256 = asset_sha256.strip().lower()
    row.source_identity = source_identity
    row.source_sha256 = _source_digest(source_text)
    row.schema_version = grammar.schema_version
    row.grammar_json = payload if isinstance(payload, dict) else {}
    row.grammar_sha256 = grammar_sha256
    row.approved_prose_json = sorted(approved)
    row.verified_values_json = {
        slot: sorted(values) for slot, values in (verified_values or {}).items()
    }
    row.compiler_version = COMPILER_VERSION
    row.compiler_ceiling = MAX_COMPILED_PROMPT_CHARS
    row.fits = fits
    row.reviewed_at = datetime.now(UTC)
    if existing is None:
        session.add(row)
    session.flush()
    return row
