from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.adapter_grammar_review import review_adapter_grammar
from local_lm.db import Base
from local_lm.models import AdapterPromptGrammar, ModelAssetInstall
from local_lm.prompt_grammar import PromptGrammarError
from local_lm.visual_prompt_compiler import COMPILER_VERSION, MAX_COMPILED_PROMPT_CHARS

BOUND = "a" * 64
GUIDANCE = "Write a natural description of the scene."
DOCUMENT = (
    "TRIGGERWORD <shape>, <description>\n\n"
    "shape may be circle or square.\n\n"
    + GUIDANCE
    + "\n\nExample: TRIGGERWORD circle, a wide shot of a lit room."
)
PAYLOAD = {
    "trigger": "TRIGGERWORD",
    "template": "TRIGGERWORD <shape>, <description>",
    "description_guidance": GUIDANCE,
    "slots": [{"name": "shape", "required": True, "values": ["circle", "square"]}],
}


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def _install(session: Session) -> str:
    install = ModelAssetInstall(name="adapter", kind="lora", local_path="adapter.safetensors")
    session.add(install)
    session.flush()
    return install.id


def _review(session: Session, install_id: str, **changes: object) -> AdapterPromptGrammar:
    values: dict[str, object] = {
        "model_asset_install_id": install_id,
        "asset_sha256": BOUND,
        "source_identity": "sidecar distributed with the adapter",
        "source_text": DOCUMENT,
        "payload": PAYLOAD,
        "approve_prose": (),
        "verified_values": {"shape": frozenset({"circle"})},
    }
    values.update(changes)
    return review_adapter_grammar(session, **values)  # type: ignore[arg-type]


def test_a_review_creates_one_row(session: Session) -> None:
    install = _install(session)
    row = _review(session, install)
    assert row.model_asset_install_id == install
    assert row.reviewed_at is not None
    assert row.compiler_version == COMPILER_VERSION
    assert row.compiler_ceiling == MAX_COMPILED_PROMPT_CHARS
    assert row.fits is True
    assert session.query(AdapterPromptGrammar).count() == 1


def test_prose_is_approved_only_if_it_was_actually_in_the_document(session: Session) -> None:
    """Approving text is meaningless if that text was never read. The digest is
    accepted only when the exact wording appears in the source being reviewed."""

    install = _install(session)
    row = _review(session, install, approve_prose=(GUIDANCE, "text nobody wrote"))
    assert len(row.approved_prose_json) == 1

    from local_lm.prompt_grammar import prose_digest

    assert row.approved_prose_json == [prose_digest(GUIDANCE)]


def test_re_reviewing_updates_the_same_row(session: Session) -> None:
    """One grammar per install. A second row would raise the question of which
    one a rewriter believed."""

    install = _install(session)
    first = _review(session, install)
    first_id, first_digest = first.id, first.grammar_sha256
    second = _review(session, install, verified_values={"shape": frozenset({"circle", "square"})})
    assert first_id == second.id
    assert session.query(AdapterPromptGrammar).count() == 1
    assert second.verified_values_json == {"shape": ["circle", "square"]}
    assert second.grammar_sha256 != first_digest, (
        "widening what was verified is a different grammar to act on"
    )


def test_a_failed_re_review_does_not_clobber_the_earlier_good_one(session: Session) -> None:
    install = _install(session)
    good = _review(session, install)
    with pytest.raises(PromptGrammarError):
        _review(session, install, payload={**PAYLOAD, "trigger": "two words"})
    still_good = session.query(AdapterPromptGrammar).one()
    assert still_good.grammar_sha256 == good.grammar_sha256
    assert still_good.fits is True


def test_the_fit_check_is_anchored_to_the_real_consumer(session: Session) -> None:
    """The compiler's own current version and floor are read here, not accepted
    as an argument - anchoring the check to a caller-supplied number would let
    review and consumer drift apart."""

    install = _install(session)
    oversized_description = "x" * (MAX_COMPILED_PROMPT_CHARS + 500)
    row = _review(
        session,
        install,
        payload={
            **PAYLOAD,
            "description_guidance": oversized_description[:2000],
        },
    )
    # The guidance being long does not itself make the grammar not fit - only
    # its overhead (template plus widest slot values) counts against the
    # consumer's minimum-description floor.
    assert row.compiler_ceiling == MAX_COMPILED_PROMPT_CHARS


def test_the_stored_grammar_matches_what_a_rewriter_would_receive(session: Session) -> None:
    install = _install(session)
    row = _review(session, install, approve_prose=(GUIDANCE,))
    from local_lm.prompt_grammar import canonical_grammar_digest, normalize_grammar
    from local_lm.prompt_grammar import prose_digest as digest_of

    rebuilt = normalize_grammar(
        PAYLOAD,
        approved_prose=frozenset({digest_of(GUIDANCE)}),
        verified_values={"shape": frozenset({"circle"})},
    )
    assert row.grammar_sha256 == canonical_grammar_digest(rebuilt)
    assert len(row.grammar_sha256) == 64


def test_an_unreadable_grammar_is_refused_before_anything_is_written(session: Session) -> None:
    install = _install(session)
    with pytest.raises(PromptGrammarError):
        _review(session, install, payload={**PAYLOAD, "trigger": "two words"})
    assert session.query(AdapterPromptGrammar).count() == 0
