from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.adapter_grammar_lookup import UNBOUND, stack_grammar
from local_lm.db import Base
from local_lm.models import AdapterPromptGrammar, ModelAssetInstall, WorkflowDependencyBinding
from local_lm.prompt_grammar import canonical_grammar_digest, normalize_grammar

ACTIVATION = "wfact_one"
BOUND = "a" * 64
COMPILER = "visual-prompt-compiler-v1"
CEILING = 900

PAYLOAD = {
    "trigger": "TRIGGERWORD",
    "template": "TRIGGERWORD <shape>, <description>",
    "slots": [{"name": "shape", "required": True, "values": ["circle", "square"]}],
}
OVERLAY = {"shape": ["circle"]}
DIGEST = canonical_grammar_digest(
    normalize_grammar(PAYLOAD, verified_values={"shape": frozenset({"circle"})})
)


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def _install(session: Session, name: str = "adapter") -> str:
    install = ModelAssetInstall(name=name, kind="lora", local_path=f"{name}.safetensors")
    session.add(install)
    session.flush()
    return install.id


_slots = 0


def _bind(session: Session, install_id: str, digest: str = BOUND) -> None:
    global _slots
    _slots += 1
    session.add(
        WorkflowDependencyBinding(
            workflow_revision_id="wfrev_one",
            workflow_activation_id=ACTIVATION,
            workflow_dependency_slot_id=f"slot-{_slots}",
            requirement_key=f"lora-{_slots}",
            model_asset_install_id=install_id,
            resource_identity_json={"sha256": digest},
            resource_identity_sha256=digest,
        )
    )
    session.flush()


def _grammar(session: Session, install_id: str, **changes: object) -> None:
    values: dict[str, object] = {
        "model_asset_install_id": install_id,
        "asset_sha256": BOUND,
        "source_identity": "sidecar",
        "source_sha256": "c" * 64,
        "schema_version": 1,
        "grammar_json": PAYLOAD,
        "grammar_sha256": DIGEST,
        "approved_prose_json": [],
        "verified_values_json": OVERLAY,
        "compiler_version": COMPILER,
        "compiler_ceiling": CEILING,
        "fits": True,
        "reviewed_at": datetime.now(UTC),
    }
    values.update(changes)
    session.add(AdapterPromptGrammar(**values))  # type: ignore[arg-type]
    session.flush()


def _resolve(
    session: Session, install_ids: list[str], *, automatic: bool = True, **changes: object
):  # type: ignore[no-untyped-def]
    values: dict[str, object] = {
        "workflow_activation_id": ACTIVATION,
        "lora_settings": [{"asset_id": item, "enabled": True} for item in install_ids],
        "compiler_version": COMPILER,
        "compiler_ceiling": CEILING,
        "automatic": automatic,
    }
    values.update(changes)
    return stack_grammar(session, **values)  # type: ignore[arg-type]


def test_a_reviewed_and_bound_adapter_supplies_its_grammar(session: Session) -> None:
    install = _install(session)
    _bind(session, install)
    _grammar(session, install)
    answer = _resolve(session, [install])
    assert answer.grammar is not None
    assert answer.grammar.trigger == "TRIGGERWORD"
    assert not answer.refuse_automatic


def test_an_unbound_adapter_never_falls_back_to_the_manifest(session: Session) -> None:
    """A review exists but nothing can say which file it describes. The manifest
    would answer, and it is the one source a review must never be checked
    against, so the answer is refusal."""

    install = _install(session)
    _grammar(session, install)  # reviewed, but the activation bound nothing
    answer = _resolve(session, [install])
    assert answer.grammar is None
    assert UNBOUND in answer.warnings
    assert answer.refuse_automatic
    assert answer.provenance[0]["applied"] is False


def test_an_adapter_with_no_grammar_bars_nothing(session: Session) -> None:
    install = _install(session)
    _bind(session, install)
    answer = _resolve(session, [install])
    assert answer.grammar is None
    assert not answer.refuse_automatic
    assert answer.warnings == ()


def test_a_changed_file_makes_the_review_stale(session: Session) -> None:
    install = _install(session)
    _bind(session, install, digest="b" * 64)
    _grammar(session, install)  # reviewed against BOUND
    answer = _resolve(session, [install])
    assert answer.grammar is None
    assert any("adapter file changed" in item for item in answer.warnings)


def test_widening_the_overlay_without_review_is_caught(session: Session) -> None:
    """The overlays change emitted text, so they are inside the digest. Widening
    verification without re-reviewing must not silently widen the prompt."""

    install = _install(session)
    _bind(session, install)
    _grammar(session, install, verified_values_json={"shape": ["circle", "square"]})
    answer = _resolve(session, [install])
    assert answer.grammar is None
    assert any("no longer matches what was reviewed" in item for item in answer.warnings)


def test_an_unreadable_grammar_is_refused_rather_than_skipped(session: Session) -> None:
    install = _install(session)
    _bind(session, install)
    _grammar(session, install, grammar_json={"trigger": "two words"})
    answer = _resolve(session, [install])
    assert answer.grammar is None
    assert answer.refuse_automatic


def test_two_grammars_refuse_the_automatic_stack(session: Session) -> None:
    first, second = _install(session, "one"), _install(session, "two")
    for install in (first, second):
        _bind(session, install)
        _grammar(session, install)
    automatic = _resolve(session, [first, second])
    assert automatic.grammar is None
    assert automatic.refuse_automatic

    explicit = _resolve(session, [first, second], automatic=False)
    assert explicit.grammar is None
    assert not explicit.refuse_automatic


def test_a_disabled_adapter_is_not_consulted(session: Session) -> None:
    install = _install(session)
    _grammar(session, install)
    answer = stack_grammar(
        session,
        workflow_activation_id=ACTIVATION,
        lora_settings=[{"asset_id": install, "enabled": False}],
        compiler_version=COMPILER,
        compiler_ceiling=CEILING,
        automatic=True,
    )
    assert answer.grammar is None
    assert not answer.refuse_automatic


def test_a_run_without_an_activation_applies_no_grammar(session: Session) -> None:
    install = _install(session)
    _bind(session, install)
    _grammar(session, install)
    answer = _resolve(session, [install], workflow_activation_id=None)
    assert answer.grammar is None
    assert UNBOUND in answer.warnings
