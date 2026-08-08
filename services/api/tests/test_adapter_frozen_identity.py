from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.adapter_frozen_identity import frozen_adapter_digest, frozen_adapter_digests
from local_lm.db import Base
from local_lm.models import WorkflowDependencyBinding

ACTIVATION = "wfact_one"
REVISION = "wfrev_one"
BOUND = "a" * 64
OTHER = "b" * 64


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


_written = 0


def _binding(session: Session, install_id: str | None, identity: object, **changes: object) -> None:
    global _written
    _written += 1
    values: dict[str, object] = {
        "workflow_revision_id": REVISION,
        "workflow_activation_id": ACTIVATION,
        "workflow_dependency_slot_id": f"slot-{_written}",
        "requirement_key": f"lora-{_written}",
        "model_asset_install_id": install_id,
        "resource_identity_json": identity,
        "resource_identity_sha256": BOUND,
    }
    values.update(changes)
    session.add(WorkflowDependencyBinding(**values))  # type: ignore[arg-type]
    session.flush()


def _digest(session: Session, install_id: str = "asset-1") -> str | None:
    return frozen_adapter_digest(
        session, workflow_activation_id=ACTIVATION, model_asset_install_id=install_id
    )


def test_the_bound_digest_is_the_one_the_activation_verified(session: Session) -> None:
    _binding(session, "asset-1", {"sha256": BOUND, "runtime_reference": "adapter.safetensors"})
    assert _digest(session) == BOUND


def test_an_adapter_the_activation_never_bound_has_no_digest(session: Session) -> None:
    """An adapter chosen from turn settings need not be a workflow dependency at
    all. Answering anyway would mean answering from the mutable manifest, which
    is the one source a review must never be checked against."""

    _binding(session, "asset-other", {"sha256": BOUND})
    assert _digest(session) is None


def test_two_bindings_for_one_adapter_answer_nothing(session: Session) -> None:
    """The activation used it for two requirements. Nothing here can say which
    file a grammar was reviewed against, and either choice is a guess wearing a
    digest."""

    _binding(session, "asset-1", {"sha256": BOUND})
    _binding(session, "asset-1", {"sha256": OTHER})
    assert _digest(session) is None


@pytest.mark.parametrize(
    "identity",
    [
        {},
        {"sha256": None},
        {"sha256": 12345},
        {"sha256": "tooshort"},
        {"sha256": "z" * 64},
        {"runtime_reference": "adapter.safetensors"},
        "not-an-object",
    ],
)
def test_an_identity_without_a_usable_digest_answers_nothing(
    session: Session, identity: object
) -> None:
    _binding(session, "asset-1", identity)
    assert _digest(session) is None


def test_a_digest_is_normalized_before_it_is_compared(session: Session) -> None:
    _binding(session, "asset-1", {"sha256": f"  {BOUND.upper()}  "})
    assert _digest(session) == BOUND


def test_an_activation_from_another_run_does_not_answer(session: Session) -> None:
    _binding(session, "asset-1", {"sha256": BOUND}, workflow_activation_id="wfact_two")
    assert _digest(session) is None


def test_every_requested_adapter_appears_even_when_unanswered(session: Session) -> None:
    """A caller iterating the result must not be able to skip an adapter by
    forgetting it had no answer."""

    _binding(session, "asset-1", {"sha256": BOUND})
    answers = frozen_adapter_digests(
        session,
        workflow_activation_id=ACTIVATION,
        model_asset_install_ids=["asset-1", "asset-2", "asset-1"],
    )
    assert answers == {"asset-1": BOUND, "asset-2": None}


def test_a_missing_activation_or_install_answers_nothing(session: Session) -> None:
    _binding(session, "asset-1", {"sha256": BOUND})
    assert (
        frozen_adapter_digest(session, workflow_activation_id="", model_asset_install_id="asset-1")
        is None
    )
    assert (
        frozen_adapter_digest(session, workflow_activation_id=ACTIVATION, model_asset_install_id="")
        is None
    )
