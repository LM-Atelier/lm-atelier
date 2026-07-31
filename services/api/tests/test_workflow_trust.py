"""Trust derived from rebuilding, not from provenance.

Import forces `trusted=False` and execution refuses untrusted revisions, so an
imported setup arrives inert. These cover when this machine may vouch for a
workflow by recompiling it, and - more importantly - when it may not.
"""

from __future__ import annotations

from local_lm.workflow_trust import (
    canonical_graph,
    derive_trust,
    recorded_template_identity,
)

_SHA = "a" * 64
_GRAPH = {"1": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}}


def _decide(**overrides: object):  # type: ignore[no-untyped-def]
    arguments: dict[str, object] = {
        "dependencies": {"template_id": "image_basic", "template_sha256": _SHA},
        "stored_api_graph": _GRAPH,
        "installed_template_sha256": _SHA,
        "recompiled_api_graph": _GRAPH,
        "uses_only_core_nodes": True,
    }
    arguments.update(overrides)
    return derive_trust(**arguments)  # type: ignore[arg-type]


def test_a_workflow_this_machine_rebuilds_identically_is_trusted() -> None:
    decision = _decide()

    assert decision.trusted
    assert decision.reason == "derived_locally"


def test_key_order_does_not_decide_trust() -> None:
    """Serialization differences are not modifications."""
    reordered = {"1": {"inputs": {"images": ["2", 0]}, "class_type": "SaveImage"}}

    assert canonical_graph(reordered) == canonical_graph(_GRAPH)
    assert _decide(recompiled_api_graph=reordered).trusted


def test_a_graph_changed_after_compiling_is_refused() -> None:
    edited = {"1": {"class_type": "SaveImage", "inputs": {"images": ["9", 0]}}}

    decision = _decide(stored_api_graph=edited)

    assert not decision.trusted
    assert decision.reason == "graph_differs"


def test_a_hand_authored_workflow_still_requires_review() -> None:
    # No recorded template identity: nothing to rebuild from.
    decision = _decide(dependencies={})

    assert not decision.trusted
    assert decision.reason == "no_template_identity"


def test_a_template_this_machine_does_not_have_is_refused() -> None:
    decision = _decide(installed_template_sha256=None)

    assert not decision.trusted
    assert decision.reason == "template_not_installed"


def test_a_template_that_has_since_changed_is_refused() -> None:
    """Rebuilding from a different template proves nothing about this graph."""
    decision = _decide(installed_template_sha256="b" * 64)

    assert not decision.trusted
    assert decision.reason == "template_changed"


def test_a_workflow_needing_custom_nodes_is_never_trusted_automatically() -> None:
    decision = _decide(uses_only_core_nodes=False)

    assert not decision.trusted
    assert decision.reason == "custom_nodes_required"


def test_an_unavailable_runtime_refuses_rather_than_trusting() -> None:
    """Not being able to check must never read as having checked."""
    decision = _decide(recompiled_api_graph=None)

    assert not decision.trusted
    assert decision.reason == "graph_differs"


def test_every_refusal_says_what_would_have_to_change() -> None:
    for decision in (
        _decide(dependencies={}),
        _decide(installed_template_sha256=None),
        _decide(installed_template_sha256="b" * 64),
        _decide(uses_only_core_nodes=False),
        _decide(recompiled_api_graph=None),
    ):
        assert not decision.trusted
        assert len(decision.message) > 40
        assert decision.message.endswith(".")


def test_malformed_template_identity_is_not_an_identity() -> None:
    assert recorded_template_identity({}) is None
    assert recorded_template_identity({"template_id": "x"}) is None
    assert recorded_template_identity({"template_id": "", "template_sha256": _SHA}) is None
    assert recorded_template_identity({"template_id": "x", "template_sha256": "short"}) is None
    # Case is normalized so a differently-cased record still matches.
    assert recorded_template_identity({"template_id": "x", "template_sha256": "A" * 64}) == (
        "x",
        _SHA,
    )
