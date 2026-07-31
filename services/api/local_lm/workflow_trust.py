"""Derive trust for a workflow the local machine can rebuild for itself.

Importing a workflow forces `trusted=False`, and execution hard-refuses untrusted
revisions, so an imported setup arrives inert: the graph is present, the models
resolve, and nothing will run it. Asking every user to review a graph they cannot
read is not a safety control, it is a wall.

But a workflow compiled from a catalog template is not arbitrary code. If this
machine recompiles the recorded template identity and gets back byte-identical
bytes, then the graph was derived here, from a template already shipped here,
using only core nodes - which is precisely the assertion the compiler makes when
it produces a workflow during a normal install. Trust follows from that
derivation rather than from where the file happened to travel.

A graph that cannot be re-derived - hand-authored, edited after compiling, built
from a template this machine does not have, or requiring custom nodes - keeps
requiring explicit review. Nothing here weakens that.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

TRUST_DERIVATION_VERSION = 1


@dataclass(frozen=True)
class TrustDecision:
    """Whether a revision may be trusted, and the reason either way."""

    trusted: bool
    reason: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": TRUST_DERIVATION_VERSION,
            "trusted": self.trusted,
            "reason": self.reason,
            "message": self.message,
        }


_REFUSALS = {
    "no_template_identity": (
        "This workflow does not record which template it was compiled from, so it "
        "cannot be rebuilt here. Review it before trusting it."
    ),
    "template_not_installed": (
        "The template this workflow was compiled from is not installed here, so it "
        "cannot be rebuilt. Review it before trusting it."
    ),
    "template_changed": (
        "The installed template no longer matches the one this workflow was "
        "compiled from, so rebuilding it would not prove anything."
    ),
    "custom_nodes_required": (
        "This workflow needs nodes outside the ComfyUI core, which cannot be "
        "trusted automatically. Review it before trusting it."
    ),
    "graph_differs": (
        "Rebuilding the template here produced a different workflow, so this one "
        "was changed after it was compiled. Review it before trusting it."
    ),
}

_DERIVED = "This machine rebuilt the workflow from its own template and got the same result."


def canonical_graph(graph: Mapping[str, Any]) -> str:
    """A byte-comparable form of a graph.

    Sorted keys and compact separators, so two graphs that mean the same thing
    compare equal regardless of how either was serialized. List order is
    preserved because it can be semantic.
    """
    return json.dumps(graph, sort_keys=True, separators=(",", ":"), allow_nan=False)


def recorded_template_identity(dependencies: Mapping[str, Any]) -> tuple[str, str] | None:
    """The template id and sha256 a revision recorded, if it recorded both."""
    template_id = dependencies.get("template_id")
    template_sha256 = dependencies.get("template_sha256")
    if not isinstance(template_id, str) or not template_id:
        return None
    if not isinstance(template_sha256, str) or len(template_sha256) != 64:
        return None
    return template_id, template_sha256.lower()


def derive_trust(
    *,
    dependencies: Mapping[str, Any],
    stored_api_graph: Mapping[str, Any],
    installed_template_sha256: str | None,
    recompiled_api_graph: Mapping[str, Any] | None,
    uses_only_core_nodes: bool,
) -> TrustDecision:
    """Whether this machine can vouch for a revision by rebuilding it.

    Every refusal names what would have to change, because "untrusted" with no
    explanation is the state that made imported setups unusable in the first
    place.
    """
    identity = recorded_template_identity(dependencies)
    if not identity:
        return TrustDecision(False, "no_template_identity", _REFUSALS["no_template_identity"])
    _, recorded_sha256 = identity

    if not installed_template_sha256:
        return TrustDecision(False, "template_not_installed", _REFUSALS["template_not_installed"])
    if installed_template_sha256.lower() != recorded_sha256:
        return TrustDecision(False, "template_changed", _REFUSALS["template_changed"])
    if not uses_only_core_nodes:
        return TrustDecision(False, "custom_nodes_required", _REFUSALS["custom_nodes_required"])
    if recompiled_api_graph is None:
        return TrustDecision(False, "graph_differs", _REFUSALS["graph_differs"])
    if canonical_graph(recompiled_api_graph) != canonical_graph(stored_api_graph):
        return TrustDecision(False, "graph_differs", _REFUSALS["graph_differs"])
    return TrustDecision(True, "derived_locally", _DERIVED)
