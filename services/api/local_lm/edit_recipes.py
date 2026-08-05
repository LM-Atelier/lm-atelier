"""Turning a run that worked into a recipe that reproduces it.

A template carried an instruction and some settings. Applying one reproduced
the words, and then ran them against whatever workflow and model happened to
be current - so the same template could produce a different picture on a
different day, and nothing in it said why.

A recipe is the rest of the answer: the workflow revision, the model profile,
the settings as they actually resolved, and whether the edit was scoped to a
selection. All of it is read from the finished run's provenance rather than
from the current state of the machine, because what is current when someone
presses Save is not necessarily what produced the result they liked.

Nothing here guesses. A run whose provenance does not say which workflow it
used yields a recipe that admits it, rather than one that names today's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MASK_MODES = frozenset({"none", "selection", "inverse"})
#: Settings that describe one particular picture rather than how to make one.
#: A recipe carrying these would reproduce the seed of a result someone liked
#: and none of the reason they liked it.
_PER_RESULT_SETTINGS = frozenset({"seed", "noise_seed", "batch_size", "mask"})


@dataclass(frozen=True)
class RecipeCapture:
    """What a recipe records about the run it was saved from."""

    workflow_revision_id: str | None
    model_profile_id: str | None
    settings: dict[str, Any]
    mask_mode: str


def capture_recipe(provenance: dict[str, Any]) -> RecipeCapture:
    """Read a finished run's provenance into the parts a recipe keeps."""
    workflow = provenance.get("workflow")
    revision_id = None
    if isinstance(workflow, dict):
        candidate = workflow.get("revision_id")
        revision_id = candidate if isinstance(candidate, str) and candidate else None
    settings = provenance.get("resolved_settings")
    resolved = dict(settings) if isinstance(settings, dict) else {}
    mask_mode = _mask_mode(resolved.get("mask"))
    return RecipeCapture(
        workflow_revision_id=revision_id,
        model_profile_id=_profile_id(provenance),
        settings={key: value for key, value in resolved.items() if key not in _PER_RESULT_SETTINGS},
        mask_mode=mask_mode,
    )


def _profile_id(provenance: dict[str, Any]) -> str | None:
    """The profile the run actually used, which lives under its model record."""
    model = provenance.get("model")
    if not isinstance(model, dict):
        return None
    candidate = model.get("profile_id")
    return candidate if isinstance(candidate, str) and candidate else None


def _mask_mode(mask: object) -> str:
    """Whether the run was scoped to a selection, and which way round.

    The selection itself is never kept - it was drawn on one picture and means
    nothing on another. What carries over is that this recipe expects one,
    which is what decides whether applying it can be a single click.
    """
    if not isinstance(mask, dict):
        return "none"
    return "inverse" if mask.get("invert") is True else "selection"
