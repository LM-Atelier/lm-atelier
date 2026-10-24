"""Which installed workflows the app may choose on its own for one turn.

A workflow declares by kind the inputs it takes beyond the picture and the
words: a selection to paint inside, margins to paint past the edge, or a
factor to enlarge by. A turn fills some of them and leaves the rest empty.
Chosen for a turn that leaves one of its inputs empty, a workflow does what
nobody asked, or cannot run at all: an instruction comes back at twice the
size, or an inpainter waits for a mask that never comes. So the app's own
choice passes such a workflow over. A turn that brings a selection or margins
is refused by a workflow with nowhere to put them, so the choice looks first
for one that declares them.

This decides the app's own choice only. A workflow someone names runs as
named, and the run's own checks refuse what it cannot honor. The studio's
report reads the same rules, so a tool is offered when its turn would find a
workflow to run on.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .matting_workflows import workflow_declares_matting
from .outpaint_workflows import workflow_declares_outpaint
from .studio_masks import workflow_accepts_mask
from .upscale_workflows import workflow_declares_upscale

DeclaredKind = Literal["mask", "outpaint", "upscale"]

_DECLARES: dict[DeclaredKind, Callable[[dict[str, Any] | None], bool]] = {
    "mask": workflow_accepts_mask,
    "outpaint": workflow_declares_outpaint,
    "upscale": workflow_declares_upscale,
}


@dataclass(frozen=True)
class TurnWorkflowKinds:
    """The declared kinds one turn leaves empty, and the one it brings."""

    #: A workflow declaring any of these is passed over: the turn gives it no value.
    unfilled: frozenset[DeclaredKind] = frozenset()
    #: A workflow declaring this is chosen ahead of the rest: the turn brings it.
    brings: DeclaredKind | None = None

    def admits(self, input_schema: dict[str, Any] | None) -> bool:
        """Whether the app may choose this workflow for the turn at all."""
        # A workflow that only cuts a subject out is chosen by name, by the tool
        # that asks for a cutout. Picked for any other turn it would hand back a
        # cutout instead of the edit asked for.
        if workflow_declares_matting(input_schema):
            return False
        return not any(_DECLARES[kind](input_schema) for kind in self.unfilled)

    def prefers(self, input_schema: dict[str, Any] | None) -> bool:
        """Whether this workflow takes what the turn brings."""
        return self.brings is not None and _DECLARES[self.brings](input_schema)

    def serves(self, input_schema: dict[str, Any] | None) -> bool:
        """Whether this workflow alone can carry the turn out."""
        return self.admits(input_schema) and (self.brings is None or self.prefers(input_schema))


#: A choice made with no turn to describe passes over only a cutout.
ANY_TURN = TurnWorkflowKinds()


def turn_workflow_kinds(
    *, margins: bool, extends_canvas: bool, selection: bool, enlargement: bool
) -> TurnWorkflowKinds:
    """The kinds an image edit leaves empty and the one it brings.

    `margins` is a turn that names how far to extend each side; `extends_canvas`
    one that gives a larger canvas to extend onto; `selection` one whose
    selection goes to the workflow's own mask input, rather than being placed
    back over a whole-picture edit; `enlargement` one that asks for a larger
    picture.
    """
    outpaint_filled = margins or extends_canvas
    if enlargement:
        # The workflow an enlargement runs on is the one its own check accepts,
        # and Enhance shows that choice before anything is sent, so it is left
        # as that check finds it.
        return TurnWorkflowKinds(
            unfilled=frozenset() if outpaint_filled else frozenset({"outpaint"})
        )
    unfilled: set[DeclaredKind] = {"upscale"}
    if not outpaint_filled:
        unfilled.add("outpaint")
    if not selection:
        unfilled.add("mask")
    # A canvas to extend onto is checked against the workflow's own route, so it
    # lets an outpainter be chosen without asking for one first.
    brings: DeclaredKind | None = "mask" if selection else "outpaint" if margins else None
    return TurnWorkflowKinds(unfilled=frozenset(unfilled), brings=brings)


#: The turn of a tool that edits the whole picture: Instruct, Text and Remove.
WHOLE_PICTURE_EDIT = turn_workflow_kinds(
    margins=False, extends_canvas=False, selection=False, enlargement=False
)
#: The turn of a selection tool, whose mask goes to the workflow.
SELECTION_EDIT = turn_workflow_kinds(
    margins=False, extends_canvas=False, selection=True, enlargement=False
)
#: The turn of Extend, which names margins.
EXTENSION = turn_workflow_kinds(
    margins=True, extends_canvas=False, selection=False, enlargement=False
)
#: The turn of Enhance, which asks for a larger picture.
ENLARGEMENT = turn_workflow_kinds(
    margins=False, extends_canvas=False, selection=False, enlargement=True
)
