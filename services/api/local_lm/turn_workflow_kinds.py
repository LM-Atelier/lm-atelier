"""Which installed workflows the app may choose on its own for one turn.

A workflow declares by kind the inputs it takes beyond the picture and the
words: a selection to paint inside, margins to paint past the edge, or a
factor to enlarge by. A turn fills some of them and leaves the rest empty.
Chosen for a turn that leaves one of its inputs empty, a workflow does what
nobody asked, or cannot run at all: an instruction comes back at twice the
size, or an inpainter waits for a mask that never comes. So the app's own
choice passes such a workflow over. A turn that brings a selection or margins
is refused by a workflow with nowhere to put them, and one that asks for a
larger picture by a workflow with no factor to enlarge by, so the choice looks
first for one that declares what the turn brings.

A video turn that starts from a picture may also bring a guide video. A
workflow reads one through its graph rather than its settings, so that kind is
judged from the graph: a workflow waiting for a video the turn does not bring
is passed over, and one that reads the videos a turn brings comes first.

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
from .media_references import unread_videos, videos_required
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
    #: How many guide videos the turn brings, or None for a turn videos play no part in.
    videos: int | None = None

    @property
    def brings_input(self) -> bool:
        """Whether a workflow that takes something this turn brings comes first."""
        return self.brings is not None or bool(self.videos)

    def admits(self, input_schema: dict[str, Any] | None) -> bool:
        """Whether the app may choose this workflow for the turn at all."""
        self._judged_without_a_graph()
        return self._admits_settings(input_schema)

    def prefers(self, input_schema: dict[str, Any] | None) -> bool:
        """Whether this workflow takes what the turn brings."""
        self._judged_without_a_graph()
        return self._prefers_settings(input_schema)

    def serves(self, input_schema: dict[str, Any] | None) -> bool:
        """Whether this workflow alone can carry the turn out."""
        return self.admits(input_schema) and (self.brings is None or self.prefers(input_schema))

    def admits_workflow(
        self, input_schema: dict[str, Any] | None, api_graph: dict[str, Any] | None
    ) -> bool:
        """Whether the app may choose this workflow, judged by its settings and its graph.

        One that reads no video stays a choice for a turn that brings one, so a
        turn whose only workflow cannot take its video is refused for that
        reason rather than told nothing is installed.
        """
        if not self._admits_settings(input_schema):
            return False
        if self.videos is None:
            return True
        required = videos_required(api_graph or {})
        if required == 0:
            return True
        return required <= self.videos and not unread_videos(api_graph or {}, self.videos)

    def prefers_workflow(
        self, input_schema: dict[str, Any] | None, api_graph: dict[str, Any] | None
    ) -> bool:
        """Whether this workflow takes what the turn brings, judged by settings and graph."""
        if self._prefers_settings(input_schema):
            return True
        return (
            bool(self.videos)
            and self.admits_workflow(input_schema, api_graph)
            and (videos_required(api_graph or {}) > 0)
        )

    def _admits_settings(self, input_schema: dict[str, Any] | None) -> bool:
        # A workflow that only cuts a subject out is chosen by name, by the tool
        # that asks for a cutout. Picked for any other turn it would hand back a
        # cutout instead of the edit asked for.
        if workflow_declares_matting(input_schema):
            return False
        return not any(_DECLARES[kind](input_schema) for kind in self.unfilled)

    def _prefers_settings(self, input_schema: dict[str, Any] | None) -> bool:
        return self.brings is not None and _DECLARES[self.brings](input_schema)

    def _judged_without_a_graph(self) -> None:
        if self.videos is not None:
            raise ValueError("Whether a workflow reads a video is judged from its graph.")


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
    unfilled: set[DeclaredKind] = set()
    if not outpaint_filled:
        unfilled.add("outpaint")
    if not selection:
        unfilled.add("mask")
    if enlargement:
        # Brought rather than left open: otherwise the newest workflow bound to
        # the model comes first, and an editor bound to it is chosen over an
        # upscaler bound to none, which the studio offered Enhance for.
        return TurnWorkflowKinds(unfilled=frozenset(unfilled), brings="upscale")
    unfilled.add("upscale")
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
