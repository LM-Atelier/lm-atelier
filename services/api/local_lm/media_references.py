"""How many reference images a workflow can actually use.

Attachments are offered to a graph three ways: `${input_image}` for the first,
`${input_images}` for the whole list, and `${input_image_N}` per numbered slot.
A workflow takes whichever of those it names.

That means a graph naming only `${input_image}` consumes the first attachment
and ignores every other one - silently, because nothing downstream knows how
many were meant. Someone attaching four references and getting a picture
conditioned on one has no way to tell that from a bad model. This is how the
count is compared against what the graph can take, so the turn can refuse
before it runs instead of quietly using less than it was given.
"""

from __future__ import annotations

import re
from typing import Any

# The whole list. A graph naming this takes as many as it is given.
_ALL_INPUTS = re.compile(r"\$\{input_images\}\Z")
_FIRST_INPUT = re.compile(r"\$\{input_image\}\Z")
_NUMBERED_INPUT = re.compile(r"\$\{input_image_(?P<index>\d{1,2})\}\Z")

MAX_NUMBERED_INPUTS = 64
UNBOUNDED = -1


def reference_capacity(workflow: dict[str, Any]) -> int:
    """How many attachments this graph can consume, or `UNBOUNDED`.

    Walks the compiled graph for the placeholders it actually contains rather
    than trusting a declared schema, because the graph is what runs. A graph
    naming none of them takes no images at all, which is a real answer: it is
    a text-to-image workflow being handed a reference.
    """
    numbered: set[int] = set()
    takes_first = False
    stack: list[Any] = [workflow]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str):
            if _ALL_INPUTS.fullmatch(value):
                return UNBOUNDED
            if _FIRST_INPUT.fullmatch(value):
                takes_first = True
            elif (match := _NUMBERED_INPUT.fullmatch(value)) and (
                index := int(match.group("index"))
            ) < MAX_NUMBERED_INPUTS:
                numbered.add(index)
    if takes_first:
        numbered.add(0)
    if not numbered:
        return 0
    prefix = 0
    while prefix in numbered:
        prefix += 1
    # Higher slots repeat the last supplied picture. That can fill one gap,
    # but every earlier picture still needs its own slot.
    return min(prefix + 1, max(numbered) + 1)


_ALL_VIDEOS = re.compile(r"\$\{input_videos\}\Z")
_FIRST_VIDEO = re.compile(r"\$\{input_video\}\Z")
_NUMBERED_VIDEO = re.compile(r"\$\{input_video_(?P<index>\d{1,2})\}\Z")


def unread_videos(workflow: dict[str, Any], supplied: int) -> list[int]:
    """The positions, among the attached videos, of each one the graph never reads.

    Videos are offered as `${input_video}` for the first, `${input_videos}`
    for the list and `${input_video_N}` per numbered slot. Unlike pictures, no
    slot past the last video repeats one, so every video needs a slot of its
    own.
    """
    if supplied <= 0:
        return []
    read: set[int] = set()
    stack: list[Any] = [workflow]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str):
            if _ALL_VIDEOS.fullmatch(value):
                return []
            if _FIRST_VIDEO.fullmatch(value):
                read.add(0)
            elif match := _NUMBERED_VIDEO.fullmatch(value):
                read.add(int(match.group("index")))
    return [index for index in range(supplied) if index not in read]


def exceeds_capacity(workflow: dict[str, Any], supplied: int) -> int | None:
    """The capacity that was overrun, or `None` when everything fits.

    Returns the number rather than a boolean so the refusal can say what the
    workflow can take, which is the part that tells someone what to do next.
    """
    if supplied <= 1:
        return None
    capacity = reference_capacity(workflow)
    if capacity == UNBOUNDED or supplied <= capacity:
        return None
    return capacity
