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


def videos_required(workflow: dict[str, Any]) -> int:
    """How many videos must be attached before every video slot the graph names is filled.

    `${input_video}` and `${input_videos}` need one, and `${input_video_N}`
    needs N + 1, because no slot repeats an earlier video. A graph naming none
    of them needs none: it reads no video at all.
    """
    required = 0
    stack: list[Any] = [workflow]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str):
            if _ALL_VIDEOS.fullmatch(value) or _FIRST_VIDEO.fullmatch(value):
                required = max(required, 1)
            elif match := _NUMBERED_VIDEO.fullmatch(value):
                required = max(required, int(match.group("index")) + 1)
    return required


#: The video types ComfyUI's LoadVideo is handed: the MP4 family, QuickTime,
#: Matroska and WebM, and AVI. A file stored as any other video type is not
#: one LM Atelier can send it.
GUIDE_VIDEO_MEDIA_TYPES = frozenset(
    {
        "video/mp4",
        "video/x-m4v",
        "video/3gpp",
        "video/3gpp2",
        "video/quicktime",
        "video/webm",
        "video/x-matroska",
        "video/x-msvideo",
        "video/avi",
        "video/msvideo",
    }
)


def guide_video_refusal(workflow: dict[str, Any], supplied: int, *, can_bring: bool) -> str | None:
    """Why the videos a turn attaches do not fit the workflow's video slots, or None.

    `can_bring` says whether the turn could bring a guide video at all: only a
    video made from a picture can. The advice in each refusal is one the
    person can follow from where they are.
    """
    required = videos_required(workflow)
    if required and not supplied:
        if not can_bring:
            return (
                "This workflow needs a guide video, which only a video made from a picture "
                "can bring. Choose another workflow."
            )
        if required == 1:
            return (
                "This workflow needs a guide video. Attach one beside the picture, or choose "
                "another workflow."
            )
        return (
            f"This workflow needs {required} guide videos. Attach them beside the picture, "
            "or choose another workflow."
        )
    if supplied and not required:
        return (
            "This workflow does not take a guide video. Choose a workflow built for one, or "
            "attach only pictures."
        )
    if supplied and required > supplied:
        return (
            f"This workflow needs {required} guide videos, and {supplied} "
            f"{'was' if supplied == 1 else 'were'} attached."
        )
    unread = unread_videos(workflow, supplied)
    if not unread:
        return None
    if unread[0] >= required:
        # Past the last slot the graph names: fewer videos would fit.
        return (
            f"This workflow does not read attached video {unread[0] + 1} of {supplied}. "
            "Choose a workflow built for more guide videos, or attach fewer."
        )
    # A slot the graph skips: no number of videos fills it.
    return (
        f"This workflow does not read attached video {unread[0] + 1} of {supplied}. "
        "Choose a workflow that reads every attached video."
    )


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
