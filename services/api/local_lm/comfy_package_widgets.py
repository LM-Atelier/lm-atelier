"""Widget layouts a package draws itself, read exactly or not at all.

Most ComfyUI nodes describe their inputs to the runtime, so `/object_info` says
what a saved value means. A few do not: they accept any number of inputs and let
their own editor code draw, order and serialize them. For those, the saved graph
carries values whose meaning lives in that package's JavaScript and is stated
nowhere a runtime can be asked.

Reading one means reproducing that package's editor code. That is worth doing -
the alternative is that a graph using a common node can never be imported - but
only under a rule that fails loudly.

Two things are checked, and neither substitutes for the other:

**Which layout applies.** A graph records the package and revision that drew each
node. That is a discriminator, never authority: it says which layout to read by,
and says nothing about what is installed or trusted. A revision this module has
not audited refuses, because a layout that has not been read cannot be
reproduced, and the nearest one that has is a guess.

**That the graph matches it.** Every element is checked in place, against what
the package is known to write. Nothing is skipped as presumed furniture and
nothing is inferred from position alone, because a package that changed its
layout would otherwise be read under the old one and produce a graph that runs
and means something else.

A complete layout proves the shape, not the meaning. Two revisions can serialize
identically and read the values differently, which is why the revision check
stands even when the shape matches.

Each audit below records the exact file that defines the layout and the digest
it was read at, so a claim to have audited a revision is checkable rather than
asserted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

POWER_LORA_LOADER = "Power Lora Loader (rgthree)"
VIDEO_COMBINE = "VHS_VideoCombine"


@dataclass(frozen=True)
class PackageClaim:
    """What a saved graph says drew a node.

    Both ids are kept because a graph records a registry name and a repository
    name for the same package, and they never match as strings - `rgthree-comfy`
    against `rgthree/rgthree-comfy`. So difference is not disagreement, and only
    the alias sets below can tell the two apart.
    """

    registry_id: str | None
    repository_id: str | None
    revision: str

    @property
    def package_id(self) -> str:
        """The name to use when saying which package this was."""

        return self.registry_id or self.repository_id or "its package"


class PackageWidgetError(ValueError):
    """A package-drawn layout is not the one this transcription describes."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# Packages publish under a registry id and a repository id; a saved graph may
# record either, and both name the same package.
_RGTHREE = frozenset({"rgthree-comfy", "rgthree/rgthree-comfy"})
_VIDEO_HELPER_SUITE = frozenset(
    {"comfyui-videohelpersuite", "Kosinkadink/ComfyUI-VideoHelperSuite"}
)

# rgthree draws the loader in `web/comfyui/power_lora_loader.js`. These two
# revisions ship that file byte-identically at sha256
# 1de8669ab958806e26e4f0ad68e9b06e376fa50cc53d574f7d20daca529559b1, which is
# what makes them one audited layout rather than two assumed to match. A
# revision is admitted here only once that file has been read at it.
_RGTHREE_AUDITED = frozenset({"1.0.2605082257", "6b76ee6f2c5a007710b5a16f97c94330d6ecc871"})

# VideoHelperSuite derives the combiner's extra widgets from the format files in
# `video_formats/`, through `gen_format_widgets`. Audited at this revision, with
# `h264-mp4.json` read at sha256
# 772c3873e8aba511454b4c3f76b2bdea6a848dc7df7bccbde5d4662c68416288.
_VIDEO_HELPER_SUITE_AUDITED = frozenset({"8343122234b61a0f8eb3d1f3f98382b0f7aff2b9"})

# What the loader's `addNonLoraWidgets` puts around the loras: a divider, the
# header, then a spacer and the add button after them. A divider serializes as
# an empty object or as null depending on how the graph was exported, so both
# are named rather than the pair being loosened into "anything".
_DIVIDER: tuple[object, ...] = ({}, None)
_HEADER = {"type": "PowerLoraLoaderHeaderWidget"}
_ADD_BUTTON = ""

# `load_loras` reads exactly these from each entry. `strengthTwo` is written
# only when the node is showing separate model and clip strengths, and absent
# and null mean the same thing to it, so both are carried through untouched.
_LORA_REQUIRED = frozenset({"lora", "on", "strength"})
_LORA_OPTIONAL = frozenset({"strengthTwo"})

# The combiner's own DOM widget. It holds player state - what was last previewed
# and how - and the node never reads it. It is recognised so that a graph
# carrying it can still be read, and dropped so that it never reaches a runtime
# that has no input by that name.
_VIDEO_PREVIEW = "videopreview"
_VIDEO_PREVIEW_KEYS = frozenset({"hidden", "paused", "params", "muted"})

# Transcribed from `video_formats/h264-mp4.json`. Every list in a format file
# becomes one widget named by its first element, so this is that file's widget
# set and nothing else. A format with no entry here refuses rather than being
# read under another format's set.
_VIDEO_FORMAT_WIDGETS: dict[str, dict[str, object]] = {
    "video/h264-mp4": {
        "pix_fmt": ("yuv420p", "yuv420p10le"),
        "crf": (0, 100),
        "save_metadata": bool,
        "trim_to_audio": bool,
    }
}


def package_widget_inputs(
    node_type: str,
    claim: PackageClaim | None,
    values: Sequence[object],
) -> dict[str, object] | None:
    """The API inputs a package-drawn node's saved widgets stand for.

    `None` when this node's layout is not transcribed here, which is the common
    case and is not an error - the caller refuses, naming the package, exactly
    as it did before any layout was known.
    """

    if node_type != POWER_LORA_LOADER or not _names(claim, _RGTHREE, node_type):
        return None
    assert claim is not None
    _require_audited(node_type, claim, _RGTHREE_AUDITED)
    return _power_lora_loader_inputs(values)


def package_named_widget_inputs(
    node_type: str,
    claim: PackageClaim | None,
    declared: Mapping[str, object],
    unread: Mapping[str, object],
) -> dict[str, object] | None:
    """The API inputs a node's extra named widgets stand for.

    Some nodes declare a fixed set of inputs to the runtime and then add more in
    the editor depending on what one of them is set to. Those extra widgets are
    named for the inputs they become, so a graph that saved them by name has
    already said what they are - what a transcription adds is knowing which of
    them the node actually takes, and which are furniture it never reads.

    `None` when this node is not one of those, which is the common case and is
    not an error: an unrecognised name means the graph saved a value for an
    input the node does not have, and the caller says so.
    """

    if node_type != VIDEO_COMBINE or not _names(claim, _VIDEO_HELPER_SUITE, node_type):
        return None
    assert claim is not None
    _require_audited(node_type, claim, _VIDEO_HELPER_SUITE_AUDITED)
    return _video_combine_extras(declared, unread)


def _names(claim: PackageClaim | None, aliases: frozenset[str], node_type: str) -> bool:
    """Whether the graph's claim names this package, and only this package.

    A node carrying one id this package answers to and one it does not is not a
    match with a stray field - it is two statements about which code drew the
    node that cannot both be true. Reading it by either one would be choosing
    which to believe.
    """

    if claim is None:
        return False
    stated = {value for value in (claim.registry_id, claim.repository_id) if value}
    if not stated:
        return False
    recognised = stated & aliases
    if not recognised:
        return False
    if recognised != stated:
        raise PackageWidgetError(
            "conflicting_package_claim",
            f"{node_type} names {' and '.join(sorted(stated))}, which are different packages",
        )
    return True


def _require_audited(node_type: str, claim: PackageClaim, audited: frozenset[str]) -> None:
    """Refuse a revision whose layout has not been read.

    There is no nearest-revision fallback on purpose. The layouts here are
    transcriptions of specific code, and applying one to a revision nobody
    checked is the guess this module exists to avoid.
    """

    if claim.revision in audited:
        return
    if not claim.revision:
        raise PackageWidgetError(
            "package_widget_revision",
            f"{node_type} does not say which {claim.package_id} revision drew it",
        )
    raise PackageWidgetError(
        "package_widget_revision",
        f"{node_type} was drawn by {claim.package_id} {claim.revision},"
        " whose layout this build has not read",
    )


def _power_lora_loader_inputs(values: Sequence[object]) -> dict[str, object]:
    """Each enabled or disabled lora entry, under the name the node gives it.

    The node names them `lora_1`, `lora_2` and so on in the order they appear,
    counting from one, and its Python reads every key it is given - so a
    disabled entry is still sent. Dropping one here would silently rewrite a
    graph whose author left it in place to turn back on.
    """

    if len(values) < 4:
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} has fewer saved widgets than it draws",
        )
    if values[0] not in _DIVIDER or values[1] != _HEADER:
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} does not begin the way this build reads it",
        )
    if values[-2] not in _DIVIDER or values[-1] != _ADD_BUTTON:
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} does not end the way this build reads it",
        )
    result: dict[str, object] = {}
    for index, value in enumerate(values[2:-2], start=1):
        result[f"lora_{index}"] = _power_lora_loader_entry(index, value)
    return result


def _power_lora_loader_entry(index: int, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} entry {index} is not a lora",
        )
    keys = {str(key) for key in value}
    missing = _LORA_REQUIRED - keys
    if missing:
        name = sorted(missing)[0]
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} entry {index} has no {name}",
        )
    unknown = keys - _LORA_REQUIRED - _LORA_OPTIONAL
    if unknown:
        name = sorted(unknown)[0]
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} entry {index} carries {name}, which this build does not read",
        )
    lora = value["lora"]
    if not isinstance(lora, str) or not lora:
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} entry {index} names no lora",
        )
    if not isinstance(value["on"], bool):
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} entry {index} is neither on nor off",
        )
    entry: dict[str, object] = {
        "on": value["on"],
        "lora": lora,
        "strength": _strength(index, "strength", value["strength"], optional=False),
    }
    if "strengthTwo" in keys:
        entry["strengthTwo"] = _strength(index, "strengthTwo", value["strengthTwo"], optional=True)
    return entry


def _strength(index: int, name: str, value: object, *, optional: bool) -> float | int | None:
    """A strength the node would apply, or the null it treats as unset.

    `bool` is excluded deliberately: it is an `int` in Python, and a strength of
    `True` would silently become 1.0 rather than being reported as the malformed
    entry it is. So are the non-finite floats, which survive JSON round-trips in
    some encoders and mean nothing to a scaling factor.
    """

    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} entry {index} has a {name} that is not a number",
        )
    if not math.isfinite(value):
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} entry {index} has a {name} that is not a finite number",
        )
    return value


def lora_entry_is_a_dependency(entry: Mapping[str, object]) -> bool:
    """Whether a saved lora entry is one the graph actually needs to run.

    An entry the author switched off, or left at zero strength for both model
    and clip, is carried into the compiled prompt so that turning it back on
    means editing the node rather than rebuilding it - but the node will not
    load its file, so treating that file as a missing dependency blocks an
    import over something the run would never open.
    """

    if entry.get("on") is not True:
        return False
    model = entry.get("strength")
    clip = entry.get("strengthTwo")
    effective = [value for value in (model, clip) if isinstance(value, int | float)]
    if not effective:
        return False
    return any(value != 0 for value in effective)


def executable_lora_entries(
    node_type: str,
    package_id: str | None,
    revision: str | None,
    widgets: object,
) -> list[object] | None:
    """The saved entries a lora loader would actually open a file for.

    `None` when that cannot be established - a node this module does not read, a
    revision it has not audited, or a layout that does not match. The caller
    then treats every string in the node as an asset reference, which is what it
    did before this existed: over-reporting a dependency is a worse answer than
    under-reporting one, and an unreadable layout is not evidence of absence.
    """

    if node_type != POWER_LORA_LOADER or package_id not in _RGTHREE:
        return None
    if revision is None or revision not in _RGTHREE_AUDITED:
        return None
    if not isinstance(widgets, Sequence) or isinstance(widgets, str | bytes):
        return None
    try:
        entries = _power_lora_loader_inputs(widgets)
    except PackageWidgetError:
        return None
    return [
        entry
        for entry in entries.values()
        if isinstance(entry, Mapping) and lora_entry_is_a_dependency(entry)
    ]


def _video_combine_extras(
    declared: Mapping[str, object],
    unread: Mapping[str, object],
) -> dict[str, object]:
    """The options the chosen video format adds, and nothing else.

    `combine_video` ends in `**kwargs`, so the runtime's declaration does not
    list these and anything sent under a name it does not expect reaches the
    node unchecked. The widget set is therefore read from the format the graph
    selected, transcribed above from that package's own format file, and a name
    outside it refuses rather than being passed on because it happened to be a
    scalar. Shape is not meaning.

    The format itself is a declared input, so it is read from what the node
    already resolved rather than from the leftovers - the leftovers are the
    thing it explains.
    """

    selected = declared.get("format")
    format_name = selected if isinstance(selected, str) else None
    if format_name is None:
        raise PackageWidgetError(
            "package_widget_layout",
            f"{VIDEO_COMBINE} saves options without saying which format they belong to",
        )
    widgets = _VIDEO_FORMAT_WIDGETS.get(format_name)
    if widgets is None:
        raise PackageWidgetError(
            "package_widget_layout",
            f"{VIDEO_COMBINE} format {format_name} has options this build has not read",
        )
    extras: dict[str, object] = {}
    for name in sorted(unread):
        value = unread[name]
        if name == _VIDEO_PREVIEW:
            _reject_unknown_preview(value)
            continue
        expected = widgets.get(name)
        if expected is None:
            raise PackageWidgetError(
                "package_widget_layout",
                f"{VIDEO_COMBINE} format {format_name} has no option named {name}",
            )
        extras[name] = _video_combine_option(name, value, expected)
    return extras


def _video_combine_option(name: str, value: object, expected: object) -> object:
    """One option, checked against what its format declares it to be."""

    if expected is bool:
        if not isinstance(value, bool):
            raise PackageWidgetError(
                "package_widget_layout", f"{VIDEO_COMBINE} option {name} is not a true or false"
            )
        return value
    if isinstance(expected, tuple) and expected and isinstance(expected[0], str):
        if value not in expected:
            raise PackageWidgetError(
                "package_widget_layout", f"{VIDEO_COMBINE} option {name} is not one of its choices"
            )
        return value
    if isinstance(expected, tuple) and len(expected) == 2:
        low, high = expected
        if isinstance(value, bool) or not isinstance(value, int):
            raise PackageWidgetError(
                "package_widget_layout", f"{VIDEO_COMBINE} option {name} is not a whole number"
            )
        if not isinstance(low, int) or not isinstance(high, int) or not low <= value <= high:
            raise PackageWidgetError(
                "package_widget_layout", f"{VIDEO_COMBINE} option {name} is outside its range"
            )
        return value
    raise PackageWidgetError(
        "package_widget_layout", f"{VIDEO_COMBINE} option {name} has no readable declaration"
    )


def _reject_unknown_preview(value: object) -> None:
    """Check the player state before discarding it.

    It is dropped either way, so this is not about what gets sent. It is about
    not treating a key as known furniture when it is carrying something this
    build has never seen - which is how a changed layout passes unnoticed.
    """

    if not isinstance(value, Mapping):
        raise PackageWidgetError(
            "package_widget_layout", f"{VIDEO_COMBINE} {_VIDEO_PREVIEW} is not the state it draws"
        )
    unknown = {str(key) for key in value} - _VIDEO_PREVIEW_KEYS
    if unknown:
        name = sorted(unknown)[0]
        raise PackageWidgetError(
            "package_widget_layout",
            f"{VIDEO_COMBINE} {_VIDEO_PREVIEW} carries {name}, which this build does not know",
        )
