"""Widget layouts a package draws itself, read exactly or not at all.

Most ComfyUI nodes describe their inputs to the runtime, so `/object_info` says
what a saved value means. A few do not: they accept any number of inputs and let
their own editor code draw, order and serialize them. For those, the saved graph
carries objects whose meaning lives in that package's JavaScript and is stated
nowhere a runtime can be asked.

Reading one means reproducing that package's editor code. That is worth doing -
the alternative is that a graph using a common node can never be imported - but
only under a rule that fails loudly. Every element of the layout is checked
against what the package is known to write, in order, and any deviation refuses.
Nothing is skipped as presumed furniture and nothing is inferred from position
alone, because a package that changed its layout would otherwise be read under
the old one and produce a graph that runs and means something else.

The layouts here are transcriptions of a package's own code, not guesses, and
each records which release it was read from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

POWER_LORA_LOADER = "Power Lora Loader (rgthree)"

# rgthree publishes under a registry id and a repository id; a saved graph may
# record either, and both name the same package.
_RGTHREE = frozenset({"rgthree-comfy", "rgthree/rgthree-comfy"})

# What the node's own `addNonLoraWidgets` puts around the loras: a divider, the
# header, then a spacer and the add button after them. Transcribed from
# rgthree-comfy `web/comfyui/power_lora_loader.js`, read at
# 6b76ee6f2c5a007710b5a16f97c94330d6ecc871.
_POWER_LORA_LOADER_LEADING: tuple[object, ...] = ({}, {"type": "PowerLoraLoaderHeaderWidget"})
_POWER_LORA_LOADER_TRAILING: tuple[object, ...] = ({}, "")

# `load_loras` reads exactly these from each entry. `strengthTwo` is written
# only when the node is showing separate model and clip strengths, and absent
# and null mean the same thing to it, so both are carried through untouched.
_LORA_REQUIRED = frozenset({"lora", "on", "strength"})
_LORA_OPTIONAL = frozenset({"strengthTwo"})


class PackageWidgetError(ValueError):
    """A package-drawn layout is not the one this transcription describes."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def package_widget_inputs(
    node_type: str,
    package: str | None,
    values: Sequence[object],
) -> dict[str, object] | None:
    """The API inputs a package-drawn node's saved widgets stand for.

    `None` when this node's layout is not transcribed here, which is the common
    case and is not an error - the caller refuses, naming the package, exactly
    as it did before any layout was known.
    """

    if node_type == POWER_LORA_LOADER and package in _RGTHREE:
        return _power_lora_loader_inputs(values)
    return None


def _power_lora_loader_inputs(values: Sequence[object]) -> dict[str, object]:
    """Each enabled or disabled lora entry, under the name the node gives it.

    The node names them `lora_1`, `lora_2` and so on in the order they appear,
    counting from one, and its Python reads every key it is given - so a
    disabled entry is still sent. Dropping one here would silently rewrite a
    graph whose author left it in place to turn back on.
    """

    leading = len(_POWER_LORA_LOADER_LEADING)
    trailing = len(_POWER_LORA_LOADER_TRAILING)
    if len(values) < leading + trailing:
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} has fewer saved widgets than it draws",
        )
    head = tuple(values[:leading])
    tail = tuple(values[len(values) - trailing :])
    if head != _POWER_LORA_LOADER_LEADING or tail != _POWER_LORA_LOADER_TRAILING:
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} is not laid out the way this build reads it",
        )
    result: dict[str, object] = {}
    for index, value in enumerate(values[leading : len(values) - trailing], start=1):
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
    entry it is.
    """

    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PackageWidgetError(
            "package_widget_layout",
            f"{POWER_LORA_LOADER} entry {index} has a {name} that is not a number",
        )
    return value
