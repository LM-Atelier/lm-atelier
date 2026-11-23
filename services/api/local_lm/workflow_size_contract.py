"""What size a workflow revision takes, read the way a turn checks it.

Families of image and video workflows take their output size differently. Some
take a width and a height, some take an aspect-ratio choice with a pixel budget
and a rounding multiple, and some fix the size in their own graph. A client that
sizes any family from one code path has to know which of these a revision takes
before it sends a turn, rather than re-deriving the settings overlay or learning
the answer from a refusal.

The answer is read from the field list a turn validates - `workflow_settings`
over the role's built-in fields and the revision's schema - with the generated
settings record saying which key each native size input was bound to, and which
inputs for the same size the graph fixes itself. Each side is judged on its own,
as the turn judges it: a workflow that fixes one side and lets a turn choose the
other advertises only the side it takes, and a client sends only the sides that
are present. Nothing here snaps or clamps a size: a value outside what is
advertised is refused by the turn, as it always was. Node ids, node titles and
the graph are not part of the answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, TypeGuard

from .domain import Operation, operation_model_role
from .schemas import (
    SettingField,
    WorkflowSizeChoiceOut,
    WorkflowSizeNumberOut,
    WorkflowSizeOut,
    WorkflowSizeSideOut,
)
from .settings_registry import builtin_settings_for_role, workflow_settings
from .workflow_graph_settings_v1 import GRAPH_SETTING_INPUT_NAMES, workflow_graph_settings

#: Only work made from nothing chooses its size. An edit keeps its source's
#: shape and a video made from a picture follows the picture, so a size
#: advertised for them would describe what the source decides.
SIZED_OPERATIONS: Final = frozenset({Operation.TEXT_TO_IMAGE, Operation.TEXT_TO_VIDEO})


def workflow_size_contract(
    operation: Operation | str, input_schema: Mapping[str, Any] | None
) -> WorkflowSizeOut | None:
    """The size a turn on one revision takes, or None where its operation chooses none.

    `input_schema` is the schema a turn lays over its role's fields, as the
    settings-schema route returns it. The role's built-in fields stand in for
    the engine's, because they are what the bundled engines advertise; a turn
    on an engine that narrows them is still checked against the narrower
    bounds. A schema the turn could not read gives None, not a guess.
    """
    try:
        kind = Operation(operation)
    except ValueError:
        return None
    if kind not in SIZED_OPERATIONS:
        return None
    try:
        fields = workflow_settings(
            builtin_settings_for_role(operation_model_role(kind)), input_schema
        )
        record = _generated_settings(input_schema)
    # A stored bound of the wrong type fails its comparison with a TypeError
    # rather than a ValueError. Lists describe every row they return, so one
    # such schema must leave its row without a size, not fail the whole list.
    except (ValueError, TypeError):
        return None
    offered = {field.key: field for field in fields if field.available}
    width = _side(_sole_field("width", offered, record))
    height = _side(_sole_field("height", offered, record))
    sides = width is not None or height is not None
    aspect = _choice(_sole_field("aspect_ratio", offered, record, bound_only=True))
    if aspect is None:
        if not sides:
            return WorkflowSizeOut(version=1, control="workflow")
        return WorkflowSizeOut(version=1, control="width_height", width=width, height=height)
    return WorkflowSizeOut(
        version=1,
        control="aspect_ratio_and_width_height" if sides else "aspect_ratio",
        width=width,
        height=height,
        aspect_ratio=aspect,
        megapixels=_number(_sole_field("megapixels", offered, record, bound_only=True)),
        round_to_multiple=_number(
            _sole_field("round_to_multiple", offered, record, bound_only=True)
        ),
    )


def _generated_settings(input_schema: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """The generated settings record, read only where `workflow_settings` reads it."""
    if not input_schema or not isinstance(input_schema.get("properties"), Mapping):
        return None
    return workflow_graph_settings(input_schema)


def _entries(record: Mapping[str, Any] | None, kind: str, name: str) -> list[Mapping[str, Any]]:
    """The record's bindings or fixed inputs, by `kind`, for native inputs that set this size.

    A grown input carries its own name, such as `mode.aspect_ratio`, so the
    size it sets is read from the name its node declares for it.
    """
    if record is None:
        return []
    return [
        entry
        for entry in record.get(kind, ())
        if GRAPH_SETTING_INPUT_NAMES.get(entry.get("declared_name", entry["input_name"])) == name
    ]


def _sole_field(
    name: str,
    offered: Mapping[str, SettingField],
    record: Mapping[str, Any] | None,
    *,
    bound_only: bool = False,
) -> SettingField | None:
    """The one field a turn takes for this size input, when there is exactly one to choose.

    A native input the generated record bound may carry a generated key, so the
    keys are read from the record rather than guessed from the name. Two inputs
    for the same size leave no single control, because which of them sizes the
    result is the graph's to say: two bindings, a binding beside a declared
    plain key, or either beside an input the graph fixes for the same size. A
    fixed input on a node that reaches no output never runs, so it does not
    count. `bound_only` holds controls that only a bound native input can
    describe, such as an aspect ratio, to a binding.
    """
    bound = {binding["parameter"] for binding in _entries(record, "bindings", name)}
    if bound_only and not bound:
        return None
    if any(entry["reason"] != "disconnected" for entry in _entries(record, "fixed", name)):
        return None
    keys = bound | ({name} if name in offered else set())
    if len(keys) != 1:
        return None
    field = offered.get(keys.pop())
    return field if field is not None and _offers_a_choice(field) else None


def _offers_a_choice(field: SettingField) -> bool:
    """Whether a turn may choose this setting, rather than the workflow fixing it.

    The same judgement the turn makes, side by side, before it refuses a saved
    size: a field that is unavailable, holds one value, or spans a single
    number leaves nothing to choose, so it is not advertised as a control.
    """
    if not field.available or len(field.choices) == 1:
        return False
    return field.minimum is None or field.minimum != field.maximum


def _plain(value: object) -> TypeGuard[int | float | str]:
    """Whether a value is a number or a string a client can send back as it is.

    A boolean, a list or an object is never a size, so a control carrying one
    is left out rather than described wrongly.
    """
    return isinstance(value, int | float | str) and not isinstance(value, bool)


def _describable(field: SettingField) -> bool:
    """Whether a field's default and choices can all be sent back as they are."""
    return (field.default is None or _plain(field.default)) and all(
        _plain(choice) for choice in field.choices
    )


def _as_sent(value: Any, field: SettingField) -> Any:
    """A number as a turn takes it back: whole, for a setting that takes whole numbers.

    A setting holds its bounds as floats, and a turn refuses 2048.0 for one
    that takes whole numbers, so a bound a client copies into a turn has to be
    the integer the turn accepts.
    """
    if field.type == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _side(field: SettingField | None) -> WorkflowSizeSideOut | None:
    """A width or height a turn can set: a range of whole numbers, or a menu of sizes."""
    if field is None or field.type not in {"integer", "enum"} or not _describable(field):
        return None
    return WorkflowSizeSideOut(
        key=field.key,
        default=_as_sent(field.default, field),
        minimum=_as_sent(field.minimum, field),
        maximum=_as_sent(field.maximum, field),
        multiple_of=_as_sent(field.multiple_of, field),
        step=_as_sent(field.step, field),
        choices=list(field.choices),
    )


def _choice(field: SettingField | None) -> WorkflowSizeChoiceOut | None:
    """An aspect ratio a turn can choose: a menu, never a free value."""
    if field is None or field.type != "enum" or not _describable(field):
        return None
    return WorkflowSizeChoiceOut(key=field.key, default=field.default, choices=list(field.choices))


def _number(field: SettingField | None) -> WorkflowSizeNumberOut | None:
    """A number that scales a chosen shape: a range a client picks from, never a menu.

    A field that lists its values is not a range, so it is left out rather
    than described by bounds it does not keep to.
    """
    if (
        field is None
        or field.type not in {"integer", "number"}
        or field.choices
        or isinstance(field.default, str)
        or not _describable(field)
    ):
        return None
    return WorkflowSizeNumberOut(
        key=field.key,
        default=_as_sent(field.default, field),
        minimum=_as_sent(field.minimum, field),
        maximum=_as_sent(field.maximum, field),
        multiple_of=_as_sent(field.multiple_of, field),
        step=_as_sent(field.step, field),
    )
