"""Validate exact image selections before transporting them to a workflow."""

from __future__ import annotations

import re
from typing import Any

_SLOT = re.compile(r"input_image(?:_(?:0|[1-9][0-9]?))?|input_images")


def validate_image_bindings(
    workflow: dict[str, Any], bindings: dict[str, tuple[int, ...]], input_count: int
) -> dict[str, tuple[int, ...]]:
    """Require every supplied image and every image slot to have an exact binding.

    Return a copy so a selection cannot change while its images are uploaded.
    Repetition is explicit; a missing slot never borrows the final image.
    """
    slots: set[str] = set()
    stack: list[Any] = [workflow]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif isinstance(value, str) and value.startswith("${input_image") and value.endswith("}"):
            name = value[2:-1]
            if not _SLOT.fullmatch(name) or (
                name.startswith("input_image_") and int(name.removeprefix("input_image_")) >= 64
            ):
                raise ValueError("The workflow image bindings contain an unsupported slot")
            slots.add(name)
    if not isinstance(bindings, dict) or set(bindings) != slots:
        raise ValueError("The workflow image bindings do not cover its exact image slots")
    used: set[int] = set()
    result: dict[str, tuple[int, ...]] = {}
    for name, indices in bindings.items():
        if (
            type(indices) is not tuple
            or not indices
            or (name != "input_images" and len(indices) != 1)
            or any(type(index) is not int or not 0 <= index < input_count for index in indices)
        ):
            raise ValueError("The workflow image bindings do not match the selected images")
        used.update(indices)
        result[name] = indices
    if used != set(range(input_count)):
        raise ValueError("The workflow image bindings omit a selected image")
    return result
