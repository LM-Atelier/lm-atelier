"""Preserve declared instruction-edit capability without inferring compatibility."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Literal

InstructionEditCapability = Literal["declared", "unknown"]

_INSTRUCTION_LABELS = frozenset(
    {"instruction edit", "instruction editing", "instruction based image editing"}
)


def instruction_edit_declaration(labels: Iterable[object]) -> InstructionEditCapability:
    """Recognize explicit labels, not descriptions, trigger words or generic edits."""

    for label in labels:
        if not isinstance(label, str) or len(label) > 200:
            continue
        normalized = " ".join(re.sub(r"[-_]", " ", label).casefold().split())
        if normalized in _INSTRUCTION_LABELS:
            return "declared"
    return "unknown"


def instruction_edit_capability(metadata: object) -> InstructionEditCapability:
    """Read the bounded declaration; unknown is not a claim of incompatibility."""

    if isinstance(metadata, Mapping) and metadata.get("instruction_edit_capability") == "declared":
        return "declared"
    return "unknown"


def combined_instruction_edit_capability(sources: Iterable[object]) -> InstructionEditCapability:
    """Require agreement across the selected files before declaring a model capable."""

    found = False
    for source in sources:
        found = True
        if instruction_edit_capability(source) != "declared":
            return "unknown"
    return "declared" if found else "unknown"
