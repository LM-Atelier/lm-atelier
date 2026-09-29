"""Bound provider descriptions independently of deterministic use-case metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

MAX_PROVIDER_DESCRIPTION_CHARS = 8_000


def normalize_provider_description(value: object) -> str:
    if not isinstance(value, str):
        return ""
    description = value[:MAX_PROVIDER_DESCRIPTION_CHARS].strip()
    try:
        description.encode("utf-8")
    except UnicodeError:
        return ""
    return description


def merge_provider_descriptions(values: Iterable[object]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    length = 0
    for value in values:
        description = normalize_provider_description(value)
        if not description or description in seen:
            continue
        available = MAX_PROVIDER_DESCRIPTION_CHARS - length - (2 if parts else 0)
        if available <= 0:
            break
        parts.append(description[:available])
        seen.add(description)
        length += len(parts[-1]) + (2 if len(parts) > 1 else 0)
    return "\n\n".join(parts)


def installed_provider_description(contract: object, source_metadata: object) -> str:
    """Keep an accepted plan authoritative, including an absent description."""
    if contract is not None:
        return normalize_provider_description(
            contract.get("provider_description") if isinstance(contract, Mapping) else None
        )
    return normalize_provider_description(
        source_metadata.get("description") if isinstance(source_metadata, Mapping) else None
    )
