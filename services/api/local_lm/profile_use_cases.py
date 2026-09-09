"""Bounded use-case suggestions from declared provider metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

_FIELDS = (
    ("tags", ("tags",)),
    ("category", ("category", "model_type", "pipeline_tag")),
    ("trained_words", ("trained_words", "trigger_words")),
    ("base_model", ("base_model", "base_models")),
)


def normalize_provider_use_case_metadata(value: object) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, list[str]] = {}
    for field, aliases in _FIELDS:
        words: list[str] = []
        seen: set[str] = set()
        for alias in aliases:
            declared = value.get(alias)
            items = (
                [declared]
                if isinstance(declared, str)
                else declared[:100]
                if isinstance(declared, list)
                else []
            )
            for item in items:
                if not isinstance(item, str):
                    continue
                cleaned = " ".join(item.split())[:200]
                if not cleaned or cleaned.casefold() in seen or len(words) >= 20:
                    continue
                seen.add(cleaned.casefold())
                words.append(cleaned)
        if words:
            result[field] = words
    return result


def merge_provider_use_case_metadata(sources: Iterable[object]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for source in sources:
        for field, words in normalize_provider_use_case_metadata(source).items():
            combined = result.setdefault(field, [])
            known = seen.setdefault(field, set())
            for word in words:
                if word.casefold() not in known and len(combined) < 20:
                    known.add(word.casefold())
                    combined.append(word)
    return result


def derive_profile_use_case(metadata: object) -> str:
    normalized = normalize_provider_use_case_metadata(metadata)
    parts: list[str] = []
    seen: set[str] = set()
    length = 0
    for field, _aliases in _FIELDS:
        for value in normalized.get(field, []):
            if value.casefold() in seen:
                continue
            added = len(value) + (2 if parts else 0)
            if length + added > 1000:
                return "; ".join(parts)
            parts.append(value)
            seen.add(value.casefold())
            length += added
    return "; ".join(parts)
