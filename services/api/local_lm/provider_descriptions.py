"""Bound provider descriptions independently of deterministic use-case metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from html.parser import HTMLParser

MAX_PROVIDER_DESCRIPTION_CHARS = 8_000
_MAX_DESCRIPTION_MARKUP_CHARS = MAX_PROVIDER_DESCRIPTION_CHARS * 4
_HIDDEN_DESCRIPTION_TAGS = frozenset({"script", "style", "template", "head"})
_DESCRIPTION_BREAK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "tr",
        "td",
        "th",
        "blockquote",
        "pre",
        "hr",
    }
)


class _DescriptionText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _HIDDEN_DESCRIPTION_TAGS:
            self.hidden += 1
        elif not self.hidden and tag in _DESCRIPTION_BREAK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _HIDDEN_DESCRIPTION_TAGS:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in _DESCRIPTION_BREAK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def normalize_html_provider_description(value: object) -> str:
    """Extract bounded prose from provider markup without including tag attributes."""
    if not isinstance(value, str):
        return ""
    parser = _DescriptionText()
    try:
        parser.feed(value[:_MAX_DESCRIPTION_MARKUP_CHARS])
    except (AssertionError, ValueError):
        return ""
    # An unfinished tag at the input bound remains buffered, not exposed as prose.
    return normalize_provider_description(" ".join("".join(parser.parts).split()))


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
