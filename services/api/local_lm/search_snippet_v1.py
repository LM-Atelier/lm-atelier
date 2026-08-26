"""Pure structured search snippet segments (item 41).

Builds bounded ordered {text, matched} segments from already-visible body
text. Never emits HTML or loads an entire chat. Matching runs over a
casefolded view through an offset map, so recall keeps full casefolding
(a query "strasse" finds "Strasse") while every slice lands on original
code-point boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, NoReturn

SCHEMA_ID: Final = "lm-atelier-search-snippet-v1"
SCHEMA_VERSION: Final = 1
INVALID_SNIPPET: Final = "search snippet facts are invalid"
MAX_SNIPPET_CHARS: Final = 240
MAX_BODY_CHARS: Final = 8192
MAX_SEGMENTS: Final = 8
MAX_RADIUS: Final = 48
MAX_QUERY_CHARS: Final = 200
MAX_TERMS: Final = 16


class SearchSnippetError(ValueError):
    """Fixed non-echoing refusal for invalid snippet facts."""


@dataclass(frozen=True, slots=True)
class SnippetSegmentV1:
    text: str
    matched: bool


@dataclass(frozen=True, slots=True)
class SearchSnippetV1:
    schema: Literal["lm-atelier-search-snippet-v1"]
    schema_version: Literal[1]
    segments: tuple[SnippetSegmentV1, ...]
    html_authorized: Literal[False] = field(default=False, init=False)
    loads_entire_chat: Literal[False] = field(default=False, init=False)


def _refuse() -> NoReturn:
    raise SearchSnippetError(INVALID_SNIPPET)


def _query_terms(query: str) -> tuple[str, ...]:
    """Split a bounded query into at most MAX_TERMS casefolded terms."""
    text = " ".join(query.split())
    if not text or len(text) > MAX_QUERY_CHARS:
        _refuse()
    terms = tuple(part.casefold() for part in text.split() if part)
    if not terms or len(terms) > MAX_TERMS:
        _refuse()
    # The raw cap alone is not enough: Unicode casefold expansion can
    # triple a term (U+FB03 folds to "ffi"), so 100 legal raw characters
    # became a 300-character folded term whose whole-code-point match
    # overran the snippet ceiling. The cap binds the NORMALIZED term,
    # which restores the reservation proof: the matched
    # original slice never exceeds the folded term, and the folded term
    # never exceeds MAX_QUERY_CHARS, which is under MAX_SNIPPET_CHARS.
    if any(len(term) > MAX_QUERY_CHARS for term in terms):
        _refuse()
    return terms


def _fold_with_offsets(body: str) -> tuple[str, tuple[int, ...]]:
    """Casefold the body keeping a folded-index to original-index map.

    ``casefold()`` is not length preserving, so an index found in the
    folded text must never slice the original: every expansion before the
    match shifts the position, and with enough expansions the highlight
    lands on unrelated text or vanishes. Each folded character remembers
    the original code point it came from.
    """
    pieces: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(body):
        folded = character.casefold()
        pieces.append(folded)
        offsets.extend([index] * len(folded))
    return "".join(pieces), tuple(offsets)


def _clip(text: str) -> str:
    if len(text) <= MAX_SNIPPET_CHARS:
        return text
    return text[: MAX_SNIPPET_CHARS - 3].rstrip() + "..."


def build_search_snippet(
    body: object, query: object, *, radius: object = MAX_RADIUS
) -> SearchSnippetV1:
    """Return bounded plain segments around the first query-term match."""
    if type(body) is not str or not body or len(body) > MAX_BODY_CHARS:
        _refuse()
    if type(query) is not str or len(query) > MAX_QUERY_CHARS:
        _refuse()
    if type(radius) is not int or radius < 1 or radius > MAX_RADIUS:
        _refuse()
    terms = _query_terms(query)
    folded, offsets = _fold_with_offsets(body)
    fold_pos = -1
    start_original = 0
    end_original = 0
    for term in terms:
        fold_pos = folded.find(term)
        if fold_pos >= 0:
            # The slice covers whole original code points whose casefold
            # contains the term, so no adjacent text is lost and nothing
            # outside the match is highlighted.
            start_original = offsets[fold_pos]
            end_original = offsets[fold_pos + len(term) - 1] + 1
            break
    segments: tuple[SnippetSegmentV1, ...]
    if fold_pos < 0:
        segments = (SnippetSegmentV1(text=_clip(body), matched=False),)
    else:
        matched = body[start_original:end_original]
        start = max(0, start_original - radius)
        end = min(len(body), end_original + radius)
        prefix = body[start:start_original]
        suffix = body[end_original:end]
        if start > 0:
            prefix = "..." + prefix.lstrip()
        if end < len(body):
            suffix = suffix.rstrip() + "..."
        parts: list[SnippetSegmentV1] = []
        if prefix:
            parts.append(SnippetSegmentV1(text=prefix, matched=False))
        parts.append(SnippetSegmentV1(text=matched, matched=True))
        if suffix:
            parts.append(SnippetSegmentV1(text=suffix, matched=False))
        # Reserve the COMPLETE match before any context: spending the
        # ceiling prefix-first truncated a maximum-legal match while still
        # marking it matched, reopening the reservation invariant.
        # The reservation always fits: every original code point folds to at
        # least one character, so the matched slice never exceeds the folded
        # term - len(matched) <= len(term) <= MAX_QUERY_CHARS, which is
        # under MAX_SNIPPET_CHARS by construction. Context parts that no
        # longer fit are skipped, never the match.
        used = len(matched)
        bounded: list[SnippetSegmentV1] = []
        for part in parts:
            if part.matched:
                bounded.append(part)
                continue
            remain = MAX_SNIPPET_CHARS - used
            if remain <= 0 or len(bounded) >= MAX_SEGMENTS:
                continue
            text = part.text if len(part.text) <= remain else part.text[:remain]
            if not text:
                continue
            bounded.append(SnippetSegmentV1(text=text, matched=part.matched))
            used += len(text)
        segments = tuple(bounded)
    if not segments:
        _refuse()
    return SearchSnippetV1(
        schema=SCHEMA_ID,
        schema_version=SCHEMA_VERSION,
        segments=segments,
    )
