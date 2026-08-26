"""Bounded exact-string helpers for conversation search.

Callers must finish this length ceiling before strip, isspace, closed-set
membership, or casefold.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, NoReturn

MAX_SEARCH_TOKEN_CHARS: Final = 40


def require_bounded_exact_str(
    value: object,
    *,
    max_len: int,
    refuse: Callable[[], NoReturn],
    allow_empty: bool = False,
) -> str:
    """Return an exact built-in str after a finite length ceiling."""
    if type(max_len) is not int or max_len < 1:
        refuse()
    if type(value) is not str:
        refuse()
    assert type(value) is str
    if not allow_empty and not value:
        refuse()
    if len(value) > max_len:
        refuse()
    return value


def casefold_with_origin(body: str) -> tuple[str, tuple[int, ...]]:
    """Casefold body and record the original index of each folded code point."""
    parts: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(body):
        folded = char.casefold()
        parts.append(folded)
        origin.extend((index,) * len(folded))
    return "".join(parts), tuple(origin)


def original_span(
    origin: tuple[int, ...],
    folded_start: int,
    folded_len: int,
    *,
    refuse: Callable[[], NoReturn],
) -> tuple[int, int]:
    """Map a casefold match onto exclusive original [start, end) indices."""
    if type(origin) is not tuple or type(folded_start) is not int or type(folded_len) is not int:
        refuse()
    if folded_start < 0 or folded_len < 1:
        refuse()
    last = folded_start + folded_len - 1
    if last >= len(origin):
        refuse()
    start = origin[folded_start]
    last_orig = origin[last]
    if type(start) is not int or type(last_orig) is not int:
        refuse()
    end = last_orig + 1
    if start < 0 or end <= start:
        refuse()
    return start, end
