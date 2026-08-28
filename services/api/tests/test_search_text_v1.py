from __future__ import annotations

from typing import NoReturn

import pytest

from local_lm.search_text_v1 import (
    MAX_SEARCH_TOKEN_CHARS,
    casefold_with_origin,
    original_span,
    require_bounded_exact_str,
)


class _Refused(ValueError):
    pass


def _refuse() -> NoReturn:
    raise _Refused("refused")


def test_require_bounded_exact_str_pins_cap_and_oversized() -> None:
    cap = "x" * MAX_SEARCH_TOKEN_CHARS
    assert require_bounded_exact_str(cap, max_len=MAX_SEARCH_TOKEN_CHARS, refuse=_refuse) == cap
    with pytest.raises(_Refused):
        require_bounded_exact_str(
            "x" * (MAX_SEARCH_TOKEN_CHARS + 1),
            max_len=MAX_SEARCH_TOKEN_CHARS,
            refuse=_refuse,
        )
    with pytest.raises(_Refused):
        require_bounded_exact_str("x" * 10_000, max_len=MAX_SEARCH_TOKEN_CHARS, refuse=_refuse)
    with pytest.raises(_Refused):
        require_bounded_exact_str("", max_len=MAX_SEARCH_TOKEN_CHARS, refuse=_refuse)
    with pytest.raises(_Refused):
        require_bounded_exact_str(b"ready", max_len=MAX_SEARCH_TOKEN_CHARS, refuse=_refuse)


def test_original_span_maps_expanding_casefold() -> None:
    folded, origin = casefold_with_origin("\u00dfhello world")
    assert folded.startswith("sshello")
    pos = folded.find("hello")
    start, end = original_span(origin, pos, len("hello"), refuse=_refuse)
    assert "\u00dfhello world"[start:end] == "hello"
    inner, inner_origin = casefold_with_origin("he\u00dfo world")
    inner_pos = inner.find("hesso")
    inner_start, inner_end = original_span(inner_origin, inner_pos, len("hesso"), refuse=_refuse)
    assert "he\u00dfo world"[inner_start:inner_end] == "he\u00dfo"
