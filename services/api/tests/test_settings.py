from __future__ import annotations

import pytest

from local_lm.settings_registry import CHAT_SETTINGS, defaults, resolve_settings, validate_settings


def test_settings_precedence_and_validation() -> None:
    resolved = resolve_settings(
        defaults(CHAT_SETTINGS),
        {"temperature": 0.4, "max_tokens": 500},
        {"temperature": 0.2},
    )
    validated = validate_settings(resolved, CHAT_SETTINGS)
    assert validated["temperature"] == 0.2
    assert validated["max_tokens"] == 500
    assert validated["context_length"] == 8192


def test_unknown_settings_are_not_silently_dropped() -> None:
    with pytest.raises(ValueError, match="unsupported settings"):
        validate_settings({"imaginary_knob": True}, CHAT_SETTINGS)


def test_out_of_range_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="temperature must be at most"):
        validate_settings({"temperature": 99}, CHAT_SETTINGS)
