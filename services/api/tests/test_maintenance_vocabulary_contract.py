"""The empty-chat cleanup page must have words for everything the server can send it.

`test_typescript_contract.py` checks the SHAPE of a payload, which fields exist.
This checks the VOCABULARY inside them, which is where the meaning lives. A reason
that arrives as a slug the page has no wording for is shown raw, in a list that
ends in a delete button, and every such mismatch passes both test suites because
a lookup table can only be tested against the values someone thought to send.

The page may know words the server no longer sends; stale wording is harmless.
The reverse is the defect: every value the server can produce must have text.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from local_lm.empty_chats import (
    CONFIGURED_REASONS,
    CONFLICT_REASONS,
    INCONSISTENT_REASONS,
    EmptyChatExecuteRefusal,
)

COMPONENT = (
    Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "EmptyChatMaintenance.tsx"
)


def _map_keys(source: str, name: str) -> set[str]:
    """The keys of a `const NAME: Record<string, string> = { ... }` literal."""

    start = source.index(f"const {name}")
    body = source[start : source.index("};", start)]
    return set(re.findall(r"^\s{2}(\w+):", body, re.M))


def _switch_cases(source: str, name: str) -> set[str]:
    """The string literals a `function NAME` switches on."""

    start = source.index(f"function {name}")
    body = source[start : source.index("\n}", start)]
    return set(re.findall(r'case "([^"]+)"', body))


@pytest.fixture(scope="module")
def component() -> str:
    return COMPONENT.read_text(encoding="utf-8")


def test_the_extractors_read_something(component: str) -> None:
    """A regex that matched nothing would make every check below vacuously pass."""

    assert len(_map_keys(component, "REASON_TEXT")) >= 5
    assert len(_map_keys(component, "CONFLICT_TEXT")) >= 5
    assert len(_switch_cases(component, "refusalText")) >= 3


def test_every_listing_reason_has_wording(component: str) -> None:
    missing = (set(CONFIGURED_REASONS) | set(INCONSISTENT_REASONS)) - _map_keys(
        component, "REASON_TEXT"
    )
    assert not missing, f"the cleanup page has no wording for listing reasons {sorted(missing)}"


def test_every_conflict_reason_has_wording(component: str) -> None:
    missing = set(CONFLICT_REASONS) - _map_keys(component, "CONFLICT_TEXT")
    assert not missing, f"the confirmation has no wording for conflicts {sorted(missing)}"


def test_every_refusal_has_wording(component: str) -> None:
    missing = {member.value for member in EmptyChatExecuteRefusal} - _switch_cases(
        component, "refusalText"
    )
    assert not missing, f"the cleanup page says nothing useful for refusals {sorted(missing)}"
