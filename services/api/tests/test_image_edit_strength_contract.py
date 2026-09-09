"""The two copies of the edit-strength classifier must agree on their vocabulary.

`services/api/local_lm/image_edit_strength.py` decides how strongly to edit an
image from what the prompt asks for, and `apps/web/src/imageEditStrength.ts`
decides the same thing again so the panel can show the answer before the run
happens. They are two independent implementations of one rule set - sixteen word
and phrase lists and a table of five strengths - with nothing that noticed if one
of them changed.

The failure that costs a user is quiet. Add a phrase to the server and the
browser predicts a strength the run will not use; the panel says one number, the
image is edited with another, and nothing is red. Nobody would think to check,
because both files look right on their own.

Measured before this test was written: both implementations classify 340 prompts
drawn from the server's entire vocabulary identically, so this pins agreement
that exists rather than announcing a break. What it protects is the next change.

Vocabulary only, and deliberately. Branch ORDER can still diverge and this would
not see it - the same word reaching a different scope because the branches were
reordered. Vocabulary is what actually gets edited, it is checkable statically
without building the browser bundle, and a static check runs in the API suite
where a cross-language runtime comparison could not.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
SERVER_FILE = REPOSITORY / "services" / "api" / "local_lm" / "image_edit_strength.py"
BROWSER_FILE = REPOSITORY / "apps" / "web" / "src" / "imageEditStrength.ts"

# The one name that is not a mechanical translation: the server calls the table
# of per-scope strengths canonical, the browser just calls it the strength.
RENAMED = {"_CANONICAL_STRENGTH": "strength"}


def _browser_name(server_name: str) -> str:
    """`_GLOBAL_PHRASES` -> `globalPhrases`, which is the whole convention."""

    if server_name in RENAMED:
        return RENAMED[server_name]
    head, *rest = server_name.lstrip("_").lower().split("_")
    return head + "".join(part.capitalize() for part in rest)


def _server_vocabularies() -> dict[str, object]:
    """Every module-level constant the classifier consults, read from the source.

    Parsed rather than imported so that a name added to the module is visible
    here even if nothing imports it yet.
    """

    tree = ast.parse(SERVER_FILE.read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("_"):
            continue
        try:
            found[target.id] = ast.literal_eval(node.value)
        except ValueError:
            # The strength table is keyed by EditScope members rather than by
            # literals, so literal_eval refuses it. Reading it anyway matters
            # more than any other entry here: these five numbers ARE the
            # strength the user is shown and the run applies. Silently skipping
            # it would have left the whole point of this file unchecked.
            enum_keyed = _enum_keyed_numbers(node.value)
            if enum_keyed is not None:
                found[target.id] = enum_keyed
    return found


def _enum_keyed_numbers(node: ast.expr) -> dict[str, float] | None:
    """`{EditScope.MINIMAL: 0.38, ...}` -> `{"minimal": 0.38, ...}`."""

    if not isinstance(node, ast.Dict):
        return None
    values: dict[str, float] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if not isinstance(key, ast.Attribute) or not isinstance(value, ast.Constant):
            return None
        if not isinstance(value.value, (int, float)):
            return None
        values[key.attr.lower()] = float(value.value)
    return values


_DECLARATION = re.compile(
    r"^const (?P<name>\w+)(?::[^=]+)? = (?P<body>new Set\(\[.*?\]\)|\[.*?\]|\{.*?\}|[0-9.]+);$",
    re.MULTILINE | re.DOTALL,
)
_STRING = re.compile(r'"([^"]*)"')
_NUMBER_FIELD = re.compile(r"(\w+):\s*([0-9.]+)")


def _browser_vocabularies() -> dict[str, object]:
    source = BROWSER_FILE.read_text(encoding="utf-8")
    found: dict[str, object] = {}
    for match in _DECLARATION.finditer(source):
        body = match.group("body")
        if body.startswith("{"):
            found[match.group("name")] = {
                key: float(value) for key, value in _NUMBER_FIELD.findall(body)
            }
        elif body.startswith(("new Set", "[")):
            found[match.group("name")] = frozenset(_STRING.findall(body))
        else:
            found[match.group("name")] = float(body)
    return found


def _comparable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): float(number) for key, number in value.items()}
    if isinstance(value, (set, frozenset, tuple, list)):
        return frozenset(value)
    return float(value) if isinstance(value, (int, float)) else value


@pytest.fixture(scope="module")
def vocabularies() -> tuple[dict[str, object], dict[str, object]]:
    return _server_vocabularies(), _browser_vocabularies()


def test_every_server_vocabulary_has_a_browser_counterpart(vocabularies) -> None:
    server, browser = vocabularies
    assert server, "no server vocabularies were parsed"

    missing = sorted(name for name in server if _browser_name(name) not in browser)
    assert missing == [], (
        "these classifier constants exist on the server with no browser "
        f"counterpart, so the browser cannot be predicting the same strength: {missing}"
    )


def test_every_browser_vocabulary_has_a_server_counterpart(vocabularies) -> None:
    server, browser = vocabularies
    assert browser, "no browser vocabularies were parsed"

    expected = {_browser_name(name) for name in server}
    extra = sorted(name for name in browser if name not in expected)
    assert extra == [], (
        "these classifier constants exist in the browser with no server "
        f"counterpart, so the server would not honour what the panel shows: {extra}"
    )


def test_both_copies_hold_the_same_words_and_phrases(vocabularies) -> None:
    server, browser = vocabularies
    differences: dict[str, tuple[object, object]] = {}
    for name, value in server.items():
        counterpart = browser.get(_browser_name(name))
        if counterpart is None:
            continue  # named by the totality test above
        if _comparable(value) != _comparable(counterpart):
            differences[name] = (_comparable(value), _comparable(counterpart))

    assert differences == {}, (
        "the two edit-strength classifiers disagree, so the strength the panel "
        f"predicts is not the strength the run will use: {differences}"
    )
