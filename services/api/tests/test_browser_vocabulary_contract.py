"""The browser writes out the server's vocabularies by hand, so pin them together.

`apps/web/src/types.ts` repeats string unions the server also declares - setup
readiness codes, job kinds, reference kinds - as a second hand-written copy with
nothing checking the two agree. Adding a member on one side is a silent, ordinary
mistake, and it has already cost something: the comment beside `review_required`
in types.ts records that while that member was missing the browser told people
their machine could not run a workflow that runs fine.

Two things this deliberately does NOT do. It does not require every server
vocabulary to have a browser twin - most have none, and should not. And it does
not treat a shared NAME as proof of a shared meaning: `WorkflowSelectionMode`
means different things on the two sides, and the pair that actually mirror each
other are named differently, so both cases are declared below rather than
guessed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_LOCAL_LM = Path(__file__).resolve().parents[1] / "local_lm"
_TYPES = Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "types.ts"

# Browser type -> the exact server vocabulary it mirrors, as "module.py:Name".
#
# An entry is needed when the two sides spell the name differently, or when the
# server declares that name more than once with different members. Both are
# stated here so a wrong pairing is visible rather than inferred, and a stale
# entry fails the test below instead of quietly excluding a real vocabulary.
_MIRRORS = {
    # The browser's shorter name is the RESPONSE vocabulary. The server also has
    # a WorkflowSelectionMode, with different members, describing how a selection
    # was made rather than what a response reports.
    "WorkflowSelectionMode": "schemas.py:WorkflowSelectionResponseMode",
    # Declared three times on the server; workflow_ownership.py carries a
    # narrower pair. The API surface the browser talks to is the schemas one.
    "WorkflowSelectorCapability": "schemas.py:WorkflowSelectorCapability",
}

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def _literal_union(node: ast.AST) -> tuple[str, set[str]] | None:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return None
    target, value = node.targets[0], node.value
    if not isinstance(target, ast.Name) or not isinstance(value, ast.Subscript):
        return None
    if not (isinstance(value.value, ast.Name) and value.value.id == "Literal"):
        return None
    elements = value.slice.elts if isinstance(value.slice, ast.Tuple) else [value.slice]
    members = {
        element.value
        for element in elements
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    if not members or len(members) != len(elements):
        return None
    return target.id, members


def _string_enum(node: ast.AST) -> tuple[str, set[str]] | None:
    if not isinstance(node, ast.ClassDef):
        return None
    if not any(isinstance(base, ast.Name) and base.id == "StrEnum" for base in node.bases):
        return None
    members = {
        item.value.value
        for item in node.body
        if isinstance(item, ast.Assign)
        and isinstance(item.value, ast.Constant)
        and isinstance(item.value.value, str)
    }
    return (node.name, members) if members else None


def _server_vocabularies() -> dict[str, dict[str, set[str]]]:
    """Every string vocabulary the API package declares, by name then module.

    Both shapes count, because the browser mirrors both: a `Literal` of string
    constants, and a `StrEnum` whose members are string constants.
    """

    found: dict[str, dict[str, set[str]]] = {}
    for path in sorted(_LOCAL_LM.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            named = _literal_union(node) or _string_enum(node)
            if named is None:
                continue
            name, members = named
            found.setdefault(name, {})[path.name] = members
    return found


def _browser_unions() -> dict[str, set[str]]:
    """Exported unions made only of string literals.

    Comments between members are ordinary here, and one of them explains why a
    member exists, so they are stripped rather than tolerated by accident. A
    union carrying anything that is not a quoted string - another type, a
    template - is skipped, because this file compares vocabularies and that is
    not one.
    """

    source = _TYPES.read_text(encoding="utf-8")
    found: dict[str, set[str]] = {}
    for match in re.finditer(r"export type (\w+)\s*=", source):
        rest = source[match.end() :]
        terminator = rest.find(";")
        if terminator == -1:
            continue
        body = _COMMENT.sub(" ", rest[:terminator])
        members = set(re.findall(r'"([^"]*)"', body))
        remainder = re.sub(r'"[^"]*"', "", body).replace("|", "").strip()
        if members and not remainder:
            found[match.group(1)] = members
    return found


def _paired() -> list[tuple[str, set[str], set[str]]]:
    """Each browser union beside the server vocabulary it mirrors."""

    server = _server_vocabularies()
    pairs: list[tuple[str, set[str], set[str]]] = []
    for name, members in sorted(_browser_unions().items()):
        declared = _MIRRORS.get(name)
        if declared is not None:
            module, server_name = declared.split(":")
            pairs.append((f"{name} -> {declared}", server[server_name][module], members))
            continue
        by_module = server.get(name)
        if not by_module:
            continue
        distinct = {frozenset(value) for value in by_module.values()}
        assert len(distinct) == 1, (
            f"{name} is declared with different members in "
            f"{sorted(by_module)}; name the intended one in _MIRRORS"
        )
        pairs.append((name, next(iter(by_module.values())), members))
    return pairs


def test_every_mirrored_vocabulary_agrees_with_the_server() -> None:
    drifted = [
        f"{label}: browser is missing {sorted(server - browser)}, "
        f"browser has extra {sorted(browser - server)}"
        for label, server, browser in _paired()
        if server != browser
    ]

    assert not drifted, (
        "apps/web/src/types.ts has drifted from the vocabulary the server sends:\n"
        + "\n".join(drifted)
    )


def test_the_guard_is_looking_at_something() -> None:
    """A parser that quietly matched nothing would make the test above vacuous.

    The count is a floor rather than today's number, so adding a vocabulary does
    not fail this and losing most of them does.
    """

    assert len(_paired()) >= 15


def test_every_declared_pairing_still_exists() -> None:
    """A stale entry in _MIRRORS would silently stop checking a real vocabulary."""

    server = _server_vocabularies()
    browser = _browser_unions()
    for name, declared in _MIRRORS.items():
        module, server_name = declared.split(":")
        assert name in browser, f"_MIRRORS names {name}, which types.ts no longer exports"
        assert server_name in server, f"_MIRRORS names {declared}, which the server no longer has"
        assert module in server[server_name], (
            f"_MIRRORS points at {declared}, but that name is declared in "
            f"{sorted(server[server_name])}"
        )
