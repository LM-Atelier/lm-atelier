"""The troubleshooting page has to quote the app exactly.

Setup shows a sentence; the user searches for that sentence. If the two drift,
the page is not merely stale - it is unfindable, which is worse than not
documenting the failure at all.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TROUBLESHOOTING = _REPOSITORY_ROOT / "docs" / "TROUBLESHOOTING.md"
_READINESS = Path(__file__).resolve().parents[1] / "local_lm" / "setup_readiness.py"

# Terminal and self-explanatory: this one is documented as prose because its
# message is assembled at runtime from the platform's own reason.
_ASSEMBLED_AT_RUNTIME = {"runtime_unsupported"}


def _documented_failures() -> dict[str, str]:
    """Every failing or pending check the setup report can emit."""
    found: dict[str, str] = {}
    for node in ast.walk(ast.parse(_READINESS.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_check"):
            continue
        literals = [arg.value if isinstance(arg, ast.Constant) else None for arg in node.args]
        if len(literals) < 3:
            continue
        code, status, message = literals[0], literals[1], literals[2]
        if not isinstance(code, str) or status not in {"fail", "pending"}:
            continue
        if isinstance(message, str):
            found[code] = message
    return found


def test_every_setup_failure_is_documented_verbatim() -> None:
    page = _TROUBLESHOOTING.read_text(encoding="utf-8")
    missing: list[str] = []
    for code, message in sorted(_documented_failures().items()):
        if code in _ASSEMBLED_AT_RUNTIME:
            continue
        if code not in page:
            missing.append(f"{code}: code absent from TROUBLESHOOTING.md")
        elif message not in page:
            missing.append(f"{code}: page does not quote {message!r}")

    assert not missing, (
        "TROUBLESHOOTING.md has drifted from the messages the app shows:\n" + "\n".join(missing)
    )


def test_the_guard_is_looking_at_something() -> None:
    """A parser that silently finds nothing would make the test above vacuous."""
    failures = _documented_failures()

    assert len(failures) >= 15
    assert failures["workflow_untrusted"] == "The compatible workflow has not been trusted."


#: Quoted on the page but not a string the app shows. Each one needs a reason,
#: the same way _ASSEMBLED_AT_RUNTIME does: an unexplained entry here is how a
#: genuinely stale quotation gets waved through.
_NOT_APP_TEXT: dict[str, str] = {}

_WEB_SOURCE = _REPOSITORY_ROOT / "apps" / "web" / "src"
_API_SOURCE = Path(__file__).resolve().parents[1] / "local_lm"


def _quoted_on_the_page() -> list[str]:
    """Every double-quoted phrase the page tells the reader to look for."""

    page = _TROUBLESHOOTING.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'"([^"\n]{2,60})"', page)))


def _app_text() -> str:
    """Everything the application itself could render, as one haystack.

    Both halves matter. A control label lives in the web source; a failure
    message can be assembled on either side, and worker failures are currently
    written in both.
    """

    parts: list[str] = []
    for root, suffixes in ((_WEB_SOURCE, (".ts", ".tsx")), (_API_SOURCE, (".py",))):
        for path in sorted(root.rglob("*")):
            if path.suffix in suffixes and path.is_file():
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def test_every_phrase_the_page_quotes_still_exists_in_the_app() -> None:
    """The page quotes controls and messages; a rename must not go unnoticed.

    This is the same argument the setup-failure test above makes, applied to
    the other half of the page. A user reads "Show recent log" here and then
    hunts for that button. Rename the button and the instruction becomes a
    wild goose chase, while every gate stays green - the page is prose, and
    nothing else compares it to the app.
    """

    haystack = _app_text()
    missing = [
        phrase
        for phrase in _quoted_on_the_page()
        if phrase not in _NOT_APP_TEXT and phrase not in haystack
    ]

    assert not missing, (
        "TROUBLESHOOTING.md quotes text the application no longer contains:\n"
        + "\n".join(f"  {phrase!r}" for phrase in missing)
        + "\nRename it on the page, or add it to _NOT_APP_TEXT with a reason."
    )


def test_the_phrase_guard_is_looking_at_something() -> None:
    """A regex that matched nothing, or a haystack that failed to load, would
    make the test above pass for the wrong reason."""

    phrases = _quoted_on_the_page()
    assert len(phrases) >= 8
    assert "Show recent log" in phrases

    haystack = _app_text()
    assert len(haystack) > 500_000
    assert "Copy log folder path" in haystack
    # A phrase of the right shape that the app has never contained must fail,
    # otherwise the containment test proves nothing.
    assert "Show the recent logs" not in haystack


def test_the_page_uses_straight_quotes_so_the_guard_can_see_them() -> None:
    """The guard matches straight quotes only, so curly ones would be invisible.

    That is the failure mode worth pinning: an editor silently substitutes a
    typographic quote, the phrase drops out of _quoted_on_the_page(), and the
    guard above keeps passing while covering one fewer control. Failing here
    is loud and the fix is a one-character edit.
    """

    page = _TROUBLESHOOTING.read_text(encoding="utf-8")
    typographic = sorted(
        {character for character in page if character in "\u201c\u201d\u2018\u2019"}
    )

    assert not typographic, (
        "TROUBLESHOOTING.md contains typographic quotation marks, which the "
        "phrase guard cannot match: "
        + ", ".join(f"U+{ord(character):04X}" for character in typographic)
    )
