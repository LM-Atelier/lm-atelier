"""The troubleshooting page has to quote the app exactly.

Setup shows a sentence; the user searches for that sentence. If the two drift,
the page is not merely stale - it is unfindable, which is worse than not
documenting the failure at all.
"""

from __future__ import annotations

import ast
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
