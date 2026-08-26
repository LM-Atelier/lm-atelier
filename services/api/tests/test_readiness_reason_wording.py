"""Every readiness reason the server can send must have wording in the browser.

All three workflow selectors rendered `readiness_reason` straight into the page.
It is a stable machine slug, and the server sets one on every non-ready path, so
the fallback sentences beside it never fired and somebody blocked from running a
workflow read `model_unavailable` in a `role="status"` region.

Nothing caught it because every fixture supplied prose where the server supplies
a slug - `readiness_reason: "Install one model file."` - so each test asserted
the component echoed a value the server never sends. The suite was green and
measuring a world that did not exist.

Sending slugs is right: a stable code survives rewording and can be matched on,
a sentence cannot. This test holds the two ends together instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORDING = Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "readinessReason.ts"
API = Path(__file__).resolve().parents[1] / "local_lm" / "api.py"


def _server_reasons() -> set[str]:
    """The reasons `_workflow_family_variant_out` and its helper can assign.

    Read from the source because they are literals at their assignment sites
    rather than a declared enum. If that ever changes, prefer importing them -
    and the assertion below fails loudly rather than comparing to an empty set.
    """

    source = API.read_text(encoding="utf-8")
    return set(re.findall(r'readiness, reason = "[a-z_]+", "([a-z_]+)"', source)) | set(
        re.findall(r'return "[a-z_]+", "([a-z_]+)"', source)
    )


@pytest.fixture(scope="module")
def browser_reasons() -> set[str]:
    if not WORDING.is_file():
        pytest.skip("the browser wording module is not present in this checkout")
    source = WORDING.read_text(encoding="utf-8")
    block = source[source.index("const REASON_TEXT") : source.index("};")]
    return set(re.findall(r"^\s{2}(\w+):", block, re.M))


def test_the_reason_vocabulary_is_not_empty() -> None:
    """If the extraction stops working, fail here rather than passing forever."""

    found = _server_reasons()
    assert len(found) >= 8, f"only found {sorted(found)}; the extraction is measuring nothing"


def test_every_server_reason_has_browser_wording(browser_reasons: set[str]) -> None:
    missing = _server_reasons() - browser_reasons
    assert not missing, (
        f"the workflow selectors have no wording for {sorted(missing)}; the raw "
        "slug would be rendered into a status region and announced"
    )


#: Reasons the browser can name before this base can send them.
#:
#: `dependency_contract_drift` is the tenth readiness reason, and the browser
#: can already word it while the server cannot yet send it. Wording it early is
#: deliberate: the alternative is a build that renders a raw slug for the
#: window between the server gaining a reason
#: and the browser gaining its sentence, which is the exact defect this file
#: exists to prevent. Delete the entry when the reason lands.
ALLOWED_AHEAD = {"dependency_contract_drift"}


def test_the_browser_invents_no_reason(browser_reasons: set[str]) -> None:
    """Wording for a reason the server cannot send describes an impossible state.

    Softer than the missing direction - a stale entry harms nobody directly -
    but it is how a vocabulary rots, and the fix is to delete a line. Reasons
    that are coming rather than gone are listed in `ALLOWED_AHEAD` with why.
    """

    invented = browser_reasons - _server_reasons() - ALLOWED_AHEAD
    assert not invented, (
        f"the workflow selectors carry wording for {sorted(invented)}, which the server never sends"
    )
