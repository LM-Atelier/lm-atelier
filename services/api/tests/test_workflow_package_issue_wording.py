"""The package review must be able to name every finding it can show.

`WorkflowPackageReview.tsx` renders each analyzer finding through a lookup with
a fallback that de-slugs the code. That fallback is reachable - the table had
ten entries and the analyzer emits eleven - so a package whose models are not on
this machine showed `missing asset` in a list where every other row was a
sentence. It is the finding most likely to be there and the only one the surface
could not explain.

The interesting part is why a reader would have concluded the table was
complete. Six of the codes are written as literals at a `WorkflowPackageIssue(...)`
call site. The other five never appear next to that constructor at all: they are
accumulated as `issue_counts["..."]` and built in a loop at the end of
`_asset_references`. Difference the call sites and you get a clean answer and
the wrong one.

So this reads the values rather than the call sites. It is deliberately a
scraper, and deliberately temporary: `WorkflowPackageIssueOut.code` is `str`
while `severity` beside it is a `Literal`, and once the code is a declared
vocabulary (`workflow-package-issue-code-parity`) the schema-derived parity
guard covers this field and this file should be deleted rather than kept beside
it. Two guards over one contract is one guard and one thing to forget.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ANALYZER = Path(__file__).resolve().parents[1] / "local_lm" / "comfy_workflow_packages.py"
COMPONENT = (
    Path(__file__).resolve().parents[3] / "apps" / "web" / "src" / "WorkflowPackageReview.tsx"
)

#: A code written directly at a construction site.
_CONSTRUCTED = re.compile(r"WorkflowPackageIssue\(\s*\n?\s*\"([a-z_]+)\"")
#: A code accumulated into the counter that a later loop constructs from.
_ACCUMULATED = re.compile(r"issue_counts\[\"([a-z_]+)\"\]")


def _analyzer_codes(source: str) -> set[str]:
    """Every code the analyzer can put on an issue, by either route."""

    return set(_CONSTRUCTED.findall(source)) | set(_ACCUMULATED.findall(source))


def _described_codes(source: str) -> set[str]:
    """The keys of the browser's `ISSUE_DESCRIPTIONS` literal."""

    start = source.index("const ISSUE_DESCRIPTIONS")
    body = source[start : source.index("};", start)]
    return set(re.findall(r"^  ([a-z_]+):", body, re.M))


@pytest.fixture(scope="module")
def analyzer() -> str:
    return ANALYZER.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def component() -> str:
    if not COMPONENT.is_file():
        pytest.skip("the web component is not present in this checkout")
    return COMPONENT.read_text(encoding="utf-8")


def test_both_production_routes_are_found(analyzer: str) -> None:
    """Guard the scraper itself, since a silent zero would pass everything.

    If either regular expression stops matching - the constructor is renamed,
    the counter becomes a different name - this test fails instead of the
    parity test quietly comparing against an empty set and reporting success.
    """

    constructed = set(_CONSTRUCTED.findall(analyzer))
    accumulated = set(_ACCUMULATED.findall(analyzer))
    assert constructed, "no directly constructed issue codes found; the scraper is stale"
    assert accumulated, "no accumulated issue codes found; the scraper is stale"
    # The accumulated ones are the point of this file. If they ever fall inside
    # the constructed set the loop has gone, and this test should be revisited
    # rather than left asserting something that no longer describes the code.
    assert accumulated - constructed, "accumulated codes are no longer a separate route"


def test_every_analyzer_finding_has_wording(analyzer: str, component: str) -> None:
    """The direction that bit: a code the reader is shown and cannot read."""

    missing = _analyzer_codes(analyzer) - _described_codes(component)
    assert not missing, (
        f"the package review has no wording for {sorted(missing)}, which the "
        "analyzer can emit; it would render the de-slugged code instead"
    )


def test_no_wording_describes_a_finding_that_cannot_happen(analyzer: str, component: str) -> None:
    """The other direction, kept separate so the two can be judged apart.

    Wording for a code the analyzer never emits is not dangerous the way a
    missing one is - nobody reads it. It is still a description of a state that
    cannot occur, and it is the residue left when a check is removed from the
    analyzer and nobody looks at the browser.
    """

    invented = _described_codes(component) - _analyzer_codes(analyzer)
    assert not invented, (
        f"the package review carries wording for {sorted(invented)}, which the "
        "analyzer never emits; it describes a finding that cannot happen"
    )
