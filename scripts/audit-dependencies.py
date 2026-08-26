#!/usr/bin/env python3
"""Decide whether the locked dependencies were really audited.

`pip-audit` answers two different questions and reports them the same way. A
package it checked and found clean, and a package it could not check at all,
both leave the run at exit zero with a line of prose. That is the shape of gate
this repository has just finished closing one level up: a branch ruleset counted
a SKIPPED required check as satisfied, so a pull request could report itself
mergeable with nothing having verified it.

The same substitution is available here. The day a dependency becomes
unresolvable - a yanked release, a private index, a rename - the audit stays
green while auditing strictly less than it did the day before, and nothing says
so.

So this refuses a skip, narrowly. `lm-atelier-api` is skipped on every run and
always will be: it is this repository's own package, it has no published
release, and there is nothing on PyPI to compare it against. Failing on that
would make the gate permanently red, which teaches everyone to ignore it - the
erosion, not the protection. An exact allowlist of distribution names is the
difference between "this specific unpublishable package of ours" and "anything
that looks local", and the second is how a genuinely skipped dependency would
slip through the guard meant to catch it.

The decision is a function so the workflow and a repository test run the same
bytes, rather than one running the rule and the other running a description of
it. Inputs are the parsed report; the exit code is the verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

#: Distributions that cannot be audited and are expected not to be. Exact
#: names, never a prefix or a pattern: this repository's own package has no
#: published release to compare against, and nothing else here has that excuse.
UNAUDITABLE_BY_NATURE = frozenset({"lm-atelier-api"})


def decide(report: Mapping[str, Any], log: list[str]) -> int:
    """Return the audit's exit status, appending its reasoning to `log`."""

    dependencies = report.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        # An empty or unreadable report is not a clean one. Reaching here means
        # the audit ran and told us nothing, which is the state this whole
        # script exists to stop treating as success.
        log.append("The audit report lists no dependencies at all.")
        log.append(
            "Refusing: a report that says nothing is not a report that says clean."
        )
        return 1

    vulnerable: list[str] = []
    unexpected_skips: list[str] = []
    incomplete: list[str] = []
    audited = 0
    for entry in dependencies:
        if not isinstance(entry, dict):
            log.append(f"Unreadable dependency entry: {entry!r}.")
            return 1
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            log.append(f"Dependency entry with no usable name: {entry!r}.")
            return 1
        skip_reason = entry.get("skip_reason")
        if skip_reason is not None:
            if name not in UNAUDITABLE_BY_NATURE:
                unexpected_skips.append(f"{name} ({skip_reason})")
            continue
        # A non-skipped entry has to CARRY its findings and its version.
        # `entry.get("vulns") or []` made a MISSING field indistinguishable from
        # an explicit empty list, so an entry that never reported anything was
        # announced as clean - the same substitution this script exists to
        # refuse, one level further in than the skip it was written to catch.
        # Reported as codex/R1959.
        version = entry.get("version")
        findings = entry.get("vulns")
        if (
            not isinstance(version, str)
            or not version
            or not isinstance(findings, list)
        ):
            incomplete.append(name)
            continue
        audited += 1
        for vulnerability in findings:
            identifier = "unidentified"
            if isinstance(vulnerability, dict):
                identifier = str(vulnerability.get("id") or identifier)
            vulnerable.append(f"{name} {version} {identifier}")

    log.append(f"Audited {audited} of {len(dependencies)} locked distributions.")

    if vulnerable:
        for line in sorted(vulnerable):
            log.append(f"Known vulnerability: {line}.")
        log.append("Refusing: a locked dependency carries a known advisory.")
        return 1

    if incomplete:
        for name in sorted(incomplete):
            log.append(f"Report entry carries no findings or no version: {name}.")
        log.append(
            "Refusing: an entry that did not report its findings is not an entry "
            "that reported none."
        )
        return 1

    if unexpected_skips:
        for line in sorted(unexpected_skips):
            log.append(f"Not audited: {line}.")
        log.append(
            "Refusing: a dependency that could not be audited is not a dependency "
            "that was found clean."
        )
        return 1

    expected = sorted(
        entry["name"]
        for entry in dependencies
        if isinstance(entry, dict) and entry.get("skip_reason") is not None
    )
    if expected:
        log.append(f"Skipped, and expected to be: {', '.join(expected)}.")

    # Every entry skipped means the audit examined NOTHING, and reporting that
    # as clean is the emptiest form of the mistake above. The empty-list case is
    # already refused; this is the same state reached with entries present.
    if not audited:
        log.append("No locked dependency was actually audited.")
        log.append("Refusing: nothing audited is not the same as nothing found.")
        return 1

    log.append("Every auditable locked dependency was audited and is clean.")
    return 0


def _run_pip_audit(argv: Sequence[str]) -> Mapping[str, Any]:
    """Run pip-audit for its REPORT, not for its exit code.

    Its status conflates "found something" with "could not look", which is the
    distinction this script exists to make, so the status is deliberately
    ignored and the JSON is the input.
    """

    completed = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--format", "json", *argv],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.stderr:
        print(completed.stderr.strip(), file=sys.stderr)
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        print("pip-audit did not produce a readable report.", file=sys.stderr)
        print(completed.stdout[-2000:], file=sys.stderr)
        raise SystemExit(1) from None
    if not isinstance(report, dict):
        print("pip-audit produced a report of an unexpected shape.", file=sys.stderr)
        raise SystemExit(1)
    return report


def main() -> int:
    cache = os.environ.get("PIP_AUDIT_CACHE_DIR", "temp/pip-audit-cache")
    os.makedirs(cache, exist_ok=True)
    report = _run_pip_audit(
        ("--skip-editable", "--progress-spinner", "off", "--cache-dir", cache)
    )
    log: list[str] = []
    status = decide(report, log)
    for line in log:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
