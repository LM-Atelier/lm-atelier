#!/usr/bin/env python3
"""Decide whether a pull request head has really been verified.

This is the merge gate's whole decision, in one place. The workflow runs it and
the workflow validator runs it, so the rule that ships and the rule under test
are the same bytes rather than two descriptions of one intention.

It used to be a shell script inside `ci.yml`. Testing it then meant finding a
bash that faithfully reports an exit code, and a developer machine without one
skipped the check while still reporting success - the validator could not
evaluate the gate, so it passed. A gate whose test passes when it cannot be run
is the same failure the gate itself exists to prevent, one level up. Python is
already required to run the validator at all, so this seam can never be
unavailable in a context that is checking it.

Inputs arrive as environment variables, matching the job's `env:` block. The
exit code is the verdict: zero authorizes the head, non-zero refuses it.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping

SUCCESS = "success"

# Every value the gate reads has an exact domain. An input that is absent,
# empty, differently cased or simply unrecognised is not a permissive default -
# it is an unknown authority state, and a gate that authorizes on one is open
# in exactly the situation it exists to close. `DRAFT` unset once meant "not a
# draft"; `WINDOWS_REQUIRED` unset meant "Windows is not required". Both
# treated missing evidence as evidence.
BOOLEANS = frozenset({"true", "false"})
ACTIONS = frozenset(
    {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "converted_to_draft",
        "edited",
    }
)
# The conclusions GitHub can report for a needed job.
RESULTS = frozenset({"success", "failure", "cancelled", "skipped"})
HEAD_SHA = re.compile(r"[0-9a-f]{40}")


def decide(environment: Mapping[str, str], log: list[str]) -> int:
    """Return the gate's exit status, appending its reasoning to `log`."""

    draft = environment.get("DRAFT", "")
    action = environment.get("ACTION", "")
    head = environment.get("HEAD_SHA", "")
    plan = environment.get("PLAN", "")
    ubuntu = environment.get("UBUNTU", "")
    windows = environment.get("WINDOWS", "")
    windows_required = environment.get("WINDOWS_REQUIRED", "")
    base_changed = environment.get("BASE_CHANGED", "")

    log.append(f"action={action} draft={draft} head={head}")
    log.append(
        f"plan={plan} ubuntu={ubuntu} windows={windows} "
        f"windows_required={windows_required} base_changed={base_changed}"
    )

    # PHASE ONE, the inputs that decide whether the rest were ever supposed to
    # exist. A draft and a title-or-body edit both legitimately produce EMPTY
    # plan outputs, because the jobs that would have set them were skipped on
    # purpose. So these are validated and answered before the general domain
    # check.
    #
    # That ordering is the whole point and it has been got wrong twice. With
    # the general check first, an empty WINDOWS_REQUIRED refused as "an unknown
    # input" and the branch naming the real reason was UNREACHABLE in
    # production, while the matrix reached it by supplying a shape production
    # never sends. Fail-closed both times, and wrong about why both times.
    preconditions = [
        f"{name}={value!r}"
        for name, value, domain in (
            ("DRAFT", draft, BOOLEANS),
            ("ACTION", action, ACTIONS),
            ("BASE_CHANGED", base_changed, BOOLEANS),
        )
        if value not in domain
    ]
    if preconditions:
        log.append(f"Unusable authority input: {', '.join(sorted(preconditions))}.")
        log.append("Refusing: an unknown input is not a permissive one.")
        return 1

    # A draft is not a verified state, and it must not leave a *successful*
    # required context on its head: if the later ready_for_review run is ever
    # lost - the exact failure this job exists for - that commit would still
    # show a green merge gate and could be merged untested.
    if draft == "true":
        log.append(
            "Draft pull request: full verification is deferred until ready_for_review."
        )
        log.append(
            "Failing closed so this head is never authorized while it is a draft."
        )
        return 1

    # An `edited` event covers two very different things. Changing the base
    # branch changes which policy applies and what the diff even is, so it must
    # reverify. Editing a title or a body verifies NOTHING, and the jobs above
    # are skipped for it, so authorizing on that would let a text edit stand in
    # for a run.
    if action == "edited" and base_changed != "true":
        log.append("Edited event that did not change the base branch.")
        log.append("Refusing: a title or body edit is not evidence about this head.")
        return 1

    # PHASE TWO, the verification results. Reaching here means a real run was
    # supposed to happen, so an empty or unrecognised result is genuinely
    # anomalous rather than expected.
    malformed = [
        f"{name}={value!r}"
        for name, value, domain in (
            ("WINDOWS_REQUIRED", windows_required, BOOLEANS),
            ("PLAN", plan, RESULTS),
            ("UBUNTU", ubuntu, RESULTS),
            ("WINDOWS", windows, RESULTS),
        )
        if value not in domain
    ]
    if not HEAD_SHA.fullmatch(head):
        malformed.append(f"HEAD_SHA={head!r}")
    if malformed:
        log.append(f"Unusable authority input: {', '.join(sorted(malformed))}.")
        log.append("Refusing: an unknown input is not a permissive one.")
        return 1

    if plan != SUCCESS:
        log.append(
            f"The verification plan did not succeed for {head} (result: {plan})."
        )
        return 1

    if ubuntu != SUCCESS:
        log.append(
            f"Ubuntu compatibility did not succeed for {head} (result: {ubuntu})."
        )
        log.append("A skipped or cancelled job is not a passing one.")
        return 1

    # Windows is conditional: the plan decides whether this change needs it.
    # Demanding it unconditionally would fail every documentation change, and
    # ignoring it would let the gate go green while a plan-required Windows leg
    # was still running or had already failed.
    if windows_required == "true":
        if windows != SUCCESS:
            log.append(
                "The plan required Windows compatibility and it did not succeed "
                f"(result: {windows})."
            )
            return 1
    else:
        log.append("The plan did not require Windows compatibility for this change.")

    log.append(f"Merge gate green for {head}.")
    return 0


def main() -> int:
    log: list[str] = []
    status = decide(os.environ, log)
    for line in log:
        print(line)
    return status


if __name__ == "__main__":
    sys.exit(main())
