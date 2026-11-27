#!/usr/bin/env python3
"""Decide whether a pull request or merge group has really been verified.

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
# The one plan that runs the API tests.
FULL = "full"

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
# The plans scripts/ci-plan.py can select.
MODES = frozenset({"full", "documentation", "promotion"})
HEAD_SHA = re.compile(r"[0-9a-f]{40}")
# A module count as scripts/api-test-part.py writes it: decimal, with no sign or
# leading zero, and short enough that reading it as a number cannot fail.
COUNT = re.compile(r"0|[1-9][0-9]{0,8}")
# The digest of the API test module list, as the same script writes it.
DIGEST = re.compile(r"[0-9a-f]{64}")
# What each part of the API tests reports: the number of modules in the whole
# suite, the number this part ran, and the digest of the suite's module list.
# Ubuntu compatibility runs the first part, and the Ubuntu API tests job the
# second.
API_TEST_REPORTS = (
    "UBUNTU_API_TOTAL",
    "UBUNTU_API_SELECTED",
    "UBUNTU_API_DIGEST",
    "API_TESTS_TOTAL",
    "API_TESTS_SELECTED",
    "API_TESTS_DIGEST",
)


def every_api_test_ran_once(
    environment: Mapping[str, str], plan_mode: str, head: str, log: list[str]
) -> bool:
    """Return whether the two parts of the API tests ran every module once between them.

    Each part lists the whole suite the way pytest collects it and takes the
    modules whose path assigns them to it, so both jobs succeeding says only
    that what each of them ran passed. A module in neither part would never
    run, and nothing would fail. The parts cover the suite when both report the
    same list, by its length and digest, and their counts add up to its length:
    a module's path alone decides its part, so no module of one list can be in
    both parts of it.
    """

    reports = {name: environment.get(name, "") for name in API_TEST_REPORTS}
    log.append(" ".join(f"{name.lower()}={value}" for name, value in reports.items()))

    # Every other plan skips the steps that would report, so it leaves every
    # report empty. One that is not came from something this gate does not
    # know about.
    if plan_mode != FULL:
        unexpected = [f"{name}={value!r}" for name, value in reports.items() if value]
        if unexpected:
            log.append(f"Unusable authority input: {', '.join(sorted(unexpected))}.")
            log.append(f"Refusing: a {plan_mode} plan runs no API tests to report on.")
            return False
        log.append(f"A {plan_mode} plan runs no API tests.")
        return True

    malformed = [
        f"{name}={value!r}"
        for name, value in reports.items()
        if not (DIGEST if name.endswith("_DIGEST") else COUNT).fullmatch(value)
    ]
    if malformed:
        log.append(f"Unusable authority input: {', '.join(sorted(malformed))}.")
        log.append("Refusing: an unknown input is not a permissive one.")
        return False

    if (
        reports["UBUNTU_API_TOTAL"] != reports["API_TESTS_TOTAL"]
        or reports["UBUNTU_API_DIGEST"] != reports["API_TESTS_DIGEST"]
    ):
        log.append(f"The two parts of the API tests listed different suites for {head}.")
        log.append("Refusing: parts of two different lists do not make up either of them.")
        return False

    total = int(reports["UBUNTU_API_TOTAL"])
    first = int(reports["UBUNTU_API_SELECTED"])
    second = int(reports["API_TESTS_SELECTED"])
    if first == 0 or second == 0:
        log.append(
            f"A part of the API tests ran no modules for {head} "
            f"(the parts ran {first} and {second})."
        )
        log.append("Refusing: a part with nothing in it passes without running a test.")
        return False
    if first + second != total:
        log.append(
            f"The two parts of the API tests ran {first} and {second} of the suite's "
            f"{total} modules for {head}."
        )
        log.append("Refusing: parts that do not add up to the suite left a module out.")
        return False
    log.append(
        f"The two parts of the API tests ran {first} and {second} of the suite's {total} modules."
    )
    return True


def decide(environment: Mapping[str, str], log: list[str]) -> int:
    """Return the gate's exit status, appending its reasoning to `log`."""

    draft = environment.get("DRAFT", "")
    action = environment.get("ACTION", "")
    head = environment.get("HEAD_SHA", "")
    plan = environment.get("PLAN", "")
    plan_mode = environment.get("PLAN_MODE", "")
    ubuntu = environment.get("UBUNTU", "")
    api_tests = environment.get("API_TESTS", "")
    browser_suites = environment.get("BROWSER_SUITES", "")
    windows = environment.get("WINDOWS", "")
    windows_required = environment.get("WINDOWS_REQUIRED", "")
    base_changed = environment.get("BASE_CHANGED", "")
    event = environment.get("EVENT_NAME", "")

    log.append(f"action={action} draft={draft} head={head}")
    log.append(
        f"plan={plan} plan_mode={plan_mode} ubuntu={ubuntu} api_tests={api_tests} "
        f"browser_suites={browser_suites} windows={windows} "
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
    if event == "merge_group":
        base = environment.get("BASE_SHA", "")
        if (
            action != "checks_requested"
            or environment.get("REPOSITORY_PRIVATE") != "false"
            or environment.get("BASE_REF") != "refs/heads/develop"
            or not environment.get("HEAD_REF", "").startswith(
                "refs/heads/gh-readonly-queue/develop/"
            )
            or not HEAD_SHA.fullmatch(base)
            or base == head
            or environment.get("RUN_SHA") != head
        ):
            log.append("Refusing: unusable develop merge group identity.")
            return 1
    elif event != "pull_request":
        log.append("Refusing: unsupported verification event.")
        return 1

    # Merge groups have no PR draft or edited-base fields.
    preconditions = [
        f"{name}={value!r}"
        for name, value, domain in (
            ("DRAFT", draft, BOOLEANS),
            ("ACTION", action, ACTIONS),
            ("BASE_CHANGED", base_changed, BOOLEANS),
        )
        if event == "pull_request" and value not in domain
    ]
    if preconditions:
        log.append(f"Unusable authority input: {', '.join(sorted(preconditions))}.")
        log.append("Refusing: an unknown input is not a permissive one.")
        return 1

    # A draft is not a verified state, and it must not leave a *successful*
    # required context on its head: if the later ready_for_review run is ever
    # lost - the exact failure this job exists for - that commit would still
    # show a green merge gate and could be merged untested.
    if event == "pull_request" and draft == "true":
        log.append("Draft pull request: full verification is deferred until ready_for_review.")
        log.append("Failing closed so this head is never authorized while it is a draft.")
        return 1

    # An `edited` event covers two very different things. Changing the base
    # branch changes which policy applies and what the diff even is, so it must
    # reverify. Editing a title or a body verifies NOTHING, and the jobs above
    # are skipped for it, so authorizing on that would let a text edit stand in
    # for a run.
    if event == "pull_request" and action == "edited" and base_changed != "true":
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
            ("PLAN_MODE", plan_mode, MODES),
            ("PLAN", plan, RESULTS),
            ("UBUNTU", ubuntu, RESULTS),
            ("API_TESTS", api_tests, RESULTS),
            ("BROWSER_SUITES", browser_suites, RESULTS),
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
        log.append(f"The verification plan did not succeed for {head} (result: {plan}).")
        return 1

    if ubuntu != SUCCESS:
        log.append(f"Ubuntu compatibility did not succeed for {head} (result: {ubuntu}).")
        log.append("A skipped or cancelled job is not a passing one.")
        return 1

    # The browser suites run in a job of their own beside Ubuntu compatibility.
    # That job runs for every event this gate can authorize and the plan only
    # decides which of its steps run, so there is no unrequired case: a skipped
    # browser leg is a missing one.
    if browser_suites != SUCCESS:
        log.append(
            f"The Ubuntu browser suites did not succeed for {head} (result: {browser_suites})."
        )
        log.append("A skipped or cancelled job is not a passing one.")
        return 1

    # The second part of the API tests runs in a job of its own too, and like
    # the browser suites' job it runs for every event this gate can authorize,
    # so a skipped one is a missing one.
    if api_tests != SUCCESS:
        log.append(
            f"The second part of the Ubuntu API tests did not succeed for {head} "
            f"(result: {api_tests})."
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

    # PHASE THREE, the evidence that the API tests ran in full. It comes after
    # both jobs that report are known to have succeeded, because a job that
    # failed may have stopped before reporting, and refusing its head as
    # unreadable would hide the real reason.
    if not every_api_test_ran_once(environment, plan_mode, head, log):
        return 1

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
