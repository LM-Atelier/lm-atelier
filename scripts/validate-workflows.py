"""Validate GitHub workflow syntax and repository security policy."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml

WORKFLOW_ROOT = Path(".github/workflows")
PUBLIC_CONFIGURATION = Path("scripts/configure-public-repository.ps1")
# The rulesets the public configuration script applies to the repository.
RULESET_ROOT = Path(".github/rulesets")
# Why the workflow policy reads the rulesets, given with each refusal of one.
RULESET_USE = "job names outside ci.yml are compared with the checks a ruleset requires"
FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
VERSION_COMMENT = re.compile(r"v[0-9]+(?:\.[0-9]+){1,2}(?:-[0-9A-Za-z.-]+)?")
# The audit comment a pinned reference carries, matched where its value ends:
# whitespace, `#`, the version, and nothing else before the end of that line.
TRAILING_VERSION = re.compile(r"[ \t]+#[ \t]*(?P<version>\S+)[ \t]*\r?$", re.MULTILINE)
STRING_TAG = "tag:yaml.org,2002:str"
MERGE_TAG = "tag:yaml.org,2002:merge"
NULL_TAG = "tag:yaml.org,2002:null"
# What opens a GitHub expression, whose value is known only once the workflow runs.
EXPRESSION_START = "${{"
PULL_REQUEST_TARGET = re.compile(r"^\s*pull_request_target\s*:", re.MULTILINE)
ALLOWED_ACTIONS_BLOCK = re.compile(
    r"^\$AllowedActionPatterns\s*=\s*@\(\s*$"
    r"(?P<body>.*?)"
    r"^\)\s*$",
    re.MULTILINE | re.DOTALL,
)
QUOTED_ACTION = re.compile(
    r'^\s*"(?P<target>[^"]+@[0-9a-f]{40})",?\s*$',
    re.MULTILINE,
)


def iter_permissions(
    value: Any,
    location: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], dict[str, Any] | str]]:
    """Return every workflow- or job-level permissions declaration."""

    declarations: list[tuple[tuple[str, ...], dict[str, Any] | str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "permissions" and isinstance(child, (dict, str)):
                declarations.append((location + (str(key),), child))
            declarations.extend(iter_permissions(child, location + (str(key),)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            declarations.extend(iter_permissions(child, location + (str(index),)))
    return declarations


def validate_permissions(path: Path, workflow: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    top_level = workflow.get("permissions")
    if top_level != {"contents": "read"}:
        errors.append(f"{path}: top-level permissions must be exactly contents: read")

    allowed_writes: dict[tuple[str, ...], set[str]] = {
        (
            "jobs",
            "release-candidate",
            "permissions",
        ): {
            "id-token",
            "attestations",
            "artifact-metadata",
        },
        (
            "jobs",
            "draft-release",
            "permissions",
        ): {
            "contents",
        },
    }
    for location, declaration in iter_permissions(workflow):
        if declaration == "read-all":
            continue
        if isinstance(declaration, str):
            errors.append(f"{path}: unsupported permissions declaration {declaration!r}")
            continue
        for scope, access in declaration.items():
            reviewed_write = (
                path.name == "release.yml"
                and scope in allowed_writes.get(location, set())
                and access == "write"
            )
            if reviewed_write:
                continue
            if access not in {"read", "none"}:
                errors.append(f"{path}: {scope} permission must not grant {access!r} access")
    return errors


def mapping_pairs(node: yaml.Node) -> list[tuple[yaml.Node, yaml.Node]]:
    """Return a mapping's key-value pairs, with every pair a `<<` merge key adds.

    Loading a mapping that merges others in gives it their pairs as well, so a
    pair that arrives that way is checked like one written in place. A node
    that is not a mapping has no pairs.
    """

    pairs: list[tuple[yaml.Node, yaml.Node]] = []
    pending = [node]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        # A mapping can merge itself in through an alias, so each is read once.
        if not isinstance(current, yaml.MappingNode) or id(current) in visited:
            continue
        visited.add(id(current))
        for key, value in current.value:
            if key.tag == MERGE_TAG:
                pending.extend(value.value if isinstance(value, yaml.SequenceNode) else [value])
            else:
                pairs.append((key, value))
    return pairs


def string_value(node: yaml.Node) -> str | None:
    """Return a scalar's text when it loads as a string, however it is quoted."""

    if isinstance(node, yaml.ScalarNode) and node.tag == STRING_TAG:
        return str(node.value)
    return None


def values_named(node: yaml.Node, name: str) -> list[yaml.Node]:
    """Return the value of every pair whose key loads as the string `name`.

    A mapping can repeat a key. A loader keeps only the last occurrence, so
    every occurrence is returned rather than the first.
    """

    return [value for key, value in mapping_pairs(node) if string_value(key) == name]


def action_references(content: str) -> list[yaml.Node]:
    """Return the value of every `uses` a workflow's jobs and steps declare, in file order.

    The document is composed rather than read line by line, so a value on the
    line after its key, a block or quoted value, a flow mapping, a quoted or
    repeated key and an alias are all found, just as the one-line spelling is.
    A job's own `uses` calls a reusable workflow and a step's runs an action.
    Composed nodes keep their place in the file, which is where an audit
    comment is looked for. A value that aliases reach more than once is
    returned once. Raises yaml.YAMLError for a document that cannot be composed.
    """

    root = yaml.compose(content, Loader=yaml.SafeLoader)
    if not isinstance(root, yaml.MappingNode):
        return []
    found: dict[int, yaml.Node] = {}
    for jobs in values_named(root, "jobs"):
        for _name, job in mapping_pairs(jobs):
            references = values_named(job, "uses")
            for steps in values_named(job, "steps"):
                if isinstance(steps, yaml.SequenceNode):
                    for step in steps.value:
                        references.extend(values_named(step, "uses"))
            found.update((id(reference), reference) for reference in references)
    return sorted(found.values(), key=lambda reference: int(reference.start_mark.index))


def validate_action_pins(path: Path, content: str) -> list[str]:
    """Refuse an action or reusable workflow reference that is not pinned and audited.

    Every reference a job or step makes is read from the composed document,
    so no spelling of one goes unchecked, and each must be a plain, unquoted
    string, so that the text a reviewer reads, with its audit comment straight
    after it, is exactly the reference that runs, with no escape, quote or
    block indicator between them. A local reference needs nothing more and a
    docker image is refused. Any other must name a full commit SHA and say which version that
    commit is in a comment straight after the value, on its line, where an
    update to the pin changes the version beside it. A document that cannot be
    composed is reported rather than passed over.
    """

    try:
        references = action_references(content)
    except yaml.YAMLError as error:
        return [f"{path}: invalid YAML: {error}"]
    errors: list[str] = []
    for node in references:
        line_number = node.start_mark.line + 1
        plain = isinstance(node, yaml.ScalarNode) and node.tag == STRING_TAG and node.style is None
        if not plain:
            errors.append(
                f"{path}:{line_number}: uses must be a plain, unquoted string "
                "naming an action or workflow"
            )
            continue
        target = str(node.value)
        if target.startswith("docker://"):
            errors.append(
                f"{path}:{line_number}: docker actions are prohibited; "
                "use a reviewed SHA-pinned action"
            )
            continue
        if target.startswith("./"):
            continue
        action, separator, reference = target.rpartition("@")
        if not separator or not action or not FULL_COMMIT.fullmatch(reference):
            errors.append(f"{path}:{line_number}: external action must use a full commit SHA")
        comment = TRAILING_VERSION.match(content, node.end_mark.index)
        if comment is None or not VERSION_COMMENT.fullmatch(comment.group("version")):
            errors.append(f"{path}:{line_number}: pinned action needs an auditable version comment")
    return errors


def external_actions(content: str) -> set[str]:
    """Return every external action and reusable workflow reference in one workflow.

    The references come from the same composed walk the pin check reads, so
    the allowlist is compared with each of them however it is spelled. The
    line break a block value keeps and the spaces a quoted one carries are no
    part of the reference: the pin check already refuses those spellings, and
    the allowlist reports the reference as a reader sees it.
    """

    texts = (text.strip() for text in map(string_value, action_references(content)) if text)
    return {text for text in texts if text and not text.startswith(("./", "docker://"))}


def validate_action_allowlist(workflow_actions: set[str]) -> list[str]:
    """Keep the applied GitHub allowlist equal to the reviewed workflow pins."""

    content = PUBLIC_CONFIGURATION.read_text(encoding="utf-8")
    block = ALLOWED_ACTIONS_BLOCK.search(content)
    if block is None:
        return [f"{PUBLIC_CONFIGURATION}: missing $AllowedActionPatterns declaration"]
    configured = {match.group("target") for match in QUOTED_ACTION.finditer(block.group("body"))}
    if configured == workflow_actions:
        return []
    missing = sorted(workflow_actions - configured)
    unused = sorted(configured - workflow_actions)
    details: list[str] = []
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unused:
        details.append(f"unused {', '.join(unused)}")
    return [
        (
            f"{PUBLIC_CONFIGURATION}: selected Actions allowlist is out of sync "
            f"with workflows ({'; '.join(details)})"
        )
    ]


def validate_checkout_credentials(path: Path, workflow: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return [f"{path}: workflow must define jobs"]
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("uses")
            if not isinstance(action, str) or not action.startswith("actions/checkout@"):
                continue
            inputs = step.get("with")
            if not isinstance(inputs, dict) or inputs.get("persist-credentials") is not False:
                errors.append(
                    f"{path}: checkout in job {job_name} must disable persisted credentials"
                )
    return errors


def validate_environment_contexts(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Reject contexts GitHub does not expose to workflow- or job-level env."""

    errors: list[str] = []
    declarations: list[tuple[str, Any]] = [("workflow", workflow.get("env"))]
    jobs = workflow.get("jobs")
    if isinstance(jobs, dict):
        declarations.extend(
            (f"job {job_name}", job.get("env"))
            for job_name, job in jobs.items()
            if isinstance(job, dict)
        )
    for location, declaration in declarations:
        if not isinstance(declaration, dict):
            continue
        for name, value in declaration.items():
            if isinstance(value, str) and "${{ runner." in value:
                errors.append(f"{path}: {location} env {name} cannot use the runner context")
    return errors


def validate_untrusted_triggers(path: Path, content: str) -> list[str]:
    errors: list[str] = []
    if PULL_REQUEST_TARGET.search(content):
        errors.append(f"{path}: pull_request_target is prohibited for untrusted contribution CI")
    if path.name == "ci.yml" and "run-ci" in content:
        errors.append(f"{path}: normal pull-request CI must not require the run-ci label")
    return errors


# The merge gate is the only required context that cannot be satisfied by a job
# not running, so it is the only thing standing between a draft-skipped pull
# request and a merge with no verification. Checking that it *exists* would be
# checking the wrong thing - the first version of it existed and still exited
# zero on a draft, which handed the draft head a green required context. So this
# runs the gate's own script against the event/result matrix and asserts the
# conclusions.
# A fully valid, fully green environment. The malformed-authority cases below
# start here and break exactly one field, so each one isolates its own input.
GREEN_MERGE_GATE = {
    "EVENT_NAME": "pull_request",
    "DRAFT": "false",
    "ACTION": "synchronize",
    "HEAD_SHA": "0" * 40,
    "PLAN": "success",
    "UBUNTU": "success",
    "BROWSER_SUITES": "success",
    "WINDOWS": "success",
    "WINDOWS_REQUIRED": "true",
}

GREEN_MERGE_GROUP: dict[str, str | None] = {
    "EVENT_NAME": "merge_group",
    "REPOSITORY_PRIVATE": "false",
    "ACTION": "checks_requested",
    "BASE_REF": "refs/heads/develop",
    "BASE_SHA": "1" * 40,
    "HEAD_REF": "refs/heads/gh-readonly-queue/develop/pr-1-" + "0" * 40,
    "HEAD_SHA": "0" * 40,
    "RUN_SHA": "0" * 40,
    "DRAFT": None,
    "BASE_CHANGED": None,
    "PLAN": "success",
    "UBUNTU": "success",
    "BROWSER_SUITES": "success",
    "WINDOWS": "success",
    "WINDOWS_REQUIRED": "true",
}

# A case is a label, the environment it runs under, and the exit status the
# gate must produce. A value of None means the variable is absent entirely,
# which is a different assertion from it being empty.
MERGE_GATE_MATRIX: tuple[tuple[str, dict[str, str | None], int], ...] = (
    ("verified merge group without PR-only fields", GREEN_MERGE_GROUP, 0),
    (
        "merge group with unrequired Windows",
        {**GREEN_MERGE_GROUP, "WINDOWS": "skipped", "WINDOWS_REQUIRED": "false"},
        0,
    ),
    *(
        (f"merge group refuses {name}={value!r}", {**GREEN_MERGE_GROUP, name: value}, 1)
        for name, value in (
            ("EVENT_NAME", None),
            ("EVENT_NAME", "push"),
            ("ACTION", "destroyed"),
            ("REPOSITORY_PRIVATE", None),
            ("REPOSITORY_PRIVATE", "true"),
            ("BASE_REF", "refs/heads/main"),
            ("BASE_REF", None),
            ("BASE_SHA", None),
            ("BASE_SHA", "0" * 40),
            ("HEAD_REF", "refs/heads/develop"),
            ("HEAD_SHA", "0" * 39),
            ("RUN_SHA", "2" * 40),
            ("PLAN", "skipped"),
            ("PLAN", "failure"),
            ("PLAN", "cancelled"),
            ("UBUNTU", "skipped"),
            ("UBUNTU", "failure"),
            ("UBUNTU", "cancelled"),
            ("BROWSER_SUITES", "skipped"),
            ("BROWSER_SUITES", "failure"),
            ("BROWSER_SUITES", "cancelled"),
            ("BROWSER_SUITES", None),
            ("WINDOWS", "skipped"),
            ("WINDOWS", "failure"),
            ("WINDOWS", "cancelled"),
            ("WINDOWS_REQUIRED", None),
        )
    ),
    (
        # Production's shape, not a synthetic green one: the plan is skipped,
        # Ubuntu runs on always() and refuses, and the browser suites refuse the
        # same way. WINDOWS_REQUIRED is absent because the job that sets it
        # never ran.
        "a title or body edit must not authorize",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "false",
            "DRAFT": "false",
            "PLAN": "skipped",
            "UBUNTU": "failure",
            "BROWSER_SUITES": "failure",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": None,
        },
        1,
    ),
    (
        "a title or body edit must not authorize even if everything went green",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "false",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    (
        "an edited event that changed the base reverifies and may authorize",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "true",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        0,
    ),
    (
        # On a synchronize the answer is not READ, so only the domain check can
        # refuse it. Without this case that check has no consequence any test
        # can observe, and deleting it would look free.
        "an unreadable base-change answer is refused even when it is not read",
        {
            "ACTION": "synchronize",
            "BASE_CHANGED": "maybe",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    (
        "an unreadable base-change answer is not a permissive one",
        {
            "ACTION": "edited",
            "BASE_CHANGED": None,
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    # (label, env, expected exit code); a None value removes the variable
    (
        "draft must fail closed",
        {
            "DRAFT": "true",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "draft fails even when nothing else ran",
        {
            "DRAFT": "true",
            "PLAN": "skipped",
            "UBUNTU": "skipped",
            "BROWSER_SUITES": "skipped",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "skipped plan is not a passing plan",
        {
            "DRAFT": "false",
            "PLAN": "skipped",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "skipped Ubuntu is not a passing Ubuntu",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "skipped",
            "BROWSER_SUITES": "success",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "cancelled Ubuntu is not a passing Ubuntu",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "cancelled",
            "BROWSER_SUITES": "success",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "failed Ubuntu fails the gate",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "failure",
            "BROWSER_SUITES": "success",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "skipped browser suites are not passing ones",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "skipped",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "cancelled browser suites are not passing ones",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "cancelled",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "failed browser suites fail the gate",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "failure",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        1,
    ),
    (
        "Windows may be absent when the plan does not require it",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "false",
        },
        0,
    ),
    (
        "a plan-required Windows must not be skipped",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "skipped",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    (
        "a plan-required Windows must not have failed",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "failure",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    (
        "everything the plan required succeeded",
        {
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        0,
    ),
    # Authority the gate cannot read is not authority it may assume. Each case
    # below starts from the fully green environment above and breaks exactly
    # one input. `None` removes the variable entirely.
    #
    # Unset `DRAFT` once meant "not a draft" and unset `WINDOWS_REQUIRED` meant
    # "Windows is not required", so a job that failed to bind its environment
    # would have been authorized by the absence of the evidence against it.
    *(
        (label, {**GREEN_MERGE_GATE, **override}, 1)
        for label, override in (
            ("a missing draft state is not a non-draft", {"DRAFT": None}),
            ("an empty draft state is not a non-draft", {"DRAFT": ""}),
            ("a capitalised draft state is not recognised", {"DRAFT": "False"}),
            (
                "a missing Windows policy is not 'not required'",
                {"WINDOWS_REQUIRED": None},
            ),
            ("a non-boolean Windows policy is refused", {"WINDOWS_REQUIRED": "yes"}),
            ("a missing action is refused", {"ACTION": None}),
            ("an unsubscribed action is refused", {"ACTION": "closed"}),
            ("a missing head is refused", {"HEAD_SHA": None}),
            ("a truncated head is refused", {"HEAD_SHA": "0" * 39}),
            ("an uppercase head is refused", {"HEAD_SHA": "A" * 40}),
            ("an unknown plan conclusion is refused", {"PLAN": "neutral"}),
            ("an uppercase Ubuntu conclusion is refused", {"UBUNTU": "SUCCESS"}),
            ("a missing browser-suites conclusion is refused", {"BROWSER_SUITES": None}),
            (
                "an uppercase browser-suites conclusion is refused",
                {"BROWSER_SUITES": "SUCCESS"},
            ),
            (
                "an unknown browser-suites conclusion is refused",
                {"BROWSER_SUITES": "timed_out"},
            ),
            ("an unknown Windows conclusion is refused", {"WINDOWS": "timed_out"}),
            # The one case where a result's domain is what refuses rather than
            # the success comparison: an unrequired Windows leg is never
            # examined, so without a domain check an unreadable conclusion
            # would sail through unread.
            (
                "an unreadable Windows conclusion is refused even when unrequired",
                {"WINDOWS": "timed_out", "WINDOWS_REQUIRED": "false"},
            ),
        )
    ),
    # Converting back to draft has to withdraw authorization even though every
    # job in this run reports success. Without the trigger the workflow never
    # re-runs and the previous green stays attached; with it, this case is what
    # replaces that green with a refusal.
    (
        "converted back to draft withdraws a fully green head",
        {
            "ACTION": "converted_to_draft",
            "DRAFT": "true",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    # An `edited` event is how a base-branch change arrives. The verdict must
    # come from this run's results rather than from which action delivered it,
    # so the same inputs decide the same way under a different action.
    (
        "an edited base is authorized only by this run's results",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "true",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        0,
    ),
    (
        "an edited base does not excuse a missing Ubuntu leg",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "true",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "skipped",
            "BROWSER_SUITES": "success",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
    (
        "an edited base does not excuse missing browser suites",
        {
            "ACTION": "edited",
            "BASE_CHANGED": "true",
            "DRAFT": "false",
            "PLAN": "success",
            "UBUNTU": "success",
            "BROWSER_SUITES": "skipped",
            "WINDOWS": "success",
            "WINDOWS_REQUIRED": "true",
        },
        1,
    ),
)


MERGE_GATE_SCRIPT = Path("scripts/ci-merge-gate.py")
MERGE_GATE_COMMAND = "python scripts/ci-merge-gate.py"
MERGE_GATE_CONDITION = (
    "always() && (github.event_name == 'pull_request' || github.event_name == 'merge_group')"
)


def event_expression(field: str, fallback: str) -> str:
    """Select a merge-group field or the existing PR/dispatch binding."""

    return (
        "${{ github.event_name == 'merge_group' && github.event.merge_group."
        + field
        + " || "
        + fallback
        + " }}"
    )


EVENT_HEAD_SHA = event_expression("head_sha", "github.event.pull_request.head.sha")
EVENT_BASE_SHA = event_expression("base_sha", "github.event.pull_request.base.sha")
EVENT_BASE_REF = event_expression("base_ref", "github.base_ref")
EVENT_HEAD_REF = event_expression("head_ref", "github.head_ref")
VERIFICATION_REF = event_expression(
    "head_sha",
    "github.event_name == 'pull_request' && github.event.pull_request.head.sha || github.sha",
)

# The whole job, declared rather than sampled. Six rounds of review on this file
# established that any check phrased as "the fields I thought of are correct"
# accepts the field nobody thought of, and GitHub has plenty: a job-level
# `continue-on-error` tolerates the decision failing, `defaults.run.shell`,
# `container` and a different `runs-on` all change what executes without
# touching a single pinned step field. So the schema is exhaustive and equality
# is the test: unknown keys are refused because they are unknown.
MERGE_GATE_JOB = {
    "name": "Merge gate",
    "if": MERGE_GATE_CONDITION,
    "needs": ["verification-plan", "compatibility", "windows-compatibility", "browser-suites"],
    "runs-on": "ubuntu-24.04",
    "timeout-minutes": 5,
}
# The decision's whole authority, bound expression by expression. A key that
# goes missing becomes an unreadable input rather than a permissive default,
# and one that reads the wrong expression answers about a different thing.
MERGE_GATE_ENV = {
    "EVENT_NAME": "${{ github.event_name }}",
    "REPOSITORY_PRIVATE": "${{ github.event.repository.private }}",
    "BASE_REF": EVENT_BASE_REF,
    "BASE_SHA": EVENT_BASE_SHA,
    "HEAD_REF": EVENT_HEAD_REF,
    "RUN_SHA": "${{ github.sha }}",
    "DRAFT": "${{ github.event.pull_request.draft }}",
    "ACTION": "${{ github.event.action }}",
    "HEAD_SHA": EVENT_HEAD_SHA,
    "PLAN": "${{ needs.verification-plan.result }}",
    "UBUNTU": "${{ needs.compatibility.result }}",
    "BROWSER_SUITES": "${{ needs.browser-suites.result }}",
    "WINDOWS": "${{ needs.windows-compatibility.result }}",
    "WINDOWS_REQUIRED": "${{ needs.verification-plan.outputs.windows }}",
    # Which KIND of edited event this is. Without it the decision cannot
    # tell a base change, which must reverify, from a title or body edit,
    # which verified nothing.
    "BASE_CHANGED": (
        "${{ github.event.changes.base.ref.from != ''"
        " || github.event.changes.base.sha.from != '' }}"
    ),
}


# The gate runs a tracked file, so the job has to fetch that file before it can
# decide anything, and it must fetch the copy belonging to the head under
# judgement. Without these two steps the job fails for want of a script rather
# than for want of verification - which fails closed, but can never authorize a
# valid head either, so the gate is useless in both directions.
class MergeGateSetupStep(TypedDict):
    """One step the merge-gate job must run before it can decide anything."""

    label: str
    missing: str
    action: str
    ref: str
    required_with: dict[str, str | bool]


MERGE_GATE_SETUP: tuple[MergeGateSetupStep, ...] = (
    {
        "label": "checkout",
        "missing": f"so {MERGE_GATE_SCRIPT.name} is not present when the job runs",
        "action": "actions/checkout",
        "ref": "3d3c42e5aac5ba805825da76410c181273ba90b1",
        "required_with": {
            # The PR head or generated merge-group head being judged.
            "ref": EVENT_HEAD_SHA,
            # This job reads a verdict. It never needs to write, and a token
            # left on disk is reachable by anything the head runs later.
            "persist-credentials": False,
        },
    },
    {
        "label": "python setup",
        "missing": "so the gate would run on whatever interpreter the runner happens to have",
        "action": "actions/setup-python",
        "ref": "5fda3b95a4ea91299a34e894583c3862153e4b97",
        "required_with": {"python-version": "3.12"},
    },
)

# Each step in full, in order. The `with` and `env` mappings are compared whole,
# so an added input is refused the same way a changed one is.
MERGE_GATE_STEPS = (
    {
        "uses": f"{MERGE_GATE_SETUP[0]['action']}@{MERGE_GATE_SETUP[0]['ref']}",
        "with": dict(MERGE_GATE_SETUP[0]["required_with"]),
    },
    {
        "uses": f"{MERGE_GATE_SETUP[1]['action']}@{MERGE_GATE_SETUP[1]['ref']}",
        "with": dict(MERGE_GATE_SETUP[1]["required_with"]),
    },
    {
        "name": "Require real verification for this exact head",
        "env": dict(MERGE_GATE_ENV),
        "run": MERGE_GATE_COMMAND,
    },
)


def exactly_equal(expected: Any, actual: Any) -> bool:
    """Equality that also requires the same type, all the way down.

    Python treats `False == 0` and `True == 1`, so a YAML
    `persist-credentials: 0` satisfies an expected `False` under ordinary
    mapping equality, and a pinned `timeout-minutes: 5` is satisfied by `True`.
    For values whose whole purpose is to be pinned, that is a bypass rather
    than a convenience, and it hides inside nested `with` and `env` mappings
    where a top-level type check never looks.
    """
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, dict):
        if len(expected) != len(actual):
            return False
        # Keys need the same treatment as values, and a set comparison cannot
        # give it: {False} == {0} and {True} == {1}, so two mappings with
        # differently typed keys compare equal and even collapse into one
        # entry. Match each key to one of the same type, consuming it, so a
        # single actual entry cannot satisfy two expected ones.
        remaining = list(actual.items())
        for key, value in expected.items():
            found = next(
                (
                    index
                    for index, (other, _) in enumerate(remaining)
                    if type(other) is type(key) and other == key
                ),
                None,
            )
            if found is None:
                return False
            if not exactly_equal(value, remaining.pop(found)[1]):
                return False
        return True
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return False
        return all(exactly_equal(item, other) for item, other in zip(expected, actual, strict=True))
    return bool(expected == actual)


def validate_merge_gate_schema(path: Path, job: dict[str, Any]) -> list[str]:
    """Compare the whole job against its declaration, key for key.

    Everything here is equality against an exhaustive schema rather than a set
    of individual assertions. The difference matters: an assertion answers the
    question it was given, while equality answers the question nobody thought
    to ask. `continue-on-error` at job level is the example that made this
    necessary - it lets the decision fail and the job succeed, and no amount of
    checking the fields we already pin would have noticed it.
    """
    problems: list[str] = []
    expected_keys = set(MERGE_GATE_JOB) | {"steps"}
    unexpected = sorted(set(job) - expected_keys)
    if unexpected:
        problems.append(
            f"{path}: merge-gate declares {unexpected}, which can change whether the "
            "decision runs or whether its result is the job's answer"
        )
    for key, want in MERGE_GATE_JOB.items():
        got = job.get(key)
        if not exactly_equal(want, got):
            problems.append(f"{path}: merge-gate {key} must be exactly {want!r}, found {got!r}")

    steps = job.get("steps")
    if not isinstance(steps, list) or len(steps) != len(MERGE_GATE_STEPS):
        found = len(steps) if isinstance(steps, list) else steps
        return problems + [
            (
                f"{path}: merge-gate must have exactly {len(MERGE_GATE_STEPS)} steps - "
                f"its audited setup then one decision - found {found!r}"
            )
        ]
    for index, (step, want_step) in enumerate(zip(steps, MERGE_GATE_STEPS, strict=True)):
        if not isinstance(step, dict):
            problems.append(f"{path}: merge-gate step {index} is not a mapping")
            continue
        extra = sorted(set(step) - set(want_step))
        if extra:
            problems.append(
                f"{path}: merge-gate step {index} declares {extra}; nothing may alter how "
                "or whether that step runs"
            )
        for key, want in want_step.items():
            got = step.get(key)
            if isinstance(want, str) and isinstance(got, str):
                got = got.strip()
            if not exactly_equal(want, got):
                problems.append(
                    f"{path}: merge-gate step {index} {key} must be exactly {want!r}, found {got!r}"
                )
    return problems


def validate_merge_gate_steps(path: Path, job: dict[str, Any]) -> list[str]:
    """The gate's steps must fetch the decision before making it.

    The job is bound exactly rather than sampled. Order and duplicates matter -
    a second checkout can replace the script that is about to run, and a
    setup-python after the decision provisions an interpreter nothing uses -
    but so does everything else in the list. An extra step between the setup
    and the decision runs with the gate's authority, and an execution control
    on the decision step can detach its exit code from what the job concludes.
    `continue-on-error: true` is the sharpest of those: the step fails, the job
    succeeds, and the gate reports green having refused.
    """
    problems: list[str] = []
    steps = job.get("steps") or []
    # A malformed `steps:` must produce a diagnostic, not a traceback. Anything
    # that is not a list of mappings crashed this function before it could
    # report, which turns a policy violation into a stack trace and loses the
    # message the operator needs.
    if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
        return problems + [
            f"{path}: merge-gate steps must be a list of mappings, found {type(steps).__name__}"
        ]
    if len(steps) != len(MERGE_GATE_SETUP) + 1:
        problems.append(
            f"{path}: merge-gate must have exactly {len(MERGE_GATE_SETUP) + 1} steps - "
            f"its audited setup then one decision - found {len(steps)}"
        )
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        for control in ("if", "continue-on-error", "working-directory", "shell"):
            if control in step:
                problems.append(
                    f"{path}: merge-gate step {index} declares {control!r}, which can "
                    "detach its result from the job's conclusion"
                )
    if str(job.get("if", "")).strip() != MERGE_GATE_CONDITION:
        problems.append(
            f"{path}: merge-gate condition must be exactly {MERGE_GATE_CONDITION!r}, "
            f"found {str(job.get('if', '')).strip()!r}"
        )
    uses = [
        (index, str(step.get("uses", ""))) for index, step in enumerate(steps) if step.get("uses")
    ]
    if len(uses) != len(MERGE_GATE_SETUP):
        problems.append(
            f"{path}: merge-gate must use exactly {len(MERGE_GATE_SETUP)} actions, found "
            f"{len(uses)}; an extra action runs with the gate's authority"
        )
    decision = [index for index, step in enumerate(steps) if step.get("run")]
    if len(decision) == 1:
        # The decision reads its authority from this mapping. A dropped key is
        # an unreadable input, which the predicate refuses - the binding is
        # pinned here so that refusal is never reached by accident.
        declared = steps[decision[0]].get("env") or {}
        if set(declared) != set(MERGE_GATE_ENV):
            missing = sorted(set(MERGE_GATE_ENV) - set(declared))
            extra = sorted(set(declared) - set(MERGE_GATE_ENV))
            problems.append(
                f"{path}: merge-gate decision env must be exactly "
                f"{sorted(MERGE_GATE_ENV)}; missing={missing} unexpected={extra}"
            )
        for name, expression in MERGE_GATE_ENV.items():
            if name in declared and str(declared[name]).strip() != expression:
                problems.append(
                    f"{path}: merge-gate {name} must read {expression!r}, found "
                    f"{str(declared[name]).strip()!r}"
                )

    for position, expected in enumerate(MERGE_GATE_SETUP):
        matches = [
            (index, value) for index, value in uses if value.startswith(f"{expected['action']}@")
        ]
        if not matches:
            problems.append(
                f"{path}: merge-gate has no {expected['label']} step, {expected['missing']}"
            )
            continue
        if len(matches) > 1:
            problems.append(
                f"{path}: merge-gate uses {expected['action']} {len(matches)} times; a second "
                "one can replace what the first fetched"
            )
        index, value = matches[0]
        pinned = value.split("@", 1)[1].split()[0]
        if not FULL_COMMIT.fullmatch(pinned):
            problems.append(
                f"{path}: merge-gate pins {expected['action']} to {pinned!r}, which is mutable; "
                "a moving ref can change what the gate runs"
            )
        elif pinned != expected["ref"]:
            problems.append(
                f"{path}: merge-gate pins {expected['action']} to {pinned}, not the "
                f"repository's audited {expected['ref']}"
            )
        if index != position:
            problems.append(
                f"{path}: merge-gate runs {expected['action']} at step {index}, expected "
                f"position {position}; the decision must come after its setup"
            )
        declared = steps[index].get("with") or {}
        for key, want in expected["required_with"].items():
            if declared.get(key) != want:
                problems.append(
                    f"{path}: merge-gate {expected['label']} needs {key}={want!r}, "
                    f"found {declared.get(key)!r}"
                )

    decision = [index for index, step in enumerate(steps) if step.get("run")]
    if decision and decision[0] < len(MERGE_GATE_SETUP):
        problems.append(
            f"{path}: merge-gate runs a command at step {decision[0]}, before its "
            "checkout and interpreter are in place"
        )
    return problems


# Every action the gate depends on. `converted_to_draft` and `edited` are here
# because without them a head keeps an authorization earned under conditions
# that no longer hold: a pull request converted back to draft, or repointed at a
# different base branch, would otherwise carry its old success forward.
REQUIRED_PULL_REQUEST_ACTIONS = frozenset(
    {
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
        "converted_to_draft",
        "edited",
    }
)
PULL_REQUEST_ACTIONS = [
    "opened",
    "synchronize",
    "reopened",
    "ready_for_review",
    "converted_to_draft",
    "edited",
]
EXPECTED_TRIGGER_KEYS = frozenset({"pull_request", "merge_group", "schedule", "workflow_dispatch"})
PROTECTED_BRANCHES = ["develop", "main"]
EXPECTED_SCHEDULE = [{"cron": "23 9 * * 2"}]
# Whitespace-normalised, because the expression is written across lines in YAML.
EXPECTED_CONCURRENCY_GROUP = (
    "ci-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}-${{ "
    "github.event_name == 'pull_request' && github.event.action || github.event_name }}"
)


def validate_pull_request_triggers(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Pin the events the gate runs on, and the identity that keeps its runs apart.

    Exact equality rather than "contains what we need". A subset check accepts
    additions, and an addition changes authority: subscribing to `closed`, for
    instance, would run the gate on a merged pull request. The same applies to
    the branch list, the concurrency expression and the cancellation policy -
    each is a decision, so each is pinned rather than sampled.
    """
    if path.name != "ci.yml":
        return []
    problems: list[str] = []
    # `on` is YAML's boolean true once parsed, which is a well-known trap here.
    # The parsed document therefore really does carry a key that is not a
    # string, so it has to be read through a mapping that admits one; declaring
    # str keys and then looking up True is the annotation being wrong, not the
    # lookup.
    # Mapping's key type is invariant, so the cast is what says the declared
    # str-keyed shape is narrower than the document actually is.
    keyed = cast("Mapping[object, Any]", workflow)
    triggers = keyed.get("on") or keyed.get(True) or {}
    pull_request = triggers.get("pull_request") if isinstance(triggers, dict) else None
    if not isinstance(pull_request, dict):
        return [f"{path}: ci.yml has no pull_request trigger"]

    # The exact list, in order, without duplicates. A set comparison accepts a
    # repeated entry, and a repeated entry is a signal that the file was edited
    # by something that did not understand it.
    declared = list(pull_request.get("types") or [])
    if declared != PULL_REQUEST_ACTIONS:
        problems.append(
            f"{path}: pull_request types must be exactly {PULL_REQUEST_ACTIONS}, found {declared}"
        )
    # No other trigger keys. A `push` trigger would run this workflow, and its
    # gate, on events with none of the pull request context the gate reads.
    if set(triggers) != EXPECTED_TRIGGER_KEYS:
        missing = sorted(EXPECTED_TRIGGER_KEYS - set(triggers))
        extra = sorted(set(triggers) - EXPECTED_TRIGGER_KEYS)
        problems.append(
            f"{path}: triggers must be exactly {sorted(EXPECTED_TRIGGER_KEYS)}; "
            f"missing={missing} unexpected={extra}"
        )
    if not exactly_equal({"types": ["checks_requested"]}, triggers.get("merge_group")):
        problems.append(f"{path}: merge_group must request exactly checks_requested")
    # workflow_dispatch takes no inputs. A dispatch contract is a way to vary a
    # run's behaviour from outside the file.
    if triggers.get("workflow_dispatch") is not None:
        problems.append(
            f"{path}: workflow_dispatch must declare no inputs, found "
            f"{triggers.get('workflow_dispatch')!r}"
        )
    if list(pull_request.get("branches") or []) != PROTECTED_BRANCHES:
        problems.append(
            f"{path}: pull_request must target exactly {PROTECTED_BRANCHES}, found "
            f"{pull_request.get('branches')!r}"
        )

    if triggers.get("schedule") != EXPECTED_SCHEDULE:
        problems.append(
            f"{path}: ci.yml must keep its schedule {EXPECTED_SCHEDULE}, found "
            f"{triggers.get('schedule')!r}"
        )
    if "workflow_dispatch" not in triggers:
        problems.append(f"{path}: ci.yml must keep its workflow_dispatch trigger")

    # The event action belongs in the concurrency identity. With one group per
    # pull request, a draft `synchronize` and the `ready_for_review` that
    # follows it cancel each other, and the survivor is whichever GitHub
    # started last - which is how a non-draft pull request came to report every
    # required check as skipped while its last real run tested an earlier
    # commit. Comparing the whole expression also catches a truncated or
    # reordered identity, which a substring check cannot.
    concurrency = workflow.get("concurrency") or {}
    group = " ".join(str(concurrency.get("group", "")).split())
    if group != EXPECTED_CONCURRENCY_GROUP:
        problems.append(
            f"{path}: the concurrency group must be exactly {EXPECTED_CONCURRENCY_GROUP!r}, "
            f"found {group!r}"
        )
    if concurrency.get("cancel-in-progress") is not True:
        problems.append(
            f"{path}: concurrency must set cancel-in-progress: true, found "
            f"{concurrency.get('cancel-in-progress')!r}"
        )
    return problems


def validate_merge_gate(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Prove the merge gate refuses, by running it rather than reading it."""

    if path.name != "ci.yml":
        return []
    problems: list[str] = []
    job = (workflow.get("jobs") or {}).get("merge-gate")
    if not isinstance(job, dict):
        return [f"{path}: ci.yml has no merge-gate job to enforce required checks"]

    # One exhaustive comparison replaces the individual condition, needs and
    # step assertions that used to live here. Those answered only the questions
    # they were given; this refuses anything the schema does not name, which is
    # where every one of the accepted drifts came from.
    problems.extend(validate_merge_gate_schema(path, job))
    problems.extend(validate_merge_gate_steps(path, job))

    declared_steps = job.get("steps") or []
    if not isinstance(declared_steps, list) or not all(
        isinstance(step, dict) for step in declared_steps
    ):
        # Already reported by the schema and the step checks; stop here rather
        # than crashing on a shape neither of them accepted.
        return problems
    scripts = [step.get("run") for step in declared_steps if step.get("run")]
    if len(scripts) != 1:
        return problems + [
            f"{path}: expected exactly one run step in merge-gate, found {len(scripts)}"
        ]
    # The step must invoke the shipped predicate, because that is the thing the
    # cases below actually execute. A step that inlined its own logic would be
    # tested by nothing.
    if scripts[0].strip() != MERGE_GATE_COMMAND:
        return problems + [
            (
                f"{path}: merge-gate must run exactly {MERGE_GATE_COMMAND!r}, so the "
                f"workflow and this validator exercise the same decision; "
                f"found {scripts[0].strip()!r}"
            )
        ]
    if not MERGE_GATE_SCRIPT.is_file():
        return problems + [f"{path}: {MERGE_GATE_SCRIPT} is missing"]
    if problems:
        return problems

    for label, environment, expected in MERGE_GATE_MATRIX:
        full = {
            **os.environ,
            "EVENT_NAME": "pull_request",
            "ACTION": "synchronize",
            "HEAD_SHA": "0" * 40,
            "BASE_CHANGED": "false",
            **{name: value for name, value in environment.items() if value is not None},
        }
        # A case can assert what happens when a variable is absent entirely,
        # which is different from asserting what happens when it is empty.
        for name, value in environment.items():
            if value is None:
                full.pop(name, None)
        try:
            finished = subprocess.run(
                [sys.executable, str(MERGE_GATE_SCRIPT)],
                env=full,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            # A failure to execute is a problem, never a pass. The previous
            # version printed a note and returned the existing list, so a
            # machine that could not run the gate reported success - the same
            # fail-open this gate exists to prevent.
            problems.append(f"{path}: merge-gate script could not be executed: {error}")
            break
        actual = 0 if finished.returncode == 0 else 1
        if actual != expected:
            problems.append(
                f"{path}: merge-gate case '{label}' returned {finished.returncode}, "
                f"expected {'success' if expected == 0 else 'failure'}. "
                f"env={environment} output={finished.stdout.strip()[-160:]}"
            )
    return problems


MERGE_GROUP_CONDITION = (
    "( github.event_name == 'merge_group' && "
    "github.event.action == 'checks_requested' && "
    "github.event.repository.private == false && "
    "github.event.merge_group.base_ref == 'refs/heads/develop' )"
)
VERIFICATION_PLAN_CONDITION = (
    "github.event_name == 'workflow_dispatch' || " + MERGE_GROUP_CONDITION + " || "
    "( github.event_name == 'pull_request' && "
    "github.event.repository.private == false && "
    "github.event.pull_request.draft == false && "
    "( github.event.action != 'edited' || "
    "github.event.changes.base.ref.from != '' || "
    "github.event.changes.base.sha.from != '' ) )"
)
COMPATIBILITY_CONDITION = (
    "always() && ( github.event_name == 'workflow_dispatch' || "
    + MERGE_GROUP_CONDITION
    + " || ( github.event_name == 'pull_request' && "
    "github.event.repository.private == false && "
    "github.event.pull_request.draft == false ) )"
)
PLAN_ENV = {
    "BASE_REF": EVENT_BASE_REF,
    "BASE_SHA": EVENT_BASE_SHA,
    "EVENT_NAME": "${{ github.event_name }}",
    "HEAD_REF": EVENT_HEAD_REF,
    "HEAD_SHA": EVENT_HEAD_SHA,
}
PLAN_COMMAND = (
    "python3 scripts/ci-plan.py \\\n"
    '  --event "$EVENT_NAME" \\\n'
    '  --base-ref "$BASE_REF" \\\n'
    '  --head-ref "$HEAD_REF" \\\n'
    '  --base-sha "$BASE_SHA" \\\n'
    '  --head-sha "$HEAD_SHA" \\\n'
    '  --github-output "$GITHUB_OUTPUT"'
)
FULL_PLAN = "needs.verification-plan.outputs.mode == 'full'"
# The first step of both Ubuntu jobs, so neither can succeed after a plan that
# failed or never ran.
REQUIRED_PLAN_STEP = {
    "name": "Require a valid verification plan",
    "if": "needs.verification-plan.result != 'success'",
    "run": "exit 1",
}
# A condition that still contains the full-plan test can be made false without
# removing it, as `... == 'full' && false` is, so the steps that do the verifying
# are pinned whole: name, condition and command, with nothing beside them.
UBUNTU_VERIFICATION_STEP = {
    "name": "Run repository verification",
    "if": FULL_PLAN,
    "run": "./scripts/verify.sh",
}
# Every command the browser-suites job runs after its plan check, in order.
BROWSER_SUITE_STEPS = (
    {
        "name": "Install browser suite dependencies",
        "if": FULL_PLAN,
        "run": (
            "uv sync --project services/api --locked --python 3.12 --extra dev --extra package\n"
            "npm ci"
        ),
    },
    {"name": "Build the production web client", "if": FULL_PLAN, "run": "npm run build"},
    {
        "name": "Install golden-path browser",
        "if": FULL_PLAN,
        # apt can hang past its own timeouts; the step's limit ends a hang.
        "timeout-minutes": 15,
        "run": "npx playwright install --with-deps chromium",
    },
    {"name": "Run browser golden path", "if": FULL_PLAN, "run": "npm run e2e:run"},
    {
        "name": "Certify synthetic workflow editor browser protocol",
        "if": FULL_PLAN,
        "run": "npm run e2e:workflow-editor:run",
    },
    {
        "name": "Certify surfaces that need a managed media engine",
        "if": FULL_PLAN,
        "run": "npm run e2e:managed-media:run",
    },
)
# The Windows job runs its whole verification in one step. That step is pinned
# like the Ubuntu one, so it cannot be replaced or made conditional.
WINDOWS_VERIFICATION_STEP = {
    "name": "Run repository verification",
    "run": ".\\scripts\\verify.ps1",
}
# The keys the Ubuntu and Windows verification jobs declare, no more and no
# fewer. `defaults`, `container`, `services` and `strategy` each change what a
# step runs without touching the step, so a job is refused for declaring any
# key beyond these, and for leaving one out. The planner declares outputs
# instead of env and needs: its outputs decide how much the other jobs verify.
VERIFICATION_JOB_KEYS = frozenset(
    {"name", "needs", "if", "runs-on", "timeout-minutes", "env", "steps"}
)
PLAN_JOB_KEYS = frozenset({"name", "if", "runs-on", "timeout-minutes", "outputs", "steps"})
VERIFICATION_RUNNERS = {
    "verification-plan": "ubuntu-24.04",
    "compatibility": "ubuntu-24.04",
    "browser-suites": "ubuntu-24.04",
    "windows-compatibility": "windows-2025",
}
# A job's env reaches every step in it, the pinned ones included, and many
# variables change what a step runs: BASH_ENV runs a file before each bash
# step, PYTEST_ADDOPTS can make the suite only collect, and an npm script shell
# can make every `npm run` succeed. A list of dangerous names would always be
# missing one, so the whole mapping is pinned.
VERIFICATION_ENV = {
    "LOCAL_LM_DATA_DIR": "${{ github.workspace }}/temp/local-lm-tests",
    "UV_PROJECT_ENVIRONMENT": "${{ github.workspace }}/.venv",
}
VERIFICATION_CHECKOUT = {
    "uses": f"{MERGE_GATE_SETUP[0]['action']}@{MERGE_GATE_SETUP[0]['ref']}",
    "with": {
        "ref": VERIFICATION_REF,
        "fetch-depth": 0,
        "persist-credentials": False,
    },
}
# The toolchain steps every job that verifies runs. Their inputs are pinned with
# them: setup-node writes a registry setting into the npm configuration later
# steps read, and setup-uv can put a folder at the front of their PATH.
TOOLCHAIN_STEPS = (
    {
        "uses": "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",
        "with": {"node-version": 22, "cache": "npm"},
    },
    {
        "uses": f"{MERGE_GATE_SETUP[1]['action']}@{MERGE_GATE_SETUP[1]['ref']}",
        "with": {"python-version": "3.12"},
    },
    {
        "uses": "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
        "with": {
            "version": "0.11.18",
            "enable-cache": True,
            "cache-dependency-glob": "services/api/uv.lock",
        },
    },
)
FFMPEG_START_SCRIPT = (
    'exec 9> "$RUNNER_TEMP/ffmpeg-install.lock"\n'
    "flock 9\n"
    "(\n"
    "  set +e\n"
    "  {\n"
    "    date -u '+%H:%M:%S refreshing the package index'\n"
    "    sudo apt-get update &&\n"
    "      date -u '+%H:%M:%S installing ffmpeg' &&\n"
    "      sudo apt-get install --yes --no-install-recommends ffmpeg\n"
    "  } 9>&-\n"
    "  status=$?\n"
    "  date -u '+%H:%M:%S finished'\n"
    '  echo "$status" > "$RUNNER_TEMP/ffmpeg-install.status"\n'
    ') < /dev/null > "$RUNNER_TEMP/ffmpeg-install.log" 2>&1 &'
)
SECRET_SCAN_SCRIPT = (
    "set -euo pipefail\n"
    'version="8.30.1"\n'
    'archive_name="gitleaks_${version}_linux_x64.tar.gz"\n'
    'archive="$RUNNER_TEMP/$archive_name"\n'
    "expected_hash=551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb\n"
    "curl --fail --location --proto '=https' --tlsv1.2 \\\n"
    '  --output "$archive" \\\n'
    '  "https://github.com/gitleaks/gitleaks/releases/download/v$version/$archive_name"\n'
    'echo "$expected_hash  $archive" | sha256sum --check --strict\n'
    'tar -xzf "$archive" -C "$RUNNER_TEMP" gitleaks\n'
    "# -v prints RuleID, File and Line for each finding. Without it the\n"
    '# step fails with only "leaks found: 1" and names nothing, so a\n'
    "# refusal on a false positive costs a person a manual bisect. Secret\n"
    "# and Match stay redacted, so nothing candidate-secret reaches a\n"
    "# public log.\n"
    '"$RUNNER_TEMP/gitleaks" git . --redact -v'
)
FFMPEG_FINISH_SCRIPT = (
    'flock "$RUNNER_TEMP/ffmpeg-install.lock" true\n'
    'cat "$RUNNER_TEMP/ffmpeg-install.log"\n'
    'test "$(cat "$RUNNER_TEMP/ffmpeg-install.status" 2> /dev/null)" = 0\n'
    "ffmpeg -version\n"
    "ffprobe -version"
)
WINDOWS_FFMPEG_SCRIPT = (
    "# The community feed returns 503 and 504 often enough to fail runs\n"
    "# on its own. The pinned version is what matters; how many attempts\n"
    "# it took to fetch it does not.\n"
    "#\n"
    "# The budget is deliberate rather than arbitrary. Four attempts with\n"
    "# linear backoff covered ninety seconds, and on 2026-09-10 an outage\n"
    "# outlasted it: every attempt failed inside 105 seconds, the step\n"
    "# never reached the verification script, and a merge-group run was\n"
    "# dequeued without a single test having run. These waits cover about\n"
    "# seven minutes, which is an order of magnitude more outage for a\n"
    "# cost paid only when the feed is actually down.\n"
    "#\n"
    "# Waiting is the right answer here rather than making the install\n"
    "# optional: three tests in test_integrated_hardening.py drive the\n"
    "# real video path and FAIL, not skip, without a working ffmpeg.\n"
    "$waits = @(15, 30, 60, 120, 180)\n"
    "$attempt = 0\n"
    "while ($true) {\n"
    "  $attempt++\n"
    "  choco install ffmpeg --version 8.1.2 --yes --no-progress\n"
    "  if ($LASTEXITCODE -eq 0) { break }\n"
    "  if ($attempt -gt $waits.Length) {"
    ' throw "choco install ffmpeg failed after $attempt attempts." }\n'
    "  $wait = $waits[$attempt - 1]\n"
    '  Write-Host "Attempt $attempt failed; retrying in ${wait}s."\n'
    "  Start-Sleep -Seconds $wait\n"
    "}"
)
# Every step of the three jobs that verify a head, in order and whole. Any step
# can change the steps after it, without touching them: it can append to the
# files GitHub reads environment variables and PATH entries from, turn on the
# workflow commands that do the same, put another tool first on PATH, edit a
# script the verification runs, or check out a different tree. A rule about one
# of those spellings would miss the next, so no step of these jobs is left
# free, and changing one means changing this file. The planner's two steps and
# the merge gate's three are pinned the same way.
VERIFICATION_STEPS: dict[str, tuple[dict[str, Any], ...]] = {
    "compatibility": (
        REQUIRED_PLAN_STEP,
        {
            "name": "Start installing ffmpeg for media pipeline tests",
            "if": FULL_PLAN,
            "run": FFMPEG_START_SCRIPT,
        },
        VERIFICATION_CHECKOUT,
        {
            "name": "Scan reachable source history for secrets",
            "if": FULL_PLAN,
            "shell": "bash",
            "run": SECRET_SCAN_SCRIPT,
        },
        *({**step, "if": FULL_PLAN} for step in TOOLCHAIN_STEPS),
        {
            "name": "Install verification dependencies",
            "if": FULL_PLAN,
            "run": (
                "uv sync --project services/api --locked --python 3.12 \\\n"
                "  --extra dev \\\n"
                "  --extra package\n"
                "npm ci"
            ),
        },
        {
            "name": "Audit locked dependencies",
            "if": "needs.verification-plan.outputs.dependency_audit == 'true'",
            "run": "npm audit --audit-level=high\n.venv/bin/python scripts/audit-dependencies.py",
        },
        {
            "name": "Finish installing ffmpeg for media pipeline tests",
            "if": FULL_PLAN,
            # A background install that hangs in apt ends the wait here.
            "timeout-minutes": 20,
            "run": FFMPEG_FINISH_SCRIPT,
        },
        UBUNTU_VERIFICATION_STEP,
        {
            "name": "Validate documentation-only change",
            "if": "needs.verification-plan.outputs.mode == 'documentation'",
            "env": {"BASE_SHA": EVENT_BASE_SHA, "HEAD_SHA": EVENT_HEAD_SHA},
            "run": (
                "python3 scripts/check-repository-hygiene.py\n"
                "git diff --check $BASE_SHA...$HEAD_SHA --"
            ),
        },
        {
            "name": "Confirm exact develop promotion",
            "if": "needs.verification-plan.outputs.mode == 'promotion'",
            "run": "echo Protected develop is ready for exact-tree promotion to main.",
        },
    ),
    "browser-suites": (
        REQUIRED_PLAN_STEP,
        VERIFICATION_CHECKOUT,
        *({**step, "if": FULL_PLAN} for step in TOOLCHAIN_STEPS),
        *BROWSER_SUITE_STEPS,
    ),
    "windows-compatibility": (
        VERIFICATION_CHECKOUT,
        *TOOLCHAIN_STEPS,
        {
            "name": "Install verification dependencies",
            "run": (
                "uv sync --project services/api --locked --python 3.12"
                " --extra dev --extra package\n"
                "npm.cmd ci"
            ),
        },
        {
            "name": "Install ffmpeg for media pipeline tests",
            "shell": "pwsh",
            "run": WINDOWS_FFMPEG_SCRIPT,
        },
        WINDOWS_VERIFICATION_STEP,
    ),
}


def step_list_problems(
    path: Path, name: str, expected: tuple[dict[str, Any], ...], steps: list[dict[str, Any]]
) -> list[str]:
    """Refuse a step list that differs from the pinned one, naming the first difference."""

    declared = [
        {**step, "run": str(step["run"]).strip()} if "run" in step else step for step in steps
    ]
    if exactly_equal(list(expected), declared):
        return []
    # Not strict: when one list is a prefix of the other, the count below says so.
    for index, (pinned, step) in enumerate(zip(expected, declared, strict=False), start=1):
        if not exactly_equal(pinned, step):
            label = step.get("name") or step.get("uses") or "unnamed"
            return [f"{path}: {name} step {index} ({label!r}) is not the step pinned there"]
    return [
        f"{path}: {name} must run exactly its {len(expected)} pinned steps, found {len(declared)}"
    ]


def job_key_problems(
    path: Path, name: str, job: dict[str, Any], expected: frozenset[str]
) -> list[str]:
    """Refuse a job that declares a key beyond the ones it needs, or leaves one out."""

    if set(job) == expected:
        return []
    missing = sorted(expected - set(job))
    unexpected = sorted(str(key) for key in set(job) - expected)
    return [
        f"{path}: {name} must declare exactly {sorted(expected)}; "
        f"missing={missing} unexpected={unexpected}"
    ]


def validate_verification_bindings(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Keep queue planning and verification bound to one integration commit."""

    if path.name != "ci.yml":
        return []
    problems: list[str] = []
    jobs = workflow.get("jobs") or {}
    for name, condition, needs in (
        ("verification-plan", VERIFICATION_PLAN_CONDITION, None),
        ("compatibility", COMPATIBILITY_CONDITION, "verification-plan"),
        ("browser-suites", COMPATIBILITY_CONDITION, "verification-plan"),
        (
            "windows-compatibility",
            "needs.verification-plan.outputs.windows == 'true'",
            "verification-plan",
        ),
    ):
        job = jobs.get(name)
        if not isinstance(job, dict):
            problems.append(f"{path}: missing {name} verification job")
            continue
        if " ".join(str(job.get("if", "")).split()) != condition:
            problems.append(f"{path}: {name} must keep its event condition")
        if not exactly_equal(needs, job.get("needs")):
            problems.append(f"{path}: {name} must depend on its verification plan")
        problems.extend(
            job_key_problems(
                path,
                name,
                job,
                PLAN_JOB_KEYS if name == "verification-plan" else VERIFICATION_JOB_KEYS,
            )
        )
        runner = VERIFICATION_RUNNERS[name]
        if not exactly_equal(runner, job.get("runs-on")):
            problems.append(f"{path}: {name} must run on {runner!r}, found {job.get('runs-on')!r}")
        if name != "verification-plan" and not exactly_equal(VERIFICATION_ENV, job.get("env")):
            problems.append(
                f"{path}: {name} job env must be exactly {VERIFICATION_ENV}, "
                f"found {job.get('env')!r}"
            )
        steps = job.get("steps")
        if not isinstance(steps, list) or not all(isinstance(step, dict) for step in steps):
            problems.append(f"{path}: {name} steps must be a list of mappings")
            continue
        checkouts = [
            step for step in steps if str(step.get("uses", "")).startswith("actions/checkout@")
        ]
        if not exactly_equal([VERIFICATION_CHECKOUT], checkouts):
            problems.append(f"{path}: {name} must check out the integration SHA exactly once")
        if "continue-on-error" in job or any("continue-on-error" in step for step in steps):
            problems.append(f"{path}: {name} must report verification failures")
        if name in VERIFICATION_STEPS:
            problems.extend(step_list_problems(path, name, VERIFICATION_STEPS[name], steps))
        if name == "verification-plan":
            expected_outputs = {
                output: "${{ steps.plan.outputs." + output + " }}"
                for output in ("mode", "dependency_audit", "windows")
            }
            expected_plan = {"id": "plan", "env": PLAN_ENV, "run": PLAN_COMMAND}
            normalized = [
                {**step, "run": step["run"].strip()} if "run" in step else step for step in steps
            ]
            if not exactly_equal([VERIFICATION_CHECKOUT, expected_plan], normalized):
                problems.append(f"{path}: verification-plan must run the bound planner exactly")
            if not exactly_equal(expected_outputs, job.get("outputs")):
                problems.append(f"{path}: verification-plan must publish all planner outputs")
        elif name == "windows-compatibility":
            commands = [
                {**step, "run": str(step["run"]).strip()} for step in steps if "run" in step
            ]
            verifying = [
                command for command in commands if exactly_equal(WINDOWS_VERIFICATION_STEP, command)
            ]
            if len(verifying) != 1:
                problems.append(
                    f"{path}: windows-compatibility must run the repository verification "
                    "exactly once, as pinned"
                )
        elif name in ("compatibility", "browser-suites"):
            if not steps or not exactly_equal(REQUIRED_PLAN_STEP, steps[0]):
                problems.append(f"{path}: {name} must refuse an unsuccessful plan")
            commands = [
                {**step, "run": str(step["run"]).strip()} for step in steps[1:] if "run" in step
            ]
            if name == "browser-suites":
                if not exactly_equal(list(BROWSER_SUITE_STEPS), commands):
                    problems.append(
                        f"{path}: browser-suites must build the web client and run every "
                        "browser suite exactly as pinned"
                    )
                continue
            verifying = [
                command for command in commands if exactly_equal(UBUNTU_VERIFICATION_STEP, command)
            ]
            if len(verifying) != 1:
                problems.append(
                    f"{path}: compatibility must run the repository verification exactly once, "
                    "as pinned"
                )
            docs = [
                step for step in steps if step.get("name") == "Validate documentation-only change"
            ]
            if len(docs) != 1 or not exactly_equal(
                {"BASE_SHA": EVENT_BASE_SHA, "HEAD_SHA": EVENT_HEAD_SHA},
                docs[0].get("env"),
            ):
                problems.append(f"{path}: documentation verification must use the event SHAs")
    return problems


# The top-level keys ci.yml may declare. A workflow-level `defaults` sets the
# shell and working directory of every run step that does not choose its own,
# and a workflow-level `env` reaches every job, so either could make each
# verification step exit 0 without changing one.
CI_WORKFLOW_KEYS = frozenset({"name", "on", "permissions", "concurrency", "jobs"})
# Every job ci.yml runs, by id and the name it reports. A required check is
# matched by that name, so an added or renamed job could report one, and a
# skipped job that reports it counts as satisfied.
CI_JOBS = {
    "verification-plan": "Select verification plan",
    "compatibility": "Ubuntu compatibility",
    "browser-suites": "Ubuntu browser suites",
    "merge-gate": "Merge gate",
    "windows-compatibility": "Windows compatibility",
    "scheduled-audit": "Scheduled dependency audit",
}


def validate_workflow_keys(path: Path, workflow: dict[str, Any]) -> list[str]:
    """Refuse ci.yml keys and jobs that change or impersonate verification."""

    if path.name != "ci.yml":
        return []
    problems: list[str] = []
    if "defaults" in workflow:
        problems.append(
            f"{path}: ci.yml must not set workflow-level defaults, which change the shell "
            "and working directory of every run step that does not choose its own"
        )
    if "env" in workflow:
        problems.append(
            f"{path}: ci.yml must not set a workflow-level env, which reaches every job"
        )
    # `on` parses as True (see validate_pull_request_triggers), so the keys are
    # read through a mapping that admits a key that is not a string. A list
    # rather than a set, so a quoted "on" beside the parsed one is refused too.
    keyed = cast("Mapping[object, Any]", workflow)
    declared = sorted("on" if key is True else str(key) for key in keyed)
    if declared != sorted(CI_WORKFLOW_KEYS):
        problems.append(
            f"{path}: ci.yml must declare exactly the top-level keys "
            f"{sorted(CI_WORKFLOW_KEYS)}, found {declared}"
        )
    jobs = workflow.get("jobs")
    reported = (
        {str(key): job.get("name") if isinstance(job, dict) else None for key, job in jobs.items()}
        if isinstance(jobs, dict)
        else None
    )
    if not exactly_equal(CI_JOBS, reported):
        problems.append(f"{path}: ci.yml must run exactly the jobs {CI_JOBS}, found {reported}")
    return problems


def required_status_checks() -> tuple[frozenset[str], list[str]]:
    """Return the status checks the repository's rulesets require, and why any is unknown.

    The public configuration script applies every ruleset in `.github/rulesets`,
    and a ruleset requires a check by naming its context in a rule of type
    `required_status_checks`. Finding no ruleset, one that cannot be read or
    parsed, one that requires a check without naming its context, or no
    required check at all is reported rather than read as requiring nothing,
    because a job name compared with nothing is never refused.
    """

    paths = sorted(RULESET_ROOT.glob("*.json"))
    if not paths:
        return frozenset(), [f"{RULESET_ROOT}: no ruleset definitions found; {RULESET_USE}"]
    contexts: set[str] = set()
    problems: list[str] = []
    for path in paths:
        # The decoder raises RecursionError, which is not a ValueError, for a
        # ruleset nested deeper than it can recurse.
        try:
            ruleset = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, RecursionError, ValueError) as error:
            problems.append(f"{path}: ruleset cannot be read: {error}; {RULESET_USE}")
            continue
        rules = ruleset.get("rules") if isinstance(ruleset, dict) else None
        if not isinstance(rules, list):
            problems.append(f"{path}: ruleset must declare its rules as a list; {RULESET_USE}")
            continue
        for rule in rules:
            if not isinstance(rule, dict) or rule.get("type") != "required_status_checks":
                continue
            parameters = rule.get("parameters")
            checks = (
                parameters.get("required_status_checks") if isinstance(parameters, dict) else None
            )
            named = [
                check.get("context") if isinstance(check, dict) else None
                for check in (checks if isinstance(checks, list) else [])
            ]
            valid = [name for name in named if isinstance(name, str) and name.strip()]
            if not named or len(valid) != len(named):
                problems.append(
                    f"{path}: a required_status_checks rule must name each check by its "
                    f"context; {RULESET_USE}"
                )
                continue
            contexts.update(valid)
    if not contexts and not problems:
        problems.append(f"{RULESET_ROOT}: no ruleset requires a status check; {RULESET_USE}")
    return frozenset(contexts), problems


def job_check_names(content: str) -> list[tuple[str, str]]:
    """Return each job's id with the name its check is reported under, once per name.

    A job's check is reported under the job's name, or under the job's id when
    the job has no name or only an empty one. The document is composed as the
    action references are, so a repeated key, a quoted one and a pair merged in
    from an anchor are all read, and a name a loader would drop is still
    returned. Raises yaml.YAMLError for a document that cannot be composed.
    """

    root = yaml.compose(content, Loader=yaml.SafeLoader)
    if not isinstance(root, yaml.MappingNode):
        return []
    reported: list[tuple[str, str]] = []
    for jobs in values_named(root, "jobs"):
        for key, job in mapping_pairs(jobs):
            job_id = str(key.value)
            names = [
                str(name.value)
                for name in values_named(job, "name")
                if isinstance(name, yaml.ScalarNode)
                and name.tag != NULL_TAG
                and str(name.value).strip()
            ]
            reported.extend((job_id, name) for name in names or [job_id])
    return reported


def validate_required_check_names(
    path: Path, content: str, required_checks: Collection[str] | None = None
) -> list[str]:
    """Refuse a job outside ci.yml that reports, or can report, a required check.

    GitHub matches a required check by the name a job reports, whatever
    workflow the job is in, so a job elsewhere that reports one satisfies it
    without any of the verification ci.yml runs. ci.yml is left to CI_JOBS,
    which pins every name it reports. A name holding an expression is refused
    too, since it is evaluated only when the workflow runs and so cannot be
    compared with the required checks. Names are compared without surrounding
    whitespace and without regard to case: GitHub matches a status context
    regardless of case, and may trim a check's name and match it as loosely,
    so a name that differs from a required check only in those ways is
    refused. `required_checks` are read from the rulesets when none are given.
    """

    if path.name == "ci.yml":
        return []
    problems: list[str] = []
    if required_checks is None:
        required_checks, problems = required_status_checks()
    try:
        reported = job_check_names(content)
    except yaml.YAMLError as error:
        return problems + [f"{path}: job names cannot be read: {error}"]
    required = {check.strip().casefold(): check for check in sorted(required_checks)}
    for job, name in reported:
        folded = name.strip().casefold()
        if EXPRESSION_START in name:
            problems.append(
                f"{path}: job {job!r} names its check with an expression, {name!r}, which is "
                "evaluated only when the workflow runs and so cannot be compared with the "
                "required checks; outside ci.yml a job's name must be literal"
            )
        elif folded in required:
            problems.append(
                f"{path}: job {job!r} reports its check as {name!r}, which matches the "
                f"required check {required[folded]!r}; only ci.yml may report a required check"
            )
    return problems


def validate_workflow_document(
    path: Path,
    content: str,
    workflow: dict[str, Any],
    required_checks: Collection[str] | None = None,
) -> list[str]:
    """Every policy this repository applies to one workflow file.

    Extracted from `main` so that a regression can enter where production
    enters. Calling the individual validators from a test proves each rule in
    isolation and proves nothing about whether the rule is still wired in - a
    check that stops being called is indistinguishable from one that passes,
    and only a caller-level entry point can tell the difference.

    `required_checks` are the status checks the repository's rulesets require.
    `main` reads them once for every file it checks; when none are given, they
    are read from the rulesets here.
    """
    errors: list[str] = []
    errors.extend(validate_permissions(path, workflow))
    errors.extend(validate_action_pins(path, content))
    errors.extend(validate_checkout_credentials(path, workflow))
    errors.extend(validate_environment_contexts(path, workflow))
    errors.extend(validate_untrusted_triggers(path, content))
    errors.extend(validate_pull_request_triggers(path, workflow))
    errors.extend(validate_workflow_keys(path, workflow))
    errors.extend(validate_required_check_names(path, content, required_checks))
    errors.extend(validate_verification_bindings(path, workflow))
    if errors:
        return errors
    errors.extend(validate_merge_gate(path, workflow))
    return errors


def main() -> None:
    paths = sorted(WORKFLOW_ROOT.glob("*.yml"))
    paths += sorted(WORKFLOW_ROOT.glob("*.yaml"))
    if not paths:
        raise SystemExit("No GitHub workflow files found")

    # Read once for every file, so a ruleset that cannot be read is refused
    # once, and is refused even when no workflow outside ci.yml needs it.
    required_checks, errors = required_status_checks()
    workflow_actions: set[str] = set()
    for path in paths:
        content = path.read_text(encoding="utf-8")
        try:
            workflow = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            errors.append(f"{path}: invalid YAML: {exc}")
            continue
        if not isinstance(workflow, dict):
            errors.append(f"{path}: workflow root must be a mapping")
            continue
        errors.extend(validate_workflow_document(path, content, workflow, required_checks))
        workflow_actions.update(external_actions(content))
        print(path)

    errors.extend(validate_action_allowlist(workflow_actions))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
