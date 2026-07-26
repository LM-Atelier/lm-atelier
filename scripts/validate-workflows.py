"""Validate GitHub workflow syntax and repository security policy."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

WORKFLOW_ROOT = Path(".github/workflows")
PUBLIC_CONFIGURATION = Path("scripts/configure-public-repository.ps1")
ACTION_USE = re.compile(
    r"^\s*(?:-\s*)?uses:\s*(?P<target>\S+)"
    r"(?:\s+#\s*(?P<version>\S+))?\s*$"
)
FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
VERSION_COMMENT = re.compile(r"v[0-9]+(?:\.[0-9]+){1,2}(?:-[0-9A-Za-z.-]+)?")
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

    allowed_writes = {
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
            errors.append(
                f"{path}: unsupported permissions declaration {declaration!r}"
            )
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
                errors.append(
                    f"{path}: {scope} permission must not grant {access!r} access"
                )
    return errors


def validate_action_pins(path: Path, content: str) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        match = ACTION_USE.match(line)
        if not match:
            continue
        target = match.group("target")
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
            errors.append(
                f"{path}:{line_number}: external action must use a full commit SHA"
            )
        version = match.group("version")
        if version is None or not VERSION_COMMENT.fullmatch(version):
            errors.append(
                f"{path}:{line_number}: pinned action needs an auditable version comment"
            )
    return errors


def external_actions(content: str) -> set[str]:
    """Return every external action reference in one workflow."""

    actions: set[str] = set()
    for line in content.splitlines():
        match = ACTION_USE.match(line)
        if not match:
            continue
        target = match.group("target")
        if not target.startswith(("./", "docker://")):
            actions.add(target)
    return actions


def validate_action_allowlist(workflow_actions: set[str]) -> list[str]:
    """Keep the applied GitHub allowlist equal to the reviewed workflow pins."""

    content = PUBLIC_CONFIGURATION.read_text(encoding="utf-8")
    block = ALLOWED_ACTIONS_BLOCK.search(content)
    if block is None:
        return [f"{PUBLIC_CONFIGURATION}: missing $AllowedActionPatterns declaration"]
    configured = {
        match.group("target") for match in QUOTED_ACTION.finditer(block.group("body"))
    }
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
            if not isinstance(action, str) or not action.startswith(
                "actions/checkout@"
            ):
                continue
            inputs = step.get("with")
            if (
                not isinstance(inputs, dict)
                or inputs.get("persist-credentials") is not False
            ):
                errors.append(
                    f"{path}: checkout in job {job_name} must disable persisted credentials"
                )
    return errors


def validate_untrusted_triggers(path: Path, content: str) -> list[str]:
    errors: list[str] = []
    if PULL_REQUEST_TARGET.search(content):
        errors.append(
            f"{path}: pull_request_target is prohibited for untrusted contribution CI"
        )
    if path.name == "ci.yml" and "run-ci" in content:
        errors.append(
            f"{path}: normal pull-request CI must not require the run-ci label"
        )
    return errors


def main() -> None:
    paths = sorted(WORKFLOW_ROOT.glob("*.yml"))
    paths += sorted(WORKFLOW_ROOT.glob("*.yaml"))
    if not paths:
        raise SystemExit("No GitHub workflow files found")

    errors: list[str] = []
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
        errors.extend(validate_permissions(path, workflow))
        errors.extend(validate_action_pins(path, content))
        errors.extend(validate_checkout_credentials(path, workflow))
        errors.extend(validate_untrusted_triggers(path, content))
        workflow_actions.update(external_actions(content))
        print(path)

    errors.extend(validate_action_allowlist(workflow_actions))
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
