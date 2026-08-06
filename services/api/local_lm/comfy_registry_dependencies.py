from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from .package_sources import bare_source_url, classify_source_url, source_refusal

# A comment starts at a `#` that follows whitespace, matching how pip reads a
# requirements file. A bare `#` inside a requirement is not a comment.
_INLINE_COMMENT = re.compile(r"\s#")

MAX_REGISTRY_PIP_DEPENDENCIES = 256
MAX_REGISTRY_PIP_DEPENDENCY_CHARACTERS = 1_000
MAX_REGISTRY_DEPENDENCY_PLAN_BYTES = 256 * 1024
MAX_REGISTRY_DEPENDENCY_EXTRAS = 32


class ComfyRegistryDependencyError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryDependency:
    name: str
    requirement: str
    marker: str | None
    extras: tuple[str, ...]
    pinned_version: str | None
    version_resolution_required: bool


@dataclass(frozen=True)
class ComfyRegistryDependencyPlan:
    dependencies: tuple[ComfyRegistryDependency, ...]
    declaration_sha256: str
    version_resolution_required: bool
    artifact_resolution_required: bool


def plan_comfy_registry_dependencies(
    declarations: Sequence[str],
) -> ComfyRegistryDependencyPlan:
    """Parse Registry pip requirements into an inert deterministic plan."""
    if isinstance(declarations, str | bytes) or not isinstance(declarations, Sequence):
        raise ComfyRegistryDependencyError(
            "invalid_dependency_list", "Registry dependencies must be an array"
        )
    if len(declarations) > MAX_REGISTRY_PIP_DEPENDENCIES:
        raise ComfyRegistryDependencyError(
            "too_many_dependencies", "Registry version declares too many pip dependencies"
        )
    parsed: list[ComfyRegistryDependency] = []
    targets: set[tuple[str, str]] = set()
    for declaration in declarations:
        requirement = _requirement_line(declaration)
        if requirement is None:
            continue
        dependency = _dependency(requirement)
        target = (dependency.name, dependency.marker or "")
        if target in targets:
            raise ComfyRegistryDependencyError(
                "ambiguous_dependency",
                f"Registry version declares {dependency.name} more than once for one environment",
            )
        targets.add(target)
        parsed.append(dependency)
    parsed.sort(key=lambda item: (item.name, item.marker or "", item.requirement))
    payload = {
        "version": 1,
        "dependencies": [
            {
                "name": item.name,
                "requirement": item.requirement,
                "marker": item.marker,
                "extras": list(item.extras),
                "pinned_version": item.pinned_version,
                "version_resolution_required": item.version_resolution_required,
            }
            for item in parsed
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_REGISTRY_DEPENDENCY_PLAN_BYTES:
        raise ComfyRegistryDependencyError(
            "dependency_plan_too_large", "Registry dependency plan exceeds the size limit"
        )
    return ComfyRegistryDependencyPlan(
        tuple(parsed),
        hashlib.sha256(encoded).hexdigest(),
        any(item.version_resolution_required for item in parsed),
        bool(parsed),
    )


def _requirement_line(value: object) -> str | None:
    """Separate what a requirements file carries from what it declares.

    Publishers fill this field by dumping a requirements file, so it arrives
    with that file's ordinary furniture: blank lines, comments, and inline
    comments after a requirement. None of those name a dependency, and
    refusing a package because it shipped a comment refuses almost every real
    package.

    Option lines are different and are not silently dropped. `-r`, `-e`, and
    `--index-url` change what gets installed or where it comes from, so
    ignoring one would install something other than what was declared. Those
    refuse with their own code rather than being mistaken for a malformed
    requirement.
    """
    if not isinstance(value, str):
        raise ComfyRegistryDependencyError(
            "invalid_dependency", "Registry version has an invalid pip dependency"
        )
    line = value.strip()
    if line.startswith("#"):
        return None
    line = _INLINE_COMMENT.split(line, maxsplit=1)[0].strip()
    if not line:
        return None
    if line.startswith("-"):
        raise ComfyRegistryDependencyError(
            "dependency_option_unsupported",
            "Registry pip dependencies cannot set installer options or extra indexes",
        )
    return line


def _dependency(value: object) -> ComfyRegistryDependency:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_REGISTRY_PIP_DEPENDENCY_CHARACTERS
        or any(character < " " or character == "\x7f" for character in value)
    ):
        raise ComfyRegistryDependencyError(
            "invalid_dependency", "Registry version has an invalid pip dependency"
        )
    try:
        parsed = Requirement(value.strip())
    except InvalidRequirement as exc:
        # pip accepts a bare `git+https://...` line; PEP 508 requires the
        # `name @ url` form, so packaging refuses it and the dependency died
        # here as "invalid" - which says nothing about what is wrong or who
        # can fix it. The same dependency written the named way already gets
        # a refusal that explains itself, and both are the same situation: a
        # source URL with no immutable identity. Send them to one answer.
        source = bare_source_url(value.strip())
        if source is not None:
            code, message = source_refusal(classify_source_url(source))
            raise ComfyRegistryDependencyError(code, message) from exc
        raise ComfyRegistryDependencyError(
            "invalid_dependency", "Registry version has an invalid pip dependency"
        ) from exc
    if parsed.url is not None:
        # Still refused, every one of them. What changed is that the refusal
        # says which situation the package is in: an exact commit that a later
        # slice could resolve, a branch that no slice can make exact, or a URL
        # that was never allowed.
        code, message = source_refusal(classify_source_url(parsed.url))
        raise ComfyRegistryDependencyError(code, message)
    name = canonicalize_name(parsed.name)
    extras = tuple(sorted(canonicalize_name(value) for value in parsed.extras))
    if len(extras) > MAX_REGISTRY_DEPENDENCY_EXTRAS:
        raise ComfyRegistryDependencyError(
            "too_many_dependency_extras", f"Registry dependency {name} declares too many extras"
        )
    marker = str(parsed.marker) if parsed.marker is not None else None
    specifier = str(parsed.specifier)
    pinned_version = _pinned_version(parsed)
    canonical: str = name
    if extras:
        canonical += f"[{','.join(extras)}]"
    canonical += specifier
    if marker:
        canonical += f"; {marker}"
    return ComfyRegistryDependency(
        name,
        canonical,
        marker,
        extras,
        pinned_version,
        pinned_version is None,
    )


def _pinned_version(requirement: Requirement) -> str | None:
    specifiers = tuple(requirement.specifier)
    if len(specifiers) != 1:
        return None
    specifier = specifiers[0]
    if specifier.operator not in {"==", "==="} or "*" in specifier.version:
        return None
    return specifier.version
