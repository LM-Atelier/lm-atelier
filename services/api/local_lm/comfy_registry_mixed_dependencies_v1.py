from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from .comfy_registry_dependencies import (
    MAX_REGISTRY_DEPENDENCY_EXTRAS,
    MAX_REGISTRY_DEPENDENCY_PLAN_BYTES,
    MAX_REGISTRY_PIP_DEPENDENCIES,
    MAX_REGISTRY_PIP_DEPENDENCY_CHARACTERS,
    ComfyRegistryDependencyError,
    ComfyRegistryDependencyPlan,
    _requirement_line,
    plan_comfy_registry_dependencies,
)
from .package_sources import classify_source_url, source_refusal


@dataclass(frozen=True)
class ComfyRegistrySourceDeclaration:
    name: str
    declaration: str
    declaration_sha256: str
    repository: str
    commit: str
    marker: str | None
    extras: tuple[str, ...]


@dataclass(frozen=True)
class ComfyRegistryMixedDependencyPlan:
    remote: ComfyRegistryDependencyPlan
    sources: tuple[ComfyRegistrySourceDeclaration, ...]
    declaration_sha256: str


def _source(requirement: Requirement) -> ComfyRegistrySourceDeclaration:
    assert requirement.url is not None
    try:
        source = classify_source_url(requirement.url)
    except ValueError as exc:
        raise ComfyRegistryDependencyError(
            "invalid_dependency", "Registry version has an invalid pip dependency"
        ) from exc
    if source.repository is None or source.commit is None or source.reference is not None:
        code, message = source_refusal(source)
        raise ComfyRegistryDependencyError(code, message)
    name = canonicalize_name(requirement.name)
    extras = tuple(sorted(canonicalize_name(extra) for extra in requirement.extras))
    if len(extras) > MAX_REGISTRY_DEPENDENCY_EXTRAS:
        raise ComfyRegistryDependencyError(
            "too_many_dependency_extras", f"Registry dependency {name} declares too many extras"
        )
    # Source reviews bind packaging's spelling, including the declared URL.
    declaration = str(requirement)
    if len(declaration) > MAX_REGISTRY_PIP_DEPENDENCY_CHARACTERS:
        raise ComfyRegistryDependencyError(
            "invalid_dependency", "Registry version has an invalid pip dependency"
        )
    return ComfyRegistrySourceDeclaration(
        name,
        declaration,
        hashlib.sha256(declaration.encode("utf-8")).hexdigest(),
        source.repository,
        source.commit,
        str(requirement.marker) if requirement.marker is not None else None,
        extras,
    )


def plan_comfy_registry_mixed_dependencies(
    declarations: Sequence[str],
) -> ComfyRegistryMixedDependencyPlan:
    """Bind remote requirements and exact source declarations without authorizing installs."""
    if isinstance(declarations, str | bytes) or not isinstance(declarations, Sequence):
        raise ComfyRegistryDependencyError(
            "invalid_dependency_list", "Registry dependencies must be an array"
        )
    if len(declarations) > MAX_REGISTRY_PIP_DEPENDENCIES:
        raise ComfyRegistryDependencyError(
            "too_many_dependencies", "Registry version declares too many pip dependencies"
        )
    remote_lines: list[str] = []
    sources: list[ComfyRegistrySourceDeclaration] = []
    for value in declarations:
        line = _requirement_line(value)
        if line is None:
            continue
        if len(line) > MAX_REGISTRY_PIP_DEPENDENCY_CHARACTERS or any(
            ord(character) < 32 or ord(character) == 127 for character in line
        ):
            raise ComfyRegistryDependencyError(
                "invalid_dependency", "Registry version has an invalid pip dependency"
            )
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            # Preserve existing diagnostics for bare URLs and malformed requirements.
            plan_comfy_registry_dependencies([line])
            raise AssertionError("An invalid requirement unexpectedly parsed") from None
        if requirement.url is None:
            remote_lines.append(line)
        else:
            sources.append(_source(requirement))
    remote = plan_comfy_registry_dependencies(remote_lines)
    targets = {(item.name, item.marker or "") for item in remote.dependencies}
    for item in sources:
        target = (item.name, item.marker or "")
        if target in targets:
            raise ComfyRegistryDependencyError(
                "ambiguous_dependency",
                f"Registry version declares {item.name} more than once for one environment",
            )
        targets.add(target)
    sources.sort(key=lambda item: (item.name, item.marker or "", item.declaration))
    if not sources:
        return ComfyRegistryMixedDependencyPlan(remote, (), remote.declaration_sha256)
    payload = {
        "version": 1,
        "remote_declaration_sha256": remote.declaration_sha256,
        "sources": [asdict(item) for item in sources],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    if len(encoded) > MAX_REGISTRY_DEPENDENCY_PLAN_BYTES:
        raise ComfyRegistryDependencyError(
            "dependency_plan_too_large", "Registry dependency plan exceeds the size limit"
        )
    return ComfyRegistryMixedDependencyPlan(
        remote, tuple(sources), hashlib.sha256(encoded).hexdigest()
    )


def validate_comfy_registry_mixed_dependency_plan(
    plan: ComfyRegistryMixedDependencyPlan,
) -> ComfyRegistryMixedDependencyPlan:
    """Rebuild the declaration identity before consuming a retained plan."""
    try:
        rebuilt = plan_comfy_registry_mixed_dependencies(
            [item.requirement for item in plan.remote.dependencies]
            + [item.declaration for item in plan.sources]
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ComfyRegistryDependencyError(
            "invalid_mixed_dependency_plan", "Registry dependency plan identity is invalid"
        ) from exc
    if plan != rebuilt:
        raise ComfyRegistryDependencyError(
            "invalid_mixed_dependency_plan", "Registry dependency plan identity is invalid"
        )
    return rebuilt
