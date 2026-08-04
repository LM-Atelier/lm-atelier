from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

MAX_COMFY_RUNTIME_DISTRIBUTIONS = 4_096
MAX_COMFY_RUNTIME_DISTRIBUTION_NAME_CHARACTERS = 200
MAX_COMFY_RUNTIME_DISTRIBUTION_VERSION_CHARACTERS = 200


class ComfyRegistryRuntimeError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryRuntimeDistribution:
    name: str
    version: str


def canonical_comfy_registry_runtime_distributions(
    value: Mapping[str, str] | Sequence[ComfyRegistryRuntimeDistribution],
) -> tuple[ComfyRegistryRuntimeDistribution, ...]:
    """Validate and canonicalize the managed interpreter's package baseline."""
    if isinstance(value, Mapping):
        items: Sequence[object] = tuple(
            ComfyRegistryRuntimeDistribution(name, version) for name, version in value.items()
        )
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        items = value
    else:
        raise ComfyRegistryRuntimeError(
            "invalid_runtime_distributions",
            "Managed runtime distributions are invalid.",
        )
    if len(items) > MAX_COMFY_RUNTIME_DISTRIBUTIONS:
        raise ComfyRegistryRuntimeError(
            "too_many_runtime_distributions",
            "Managed runtime declares too many distributions.",
        )
    resolved: list[ComfyRegistryRuntimeDistribution] = []
    names: set[str] = set()
    for item in items:
        if not isinstance(item, ComfyRegistryRuntimeDistribution):
            raise ComfyRegistryRuntimeError(
                "invalid_runtime_distributions",
                "Managed runtime distributions are invalid.",
            )
        if (
            not isinstance(item.name, str)
            or not item.name
            or len(item.name) > MAX_COMFY_RUNTIME_DISTRIBUTION_NAME_CHARACTERS
            or not isinstance(item.version, str)
            or not item.version
            or len(item.version) > MAX_COMFY_RUNTIME_DISTRIBUTION_VERSION_CHARACTERS
            or _has_control(item.name)
            or _has_control(item.version)
        ):
            raise ComfyRegistryRuntimeError(
                "invalid_runtime_distributions",
                "Managed runtime distributions are invalid.",
            )
        name = canonicalize_name(item.name)
        try:
            version = str(Version(item.version))
        except InvalidVersion as exc:
            raise ComfyRegistryRuntimeError(
                "invalid_runtime_distributions",
                "Managed runtime distribution version is invalid.",
            ) from exc
        if name in names:
            raise ComfyRegistryRuntimeError(
                "duplicate_runtime_distribution",
                f"Managed runtime declares {name} more than once.",
            )
        names.add(name)
        resolved.append(ComfyRegistryRuntimeDistribution(name, version))
    return tuple(sorted(resolved, key=lambda item: item.name))


def comfy_registry_runtime_distribution_map(
    value: Mapping[str, str] | Sequence[ComfyRegistryRuntimeDistribution],
) -> dict[str, str]:
    return {
        item.name: item.version for item in canonical_comfy_registry_runtime_distributions(value)
    }


def comfy_registry_runtime_distribution_payload(
    value: Mapping[str, str] | Sequence[ComfyRegistryRuntimeDistribution],
) -> list[dict[str, str]]:
    return [
        {"name": item.name, "version": item.version}
        for item in canonical_comfy_registry_runtime_distributions(value)
    ]


def _has_control(value: str) -> bool:
    return any(character < " " or character == "\x7f" for character in value)
