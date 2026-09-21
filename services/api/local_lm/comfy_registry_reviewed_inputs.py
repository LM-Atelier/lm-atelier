"""Revalidate retained source identities using a fresh database session."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .comfy_registry_wheel_artifacts import comfy_registry_wheel_target_sha256
from .comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryWheelInputError,
    ComfyRegistryWheelInputManifest,
    reviewed_wheel_input,
    validate_comfy_registry_wheel_input_manifest,
)


@dataclass(frozen=True)
class ComfyRegistryReviewedInputContext:
    session_factory: Callable[[], Session]
    store: ArtifactStore
    marker_environment: Mapping[str, str]
    supported_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "marker_environment", MappingProxyType(dict(self.marker_environment))
        )
        object.__setattr__(self, "supported_tags", tuple(self.supported_tags))

    def validate(self, manifest: ComfyRegistryWheelInputManifest) -> None:
        """Require the exact current review, retained bytes and target for every local input."""
        inputs = validate_comfy_registry_wheel_input_manifest(manifest)
        target = comfy_registry_wheel_target_sha256(self.marker_environment, self.supported_tags)
        if target != inputs.remote.target_sha256:
            raise ComfyRegistryWheelInputError("wheel_input_target_mismatch")
        with self.session_factory() as session:
            for expected in inputs.reviewed_local:
                current = reviewed_wheel_input(
                    session,
                    self.store,
                    declaration=expected.declaration,
                    marker_environment=self.marker_environment,
                    supported_tags=self.supported_tags,
                )
                if current != expected:
                    raise ComfyRegistryWheelInputError("reviewed_wheel_input_changed")
