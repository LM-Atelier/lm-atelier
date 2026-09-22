"""Revalidate retained source identities using a fresh database session."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sqlalchemy import select
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .comfy_registry_wheel_artifacts import comfy_registry_wheel_target_sha256
from .comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryWheelInputError,
    ComfyRegistryWheelInputManifest,
    reviewed_wheel_input,
    validate_comfy_registry_wheel_input_manifest,
)
from .models import Artifact, ComfyRegistrySourceArtifactReview

_AUTHORITY_SEAL = object()


@dataclass(frozen=True)
class ComfyRegistryReviewedInputAuthority:
    manifest: ComfyRegistryWheelInputManifest
    rows: tuple[str, ...]
    seal: object

    def require_current(self, session: Session) -> None:
        """Compare database facts after the caller has reserved its commit's writer."""
        if self.seal is not _AUTHORITY_SEAL or _authority_rows(session, self.manifest) != self.rows:
            raise ComfyRegistryWheelInputError("reviewed_wheel_input_changed")


def _authority_rows(session: Session, manifest: ComfyRegistryWheelInputManifest) -> tuple[str, ...]:
    rows: list[str] = []
    review = ComfyRegistrySourceArtifactReview
    for expected in manifest.reviewed_local:
        # Select columns so a caller's identity map cannot conceal a revocation
        # or replacement committed while the files were being verified.
        row = session.execute(
            select(
                review.id,
                review.source_declaration,
                review.source_declaration_sha256,
                review.repository,
                review.source_commit,
                review.artifact_id,
                review.artifact_sha256,
                review.artifact_size_bytes,
                review.wheel_filename,
                review.wheel_distribution,
                review.wheel_version,
                review.evidence_json,
                review.reviewer_kind,
                review.review_sha256,
                review.reviewed_at,
                Artifact.id,
                Artifact.sha256,
                Artifact.size_bytes,
                Artifact.relative_path,
                Artifact.original_name,
            )
            .join(Artifact, Artifact.id == review.artifact_id)
            .where(review.source_declaration_sha256 == expected.declaration_sha256)
        ).one_or_none()
        if row is None:
            raise ComfyRegistryWheelInputError("reviewed_wheel_input_changed")
        values = list(row)
        values[14] = row.reviewed_at.isoformat() if row.reviewed_at is not None else None
        rows.append(json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return tuple(rows)


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
        self.verified_authority(manifest)

    def verified_authority(
        self, manifest: ComfyRegistryWheelInputManifest
    ) -> ComfyRegistryReviewedInputAuthority:
        """Verify bytes and capture their authority from one consistent read snapshot."""
        inputs = validate_comfy_registry_wheel_input_manifest(manifest)
        target = comfy_registry_wheel_target_sha256(self.marker_environment, self.supported_tags)
        if target != inputs.remote.target_sha256:
            raise ComfyRegistryWheelInputError("wheel_input_target_mismatch")
        with self.session_factory() as session:
            connection = session.connection()
            driver = connection.connection.driver_connection
            if connection.dialect.name == "sqlite" and not bool(
                getattr(driver, "in_transaction", False)
            ):
                # sqlite3's legacy transaction mode does not begin on SELECT.
                # Keep the review and its artifact metadata in the same read
                # snapshot; WAL still permits another connection to revoke it.
                connection.exec_driver_sql("BEGIN")
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
            return ComfyRegistryReviewedInputAuthority(
                inputs, _authority_rows(session, inputs), _AUTHORITY_SEAL
            )
