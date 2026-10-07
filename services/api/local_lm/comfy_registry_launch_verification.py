"""Verify package files before reserving the transaction that grants their trust."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import false, or_, select, update
from sqlalchemy.orm import Session, object_session
from sqlalchemy.sql.elements import ColumnElement

from .comfy_registry import MAX_REGISTRY_PACKAGES
from .comfy_registry_installs import (
    ComfyRegistryInstallError,
    ComfyRegistryLaunchContract,
    _verified_comfy_registry_launch_contract,
)
from .comfy_registry_reviewed_inputs import (
    ComfyRegistryReviewedInputAuthority,
    ComfyRegistryReviewedInputContext,
)
from .models import ComfyRegistryInstall

_SEAL = object()
_FIELDS = (
    "id",
    "package_id",
    "package_version",
    "registry_record_id",
    "repository_url",
    "download_url",
    "archive_sha256",
    "manifest_sha256",
    "installed_path",
    "node_types_json",
    "pip_dependencies_json",
    "review_json",
    "wheel_closure_sha256",
    "wheel_environment_sha256",
    "wheel_environment_path",
    "trusted",
    "active",
)


def _snapshots(
    session: Session, install_ids: tuple[str, ...], include_active: bool
) -> tuple[str, ...]:
    model = ComfyRegistryInstall
    selected: ColumnElement[bool] = model.id.in_(install_ids)
    if include_active:
        selected = or_(selected, model.trusted.is_(True) & model.active.is_(True))
    rows = session.execute(
        select(*(getattr(model, name) for name in _FIELDS))
        .where(selected)
        .order_by(model.id)
        .limit(MAX_REGISTRY_PACKAGES + 1)
    ).all()
    if len(rows) > MAX_REGISTRY_PACKAGES or not set(install_ids).issubset(row.id for row in rows):
        raise ComfyRegistryInstallError("Registry verification package set changed")
    return tuple(
        json.dumps(dict(row._mapping), sort_keys=True, separators=(",", ":"), allow_nan=False)
        for row in rows
    )


@dataclass(frozen=True)
class VerifiedComfyRegistryLaunch:
    install_ids: tuple[str, ...]
    include_active: bool
    custom_node_root: Path
    environment_root: Path
    snapshots: tuple[str, ...]
    contract: ComfyRegistryLaunchContract
    authorities: tuple[ComfyRegistryReviewedInputAuthority, ...]
    seal: object

    def require_current(self, session: Session) -> None:
        """Reserve the writer and compare current columns without reading any files."""
        if self.seal is not _SEAL:
            raise ComfyRegistryInstallError("Registry launch verification is invalid")
        session.flush()
        session.execute(
            update(ComfyRegistryInstall)
            .where(false())
            .values(active=ComfyRegistryInstall.active)
            .execution_options(synchronize_session=False)
        )
        if _snapshots(session, self.install_ids, self.include_active) != self.snapshots:
            raise ComfyRegistryInstallError("Registry packages changed after file verification")
        for authority in self.authorities:
            try:
                authority.require_current(session)
            except ValueError as exc:
                raise ComfyRegistryInstallError(
                    "Registry source dependency review changed before commit",
                    code="source_review_verification_failed",
                ) from exc

    def require_install(
        self,
        session: Session,
        install: ComfyRegistryInstall,
        *,
        custom_node_root: Path,
        environment_root: Path,
    ) -> None:
        """Bind an individual trust grant to the verified broad launch selection."""
        if (
            object_session(install) is not session
            or install.id not in self.install_ids
            or not self.include_active
            or custom_node_root != self.custom_node_root
            or environment_root != self.environment_root
        ):
            raise ComfyRegistryInstallError("Registry trust verification scope changed")
        current = json.dumps(
            {name: getattr(install, name) for name in _FIELDS},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if current not in self.snapshots:
            raise ComfyRegistryInstallError("Registry install changed after file verification")
        self.require_current(session)


def verify_comfy_registry_launch(
    session_factory: Callable[[], Session],
    install_ids: Sequence[str],
    *,
    include_active: bool,
    custom_node_root: Path,
    environment_root: Path,
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None,
) -> VerifiedComfyRegistryLaunch:
    """Read detached install facts and verify their files without holding a writer."""
    return _verify_comfy_registry_launch(
        session_factory,
        install_ids,
        include_active=include_active,
        custom_node_root=custom_node_root,
        environment_root=environment_root,
        reviewed_inputs=reviewed_inputs,
    )


def _verify_comfy_registry_launch(
    session_factory: Callable[[], Session],
    install_ids: Sequence[str],
    *,
    include_active: bool,
    custom_node_root: Path,
    environment_root: Path,
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None,
    prepare_installs: Callable[[list[ComfyRegistryInstall]], None] | None = None,
) -> VerifiedComfyRegistryLaunch:
    """Verify a staged transition while retaining the original database comparison."""
    if (
        isinstance(install_ids, str | bytes)
        or not isinstance(install_ids, Sequence)
        or len(install_ids) > MAX_REGISTRY_PACKAGES
        or any(not isinstance(item, str) or not item or len(item) > 64 for item in install_ids)
        or len(set(install_ids)) != len(install_ids)
        or type(include_active) is not bool
    ):
        raise ComfyRegistryInstallError("Registry verification package set is invalid")
    identifiers = tuple(sorted(install_ids))
    with session_factory() as session:
        snapshots = _snapshots(session, identifiers, include_active)
    installs = [ComfyRegistryInstall(**json.loads(item)) for item in snapshots]
    if prepare_installs is not None:
        prepare_installs(installs)
    authorities: list[ComfyRegistryReviewedInputAuthority] = []
    contract = _verified_comfy_registry_launch_contract(
        installs,
        custom_node_root=custom_node_root,
        environment_root=environment_root,
        reviewed_inputs=reviewed_inputs,
        review_authorities=authorities,
    )
    return VerifiedComfyRegistryLaunch(
        identifiers,
        include_active,
        custom_node_root,
        environment_root,
        snapshots,
        contract,
        tuple(authorities),
        _SEAL,
    )
