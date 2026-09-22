"""Verify direct activation before committing its launch and completion states."""

from __future__ import annotations

import asyncio
import json

from sqlalchemy.orm import Session

from .comfy_registry_activation import (
    ComfyRegistryActivationError,
    ComfyRegistryActivationState,
    MediaStarter,
    NodeInventoryReader,
    _archive_path,
    _deactivate,
    _install,
    _require_stopped,
    _restore_after_cancellation,
    _review_count,
    _state,
)
from .comfy_registry_archives import (
    ComfyRegistryRuntimeFile,
    capture_staged_comfy_registry_runtime_files,
    comfy_registry_runtime_files_json,
    parse_comfy_registry_runtime_files,
    snapshot_staged_comfy_registry_files,
)
from .comfy_registry_installs import ComfyRegistryInstallError
from .comfy_registry_launch_verification import _FIELDS
from .comfy_registry_target_verification import (
    ComfyRegistryVerificationTarget,
    _verification_worker,
)
from .domain import utcnow
from .models import ComfyRegistryInstall
from .source_omission_proof import (
    OmissionProofError,
    evidence_digest,
    pending_omission_requirement,
    prove_omission,
)


async def activate_verified_comfy_registry_install(
    session: Session,
    *,
    install_id: str,
    target: ComfyRegistryVerificationTarget,
    media_worker_stopped: bool,
    start_media: MediaStarter,
    read_node_inventory: NodeInventoryReader | None,
) -> ComfyRegistryActivationState:
    """Keep file work outside the writer and recheck authority after the worker starts."""
    _require_stopped(media_worker_stopped)
    if not _install(session, install_id).trusted:
        raise ComfyRegistryActivationError(
            "registry_install_untrusted",
            "The Registry package must be explicitly trusted before activation",
        )
    try:
        verified = await target.verify((install_id,))
        original = next(
            json.loads(item) for item in verified.snapshots if json.loads(item)["id"] == install_id
        )
        omission = pending_omission_requirement(install_id, original["review_json"])

        def snapshot() -> tuple[ComfyRegistryRuntimeFile, ...]:
            return snapshot_staged_comfy_registry_files(
                _archive_path(target.custom_node_root, original["installed_path"])
            )

        before_start = await _verification_worker(snapshot)
        target.require_configuration()
        session.expire_all()
        install = _install(session, install_id)
        if not install.trusted:
            raise ComfyRegistryInstallError("Registry trust changed before activation")
        verified.require_install(
            session,
            install,
            custom_node_root=target.custom_node_root,
            environment_root=target.environment_root,
        )
        install.active = True
        session.commit()
    except (ValueError, OSError) as exc:
        session.rollback()
        raise ComfyRegistryActivationError(
            "registry_install_verification_failed",
            "Registry package files or dependencies failed verification",
        ) from exc

    failure_code = "activation_start_failed"
    try:
        target.require_configuration()
        await start_media()
        omission_proof = None
        if omission is not None:
            failure_code = "omission_unverifiable"
            if read_node_inventory is None:
                raise ComfyRegistryActivationError(
                    failure_code, "The runtime's node inventory could not be verified"
                )
            observed = await read_node_inventory()
            omission_proof = prove_omission(omission, observed_node_types=observed)
        failure_code = "activation_runtime_files_failed"
        expected = []
        for item in verified.snapshots:
            values = json.loads(item)
            if values["id"] == install_id:
                values["active"] = True
            expected.append(
                json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
        expected_snapshots = tuple(expected)
        runtime_json = None

        def capture(rows: list[ComfyRegistryInstall]) -> None:
            nonlocal runtime_json
            snapshots = tuple(
                json.dumps(
                    {name: getattr(row, name) for name in _FIELDS},
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for row in rows
            )
            if snapshots != expected_snapshots:
                raise ComfyRegistryInstallError("Registry activation inputs changed during startup")
            row = next(row for row in rows if row.id == install_id)
            review = row.review_json
            runtime_files = capture_staged_comfy_registry_runtime_files(
                _archive_path(target.custom_node_root, row.installed_path),
                before_start=before_start,
                expected_manifest_sha256=row.manifest_sha256,
                expected_file_count=_review_count(review, "file_count"),
                expected_expanded_bytes=_review_count(review, "expanded_bytes"),
                runtime_files=parse_comfy_registry_runtime_files(review.get("runtime_files")),
            )
            runtime_json = comfy_registry_runtime_files_json(runtime_files)
            row.review_json = {**review, "runtime_files": runtime_json}

        completed = await target.verify((install_id,), prepare_installs=capture)
        failure_code = "registry_install_verification_failed"
        if completed.contract != verified.contract:
            raise ComfyRegistryInstallError("Registry launch selection changed during startup")
        session.expire_all()
        target.require_configuration()
        completed.require_current(session)
        activated = _install(session, install_id)
        activated.review_json = {
            **activated.review_json,
            "activated_at": utcnow().isoformat(),
            "activation_failure_code": None,
            "runtime_files": runtime_json,
            "source_omission_proof": omission_proof,
            "source_omission_digest": evidence_digest(omission_proof)
            if omission_proof is not None
            else None,
        }
        session.commit()
        session.refresh(activated)
        return _state(activated)
    except (Exception, asyncio.CancelledError) as exc:
        if isinstance(exc, asyncio.CancelledError):
            failure_code = "activation_cancelled"
        elif isinstance(exc, OmissionProofError):
            failure_code = exc.code
        elif isinstance(exc, ComfyRegistryInstallError):
            failure_code = "registry_install_verification_failed"
        _deactivate(session, install_id, failure_code=failure_code)
        await _restore_after_cancellation(start_media)
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise ComfyRegistryActivationError(
            failure_code, "Registry activation failed; the prior media runtime was restored"
        ) from exc
