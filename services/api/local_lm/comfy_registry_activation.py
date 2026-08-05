from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from .comfy_registry_archives import (
    ComfyRegistryArchiveError,
    capture_staged_comfy_registry_runtime_files,
    comfy_registry_runtime_files_json,
    parse_comfy_registry_runtime_files,
    snapshot_staged_comfy_registry_files,
)
from .comfy_registry_installs import trusted_comfy_registry_launch_contract
from .domain import utcnow
from .models import ComfyRegistryInstall
from .source_omission_proof import (
    OmissionProofError,
    OmissionRequirement,
    evidence_digest,
    prove_omission,
)

MediaStarter = Callable[[], Awaitable[object]]
#: Reads the running worker's loaded node types. Injected rather than imported
#: so the proof can be tested without a worker, and so this module keeps
#: knowing nothing about how the runtime is reached.
NodeInventoryReader = Callable[[], Awaitable[frozenset[str]]]


class ComfyRegistryActivationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ComfyRegistryActivationState:
    install_id: str
    trusted: bool
    active: bool
    reviewed_at: str | None
    activated_at: str | None


def review_comfy_registry_install(
    session: Session,
    *,
    install_id: str,
    trusted: bool,
    custom_node_root: Path,
    environment_root: Path,
    media_worker_stopped: bool,
) -> ComfyRegistryActivationState:
    """Record an explicit local trust decision after exact stopped-worker verification."""
    _require_stopped(media_worker_stopped)
    install = _install(session, install_id)
    if trusted:
        _verify_install(
            session,
            install,
            custom_node_root=custom_node_root,
            environment_root=environment_root,
        )
        install.trusted = True
    else:
        install.trusted = False
        install.active = False
    reviewed_at = utcnow().isoformat()
    install.review_json = {
        **install.review_json,
        "reviewed_at": reviewed_at,
        "trusted_by_local_user": trusted,
    }
    session.commit()
    session.refresh(install)
    return _state(install)


async def activate_comfy_registry_install(
    session: Session,
    *,
    install_id: str,
    custom_node_root: Path,
    environment_root: Path,
    media_worker_stopped: bool,
    start_media: MediaStarter,
    omission: OmissionRequirement | None = None,
    read_node_inventory: NodeInventoryReader | None = None,
) -> ComfyRegistryActivationState:
    """Activate one trusted package and restore the prior runtime if startup fails.

    When `omission` is given, this activation is a trial: the package declares
    a source dependency that was left out, and starting is not enough to
    conclude it was unnecessary. The authorized workflow's exact required node
    types must be present afterwards, and a runtime that cannot show them is
    rolled back like any other failed activation.
    """
    _require_stopped(media_worker_stopped)
    install = _install(session, install_id)
    if not install.trusted:
        raise ComfyRegistryActivationError(
            "registry_install_untrusted",
            "The Registry package must be explicitly trusted before activation",
        )
    _verify_install(
        session,
        install,
        custom_node_root=custom_node_root,
        environment_root=environment_root,
    )
    archive_path = _archive_path(custom_node_root, install.installed_path)
    before_start = snapshot_staged_comfy_registry_files(archive_path)
    install.active = True
    session.commit()
    try:
        await start_media()
    except asyncio.CancelledError:
        _deactivate(session, install_id, failure_code="activation_cancelled")
        await _restore_after_cancellation(start_media)
        raise
    except Exception as exc:
        _deactivate(session, install_id, failure_code="activation_start_failed")
        try:
            await start_media()
        except (Exception, asyncio.CancelledError) as restore_exc:
            raise ComfyRegistryActivationError(
                "activation_restore_failed",
                "Registry activation failed and the prior media runtime could not be restored",
            ) from restore_exc
        raise ComfyRegistryActivationError(
            "activation_start_failed",
            "Registry activation failed; the prior media runtime was restored",
        ) from exc
    activated = _install(session, install_id)
    try:
        review = activated.review_json
        runtime_files = capture_staged_comfy_registry_runtime_files(
            archive_path,
            before_start=before_start,
            expected_manifest_sha256=activated.manifest_sha256,
            expected_file_count=_review_count(review, "file_count"),
            expected_expanded_bytes=_review_count(review, "expanded_bytes"),
            runtime_files=parse_comfy_registry_runtime_files(review.get("runtime_files")),
        )
    except (ComfyRegistryArchiveError, OSError, ValueError) as exc:
        _deactivate(session, install_id, failure_code="activation_runtime_files_failed")
        try:
            await start_media()
        except (Exception, asyncio.CancelledError) as restore_exc:
            raise ComfyRegistryActivationError(
                "activation_restore_failed",
                "Registry activation changed its package files and the prior media "
                "runtime could not be restored",
            ) from restore_exc
        raise ComfyRegistryActivationError(
            "activation_runtime_files_failed",
            "Registry activation changed package files outside the bounded data contract; "
            "the prior media runtime was restored",
        ) from exc
    proof: dict[str, Any] | None = None
    if omission is not None:
        # After startup and after the file contract, because both of those
        # can restore on their own terms; this one restores the same way.
        if read_node_inventory is None:
            await _roll_back(session, install_id, start_media, "omission_unverifiable")
            raise ComfyRegistryActivationError(
                "omission_unverifiable",
                "An omitted source dependency cannot be proven unnecessary without "
                "reading the runtime's node inventory; the prior runtime was restored",
            )
        try:
            observed = await read_node_inventory()
            proof = prove_omission(omission, observed_node_types=observed)
        except Exception as exc:
            # A reader that fails is not a proof either, and it fails the same
            # way a refused proof does rather than leaving the trial standing.
            code = exc.code if isinstance(exc, OmissionProofError) else "omission_unverifiable"
            await _roll_back(session, install_id, start_media, code)
            raise ComfyRegistryActivationError(
                code,
                f"{exc} - the prior media runtime was restored",
            ) from exc
    activated = _install(session, install_id)
    activated.review_json = {
        **activated.review_json,
        "activated_at": utcnow().isoformat(),
        "activation_failure_code": None,
        "runtime_files": comfy_registry_runtime_files_json(runtime_files),
        **(
            {"source_omission_proof": proof, "source_omission_digest": evidence_digest(proof)}
            if proof is not None
            else {}
        ),
    }
    session.commit()
    session.refresh(activated)
    return _state(activated)


async def _roll_back(
    session: Session,
    install_id: str,
    start_media: MediaStarter,
    failure_code: str,
) -> None:
    """Undo a trial activation, restoring exactly what was running before."""
    _deactivate(session, install_id, failure_code=failure_code)
    try:
        await start_media()
    except (Exception, asyncio.CancelledError) as restore_exc:
        raise ComfyRegistryActivationError(
            "activation_restore_failed",
            "The trial activation was undone but the prior media runtime could not be restored",
        ) from restore_exc


async def deactivate_comfy_registry_install(
    session: Session,
    *,
    install_id: str,
    media_worker_stopped: bool,
    start_media: MediaStarter,
) -> ComfyRegistryActivationState:
    """Deactivate one package and restart the media worker without it."""
    _require_stopped(media_worker_stopped)
    install = _install(session, install_id)
    install.active = False
    install.review_json = {
        **install.review_json,
        "deactivated_at": utcnow().isoformat(),
    }
    session.commit()
    try:
        await start_media()
    except (Exception, asyncio.CancelledError) as exc:
        raise ComfyRegistryActivationError(
            "deactivation_restart_failed",
            "The Registry package is inactive but the media runtime did not restart",
        ) from exc
    current = _install(session, install_id)
    return _state(current)


def _verify_install(
    session: Session,
    install: ComfyRegistryInstall,
    *,
    custom_node_root: Path,
    environment_root: Path,
) -> None:
    original_trusted = install.trusted
    original_active = install.active
    install.trusted = True
    install.active = True
    session.flush()
    try:
        contract = trusted_comfy_registry_launch_contract(
            session,
            custom_node_root=custom_node_root,
            environment_root=environment_root,
        )
        if install.installed_path not in contract.custom_node_folders:
            raise ComfyRegistryActivationError(
                "registry_install_verification_failed",
                "Registry package verification did not produce its launch binding",
            )
    except Exception as exc:
        session.rollback()
        if isinstance(exc, ComfyRegistryActivationError):
            raise
        raise ComfyRegistryActivationError(
            "registry_install_verification_failed",
            "Registry package files or dependencies failed verification",
        ) from exc
    install.trusted = original_trusted
    install.active = original_active


def _deactivate(session: Session, install_id: str, *, failure_code: str) -> None:
    session.rollback()
    install = _install(session, install_id)
    install.active = False
    install.review_json = {
        **install.review_json,
        "activation_failed_at": utcnow().isoformat(),
        "activation_failure_code": failure_code,
    }
    session.commit()


def _archive_path(root: Path, installed_path: str) -> Path:
    try:
        managed_root = root.resolve(strict=True)
        path = (managed_root / installed_path).resolve(strict=True)
    except OSError as exc:
        raise ComfyRegistryActivationError(
            "registry_install_verification_failed",
            "Registry package files failed verification",
        ) from exc
    if path.parent != managed_root:
        raise ComfyRegistryActivationError(
            "registry_install_verification_failed",
            "Registry package files failed verification",
        )
    return path


def _review_count(review: object, key: str) -> int:
    value = review.get(key) if isinstance(review, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ComfyRegistryArchiveError("invalid staged Registry archive identity")
    return value


async def _restore_after_cancellation(start_media: MediaStarter) -> None:
    task: asyncio.Future[object] = asyncio.ensure_future(start_media())
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception as exc:
            raise ComfyRegistryActivationError(
                "activation_restore_failed",
                "Cancelled Registry activation could not restore the prior media runtime",
            ) from exc
    try:
        task.result()
    except Exception as exc:
        raise ComfyRegistryActivationError(
            "activation_restore_failed",
            "Cancelled Registry activation could not restore the prior media runtime",
        ) from exc


def _require_stopped(value: bool) -> None:
    if value is not True:
        raise ComfyRegistryActivationError(
            "media_worker_running",
            "The media worker must be stopped before changing Registry activation",
        )


def _install(session: Session, install_id: str) -> ComfyRegistryInstall:
    if not isinstance(install_id, str) or not install_id or len(install_id) > 80:
        raise ComfyRegistryActivationError(
            "registry_install_not_found", "Registry package install was not found"
        )
    install = session.get(ComfyRegistryInstall, install_id)
    if install is None:
        raise ComfyRegistryActivationError(
            "registry_install_not_found", "Registry package install was not found"
        )
    return install


def _state(install: ComfyRegistryInstall) -> ComfyRegistryActivationState:
    review = install.review_json
    reviewed_at = review.get("reviewed_at") if isinstance(review, dict) else None
    activated_at = review.get("activated_at") if isinstance(review, dict) else None
    return ComfyRegistryActivationState(
        install.id,
        install.trusted,
        install.active,
        reviewed_at if isinstance(reviewed_at, str) else None,
        activated_at if isinstance(activated_at, str) else None,
    )
