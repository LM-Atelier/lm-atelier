"""Activate exact prepared packages together and retain their rollback state."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from .comfy_registry import MAX_REGISTRY_PACKAGES
from .comfy_registry_activation import ComfyRegistryActivationError, _review_count
from .comfy_registry_archives import (
    ComfyRegistryRuntimeFile,
    capture_staged_comfy_registry_runtime_files,
    comfy_registry_runtime_files_json,
    parse_comfy_registry_runtime_files,
    snapshot_staged_comfy_registry_files,
)
from .comfy_registry_dependencies import (
    ComfyRegistryDependencyError,
    plan_comfy_registry_dependencies,
)
from .comfy_registry_launch_verification import (
    VerifiedComfyRegistryLaunch,
    _verify_comfy_registry_launch,
)
from .comfy_registry_lifecycle import ComfyRegistryPreparation
from .comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies
from .comfy_registry_mixed_wheel_closure import ComfyRegistryMixedWheelClosure
from .comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from .comfy_registry_sources import resolve_comfy_package_source
from .domain import utcnow
from .models import ComfyRegistryInstall
from .schemas import ApiModel
from .source_omission_proof import evidence_digest, pending_omission_requirement, prove_omission
from .workflow_activations import (
    WorkflowRegistryLaunchBinding,
    _registry_install_launch_binding,
    _reject_node_type_collisions,
)
from .workflow_package_execution_plan import WorkflowPackageExecutionPlan
from .workflow_trust import canonical_graph

_BATCH_KEY = "activation_batch_v1"
SessionFactory = Callable[[], Session]
BatchStarter = Callable[[tuple[WorkflowRegistryLaunchBinding, ...]], Awaitable[object]]
StoppedVerifier = Callable[[], Awaitable[bool]]
NodeReader = Callable[[], Awaitable[frozenset[str]]]


class _BatchRecord(ApiModel):
    id: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: Literal["starting", "verified", "complete", "failed"]
    previous_active: dict[str, bool]
    before_start: tuple[ComfyRegistryRuntimeFile, ...]
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _review(row: ComfyRegistryInstall) -> dict[str, Any]:
    return {key: value for key, value in row.review_json.items() if key != _BATCH_KEY}


def _review_digest(row: ComfyRegistryInstall) -> str:
    return hashlib.sha256(canonical_graph(_review(row)).encode("utf-8")).hexdigest()


def _write_record(
    row: ComfyRegistryInstall,
    identifier: str,
    state: Literal["starting", "verified", "complete", "failed"],
    previous: dict[str, bool],
    before: tuple[ComfyRegistryRuntimeFile, ...],
) -> None:
    row.review_json = {
        **_review(row),
        _BATCH_KEY: _BatchRecord(
            id=identifier,
            state=state,
            previous_active=previous,
            before_start=before,
            review_sha256=_review_digest(row),
        ).model_dump(mode="json"),
    }


def _refuse(code: str) -> ComfyRegistryActivationError:
    return ComfyRegistryActivationError(code, "The extension activation could not be completed.")


def _reserve(session: Session) -> None:
    session.execute(text("UPDATE comfy_registry_installs SET active = active WHERE 0"))
    session.expire_all()


def _row(
    session: Session,
    preparation: ComfyRegistryPreparation,
    plan: WorkflowPackageExecutionPlan,
) -> ComfyRegistryInstall:
    return _check_row(session.get(ComfyRegistryInstall, preparation.install_id), preparation, plan)


def _check_row(
    row: ComfyRegistryInstall | None,
    preparation: ComfyRegistryPreparation,
    plan: WorkflowPackageExecutionPlan,
) -> ComfyRegistryInstall:
    plan.verify()
    source = resolve_comfy_package_source(plan.resolution)
    if (
        row is None
        or row.id != preparation.install_id
        or any(
            getattr(row, name) != getattr(preparation, name)
            for name in (
                "installed_path",
                "wheel_environment_path",
                "archive_sha256",
                "manifest_sha256",
                "wheel_closure_sha256",
                "wheel_environment_sha256",
            )
        )
    ):
        raise _refuse("registry_batch_identity_changed")
    try:
        dependencies = (
            plan_comfy_registry_mixed_dependencies(row.pip_dependencies_json)
            if isinstance(plan.closure, ComfyRegistryMixedWheelClosure)
            else plan_comfy_registry_dependencies(row.pip_dependencies_json)
        )
    except ComfyRegistryDependencyError as exc:
        raise _refuse("registry_batch_identity_changed") from exc
    if (
        (
            row.package_id,
            row.package_version,
            row.registry_record_id,
            row.repository_url,
            row.download_url,
        )
        != (
            source.package_id,
            source.package_version,
            source.source_record_id,
            source.repository_url,
            source.download_url,
        )
        or sorted(row.node_types_json) != sorted(plan.resolution.node_types)
        or row.review_json.get("registry_warnings") != list(plan.resolution.warnings)
        or row.archive_sha256 != plan.archive.archive_sha256
        or row.manifest_sha256 != plan.archive.manifest_sha256
        or row.wheel_closure_sha256 != plan.closure.closure_sha256
        or dependencies.declaration_sha256 != plan.closure.manifest.declaration_sha256
    ):
        raise _refuse("registry_batch_identity_changed")
    return row


def _record(row: ComfyRegistryInstall) -> _BatchRecord | None:
    value = row.review_json.get(_BATCH_KEY)
    if value is None:
        return None
    try:
        return _BatchRecord.model_validate(value)
    except ValueError as exc:
        raise _refuse("registry_batch_record_changed") from exc


@dataclass(frozen=True)
class RegistryActivationBatch:
    id: str
    preparations: tuple[ComfyRegistryPreparation, ...]
    execution_plans: dict[str, WorkflowPackageExecutionPlan]
    bindings: tuple[WorkflowRegistryLaunchBinding, ...]
    previous_active: dict[str, bool]
    custom_node_root: Path
    environment_root: Path
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None
    completion_verification: VerifiedComfyRegistryLaunch | None = None

    def _rows(self, session: Session, state: str) -> list[ComfyRegistryInstall]:
        rows = [
            _row(session, item, self.execution_plans[item.install_id]) for item in self.preparations
        ]
        self._check_rows(rows, state)
        return rows

    def _check_rows(self, rows: list[ComfyRegistryInstall], state: str) -> None:
        if tuple(row.id for row in rows) != tuple(item.install_id for item in self.preparations):
            raise _refuse("registry_batch_identity_changed")
        for row, preparation in zip(rows, self.preparations, strict=True):
            _check_row(row, preparation, self.execution_plans[row.id])
            record = _record(row)
            if (
                record is None
                or record.id != self.id
                or record.state != state
                or record.previous_active != self.previous_active
                or not row.trusted
                or not row.active
                or record.review_sha256 != _review_digest(row)
            ):
                raise _refuse("registry_batch_record_changed")

    def _check_bindings(self, rows: list[ComfyRegistryInstall]) -> None:
        bindings = tuple(
            _registry_install_launch_binding(
                row,
                custom_node_root=self.custom_node_root,
                environment_root=self.environment_root,
            )
            for row in rows
        )
        if bindings != self.bindings:
            raise _refuse("registry_batch_identity_changed")

    def _verify(self, session_factory: SessionFactory, state: str) -> VerifiedComfyRegistryLaunch:
        def prepare(rows: list[ComfyRegistryInstall]) -> None:
            self._check_rows(rows, state)
            self._check_bindings(rows)

        return _verify_comfy_registry_launch(
            session_factory,
            [item.install_id for item in self.preparations],
            include_active=False,
            custom_node_root=self.custom_node_root,
            environment_root=self.environment_root,
            reviewed_inputs=self.reviewed_inputs,
            prepare_installs=prepare,
        )

    def complete(self, session: Session) -> None:
        """Stage completion in the transaction that accepts the executable workflow."""
        if self.completion_verification is None:
            raise _refuse("registry_batch_not_verified")
        self.completion_verification.require_current(session)
        session.expire_all()
        rows = self._rows(session, "verified")
        for row in rows:
            record = _record(row)
            assert record is not None
            _write_record(row, self.id, "complete", self.previous_active, record.before_start)
        session.flush()

    def verify_completion(self, session_factory: SessionFactory) -> RegistryActivationBatch:
        """Refresh physical verification after consumer work and before its final writer."""
        return replace(self, completion_verification=self._verify(session_factory, "verified"))


def _begin(
    session_factory: SessionFactory,
    purpose: str,
    preparations: Sequence[ComfyRegistryPreparation],
    execution_plans: Mapping[str, WorkflowPackageExecutionPlan],
    custom_node_root: Path,
    environment_root: Path,
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None,
) -> tuple[RegistryActivationBatch, dict[str, tuple[ComfyRegistryRuntimeFile, ...]]]:
    ordered = tuple(sorted(preparations, key=lambda item: item.install_id))
    identifiers = {item.install_id for item in ordered}
    if (
        not purpose
        or len(purpose) > 160
        or not ordered
        or len(ordered) > MAX_REGISTRY_PACKAGES
        or len(identifiers) != len(ordered)
        or set(execution_plans) != identifiers
    ):
        raise _refuse("registry_batch_invalid")
    plans = {key: execution_plans[key].model_copy(deep=True) for key in sorted(identifiers)}
    digest = hashlib.sha256(
        canonical_graph(
            {
                "purpose": purpose,
                "execution_plans": {key: plan.plan_sha256 for key, plan in plans.items()},
                "preparations": [
                    {
                        key: value
                        for key, value in asdict(item).items()
                        if key != "reused_wheel_environment"
                    }
                    for item in ordered
                ],
            }
        ).encode("utf-8")
    ).hexdigest()
    previous: dict[str, bool] = {}
    bindings: tuple[WorkflowRegistryLaunchBinding, ...] = ()
    before: dict[str, tuple[ComfyRegistryRuntimeFile, ...]] = {}
    staged: list[ComfyRegistryInstall] = []

    def prepare(rows: list[ComfyRegistryInstall]) -> None:
        nonlocal previous, bindings, before
        for row, item in zip(rows, ordered, strict=True):
            _check_row(row, item, plans[item.install_id])
        previous = {row.id: row.active for row in rows}
        pending = [_record(row) for row in rows]
        unfinished = [item for item in pending if item and item.state in {"starting", "verified"}]
        if unfinished:
            first = unfinished[0]
            if (
                len(unfinished) != len(rows)
                or first.id != digest
                or set(first.previous_active) != identifiers
                or any(
                    item.id != first.id
                    or item.state != first.state
                    or item.previous_active != first.previous_active
                    for item in unfinished
                )
            ):
                raise _refuse("registry_batch_in_progress")
            previous = first.previous_active
        elif any(item and item.id == digest and item.state == "complete" for item in pending):
            raise _refuse("registry_batch_already_complete")
        if any(not row.trusted for row in rows):
            raise _refuse("registry_install_untrusted")
        for row in rows:
            row.active = True
        bindings = tuple(
            _registry_install_launch_binding(
                row,
                custom_node_root=custom_node_root,
                environment_root=environment_root,
            )
            for row in rows
        )
        _reject_node_type_collisions((), bindings)
        if unfinished:
            for row, binding in zip(rows, bindings, strict=True):
                record = _record(row)
                assert record is not None
                if record.review_sha256 != _review_digest(row):
                    raise _refuse("registry_batch_record_changed")
                if record.state == "verified":
                    continue
                recovered = capture_staged_comfy_registry_runtime_files(
                    binding.installed_path,
                    before_start=record.before_start,
                    expected_manifest_sha256=row.manifest_sha256,
                    expected_file_count=_review_count(row.review_json, "file_count"),
                    expected_expanded_bytes=_review_count(row.review_json, "expanded_bytes"),
                    runtime_files=parse_comfy_registry_runtime_files(
                        row.review_json.get("runtime_files")
                    ),
                )
                row.review_json = {
                    **row.review_json,
                    "runtime_files": comfy_registry_runtime_files_json(recovered),
                }
        before = {
            item.registry_install_id: snapshot_staged_comfy_registry_files(item.installed_path)
            for item in bindings
        }
        for row in rows:
            pending_omission_requirement(row.id, row.review_json)
            _write_record(row, digest, "starting", previous, before[row.id])
        staged.extend(rows)

    verified = _verify_comfy_registry_launch(
        session_factory,
        sorted(identifiers),
        include_active=False,
        custom_node_root=custom_node_root,
        environment_root=environment_root,
        reviewed_inputs=reviewed_inputs,
        prepare_installs=prepare,
    )
    with session_factory() as session:
        verified.require_current(session)
        for install in staged:
            current = session.get(ComfyRegistryInstall, install.id)
            assert current is not None
            current.active = True
            current.review_json = install.review_json
        session.commit()
    return RegistryActivationBatch(
        digest,
        ordered,
        plans,
        bindings,
        previous,
        custom_node_root,
        environment_root,
        reviewed_inputs,
    ), before


def _verified(
    session_factory: SessionFactory,
    batch: RegistryActivationBatch,
    before: dict[str, tuple[ComfyRegistryRuntimeFile, ...]],
    observed: frozenset[str],
) -> RegistryActivationBatch:
    staged: list[ComfyRegistryInstall] = []

    def prepare(rows: list[ComfyRegistryInstall]) -> None:
        batch._check_rows(rows, "starting")
        batch._check_bindings(rows)
        expected = {node for binding in batch.bindings for node in binding.node_types}
        if not expected.issubset(observed):
            raise _refuse("registry_node_types_missing")
        for row, binding in zip(rows, batch.bindings, strict=True):
            review = row.review_json
            runtime_files = capture_staged_comfy_registry_runtime_files(
                binding.installed_path,
                before_start=before[row.id],
                expected_manifest_sha256=row.manifest_sha256,
                expected_file_count=_review_count(review, "file_count"),
                expected_expanded_bytes=_review_count(review, "expanded_bytes"),
                runtime_files=parse_comfy_registry_runtime_files(review.get("runtime_files")),
            )
            omission = pending_omission_requirement(row.id, review)
            proof = prove_omission(omission, observed_node_types=observed) if omission else None
            row.review_json = {
                **review,
                "activated_at": utcnow().isoformat(),
                "activation_failure_code": None,
                "runtime_files": comfy_registry_runtime_files_json(runtime_files),
                "source_omission_proof": proof,
                "source_omission_digest": evidence_digest(proof) if proof is not None else None,
            }
            _write_record(row, batch.id, "verified", batch.previous_active, before[row.id])
        staged.extend(rows)

    verified = _verify_comfy_registry_launch(
        session_factory,
        [item.install_id for item in batch.preparations],
        include_active=False,
        custom_node_root=batch.custom_node_root,
        environment_root=batch.environment_root,
        reviewed_inputs=batch.reviewed_inputs,
        prepare_installs=prepare,
    )
    with session_factory() as session:
        verified.require_current(session)
        for install in staged:
            current = session.get(ComfyRegistryInstall, install.id)
            assert current is not None
            current.review_json = install.review_json
        session.commit()
    return batch.verify_completion(session_factory)


async def _rollback(
    session_factory: SessionFactory,
    batch: RegistryActivationBatch,
    stop_media: StoppedVerifier,
) -> None:
    if await stop_media() is not True:
        raise _refuse("media_worker_running")
    try:
        await _worker(lambda: _restore_flags(session_factory, batch))
    except (ValueError, OSError):
        await _worker(
            lambda: _deactivate_pending(
                session_factory, [item.install_id for item in batch.preparations], force=True
            )
        )
        raise


def _restore_flags(session_factory: SessionFactory, batch: RegistryActivationBatch) -> None:
    proofs: dict[str, VerifiedComfyRegistryLaunch] = {}
    for binding in batch.bindings:
        if not batch.previous_active[binding.registry_install_id]:
            continue

        def prepare(
            rows: list[ComfyRegistryInstall], expected: WorkflowRegistryLaunchBinding = binding
        ) -> None:
            row = rows[0]
            if not row.trusted:
                raise _refuse("registry_install_untrusted")
            row.active = True
            current = _registry_install_launch_binding(
                row,
                custom_node_root=batch.custom_node_root,
                environment_root=batch.environment_root,
            )
            if current != expected:
                raise _refuse("registry_batch_identity_changed")

        try:
            proofs[binding.registry_install_id] = _verify_comfy_registry_launch(
                session_factory,
                [binding.registry_install_id],
                include_active=False,
                custom_node_root=batch.custom_node_root,
                environment_root=batch.environment_root,
                reviewed_inputs=batch.reviewed_inputs,
                prepare_installs=prepare,
            )
        except (ValueError, OSError):
            continue
    with session_factory() as session:
        _reserve(session)
        rows = [
            _row(session, item, batch.execution_plans[item.install_id])
            for item in batch.preparations
        ]
        for row in rows:
            record = _record(row)
            if (
                record is None
                or record.id != batch.id
                or record.previous_active != batch.previous_active
            ):
                raise _refuse("registry_batch_record_changed")
        restored: set[str] = set()
        for identifier, proof in proofs.items():
            try:
                proof.require_current(session)
            except (ValueError, OSError):
                continue
            restored.add(identifier)
        for row in rows:
            row.active = row.id in restored
            row.review_json = {
                **row.review_json,
                "activation_failure_code": "registry_batch_failed",
            }
            record = _record(row)
            assert record is not None
            _write_record(row, batch.id, "failed", batch.previous_active, record.before_start)
        session.commit()


def _deactivate_pending(
    session_factory: SessionFactory,
    install_ids: Sequence[str],
    *,
    force: bool = False,
) -> None:
    """Quarantine activation flags after the caller proves the media worker stopped."""
    with session_factory() as session:
        _reserve(session)
        deactivate_pending_registry_packages(session, install_ids, force=force)
        session.commit()


def registry_activation_pending(row: ComfyRegistryInstall) -> bool:
    return _BATCH_KEY in row.review_json and (
        not isinstance(row.review_json[_BATCH_KEY], dict)
        or row.review_json[_BATCH_KEY].get("state") not in {"complete", "failed"}
    )


def deactivate_pending_registry_packages(
    session: Session, install_ids: Sequence[str], *, force: bool = False
) -> None:
    """Clear pending flags within a reserved writer after the media worker has stopped."""
    rows = [session.get(ComfyRegistryInstall, identifier) for identifier in install_ids]
    pending = any(row is not None and registry_activation_pending(row) for row in rows)
    if not force and not pending:
        return
    for row in rows:
        if row is None:
            continue
        value = row.review_json.get(_BATCH_KEY)
        if not force and isinstance(value, dict) and value.get("state") == "complete":
            continue
        row.active = False
        row.review_json = {
            **row.review_json,
            "activation_failure_code": "registry_batch_failed",
        }


def recover_registry_package_batch(
    session_factory: SessionFactory,
    *,
    purpose: str,
    preparations: Sequence[ComfyRegistryPreparation],
    execution_plans: Mapping[str, WorkflowPackageExecutionPlan],
    custom_node_root: Path,
    environment_root: Path,
    media_worker_stopped: bool,
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None,
) -> None:
    """Adopt interrupted runtime files before ordinary preparation verifies the package."""
    if media_worker_stopped is not True:
        raise _refuse("media_worker_running")
    try:
        with session_factory() as session:
            records = [
                _record(_row(session, item, execution_plans[item.install_id]))
                for item in preparations
            ]
        if not any(record and record.state in {"starting", "verified"} for record in records):
            return
        batch, _before = _begin(
            session_factory,
            purpose,
            preparations,
            execution_plans,
            custom_node_root,
            environment_root,
            reviewed_inputs,
        )
        _restore_flags(session_factory, batch)
    except (ValueError, OSError):
        _deactivate_pending(session_factory, [item.install_id for item in preparations])
        raise


@asynccontextmanager
async def activate_registry_package_batch(
    session_factory: SessionFactory,
    *,
    purpose: str,
    preparations: Sequence[ComfyRegistryPreparation],
    execution_plans: Mapping[str, WorkflowPackageExecutionPlan],
    custom_node_root: Path,
    environment_root: Path,
    media_worker_stopped: bool,
    start_media: BatchStarter,
    stop_media: StoppedVerifier,
    read_node_inventory: NodeReader,
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None,
) -> AsyncIterator[RegistryActivationBatch]:
    """Hold the caller's primary lease through one start, completion or verified rollback."""
    if media_worker_stopped is not True:
        raise _refuse("media_worker_running")
    beginning = asyncio.create_task(
        asyncio.to_thread(
            _begin,
            session_factory,
            purpose,
            preparations,
            execution_plans,
            custom_node_root,
            environment_root,
            reviewed_inputs,
        )
    )
    try:
        batch, before = await asyncio.shield(beginning)
    except asyncio.CancelledError:
        try:
            batch, before = await _drain(beginning)
        except (ValueError, OSError):
            await _drain(
                asyncio.create_task(
                    asyncio.to_thread(
                        _deactivate_pending,
                        session_factory,
                        [item.install_id for item in preparations],
                    )
                )
            )
            raise
        await _drain(asyncio.create_task(_rollback(session_factory, batch, stop_media)))
        raise
    except (ValueError, OSError):
        await _worker(
            lambda: _deactivate_pending(session_factory, [item.install_id for item in preparations])
        )
        raise
    try:
        await start_media(batch.bindings)
        observed = await read_node_inventory()
        batch = await _worker(lambda: _verified(session_factory, batch, before, observed))
        yield batch
        with session_factory() as session:
            batch._rows(session, "complete")
    except (Exception, asyncio.CancelledError):
        rollback = asyncio.create_task(_rollback(session_factory, batch, stop_media))
        await _drain(rollback)
        raise


async def _drain[T](task: asyncio.Task[T]) -> T:
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _worker[T](operation: Callable[[], T]) -> T:
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _drain(task)
        raise
