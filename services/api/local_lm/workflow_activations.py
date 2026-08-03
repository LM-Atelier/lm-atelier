from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapters.contracts import ADAPTER_CONTRACT_VERSION
from .auxiliary_assets import COMFY_AUXILIARY_FOLDERS
from .domain import new_id, utcnow
from .model_planner import LAUNCH_CONTRACT_VERSION
from .models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    ModelComponentManifest,
    ModelInstall,
    ModelProfile,
    WorkflowActivation,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowRevision,
)
from .workflow_bindings import (
    MaterializedWorkflowDependency,
    ResolvedWorkflowBinding,
    WorkflowActivationResolution,
    WorkflowBindingError,
    WorkflowBindingIssue,
    WorkflowBindingSelection,
    materialize_custom_node,
    materialize_model_asset,
    materialize_model_install,
    materialize_model_profile,
    materialize_registry_package,
    materialize_runtime,
    resolve_workflow_activation,
)
from .workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyError,
    WorkflowDependencyRequirement,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_sha256,
)

if TYPE_CHECKING:
    from .runtime_provisioning import RuntimeProvisioner

WORKFLOW_ACTIVATION_RESOLVER_VERSION = "workflow-activation-v1"
MAX_CUSTOM_NODE_TYPES = 4_096
MAX_NODE_TYPE_LENGTH = 200

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RESOLVER_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,39}$")
_COMFY_MODEL_FOLDERS = frozenset(
    {
        "checkpoints",
        "diffusion_models",
        "text_encoders",
        "vae",
        "clip_vision",
        "loras",
        "controlnet",
        "upscale_models",
        "embeddings",
        "ipadapter",
    }
)


class WorkflowActivationError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        issues: Sequence[WorkflowBindingIssue] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.issues = tuple(issues)


@dataclass(frozen=True)
class WorkflowModelComponentLaunchBinding:
    target_folder: str
    runtime_reference: str
    sha256: str


@dataclass(frozen=True)
class WorkflowModelLaunchBinding:
    model_install_id: str
    base_path: Path
    comfy_paths: tuple[tuple[str, str], ...]
    components: tuple[WorkflowModelComponentLaunchBinding, ...]


@dataclass(frozen=True)
class WorkflowAssetLaunchBinding:
    model_asset_install_id: str
    base_path: Path
    loader_folder: str
    runtime_reference: str
    sha256: str


@dataclass(frozen=True)
class WorkflowCustomNodeLaunchBinding:
    custom_node_install_id: str
    installed_path: Path
    source_url: str
    revision: str
    tree_hash: str
    node_types: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowRegistryLaunchBinding:
    registry_install_id: str
    installed_path: Path
    site_packages: Path
    package_id: str
    package_version: str
    archive_sha256: str
    manifest_sha256: str
    wheel_closure_sha256: str
    wheel_environment_sha256: str
    node_types: tuple[str, ...]


@dataclass(frozen=True)
class WorkflowRuntimeLaunchBinding:
    runtime_key: str
    resource_identity_sha256: str
    identity_json: str


@dataclass(frozen=True)
class WorkflowActivationLaunchScope:
    activation_id: str
    workflow_revision_id: str
    binding_sha256: str
    launch_sha256: str
    model_install_ids: tuple[str, ...]
    model_asset_install_ids: tuple[str, ...]
    custom_node_install_ids: tuple[str, ...]
    registry_install_ids: tuple[str, ...]
    runtime_keys: tuple[str, ...]
    models: tuple[WorkflowModelLaunchBinding, ...]
    assets: tuple[WorkflowAssetLaunchBinding, ...]
    custom_nodes: tuple[WorkflowCustomNodeLaunchBinding, ...]
    registry_packages: tuple[WorkflowRegistryLaunchBinding, ...]
    runtimes: tuple[WorkflowRuntimeLaunchBinding, ...]


WorkflowRuntimeMaterializer = Callable[
    [WorkflowDependencyRequirement, WorkflowBindingSelection],
    MaterializedWorkflowDependency | None,
]


def materialize_comfy_runtime_dependency(
    provisioner: RuntimeProvisioner,
    _requirement: WorkflowDependencyRequirement,
    _selection: WorkflowBindingSelection,
) -> MaterializedWorkflowDependency | None:
    """Materialize the verified current Comfy runtime using stable launch contracts."""

    status = provisioner.status("comfyui")
    if status.state != "ready":
        return None
    return materialize_runtime(
        engine="comfyui",
        runtime_build=status.release,
        adapter_contract_version=ADAPTER_CONTRACT_VERSION,
        launch_contract_version=LAUNCH_CONTRACT_VERSION,
    )


@dataclass(frozen=True)
class _LaunchResources:
    model_install_ids: tuple[str, ...]
    model_asset_install_ids: tuple[str, ...]
    custom_node_install_ids: tuple[str, ...]
    registry_install_ids: tuple[str, ...]
    runtime_keys: tuple[str, ...]
    models: tuple[WorkflowModelLaunchBinding, ...]
    assets: tuple[WorkflowAssetLaunchBinding, ...]
    custom_nodes: tuple[WorkflowCustomNodeLaunchBinding, ...]
    registry_packages: tuple[WorkflowRegistryLaunchBinding, ...]
    runtimes: tuple[WorkflowRuntimeLaunchBinding, ...]


def activate_workflow_revision(
    session: Session,
    revision: WorkflowRevision | str,
    selections: Sequence[WorkflowBindingSelection],
    *,
    runtime_materializer: WorkflowRuntimeMaterializer | None = None,
    resolver_version: str = WORKFLOW_ACTIVATION_RESOLVER_VERSION,
    custom_node_root: Path | None = None,
    registry_environment_root: Path | None = None,
) -> WorkflowActivationLaunchScope:
    """Resolve and persist one exact activation without committing the caller's transaction."""

    revision_row = _revision_row(session, revision)
    contract, slots = _hydrate_contract(session, revision_row)
    version = _resolver_version(resolver_version)
    resolution = _resolve(
        session,
        contract,
        selections,
        runtime_materializer=runtime_materializer,
    )
    if not resolution.complete or resolution.binding_sha256 is None:
        raise WorkflowActivationError(
            "workflow_activation_incomplete",
            "Workflow dependencies are incomplete or unavailable",
            issues=resolution.issues,
        )

    existing = session.scalar(
        select(WorkflowActivation).where(
            WorkflowActivation.workflow_revision_id == revision_row.id,
            WorkflowActivation.binding_sha256 == resolution.binding_sha256,
        )
    )
    try:
        resources = _launch_resources(
            session,
            resolution.bindings,
            selections,
            custom_node_root=custom_node_root,
            registry_environment_root=registry_environment_root,
        )
        if existing is not None:
            if existing.dependency_contract_sha256 != revision_row.dependency_contract_sha256:
                raise WorkflowActivationError(
                    "workflow_contract_drift",
                    "Stored workflow activation uses a different dependency contract",
                )
            _assert_persisted_snapshot(existing, slots, resolution.bindings, selections)
    except WorkflowActivationError as exc:
        if existing is not None:
            _mark_stale(session, existing, exc.code, str(exc))
        raise

    with session.begin_nested():
        activation = existing
        if activation is None:
            activation = WorkflowActivation(
                id=new_id("wfact"),
                workflow_revision_id=revision_row.id,
                resolver_version=version,
                dependency_contract_sha256=cast(str, revision_row.dependency_contract_sha256),
                binding_sha256=resolution.binding_sha256,
                state="ready",
                is_active=False,
            )
            session.add(activation)
            session.flush()
            _persist_bindings(activation, slots, resolution.bindings, selections, session)
        activation.resolver_version = version
        activation.dependency_contract_sha256 = cast(str, revision_row.dependency_contract_sha256)
        activation.state = "ready"
        activation.is_active = False
        activation.last_validated_at = utcnow()
        activation.invalidated_at = None
        activation.invalidation_code = None
        activation.invalidation_reason = None
        session.flush()

        active_rows = session.scalars(
            select(WorkflowActivation).where(
                WorkflowActivation.workflow_revision_id == revision_row.id,
                WorkflowActivation.is_active.is_(True),
                WorkflowActivation.id != activation.id,
            )
        ).all()
        for active in active_rows:
            active.is_active = False
        session.flush()

        launch_sha256 = _launch_sha256(resolution.binding_sha256, resources)
        activation.details_json = {"launch_sha256": launch_sha256}
        activation.is_active = True
        session.flush()

    return _launch_scope(activation, resources, launch_sha256)


def revalidate_workflow_activation(
    session: Session,
    activation: WorkflowActivation | str,
    *,
    runtime_materializer: WorkflowRuntimeMaterializer | None = None,
    custom_node_root: Path | None = None,
    registry_environment_root: Path | None = None,
) -> WorkflowActivationLaunchScope:
    """Revalidate a stored snapshot and mark it stale, never active, on drift."""

    activation_row = _activation_row(session, activation)
    if activation_row.state == "disabled":
        raise WorkflowActivationError(
            "workflow_activation_disabled", "Disabled workflow activations cannot be launched"
        )
    try:
        revision = _revision_row(session, activation_row.workflow_revision_id)
        contract, slots = _hydrate_contract(session, revision)
        if activation_row.dependency_contract_sha256 != revision.dependency_contract_sha256:
            raise WorkflowActivationError(
                "workflow_contract_drift", "Workflow activation contract identity has changed"
            )
        selections = _stored_selections(session, activation_row)
        resolution = _resolve(
            session,
            contract,
            selections,
            runtime_materializer=runtime_materializer,
        )
        if (
            not resolution.complete
            or resolution.binding_sha256 is None
            or resolution.binding_sha256 != activation_row.binding_sha256
        ):
            raise WorkflowActivationError(
                "dependency_binding_drift",
                "Workflow activation dependencies no longer match their recorded identities",
                issues=resolution.issues,
            )
        _assert_persisted_snapshot(activation_row, slots, resolution.bindings, selections)
        resources = _launch_resources(
            session,
            resolution.bindings,
            selections,
            custom_node_root=custom_node_root,
            registry_environment_root=registry_environment_root,
        )
    except WorkflowActivationError as exc:
        _mark_stale(session, activation_row, exc.code, str(exc))
        raise

    launch_sha256 = _launch_sha256(activation_row.binding_sha256, resources)
    with session.begin_nested():
        activation_row.state = "ready"
        activation_row.last_validated_at = utcnow()
        activation_row.invalidated_at = None
        activation_row.invalidation_code = None
        activation_row.invalidation_reason = None
        activation_row.details_json = {"launch_sha256": launch_sha256}
        session.flush()
    return _launch_scope(activation_row, resources, launch_sha256)


def _revision_row(session: Session, revision: WorkflowRevision | str) -> WorkflowRevision:
    revision_id = revision if isinstance(revision, str) else revision.id
    row = session.get(WorkflowRevision, revision_id)
    if row is None:
        raise WorkflowActivationError(
            "workflow_revision_unavailable", "Workflow revision is unavailable"
        )
    return row


def _activation_row(session: Session, activation: WorkflowActivation | str) -> WorkflowActivation:
    activation_id = activation if isinstance(activation, str) else activation.id
    row = session.get(WorkflowActivation, activation_id)
    if row is None:
        raise WorkflowActivationError(
            "workflow_activation_unavailable", "Workflow activation is unavailable"
        )
    return row


def _hydrate_contract(
    session: Session, revision: WorkflowRevision
) -> tuple[WorkflowDependencyContract, dict[str, WorkflowDependencySlot]]:
    if revision.dependency_contract_sha256 is None:
        raise WorkflowActivationError(
            "legacy_workflow_revision_unsupported",
            "Legacy workflow revisions do not have an activation dependency contract",
        )
    rows = list(
        session.scalars(
            select(WorkflowDependencySlot)
            .where(WorkflowDependencySlot.workflow_revision_id == revision.id)
            .order_by(WorkflowDependencySlot.ordinal)
        ).all()
    )
    if [row.ordinal for row in rows] != list(range(len(rows))):
        raise WorkflowActivationError(
            "invalid_workflow_dependency_snapshot",
            "Workflow dependency slot ordering is not canonical",
        )
    payload = {
        "version": 1,
        "slots": [
            {
                "name": row.name,
                "resource_kind": row.resource_kind,
                "required": row.required,
                "satisfaction": row.satisfaction,
                "requirements": row.requirements_json,
            }
            for row in rows
        ],
    }
    try:
        contract = parse_workflow_dependency_contract(payload)
    except WorkflowDependencyError as exc:
        raise WorkflowActivationError(exc.code, str(exc)) from exc
    by_name = {slot.name: slot for slot in contract.slots}
    for row in rows:
        slot = by_name[row.name]
        if row.contract_sha256 != workflow_dependency_slot_sha256(slot):
            raise WorkflowActivationError(
                "workflow_contract_drift",
                f"Workflow dependency slot {row.name} no longer matches its recorded identity",
            )
    digest = workflow_dependency_contract_sha256(contract)
    if revision.dependency_contract_sha256 != digest:
        raise WorkflowActivationError(
            "workflow_contract_drift",
            "Workflow dependency contract no longer matches its recorded identity",
        )
    return contract, {row.name: row for row in rows}


def _resolve(
    session: Session,
    contract: WorkflowDependencyContract,
    selections: Sequence[WorkflowBindingSelection],
    *,
    runtime_materializer: WorkflowRuntimeMaterializer | None,
) -> WorkflowActivationResolution:
    def materialize(
        requirement: WorkflowDependencyRequirement,
        selection: WorkflowBindingSelection,
    ) -> MaterializedWorkflowDependency | None:
        if selection.local_kind == "runtime":
            if runtime_materializer is None:
                raise WorkflowBindingError(
                    "dependency_unavailable", "Runtime dependency is unavailable"
                )
            return runtime_materializer(requirement, selection)
        model: object | None
        if selection.local_kind == "model_install":
            model = session.get(ModelInstall, selection.local_id)
            return (
                materialize_model_install(session, model)
                if isinstance(model, ModelInstall)
                else None
            )
        if selection.local_kind == "model_profile":
            model = session.get(ModelProfile, selection.local_id)
            return (
                materialize_model_profile(session, model)
                if isinstance(model, ModelProfile)
                else None
            )
        if selection.local_kind == "model_asset":
            model = session.get(ModelAssetInstall, selection.local_id)
            return materialize_model_asset(model) if isinstance(model, ModelAssetInstall) else None
        if selection.local_kind == "registry_package":
            model = session.get(ComfyRegistryInstall, selection.local_id)
            return (
                materialize_registry_package(model)
                if isinstance(model, ComfyRegistryInstall)
                else None
            )
        if selection.local_kind == "custom_node":
            model = session.get(CustomNodeInstall, selection.local_id)
            if not isinstance(model, CustomNodeInstall):
                return None
            node_types = _custom_node_types(model.security_json)
            base = materialize_custom_node(model)
            return MaterializedWorkflowDependency(
                "custom_node", {**base.identity, "node_types": list(node_types)}
            )
        raise WorkflowBindingError(
            "dependency_locator_kind_mismatch", "Workflow dependency locator kind is invalid"
        )

    try:
        return resolve_workflow_activation(contract, selections, materialize)
    except WorkflowBindingError as exc:
        raise WorkflowActivationError(exc.code, str(exc)) from exc


def _custom_node_types(value: object) -> tuple[str, ...]:
    if not isinstance(value, dict):
        raise WorkflowBindingError(
            "dependency_unavailable", "Custom node has no verified node type evidence"
        )
    raw = value.get("node_types")
    if not isinstance(raw, list) or not raw or len(raw) > MAX_CUSTOM_NODE_TYPES:
        raise WorkflowBindingError(
            "dependency_unavailable", "Custom node has no verified node type evidence"
        )
    result: list[str] = []
    folded: set[str] = set()
    for item in raw:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > MAX_NODE_TYPE_LENGTH
            or any(character < " " or ord(character) == 127 for character in item)
            or item.casefold() in folded
        ):
            raise WorkflowBindingError(
                "dependency_unavailable", "Custom node has invalid node type evidence"
            )
        folded.add(item.casefold())
        result.append(item)
    return tuple(sorted(result, key=lambda item: (item.casefold(), item)))


def _launch_resources(
    session: Session,
    bindings: Sequence[ResolvedWorkflowBinding],
    selections: Sequence[WorkflowBindingSelection],
    *,
    custom_node_root: Path | None,
    registry_environment_root: Path | None,
) -> _LaunchResources:
    selected = {(item.slot_name, item.requirement_key): item for item in selections}
    model_ids: set[str] = set()
    asset_ids: set[str] = set()
    custom_ids: set[str] = set()
    registry_ids: set[str] = set()
    runtime_keys: set[str] = set()
    runtime_bindings: dict[str, WorkflowRuntimeLaunchBinding] = {}
    for binding in bindings:
        selection = selected[(binding.slot_name, binding.requirement_key)]
        if selection.local_kind == "model_install":
            model_ids.add(selection.local_id)
        elif selection.local_kind == "model_profile":
            profile = session.get(ModelProfile, selection.local_id)
            if profile is None or profile.model_install_id is None:
                raise WorkflowActivationError(
                    "dependency_unavailable", "Selected model profile is no longer bound"
                )
            model_ids.add(profile.model_install_id)
        elif selection.local_kind == "model_asset":
            asset_ids.add(selection.local_id)
        elif selection.local_kind == "custom_node":
            custom_ids.add(selection.local_id)
        elif selection.local_kind == "registry_package":
            registry_ids.add(selection.local_id)
        elif selection.local_kind == "runtime":
            runtime_keys.add(selection.local_id)
            candidate = WorkflowRuntimeLaunchBinding(
                selection.local_id,
                binding.resource_identity_sha256,
                _canonical_json(binding.identity),
            )
            previous = runtime_bindings.get(selection.local_id)
            if previous is not None and previous != candidate:
                raise WorkflowActivationError(
                    "runtime_identity_collision",
                    "One selected runtime key resolves to conflicting runtime identities",
                )
            runtime_bindings[selection.local_id] = candidate

    models = tuple(_model_launch_binding(session, item) for item in sorted(model_ids))
    assets = tuple(_asset_launch_binding(session, item) for item in sorted(asset_ids))
    custom_nodes = tuple(
        _custom_node_launch_binding(session, item, custom_node_root) for item in sorted(custom_ids)
    )
    registry_packages = tuple(
        _registry_launch_binding(
            session,
            item,
            custom_node_root=custom_node_root,
            environment_root=registry_environment_root,
        )
        for item in sorted(registry_ids)
    )
    _reject_loader_collisions(models, assets)
    _reject_node_type_collisions(custom_nodes, registry_packages)
    runtimes = tuple(
        sorted(
            runtime_bindings.values(),
            key=lambda item: (
                item.runtime_key,
                item.resource_identity_sha256,
                item.identity_json,
            ),
        )
    )
    return _LaunchResources(
        tuple(sorted(model_ids)),
        tuple(sorted(asset_ids)),
        tuple(sorted(custom_ids)),
        tuple(sorted(registry_ids)),
        tuple(sorted(runtime_keys)),
        models,
        assets,
        custom_nodes,
        registry_packages,
        runtimes,
    )


def _model_launch_binding(session: Session, install_id: str) -> WorkflowModelLaunchBinding:
    install = session.get(ModelInstall, install_id)
    if install is None:
        raise WorkflowActivationError("dependency_unavailable", "Selected model is unavailable")
    try:
        identity = materialize_model_install(session, install).identity
    except WorkflowBindingError as exc:
        raise WorkflowActivationError(exc.code, str(exc)) from exc
    raw_components = identity.get("components")
    if not isinstance(raw_components, list):
        raise WorkflowActivationError(
            "invalid_dependency_identity", "Selected model component identity is invalid"
        )
    components = tuple(
        sorted(
            (
                WorkflowModelComponentLaunchBinding(
                    _identity_text(item, "target_folder"),
                    _identity_text(item, "runtime_reference"),
                    _identity_digest(item, "sha256"),
                )
                for item in raw_components
                if isinstance(item, dict)
            ),
            key=lambda item: (
                item.target_folder,
                item.runtime_reference,
                item.sha256,
            ),
        )
    )
    if len(components) != len(raw_components):
        raise WorkflowActivationError(
            "invalid_dependency_identity", "Selected model component identity is invalid"
        )
    base_path = _directory(Path(install.local_path), "Selected model directory is unavailable")
    rows = list(
        session.scalars(
            select(ModelComponentManifest).where(
                ModelComponentManifest.model_install_id == install.id,
                ModelComponentManifest.required.is_(True),
            )
        ).all()
    )
    if len(rows) != len(components):
        raise WorkflowActivationError(
            "dependency_content_drift", "Selected model component closure has changed"
        )
    for row in rows:
        if not isinstance(row.sha256, str) or not _DIGEST.fullmatch(row.sha256.lower()):
            raise WorkflowActivationError(
                "invalid_dependency_identity", "Selected model component hash is invalid"
            )
        path = _contained_file(base_path, row.relative_path)
        _verify_file_digest(path, row.sha256.lower(), "Selected model component bytes changed")
    comfy_paths = _comfy_paths(install)
    return WorkflowModelLaunchBinding(install.id, base_path, comfy_paths, components)


def _asset_launch_binding(session: Session, asset_id: str) -> WorkflowAssetLaunchBinding:
    asset = session.get(ModelAssetInstall, asset_id)
    if asset is None:
        raise WorkflowActivationError(
            "dependency_unavailable", "Selected model asset is unavailable"
        )
    try:
        identity = materialize_model_asset(asset).identity
    except WorkflowBindingError as exc:
        raise WorkflowActivationError(exc.code, str(exc)) from exc
    root = _directory(Path(asset.local_path), "Selected model asset directory is unavailable")
    runtime_reference = _identity_text(identity, "runtime_reference")
    digest = _identity_digest(identity, "sha256")
    loader_folder = COMFY_AUXILIARY_FOLDERS.get(asset.kind)
    if loader_folder is None:
        raise WorkflowActivationError(
            "invalid_dependency_identity", "Selected model asset loader is invalid"
        )
    path = _contained_file(root, runtime_reference)
    _verify_file_digest(path, digest, "Selected model asset bytes changed")
    return WorkflowAssetLaunchBinding(
        asset.id,
        root,
        loader_folder,
        runtime_reference,
        digest,
    )


def _custom_node_launch_binding(
    session: Session,
    install_id: str,
    custom_node_root: Path | None,
) -> WorkflowCustomNodeLaunchBinding:
    install = session.get(CustomNodeInstall, install_id)
    if install is None:
        raise WorkflowActivationError(
            "dependency_unavailable", "Selected custom node is unavailable"
        )
    try:
        identity = materialize_custom_node(install).identity
        node_types = _custom_node_types(install.security_json)
    except WorkflowBindingError as exc:
        raise WorkflowActivationError(exc.code, str(exc)) from exc
    path = _managed_directory(install.installed_path, custom_node_root, "custom node")
    return WorkflowCustomNodeLaunchBinding(
        install.id,
        path,
        _identity_text(identity, "source_url"),
        _identity_text(identity, "revision"),
        _identity_text(identity, "tree_hash"),
        node_types,
    )


def _registry_launch_binding(
    session: Session,
    install_id: str,
    *,
    custom_node_root: Path | None,
    environment_root: Path | None,
) -> WorkflowRegistryLaunchBinding:
    install = session.get(ComfyRegistryInstall, install_id)
    if install is None:
        raise WorkflowActivationError(
            "dependency_unavailable", "Selected Registry package is unavailable"
        )
    try:
        identity = materialize_registry_package(install).identity
    except WorkflowBindingError as exc:
        raise WorkflowActivationError(exc.code, str(exc)) from exc
    installed_path = _managed_directory(install.installed_path, custom_node_root, "Registry node")
    if install.wheel_environment_path is None:
        raise WorkflowActivationError(
            "dependency_unavailable", "Selected Registry package environment is unavailable"
        )
    environment = _managed_directory(
        install.wheel_environment_path,
        environment_root,
        "Registry wheel environment",
    )
    site_packages = _directory(
        environment / "site-packages", "Selected Registry site-packages is unavailable"
    )
    raw_nodes = identity.get("node_types")
    if not isinstance(raw_nodes, list):
        raise WorkflowActivationError(
            "invalid_dependency_identity", "Selected Registry node identities are invalid"
        )
    node_types = tuple(_node_type(item) for item in raw_nodes)
    _casefold_unique(node_types, "Selected Registry node identities collide")
    return WorkflowRegistryLaunchBinding(
        install.id,
        installed_path,
        site_packages,
        _identity_text(identity, "package_id"),
        _identity_text(identity, "package_version"),
        _identity_digest(identity, "archive_sha256"),
        _identity_digest(identity, "manifest_sha256"),
        _identity_digest(identity, "wheel_closure_sha256"),
        _identity_digest(identity, "wheel_environment_sha256"),
        tuple(sorted(node_types, key=lambda item: (item.casefold(), item))),
    )


def _comfy_paths(install: ModelInstall) -> tuple[tuple[str, str], ...]:
    if install.engine != "comfyui":
        return ()
    raw = (
        install.manifest_json.get("comfy_paths")
        if isinstance(install.manifest_json, dict)
        else None
    )
    if not isinstance(raw, dict) or not raw:
        raise WorkflowActivationError(
            "dependency_unavailable", "Selected Comfy model has no runtime path mapping"
        )
    result: list[tuple[str, str]] = []
    for key, value in raw.items():
        if not isinstance(key, str) or key not in _COMFY_MODEL_FOLDERS:
            raise WorkflowActivationError(
                "invalid_dependency_identity", "Selected Comfy model path mapping is invalid"
            )
        result.append((key, _relative_path(value, allow_root=True)))
    return tuple(sorted(result))


def _reject_loader_collisions(
    models: Sequence[WorkflowModelLaunchBinding],
    assets: Sequence[WorkflowAssetLaunchBinding],
) -> None:
    seen: set[tuple[str, str]] = set()
    for model in models:
        local: set[tuple[str, str]] = set()
        for component in model.components:
            key = (component.target_folder.casefold(), component.runtime_reference.casefold())
            if key in local or key in seen:
                raise WorkflowActivationError(
                    "runtime_loader_collision",
                    "Selected model resources have colliding runtime loader names",
                )
            local.add(key)
            seen.add(key)
    for asset in assets:
        key = (asset.loader_folder.casefold(), asset.runtime_reference.casefold())
        if key in seen:
            raise WorkflowActivationError(
                "runtime_loader_collision",
                "Selected model resources have colliding runtime loader names",
            )
        seen.add(key)


def _reject_node_type_collisions(
    custom_nodes: Sequence[WorkflowCustomNodeLaunchBinding],
    registry_packages: Sequence[WorkflowRegistryLaunchBinding],
) -> None:
    folders: set[str] = set()
    for path in (
        *(item.installed_path for item in custom_nodes),
        *(item.installed_path for item in registry_packages),
    ):
        folder = path.name.casefold()
        if folder in folders:
            raise WorkflowActivationError(
                "extension_folder_collision",
                "Selected workflow extensions have colliding install folders",
            )
        folders.add(folder)

    environments: dict[str, tuple[str, str]] = {}
    for item in registry_packages:
        path_key = str(item.site_packages).casefold()
        identity = (item.wheel_closure_sha256, item.wheel_environment_sha256)
        previous = environments.get(path_key)
        if previous is not None and previous != identity:
            raise WorkflowActivationError(
                "registry_environment_collision",
                "Selected Registry packages conflict in one wheel environment",
            )
        environments[path_key] = identity

    seen: set[str] = set()
    providers = [item.node_types for item in custom_nodes]
    providers.extend(item.node_types for item in registry_packages)
    for node_types in providers:
        local: set[str] = set()
        for node_type in node_types:
            folded = node_type.casefold()
            if folded in local or folded in seen:
                raise WorkflowActivationError(
                    "node_type_collision",
                    "Selected workflow extensions provide colliding node types",
                )
            local.add(folded)
            seen.add(folded)


def _stored_selections(
    session: Session, activation: WorkflowActivation
) -> tuple[WorkflowBindingSelection, ...]:
    rows = list(
        session.scalars(
            select(WorkflowDependencyBinding)
            .where(WorkflowDependencyBinding.workflow_activation_id == activation.id)
            .order_by(
                WorkflowDependencyBinding.workflow_dependency_slot_id,
                WorkflowDependencyBinding.requirement_key,
            )
        ).all()
    )
    slots = {
        row.id: row.name
        for row in session.scalars(
            select(WorkflowDependencySlot).where(
                WorkflowDependencySlot.workflow_revision_id == activation.workflow_revision_id
            )
        ).all()
    }
    selections: list[WorkflowBindingSelection] = []
    for row in rows:
        slot_name = slots.get(row.workflow_dependency_slot_id)
        if slot_name is None:
            raise WorkflowActivationError(
                "invalid_activation_snapshot", "Workflow activation refers to an unknown slot"
            )
        local_kind, local_id = _stored_locator(row)
        selections.append(
            WorkflowBindingSelection(
                slot_name,
                row.requirement_key,
                local_kind,
                local_id,
                row.resource_identity_sha256,
                row.mount_json,
            )
        )
    return tuple(sorted(selections, key=lambda item: (item.slot_name, item.requirement_key)))


def _persist_bindings(
    activation: WorkflowActivation,
    slots: dict[str, WorkflowDependencySlot],
    bindings: Sequence[ResolvedWorkflowBinding],
    selections: Sequence[WorkflowBindingSelection],
    session: Session,
) -> None:
    selected = {(item.slot_name, item.requirement_key): item for item in selections}
    for binding in bindings:
        selection = selected[(binding.slot_name, binding.requirement_key)]
        locator = _locator_values(selection.local_kind, selection.local_id)
        session.add(
            WorkflowDependencyBinding(
                id=new_id("wfbind"),
                workflow_revision_id=activation.workflow_revision_id,
                workflow_activation_id=activation.id,
                workflow_dependency_slot_id=slots[binding.slot_name].id,
                requirement_key=binding.requirement_key,
                mount_json=binding.mount,
                resource_identity_json=binding.identity,
                resource_identity_sha256=binding.resource_identity_sha256,
                **locator,
            )
        )
    session.flush()


def _assert_persisted_snapshot(
    activation: WorkflowActivation,
    slots: dict[str, WorkflowDependencySlot],
    bindings: Sequence[ResolvedWorkflowBinding],
    selections: Sequence[WorkflowBindingSelection],
) -> None:
    slot_names = {row.id: name for name, row in slots.items()}
    expected_selection = {(item.slot_name, item.requirement_key): item for item in selections}
    expected_binding = {(item.slot_name, item.requirement_key): item for item in bindings}
    actual: dict[tuple[str, str], WorkflowDependencyBinding] = {}
    for row in activation.bindings:
        slot_name = slot_names.get(row.workflow_dependency_slot_id)
        if slot_name is None:
            raise WorkflowActivationError(
                "invalid_activation_snapshot", "Stored activation contains an unknown slot"
            )
        key = (slot_name, row.requirement_key)
        if key in actual:
            raise WorkflowActivationError(
                "invalid_activation_snapshot", "Stored activation contains duplicate bindings"
            )
        actual[key] = row
    if set(actual) != set(expected_binding):
        raise WorkflowActivationError(
            "invalid_activation_snapshot", "Stored activation binding set has changed"
        )
    for key, resolved in expected_binding.items():
        row = actual[key]
        selection = expected_selection[key]
        if (
            _stored_locator(row) != (selection.local_kind, selection.local_id)
            or row.resource_identity_sha256 != resolved.resource_identity_sha256
            or row.resource_identity_json != resolved.identity
            or row.mount_json != resolved.mount
        ):
            raise WorkflowActivationError(
                "invalid_activation_snapshot", "Stored activation binding data has changed"
            )


def _locator_values(local_kind: str, local_id: str) -> dict[str, str | None]:
    values: dict[str, str | None] = {
        "model_profile_id": None,
        "model_install_id": None,
        "model_asset_install_id": None,
        "custom_node_install_id": None,
        "comfy_registry_install_id": None,
        "runtime_key": None,
    }
    field = {
        "model_profile": "model_profile_id",
        "model_install": "model_install_id",
        "model_asset": "model_asset_install_id",
        "custom_node": "custom_node_install_id",
        "registry_package": "comfy_registry_install_id",
        "runtime": "runtime_key",
    }.get(local_kind)
    if field is None:
        raise WorkflowActivationError(
            "dependency_locator_kind_mismatch", "Workflow dependency locator kind is invalid"
        )
    values[field] = local_id
    return values


def _stored_locator(binding: WorkflowDependencyBinding) -> tuple[str, str]:
    values = (
        ("model_profile", binding.model_profile_id),
        ("model_install", binding.model_install_id),
        ("model_asset", binding.model_asset_install_id),
        ("custom_node", binding.custom_node_install_id),
        ("registry_package", binding.comfy_registry_install_id),
        ("runtime", binding.runtime_key),
    )
    present = [(kind, value) for kind, value in values if value is not None]
    if len(present) != 1:
        raise WorkflowActivationError(
            "invalid_activation_snapshot", "Stored activation binding locator is invalid"
        )
    kind, value = present[0]
    return kind, value


def _mark_stale(
    session: Session,
    activation: WorkflowActivation,
    code: str,
    reason: str,
) -> None:
    safe_code = code[:80] if code else "activation_invalid"
    safe_reason = reason[:2_000] if reason else "Workflow activation is stale"
    with session.begin_nested():
        activation.is_active = False
        activation.state = "stale"
        activation.invalidated_at = utcnow()
        activation.invalidation_code = safe_code
        activation.invalidation_reason = safe_reason
        session.flush()


def _launch_scope(
    activation: WorkflowActivation,
    resources: _LaunchResources,
    launch_sha256: str,
) -> WorkflowActivationLaunchScope:
    return WorkflowActivationLaunchScope(
        activation.id,
        activation.workflow_revision_id,
        activation.binding_sha256,
        launch_sha256,
        resources.model_install_ids,
        resources.model_asset_install_ids,
        resources.custom_node_install_ids,
        resources.registry_install_ids,
        resources.runtime_keys,
        resources.models,
        resources.assets,
        resources.custom_nodes,
        resources.registry_packages,
        resources.runtimes,
    )


def _launch_sha256(binding_sha256: str, resources: _LaunchResources) -> str:
    payload = {
        "version": 1,
        "binding_sha256": binding_sha256,
        "models": [
            {
                "id": item.model_install_id,
                "base_path": str(item.base_path),
                "comfy_paths": list(item.comfy_paths),
                "components": [
                    [component.target_folder, component.runtime_reference, component.sha256]
                    for component in item.components
                ],
            }
            for item in resources.models
        ],
        "assets": [
            [
                item.model_asset_install_id,
                str(item.base_path),
                item.loader_folder,
                item.runtime_reference,
                item.sha256,
            ]
            for item in resources.assets
        ],
        "custom_nodes": [
            [
                item.custom_node_install_id,
                str(item.installed_path),
                item.source_url,
                item.revision,
                item.tree_hash,
                list(item.node_types),
            ]
            for item in resources.custom_nodes
        ],
        "registry_packages": [
            [
                item.registry_install_id,
                str(item.installed_path),
                str(item.site_packages),
                item.package_id,
                item.package_version,
                item.archive_sha256,
                item.manifest_sha256,
                item.wheel_closure_sha256,
                item.wheel_environment_sha256,
                list(item.node_types),
            ]
            for item in resources.registry_packages
        ],
        "runtimes": [
            [item.runtime_key, item.resource_identity_sha256, item.identity_json]
            for item in resources.runtimes
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _directory(path: Path, message: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise WorkflowActivationError("dependency_content_drift", message) from exc
    if not resolved.is_dir():
        raise WorkflowActivationError("dependency_content_drift", message)
    return resolved


def _managed_directory(value: str, root: Path | None, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = _directory(path, f"Selected {label} directory is unavailable")
        if root is not None:
            managed_root = _directory(root, f"Managed {label} root is unavailable")
            if resolved.parent != managed_root:
                raise WorkflowActivationError(
                    "dependency_content_drift", f"Selected {label} escapes its managed root"
                )
        return resolved
    if root is None or path.name != value or value in {"", ".", ".."}:
        raise WorkflowActivationError(
            "dependency_unavailable", f"Selected {label} has no managed root"
        )
    managed_root = _directory(root, f"Managed {label} root is unavailable")
    resolved = _directory(managed_root / value, f"Selected {label} directory is unavailable")
    if resolved.parent != managed_root:
        raise WorkflowActivationError(
            "dependency_content_drift", f"Selected {label} escapes its managed root"
        )
    return resolved


def _contained_file(root: Path, relative: object) -> Path:
    canonical = _relative_path(relative, allow_root=False)
    try:
        candidate = root.joinpath(*PurePosixPath(canonical).parts).resolve(strict=True)
    except OSError as exc:
        raise WorkflowActivationError(
            "dependency_content_drift", "Selected dependency file is unavailable"
        ) from exc
    if root not in candidate.parents or not candidate.is_file():
        raise WorkflowActivationError(
            "dependency_content_drift", "Selected dependency file escapes its install"
        )
    return candidate


def _relative_path(value: object, *, allow_root: bool) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1_000
        or ":" in value
        or any(character < " " or ord(character) == 127 for character in value)
    ):
        raise WorkflowActivationError(
            "invalid_dependency_identity", "Selected dependency path is invalid"
        )
    path = PurePosixPath(value.replace("\\", "/"))
    canonical = path.as_posix()
    if path.is_absolute() or ".." in path.parts or (canonical == "." and not allow_root):
        raise WorkflowActivationError(
            "invalid_dependency_identity", "Selected dependency path is invalid"
        )
    return canonical


def _verify_file_digest(path: Path, expected: str, message: str) -> None:
    try:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise WorkflowActivationError("dependency_content_drift", message) from exc
    before_identity = (before.st_size, before.st_mtime_ns)
    after_identity = (after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or digest.hexdigest() != expected:
        raise WorkflowActivationError("dependency_content_drift", message)


def _identity_text(identity: dict[str, Any], key: str) -> str:
    value = identity.get(key)
    if not isinstance(value, str) or not value:
        raise WorkflowActivationError(
            "invalid_dependency_identity", "Selected dependency identity is invalid"
        )
    return value


def _identity_digest(identity: dict[str, Any], key: str) -> str:
    value = _identity_text(identity, key).lower()
    if not _DIGEST.fullmatch(value):
        raise WorkflowActivationError(
            "invalid_dependency_identity", "Selected dependency hash is invalid"
        )
    return value


def _node_type(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_NODE_TYPE_LENGTH
        or any(character < " " or ord(character) == 127 for character in value)
    ):
        raise WorkflowActivationError(
            "invalid_dependency_identity", "Selected extension node identity is invalid"
        )
    return value


def _casefold_unique(values: Sequence[str], message: str) -> None:
    folded = {value.casefold() for value in values}
    if len(folded) != len(values):
        raise WorkflowActivationError("node_type_collision", message)


def _resolver_version(value: str) -> str:
    if not isinstance(value, str) or not _RESOLVER_VERSION.fullmatch(value):
        raise WorkflowActivationError(
            "invalid_resolver_version", "Workflow activation resolver version is invalid"
        )
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
