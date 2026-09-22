"""Bind a temporary media launch to an accepted source and its exact dependencies."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from sqlalchemy.orm import Session

from .comfy_registry_activation_batches import _row
from .comfy_registry_installs import scoped_comfy_registry_launch_contract
from .comfy_registry_paths import registry_wheel_environment_root
from .comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from .revision_dependency_contract import declared_dependency_contract
from .workflow_activation_preparation import prepare_workflow_dependency_choices
from .workflow_activations import (
    WorkflowActivationError,
    WorkflowActivationLaunchScope,
    WorkflowRuntimeLaunchBinding,
    WorkflowRuntimeMaterializer,
    WorkflowSourceLaunchScope,
    _asset_launch_binding,
    _canonical_json,
    _launch_resources,
    _launch_sha256,
    _LaunchResources,
    _model_launch_binding,
    _registry_launch_binding,
    _reject_loader_collisions,
    _reject_node_type_collisions,
    resolve_workflow_dependencies,
)
from .workflow_bindings import WorkflowBindingSelection
from .workflow_completion_jobs import workflow_completion_job
from .workflow_dependencies import (
    WorkflowDependencyContract,
    WorkflowDependencyRequirement,
    WorkflowDependencySlotContract,
    workflow_dependency_contract_sha256,
)
from .workflow_offer_completion import WorkflowOfferCompletionError, _accepted_results
from .workflow_package_acceptance import accepted_workflow_package_jobs
from .workflow_package_preparation import PreparationContext
from .workflow_source_extensions import SessionFactory, _accepted


def _runtime(
    session: Session, materializer: WorkflowRuntimeMaterializer
) -> WorkflowRuntimeLaunchBinding:
    contract = WorkflowDependencyContract(
        1,
        (
            WorkflowDependencySlotContract(
                "runtime",
                "runtime",
                True,
                "all_of",
                (WorkflowDependencyRequirement("comfyui", {"engine": "comfyui"}),),
            ),
        ),
    )
    resolution = resolve_workflow_dependencies(
        session,
        contract,
        [WorkflowBindingSelection("runtime", "comfyui", "runtime", "comfyui")],
        runtime_materializer=materializer,
    )
    if not resolution.complete or len(resolution.bindings) != 1:
        raise WorkflowOfferCompletionError("workflow-runtime-unavailable")
    binding = resolution.bindings[0]
    return WorkflowRuntimeLaunchBinding(
        "comfyui", binding.resource_identity_sha256, _canonical_json(binding.identity)
    )


def prepare_workflow_source_launch_scope(
    session_factory: SessionFactory,
    offer_id: str,
    *,
    context: PreparationContext,
    runtime_materializer: WorkflowRuntimeMaterializer,
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None,
) -> WorkflowSourceLaunchScope:
    """Read a consistent accepted snapshot; the launcher revalidates it before execution."""
    environment_root = registry_wheel_environment_root(context.state_root)
    with session_factory() as session:
        session.connection().exec_driver_sql("BEGIN")
        offer, saved, packages = _accepted(session, offer_id)
        completion = workflow_completion_job(session, offer)
        if completion.status not in {"queued", "running"}:
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        payload, _saved, _jobs = accepted_workflow_package_jobs(session, offer)
        accepted_downloads = _accepted_results(session, offer)
        contract = declared_dependency_contract(payload.dependencies)
        if accepted_downloads is None:
            raise WorkflowOfferCompletionError("workflow-downloads-pending")
        if (
            contract is None
            or workflow_dependency_contract_sha256(contract) != saved.dependency_contract_sha256
        ):
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        accepted = {resource for result in accepted_downloads for resource in result}
        package_facts = []
        registry_ids = set()
        for package in packages:
            if package.preparation is None:
                raise WorkflowOfferCompletionError("workflow-extensions-pending")
            row = _row(session, package.preparation, package.plan)
            if not row.trusted or not row.active:
                raise WorkflowOfferCompletionError("workflow-extension-review-required")
            registry_ids.add(row.id)
            accepted.add(("registry_package", row.id))
            package_facts.append(
                {
                    "package_id": package.link.package_id,
                    "execution_plan_sha256": package.plan.plan_sha256,
                    "preparation": asdict(package.preparation),
                }
            )
        choices = prepare_workflow_dependency_choices(
            session,
            contract,
            runtime_materializer=runtime_materializer,
            accepted_resources=frozenset(accepted),
        )
        if choices.state != "prepared" or choices.selections is None:
            raise WorkflowOfferCompletionError("workflow-dependencies-need-selection")
        selections = [choice.binding() for choice in choices.selections]
        resolution = resolve_workflow_dependencies(
            session,
            contract,
            selections,
            runtime_materializer=runtime_materializer,
        )
        if not resolution.complete or resolution.binding_sha256 is None:
            raise WorkflowOfferCompletionError("workflow-dependencies-need-selection")
        declared = _launch_resources(
            session,
            resolution.bindings,
            selections,
            custom_node_root=context.custom_node_root,
            registry_environment_root=environment_root,
        )
        model_ids = set(declared.model_install_ids) | {
            value for kind, value in accepted if kind == "model_install"
        }
        asset_ids = set(declared.model_asset_install_ids) | {
            value for kind, value in accepted if kind == "model_asset"
        }
        registry_ids.update(declared.registry_install_ids)
        models = tuple(
            _model_launch_binding(session, identifier) for identifier in sorted(model_ids)
        )
        assets = tuple(
            _asset_launch_binding(session, identifier) for identifier in sorted(asset_ids)
        )
        registry = tuple(
            _registry_launch_binding(
                session,
                identifier,
                custom_node_root=context.custom_node_root,
                environment_root=environment_root,
            )
            for identifier in sorted(registry_ids)
        )
        scoped_comfy_registry_launch_contract(
            session,
            registry,
            custom_node_root=context.custom_node_root,
            environment_root=environment_root,
            reviewed_inputs=reviewed_inputs,
        )
        _reject_loader_collisions(models, assets)
        _reject_node_type_collisions(declared.custom_nodes, registry)
        runtime = _runtime(session, runtime_materializer)
        if any(item != runtime for item in declared.runtimes):
            raise WorkflowOfferCompletionError("workflow-runtime-changed")
        resources = _LaunchResources(
            tuple(sorted(model_ids)),
            tuple(sorted(asset_ids)),
            declared.custom_node_install_ids,
            tuple(sorted(registry_ids)),
            (runtime.runtime_key,),
            models,
            assets,
            declared.custom_nodes,
            registry,
            (runtime,),
        )
        binding_sha256 = hashlib.sha256(
            _canonical_json(
                {
                    "version": 1,
                    "offer_id": offer_id,
                    "completion_job_id": completion.id,
                    "completion_attempt": completion.attempt,
                    "plan_sha256": saved.plan_sha256,
                    "declared_binding_sha256": resolution.binding_sha256,
                    "download_results": sorted(sorted(result) for result in accepted_downloads),
                    "packages": package_facts,
                }
            ).encode("utf-8")
        ).hexdigest()
        return WorkflowSourceLaunchScope(
            offer_id,
            saved.plan_sha256,
            binding_sha256,
            _launch_sha256(binding_sha256, resources),
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


def require_workflow_source_resource_match(
    source: WorkflowSourceLaunchScope, installed: WorkflowActivationLaunchScope
) -> None:
    """The executable must use exactly the resources its temporary worker validated.

    Source approval and executable activation have different binding identities.
    Their launch resources must nevertheless agree, including file identities,
    mounts, extension closures and runtime contracts.
    """
    if not (
        source.model_install_ids == installed.model_install_ids
        and source.model_asset_install_ids == installed.model_asset_install_ids
        and source.custom_node_install_ids == installed.custom_node_install_ids
        and source.registry_install_ids == installed.registry_install_ids
        and source.runtime_keys == installed.runtime_keys
        and source.models == installed.models
        and source.assets == installed.assets
        and source.custom_nodes == installed.custom_nodes
        and source.registry_packages == installed.registry_packages
        and source.runtimes == installed.runtimes
    ):
        raise WorkflowOfferCompletionError("workflow-install-offer-changed")


def revalidate_workflow_source_launch_scope(
    session_factory: SessionFactory,
    expected: WorkflowSourceLaunchScope,
    *,
    context: PreparationContext,
    runtime_materializer: WorkflowRuntimeMaterializer,
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None,
) -> None:
    """Refuse any change to the accepted launch, including its physical dependency files."""
    current = prepare_workflow_source_launch_scope(
        session_factory,
        expected.offer_id,
        context=context,
        runtime_materializer=runtime_materializer,
        reviewed_inputs=reviewed_inputs,
    )
    if current != expected:
        raise WorkflowActivationError(
            "workflow_source_launch_changed", "The accepted workflow launch changed before startup."
        )
