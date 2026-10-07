"""Inspect declared extensions before saving a workflow installation preview."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from pydantic import Field
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .comfy_registry import MAX_REGISTRY_PACKAGES, ComfyRegistryClient
from .comfy_registry_closure_driver import ComfyRegistryWheelMetadataClient
from .comfy_registry_downloads import ComfyRegistryArchiveDownloader
from .comfy_registry_interpreter import probe_comfy_registry_runtime_target
from .comfy_registry_wheel_projects import ComfyRegistryWheelProjectClient
from .comfy_workflow_packages import analyze_comfyui_workflow_package
from .config import Settings
from .runtime_provisioning import RuntimeProvisioner
from .runtime_provisioning_plans import RuntimeProvisioningPlan
from .schemas import ApiModel
from .workflow_package_execution_plan import (
    WorkflowPackageExecutionPlan,
    plan_workflow_package_execution,
)
from .workflow_package_preparation import InterpreterProbe, PreparationContext
from .workflow_runtime_targets import preflight_workflow_runtime_target


class WorkflowPackageExtensionPreflight(ApiModel):
    plans: dict[str, WorkflowPackageExecutionPlan] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)


async def preflight_workflow_extensions(
    graph: dict[str, Any],
    settings: Settings,
    *,
    session_factory: Callable[[], Session],
    source_store: ArtifactStore,
    runtimes: RuntimeProvisioner | None = None,
    runtime_plan: RuntimeProvisioningPlan | None = None,
) -> WorkflowPackageExtensionPreflight:
    """Resolve exact source requirements without changing workers or installation state."""
    requirements = analyze_comfyui_workflow_package(graph).custom_packages
    result = WorkflowPackageExtensionPreflight()
    if not requirements:
        return result
    if len(requirements) > MAX_REGISTRY_PACKAGES:
        result.errors = {
            item.package_id: "workflow-package-limit-exceeded" for item in requirements
        }
        return result
    try:
        interpreter_probe: InterpreterProbe = probe_comfy_registry_runtime_target
        if runtimes is not None:
            if runtime_plan is None:
                result.errors = {
                    item.package_id: "workflow-runtime-plan-unavailable" for item in requirements
                }
                return result
            target = await preflight_workflow_runtime_target(runtimes, runtime_plan)
            python_executable = target.python_executable
            interpreter_probe = target.probe
        else:
            python_executable = PreparationContext.from_settings(settings).python_executable
    except ValueError as exc:
        result.errors = {
            item.package_id: getattr(exc, "code", "workflow-runtime-target-unavailable")
            for item in requirements
        }
        return result
    async with AsyncExitStack() as cleanup:
        registry = ComfyRegistryClient()
        cleanup.push_async_callback(registry.close)
        projects = ComfyRegistryWheelProjectClient()
        cleanup.push_async_callback(projects.close)
        metadata = ComfyRegistryWheelMetadataClient()
        cleanup.push_async_callback(metadata.close)
        archive = ComfyRegistryArchiveDownloader()
        cleanup.push_async_callback(archive.close)
        for requirement in requirements:
            try:
                result.plans[requirement.package_id] = await plan_workflow_package_execution(
                    requirement=requirement,
                    python_executable=python_executable,
                    interpreter_probe=interpreter_probe,
                    registry_client=registry,
                    project_client=projects,
                    metadata_client=metadata,
                    archive_downloader=archive,
                    source_session_factory=session_factory,
                    source_store=source_store,
                )
            except ValueError as exc:
                result.errors[requirement.package_id] = getattr(
                    exc, "code", "workflow-package-execution-plan-unavailable"
                )
    return result
