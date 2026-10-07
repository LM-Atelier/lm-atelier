"""Plan extension wheels for the exact runtime a workflow will install or reuse."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.markers import default_environment
from packaging.tags import compatible_tags, cpython_tags

from .comfy_registry_interpreter import probe_comfy_registry_runtime_target
from .comfy_registry_runtime import (
    ComfyRegistryRuntimeDistribution,
    canonical_comfy_registry_runtime_distributions,
)
from .comfy_registry_wheel_artifacts import comfy_registry_wheel_target_sha256
from .runtime_provisioning import RuntimeProvisioner, RuntimeProvisioningError
from .runtime_provisioning_plans import RuntimeProvisioningPlan

# This exact embeddable interpreter, with packaging 26.2, supplies the CPython
# 3.13 Windows AMD64 target. Other components need their own measured target.
_PYTHON_ARCHIVE_SHA256 = "90b4e5b9898b72d744650524bff92377c367f44bd5fbd09e3148656c080ad907"


class WorkflowRuntimeTargetError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__("The workflow runtime's extension target is unavailable or has changed.")
        self.code = code


@dataclass(frozen=True)
class WorkflowRuntimeTarget:
    python_executable: Path
    environment: tuple[tuple[str, str], ...]
    supported_tags: tuple[str, ...]
    distributions: tuple[ComfyRegistryRuntimeDistribution, ...]

    @classmethod
    def from_values(
        cls,
        python_executable: Path,
        environment: Mapping[str, str],
        tags: tuple[str, ...],
        distributions: tuple[ComfyRegistryRuntimeDistribution, ...],
    ) -> WorkflowRuntimeTarget:
        comfy_registry_wheel_target_sha256(environment, tags)
        return cls(
            python_executable,
            tuple(sorted(environment.items())),
            tags,
            canonical_comfy_registry_runtime_distributions(distributions),
        )

    async def probe(
        self, _executable: Path
    ) -> tuple[dict[str, str], tuple[str, ...], tuple[ComfyRegistryRuntimeDistribution, ...]]:
        """Supply an immutable preview to wheel planning, never to installation."""
        return dict(self.environment), self.supported_tags, self.distributions


def _planned_target(
    provisioner: RuntimeProvisioner, definition: dict[str, Any], asset: dict[str, Any]
) -> WorkflowRuntimeTarget:
    environment = {key: str(value) for key, value in default_environment().items()}
    if any(
        environment.get(key) != expected
        for key, expected in {
            "os_name": "nt",
            "sys_platform": "win32",
            "platform_system": "Windows",
            "platform_machine": "AMD64",
        }.items()
    ):
        raise WorkflowRuntimeTargetError("workflow-runtime-target-unsupported")
    overlays = asset.get("security_overlays", [])
    executable = PurePosixPath(str(asset.get("executable", "")))
    if (
        len(overlays) != 1
        or overlays[0].get("sha256") != _PYTHON_ARCHIVE_SHA256
        or overlays[0].get("compatibility_tag") != "cp313-win_amd64-embeddable"
        or PurePosixPath(str(overlays[0].get("target_directory", ""))) != executable.parent
        or executable.name != "python.exe"
        or asset.get("runtime_probe", {}).get("python") != "3.13.14"
        or overlays[0].get("rewrite_files")
        != {"python313._pth": "../ComfyUI\npython313.zip\n.\nimport site\n"}
    ):
        raise WorkflowRuntimeTargetError("workflow-runtime-target-unsupported")

    root = provisioner.manifest_path.parent
    reference = asset["dependency_review"]
    provisioner._validate_dependency_review(
        root,
        definition,
        reference["asset_key"],
        asset,
        count=asset["dependency_inventory_count"],
        inventory_hash=asset["dependency_inventory_sha256"],
    )
    relative = provisioner._safe_archive_path(reference["file"])
    path = root.joinpath(*relative.parts).resolve()
    provisioner._ensure_inside(root, path)
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != reference["sha256"]:
        raise WorkflowRuntimeTargetError("workflow-runtime-target-changed")
    review = json.loads(content)["assets"][reference["asset_key"]]
    distributions = canonical_comfy_registry_runtime_distributions(
        tuple(
            ComfyRegistryRuntimeDistribution(item["name"], item["version"])
            for item in review["distributions"]
        )
    )
    if dict((item.name, item.version) for item in distributions).get("packaging") != "26.2":
        raise WorkflowRuntimeTargetError("workflow-runtime-target-unsupported")
    environment.update(
        implementation_name="cpython",
        implementation_version="3.13.14",
        platform_python_implementation="CPython",
        python_full_version="3.13.14",
        python_version="3.13",
        extra="",
    )
    tags = tuple(
        str(tag)
        for tag in (
            *cpython_tags(python_version=(3, 13), abis=["cp313"], platforms=["win_amd64"]),
            *compatible_tags(python_version=(3, 13), interpreter="cp313", platforms=["win_amd64"]),
        )
    )
    destination = provisioner._installation_path("comfyui", definition).joinpath(*executable.parts)
    return WorkflowRuntimeTarget.from_values(destination, environment, tags, distributions)


async def preflight_workflow_runtime_plan(
    provisioner: RuntimeProvisioner,
) -> RuntimeProvisioningPlan | None:
    """Keep an unavailable runtime visible as an incomplete installation preview."""
    try:
        return await asyncio.to_thread(provisioner.preflight, "comfyui")
    except (OSError, RuntimeProvisioningError):
        return None


async def preflight_workflow_runtime_target(
    provisioner: RuntimeProvisioner, expected_plan: RuntimeProvisioningPlan
) -> WorkflowRuntimeTarget:
    """Bind planning metadata to the same runtime operation before and after inspection."""
    try:
        plan, definition, configured, asset = await asyncio.to_thread(
            provisioner._preflight_snapshot, "comfyui"
        )
        if plan != expected_plan:
            raise WorkflowRuntimeTargetError("workflow-runtime-target-changed")
        if configured is not None:
            executable = provisioner.settings.comfy_executable
            if executable is None:
                raise WorkflowRuntimeTargetError("workflow-runtime-target-changed")
            environment, tags, distributions = await probe_comfy_registry_runtime_target(
                Path(executable)
            )
            target = WorkflowRuntimeTarget.from_values(
                Path(executable), environment, tags, distributions
            )
        elif asset is not None and plan.operation == "install_managed":
            target = await asyncio.to_thread(_planned_target, provisioner, definition, asset)
        else:
            raise WorkflowRuntimeTargetError("workflow-runtime-target-unavailable")
        if await asyncio.to_thread(provisioner.preflight, "comfyui") != expected_plan:
            raise WorkflowRuntimeTargetError("workflow-runtime-target-changed")
        return target
    except (OSError, RuntimeProvisioningError) as exc:
        raise WorkflowRuntimeTargetError("workflow-runtime-target-unavailable") from exc
