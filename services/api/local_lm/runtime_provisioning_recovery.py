"""Recover an approved runtime only from its verified published files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .filesystem_links import is_link_or_reparse
from .runtime_provisioning import RuntimeName, RuntimeProvisioner, RuntimeProvisioningError
from .runtime_provisioning_plans import RuntimeProvisioningPlan


@dataclass(frozen=True)
class RecoveredRuntimeSetup:
    installed: dict[str, Path]
    definition: dict[str, Any]
    asset: dict[str, Any]
    current_inputs_sha256: str


def _changed() -> RuntimeProvisioningError:
    return RuntimeProvisioningError("The approved runtime setup changed. Preview it again.")


def recover_approved_runtime(
    provisioner: RuntimeProvisioner, engine: RuntimeName, expected: RuntimeProvisioningPlan
) -> RecoveredRuntimeSetup | None:
    """Inspect files under the engine lock, off the event loop, before saving configuration."""
    if expected.engine != engine:
        raise _changed()
    if expected.operation != "install_managed":
        return None
    inputs = provisioner._provisioning_inputs(engine)
    definition = {**inputs["definition"], "runtime_assets": {inputs["platform"]: inputs["asset"]}}
    asset = inputs["asset"]
    if not isinstance(asset, dict):
        return None
    root = provisioner._installation_path(engine, definition)
    for path in (root, root.parent, provisioner.runtime_root):
        if is_link_or_reparse(path, missing="assume_regular", unreadable="assume_link"):
            raise _changed()
    marker_path = root / ".lm-atelier-runtime.json"
    if is_link_or_reparse(marker_path, missing="assume_regular", unreadable="assume_link"):
        raise _changed()
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise _changed() from None
    approved = marker.get("approved_setup") if isinstance(marker, dict) else None
    if not isinstance(approved, dict) or approved.get("plan") != asdict(expected):
        return None
    original_paths = approved.get("configured_paths")
    if (
        set(approved) != {"plan", "configured_paths"}
        or not isinstance(original_paths, list)
        or any(not isinstance(path, str) for path in original_paths)
        or provisioner._json_sha256({**inputs, "configured_paths": original_paths})
        != expected.inputs_sha256
        or provisioner._security_blocked(definition, asset)
    ):
        raise _changed()
    installed = provisioner._resolve_installed_paths(root, asset)
    target_paths = [installed["executable"].resolve()]
    if engine == "comfyui":
        target_paths.append(installed["directory"].resolve())
    current_paths = [Path(path).resolve() for path in inputs["configured_paths"]]
    # Before configuration was saved, the original inputs must still be
    # untouched, including any file that has appeared at a configured path.
    if current_paths != target_paths and (
        inputs["configured_paths"] != original_paths or provisioner.preflight(engine) != expected
    ):
        raise _changed()
    if not provisioner._managed_marker_matches(root, engine, definition, asset):
        raise _changed()
    if inputs != provisioner._provisioning_inputs(engine):
        raise _changed()
    return RecoveredRuntimeSetup(installed, definition, asset, provisioner._json_sha256(inputs))
