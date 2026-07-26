from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapters.contracts import ADAPTER_CONTRACT_VERSION
from .config import Settings
from .hardware import hardware_capability_class
from .model_planner import (
    ACTIVATION_PROBE_VERSION,
    LAUNCH_CONTRACT_VERSION,
    media_workflow_contract_version,
)
from .models import ModelCapabilityEvidence, ModelInstall

if TYPE_CHECKING:
    from .runtime_provisioning import RuntimeProvisioner


def current_capability_evidence(
    session: Session,
    install: ModelInstall,
    settings: Settings,
    runtimes: RuntimeProvisioner | None,
) -> ModelCapabilityEvidence | None:
    """Return only evidence that still matches every activation contract."""

    expected_hashes = {
        str(path): str(digest)
        for path, digest in (install.manifest_json.get("expected_sha256") or {}).items()
        if isinstance(path, str) and isinstance(digest, str)
    }
    template_sha256 = install.manifest_json.get("workflow_template_sha256")
    expected_workflow = (
        media_workflow_contract_version(template_sha256)
        if isinstance(template_sha256, str)
        else None
    )
    runtime_name = (
        cast(Literal["llama.cpp", "comfyui"], install.engine)
        if install.engine in {"llama.cpp", "comfyui"}
        else None
    )
    current_runtime = runtimes.status(runtime_name) if runtimes and runtime_name else None
    for evidence in session.scalars(
        select(ModelCapabilityEvidence)
        .where(
            ModelCapabilityEvidence.model_install_id == install.id,
            ModelCapabilityEvidence.result == "ready",
        )
        .order_by(ModelCapabilityEvidence.probed_at.desc())
    ).all():
        runtime_release = evidence.details_json.get("runtime_release")
        if (
            evidence.adapter_contract_version != ADAPTER_CONTRACT_VERSION
            or evidence.launch_contract_version != LAUNCH_CONTRACT_VERSION
            or evidence.probe_version != ACTIVATION_PROBE_VERSION
            or evidence.hardware_class != hardware_capability_class(settings)
            or evidence.component_hashes_json != expected_hashes
            or evidence.workflow_contract_version != expected_workflow
            or (
                isinstance(runtime_release, str)
                and (
                    not current_runtime
                    or current_runtime.state != "ready"
                    or current_runtime.release != runtime_release
                )
            )
        ):
            continue
        return evidence
    return None


def evidence_input_modalities(evidence: ModelCapabilityEvidence | None) -> list[str]:
    if not evidence:
        return ["text"]
    raw = evidence.details_json.get("input_modalities")
    if not isinstance(raw, list):
        return ["text"]
    modalities = [value for value in raw if isinstance(value, str) and value in {"text", "image"}]
    return list(dict.fromkeys(modalities)) or ["text"]
