from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from .adapters.contracts import ADAPTER_CONTRACT_VERSION
from .config import Settings
from .hardware import (
    hardware_capability_class,
    hardware_envelope,
    hardware_envelope_satisfied,
)
from .model_planner import (
    ACTIVATION_PROBE_VERSION,
    LAUNCH_CONTRACT_VERSION,
    media_workflow_contract_version,
)
from .models import ModelCapabilityEvidence, ModelInstall, WorkflowRevision

if TYPE_CHECKING:
    from .runtime_provisioning import RuntimeProvisioner


ACTIVATION_ARTIFACT_KEY = "activation_artifact_sha256"


def _accepted_workflow_contracts(session: Session, install: ModelInstall) -> set[str | None]:
    """Workflow contract values that still describe this install's activation.

    The primary value is the artifact hash of the workflow the install was
    activated against, recorded at activation time. It is deliberately the
    *primary* one: an install can also have derived edit workflows, and comparing
    against whichever revision ran most recently would let an edit invalidate
    creation readiness even though both workflows are healthy.

    A legacy value - the old compiler-version key - is accepted alongside it, but
    only when it matches the compiler version that revision actually recorded, so
    it cannot validate an unrelated revision. Evidence recorded from then on uses
    the artifact contract, so the legacy path disappears on the next successful
    output or explicit activation.
    """

    template_sha256 = install.manifest_json.get("workflow_template_sha256")
    if not isinstance(template_sha256, str):
        # Not workflow-driven; the contract does not apply.
        return {None}
    accepted: set[str | None] = set()
    recorded = install.manifest_json.get(ACTIVATION_ARTIFACT_KEY)
    if isinstance(recorded, str) and recorded:
        accepted.add(recorded)
    for version in _recorded_compiler_versions(session, install, template_sha256):
        accepted.add(media_workflow_contract_version(template_sha256, version))
    return accepted or {None}


def _recorded_compiler_versions(
    session: Session,
    install: ModelInstall,
    template_sha256: str,
) -> set[int]:
    """Compiler versions declared by this install's own primary revisions."""

    versions: set[int] = set()
    template_id = install.manifest_json.get("workflow_template_id")
    for revision in session.scalars(
        select(WorkflowRevision).where(WorkflowRevision.engine == install.engine)
    ).all():
        dependencies = revision.dependencies_json or {}
        if dependencies.get("template_sha256") != template_sha256:
            continue
        if template_id and dependencies.get("template_id") != template_id:
            continue
        declared = dependencies.get("model_install_ids")
        if isinstance(declared, list) and install.id not in {str(item) for item in declared}:
            continue
        compiler = dependencies.get("compiler_version")
        if isinstance(compiler, int):
            versions.add(compiler)
    return versions


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
    accepted_workflows = _accepted_workflow_contracts(session, install)
    runtime_name = (
        cast(Literal["llama.cpp", "vllm", "comfyui"], install.engine)
        if install.engine in {"llama.cpp", "vllm", "comfyui"}
        else None
    )
    current_runtime = runtimes.status(runtime_name) if runtimes and runtime_name else None
    current_envelope = hardware_envelope(settings)
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
            or not _hardware_still_sufficient(evidence, settings, current_envelope)
            or evidence.component_hashes_json != expected_hashes
            or evidence.workflow_contract_version not in accepted_workflows
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


def _hardware_still_sufficient(
    evidence: ModelCapabilityEvidence,
    settings: Settings,
    current_envelope: Mapping[str, Any],
) -> bool:
    """Whether this machine can still do what the machine that proved it could.

    `hardware_class` is an equality token over the CPU model, total memory and the
    device list - and that device list is only populated when `llama-server` and
    `nvidia-smi` resolve on PATH. So a driver update or a PATH change threw away
    every proof on a machine that had not changed. Equality is kept as a fast
    accept, and as the only test for rows recorded before envelopes existed, but
    it is no longer allowed to reject on its own.
    """
    if evidence.hardware_class == hardware_capability_class(settings):
        return True
    if not evidence.hardware_envelope_json:
        return False
    return hardware_envelope_satisfied(evidence.hardware_envelope_json, current_envelope)


def record_capability_evidence(
    session: Session,
    install: ModelInstall,
    settings: Settings,
    runtimes: RuntimeProvisioner | None,
    *,
    component_hashes: dict[str, str],
    runtime_build: str,
    workflow_contract_version: str | None,
    details: dict[str, Any],
) -> ModelCapabilityEvidence:
    """Persist idempotent capability evidence for one exact runtime contract."""

    hardware_class = hardware_capability_class(settings)
    runtime_release: str | None = None
    runtime_managed: bool | None = None
    if runtimes and install.engine in {"llama.cpp", "vllm", "comfyui"}:
        runtime_status = runtimes.status(
            cast(Literal["llama.cpp", "vllm", "comfyui"], install.engine)
        )
        runtime_release = runtime_status.release
        runtime_managed = runtime_status.managed
    evidence_payload = {
        "component_hashes": dict(sorted(component_hashes.items())),
        "runtime_build": runtime_build,
        "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
        "launch_contract_version": LAUNCH_CONTRACT_VERSION,
        "workflow_contract_version": workflow_contract_version,
        "hardware_class": hardware_class,
        "probe_version": ACTIVATION_PROBE_VERSION,
        "runtime_release": runtime_release,
    }
    evidence_key = hashlib.sha256(
        json.dumps(
            evidence_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    existing = session.scalar(
        select(ModelCapabilityEvidence).where(
            ModelCapabilityEvidence.model_install_id == install.id,
            ModelCapabilityEvidence.evidence_key == evidence_key,
        )
    )
    if existing:
        return existing
    evidence = ModelCapabilityEvidence(
        model_install_id=install.id,
        evidence_key=evidence_key,
        result="ready",
        component_hashes_json=component_hashes,
        runtime_build=runtime_build[:200],
        adapter_contract_version=ADAPTER_CONTRACT_VERSION,
        launch_contract_version=LAUNCH_CONTRACT_VERSION,
        workflow_contract_version=workflow_contract_version,
        hardware_class=hardware_class[:200],
        hardware_envelope_json=hardware_envelope(settings),
        probe_version=ACTIVATION_PROBE_VERSION,
        details_json={
            **details,
            "runtime_release": runtime_release,
            "runtime_managed": runtime_managed,
        },
    )
    session.add(evidence)
    return evidence


def evidence_input_modalities(evidence: ModelCapabilityEvidence | None) -> list[str]:
    if not evidence:
        return ["text"]
    raw = evidence.details_json.get("input_modalities")
    if not isinstance(raw, list):
        return ["text"]
    modalities = [value for value in raw if isinstance(value, str) and value in {"text", "image"}]
    return list(dict.fromkeys(modalities)) or ["text"]
