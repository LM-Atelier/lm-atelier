"""Durable accepted conversation inputs and their retained artifact edges."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import JobStatus, MessageStatus, Operation, PartType, RoutingMode
from .models import (
    Artifact,
    Message,
    ModelInstall,
    ModelProfile,
    ModelSource,
    ResponseRevision,
    Run,
    RunContextArtifact,
    RunContextSnapshot,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from .vision import VisionSamplingPolicy
from .workflow_revision_reviews import revision_is_trusted


class ContextMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: str
    source_message_id: str | None
    response_revision_id: str | None
    content_sha256: str


class ContextDependency(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text", "artifact"]
    step_id: str
    plan_id: str
    run_id: str
    message_id: str


class AcceptedInstall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    role: str
    engine: str
    local_path: str
    manifest_json: dict[str, Any]
    source_id: str | None
    source_provenance: dict[str, str] | None
    shared_package_binding_id: str | None


class AcceptedProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    role: str
    engine: str
    load_settings_json: dict[str, Any]
    request_settings_json: dict[str, Any]
    install: AcceptedInstall | None


def capture_profile(session: Session, profile_id: str | None) -> AcceptedProfile | None:
    if profile_id is None:
        return None
    profile = session.get(ModelProfile, profile_id)
    if profile is None:
        raise ValueError("Accepted model configuration is unavailable.")
    install = (
        session.get(ModelInstall, profile.model_install_id) if profile.model_install_id else None
    )
    if profile.model_install_id and install is None:
        raise ValueError("Accepted model configuration is unavailable.")
    source = session.get(ModelSource, install.source_id) if install and install.source_id else None
    return AcceptedProfile(
        id=profile.id,
        name=profile.name,
        role=profile.role,
        engine=profile.engine,
        load_settings_json=copy.deepcopy(profile.load_settings_json),
        request_settings_json=copy.deepcopy(profile.request_settings_json),
        install=AcceptedInstall(
            id=install.id,
            name=install.name,
            role=install.role,
            engine=install.engine,
            local_path=install.local_path,
            manifest_json=copy.deepcopy(install.manifest_json),
            source_id=install.source_id,
            source_provenance={
                "provider": source.provider,
                "remote_id": source.remote_id,
                "revision": source.revision,
            }
            if source is not None
            else None,
            shared_package_binding_id=install.shared_package_binding_id,
        )
        if install is not None
        else None,
    )


def accepted_profile_provenance(profile: AcceptedProfile | None) -> dict[str, Any] | None:
    if profile is None or profile.install is None:
        return None
    return {
        "profile_id": profile.id,
        "profile_name": profile.name,
        "install_id": profile.install.id,
        "engine": profile.install.engine,
        "component_hashes": copy.deepcopy(
            profile.install.manifest_json.get("expected_sha256") or {}
        ),
        "source": copy.deepcopy(profile.install.source_provenance),
    }


def resolve_accepted_profile(
    session: Session, accepted: AcceptedProfile
) -> tuple[ModelProfile, ModelInstall, str]:
    """Use accepted configuration without changing the persisted profile."""
    frozen = accepted.install
    current = session.get(ModelInstall, frozen.id) if frozen is not None else None
    if (
        frozen is None
        or current is None
        or session.get(ModelProfile, accepted.id) is None
        or current.engine != frozen.engine
        or current.local_path != frozen.local_path
        or current.manifest_json != frozen.manifest_json
        or current.shared_package_binding_id != frozen.shared_package_binding_id
    ):
        raise RuntimeError("Accepted model installation is unavailable.")
    install = ModelInstall(**frozen.model_dump(exclude={"source_provenance"}), active=True)
    profile = ModelProfile(
        id=accepted.id,
        model_install_id=frozen.id,
        name=accepted.name,
        role=accepted.role,
        engine=accepted.engine,
        load_settings_json=copy.deepcopy(accepted.load_settings_json),
        request_settings_json=copy.deepcopy(accepted.request_settings_json),
    )
    scope = _digest(
        {
            "profile_id": accepted.id,
            "engine": accepted.engine,
            "load_settings": accepted.load_settings_json,
            "install": {
                "id": frozen.id,
                "engine": frozen.engine,
                "local_path": frozen.local_path,
                "manifest": frozen.manifest_json,
                "shared_package_binding_id": frozen.shared_package_binding_id,
            },
        }
    )
    return profile, install, scope


class AcceptedWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    id: str
    workflow_id: str
    version: int
    engine: str
    engine_version: str | None
    api_graph_json: dict[str, Any]
    input_schema_json: dict[str, Any]
    capabilities_json: list[str]
    dependencies_json: dict[str, Any]
    dependency_contract_sha256: str | None
    artifact_sha256: str | None
    trusted: bool


def capture_workflow(session: Session, revision_id: str | None) -> AcceptedWorkflow | None:
    if revision_id is None:
        return None
    revision = session.get(WorkflowRevision, revision_id)
    if revision is None:
        raise ValueError("Accepted workflow revision is unavailable.")
    return AcceptedWorkflow.model_validate(revision).model_copy(deep=True)


def resolve_accepted_workflow(
    session: Session, accepted: AcceptedWorkflow | None
) -> WorkflowRevision | None:
    if accepted is None:
        return None
    current = session.get(WorkflowRevision, accepted.id)
    if (
        current is None
        or current.engine != accepted.engine
        or current.workflow_id != accepted.workflow_id
        or current.dependency_contract_sha256 != accepted.dependency_contract_sha256
        or (accepted.engine == "comfyui" and (not current.trusted or not accepted.trusted))
    ):
        raise RuntimeError("Accepted workflow revision is unavailable or no longer trusted.")
    projection = WorkflowRevision(**accepted.model_dump())
    if accepted.engine == "comfyui" and (
        not revision_is_trusted(session, current) or not revision_is_trusted(session, projection)
    ):
        raise RuntimeError("Accepted workflow revision is unavailable or no longer trusted.")
    return projection


def capture_image_edit_strength(run: Run) -> dict[str, Any] | None:
    image_edit = run.provenance_json.get("image_edit")
    strength = image_edit.get("strength") if isinstance(image_edit, dict) else None
    return copy.deepcopy(strength) if isinstance(strength, dict) else None


class AcceptedContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    unavailable_reason: Literal["removed_context", "imported_media_unavailable"] | None = None
    run_id: str
    chat_id: str
    source_message_id: str | None = None
    source_run_id: str | None = None
    source_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    messages: list[ContextMessage]
    input_artifact_ids: list[str]
    visual_artifact_ids: list[str]
    visual_posters: dict[str, str] = Field(default_factory=dict)
    strict_artifact_ids: list[str]
    dependencies: list[ContextDependency]
    compiled_prompt: str | None
    standalone_prompt: str
    artifact_ids: list[str]
    context_artifact_ids: list[str] | None = None
    vision_settings: dict[str, Any]
    vision_sampling: VisionSamplingPolicy
    vision_bridge_max_tokens: int = Field(ge=1)
    context_limit: int = Field(ge=1)
    operation: str
    routing_mode: RoutingMode | None = None
    image_edit_strength: dict[str, Any] | None = None
    profile_id: str | None
    vision_profile_id: str | None
    workflow_revision_id: str | None
    settings: dict[str, Any]
    preset: dict[str, Any] | None = None
    preset_layers: list[dict[str, Any]] = Field(default_factory=list)
    chat_engine: str
    media_engine: str
    media_prompt: str
    workflow: AcceptedWorkflow | None
    workflow_activation: dict[str, Any] | None
    auxiliary_assets: dict[str, Any]
    profile: AcceptedProfile | None
    vision_profile: AcceptedProfile | None
    verification_profile: AcceptedProfile | None = None


def _digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def save_accepted_context(
    session: Session,
    run: Run,
    *,
    messages: list[dict[str, str]],
    sources: list[str | None],
    artifact_ids: set[str],
    input_artifact_ids: list[str],
    visual_artifact_ids: list[str],
    strict_artifact_ids: list[str],
    vision_settings: dict[str, Any],
    vision_sampling: VisionSamplingPolicy,
    vision_bridge_max_tokens: int,
    context_limit: int,
    chat_engine: str,
    media_engine: str,
    media_prompt: str,
    context_artifact_ids: set[str],
    verification_profile_id: str | None = None,
    inherited_context: AcceptedContext | None = None,
    inherited_configuration: AcceptedContext | None = None,
    inherit_profile_configuration: bool = False,
    inherit_vision_configuration: bool = False,
    inherit_workflow_configuration: bool = False,
) -> None:
    if session.get(RunContextSnapshot, run.id) is not None:
        raise ValueError("Conversation context was already accepted.")
    entries: list[ContextMessage] = []
    for value, source_id in zip(messages, sources, strict=True):
        source = session.get(Message, source_id) if source_id is not None else None
        if source_id is not None and (
            source is None or source.chat_id != run.chat_id or source.content_removed_at is not None
        ):
            raise ValueError("Accepted conversation context is unavailable.")
        entries.append(
            ContextMessage(
                role=value["role"],
                content=value["content"],
                source_message_id=source_id,
                response_revision_id=source.active_response_revision_id
                if source is not None
                else None,
                content_sha256=hashlib.sha256(value["content"].encode("utf-8")).hexdigest(),
            )
        )
    if inherited_context is not None:
        edit_source = (
            run.provenance_json.get("edit_source")
            or run.provenance_json.get("image_edit_verification_retry")
            or {}
        )
        if (
            edit_source.get("source_run_id") != inherited_context.run_id
            or inherited_context.chat_id != run.chat_id
        ):
            raise ValueError("Accepted edit context does not match its source.")
        source_id = edit_source.get("source_message_id")
        entries = [
            entry.model_copy(deep=True)
            for entry in inherited_context.messages
            if entry.source_message_id != source_id
        ] + [entry for entry in entries if entry.source_message_id == run.user_message_id]
        vision_settings = copy.deepcopy(inherited_context.vision_settings)
        vision_sampling = inherited_context.vision_sampling.model_copy(deep=True)
        vision_bridge_max_tokens = inherited_context.vision_bridge_max_tokens
        if run.provenance_json.get("image_edit_verification_retry"):
            media_prompt = inherited_context.media_prompt
    compiled = run.provenance_json.get("compiled_step")
    prompt = compiled.get("prompt") if isinstance(compiled, dict) else None
    workflow_provenance = run.provenance_json.get("workflow")
    activation = (
        workflow_provenance.get("activation") if isinstance(workflow_provenance, dict) else None
    )
    edit_source = (
        run.provenance_json.get("edit_source")
        or run.provenance_json.get("image_edit_verification_retry")
        or {}
    )
    configuration_context = inherited_configuration or inherited_context
    if inherited_configuration is not None:
        configuration_run = session.get(Run, inherited_configuration.run_id)
        if (
            configuration_run is None
            or inherited_configuration.chat_id != run.chat_id
            or configuration_run.user_message_id != edit_source.get("source_message_id")
        ):
            raise ValueError("Accepted step configuration does not match its source.")
        if inherit_vision_configuration:
            vision_settings = copy.deepcopy(inherited_configuration.vision_settings)
            vision_sampling = inherited_configuration.vision_sampling.model_copy(deep=True)
            vision_bridge_max_tokens = inherited_configuration.vision_bridge_max_tokens
    if inherit_profile_configuration:
        if configuration_context is None or run.profile_id != configuration_context.profile_id:
            raise ValueError("Accepted model configuration does not match its source.")
        profile = (
            configuration_context.profile.model_copy(deep=True)
            if configuration_context.profile is not None
            else None
        )
        context_limit = configuration_context.context_limit
        chat_engine = configuration_context.chat_engine
        media_engine = configuration_context.media_engine
    else:
        profile = capture_profile(session, run.profile_id)
    if inherit_vision_configuration:
        if (
            configuration_context is None
            or run.vision_profile_id != configuration_context.vision_profile_id
        ):
            raise ValueError("Accepted vision configuration does not match its source.")
        vision_profile = (
            configuration_context.vision_profile.model_copy(deep=True)
            if configuration_context.vision_profile is not None
            else None
        )
    else:
        vision_profile = capture_profile(session, run.vision_profile_id)
    inherit_verifier = inherit_vision_configuration or (
        configuration_context is not None
        and configuration_context.operation == run.operation == Operation.IMAGE_TO_IMAGE.value
    )
    verification_profile = (
        configuration_context.verification_profile.model_copy(deep=True)
        if inherit_verifier
        and configuration_context is not None
        and configuration_context.verification_profile is not None
        else None
        if inherit_verifier
        else capture_profile(session, verification_profile_id)
    )
    if inherit_workflow_configuration:
        if (
            configuration_context is None
            or run.workflow_revision_id != configuration_context.workflow_revision_id
        ):
            raise ValueError("Accepted workflow configuration does not match its source.")
        workflow = (
            configuration_context.workflow.model_copy(deep=True)
            if configuration_context.workflow is not None
            else None
        )
        activation = copy.deepcopy(configuration_context.workflow_activation)
    else:
        workflow = capture_workflow(session, run.workflow_revision_id)
    work_plan = session.get(WorkPlan, run.work_plan_id) if run.work_plan_id else None
    visual_posters: dict[str, str] = {}
    for artifact_id in visual_artifact_ids:
        if artifact_id in strict_artifact_ids:
            continue
        artifact = session.get(Artifact, artifact_id)
        if artifact is None or not artifact.media_type.casefold().startswith("video/"):
            continue
        poster_id = (
            inherited_context.visual_posters.get(artifact_id)
            if inherited_context is not None
            and artifact_id in inherited_context.visual_artifact_ids
            else artifact.metadata_json.get("poster_artifact_id")
        )
        poster = session.get(Artifact, poster_id) if isinstance(poster_id, str) else None
        if poster is not None and poster.media_type.casefold().startswith("image/"):
            visual_posters[artifact_id] = poster.id
            artifact_ids.add(poster.id)
            if artifact_id in context_artifact_ids:
                context_artifact_ids.add(poster.id)
    snapshot = AcceptedContext(
        run_id=run.id,
        chat_id=run.chat_id,
        source_message_id=edit_source.get("source_message_id"),
        source_run_id=edit_source.get("source_run_id"),
        source_snapshot_sha256=edit_source.get("source_snapshot_sha256"),
        messages=entries,
        input_artifact_ids=input_artifact_ids,
        visual_artifact_ids=visual_artifact_ids,
        visual_posters=visual_posters,
        strict_artifact_ids=strict_artifact_ids,
        dependencies=_accepted_dependencies(session, run),
        compiled_prompt=prompt if isinstance(prompt, str) else None,
        standalone_prompt=run.standalone_prompt,
        artifact_ids=sorted(artifact_ids | set(visual_artifact_ids)),
        context_artifact_ids=sorted(context_artifact_ids),
        vision_settings=vision_settings,
        vision_sampling=vision_sampling,
        vision_bridge_max_tokens=vision_bridge_max_tokens,
        context_limit=context_limit,
        operation=run.operation,
        routing_mode=work_plan.summary_json.get("routing_mode") if work_plan else None,
        profile_id=run.profile_id,
        vision_profile_id=run.vision_profile_id,
        workflow_revision_id=run.workflow_revision_id,
        settings=run.settings_json,
        image_edit_strength=capture_image_edit_strength(run),
        preset=copy.deepcopy(run.provenance_json.get("preset")),
        preset_layers=copy.deepcopy(run.provenance_json.get("preset_layers") or []),
        chat_engine=chat_engine,
        media_engine=media_engine,
        media_prompt=media_prompt,
        workflow=workflow,
        workflow_activation=copy.deepcopy(activation) if isinstance(activation, dict) else None,
        auxiliary_assets=copy.deepcopy(run.provenance_json.get("auxiliary_assets") or {}),
        profile=profile,
        vision_profile=vision_profile,
        verification_profile=verification_profile,
    )
    payload = snapshot.model_dump(mode="json")
    digest = _digest(payload)
    session.add(RunContextSnapshot(run_id=run.id, payload_json=payload, sha256=digest))
    session.flush()
    for artifact_id in snapshot.artifact_ids:
        session.add(RunContextArtifact(run_id=run.id, artifact_id=artifact_id))
    run.provenance_json = {**run.provenance_json, "accepted_context_sha256": digest}


def _accepted_dependencies(session: Session, run: Run) -> list[ContextDependency]:
    step = session.get(WorkStep, run.work_step_id) if run.work_step_id else None
    if step is None:
        return []
    edges = set(
        session.scalars(
            select(WorkStepDependency.depends_on_step_id).where(
                WorkStepDependency.step_id == step.id
            )
        )
    )
    requests: list[tuple[str, Literal["text", "artifact"] | None]] = []
    for binding in step.input_bindings_json:
        kind = binding.get("type")
        if kind not in {"step_output.text", "step_output.artifact"}:
            continue
        source_id = binding.get("source_step_id")
        if not isinstance(source_id, str) or source_id not in edges:
            raise ValueError("Accepted dependency identity is unavailable.")
        requests.append((source_id, "text" if kind == "step_output.text" else "artifact"))
    requested = {source_id for source_id, _ in requests}
    requests.extend((source_id, None) for source_id in sorted(edges - requested))
    result: list[ContextDependency] = []
    for source_id, kind in requests:
        producer_step = session.get(WorkStep, source_id)
        producer = (
            session.get(Run, producer_step.run_id)
            if producer_step and producer_step.run_id
            else None
        )
        if producer_step is None or producer is None or producer.chat_id != run.chat_id:
            raise ValueError("Accepted dependency identity is unavailable.")
        result.append(
            ContextDependency(
                kind=kind or ("text" if producer.operation == Operation.TEXT.value else "artifact"),
                step_id=producer_step.id,
                plan_id=producer_step.plan_id,
                run_id=producer.id,
                message_id=producer.assistant_message_id,
            )
        )
    return result


class ResolvedContextDependencies(NamedTuple):
    text_inputs: list[dict[str, str]]
    artifact_ids: list[str]
    visual_posters: dict[str, str]


def resolve_context_dependencies(
    session: Session, run: Run, snapshot: AcceptedContext
) -> ResolvedContextDependencies:
    text_inputs: list[dict[str, str]] = []
    artifact_ids: list[str] = []
    visual_posters: dict[str, str] = {}
    for dependency in snapshot.dependencies:
        step = session.get(WorkStep, dependency.step_id)
        producer = session.get(Run, dependency.run_id)
        message = session.get(Message, dependency.message_id)
        if (
            step is None
            or step.plan_id != dependency.plan_id
            or step.run_id != dependency.run_id
            or step.status != JobStatus.COMPLETE.value
            or producer is None
            or producer.chat_id != run.chat_id
            or producer.assistant_message_id != dependency.message_id
            or message is None
            or message.chat_id != run.chat_id
            or message.content_removed_at is not None
        ):
            raise RuntimeError("Accepted dependency output is unavailable.")
        revision = session.scalar(
            select(ResponseRevision).where(
                ResponseRevision.run_id == dependency.run_id,
                ResponseRevision.message_id == dependency.message_id,
            )
        )
        if revision is None or revision.status != MessageStatus.COMPLETE.value:
            raise RuntimeError("Accepted dependency output is unavailable.")
        if dependency.kind == "text":
            text = "\n".join(
                part.text
                for part in sorted(revision.parts, key=lambda part: part.position)
                if part.type == PartType.TEXT.value and part.text
            ).strip()
            if not text or len(text) > 50_000:
                raise RuntimeError("Accepted dependency text is unavailable.")
            text_inputs.append({"source_step_id": dependency.step_id, "text": text})
            continue
        selected: list[str] = []
        for part in sorted(revision.parts, key=lambda part: part.position):
            if (
                not part.artifact_id
                or part.metadata_json.get("preview")
                or part.metadata_json.get("input_reference")
            ):
                continue
            artifact = session.get(Artifact, part.artifact_id)
            if artifact is None:
                raise RuntimeError("Accepted dependency media is unavailable.")
            if snapshot.operation in {
                Operation.IMAGE_TO_IMAGE.value,
                Operation.IMAGE_TO_VIDEO.value,
            } and not artifact.media_type.casefold().startswith("image/"):
                continue
            if (
                snapshot.operation == Operation.TEXT.value
                and not artifact.media_type.casefold().startswith(("image/", "video/"))
            ):
                continue
            selected.append(artifact.id)
            poster_id = part.metadata_json.get("poster_artifact_id")
            if artifact.media_type.casefold().startswith("video/") and poster_id is not None:
                poster = session.get(Artifact, poster_id) if isinstance(poster_id, str) else None
                if poster is None or not poster.media_type.casefold().startswith("image/"):
                    raise RuntimeError("Accepted dependency poster is unavailable.")
                visual_posters.setdefault(artifact.id, poster.id)
        if not selected:
            raise RuntimeError("Accepted dependency media is unavailable.")
        artifact_ids.extend(selected)
    return ResolvedContextDependencies(
        text_inputs, list(dict.fromkeys(artifact_ids)), visual_posters
    )


def accepted_context(session: Session, run: Run) -> AcceptedContext | None:
    expected = run.provenance_json.get("accepted_context_sha256")
    row = session.get(RunContextSnapshot, run.id)
    if expected is None and row is None:
        return None
    if row is None or expected is None:
        raise ValueError("Accepted conversation context is unavailable.")
    try:
        snapshot = AcceptedContext.model_validate(row.payload_json)
        valid = (
            snapshot.run_id == run.id
            and snapshot.chat_id == run.chat_id
            and row.sha256 == expected
            and _digest(row.payload_json) == expected
        )
    except (ValidationError, ValueError, TypeError):
        raise ValueError("Accepted conversation context is unavailable.") from None
    if not valid or snapshot.unavailable_reason is not None:
        raise ValueError("Accepted conversation context is unavailable.")
    retained = set(
        session.scalars(
            select(RunContextArtifact.artifact_id).where(RunContextArtifact.run_id == run.id)
        )
    )
    if retained != set(snapshot.artifact_ids):
        raise ValueError("Accepted conversation context is unavailable.")
    for entry in snapshot.messages:
        if entry.source_message_id is None:
            continue
        source = session.get(Message, entry.source_message_id)
        if source is None or source.chat_id != run.chat_id or source.content_removed_at is not None:
            raise ValueError("Accepted conversation context is unavailable.")
    return snapshot
