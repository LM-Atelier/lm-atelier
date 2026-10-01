"""Accept a new queued version without changing the source exchange."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .accepted_turn_context import (
    AcceptedContext,
    accepted_context,
    capture_image_edit_strength,
    recorded_enlargement,
    settings_workflow,
)
from .auxiliary_assets import revision_accepts_added_loras
from .domain import MessageRole, Operation, RoutingMode
from .models import (
    Artifact,
    Message,
    ModelProfile,
    Run,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from .prior_turn_inheritance import PriorTurnInheritance
from .schemas import (
    ArtifactOut,
    MessageReferenceOut,
    PriorTurnEditAccepted,
    PriorTurnEditBinding,
    PriorTurnEditConfiguration,
    PriorTurnEditRequest,
    PriorTurnEditSource,
    PriorTurnEditStepSource,
    SourceFitRequest,
    TurnAccepted,
    TurnRequest,
    WorkflowSelectionOut,
)
from .source_fit_preview import SourceFitPreviewOut
from .turn_inheritance import inherited_edit_strength
from .upscale_workflows import effective_upscale_schema
from .workflow_use_case_execution import InheritedWorkflowUseCasePreset
from .workflow_use_case_preset_provenance import read_workflow_use_case_preset

if TYPE_CHECKING:
    from .orchestrator import ConversationOrchestrator


class EditRequestConflict(ValueError):
    pass


def _source(session: Session, message_id: str, run_id: str | None) -> tuple[Message, Run]:
    message = session.scalar(
        select(Message).options(selectinload(Message.parts)).where(Message.id == message_id)
    )
    if message is None or message.role != MessageRole.USER.value or not message.transcript_visible:
        raise LookupError("Source user message not found.")
    if message.content_removed_at is not None:
        raise EditRequestConflict("Source content was removed and cannot be edited.")
    query = select(Run).where(Run.user_message_id == message.id, Run.chat_id == message.chat_id)
    if run_id is not None:
        query = query.where(Run.id == run_id)
    run = session.scalar(query.order_by(Run.created_at, Run.id).limit(1))
    if run is None:
        raise LookupError("Source run not found for this message.")
    return message, run


def _fingerprint(message_id: str, payload: PriorTurnEditRequest) -> str:
    value = {
        "source_message_id": message_id,
        "request": payload.model_dump(mode="json", exclude_unset=True, exclude={"idempotency_key"}),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _edit_result(
    session: Session, accepted: TurnAccepted, fingerprint: str
) -> PriorTurnEditAccepted:
    source = accepted.run.provenance_json.get("edit_source")
    if not isinstance(source, dict) or source.get("request_sha256") != fingerprint:
        raise EditRequestConflict("This request ID already belongs to a different source or edit.")
    digest = accepted.run.provenance_json.get("accepted_context_sha256")
    if not isinstance(digest, str) or not accepted.run.work_plan_id:
        raise EditRequestConflict("The accepted edit is unavailable.")
    head_id = session.scalar(
        select(Run.assistant_message_id)
        .join(WorkStep, WorkStep.id == Run.work_step_id)
        .where(Run.work_plan_id == accepted.run.work_plan_id)
        .order_by(WorkStep.ordinal.desc())
        .limit(1)
    )
    if head_id is None:
        raise EditRequestConflict("The accepted edit branch is unavailable.")
    return PriorTurnEditAccepted(
        **accepted.model_dump(),
        source_message_id=source["source_message_id"],
        source_run_id=source["source_run_id"],
        work_plan_id=accepted.run.work_plan_id,
        branch_head_message_id=head_id,
        accepted_context_sha256=digest,
    )


def _configuration(
    session: Session, prior: Run, snapshot: AcceptedContext | None
) -> tuple[PriorTurnEditConfiguration, str | None, bool]:
    """A turn's configuration, its engine, and whether its workflow takes a LoRA stack."""

    operation = Operation(snapshot.operation if snapshot is not None else prior.operation)
    role = (
        "chat"
        if operation == Operation.TEXT
        else "video"
        if "video" in operation.value
        else "image"
    )
    revision_id = (
        snapshot.workflow_revision_id if snapshot is not None else prior.workflow_revision_id
    )
    workflow = settings_workflow(session, prior, snapshot)
    profile_id = snapshot.profile_id if snapshot is not None else prior.profile_id
    profile = session.get(ModelProfile, profile_id) if profile_id else None
    input_schema = (
        copy.deepcopy(effective_upscale_schema(workflow.api_graph_json, workflow.input_schema_json))
        if workflow is not None
        else None
    )
    resolved_settings = copy.deepcopy(
        snapshot.settings if snapshot is not None else prior.settings_json
    )
    engine = (
        snapshot.profile.engine
        if snapshot and snapshot.profile
        else profile.engine
        if profile
        else None
    )
    preset = copy.deepcopy(
        snapshot.preset if snapshot is not None else prior.provenance_json.get("preset")
    )
    selection = copy.deepcopy(prior.provenance_json.get("model_selection") or {})
    output = prior.provenance_json.get("media_output") or {}
    return (
        PriorTurnEditConfiguration(
            upscale=recorded_enlargement(prior, snapshot),
            source_fit=(
                SourceFitRequest(
                    mode=snapshot.source_fit.mode,
                    width=snapshot.source_fit.canvas_width,
                    height=snapshot.source_fit.canvas_height,
                )
                if snapshot is not None and snapshot.source_fit is not None
                else None
            ),
            image_edit_strength=(
                snapshot.image_edit_strength
                if snapshot is not None and "image_edit_strength" in snapshot.model_fields_set
                else capture_image_edit_strength(prior)
            ),
            operation=operation.value,
            profile_engine=engine,
            settings=copy.deepcopy(resolved_settings),
            resolved_settings=resolved_settings,
            settings_role=role,
            output_count=output.get("count", 1),
            profile_id=profile_id,
            vision_profile_id=snapshot.vision_profile_id
            if snapshot is not None
            else prior.vision_profile_id,
            preset_id=preset.get("id") if isinstance(preset, dict) else None,
            preset=preset if isinstance(preset, dict) else None,
            model_selection=selection,
            workflow_selection=WorkflowSelectionOut(
                selector_capability="chat"
                if role == "chat"
                else "image"
                if role == "image"
                else "video",
                mode="revision" if revision_id else "legacy",
                workflow_family_id=selection.get("workflow_family_id"),
                workflow_revision_id=revision_id,
                legacy_profile_id=profile_id,
            ),
            workflow_revision_id=revision_id,
            workflow_schema=input_schema,
            profile_settings=copy.deepcopy(
                {**snapshot.profile.load_settings_json, **snapshot.profile.request_settings_json}
                if snapshot and snapshot.profile
                else {**profile.load_settings_json, **profile.request_settings_json}
                if profile
                else {}
            ),
        ),
        engine,
        workflow is not None and revision_accepts_added_loras(workflow),
    )


def _source_view(
    orchestrator: ConversationOrchestrator,
    session: Session,
    message_id: str,
    source_run_id: str | None = None,
) -> tuple[PriorTurnEditSource, str | None, bool]:
    source, prior = _source(session, message_id, source_run_id)
    snapshot = accepted_context(session, prior)
    operation = Operation(snapshot.operation if snapshot is not None else prior.operation)
    mode = (
        RoutingMode.TEXT
        if operation == Operation.TEXT
        else (RoutingMode.VIDEO if "video" in operation.value else RoutingMode.IMAGE)
    )
    input_ids = (
        list(snapshot.input_artifact_ids)
        if snapshot is not None
        else orchestrator.input_artifact_ids_for_run(session, prior)
    )
    artifacts = []
    for artifact_id in input_ids:
        artifact = session.get(Artifact, artifact_id)
        if artifact is None:
            raise EditRequestConflict("A source input is no longer available.")
        artifacts.append(ArtifactOut.model_validate(artifact))
    configuration, engine, takes_added_loras = _configuration(session, prior, snapshot)
    work_plan = session.get(WorkPlan, prior.work_plan_id) if prior.work_plan_id else None
    summary = work_plan.summary_json if work_plan else {}
    recorded_mode = snapshot.routing_mode if snapshot is not None else None
    if recorded_mode is None:
        raw_mode = summary.get("routing_mode")
        recorded_mode = (
            RoutingMode(raw_mode)
            if isinstance(raw_mode, str) and raw_mode in {mode.value for mode in RoutingMode}
            else None
        )
    step_sources = []
    if work_plan and summary.get("operation") == "ordered":
        rows = session.execute(
            select(WorkStep, Run)
            .outerjoin(Run, Run.work_step_id == WorkStep.id)
            .where(WorkStep.plan_id == work_plan.id)
            .order_by(WorkStep.ordinal, Run.id)
        ).all()
        for step, run in rows:
            if run is None or run.user_message_id != source.id:
                raise EditRequestConflict("A source step is no longer available.")
            selected, _, _ = _configuration(session, run, accepted_context(session, run))
            step_sources.append(
                PriorTurnEditStepSource(
                    **selected.model_dump(),
                    step_id=step.id,
                    ordinal=step.ordinal,
                    source_run_id=run.id,
                    depends_on=sorted(
                        session.scalars(
                            select(WorkStepDependency.depends_on_step_id).where(
                                WorkStepDependency.step_id == step.id
                            )
                        )
                    ),
                )
            )
    messages, sources = orchestrator._context_messages_with_sources(
        session, prior, include_step_context=False
    )
    context = [
        value for value, identity in zip(messages, sources, strict=True) if identity != source.id
    ]
    prompt_source = copy.deepcopy(prior.provenance_json.get("prompt_source"))
    text = "\n".join(
        part.text
        for part in sorted(source.parts, key=lambda part: part.position)
        if part.type == "text" and part.text
    )
    result = PriorTurnEditSource(
        source_user_message_id=source.id,
        source_run_id=prior.id,
        source_snapshot_sha256="",
        chat_id=source.chat_id,
        text=text,
        mode=mode,
        original_mode=recorded_mode,
        plan_kind="ordered" if summary.get("operation") == "ordered" else "single",
        steps=step_sources,
        input_artifact_ids=input_ids,
        input_artifacts=artifacts,
        references=[MessageReferenceOut.model_validate(row) for row in source.references],
        **configuration.model_dump(),
        context_messages=context,
        context_visual_artifacts=[
            ArtifactOut.model_validate(artifact)
            for artifact in orchestrator._visual_context_artifacts(
                session, prior, lookback=orchestrator.engines.settings.vision_prior_visual_lookback
            )
            if artifact.id not in input_ids
        ],
        prompt_source=prompt_source if isinstance(prompt_source, dict) else None,
    )
    # Settings shown in the editor are a filtered projection of these resolved
    # settings. Bind the source values and context identities, not transport-only
    # artifact decoration or a later change to the source run's progress.
    identity = result.model_dump(
        mode="json",
        exclude={
            "source_snapshot_sha256",
            "settings",
            "input_artifacts",
            "context_visual_artifacts",
        },
    )
    for step in identity["steps"]:
        step.pop("settings", None)
    identity["parent_message_id"] = source.parent_id
    identity["engine"] = engine
    identity["input_artifacts"] = [
        item.model_dump(mode="json", exclude={"favorite", "url", "generation_identity"})
        for item in artifacts
    ]
    identity["context_visual_artifacts"] = [
        item.model_dump(mode="json", exclude={"favorite", "url", "generation_identity"})
        for item in result.context_visual_artifacts
    ]
    identity["context_identities"] = (
        [
            (entry.source_message_id, entry.response_revision_id)
            for entry in snapshot.messages
            if entry.source_message_id != source.id
        ]
        if snapshot is not None
        else [
            (
                source_id,
                message.active_response_revision_id
                if source_id is not None and (message := session.get(Message, source_id))
                else None,
            )
            for source_id in sources
            if source_id != source.id
        ]
    )
    result.source_snapshot_sha256 = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return result, engine, takes_added_loras


def classify_prior_turn_edit(
    orchestrator: ConversationOrchestrator,
    session: Session,
    chat_id: str,
    binding: PriorTurnEditBinding,
    *,
    text: str,
    mode: RoutingMode | None,
) -> bool:
    source, _, _ = _source_view(
        orchestrator, session, binding.source_message_id, binding.source_run_id
    )
    if source.chat_id != chat_id:
        raise LookupError("Source turn not found in this chat.")
    if source.source_snapshot_sha256 != binding.source_snapshot_sha256:
        raise EditRequestConflict("The source changed. Reload it before classifying this edit.")
    return orchestrator.router.references_prior_visual(
        text=text, mode=mode or source.mode, conversation=source.context_messages
    )


async def prior_turn_edit_source(
    orchestrator: ConversationOrchestrator,
    session: Session,
    message_id: str,
    source_run_id: str | None = None,
) -> PriorTurnEditSource:
    result, engine, takes_added_loras = _source_view(
        orchestrator, session, message_id, source_run_id
    )
    result.settings = await orchestrator.request_settings_for_operation(
        Operation(result.operation),
        result.resolved_settings,
        input_schema=result.workflow_schema,
        engine=engine,
        accepts_added_loras=takes_added_loras,
    )
    for step in result.steps:
        run = session.get(Run, step.source_run_id)
        if run is None:
            raise EditRequestConflict("A source step is no longer available.")
        _, step_engine, step_takes_added_loras = _configuration(
            session, run, accepted_context(session, run)
        )
        step.settings = await orchestrator.request_settings_for_operation(
            Operation(step.operation),
            step.resolved_settings,
            input_schema=step.workflow_schema,
            engine=step_engine,
            accepts_added_loras=step_takes_added_loras,
        )
    session.expire_all()
    current, _, _ = _source_view(orchestrator, session, message_id, result.source_run_id)
    if current.source_snapshot_sha256 != result.source_snapshot_sha256:
        raise EditRequestConflict("The source changed while loading. Reload the edited version.")
    return result


@dataclass(frozen=True, slots=True)
class PreparedPriorTurnEdit:
    """Read-only edit configuration, requiring fresh validation at admission."""

    source: Message
    prior: Run
    payload: PriorTurnEditRequest
    snapshot: AcceptedContext | None
    source_digest: str
    prior_operation: Operation
    source_inheritance: PriorTurnInheritance | None
    same_role: bool
    workflow_override: bool
    inherit_preset: bool
    inherited_preset: dict[str, Any] | None
    inherited_preset_layers: list[dict[str, Any]]
    inherited_prompt_source: object | None
    inherited_strength: dict[str, Any] | None
    inherited_auxiliary: dict[str, Any]
    inherit_loras: bool
    request: TurnRequest


async def prepare_prior_turn_edit(
    orchestrator: ConversationOrchestrator,
    session: Session,
    message_id: str,
    payload: PriorTurnEditRequest,
) -> PreparedPriorTurnEdit:
    """Reuse queued-edit selection without admitting work or changing the source."""
    source, prior = _source(session, message_id, payload.source_run_id)
    editor_source, _, _ = _source_view(orchestrator, session, message_id, prior.id)
    source_digest = editor_source.source_snapshot_sha256
    if (
        payload.source_snapshot_sha256 is not None
        and payload.source_snapshot_sha256 != source_digest
    ):
        raise EditRequestConflict("The source changed. Reload it before queuing this edit.")
    snapshot = accepted_context(session, prior)
    prior_operation = Operation(snapshot.operation if snapshot is not None else prior.operation)
    prior_mode = (
        RoutingMode.TEXT
        if prior_operation == Operation.TEXT
        else (RoutingMode.VIDEO if "video" in prior_operation.value else RoutingMode.IMAGE)
    )
    target_mode = payload.mode or editor_source.original_mode or prior_mode
    use_source_configurations = (
        target_mode == RoutingMode.AUTO or bool(editor_source.steps) or bool(payload.step_overrides)
    )
    source_inheritance = (
        PriorTurnInheritance(orchestrator, session, editor_source, payload)
        if use_source_configurations
        else None
    )
    same_role = target_mode == prior_mode and not use_source_configurations
    if target_mode != RoutingMode.AUTO:
        payload = payload.for_role("chat" if target_mode == RoutingMode.TEXT else target_mode.value)
    values = payload.model_dump(
        exclude={"source_run_id", "source_snapshot_sha256", "step_overrides"},
        exclude_unset=True,
    )
    values.update(parent_message_id=source.parent_id, mode=target_mode)
    workflow_override = bool(
        {"workflow_selection", "workflow_revision_id"} & payload.model_fields_set
    )
    if same_role and "source_fit" not in payload.model_fields_set:
        values["source_fit"] = (
            editor_source.source_fit.model_dump(mode="json")
            if editor_source.source_fit is not None
            else None
        )
    if same_role and "upscale" not in payload.model_fields_set:
        values["upscale"] = editor_source.upscale
    if same_role and not workflow_override and "profile_id" not in payload.model_fields_set:
        values["profile_id"] = snapshot.profile_id if snapshot is not None else prior.profile_id
    if (
        same_role
        and prior_operation == Operation.TEXT
        and "vision_profile_id" not in payload.model_fields_set
    ):
        values["vision_profile_id"] = (
            snapshot.vision_profile_id if snapshot is not None else prior.vision_profile_id
        )
    if "input_artifact_ids" not in payload.model_fields_set:
        values["input_artifact_ids"] = (
            list(snapshot.input_artifact_ids)
            if snapshot is not None
            else orchestrator.input_artifact_ids_for_run(session, prior)
        )
    inherit_preset = same_role and "preset_id" not in payload.model_fields_set
    inherited_preset = copy.deepcopy(editor_source.preset)
    inherited_preset_layers = copy.deepcopy(
        snapshot.preset_layers
        if snapshot is not None
        else prior.provenance_json.get("preset_layers") or []
    )
    if inherit_preset:
        # The accepted source already carries the preset's resolved values.
        # Do not reapply the mutable preset or the chat's newer selection.
        values["preset_id"] = None
    settings = snapshot.settings if snapshot is not None else prior.settings_json
    workflow_id = (
        snapshot.workflow_revision_id if snapshot is not None else prior.workflow_revision_id
    )
    if same_role and not workflow_override:
        values["workflow_revision_id"] = workflow_id
    if same_role and payload.preset_id is None:
        workflow = settings_workflow(session, prior, snapshot)
        profile_id = snapshot.profile_id if snapshot is not None else prior.profile_id
        profile = session.get(ModelProfile, profile_id) if profile_id else None
        values["settings"] = await orchestrator.request_settings_for_operation(
            prior_operation,
            settings,
            input_schema=workflow.input_schema_json if workflow is not None else None,
            api_graph=workflow.api_graph_json if workflow is not None else None,
            engine=snapshot.profile.engine
            if snapshot and snapshot.profile
            else profile.engine
            if profile
            else None,
            accepts_added_loras=(workflow is not None and revision_accepts_added_loras(workflow)),
        )
        if values.get("source_fit") is not None:
            # Legacy resolved dimensions are not a second explicit canvas choice.
            values["settings"].pop("width", None)
            values["settings"].pop("height", None)
        values["settings"] = {**values["settings"], **payload.settings}
    if same_role and "output_count" not in payload.model_fields_set:
        output = prior.provenance_json.get("media_output") or {}
        values["output_count"] = output.get("count", 1)
    inherited_prompt_source: object | None = None
    if "prompt_source" not in payload.model_fields_set and target_mode in {
        RoutingMode.IMAGE,
        RoutingMode.AUTO,
    }:
        inherited_prompt_source = prior.provenance_json.get("prompt_source")
    inherited_strength = (
        inherited_edit_strength(editor_source.image_edit_strength, payload, prior_operation)
        if same_role
        else None
    )
    inherited_auxiliary = copy.deepcopy(
        snapshot.auxiliary_assets
        if snapshot is not None
        else prior.provenance_json.get("auxiliary_assets") or {}
    )
    inherit_loras = same_role and payload.preset_id is None and "loras" not in payload.settings
    # Schema loading may yield. A preview must not return a source projection
    # assembled across different versions, even though admission rechecks again.
    session.expire_all()
    current, _, _ = _source_view(orchestrator, session, message_id, prior.id)
    if current.source_snapshot_sha256 != source_digest:
        raise EditRequestConflict("The source changed. Reload it before preparing this edit.")
    return PreparedPriorTurnEdit(
        source=source,
        prior=prior,
        payload=payload,
        snapshot=snapshot,
        source_digest=source_digest,
        prior_operation=prior_operation,
        source_inheritance=source_inheritance,
        same_role=same_role,
        workflow_override=workflow_override,
        inherit_preset=inherit_preset,
        inherited_preset=inherited_preset,
        inherited_preset_layers=inherited_preset_layers,
        inherited_prompt_source=inherited_prompt_source,
        inherited_strength=inherited_strength,
        inherited_auxiliary=inherited_auxiliary,
        inherit_loras=inherit_loras,
        request=TurnRequest.model_validate(values),
    )


async def preview_prior_turn_source_fit(
    orchestrator: ConversationOrchestrator,
    session: Session,
    message_id: str,
    payload: PriorTurnEditRequest,
) -> SourceFitPreviewOut:
    """Preview the same edit selection, then revalidate its source after awaits."""
    if session.new or session.dirty or session.deleted:
        raise ValueError("A source preview requires a read-only session.")
    with session.no_autoflush:
        prepared = await prepare_prior_turn_edit(orchestrator, session, message_id, payload)
        snapshot = prepared.snapshot
        inherit_workflow = prepared.same_role and not prepared.workflow_override
        result = await orchestrator.preview_turn_source_fit(
            session,
            prepared.source.chat_id,
            prepared.request,
            use_explicit_parent=True,
            source_action="edit_and_branch",
            inherited_image_edit_strength=prepared.inherited_strength,
            inherited_prompt_source=prepared.inherited_prompt_source,
            inherited_workflow=snapshot.workflow if snapshot and inherit_workflow else None,
            inherited_source_fit=snapshot.source_fit if snapshot and inherit_workflow else None,
            inherited_use_case_preset=(
                InheritedWorkflowUseCasePreset(
                    snapshot.workflow_use_case_preset
                    if snapshot
                    else read_workflow_use_case_preset(
                        prepared.prior.provenance_json.get("workflow_use_case_preset"),
                        workflow_revision_id=prepared.prior.workflow_revision_id,
                    )
                )
                if prepared.same_role
                else None
            ),
            reference_source_message_id=prepared.source.id
            if "references" not in prepared.payload.model_fields_set
            else None,
            resolve_source=prepared.source_inheritance.resolve
            if prepared.source_inheritance
            else None,
        )
        if prepared.source_inheritance is not None:
            prepared.source_inheritance.validate_consumed()
        prior_id = prepared.prior.id
        session.expire_all()
        current, _, _ = _source_view(orchestrator, session, message_id, prior_id)
        if current.source_snapshot_sha256 != prepared.source_digest:
            raise EditRequestConflict("The source changed. Reload it before previewing this edit.")
    return result


async def queue_prior_turn_edit(
    orchestrator: ConversationOrchestrator,
    session: Session,
    message_id: str,
    payload: PriorTurnEditRequest,
) -> PriorTurnEditAccepted:
    if not orchestrator._admission_open:
        raise RuntimeError("This conversation service cannot accept new work while shutting down.")
    source, _ = _source(session, message_id, payload.source_run_id)
    chat_id = source.chat_id
    fingerprint = _fingerprint(message_id, payload)
    async with orchestrator.chat_guard(chat_id):
        session.expire_all()
        source, prior = _source(session, message_id, payload.source_run_id)
        replay = orchestrator._idempotent_run(session, chat_id, payload.idempotency_key)
        if replay is not None:
            return _edit_result(
                session, orchestrator._accepted_for_run(session, replay), fingerprint
            )
        prepared = await prepare_prior_turn_edit(orchestrator, session, message_id, payload)
        source = prepared.source
        prior = prepared.prior
        payload = prepared.payload
        snapshot = prepared.snapshot
        source_digest = prepared.source_digest
        prior_operation = prepared.prior_operation
        source_inheritance = prepared.source_inheritance
        same_role = prepared.same_role
        workflow_override = prepared.workflow_override
        inherit_preset = prepared.inherit_preset
        inherited_preset = prepared.inherited_preset
        inherited_preset_layers = prepared.inherited_preset_layers
        inherited_prompt_source = prepared.inherited_prompt_source
        inherited_strength = prepared.inherited_strength
        inherited_auxiliary = prepared.inherited_auxiliary
        inherit_loras = prepared.inherit_loras
        prior_id = prior.id

        def bind_source(transaction: Session, first: Run) -> None:
            # This hook runs after all awaits, inside the acceptance transaction.
            transaction.flush()
            transaction.expire_all()
            current, _, _ = _source_view(orchestrator, transaction, message_id, prior_id)
            if current.source_snapshot_sha256 != source_digest:
                raise EditRequestConflict("The source changed. Reload it before queuing this edit.")
            runs = transaction.scalars(
                select(Run).where(Run.work_plan_id == first.work_plan_id)
            ).all()
            if source_inheritance is not None:
                source_inheritance.validate_consumed()
            for run in runs:
                configuration = None
                if source_inheritance is not None:
                    step = transaction.get(WorkStep, run.work_step_id) if run.work_step_id else None
                    ordinal = (
                        None if None in source_inheritance.bound else step.ordinal if step else None
                    )
                    configuration = source_inheritance.bind(transaction, run, ordinal)
                if inherit_loras and orchestrator._role_for_operation(
                    Operation(run.operation)
                ) == orchestrator._role_for_operation(prior_operation):
                    resolved_auxiliary = run.provenance_json.get("auxiliary_assets") or {}
                    if (resolved_auxiliary.get("lora_stack") or []) != (
                        inherited_auxiliary.get("lora_stack") or []
                    ):
                        raise EditRequestConflict(
                            "An inherited LoRA changed. Select it again before queuing this edit."
                        )
                if inherit_preset and orchestrator._role_for_operation(
                    Operation(run.operation)
                ) == orchestrator._role_for_operation(prior_operation):
                    run.provenance_json = {
                        **run.provenance_json,
                        "preset": copy.deepcopy(inherited_preset),
                        "preset_layers": copy.deepcopy(inherited_preset_layers),
                    }
                run.provenance_json = {
                    **run.provenance_json,
                    "edit_source": {
                        "source_message_id": message_id,
                        "source_run_id": prior_id,
                        "request_sha256": fingerprint,
                        "source_snapshot_sha256": source_digest,
                    },
                }
                orchestrator._freeze_turn_context(
                    transaction,
                    run,
                    inherited_context=snapshot,
                    inherited_configuration=configuration.source.snapshot
                    if configuration
                    else None,
                    inherit_workflow_configuration=(
                        bool(configuration.source.snapshot and configuration.inherit_workflow)
                        if configuration
                        else snapshot is not None
                        and same_role
                        and not workflow_override
                        and orchestrator._role_for_operation(Operation(run.operation))
                        == orchestrator._role_for_operation(prior_operation)
                    ),
                    inherit_vision_configuration=(
                        bool(configuration.source.snapshot and configuration.inherit_vision)
                        if configuration
                        else snapshot is not None
                        and same_role
                        and prior_operation == Operation.TEXT
                        and run.operation == Operation.TEXT.value
                        and "vision_profile_id" not in payload.model_fields_set
                    ),
                    inherit_profile_configuration=(
                        bool(configuration.source.snapshot and configuration.inherit_profile)
                        if configuration
                        else snapshot is not None
                        and same_role
                        and not workflow_override
                        and "profile_id" not in payload.model_fields_set
                        and orchestrator._role_for_operation(Operation(run.operation))
                        == orchestrator._role_for_operation(prior_operation)
                    ),
                )

            plan = transaction.get(WorkPlan, first.work_plan_id)
            head_id = transaction.scalar(
                select(Run.assistant_message_id)
                .join(WorkStep, WorkStep.id == Run.work_step_id)
                .where(Run.work_plan_id == first.work_plan_id)
                .order_by(WorkStep.ordinal.desc())
                .limit(1)
            )
            if plan is None or head_id is None:
                raise EditRequestConflict("The accepted edit branch is unavailable.")
            plan.summary_json = {
                **plan.summary_json,
                "edit_source": copy.deepcopy(first.provenance_json["edit_source"]),
                "branch_head_message_id": head_id,
            }

        accepted = await orchestrator._create_turn(
            session,
            chat_id,
            prepared.request,
            use_explicit_parent=True,
            source_action="edit_and_branch",
            activate_branch=False,
            inherited_image_edit_strength=inherited_strength,
            inherited_prompt_source=inherited_prompt_source,
            inherited_workflow=(
                snapshot.workflow
                if snapshot is not None and same_role and not workflow_override
                else None
            ),
            inherited_source_fit=(
                snapshot.source_fit
                if snapshot is not None and same_role and not workflow_override
                else None
            ),
            reference_source_message_id=source.id
            if "references" not in payload.model_fields_set
            else None,
            before_commit=bind_source,
            inherited_use_case_preset=(
                InheritedWorkflowUseCasePreset(
                    snapshot.workflow_use_case_preset
                    if snapshot
                    else read_workflow_use_case_preset(
                        prepared.prior.provenance_json.get("workflow_use_case_preset"),
                        workflow_revision_id=prepared.prior.workflow_revision_id,
                    )
                )
                if same_role
                else None
            ),
            resolve_source=source_inheritance.resolve if source_inheritance else None,
        )
        return _edit_result(session, accepted, fingerprint)
