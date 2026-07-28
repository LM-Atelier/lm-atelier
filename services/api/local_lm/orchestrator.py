from __future__ import annotations

import asyncio
import copy
import hashlib
import logging
import re
import secrets
import shutil
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, object_session, selectinload

from .adapters.base import ChatRequest, MediaEvent, MediaRequest
from .artifacts import ArtifactStore
from .auxiliary_assets import (
    LORA_GRAPH_TRANSFORM_VERSION,
    resolve_lora_stack,
    select_automatic_lora_stack,
    transform_lora_graph,
    workflow_lora_extension,
)
from .capability_evidence import current_capability_evidence, evidence_input_modalities
from .context_compaction import (
    CONTEXT_COMPACTION_VERSION,
    MAX_COMPACTION_CHARACTERS,
    MIN_COMPACTION_CHARACTERS,
    compact_context_messages,
)
from .custom_nodes import custom_node_dependency_errors
from .db import SessionLocal
from .domain import (
    ArtifactKind,
    JobKind,
    JobStatus,
    MessageRole,
    MessageStatus,
    Operation,
    PartType,
    RoutingMode,
    RunStatus,
    elapsed_milliseconds,
    utcnow,
)
from .engines import EngineRegistry
from .events import EventBroker
from .generation_offers import (
    extract_generation_offer,
    generation_offer_from_metadata,
    generation_offer_metadata,
    is_explicit_generation_assent,
    ordered_intent_for_offer,
    routing_plan_for_offer,
    should_extract_generation_offer,
)
from .image_edit_strength import (
    EditSettingSource,
    ImageEditStrengthResolution,
    resolve_image_edit_strength,
)
from .models import (
    Artifact,
    Chat,
    GenerationPreset,
    Job,
    Message,
    MessagePart,
    ModelInstall,
    ModelProfile,
    ModelSource,
    Project,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
    TurnCreationClaim,
    WorkflowDefinition,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from .ordered_planning import OrderedPlanCompiler, OrderedPlanConfirmationRequired
from .processes import ProcessSupervisor
from .profile_service import AUTO_PROFILE_ID
from .progress import completed_progress, update_job_progress
from .routing import ModalityRouter, RouteConfirmationRequired
from .scheduler import ResourceScheduler
from .schemas import (
    GenerationOffer,
    MessageOut,
    OrderedWorkIntent,
    RunOut,
    TurnAccepted,
    TurnRequest,
    WorkerStatus,
)
from .settings_registry import (
    compatible_stored_settings,
    resolve_generation_settings,
    validate_settings,
    workflow_settings,
)
from .vision import PreparedVisualContext, VisionContextService
from .work_plans import plan_status_summary, refresh_plan_status

logger = logging.getLogger(__name__)

IDEMPOTENCY_CLAIM_WAIT_SECONDS = 120.0
MAX_PENDING_WORK_PER_CHAT = 32
PENDING_OUTPUT_REFERENCE = re.compile(
    r"\b(?:"
    r"(?:that|this|it|its)(?:\s+(?:image|video|answer|response|story|result|output))?"
    r"|(?:previous|last|above|earlier)\s+(?:image|video|answer|response|story|result|output)"
    r"|based\s+on\s+(?:that|this|it|the\s+(?:previous|last|above|earlier|story|response))"
    r")\b",
    re.IGNORECASE,
)


class ResponseRevisionConflict(ValueError):
    """A stable response cannot accept the requested revision transition."""


SELECTION_TERM_ALIASES = {
    "animation": "video",
    "animations": "video",
    "artwork": "image",
    "cinematic": "video",
    "coding": "code",
    "debugging": "debug",
    "developer": "code",
    "development": "code",
    "draw": "image",
    "drawing": "image",
    "fiction": "writing",
    "illustration": "image",
    "illustrations": "image",
    "images": "image",
    "motion": "video",
    "narrative": "writing",
    "photo": "image",
    "photos": "image",
    "photography": "image",
    "picture": "image",
    "pictures": "image",
    "programming": "code",
    "prose": "writing",
    "software": "code",
    "stories": "writing",
    "story": "writing",
    "storytelling": "writing",
    "summarization": "summarize",
    "summary": "summarize",
    "translation": "translate",
    "translator": "translate",
    "troubleshooting": "debug",
    "videos": "video",
    "write": "writing",
    "writer": "writing",
}


def _preview_media_type(content: bytes) -> str:
    if content.startswith(b"\x89PNG"):
        return "image/png"
    if content.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    if content.lstrip().startswith(b"<svg"):
        return "image/svg+xml"
    return "application/octet-stream"


class ConversationOrchestrator:
    def __init__(
        self,
        engines: EngineRegistry,
        artifacts: ArtifactStore,
        events: EventBroker,
        scheduler: ResourceScheduler,
        processes: ProcessSupervisor,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        persistence_scope: str = "durable",
        scope_id: str | None = None,
    ) -> None:
        self.engines = engines
        self.artifacts = artifacts
        self.events = events
        self.scheduler = scheduler
        self.processes = processes
        self.session_factory = session_factory
        self.persistence_scope = persistence_scope
        self.scope_id = scope_id
        self.router = ModalityRouter()
        self.vision = VisionContextService(engines.settings, artifacts)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._media_restart_task: asyncio.Task[None] | None = None
        self._media_restart_after_chat_activity = False
        self._step_prewarm_plan_id: str | None = None
        self._step_prewarm_task: asyncio.Task[None] | None = None
        self._chat_guards: dict[str, asyncio.Lock] = {}
        self._chat_planner_ready = asyncio.Event()
        self._chat_planner_ready.set()
        self._admission_open = True

    def recover_interrupted(self) -> None:
        queued: list[tuple[str, str]] = []
        with self.session_factory() as session:
            # Turn-creation claims only live while one API process is planning.
            session.execute(delete(TurnCreationClaim))
            queued_jobs = session.scalars(
                select(Job)
                .where(Job.status == JobStatus.QUEUED.value)
                .order_by(
                    Job.enqueued_at,
                    Job.queue_ticket,
                    Job.created_at,
                    Job.id,
                )
            ).all()
            for job in queued_jobs:
                job.claim_owner = None
                job.claim_expires_at = None
                job.heartbeat_at = None
                update_job_progress(
                    job,
                    stage="queued",
                    queue_resource=job.queue_resource,
                    indeterminate=True,
                )
                if job.run_id:
                    run = session.get(Run, job.run_id)
                    if run:
                        run.status = RunStatus.QUEUED.value
                        self._set_work_status(session, run, JobStatus.QUEUED.value)
                        queued.append((job.id, run.id))

            # A running backend operation cannot be proven safe to replay after
            # its process disappears. Preserve partial output and interrupt it.
            running_jobs = session.scalars(
                select(Job).where(Job.status == JobStatus.RUNNING.value)
            ).all()
            for job in running_jobs:
                job.status = JobStatus.INTERRUPTED.value
                job.error = "The application restarted before this job completed."
                job.completed_at = utcnow()
                job.claim_owner = None
                job.claim_expires_at = None
                job.heartbeat_at = None
                update_job_progress(
                    job,
                    stage="interrupted by application restart",
                    queue_resource=job.queue_resource,
                    indeterminate=True,
                )
                if job.run_id:
                    run = session.get(Run, job.run_id)
                    if run and run.status not in {
                        RunStatus.COMPLETE.value,
                        RunStatus.FAILED.value,
                        RunStatus.CANCELLED.value,
                    }:
                        run.status = RunStatus.FAILED.value
                        run.error = job.error
                        run.completed_at = utcnow()
                        self._set_work_status(
                            session,
                            run,
                            JobStatus.INTERRUPTED.value,
                            error=job.error,
                        )
                        message = session.get(Message, run.assistant_message_id)
                        if message:
                            message.status = MessageStatus.FAILED.value
                            preview_ids = self._temporary_preview_ids(message)
                            for part in list(message.parts):
                                if part.type == PartType.PROGRESS.value or (
                                    part.artifact_id and part.metadata_json.get("preview")
                                ):
                                    message.parts.remove(part)
                            session.flush()
                            error_part = next(
                                (
                                    part
                                    for part in message.parts
                                    if part.type == PartType.ERROR.value
                                ),
                                None,
                            )
                            if error_part:
                                error_part.text = job.error
                            else:
                                message.parts.append(
                                    MessagePart(
                                        position=max(
                                            (part.position for part in message.parts),
                                            default=-1,
                                        )
                                        + 1,
                                        type=PartType.ERROR.value,
                                        text=job.error,
                                    )
                                )
                            for artifact_id in preview_ids:
                                self.artifacts.delete_temporary_preview(session, artifact_id)
            session.commit()
        for job_id, run_id in queued:
            self.start(job_id, run_id)

    async def create_turn(
        self,
        session: Session,
        chat_id: str,
        request: TurnRequest,
        *,
        use_explicit_parent: bool = False,
        replacement_message_id: str | None = None,
        source_action: str = "send",
        inherited_image_edit_strength: dict[str, Any] | None = None,
    ) -> TurnAccepted:
        if not self._admission_open:
            raise RuntimeError(
                "This conversation service is shutting down and cannot accept new work."
            )
        async with self.chat_guard(chat_id):
            return await self._create_turn(
                session,
                chat_id,
                request,
                use_explicit_parent=use_explicit_parent,
                replacement_message_id=replacement_message_id,
                source_action=source_action,
                inherited_image_edit_strength=inherited_image_edit_strength,
            )

    async def _create_turn(
        self,
        session: Session,
        chat_id: str,
        request: TurnRequest,
        *,
        use_explicit_parent: bool = False,
        replacement_message_id: str | None = None,
        source_action: str = "send",
        inherited_image_edit_strength: dict[str, Any] | None = None,
    ) -> TurnAccepted:
        # Never resolve an idempotency key until its URL-scoped chat has been
        # validated. Otherwise a key from one chat could disclose another
        # chat's run, including when the requested chat never existed.
        chat = session.get(Chat, chat_id)
        if not chat:
            raise LookupError("chat not found")
        pending_count = session.scalar(
            select(func.count(Job.id))
            .join(Run, Job.run_id == Run.id)
            .where(
                Run.chat_id == chat_id,
                Job.status.in_(
                    [
                        JobStatus.QUEUED.value,
                        JobStatus.RUNNING.value,
                        JobStatus.PAUSED.value,
                    ]
                ),
            )
        )
        if (pending_count or 0) >= MAX_PENDING_WORK_PER_CHAT:
            raise ValueError(
                f"This chat already has {MAX_PENDING_WORK_PER_CHAT} pending items. "
                "Cancel one or wait for work to finish before sending another."
            )

        key = request.idempotency_key
        if key is None:
            return await self._create_new_turn(
                session,
                chat_id,
                request,
                use_explicit_parent=use_explicit_parent,
                replacement_message_id=replacement_message_id,
                source_action=source_action,
                inherited_image_edit_strength=inherited_image_edit_strength,
            )

        owner_token, replay = await self._claim_or_replay_turn(
            session,
            chat_id,
            key,
        )
        if replay:
            return replay
        if not owner_token:
            raise TimeoutError("turn idempotency claim could not be acquired")

        try:
            # Revalidate after the claim commit. This also refreshes state that
            # may have changed while a competing request held the claim.
            session.expire_all()
            if not session.get(Chat, chat_id):
                raise LookupError("chat not found")
            existing = self._idempotent_run(session, chat_id, key)
            if existing:
                return self._accepted_for_run(session, existing)
            return await self._create_new_turn(
                session,
                chat_id,
                request,
                use_explicit_parent=use_explicit_parent,
                replacement_message_id=replacement_message_id,
                source_action=source_action,
                inherited_image_edit_strength=inherited_image_edit_strength,
            )
        finally:
            self._release_turn_claim(session, chat_id, key, owner_token)

    async def _claim_or_replay_turn(
        self,
        session: Session,
        chat_id: str,
        idempotency_key: str,
    ) -> tuple[str | None, TurnAccepted | None]:
        deadline = asyncio.get_running_loop().time() + IDEMPOTENCY_CLAIM_WAIT_SECONDS
        while True:
            session.expire_all()
            if not session.get(Chat, chat_id):
                raise LookupError("chat not found")
            existing = self._idempotent_run(session, chat_id, idempotency_key)
            if existing:
                return None, self._accepted_for_run(session, existing)

            owner_token = secrets.token_hex(24)
            claim = TurnCreationClaim(
                chat_id=chat_id,
                idempotency_key=idempotency_key,
                owner_token=owner_token,
            )
            session.add(claim)
            try:
                session.commit()
                return owner_token, None
            except IntegrityError:
                session.rollback()
                # A chat may be deleted between the ownership check and claim
                # insertion. Distinguish that from a legitimate duplicate.
                if not session.get(Chat, chat_id):
                    raise LookupError("chat not found") from None

            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    "another request is still creating this turn; retry with the same key"
                )
            await asyncio.sleep(0.01)

    @staticmethod
    def _idempotent_run(
        session: Session,
        chat_id: str,
        idempotency_key: str,
    ) -> Run | None:
        planned = session.scalar(
            select(Run)
            .join(WorkStep, WorkStep.id == Run.work_step_id)
            .join(WorkPlan, WorkPlan.id == WorkStep.plan_id)
            .where(
                WorkPlan.chat_id == chat_id,
                WorkPlan.idempotency_key == idempotency_key,
            )
            .order_by(WorkStep.ordinal)
            .limit(1)
        )
        if planned:
            return planned
        # Legacy rows created before work plans remain replayable.
        return session.scalar(
            select(Run).where(
                Run.chat_id == chat_id,
                Run.idempotency_key == idempotency_key,
                Run.work_plan_id.is_(None),
            )
        )

    @staticmethod
    def _release_turn_claim(
        session: Session,
        chat_id: str,
        idempotency_key: str,
        owner_token: str,
    ) -> None:
        try:
            # If turn construction failed after flushing messages, discard that
            # partial graph before releasing ownership to a retry.
            session.rollback()
            session.execute(
                delete(TurnCreationClaim).where(
                    TurnCreationClaim.chat_id == chat_id,
                    TurnCreationClaim.idempotency_key == idempotency_key,
                    TurnCreationClaim.owner_token == owner_token,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            logger.exception(
                "Failed to release turn idempotency claim for chat %s",
                chat_id,
            )

    async def _create_new_turn(
        self,
        session: Session,
        chat_id: str,
        request: TurnRequest,
        *,
        use_explicit_parent: bool = False,
        replacement_message_id: str | None = None,
        source_action: str = "send",
        inherited_image_edit_strength: dict[str, Any] | None = None,
    ) -> TurnAccepted:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise LookupError("chat not found")
        pending_count = session.scalar(
            select(func.count(Job.id))
            .join(Run, Job.run_id == Run.id)
            .where(
                Run.chat_id == chat_id,
                Job.status.in_(
                    [
                        JobStatus.QUEUED.value,
                        JobStatus.RUNNING.value,
                        JobStatus.PAUSED.value,
                    ]
                ),
            )
        )
        replacement_message = (
            session.get(Message, replacement_message_id) if replacement_message_id else None
        )
        if replacement_message_id and (
            not replacement_message
            or replacement_message.chat_id != chat_id
            or replacement_message.role != MessageRole.ASSISTANT.value
            or not replacement_message.transcript_visible
        ):
            raise LookupError("replacement assistant message not found in this chat")
        if replacement_message:
            if replacement_message.status != MessageStatus.COMPLETE.value:
                raise ResponseRevisionConflict(
                    "only a completed visible response can be regenerated"
                )
            pending_revision = session.scalar(
                select(ResponseRevision.id).where(
                    ResponseRevision.message_id == replacement_message.id,
                    ResponseRevision.status == MessageStatus.PENDING.value,
                )
            )
            if pending_revision:
                raise ResponseRevisionConflict("this response is already being regenerated")
            self._ensure_response_revision(session, replacement_message)
        parent_message_id = request.parent_message_id
        if parent_message_id:
            parent = session.get(Message, parent_message_id)
            if not parent or parent.chat_id != chat_id:
                raise LookupError("parent message not found in this chat")
        elif not use_explicit_parent:
            parent_message_id = chat.active_head_message_id
            if not parent_message_id:
                parent_message_id = session.scalar(
                    select(Message.id)
                    .where(Message.chat_id == chat_id)
                    .order_by(
                        Message.updated_at.desc(), Message.created_at.desc(), Message.id.desc()
                    )
                    .limit(1)
                )
        pending_dependency_step_id = self._pending_parent_step_id(
            session,
            chat_id,
            parent_message_id,
        )
        references_pending_output = bool(
            pending_dependency_step_id and PENDING_OUTPUT_REFERENCE.search(request.text)
        )
        context_head_message_id = (
            parent_message_id
            if references_pending_output
            else self._latest_completed_context_head(
                session,
                chat_id,
                parent_message_id,
            )
        )
        explicit_artifacts: dict[str, Artifact] = {}
        for artifact_id in request.input_artifact_ids:
            artifact = session.get(Artifact, artifact_id)
            if not artifact:
                raise LookupError(f"input artifact not found: {artifact_id}")
            explicit_artifacts[artifact_id] = artifact

        accepted_offer = None
        if (
            replacement_message is None
            and not explicit_artifacts
            and is_explicit_generation_assent(request.text)
        ):
            parent_message = session.get(Message, parent_message_id) if parent_message_id else None
            accepted_offer = self._generation_offer_for_message(parent_message)
        if accepted_offer:
            request = request.model_copy(
                update={"settings": {}, "ordered_settings": {}, "output_count": None}
            )

        mode = request.mode or RoutingMode(chat.routing_mode)
        ordered_intent = None
        if accepted_offer and len(accepted_offer.items) > 1:
            ordered_intent = ordered_intent_for_offer(accepted_offer)
        elif replacement_message is None:
            ordered_intent = OrderedPlanCompiler.deterministic(
                request.text,
                mode,
                has_media_input=bool(explicit_artifacts),
            )
            if (
                ordered_intent is None
                and mode == RoutingMode.AUTO
                and await self._chat_planner_available()
            ):
                ordered_intent = await OrderedPlanCompiler.plan_with_model(
                    adapter=self.engines.chat,
                    text=request.text,
                    mode=mode,
                    conversation=self._routing_context(
                        session,
                        chat,
                        context_head_message_id,
                    ),
                    has_media_input=bool(explicit_artifacts),
                )
        if accepted_offer and ordered_intent and not request.confirm_media:
            raise OrderedPlanConfirmationRequired(ordered_intent)
        if (
            ordered_intent
            and chat.confirm_uncertain_media
            and (ordered_intent.requires_confirmation or ordered_intent.confidence < 0.8)
            and not request.confirm_media
        ):
            raise OrderedPlanConfirmationRequired(ordered_intent)
        if ordered_intent:
            return await self._create_ordered_turn(
                session,
                chat,
                request,
                ordered_intent,
                parent_message_id=parent_message_id,
                context_head_message_id=context_head_message_id,
                pending_dependency_step_id=pending_dependency_step_id,
                references_pending_output=references_pending_output,
                explicit_artifacts=explicit_artifacts,
                pending_count=pending_count or 0,
                source_action=source_action,
            )
        prior_image, prior_image_prompt = self._latest_image_context(
            session,
            chat.id,
            context_head_message_id,
        )
        has_prior_image = prior_image is not None
        routing_context = self._routing_context(session, chat, context_head_message_id)
        if accepted_offer:
            plan = routing_plan_for_offer(accepted_offer)
            if not request.confirm_media:
                raise RouteConfirmationRequired(plan)
        else:
            planner_available = (
                await self._chat_planner_available() if mode == RoutingMode.AUTO else True
            )
            if planner_available:
                plan = await self.router.plan_with_model(
                    adapter=self.engines.chat,
                    text=request.text,
                    mode=mode,
                    input_artifact_ids=request.input_artifact_ids,
                    has_prior_image=has_prior_image,
                    conversation=routing_context,
                )
            else:
                plan = self.router.plan(
                    text=request.text,
                    mode=mode,
                    input_artifact_ids=request.input_artifact_ids,
                    has_prior_image=has_prior_image,
                    conversation=routing_context,
                )
        if (
            mode == RoutingMode.AUTO
            and chat.confirm_uncertain_media
            and plan.operation != Operation.TEXT
            and plan.confidence < 0.8
            and not request.confirm_media
        ):
            raise RouteConfirmationRequired(plan)
        resolved_input_ids = list(dict.fromkeys(request.input_artifact_ids))
        prior_prompt: str | None = None
        if plan.operation in {Operation.IMAGE_TO_IMAGE, Operation.IMAGE_TO_VIDEO}:
            if not resolved_input_ids and prior_image:
                resolved_input_ids.append(prior_image)
                prior_prompt = prior_image_prompt
            elif resolved_input_ids:
                prior_prompt = self._latest_media_prompt(
                    session,
                    chat.id,
                    context_head_message_id,
                )
            if prior_prompt:
                plan.standalone_prompt = f"{prior_prompt}. Follow-up instruction: {request.text}"
        plan.input_artifact_ids = resolved_input_ids

        profile, model_selection = self._profile_for_operation(
            session,
            chat,
            plan.operation,
            f"{request.text}\n{plan.standalone_prompt}",
        )
        profile_id = profile.id if profile else None
        vision_profile = (
            self._vision_profile_for_chat(session, chat, profile)
            if plan.operation == Operation.TEXT
            else None
        )
        vision_profile_id = vision_profile.id if vision_profile else None
        workflow_revision = self._workflow_for_operation(
            session,
            plan.operation,
            project_id=chat.project_id,
            model_install_id=profile.model_install_id if profile else None,
        )
        if plan.operation != Operation.TEXT and not workflow_revision:
            semantic_fallback = {
                Operation.IMAGE_TO_IMAGE: Operation.TEXT_TO_IMAGE,
                Operation.IMAGE_TO_VIDEO: Operation.TEXT_TO_VIDEO,
            }.get(plan.operation)
            if semantic_fallback and not request.input_artifact_ids and prior_prompt:
                plan.operation = semantic_fallback
                plan.input_artifact_ids = []
                resolved_input_ids = []
                workflow_revision = self._workflow_for_operation(
                    session,
                    plan.operation,
                    project_id=chat.project_id,
                    model_install_id=profile.model_install_id if profile else None,
                )
            if not workflow_revision:
                raise ValueError(
                    "No ready workflow matches the active media engine. Install a supported "
                    "image or video model and LM Atelier will configure it automatically."
                )
        role = self._role_for_operation(plan.operation)
        engine = (
            profile.engine if profile else workflow_revision.engine if workflow_revision else None
        )
        fields = workflow_settings(
            await self.engines.settings_for_role(role, engine=engine),
            workflow_revision.input_schema_json if workflow_revision else None,
        )
        request_fields = [field for field in fields if field.scope != "load"]
        request_settings = validate_settings(request.settings, request_fields)
        project = session.get(Project, chat.project_id) if chat.project_id else None
        default_preset = self._default_preset(session, plan.operation)
        project_preset = self._bound_preset(session, project, role)
        chat_preset = self._bound_preset(session, chat, role)
        preset_layers = [
            (scope, preset, compatible_stored_settings(preset.settings_json, request_fields))
            for scope, preset in (
                ("default", default_preset),
                ("project", project_preset),
                ("chat", chat_preset),
            )
            if preset
        ]
        effective_settings = resolve_generation_settings(
            fields,
            request_fields=request_fields,
            profile_defaults=(
                profile.load_settings_json if profile else {},
                profile.request_settings_json if profile else {},
                default_preset.settings_json if default_preset else {},
            ),
            project_defaults=(
                project_preset.settings_json if project_preset else {},
                self._scoped_generation_settings(project, role),
            ),
            chat_defaults=(
                chat_preset.settings_json if chat_preset else {},
                self._scoped_generation_settings(chat, role),
            ),
            turn_overrides=request_settings,
        )
        lora_selection = None
        lora_setting_layers = (
            profile.load_settings_json if profile else {},
            profile.request_settings_json if profile else {},
            default_preset.settings_json if default_preset else {},
            project_preset.settings_json if project_preset else {},
            self._scoped_generation_settings(project, role),
            chat_preset.settings_json if chat_preset else {},
            self._scoped_generation_settings(chat, role),
            request_settings,
        )
        if (
            plan.operation in {Operation.TEXT_TO_IMAGE, Operation.IMAGE_TO_IMAGE}
            and workflow_revision
            and not any("loras" in layer for layer in lora_setting_layers)
        ):
            lora_selection = select_automatic_lora_stack(
                session,
                workflow_revision,
                plan.standalone_prompt if accepted_offer else request.text,
            )
            if lora_selection.settings:
                effective_settings["loras"] = lora_selection.settings
        image_edit_strength = resolve_image_edit_strength(
            plan.operation,
            plan.standalone_prompt if accepted_offer else request.text,
            fields,
            effective_settings,
            (
                (EditSettingSource.PROFILE_LOAD, profile.load_settings_json if profile else {}),
                (
                    EditSettingSource.PROFILE_REQUEST,
                    profile.request_settings_json if profile else {},
                ),
                (
                    EditSettingSource.DEFAULT_PRESET,
                    default_preset.settings_json if default_preset else {},
                ),
                (
                    EditSettingSource.PROJECT_PRESET,
                    project_preset.settings_json if project_preset else {},
                ),
                (EditSettingSource.PROJECT, self._scoped_generation_settings(project, role)),
                (
                    EditSettingSource.CHAT_PRESET,
                    chat_preset.settings_json if chat_preset else {},
                ),
                (EditSettingSource.CHAT, self._scoped_generation_settings(chat, role)),
                (EditSettingSource.TURN, request_settings),
            ),
            inherited_auto=inherited_image_edit_strength,
            workflow_schema=(workflow_revision.input_schema_json if workflow_revision else None),
        )
        lora_resolution = None
        if plan.operation != Operation.TEXT and effective_settings.get("loras"):
            if not workflow_revision:
                raise ValueError("LoRA settings require a selected media workflow.")
            lora_resolution = resolve_lora_stack(
                session,
                workflow_revision,
                effective_settings["loras"],
            )
            effective_settings["loras"] = lora_resolution.settings
        effective_preset = preset_layers[-1] if preset_layers else None
        if plan.operation != Operation.TEXT and effective_settings.get("seed") == -1:
            effective_settings["seed"] = secrets.randbelow(2_147_483_648)
        configured_batch_size = effective_settings.get("batch_size", 1)
        try:
            configured_output_count = max(1, int(configured_batch_size))
        except (TypeError, ValueError):
            configured_output_count = 1
        output_count = (
            1
            if plan.operation == Operation.TEXT or replacement_message is not None
            else max(
                request.output_count or 1,
                plan.output_count,
                configured_output_count,
            )
        )
        per_output_prompt = ModalityRouter.per_output_media_prompt(
            plan.standalone_prompt,
            plan.operation,
            output_count,
        )
        if output_count > self.engines.settings.max_media_outputs_per_plan:
            raise ValueError(
                f"A media request can create at most "
                f"{self.engines.settings.max_media_outputs_per_plan} outputs."
            )
        if (pending_count or 0) + output_count > MAX_PENDING_WORK_PER_CHAT:
            raise ValueError(
                f"This request would exceed the limit of "
                f"{MAX_PENDING_WORK_PER_CHAT} pending items in one chat."
            )
        if output_count > 1 and "batch_size" in effective_settings:
            # Each planned output owns its own lifecycle. Prevent an engine-native
            # batch from multiplying the visible output slots a second time.
            effective_settings["batch_size"] = 1
        generation_estimate = (
            self._video_estimate(effective_settings) if "video" in plan.operation.value else None
        )
        media_plan_estimate = (
            self._media_plan_estimate(plan.operation, effective_settings, output_count)
            if plan.operation != Operation.TEXT
            else None
        )
        if media_plan_estimate:
            if media_plan_estimate["work_units"] > self.engines.settings.max_media_plan_work_units:
                raise ValueError(
                    "This media request is too large to queue safely. "
                    "Reduce the output count, resolution, frames, or steps."
                )
            if (
                media_plan_estimate["estimated_bytes"]
                > self.engines.settings.max_media_plan_estimated_bytes
            ):
                raise ValueError(
                    "This media request has an unsafe storage estimate. "
                    "Reduce the output count, resolution, or duration."
                )
            artifact_root = self.artifacts.root
            artifact_root.mkdir(parents=True, exist_ok=True)
            available_bytes = shutil.disk_usage(artifact_root).free
            if media_plan_estimate["estimated_bytes"] > available_bytes:
                raise ValueError("There is not enough free storage to queue this media request.")
            media_plan_estimate["available_bytes_at_admission"] = available_bytes
            plan.parameter_overrides = {
                **plan.parameter_overrides,
                "_media_plan_estimate": media_plan_estimate,
            }
        if generation_estimate:
            plan.parameter_overrides = {
                **plan.parameter_overrides,
                "_generation_estimate": generation_estimate,
            }
        if (
            mode == RoutingMode.AUTO
            and chat.confirm_uncertain_media
            and generation_estimate
            and self.engines.settings.video_confirmation_work_units > 0
            and generation_estimate["work_units"]
            >= self.engines.settings.video_confirmation_work_units
            and not request.confirm_media
        ):
            raise RouteConfirmationRequired(plan)

        input_parts: list[MessagePart] = [
            MessagePart(position=0, type=PartType.TEXT.value, text=request.text)
        ]
        explicit_ids = set(explicit_artifacts)
        for artifact_id in resolved_input_ids:
            artifact = explicit_artifacts.get(artifact_id) or session.get(Artifact, artifact_id)
            if not artifact:
                # The ancestor reference was resolved in this transaction, so this
                # guard only protects against corrupt legacy rows.
                raise LookupError(f"input artifact not found: {artifact_id}")
            input_parts.append(
                MessagePart(
                    position=len(input_parts),
                    type=self._input_part_type(artifact),
                    artifact_id=artifact.id,
                    metadata_json={
                        "input_reference": True,
                        "input_reference_source": (
                            "explicit" if artifact.id in explicit_ids else "ancestor"
                        ),
                    },
                )
            )
        user_message = Message(
            chat_id=chat.id,
            parent_id=parent_message_id,
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
            transcript_visible=replacement_message is None,
            parts=input_parts,
        )
        assistant_messages = [
            Message(
                chat_id=chat.id,
                parent_id=None,
                role=MessageRole.ASSISTANT.value,
                status=MessageStatus.PENDING.value,
                transcript_visible=replacement_message is None,
                parts=self._initial_output_parts(plan.operation, ordinal, output_count),
            )
            for ordinal in range(1, output_count + 1)
        ]
        session.add_all([user_message, *assistant_messages])
        session.flush()
        previous_message_id = user_message.id
        for assistant_message in assistant_messages:
            assistant_message.parent_id = previous_message_id
            previous_message_id = assistant_message.id
        if replacement_message is None:
            chat.active_head_message_id = assistant_messages[-1].id
        assistant_message = assistant_messages[0]

        model_provenance: dict[str, Any] | None = None
        if profile and profile.model_install_id:
            install = session.get(ModelInstall, profile.model_install_id)
            source = (
                session.get(ModelSource, install.source_id)
                if install and install.source_id
                else None
            )
            if install:
                model_provenance = {
                    "profile_id": profile.id,
                    "profile_name": profile.name,
                    "profile_use_case": profile.use_case,
                    "install_id": install.id,
                    "engine": install.engine,
                    "local_path": install.local_path,
                    "size_bytes": install.size_bytes,
                    "manifest": install.manifest_json,
                    "source": {
                        "provider": source.provider,
                        "remote_id": source.remote_id,
                        "revision": source.revision,
                        "metadata": source.metadata_json,
                    }
                    if source
                    else None,
                }
        workflow_provenance = (
            {
                "definition_id": workflow_revision.workflow_id,
                "revision_id": workflow_revision.id,
                "version": workflow_revision.version,
                "engine": workflow_revision.engine,
                "engine_version": workflow_revision.engine_version,
                "trusted": workflow_revision.trusted,
                "dependencies": workflow_revision.dependencies_json,
            }
            if workflow_revision
            else None
        )

        transcript_sequence = (
            session.scalar(
                select(WorkPlan.transcript_sequence)
                .where(WorkPlan.chat_id == chat.id)
                .order_by(WorkPlan.transcript_sequence.desc())
                .limit(1)
            )
            or 0
        ) + 1
        queue_class = "interactive_compute" if plan.operation == Operation.TEXT else "media_compute"
        work_plan = WorkPlan(
            chat_id=chat.id,
            idempotency_key=request.idempotency_key,
            source_action=source_action,
            persistence_scope=self.persistence_scope,
            status=JobStatus.QUEUED.value,
            context_head_message_id=context_head_message_id,
            transcript_sequence=transcript_sequence,
            priority=10 if plan.operation == Operation.TEXT else 0,
            planner_version=("media-outputs-v1" if output_count > 1 else "legacy-turn-v1"),
            failure_policy="stop_dependents",
            summary_json={
                "operation": plan.operation.value,
                "step_count": output_count,
                "output_count": output_count,
                "source_action": source_action,
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
                "assistant_message_ids": [message.id for message in assistant_messages],
                "dependency_step_ids": (
                    [pending_dependency_step_id] if references_pending_output else []
                ),
                "media_plan_estimate": media_plan_estimate,
                "status_counts": {"queued": output_count},
            },
        )
        input_bindings: list[dict[str, Any]] = [
            {
                "type": "explicit_artifact"
                if artifact_id in explicit_ids
                else "response_revision.artifact",
                "artifact_id": artifact_id,
            }
            for artifact_id in resolved_input_ids
        ]
        if context_head_message_id:
            input_bindings.insert(
                0,
                {
                    "type": "context_text",
                    "context_head_message_id": context_head_message_id,
                },
            )
        output_type = (
            "text"
            if plan.operation == Operation.TEXT
            else "video"
            if plan.operation in {Operation.TEXT_TO_VIDEO, Operation.IMAGE_TO_VIDEO}
            else "image"
        )
        session.add(work_plan)
        session.flush()
        work_steps: list[WorkStep] = []
        runs: list[Run] = []
        jobs: list[Job] = []
        base_seed = effective_settings.get("seed")
        for ordinal, output_message in enumerate(assistant_messages, start=1):
            output_settings = copy.deepcopy(effective_settings)
            if (
                output_count > 1
                and isinstance(base_seed, int)
                and not isinstance(base_seed, bool)
                and base_seed >= 0
            ):
                output_settings["seed"] = (base_seed + ordinal - 1) % 2_147_483_648
            output_slot = "response" if output_count == 1 else f"output-{ordinal}"
            work_step = WorkStep(
                plan=work_plan,
                ordinal=ordinal,
                display_group="media_outputs" if output_count > 1 else None,
                operation=plan.operation.value,
                status=JobStatus.QUEUED.value,
                prompt=per_output_prompt,
                profile_id=profile_id,
                workflow_revision_id=workflow_revision.id if workflow_revision else None,
                settings_json=output_settings,
                input_bindings_json=copy.deepcopy(input_bindings),
                output_contract_json=[
                    {
                        "slot": output_slot,
                        "type": output_type,
                        "index": ordinal,
                        "count": output_count,
                    }
                ],
                queue_class=queue_class,
            )
            session.add(work_step)
            session.flush()
            if references_pending_output and pending_dependency_step_id:
                session.add(
                    WorkStepDependency(
                        step_id=work_step.id,
                        depends_on_step_id=pending_dependency_step_id,
                    )
                )
            provenance: dict[str, Any] = {
                "routing": plan.model_dump(mode="json"),
                "model_selection": model_selection,
                "input_artifact_ids": resolved_input_ids,
                "model": model_provenance,
                "preset": (
                    {
                        "id": effective_preset[1].id,
                        "name": effective_preset[1].name,
                        "role": effective_preset[1].role,
                        "settings": effective_preset[2],
                    }
                    if effective_preset
                    else None
                ),
                "preset_layers": [
                    {
                        "scope": scope,
                        "id": preset.id,
                        "name": preset.name,
                        "role": preset.role,
                        "settings": settings,
                    }
                    for scope, preset, settings in preset_layers
                ],
                "workflow": workflow_provenance,
                "resolved_settings": output_settings,
                "generation_estimate": generation_estimate,
                "media_plan_estimate": media_plan_estimate,
                "media_output": {
                    "index": ordinal,
                    "count": output_count,
                    "slot": output_slot,
                },
                "image_edit": self._image_edit_provenance(image_edit_strength),
                "auxiliary_assets": (
                    {
                        "lora_stack": lora_resolution.provenance,
                        "selection": (
                            lora_selection.provenance if lora_selection else {"mode": "explicit"}
                        ),
                        "graph_transform_version": LORA_GRAPH_TRANSFORM_VERSION,
                        "effective_graph_sha256": lora_resolution.graph_sha256,
                    }
                    if lora_resolution
                    else None
                ),
                **(
                    {
                        "response_replacement": {
                            "message_id": replacement_message.id,
                            "source_user_message_id": replacement_message.parent_id,
                        }
                    }
                    if replacement_message
                    else {}
                ),
            }
            run = Run(
                idempotency_key=request.idempotency_key if ordinal == 1 else None,
                chat_id=chat.id,
                user_message_id=user_message.id,
                assistant_message_id=output_message.id,
                work_plan_id=work_plan.id,
                work_step_id=work_step.id,
                operation=plan.operation.value,
                status=RunStatus.QUEUED.value,
                standalone_prompt=per_output_prompt,
                profile_id=profile_id,
                vision_profile_id=vision_profile_id,
                workflow_revision_id=workflow_revision.id if workflow_revision else None,
                settings_json=output_settings,
                provenance_json=provenance,
            )
            session.add(run)
            session.flush()
            work_step.run_id = run.id
            if replacement_message:
                latest_sequence = session.scalar(
                    select(ResponseRevision.sequence)
                    .where(ResponseRevision.message_id == replacement_message.id)
                    .order_by(ResponseRevision.sequence.desc())
                    .limit(1)
                )
                revision = ResponseRevision(
                    message_id=replacement_message.id,
                    run_id=run.id,
                    sequence=(latest_sequence or 0) + 1,
                    status=MessageStatus.PENDING.value,
                )
                session.add(revision)
                try:
                    session.flush()
                except IntegrityError as exc:
                    session.rollback()
                    raise ResponseRevisionConflict(
                        "this response is already being regenerated"
                    ) from exc
                run.provenance_json = {
                    **run.provenance_json,
                    "response_replacement": {
                        **run.provenance_json["response_replacement"],
                        "revision_id": revision.id,
                    },
                }
            job = Job(
                kind=self._job_kind(plan.operation).value,
                status=JobStatus.QUEUED.value,
                run_id=run.id,
                work_plan_id=work_plan.id,
                work_step_id=work_step.id,
                progress=0,
                phase="queued",
                queue_resource=queue_class,
                queue_group="primary",
                queue_priority=work_plan.priority,
                queue_ticket=f"{transcript_sequence:020d}:{ordinal:04d}:{run.id}",
                enqueued_at=utcnow(),
                payload_json={
                    "operation": plan.operation.value,
                    "output_index": ordinal,
                    "output_count": output_count,
                },
            )
            update_job_progress(
                job,
                stage="queued",
                queue_resource=queue_class,
                queue_position=ordinal - 1,
                queue_length=output_count,
                indeterminate=True,
            )
            session.add(job)
            work_steps.append(work_step)
            runs.append(run)
            jobs.append(job)
        work_plan.summary_json = {
            **work_plan.summary_json,
            "step_ids": [step.id for step in work_steps],
            "run_ids": [run.id for run in runs],
            "job_ids": [job.id for job in jobs],
        }
        if chat.title == "New chat":
            chat.title = request.text.strip().replace("\n", " ")[:72] or "New chat"
        session.commit()
        accepted = self._accepted_for_run(session, runs[0])
        await self.events.publish(
            "work_plan.created",
            work_plan.id,
            {
                "plan_id": work_plan.id,
                "step_id": work_steps[0].id,
                "run_id": runs[0].id,
                "job_id": jobs[0].id,
                "step_ids": [step.id for step in work_steps],
                "run_ids": [run.id for run in runs],
                "job_ids": [job.id for job in jobs],
                "chat_id": chat.id,
            },
        )
        for queued_job, queued_run in zip(jobs, runs, strict=True):
            self.start(queued_job.id, queued_run.id)
        return accepted

    async def _create_ordered_turn(
        self,
        session: Session,
        chat: Chat,
        request: TurnRequest,
        intent: OrderedWorkIntent,
        *,
        parent_message_id: str | None,
        context_head_message_id: str | None,
        pending_dependency_step_id: str | None,
        references_pending_output: bool,
        explicit_artifacts: dict[str, Artifact],
        pending_count: int,
        source_action: str,
    ) -> TurnAccepted:
        intent = OrderedPlanCompiler.validate(intent)
        if pending_count + len(intent.steps) > MAX_PENDING_WORK_PER_CHAT:
            raise ValueError(
                f"This ordered request would exceed the limit of "
                f"{MAX_PENDING_WORK_PER_CHAT} pending items in one chat."
            )
        if request.output_count not in {None, 1}:
            raise ValueError(
                "A heterogeneous ordered plan cannot multiply all steps. "
                "Request variations in a separate media turn."
            )
        allowed_setting_roles = {"chat", "image", "video"}
        unknown_setting_roles = set(request.ordered_settings) - allowed_setting_roles
        if unknown_setting_roles:
            raise ValueError("Ordered settings contain an unsupported role.")
        if request.settings:
            raise ValueError(
                "Ordered plans accept role-specific ordered_settings, not shared settings."
            )

        explicit_ids = set(explicit_artifacts)
        first_step = intent.steps[0]
        first_explicit_images = [
            artifact.id
            for artifact in explicit_artifacts.values()
            if artifact.media_type.casefold().startswith("image/")
        ]
        if (
            first_step.mode in {"image", "video"}
            and explicit_artifacts
            and not first_explicit_images
        ):
            raise ValueError("The first media step requires an image-compatible input.")

        resolved_steps: list[dict[str, Any]] = []
        intent_by_id = {step.id: step for step in intent.steps}
        total_work_units = 0
        total_estimated_bytes = 0
        total_video_duration_seconds = 0.0
        project = session.get(Project, chat.project_id) if chat.project_id else None
        for index, step_intent in enumerate(intent.steps):
            artifact_source_modes = [
                intent_by_id[binding.source_step_id].mode
                for binding in step_intent.inputs
                if binding.kind == "artifact"
            ]
            if step_intent.mode == "text":
                operation = Operation.TEXT
            elif step_intent.mode == "image":
                operation = (
                    Operation.IMAGE_TO_IMAGE
                    if "image" in artifact_source_modes
                    or (index == 0 and bool(first_explicit_images))
                    else Operation.TEXT_TO_IMAGE
                )
            else:
                operation = (
                    Operation.IMAGE_TO_VIDEO
                    if "image" in artifact_source_modes
                    or (index == 0 and bool(first_explicit_images))
                    else Operation.TEXT_TO_VIDEO
                )

            profile, model_selection = self._profile_for_operation(
                session,
                chat,
                operation,
                step_intent.prompt,
            )
            if profile and profile.model_install_id:
                install = session.get(ModelInstall, profile.model_install_id)
                if not install or not install.active:
                    raise ValueError(
                        f"Ordered step {index + 1} selected a model that is not ready."
                    )
            profile_id = profile.id if profile else None
            vision_profile = (
                self._vision_profile_for_chat(session, chat, profile)
                if operation == Operation.TEXT
                and (
                    any(binding.kind == "artifact" for binding in step_intent.inputs)
                    or (index == 0 and bool(explicit_artifacts))
                )
                else None
            )
            workflow_revision = self._workflow_for_operation(
                session,
                operation,
                project_id=chat.project_id,
                model_install_id=profile.model_install_id if profile else None,
            )
            if operation != Operation.TEXT and not workflow_revision:
                raise ValueError(
                    f"No ready workflow can perform ordered step {index + 1} ({operation.value})."
                )
            if workflow_revision:
                if workflow_revision.engine == "comfyui" and not workflow_revision.trusted:
                    raise ValueError(f"Ordered step {index + 1} selected an untrusted workflow.")
                dependency_errors = custom_node_dependency_errors(
                    session,
                    workflow_revision.dependencies_json.get("custom_nodes"),
                )
                if dependency_errors:
                    raise ValueError(
                        f"Ordered step {index + 1} is not ready: " + "; ".join(dependency_errors)
                    )
            role = self._role_for_operation(operation)
            engine = (
                profile.engine
                if profile
                else workflow_revision.engine
                if workflow_revision
                else None
            )
            fields = workflow_settings(
                await self.engines.settings_for_role(role, engine=engine),
                workflow_revision.input_schema_json if workflow_revision else None,
            )
            request_fields = [field for field in fields if field.scope != "load"]
            step_overrides = validate_settings(
                request.ordered_settings.get(role, {}),
                request_fields,
            )
            default_preset = self._default_preset(session, operation)
            project_preset = self._bound_preset(session, project, role)
            chat_preset = self._bound_preset(session, chat, role)
            preset_layers = [
                (scope, preset, compatible_stored_settings(preset.settings_json, request_fields))
                for scope, preset in (
                    ("default", default_preset),
                    ("project", project_preset),
                    ("chat", chat_preset),
                )
                if preset
            ]
            effective_settings = resolve_generation_settings(
                fields,
                request_fields=request_fields,
                profile_defaults=(
                    profile.load_settings_json if profile else {},
                    profile.request_settings_json if profile else {},
                    default_preset.settings_json if default_preset else {},
                ),
                project_defaults=(
                    project_preset.settings_json if project_preset else {},
                    self._scoped_generation_settings(project, role),
                ),
                chat_defaults=(
                    chat_preset.settings_json if chat_preset else {},
                    self._scoped_generation_settings(chat, role),
                ),
                turn_overrides=step_overrides,
            )
            lora_selection = None
            lora_setting_layers = (
                profile.load_settings_json if profile else {},
                profile.request_settings_json if profile else {},
                default_preset.settings_json if default_preset else {},
                project_preset.settings_json if project_preset else {},
                self._scoped_generation_settings(project, role),
                chat_preset.settings_json if chat_preset else {},
                self._scoped_generation_settings(chat, role),
                step_overrides,
            )
            if (
                operation in {Operation.TEXT_TO_IMAGE, Operation.IMAGE_TO_IMAGE}
                and workflow_revision
                and not any("loras" in layer for layer in lora_setting_layers)
            ):
                lora_selection = select_automatic_lora_stack(
                    session,
                    workflow_revision,
                    step_intent.prompt,
                )
                if lora_selection.settings:
                    effective_settings["loras"] = lora_selection.settings
            image_edit_strength = resolve_image_edit_strength(
                operation,
                step_intent.prompt,
                fields,
                effective_settings,
                (
                    (
                        EditSettingSource.PROFILE_LOAD,
                        profile.load_settings_json if profile else {},
                    ),
                    (
                        EditSettingSource.PROFILE_REQUEST,
                        profile.request_settings_json if profile else {},
                    ),
                    (
                        EditSettingSource.DEFAULT_PRESET,
                        default_preset.settings_json if default_preset else {},
                    ),
                    (
                        EditSettingSource.PROJECT_PRESET,
                        project_preset.settings_json if project_preset else {},
                    ),
                    (EditSettingSource.PROJECT, self._scoped_generation_settings(project, role)),
                    (
                        EditSettingSource.CHAT_PRESET,
                        chat_preset.settings_json if chat_preset else {},
                    ),
                    (EditSettingSource.CHAT, self._scoped_generation_settings(chat, role)),
                    (EditSettingSource.TURN, step_overrides),
                ),
                workflow_schema=(
                    workflow_revision.input_schema_json if workflow_revision else None
                ),
            )
            lora_resolution = None
            if operation != Operation.TEXT and effective_settings.get("loras"):
                if not workflow_revision:
                    raise ValueError("LoRA settings require a selected media workflow.")
                lora_resolution = resolve_lora_stack(
                    session,
                    workflow_revision,
                    effective_settings["loras"],
                )
                effective_settings["loras"] = lora_resolution.settings
            if operation != Operation.TEXT and effective_settings.get("seed") == -1:
                effective_settings["seed"] = secrets.randbelow(2_147_483_648)
            estimate = (
                self._media_plan_estimate(operation, effective_settings, 1)
                if operation != Operation.TEXT
                else None
            )
            generation_estimate = (
                self._video_estimate(effective_settings) if "video" in operation.value else None
            )
            if estimate:
                total_work_units += estimate["work_units"]
                total_estimated_bytes += estimate["estimated_bytes"]
            if generation_estimate:
                total_video_duration_seconds += float(generation_estimate["duration_seconds"])
            resolved_steps.append(
                {
                    "intent": step_intent,
                    "operation": operation,
                    "profile": profile,
                    "profile_id": profile_id,
                    "vision_profile_id": vision_profile.id if vision_profile else None,
                    "workflow": workflow_revision,
                    "role": role,
                    "settings": effective_settings,
                    "model_selection": model_selection,
                    "image_edit_strength": image_edit_strength,
                    "preset_layers": preset_layers,
                    "effective_preset": preset_layers[-1] if preset_layers else None,
                    "estimate": estimate,
                    "generation_estimate": generation_estimate,
                    "lora_resolution": lora_resolution,
                    "lora_selection": lora_selection,
                }
            )

        if total_work_units > self.engines.settings.max_media_plan_work_units:
            raise ValueError(
                "This ordered plan is too large to queue safely. "
                "Reduce its media steps, resolution, frames, or generation steps."
            )
        if total_estimated_bytes > self.engines.settings.max_media_plan_estimated_bytes:
            raise ValueError("This ordered plan has an unsafe storage estimate.")
        if total_video_duration_seconds > self.engines.settings.max_media_plan_duration_seconds:
            raise ValueError("This ordered plan requests too much total video duration.")
        plan_estimate: dict[str, int | float] = {
            "step_count": len(intent.steps),
            "work_units": total_work_units,
            "video_duration_seconds": round(total_video_duration_seconds, 2),
            "estimated_bytes": total_estimated_bytes,
        }
        if (
            self.engines.settings.video_confirmation_work_units > 0
            and total_work_units >= self.engines.settings.video_confirmation_work_units
            and not request.confirm_media
        ):
            raise OrderedPlanConfirmationRequired(intent, estimate=plan_estimate)
        artifact_root = self.artifacts.root
        artifact_root.mkdir(parents=True, exist_ok=True)
        available_bytes = shutil.disk_usage(artifact_root).free
        if total_estimated_bytes > available_bytes:
            raise ValueError("There is not enough free storage for this ordered plan.")
        plan_estimate["available_bytes_at_admission"] = available_bytes

        input_parts: list[MessagePart] = [
            MessagePart(position=0, type=PartType.TEXT.value, text=request.text)
        ]
        for artifact in explicit_artifacts.values():
            input_parts.append(
                MessagePart(
                    position=len(input_parts),
                    type=self._input_part_type(artifact),
                    artifact_id=artifact.id,
                    metadata_json={
                        "input_reference": True,
                        "input_reference_source": "explicit",
                    },
                )
            )
        user_message = Message(
            chat_id=chat.id,
            parent_id=parent_message_id,
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
            parts=input_parts,
        )
        assistant_messages = [
            Message(
                chat_id=chat.id,
                role=MessageRole.ASSISTANT.value,
                status=MessageStatus.PENDING.value,
                parts=self._initial_output_parts(step["operation"], 1, 1),
            )
            for step in resolved_steps
        ]
        session.add_all([user_message, *assistant_messages])
        session.flush()
        previous_message_id = user_message.id
        for assistant_message in assistant_messages:
            assistant_message.parent_id = previous_message_id
            previous_message_id = assistant_message.id
        chat.active_head_message_id = assistant_messages[-1].id

        transcript_sequence = (
            session.scalar(
                select(WorkPlan.transcript_sequence)
                .where(WorkPlan.chat_id == chat.id)
                .order_by(WorkPlan.transcript_sequence.desc())
                .limit(1)
            )
            or 0
        ) + 1
        work_plan = WorkPlan(
            chat_id=chat.id,
            idempotency_key=request.idempotency_key,
            source_action=source_action,
            persistence_scope=self.persistence_scope,
            status=JobStatus.QUEUED.value,
            context_head_message_id=context_head_message_id,
            transcript_sequence=transcript_sequence,
            priority=0,
            planner_version=intent.planner_version,
            failure_policy="preserve_completed_block_dependents",
            summary_json={
                "operation": "ordered",
                "step_count": len(intent.steps),
                "source_action": source_action,
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_messages[0].id,
                "assistant_message_ids": [message.id for message in assistant_messages],
                "intent": intent.model_dump(mode="json"),
                "plan_estimate": plan_estimate,
                "status_counts": {"queued": len(intent.steps)},
            },
        )
        session.add(work_plan)
        session.flush()

        database_steps_by_intent_id: dict[str, WorkStep] = {}
        work_steps: list[WorkStep] = []
        runs: list[Run] = []
        jobs: list[Job] = []
        for ordinal, (resolved, output_message) in enumerate(
            zip(resolved_steps, assistant_messages, strict=True),
            start=1,
        ):
            step_intent = resolved["intent"]
            operation = resolved["operation"]
            profile = resolved["profile"]
            workflow_revision = resolved["workflow"]
            input_bindings: list[dict[str, Any]] = []
            if ordinal == 1:
                input_bindings.extend(
                    {
                        "type": "explicit_artifact",
                        "artifact_id": artifact_id,
                    }
                    for artifact_id in explicit_ids
                )
                if context_head_message_id:
                    input_bindings.insert(
                        0,
                        {
                            "type": "context_text",
                            "context_head_message_id": context_head_message_id,
                        },
                    )
            for binding in step_intent.inputs:
                source_step = database_steps_by_intent_id[binding.source_step_id]
                input_bindings.append(
                    {
                        "type": (
                            "step_output.text"
                            if binding.kind == "text_context"
                            else "step_output.artifact"
                        ),
                        "source_step_id": source_step.id,
                    }
                )
            output_type = step_intent.mode
            work_step = WorkStep(
                plan=work_plan,
                ordinal=ordinal,
                display_group="ordered_work",
                operation=operation.value,
                status=JobStatus.QUEUED.value,
                prompt=step_intent.prompt,
                profile_id=resolved["profile_id"],
                workflow_revision_id=(workflow_revision.id if workflow_revision else None),
                settings_json=resolved["settings"],
                input_bindings_json=input_bindings,
                output_contract_json=[
                    {
                        "slot": step_intent.id,
                        "type": output_type,
                        "index": ordinal,
                        "count": len(intent.steps),
                    }
                ],
                queue_class=(
                    "interactive_compute" if operation == Operation.TEXT else "media_compute"
                ),
            )
            session.add(work_step)
            session.flush()
            database_steps_by_intent_id[step_intent.id] = work_step
            dependency_ids = [
                database_steps_by_intent_id[dependency_id].id
                for dependency_id in step_intent.depends_on
            ]
            if ordinal == 1 and references_pending_output and pending_dependency_step_id:
                dependency_ids.append(pending_dependency_step_id)
            session.add_all(
                [
                    WorkStepDependency(
                        step_id=work_step.id,
                        depends_on_step_id=dependency_id,
                    )
                    for dependency_id in dict.fromkeys(dependency_ids)
                ]
            )
            model_provenance = self._model_provenance(session, profile)
            workflow_provenance = (
                {
                    "definition_id": workflow_revision.workflow_id,
                    "revision_id": workflow_revision.id,
                    "version": workflow_revision.version,
                    "engine": workflow_revision.engine,
                    "engine_version": workflow_revision.engine_version,
                    "trusted": workflow_revision.trusted,
                    "dependencies": workflow_revision.dependencies_json,
                }
                if workflow_revision
                else None
            )
            effective_preset = resolved["effective_preset"]
            run = Run(
                idempotency_key=request.idempotency_key if ordinal == 1 else None,
                chat_id=chat.id,
                user_message_id=user_message.id,
                assistant_message_id=output_message.id,
                work_plan_id=work_plan.id,
                work_step_id=work_step.id,
                operation=operation.value,
                status=RunStatus.QUEUED.value,
                standalone_prompt=step_intent.prompt,
                profile_id=resolved["profile_id"],
                vision_profile_id=resolved["vision_profile_id"],
                workflow_revision_id=(workflow_revision.id if workflow_revision else None),
                settings_json=resolved["settings"],
                provenance_json={
                    "planner_version": intent.planner_version,
                    "compiled_step": step_intent.model_dump(mode="json"),
                    "model_selection": resolved["model_selection"],
                    "input_artifact_ids": (list(explicit_ids) if ordinal == 1 else []),
                    "model": model_provenance,
                    "preset": (
                        {
                            "id": effective_preset[1].id,
                            "name": effective_preset[1].name,
                            "role": effective_preset[1].role,
                            "settings": effective_preset[2],
                        }
                        if effective_preset
                        else None
                    ),
                    "preset_layers": [
                        {
                            "scope": scope,
                            "id": preset.id,
                            "name": preset.name,
                            "role": preset.role,
                            "settings": preset_settings,
                        }
                        for scope, preset, preset_settings in resolved["preset_layers"]
                    ],
                    "workflow": workflow_provenance,
                    "resolved_settings": resolved["settings"],
                    "generation_estimate": resolved["generation_estimate"],
                    "plan_step_estimate": resolved["estimate"],
                    "image_edit": self._image_edit_provenance(resolved["image_edit_strength"]),
                    "auxiliary_assets": (
                        {
                            "lora_stack": resolved["lora_resolution"].provenance,
                            "selection": (
                                resolved["lora_selection"].provenance
                                if resolved["lora_selection"]
                                else {"mode": "explicit"}
                            ),
                            "graph_transform_version": LORA_GRAPH_TRANSFORM_VERSION,
                            "effective_graph_sha256": resolved["lora_resolution"].graph_sha256,
                        }
                        if resolved["lora_resolution"]
                        else None
                    ),
                },
            )
            session.add(run)
            session.flush()
            work_step.run_id = run.id
            job = Job(
                kind=self._job_kind(operation).value,
                status=JobStatus.QUEUED.value,
                run_id=run.id,
                work_plan_id=work_plan.id,
                work_step_id=work_step.id,
                progress=0,
                phase="queued",
                queue_resource=work_step.queue_class,
                queue_group="primary",
                queue_priority=0,
                queue_ticket=f"{transcript_sequence:020d}:{ordinal:04d}:{run.id}",
                enqueued_at=utcnow(),
                payload_json={
                    "operation": operation.value,
                    "ordered_step_id": step_intent.id,
                    "step_index": ordinal,
                    "step_count": len(intent.steps),
                },
            )
            update_job_progress(
                job,
                stage="queued",
                queue_resource=work_step.queue_class,
                queue_position=ordinal - 1,
                queue_length=len(intent.steps),
                blocked_by=dependency_ids,
                indeterminate=True,
            )
            session.add(job)
            work_steps.append(work_step)
            runs.append(run)
            jobs.append(job)

        work_plan.summary_json = {
            **work_plan.summary_json,
            "step_ids": [step.id for step in work_steps],
            "run_ids": [run.id for run in runs],
            "job_ids": [job.id for job in jobs],
        }
        if chat.title == "New chat":
            chat.title = request.text.strip().replace("\n", " ")[:72] or "New chat"
        session.commit()
        accepted = self._accepted_for_run(session, runs[0])
        await self.events.publish(
            "work_plan.created",
            work_plan.id,
            {
                "plan_id": work_plan.id,
                "step_id": work_steps[0].id,
                "run_id": runs[0].id,
                "job_id": jobs[0].id,
                "step_ids": [step.id for step in work_steps],
                "run_ids": [run.id for run in runs],
                "job_ids": [job.id for job in jobs],
                "chat_id": chat.id,
            },
        )
        for queued_job, queued_run in zip(jobs, runs, strict=True):
            self.start(queued_job.id, queued_run.id)
        return accepted

    def start(self, job_id: str, run_id: str) -> None:
        if job_id in self._tasks and not self._tasks[job_id].done():
            return
        task = asyncio.create_task(self._execute(job_id, run_id), name=f"local-lm-{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda finished: self._task_done(job_id, finished))

    def prepare_retry(self, session: Session, run: Run) -> None:
        """Reset the existing assistant slot before dispatching a retry."""

        message = session.get(Message, run.assistant_message_id)
        if not message:
            raise LookupError("run assistant message not found")
        preview_ids = self._temporary_preview_ids(message)
        message.status = MessageStatus.PENDING.value
        if run.operation == Operation.TEXT.value:
            parts = [
                MessagePart(position=0, type=PartType.TEXT.value, text=""),
                MessagePart(
                    position=1,
                    type=PartType.PROGRESS.value,
                    text="Queued",
                    metadata_json={
                        "activity": "chat",
                        "progress": 0,
                        "phase": "queued",
                    },
                ),
            ]
        else:
            parts = [
                MessagePart(
                    position=0,
                    type=PartType.PROGRESS.value,
                    text="Queued",
                    metadata_json={"progress": 0, "phase": "queued"},
                )
            ]
        self._replace_parts(message, parts)
        replacement = run.provenance_json.get("response_replacement")
        if isinstance(replacement, dict):
            revision_id = replacement.get("revision_id")
            revision = (
                session.get(ResponseRevision, revision_id) if isinstance(revision_id, str) else None
            )
            if revision and revision.run_id == run.id:
                revision.status = MessageStatus.PENDING.value
                revision.parts.clear()
        self._set_work_status(session, run, JobStatus.QUEUED.value)
        session.flush()
        for artifact_id in preview_ids:
            self.artifacts.delete_temporary_preview(session, artifact_id)

    def chat_guard(self, chat_id: str) -> asyncio.Lock:
        """Serialize turn creation, retries, and deletion for one chat."""
        return self._chat_guards.setdefault(chat_id, asyncio.Lock())

    @asynccontextmanager
    async def prepare_chat_deletion(self, chat_id: str) -> AsyncIterator[None]:
        """Stop every chat generation and hold its lifecycle lock through deletion."""
        async with self.chat_guard(chat_id):
            await self._cancel_chat_runs(chat_id)
            yield

    async def _cancel_chat_runs(self, chat_id: str) -> None:
        active_statuses = {
            JobStatus.QUEUED.value,
            JobStatus.RUNNING.value,
            JobStatus.PAUSED.value,
        }
        with self.session_factory() as session:
            rows = session.execute(
                select(Job.id, Job.run_id, Job.status, Run.operation)
                .join(Run, Job.run_id == Run.id)
                .where(Run.chat_id == chat_id)
            ).all()

        cancellable_rows = [
            (job_id, run_id, operation)
            for job_id, run_id, status, operation in rows
            if status in active_statuses or self._task_is_active(job_id)
        ]
        for job_id, run_id, operation in cancellable_rows:
            try:
                if operation == Operation.TEXT.value:
                    await self.engines.chat.cancel(run_id)
                else:
                    await self.engines.media.cancel(run_id)
            except Exception:
                logger.exception(
                    "Engine cancellation failed while deleting chat %s (job %s)",
                    chat_id,
                    job_id,
                )

        tasks = {
            job_id: task
            for job_id, _run_id, _operation in cancellable_rows
            if (task := self._tasks.get(job_id)) is not None and not task.done()
        }
        for task in tasks.values():
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks.values(), return_exceptions=True)
            for job_id, task in tasks.items():
                if self._tasks.get(job_id) is task:
                    self._tasks.pop(job_id, None)

        cancelled: list[tuple[str, str]] = []
        with self.session_factory() as session:
            for job_id, run_id, _status, _operation in rows:
                job = session.get(Job, job_id)
                if not job:
                    continue
                if job.status in active_statuses:
                    self._mark_cancelled(session, job)
                if job.status == JobStatus.CANCELLED.value:
                    cancelled.append((job.id, run_id))
            session.commit()

        for job_id, run_id in cancelled:
            await self.scheduler.publish_job(job_id)
            await self.events.publish("run.cancelled", run_id, {"job_id": job_id})

    def _task_is_active(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        return task is not None and not task.done()

    def _task_done(self, job_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(job_id) is task:
            self._tasks.pop(job_id, None)
        if task.cancelled():
            return
        exception = task.exception()
        if exception:
            logger.exception(
                "Background job %s terminated unexpectedly", job_id, exc_info=exception
            )

    async def cancel(self, job_id: str) -> bool:
        cancelled_task: asyncio.Task[None] | None = None
        run_id: str | None = None
        operation: str | None = None
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if not job or job.status in {
                JobStatus.COMPLETE.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            }:
                return False
            if job.run_id:
                run = session.get(Run, job.run_id)
                if run:
                    run_id = run.id
                    operation = run.operation
            task = self._tasks.get(job_id)
            if task:
                task.cancel()
                cancelled_task = task
            self._mark_cancelled(session, job)
            session.commit()
        if run_id:
            try:
                if operation == Operation.TEXT.value:
                    await self.engines.chat.cancel(run_id)
                else:
                    await self.engines.media.cancel(run_id)
            except Exception:
                logger.warning(
                    "Could not signal engine cancellation for run %s", run_id, exc_info=True
                )
        if cancelled_task:
            await asyncio.gather(cancelled_task, return_exceptions=True)
        await self.scheduler.publish_job(job_id)
        await self.events.publish("run.cancelled", run_id, {"job_id": job_id})
        return True

    async def close(self) -> None:
        self._admission_open = False
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        if self._media_restart_task:
            self._media_restart_task.cancel()
            await asyncio.gather(self._media_restart_task, return_exceptions=True)
            self._media_restart_task = None
        self._media_restart_after_chat_activity = False
        self._step_prewarm_plan_id = None
        self._step_prewarm_task = None

    def stop_admission(self) -> None:
        self._admission_open = False

    async def _execute(self, job_id: str, run_id: str) -> None:
        try:
            with self.session_factory() as session:
                job = session.get(Job, job_id)
                run = session.get(Run, run_id)
                if not job or not run:
                    return
                operation = run.operation
                resource = job.queue_resource or (
                    "interactive_compute" if operation == Operation.TEXT.value else "media_compute"
                )
                group = job.queue_group or "primary"
                priority = job.queue_priority
            async with self.scheduler.job_lease(
                job_id,
                resource=resource,
                group=group,
                priority=priority,
            ):
                with self.session_factory() as session:
                    job = session.get(Job, job_id)
                    run = session.get(Run, run_id)
                    if (
                        not job
                        or not run
                        or job.status
                        in {
                            JobStatus.CANCELLED.value,
                            JobStatus.FAILED.value,
                            JobStatus.INTERRUPTED.value,
                        }
                    ):
                        return
                    self._resolve_step_inputs(session, run)
                    run.status = RunStatus.RUNNING.value
                    run.started_at = job.started_at or utcnow()
                    self._set_work_status(session, run, JobStatus.RUNNING.value)
                    session.commit()
                    event_payload = {
                        "job_id": job_id,
                        "plan_id": run.work_plan_id,
                        "step_id": run.work_step_id,
                    }
                    operation = run.operation
                    prompt = run.standalone_prompt
                    self._arm_step_prewarm(session, run)
                await self.events.publish("run.created", run_id, event_payload)
                await self.events.publish(
                    "plan.selected",
                    run_id,
                    {
                        "operation": operation,
                        "prompt": prompt,
                        "plan_id": event_payload["plan_id"],
                        "step_id": event_payload["step_id"],
                    },
                )
                resume_chat_profile = await self._prepare_device_handoff(operation)
                try:
                    if operation == Operation.TEXT.value:
                        await self._execute_chat(job_id, run_id)
                    else:
                        await self._execute_media(job_id, run_id)
                finally:
                    if operation == Operation.TEXT.value:
                        self._release_deferred_media_restart()
                        await self._settle_step_prewarm(job_id)
                    if resume_chat_profile:
                        await self._complete_media_handoff(resume_chat_profile)
        except asyncio.CancelledError:
            with self.session_factory() as session:
                job = session.get(Job, job_id)
                if job and job.status != JobStatus.CANCELLED.value:
                    self._mark_cancelled(session, job)
                    session.commit()
            raise
        except Exception as exc:
            detail = str(exc).strip() or f"Generation failed ({type(exc).__name__})"
            await self._fail(job_id, run_id, detail)

    async def _execute_chat(self, job_id: str, run_id: str) -> None:
        await self._set_chat_phase(job_id, run_id, "Preparing chat model")
        worker = await self._ensure_chat_worker(run_id)
        await self._set_chat_phase(job_id, run_id, "Preparing conversation")
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            if not run:
                return
            (
                messages,
                request_settings,
                context_metadata,
                tool_calling_available,
            ) = await self._prepare_chat_context(session, run)
            run.provenance_json = {
                **run.provenance_json,
                "context": context_metadata,
                **(
                    {
                        "worker": worker.model_dump(
                            mode="json",
                            exclude={"active_jobs", "queued_jobs"},
                        )
                    }
                    if worker
                    else {}
                ),
            }
            session.commit()
            request = ChatRequest(
                run_id=run.id,
                messages=messages,
                settings=request_settings,
                persistence_scope=self.persistence_scope,
                scope_id=self.scope_id,
            )
            assistant_id = run.assistant_message_id

        await self._set_chat_phase(job_id, run_id, "Waiting for first token")
        accumulated = ""
        completion_metadata: dict[str, Any] = {}
        last_persisted_length = 0
        last_persisted_at = time.monotonic()
        try:
            async for event in self.engines.chat.stream(request):
                if event.type == "delta":
                    self._release_deferred_media_restart()
                    self._begin_step_prewarm()
                    accumulated += event.text
                    await self.events.publish(
                        "text.delta",
                        run_id,
                        {
                            "text": event.text,
                            "job_id": job_id,
                            "assistant_message_id": assistant_id,
                        },
                    )
                    now = time.monotonic()
                    if (
                        last_persisted_length == 0
                        or len(accumulated) - last_persisted_length >= 32
                        or now - last_persisted_at >= 0.25
                    ):
                        self._persist_streamed_text(assistant_id, accumulated)
                        last_persisted_length = len(accumulated)
                        last_persisted_at = now

                elif event.type == "cancelled":
                    if accumulated:
                        self._persist_streamed_text(assistant_id, accumulated.rstrip())
                    return
                elif event.type == "error":
                    detail = str(event.data.get("error") or "").strip()
                    raise RuntimeError(detail or "Chat engine stream failed")
                elif event.type in {"usage", "complete"}:
                    completion_metadata.update(event.data)
        except asyncio.CancelledError:
            if accumulated:
                self._persist_streamed_text(assistant_id, accumulated.rstrip())
            raise
        except Exception:
            if accumulated:
                self._persist_streamed_text(assistant_id, accumulated.rstrip())
            raise

        text_output = accumulated.rstrip()
        offer_relevant = should_extract_generation_offer(text_output) or any(
            message.get("role") == MessageRole.USER.value
            and isinstance(message.get("content"), str)
            and should_extract_generation_offer(message["content"])
            for message in messages[-2:]
        )
        offer = (
            await extract_generation_offer(self.engines.chat, text_output)
            if tool_calling_available and offer_relevant
            else None
        )
        if offer and offer.message not in text_output:
            text_output = "\n\n".join(value for value in (text_output, offer.message) if value)

        completed_assistant_id = assistant_id
        with self.session_factory() as session:
            message = session.get(Message, assistant_id)
            run = session.get(Run, run_id)
            job = session.get(Job, job_id)
            if not message or not run or not job:
                return
            self._remove_chat_progress(message)
            text_part = next(
                (part for part in message.parts if part.type == PartType.TEXT.value), None
            )
            if text_part:
                text_part.text = text_output
            else:
                self._replace_parts(
                    message,
                    [MessagePart(position=0, type=PartType.TEXT.value, text=text_output)],
                )
            context_metadata = dict(run.provenance_json.get("context", {}))
            if usage := completion_metadata.get("usage"):
                context_metadata["usage"] = usage
            self._complete(
                session,
                run,
                job,
                {"characters": len(text_output), "context": context_metadata},
            )
            run.provenance_json = {
                **run.provenance_json,
                "context": context_metadata,
                "completion": completion_metadata,
                "output": {
                    "kind": "text",
                    "characters": len(text_output),
                    "sha256": hashlib.sha256(text_output.encode()).hexdigest(),
                },
                "timings": {"duration_ms": run.duration_ms},
            }
            metadata_part = next(
                (part for part in message.parts if part.type == PartType.GENERATION_METADATA.value),
                None,
            )
            metadata = {
                "run_id": run.id,
                "context": context_metadata,
                "completion": completion_metadata,
                "provenance": run.provenance_json,
                **({"generation_offer": generation_offer_metadata(offer)} if offer else {}),
            }
            if metadata_part:
                metadata_part.metadata_json = metadata
            else:
                message.parts.append(
                    MessagePart(
                        position=max((part.position for part in message.parts), default=-1) + 1,
                        type=PartType.GENERATION_METADATA.value,
                        metadata_json=metadata,
                    )
                )
            completed_assistant_id = self._finalize_response_revision(
                session,
                run,
                message,
                promote=True,
            )
            session.commit()
        await self.scheduler.publish_job(job_id)
        await self.events.publish(
            "run.completed",
            run_id,
            {
                "job_id": job_id,
                "assistant_message_id": completed_assistant_id,
            },
        )

    async def _set_chat_phase(self, job_id: str, run_id: str, label: str) -> None:
        assistant_id = ""
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            if (
                not job
                or not run
                or job.status
                in {
                    JobStatus.COMPLETE.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                }
            ):
                return
            assistant_id = run.assistant_message_id
            update_job_progress(
                job,
                stage=label.lower(),
                queue_resource=job.queue_resource,
                indeterminate=True,
            )
            message = session.get(Message, assistant_id)
            if message:
                progress_part = next(
                    (
                        part
                        for part in message.parts
                        if part.type == PartType.PROGRESS.value
                        and part.metadata_json.get("activity") == "chat"
                    ),
                    None,
                )
                if progress_part:
                    progress_part.text = label
                    progress_part.metadata_json = {
                        "activity": "chat",
                        "progress": 0,
                        "phase": label.lower(),
                    }
                else:
                    message.parts.append(
                        MessagePart(
                            position=max(
                                (part.position for part in message.parts),
                                default=-1,
                            )
                            + 1,
                            type=PartType.PROGRESS.value,
                            text=label,
                            metadata_json={
                                "activity": "chat",
                                "progress": 0,
                                "phase": label.lower(),
                            },
                        )
                    )
            session.commit()
        await self.scheduler.publish_job(job_id)
        await self.events.publish(
            "run.progress",
            run_id,
            {
                "assistant_message_id": assistant_id,
                "job_id": job_id,
                "phase": label.lower(),
                "label": label,
            },
        )

    async def _ensure_chat_worker(self, run_id: str) -> WorkerStatus | None:
        if self.engines.settings.chat_engine not in {"llama.cpp", "vllm"}:
            return None
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            profile = session.get(ModelProfile, run.profile_id) if run and run.profile_id else None
            install = (
                session.get(ModelInstall, profile.model_install_id)
                if profile and profile.model_install_id
                else None
            )
            if not profile or not install:
                raise RuntimeError("the selected chat profile does not have an installed model")
            session.expunge(profile)
            session.expunge(install)
        status = next(item for item in self.processes.statuses() if item.name == "chat")
        if status.running and status.state == "ready" and status.profile_id == profile.id:
            return status
        return await self.processes.load_chat(profile, install)

    async def _chat_planner_available(self) -> bool:
        if self.engines.settings.chat_engine not in {"llama.cpp", "vllm"}:
            return True
        executable = (
            self.processes.settings.vllm_executable
            if self.engines.settings.chat_engine == "vllm"
            else self.processes.settings.llama_executable
        )
        if not executable:
            return False
        if not self._chat_planner_ready.is_set():
            return False
        status = next(item for item in self.processes.statuses() if item.name == "chat")
        return status.state == "ready"

    async def _prepare_chat_context(
        self, session: Session, run: Run
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], bool]:
        messages, source_message_ids = self._context_messages_with_sources(session, run)
        self._commit_before_await(session)
        capabilities = await self.engines.chat_capabilities()
        candidates = self._visual_context_artifacts(session, run)
        self._commit_before_await(session)
        direct_profile_selected = run.vision_profile_id == run.profile_id and bool(run.profile_id)
        if (
            self.engines.settings.chat_engine == "mock"
            and candidates
            and "image" in capabilities.input_modalities
        ):
            direct_profile_selected = True
        vision_metadata: dict[str, Any] = {
            "available": "image" in capabilities.input_modalities,
            "images_included": 0,
            "artifact_ids": [],
            "mode": "none",
            "visual_contents_inspected": False,
        }
        if candidates and vision_metadata["available"] and direct_profile_selected:
            job_id = self._job_id_for_run(session, run)
            self._commit_before_await(session)
            await self._set_chat_phase(
                job_id,
                run.id,
                "Preparing visual context",
            )
            messages, vision_metadata = await self._attach_visual_context(
                session,
                run,
                messages,
                candidates=candidates,
            )
            run.vision_profile_id = run.profile_id
            vision_metadata.update(
                {
                    "mode": "direct",
                    "profile_id": run.profile_id,
                    "profile": self._vision_profile_provenance(session, run.profile_id),
                    "visual_contents_inspected": bool(vision_metadata.get("images_included")),
                }
            )
        elif candidates and direct_profile_selected:
            raise RuntimeError(
                "The verified vision profile did not expose image input after loading."
            )
        elif candidates and run.vision_profile_id and run.vision_profile_id != run.profile_id:
            observation, bridge_metadata = await self._bridge_visual_context(
                session,
                run,
                candidates,
            )
            vision_metadata = bridge_metadata
            vision_metadata["profile"] = self._vision_profile_provenance(
                session,
                run.vision_profile_id,
            )
            if observation:
                messages = self._append_vision_observation(
                    messages,
                    observation,
                    run.vision_profile_id,
                )
        elif candidates:
            vision_metadata.update(
                {
                    "reason": "No runtime-verified vision profile is available.",
                    "images_skipped": len(candidates),
                }
            )
        source_message_ids = source_message_ids[: len(messages)] + [None] * max(
            0,
            len(messages) - len(source_message_ids),
        )
        profile = session.get(ModelProfile, run.profile_id) if run.profile_id else None
        context_limit = int(
            (profile.load_settings_json if profile else {}).get("context_length", 8192)
        )
        requested_output = int(run.settings_json.get("max_tokens", 1024))
        safety_tokens = min(128, max(32, context_limit // 100))
        maximum_output = max(1, context_limit - safety_tokens - 64)
        output_limit = min(requested_output, maximum_output)
        input_budget = max(64, context_limit - output_limit - safety_tokens)

        self._commit_before_await(session)
        messages, input_tokens, omitted, compaction = await self._fit_chat_context(
            messages,
            source_message_ids,
            input_budget,
        )
        if input_tokens > input_budget:
            raise ValueError(
                "The current instructions and message exceed this profile's context window. "
                "Increase Context length, reduce Maximum output, or shorten the message."
            )

        request_settings = {**run.settings_json, "max_tokens": output_limit}
        metadata = {
            "policy": "compact-oldest-preserve-system-and-newest",
            "context_limit": context_limit,
            "input_budget": input_budget,
            "input_tokens": input_tokens,
            "requested_output_tokens": requested_output,
            "output_limit": output_limit,
            "safety_tokens": safety_tokens,
            "messages_included": len(messages),
            "messages_omitted": omitted,
            "compaction": compaction,
            "output_adjusted": output_limit != requested_output,
            "vision": vision_metadata,
        }
        return messages, request_settings, metadata, capabilities.tool_calling

    async def _fit_chat_context(
        self,
        messages: list[dict[str, Any]],
        source_message_ids: list[str | None],
        input_budget: int,
    ) -> tuple[list[dict[str, Any]], int, int, dict[str, Any]]:
        remaining = list(messages)
        remaining_sources = list(source_message_ids)
        removed: list[dict[str, Any]] = []
        removed_sources: list[str | None] = []
        system_messages = (
            1 if remaining and remaining[0].get("role") == MessageRole.SYSTEM.value else 0
        )

        def remove_oldest() -> bool:
            if len(remaining) <= system_messages + 1:
                return False
            remove_count = 1
            if (
                remaining[system_messages].get("role") == MessageRole.USER.value
                and len(remaining) > system_messages + 2
                and remaining[system_messages + 1].get("role") == MessageRole.ASSISTANT.value
            ):
                remove_count = 2
            removed.extend(remaining[system_messages : system_messages + remove_count])
            removed_sources.extend(
                remaining_sources[system_messages : system_messages + remove_count]
            )
            del remaining[system_messages : system_messages + remove_count]
            del remaining_sources[system_messages : system_messages + remove_count]
            return True

        input_tokens = await self.engines.chat.count_tokens(remaining)
        while input_tokens > input_budget and remove_oldest():
            input_tokens = await self.engines.chat.count_tokens(remaining)
        if not removed:
            return (
                remaining,
                input_tokens,
                0,
                {
                    "active": False,
                    "version": CONTEXT_COMPACTION_VERSION,
                    "reason": "not_needed",
                    "transcript_preserved": True,
                    "reversible": True,
                },
            )

        max_characters = min(
            MAX_COMPACTION_CHARACTERS,
            max(MIN_COMPACTION_CHARACTERS, input_budget * 2),
        )
        while removed:
            folded = compact_context_messages(
                removed,
                removed_sources,
                max_characters=max_characters,
            )
            candidate = list(remaining)
            candidate.insert(system_messages, folded.message)
            candidate_tokens = await self.engines.chat.count_tokens(candidate)
            if candidate_tokens <= input_budget:
                return (
                    candidate,
                    candidate_tokens,
                    len(removed),
                    {
                        **folded.provenance,
                        "fold_tokens": max(0, candidate_tokens - input_tokens),
                    },
                )
            if max_characters > MIN_COMPACTION_CHARACTERS:
                max_characters = max(
                    MIN_COMPACTION_CHARACTERS,
                    int(max_characters * 0.7),
                )
                continue
            if not remove_oldest():
                break
            input_tokens = await self.engines.chat.count_tokens(remaining)
            while input_tokens > input_budget and remove_oldest():
                input_tokens = await self.engines.chat.count_tokens(remaining)

        return (
            remaining,
            input_tokens,
            len(removed),
            {
                "active": False,
                "version": CONTEXT_COMPACTION_VERSION,
                "reason": "insufficient_budget",
                "source_message_count": len(removed),
                "transcript_preserved": True,
                "reversible": True,
            },
        )

    async def _attach_visual_context(
        self,
        session: Session,
        run: Run,
        messages: list[dict[str, Any]],
        *,
        candidates: list[Artifact] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidates = candidates or self._visual_context_artifacts(session, run)
        current = session.get(Message, run.user_message_id)
        strict_ids = {
            part.artifact_id
            for part in (current.parts if current else [])
            if part.artifact_id and part.metadata_json.get("input_reference_source") == "explicit"
        }
        chat = session.get(Chat, run.chat_id)
        vision_settings = dict(chat.vision_settings_json) if chat else {}
        self._commit_before_await(session)
        visual = await self.vision.prepare(
            candidates,
            strict_artifact_ids=strict_ids,
            vision_settings=vision_settings,
        )
        if not visual.frames:
            posters = [
                poster
                for artifact in candidates
                if artifact.id not in strict_ids
                and artifact.media_type.casefold().startswith("video/")
                and isinstance(
                    poster_id := artifact.metadata_json.get("poster_artifact_id"),
                    str,
                )
                and (poster := session.get(Artifact, poster_id)) is not None
            ]
            if posters:
                self._commit_before_await(session)
                poster_visual = await self.vision.prepare(
                    posters,
                    strict_artifact_ids=set(),
                    vision_settings=vision_settings,
                )
                visual = PreparedVisualContext(
                    frames=poster_visual.frames,
                    skipped_artifact_ids=tuple(
                        dict.fromkeys(
                            (
                                *visual.skipped_artifact_ids,
                                *poster_visual.skipped_artifact_ids,
                            )
                        )
                    ),
                )
        return self.vision.attach_to_latest_user(messages, visual), {
            "available": True,
            "images_included": len(visual.frames),
            "artifact_ids": visual.inspected_artifact_ids,
            "bytes_included": sum(len(frame.content) for frame in visual.frames),
            "images_skipped": len(visual.skipped_artifact_ids),
            "sampled_frame_timestamps": [
                {
                    "artifact_id": frame.artifact_id,
                    "timestamp_seconds": frame.timestamp_seconds,
                }
                for frame in visual.frames
                if frame.timestamp_seconds is not None
            ],
            **visual.provenance(),
        }

    async def _bridge_visual_context(
        self,
        session: Session,
        run: Run,
        candidates: list[Artifact],
    ) -> tuple[str, dict[str, Any]]:
        bridge_profile = session.get(ModelProfile, run.vision_profile_id)
        bridge_install = (
            session.get(ModelInstall, bridge_profile.model_install_id)
            if bridge_profile and bridge_profile.model_install_id
            else None
        )
        text_profile = session.get(ModelProfile, run.profile_id) if run.profile_id else None
        text_install = (
            session.get(ModelInstall, text_profile.model_install_id)
            if text_profile and text_profile.model_install_id
            else None
        )
        if not bridge_profile or not bridge_install:
            return "", {
                "available": False,
                "mode": "none",
                "visual_contents_inspected": False,
                "images_included": 0,
                "artifact_ids": [],
                "images_skipped": len(candidates),
                "reason": "The selected vision profile is unavailable.",
            }
        session.expunge(bridge_profile)
        session.expunge(bridge_install)
        if text_profile:
            session.expunge(text_profile)
        if text_install:
            session.expunge(text_install)
        observation = ""
        metadata: dict[str, Any] = {
            "available": True,
            "mode": "bridge",
            "profile_id": bridge_profile.id,
            "visual_contents_inspected": False,
            "images_included": 0,
            "artifact_ids": [],
        }
        try:
            job_id = self._job_id_for_run(session, run)
            self._commit_before_await(session)
            await self._set_chat_phase(
                job_id,
                run.id,
                "Loading vision model",
            )
            await self.processes.load_chat(bridge_profile, bridge_install)
            capabilities = await self.engines.chat_capabilities()
            if "image" not in capabilities.input_modalities:
                raise RuntimeError("the selected vision profile did not accept image input")
            bridge_messages: list[dict[str, Any]] = [
                {
                    "role": MessageRole.USER.value,
                    "content": (
                        "Inspect the attached visual material for the user's request. "
                        "Describe only relevant visible facts. If frames have timestamp labels, "
                        "identify observations by those sampled timestamps and do not claim to "
                        "have inspected unsampled portions.\n\n"
                        f"User request: {run.standalone_prompt}"
                    ),
                }
            ]
            bridge_messages, metadata = await self._attach_visual_context(
                session,
                run,
                bridge_messages,
                candidates=candidates,
            )
            metadata.update(
                {
                    "available": True,
                    "mode": "bridge",
                    "profile_id": bridge_profile.id,
                    "visual_contents_inspected": bool(metadata.get("images_included")),
                }
            )
            if not metadata["visual_contents_inspected"]:
                return "", metadata
            job_id = self._job_id_for_run(session, run)
            self._commit_before_await(session)
            await self._set_chat_phase(
                job_id,
                run.id,
                f"Analyzing {metadata['images_included']} visual frame"
                f"{'' if metadata['images_included'] == 1 else 's'}",
            )
            max_tokens = self.engines.settings.vision_bridge_max_tokens
            completion_seen = False
            completion_metadata: dict[str, Any] = {}
            async with asyncio.timeout(180):
                async for event in self.engines.chat.stream(
                    ChatRequest(
                        run_id=run.id,
                        messages=bridge_messages,
                        settings={"temperature": 0, "max_tokens": max_tokens},
                        persistence_scope=self.persistence_scope,
                        scope_id=self.scope_id,
                    )
                ):
                    if event.type == "delta":
                        observation += event.text
                        if len(observation) > 16_000:
                            raise RuntimeError("vision observation exceeded its safety limit")
                    elif event.type == "error":
                        raise RuntimeError(str(event.data.get("error") or "vision analysis failed"))
                    elif event.type == "cancelled":
                        raise asyncio.CancelledError
                    elif event.type in {"usage", "complete"}:
                        completion_metadata.update(event.data)
                        completion_seen = completion_seen or event.type == "complete"
            observation = observation.strip()
            if not observation or not completion_seen:
                raise RuntimeError("vision profile returned no observation")
            metadata["observation_sha256"] = hashlib.sha256(observation.encode()).hexdigest()
            metadata["observation_characters"] = len(observation)
            metadata["completion"] = completion_metadata
            return observation, metadata
        finally:
            job_id = self._job_id_for_run(session, run)
            self._commit_before_await(session)
            await self._set_chat_phase(
                job_id,
                run.id,
                "Restoring chat model",
            )
            if text_profile and text_install:
                await self.processes.load_chat(text_profile, text_install)
            else:
                await self.processes.stop("chat")

    @staticmethod
    def _append_vision_observation(
        messages: list[dict[str, Any]],
        observation: str,
        profile_id: str | None,
    ) -> list[dict[str, Any]]:
        attributed = {
            "role": "system",
            "content": (
                "Attributed visual observation from the selected local vision profile "
                f"({profile_id or 'unknown'}). Treat it as model-produced context, not as the "
                f"user's words:\n{observation}"
            ),
        }
        insert_at = 1 if messages and messages[0].get("role") == "system" else 0
        return [*messages[:insert_at], attributed, *messages[insert_at:]]

    @staticmethod
    def _job_id_for_run(session: Session, run: Run) -> str:
        job_id = session.scalar(
            select(Job.id).where(Job.run_id == run.id).order_by(Job.created_at.desc()).limit(1)
        )
        return job_id or ""

    @staticmethod
    def _commit_before_await(session: Session) -> None:
        """Release SQLite state before an external or potentially long await."""

        if session.in_transaction():
            session.commit()

    @classmethod
    def _visual_context_artifacts(cls, session: Session, run: Run) -> list[Artifact]:
        """Return explicit current inputs, then the newest prior branch visual."""

        current = session.get(Message, run.user_message_id)
        candidates = [
            artifact
            for artifact_id in cls.input_artifact_ids_for_run(session, run)
            if (artifact := session.get(Artifact, artifact_id)) is not None
        ]
        seen = {artifact.id for artifact in candidates}
        chat = session.get(Chat, run.chat_id)
        vision_settings = chat.vision_settings_json if chat else {}
        if (
            isinstance(vision_settings, dict)
            and vision_settings.get("include_prior_visual") is False
        ):
            return candidates

        current_id = current.parent_id if current else None
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            message = session.scalar(
                select(Message)
                .options(selectinload(Message.parts))
                .where(Message.id == current_id, Message.chat_id == run.chat_id)
            )
            if not message:
                break
            for part in sorted(message.parts, key=lambda value: value.position, reverse=True):
                if not part.artifact_id:
                    continue
                referenced = session.get(Artifact, part.artifact_id)
                if not referenced:
                    continue
                if (
                    part.type in {PartType.IMAGE.value, PartType.VIDEO.value}
                    and referenced.id not in seen
                ):
                    candidates.append(referenced)
                    return candidates
            current_id = message.parent_id
        return candidates

    async def _prepare_device_handoff(self, operation: str) -> str | None:
        if (
            operation == Operation.TEXT.value
            or not self.processes.settings.auto_unload_chat_for_media
        ):
            return None
        chat_worker = next(item for item in self.processes.statuses() if item.name == "chat")
        if not chat_worker.running or not chat_worker.managed:
            return None
        profile_id = chat_worker.profile_id
        if profile_id:
            self._chat_planner_ready.clear()
        try:
            await self.processes.stop("chat")
        except Exception:
            self._chat_planner_ready.set()
            raise
        return profile_id

    async def _resume_chat_worker(self, profile_id: str) -> None:
        try:
            with self.session_factory() as session:
                profile = session.get(ModelProfile, profile_id)
                install = (
                    session.get(ModelInstall, profile.model_install_id)
                    if profile and profile.model_install_id
                    else None
                )
                if not profile or not install:
                    return
                session.expunge(profile)
                session.expunge(install)
            await self.processes.load_chat(profile, install)
        except Exception:
            logger.exception("Could not reload chat profile %s after media handoff", profile_id)
        finally:
            self._chat_planner_ready.set()

    def _handoff_chat_target(self, fallback_profile_id: str) -> tuple[str, bool]:
        """Prefer the profile required by the next dispatchable text job."""

        try:
            candidate = self.scheduler.peek_next_eligible_job("primary")
        except Exception:
            logger.exception("Could not inspect the next job during media handoff")
            return fallback_profile_id, False
        if (
            not isinstance(candidate, tuple)
            or len(candidate) != 2
            or not isinstance(candidate[1], str)
        ):
            return fallback_profile_id, False
        with self.session_factory() as session:
            run = session.get(Run, candidate[1])
            if run and run.operation == Operation.TEXT.value and isinstance(run.profile_id, str):
                return run.profile_id, True
        return fallback_profile_id, False

    async def _complete_media_handoff(self, chat_profile_id: str) -> None:
        """Release retained Comfy state before restoring a managed chat model."""

        selected_chat_profile_id, queued_text_next = self._handoff_chat_target(chat_profile_id)
        recycle_managed_media = False
        try:
            media_worker = next(item for item in self.processes.statuses() if item.name == "media")
            if self.engines.settings.media_engine == "comfyui" and media_worker.managed:
                # ComfyUI retains model allocations after generation. Loading a
                # large chat model alongside that cache can push Windows/WDDM
                # into system-memory paging; the next media run then appears
                # stalled before sampling. A managed worker recycle releases
                # both VRAM and host allocations while preserving the automatic
                # Ready media service expected by the desktop application.
                await self.processes.stop("media")
                recycle_managed_media = True
        except Exception:
            logger.exception("Could not recycle the media worker after device handoff")

        if not recycle_managed_media:
            await self._resume_chat_worker(selected_chat_profile_id)
            return

        # Restore chat without competing with Python/Torch startup for disk and
        # CPU. Once chat is ready, warm the empty ComfyUI service in a tracked
        # background task so the queued text job can proceed immediately.
        await self._resume_chat_worker(selected_chat_profile_id)
        if queued_text_next:
            self._media_restart_after_chat_activity = True
        else:
            self._schedule_media_restart()

    def _schedule_media_restart(self) -> None:
        if self._media_restart_task and not self._media_restart_task.done():
            return
        task = asyncio.create_task(
            self._restart_media_worker(),
            name="media-worker-handoff-restart",
        )
        self._media_restart_task = task
        task.add_done_callback(self._media_restart_finished)

    def _release_deferred_media_restart(self) -> None:
        if not self._media_restart_after_chat_activity:
            return
        self._media_restart_after_chat_activity = False
        self._schedule_media_restart()

    def _arm_step_prewarm(self, session: Session, run: Run) -> None:
        """Remember that the following ordered step will need the media worker.

        Only a text step whose successor in the same work plan runs on the
        media worker arms a prewarm, and only while that worker is down.
        Single-step turns and same-role consecutive steps change nothing.
        """

        self._step_prewarm_plan_id = None
        self._step_prewarm_task = None
        if (
            run.operation != Operation.TEXT.value
            or not run.work_plan_id
            or not run.work_step_id
            or self.engines.settings.media_engine != "comfyui"
        ):
            return
        step = session.get(WorkStep, run.work_step_id)
        if not step:
            return
        next_step = session.scalar(
            select(WorkStep).where(
                WorkStep.plan_id == run.work_plan_id,
                WorkStep.ordinal == step.ordinal + 1,
            )
        )
        if (
            not next_step
            or next_step.operation == Operation.TEXT.value
            or next_step.status != JobStatus.QUEUED.value
        ):
            return
        media_worker = next(
            (item for item in self.processes.statuses() if item.name == "media"), None
        )
        if media_worker is None or media_worker.running:
            return
        self._step_prewarm_plan_id = run.work_plan_id

    def _begin_step_prewarm(self) -> None:
        """Start the next ordered step's media worker while chat still streams.

        Mirrors the deferred media restart released on chat activity: only the
        worker process launches now. ComfyUI defers model loads until a prompt
        executes, so the running text step keeps exclusive use of the device
        and the media job still claims the primary lease before generating.
        """

        if self._step_prewarm_plan_id is None or self._step_prewarm_task is not None:
            return
        self._schedule_media_restart()
        self._step_prewarm_task = self._media_restart_task

    async def _settle_step_prewarm(self, job_id: str) -> None:
        """Finish or abort a plan prewarm once its triggering step stops."""

        if self._step_prewarm_plan_id is None:
            return
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            completed = job is not None and job.status == JobStatus.COMPLETE.value
        if completed:
            # A step that never streamed still warms its successor on success.
            self._begin_step_prewarm()
            self._step_prewarm_plan_id = None
            self._step_prewarm_task = None
            return
        task = self._step_prewarm_task
        self._step_prewarm_plan_id = None
        self._step_prewarm_task = None
        if task and not task.done():
            # Cancellation or failure blocks the rest of the plan, so stop the
            # in-flight worker launch exactly as close() stops restarts.
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self._media_restart_task is task:
                self._media_restart_task = None

    async def _restart_media_worker(self) -> None:
        try:
            await self.processes.start_media()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Could not restart the managed media worker after device handoff")

    def _media_restart_finished(self, task: asyncio.Task[None]) -> None:
        if self._media_restart_task is task:
            self._media_restart_task = None

    async def _ensure_media_worker(self) -> None:
        restart_task = self._media_restart_task
        if restart_task and not restart_task.done():
            await asyncio.shield(restart_task)
        status = next(item for item in self.processes.statuses() if item.name == "media")
        if not status.running or status.state != "ready":
            await self.processes.start_media()

    async def _execute_media(self, job_id: str, run_id: str) -> None:
        if self.engines.settings.media_engine == "comfyui":
            status = next(item for item in self.processes.statuses() if item.name == "media")
            if not status.running or status.state != "ready":
                with self.session_factory() as session:
                    job = session.get(Job, job_id)
                    run = session.get(Run, run_id)
                    message = session.get(Message, run.assistant_message_id) if run else None
                    if job and message:
                        event = MediaEvent(
                            type="progress",
                            progress=0,
                            phase="preparing media runtime",
                        )
                        update_job_progress(
                            job,
                            stage=event.phase,
                            queue_resource=job.queue_resource,
                            indeterminate=True,
                        )
                        self._replace_parts(
                            message,
                            self._media_progress_parts(message, event),
                        )
                        session.commit()
                await self.scheduler.publish_job(job_id)
                await self.events.publish(
                    "generation.progress",
                    run_id,
                    {
                        "progress": 0,
                        "phase": "preparing media runtime",
                        "job_id": job_id,
                    },
                )
                await self._ensure_media_worker()
        with self.session_factory() as session:
            run = session.get(Run, run_id)
            if not run:
                return
            input_paths: list[Path] = []
            for artifact_id in self.input_artifact_ids_for_run(session, run):
                artifact = session.get(Artifact, artifact_id)
                if artifact:
                    input_paths.append(self.artifacts.resolve(artifact))
            workflow: dict[str, Any] = {}
            if run.workflow_revision_id:
                revision = session.get(WorkflowRevision, run.workflow_revision_id)
                if revision:
                    if revision.engine == "comfyui" and not revision.trusted:
                        raise RuntimeError(
                            "The selected ComfyUI workflow is not trusted. Review its nodes and "
                            "create a trusted revision before execution."
                        )
                    dependency_errors = custom_node_dependency_errors(
                        session, revision.dependencies_json.get("custom_nodes")
                    )
                    if dependency_errors:
                        raise RuntimeError("; ".join(dependency_errors))
                    workflow = revision.api_graph_json
                    if run.settings_json.get("loras"):
                        resolved_loras = resolve_lora_stack(
                            session,
                            revision,
                            run.settings_json["loras"],
                        )
                        extension = workflow_lora_extension(revision)
                        if not extension:
                            raise RuntimeError(
                                "The selected workflow no longer provides its LoRA extension."
                            )
                        workflow = transform_lora_graph(
                            workflow,
                            extension,
                            [
                                {
                                    "comfy_name": item["comfy_name"],
                                    "model_strength": item["model_strength"],
                                    "clip_strength": item["clip_strength"],
                                }
                                for item in resolved_loras.provenance
                                if item["enabled"]
                            ],
                        )
                        expected_graph = (run.provenance_json.get("auxiliary_assets") or {}).get(
                            "effective_graph_sha256"
                        )
                        if expected_graph != resolved_loras.graph_sha256:
                            raise RuntimeError(
                                "The effective LoRA graph changed after this run was queued."
                            )
            request = MediaRequest(
                run_id=run.id,
                operation=run.operation,
                prompt=self._media_prompt(run),
                negative_prompt=str(run.settings_json.get("negative_prompt", "")) or None,
                input_paths=input_paths,
                workflow=workflow,
                parameters=run.settings_json,
                persistence_scope=self.persistence_scope,
                scope_id=self.scope_id,
            )
            assistant_id = run.assistant_message_id

        completed_assets = []
        preview_artifact_id: str | None = None
        async for event in self.engines.media.generate(request):
            if event.type in {"progress", "queued"}:
                with self.session_factory() as session:
                    job = session.get(Job, job_id)
                    message = session.get(Message, assistant_id)
                    if job and message:
                        update_job_progress(
                            job,
                            stage=event.phase,
                            stage_progress=event.progress,
                            queue_resource=job.queue_resource,
                        )
                        self._replace_parts(message, self._media_progress_parts(message, event))
                        session.commit()
                await self.scheduler.publish_job(job_id)
                await self.events.publish(
                    "generation.progress",
                    run_id,
                    {"progress": event.progress, "phase": event.phase, "job_id": job_id},
                )
            elif event.type == "preview" and event.preview:
                old_preview_id = preview_artifact_id
                with self.session_factory() as session:
                    message = session.get(Message, assistant_id)
                    job = session.get(Job, job_id)
                    if message and job:
                        update_job_progress(
                            job,
                            stage=event.phase or "preview",
                            stage_progress=event.progress,
                            queue_resource=job.queue_resource,
                        )
                        preview = self.artifacts.ingest_bytes(
                            session,
                            event.preview,
                            kind=ArtifactKind.THUMBNAIL,
                            media_type=_preview_media_type(event.preview),
                            original_name="generation-preview",
                            metadata={
                                "run_id": run_id,
                                "temporary_preview": True,
                            },
                        )
                        preview_artifact_id = preview.id
                        self._replace_parts(
                            message,
                            [
                                MessagePart(
                                    position=0,
                                    type=PartType.PROGRESS.value,
                                    text=event.phase.title() or "Preview",
                                    metadata_json={
                                        "progress": event.progress,
                                        "phase": event.phase or "preview",
                                    },
                                ),
                                MessagePart(
                                    position=1,
                                    type=PartType.IMAGE.value,
                                    artifact_id=preview.id,
                                    metadata_json={"preview": True},
                                ),
                            ],
                        )
                        session.commit()
                        if old_preview_id and old_preview_id != preview.id:
                            self.artifacts.delete_temporary_preview(session, old_preview_id)
                            session.commit()
                await self.scheduler.publish_job(job_id)
                await self.events.publish(
                    "generation.preview",
                    run_id,
                    {
                        "job_id": job_id,
                        "bytes": len(event.preview),
                        "artifact_id": preview_artifact_id,
                    },
                )
            elif event.type == "cancelled":
                return
            elif event.type == "complete":
                completed_assets.extend(event.assets)

        completed_assistant_id = assistant_id
        with self.session_factory() as session:
            message = session.get(Message, assistant_id)
            run = session.get(Run, run_id)
            job = session.get(Job, job_id)
            if not message or not run or not job:
                return
            parts: list[MessagePart] = []
            artifact_ids: list[str] = []
            output_provenance: list[dict[str, Any]] = []
            for generated in completed_assets:
                kind = ArtifactKind(generated.kind)
                artifact = self.artifacts.ingest_bytes(
                    session,
                    generated.content,
                    kind=kind,
                    media_type=generated.media_type,
                    original_name=generated.name,
                    metadata={
                        **generated.metadata,
                        "run_id": run.id,
                        "semantic_description": run.standalone_prompt,
                        "semantic_description_source": "generation_prompt",
                        "semantic_description_confidence": "intent-only",
                        "visual_contents_inspected": False,
                        "settings": run.settings_json,
                    },
                )
                # Derived-video helpers can spend minutes in ffmpeg. Persist the
                # content-addressed source before awaiting them so SQLite never
                # carries a write transaction across external process work.
                session.commit()
                poster_artifact_id: str | None = None
                proxy_artifact_id: str | None = None
                if generated.kind == "video":
                    playback_artifact = artifact
                    proxy_result = await self.artifacts.browser_video_proxy(artifact)
                    if proxy_result:
                        try:
                            proxy = self.artifacts.ingest_path(
                                session,
                                proxy_result.path,
                                kind=ArtifactKind.OTHER,
                                media_type=proxy_result.media_type,
                                original_name=proxy_result.original_name,
                                metadata={
                                    "run_id": run.id,
                                    "browser_proxy": True,
                                    "proxy_for": artifact.id,
                                },
                            )
                        finally:
                            proxy_result.discard()
                        proxy_artifact_id = proxy.id
                        playback_artifact = proxy
                        artifact.metadata_json = {
                            **artifact.metadata_json,
                            "browser_proxy_artifact_id": proxy.id,
                        }
                        # The poster helper awaits ffmpeg too. End the proxy
                        # transaction first; incomplete derived artifacts remain
                        # unreferenced and are handled by normal retention.
                        session.commit()
                    poster_content = await self.artifacts.video_poster(playback_artifact)
                    if poster_content:
                        poster = self.artifacts.ingest_bytes(
                            session,
                            poster_content,
                            kind=ArtifactKind.THUMBNAIL,
                            media_type="image/jpeg",
                            original_name=f"{generated.name}.poster.jpg",
                            metadata={"run_id": run.id, "poster_for": artifact.id},
                        )
                        poster_artifact_id = poster.id
                        artifact.metadata_json = {
                            **artifact.metadata_json,
                            "poster_artifact_id": poster.id,
                        }
                artifact_ids.append(artifact.id)
                output_provenance.append(
                    {
                        "artifact_id": artifact.id,
                        "sha256": artifact.sha256,
                        "kind": artifact.kind,
                        "media_type": artifact.media_type,
                        "size_bytes": artifact.size_bytes,
                        "poster_artifact_id": poster_artifact_id,
                        "browser_proxy_artifact_id": proxy_artifact_id,
                    }
                )
                parts.append(
                    MessagePart(
                        position=len(parts),
                        type=PartType.IMAGE.value
                        if generated.kind == "image"
                        else PartType.VIDEO.value,
                        artifact_id=artifact.id,
                        metadata_json={
                            "media_type": artifact.media_type,
                            "poster_artifact_id": poster_artifact_id,
                            "browser_proxy_artifact_id": proxy_artifact_id,
                        },
                    )
                )
            self._complete(session, run, job, {"artifact_ids": artifact_ids})
            run.provenance_json = {
                **run.provenance_json,
                "outputs": output_provenance,
                "timings": {"duration_ms": run.duration_ms},
            }
            parts.append(
                MessagePart(
                    position=len(parts),
                    type=PartType.GENERATION_METADATA.value,
                    metadata_json={"run_id": run.id, "provenance": run.provenance_json},
                )
            )
            self._replace_parts(message, parts)
            completed_assistant_id = self._finalize_response_revision(
                session,
                run,
                message,
                promote=True,
            )
            session.commit()
            if preview_artifact_id and preview_artifact_id not in artifact_ids:
                self.artifacts.delete_temporary_preview(session, preview_artifact_id)
                session.commit()
        await self.scheduler.publish_job(job_id)
        for artifact_id in artifact_ids:
            await self.events.publish(
                "artifact.ready", run_id, {"artifact_id": artifact_id, "job_id": job_id}
            )
        await self.events.publish(
            "run.completed",
            run_id,
            {
                "job_id": job_id,
                "assistant_message_id": completed_assistant_id,
            },
        )

    async def _fail(self, job_id: str, run_id: str, error: str) -> None:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            if not job or not run:
                return
            now = utcnow()
            job.status = JobStatus.FAILED.value
            job.error = error
            job.completed_at = now
            run.status = RunStatus.FAILED.value
            run.error = error
            run.completed_at = now
            self._set_work_status(session, run, JobStatus.FAILED.value, error=error)
            update_job_progress(job, stage="failed", indeterminate=True, now=now)
            message = session.get(Message, run.assistant_message_id)
            if message:
                preview_ids = self._temporary_preview_ids(message)
                message.status = MessageStatus.FAILED.value
                if run.operation == Operation.TEXT.value:
                    self._remove_chat_progress(message)
                    # Flush removed progress rows before reusing their positions
                    # for a terminal error part.
                    session.flush()
                    error_part = next(
                        (part for part in message.parts if part.type == PartType.ERROR.value),
                        None,
                    )
                    if error_part:
                        error_part.text = error
                    else:
                        message.parts.append(
                            MessagePart(
                                position=max((part.position for part in message.parts), default=-1)
                                + 1,
                                type=PartType.ERROR.value,
                                text=error,
                            )
                        )
                else:
                    self._replace_parts(
                        message,
                        [MessagePart(position=0, type=PartType.ERROR.value, text=error)],
                    )
                self._finalize_response_revision(
                    session,
                    run,
                    message,
                    promote=False,
                )
                session.flush()
                for artifact_id in preview_ids:
                    self.artifacts.delete_temporary_preview(session, artifact_id)
            session.commit()
        await self.scheduler.publish_job(job_id)
        await self.events.publish("run.failed", run_id, {"job_id": job_id, "error": error})

    def _complete(self, session: Session, run: Run, job: Job, result: dict[str, Any]) -> None:
        now = utcnow()
        run.status = RunStatus.COMPLETE.value
        run.completed_at = now
        if run.started_at:
            run.duration_ms = elapsed_milliseconds(run.started_at, now)
        message = session.get(Message, run.assistant_message_id)
        if message:
            message.status = MessageStatus.COMPLETE.value
        job.status = JobStatus.COMPLETE.value
        completed_progress(job, now=now)
        job.result_json = result
        job.completed_at = now
        self._set_work_status(session, run, JobStatus.COMPLETE.value)

    def _mark_cancelled(self, session: Session, job: Job) -> None:
        now = utcnow()
        job.status = JobStatus.CANCELLED.value
        job.completed_at = now
        if not job.run_id:
            return
        run = session.get(Run, job.run_id)
        if not run:
            return
        run.status = RunStatus.CANCELLED.value
        run.completed_at = now
        self._set_work_status(session, run, JobStatus.CANCELLED.value)
        update_job_progress(job, stage="cancelled", indeterminate=True, now=now)
        message = session.get(Message, run.assistant_message_id)
        if message:
            preview_ids = self._temporary_preview_ids(message)
            message.status = MessageStatus.CANCELLED.value
            if run.operation == Operation.TEXT.value:
                self._remove_chat_progress(message)
            else:
                # Cancellation is a state, not a generation failure. Remove
                # transient progress/preview parts and let the client render
                # the neutral cancelled subtext below any durable content.
                self._replace_parts(
                    message,
                    [
                        self._message_part_copy(part)
                        for part in message.parts
                        if part.type != PartType.PROGRESS.value
                        and part.artifact_id not in preview_ids
                    ],
                )
            self._finalize_response_revision(
                session,
                run,
                message,
                promote=False,
            )
            session.flush()
            for artifact_id in preview_ids:
                self.artifacts.delete_temporary_preview(session, artifact_id)

    @staticmethod
    def _set_work_status(
        session: Session,
        run: Run,
        status: str,
        *,
        error: str | None = None,
    ) -> None:
        if run.work_step_id:
            step = session.get(WorkStep, run.work_step_id)
            if step:
                step.status = status
                step.error = error
        if run.work_plan_id:
            plan = session.get(WorkPlan, run.work_plan_id)
            if plan:
                session.flush()
                refresh_plan_status(session, plan.id)
                plan.summary_json = {
                    **plan.summary_json,
                    "status_counts": plan_status_summary(session, plan.id),
                }

    @staticmethod
    def _temporary_preview_ids(message: Message) -> list[str]:
        return [
            part.artifact_id
            for part in message.parts
            if part.artifact_id and part.metadata_json.get("preview")
        ]

    @staticmethod
    def _media_progress_parts(message: Message, event: MediaEvent) -> list[MessagePart]:
        parts = [
            MessagePart(
                position=0,
                type=PartType.PROGRESS.value,
                text=event.phase.title(),
                metadata_json={
                    "progress": event.progress,
                    "phase": event.phase,
                    "indeterminate": event.type == "queued"
                    or (event.progress <= 0 and "prepar" in event.phase.casefold()),
                },
            )
        ]
        preview = next(
            (
                part
                for part in message.parts
                if part.artifact_id and part.metadata_json.get("preview")
            ),
            None,
        )
        if preview:
            parts.append(
                MessagePart(
                    position=1,
                    type=PartType.IMAGE.value,
                    artifact_id=preview.artifact_id,
                    metadata_json=dict(preview.metadata_json),
                )
            )
        return parts

    def _persist_streamed_text(self, message_id: str, text: str) -> None:
        with self.session_factory() as session:
            message = session.get(Message, message_id)
            if not message:
                return
            ConversationOrchestrator._remove_chat_progress(message)
            text_part = next(
                (part for part in message.parts if part.type == PartType.TEXT.value), None
            )
            if text_part:
                text_part.text = text
            else:
                ConversationOrchestrator._replace_parts(
                    message,
                    [MessagePart(position=0, type=PartType.TEXT.value, text=text)],
                )
            session.commit()

    @staticmethod
    def _remove_chat_progress(message: Message) -> None:
        for part in list(message.parts):
            if (
                part.type == PartType.PROGRESS.value
                and part.metadata_json.get("activity") == "chat"
            ):
                message.parts.remove(part)

    @staticmethod
    def _replace_parts(message: Message, parts: list[MessagePart]) -> None:
        session = object_session(message)
        message.parts.clear()
        if session is not None:
            # Flush orphan deletes before inserting replacement rows that reuse
            # the unique (message_id, position) values.
            session.flush()
        message.parts.extend(parts)

    @staticmethod
    def _message_part_copy(part: MessagePart | ResponseRevisionPart) -> MessagePart:
        return MessagePart(
            position=part.position,
            type=part.type,
            text=part.text,
            artifact_id=part.artifact_id,
            metadata_json=dict(part.metadata_json),
        )

    @staticmethod
    def _revision_part_copy(part: MessagePart) -> ResponseRevisionPart:
        return ResponseRevisionPart(
            position=part.position,
            type=part.type,
            text=part.text,
            artifact_id=part.artifact_id,
            metadata_json=dict(part.metadata_json),
        )

    def _ensure_response_revision(
        self,
        session: Session,
        message: Message,
    ) -> ResponseRevision:
        if message.active_response_revision_id:
            existing = session.get(ResponseRevision, message.active_response_revision_id)
            if existing and existing.message_id == message.id:
                return existing
        source_run = session.scalar(
            select(Run)
            .where(Run.assistant_message_id == message.id)
            .order_by(Run.created_at.asc(), Run.id.asc())
            .limit(1)
        )
        revision = ResponseRevision(
            message_id=message.id,
            run_id=source_run.id if source_run else None,
            sequence=max(
                (item.sequence for item in message.response_revisions),
                default=0,
            )
            + 1,
            status=message.status,
            parts=[self._revision_part_copy(part) for part in message.parts],
        )
        session.add(revision)
        session.flush()
        message.active_response_revision_id = revision.id
        return revision

    def _finalize_response_revision(
        self,
        session: Session,
        run: Run,
        staged_message: Message,
        *,
        promote: bool,
    ) -> str:
        replacement = run.provenance_json.get("response_replacement")
        if not isinstance(replacement, dict):
            revision = self._ensure_response_revision(session, staged_message)
            revision.parts.clear()
            session.flush()
            revision.parts.extend(
                self._revision_part_copy(part)
                for part in sorted(staged_message.parts, key=lambda item: item.position)
            )
            revision.status = staged_message.status
            return staged_message.id
        message_id = replacement.get("message_id")
        revision_id = replacement.get("revision_id")
        if not isinstance(message_id, str) or not isinstance(revision_id, str):
            return staged_message.id
        message = session.get(Message, message_id)
        target_revision = session.get(ResponseRevision, revision_id)
        if (
            not message
            or not target_revision
            or target_revision.message_id != message.id
            or target_revision.run_id != run.id
        ):
            raise RuntimeError("response revision target is invalid")
        target_revision.parts.clear()
        session.flush()
        target_revision.parts.extend(
            self._revision_part_copy(part)
            for part in sorted(staged_message.parts, key=lambda item: item.position)
        )
        target_revision.status = staged_message.status
        if promote:
            self._replace_parts(
                message,
                [
                    self._message_part_copy(part)
                    for part in sorted(target_revision.parts, key=lambda item: item.position)
                ],
            )
            message.status = MessageStatus.COMPLETE.value
            message.active_response_revision_id = target_revision.id
        return message.id

    def select_response_revision(
        self,
        session: Session,
        message_id: str,
        revision_id: str,
    ) -> Message:
        message = session.get(Message, message_id)
        revision = session.get(ResponseRevision, revision_id)
        if (
            not message
            or not message.transcript_visible
            or message.role != MessageRole.ASSISTANT.value
        ):
            raise LookupError("assistant message not found")
        if not revision or revision.message_id != message.id:
            raise LookupError("response revision not found")
        if revision.status != MessageStatus.COMPLETE.value:
            raise ValueError("only a completed response revision can be selected")
        self._replace_parts(
            message,
            [
                self._message_part_copy(part)
                for part in sorted(revision.parts, key=lambda item: item.position)
            ],
        )
        message.status = MessageStatus.COMPLETE.value
        message.active_response_revision_id = revision.id
        session.commit()
        selected = session.scalar(
            select(Message)
            .options(
                selectinload(Message.parts).selectinload(MessagePart.artifact),
                selectinload(Message.response_revisions)
                .selectinload(ResponseRevision.parts)
                .selectinload(ResponseRevisionPart.artifact),
            )
            .where(Message.id == message.id)
        )
        if not selected:
            raise LookupError("assistant message not found")
        return selected

    @staticmethod
    def _ancestor_messages(
        session: Session,
        chat_id: str,
        head_message_id: str | None,
        *,
        limit: int | None = None,
    ) -> list[Message]:
        """Return one active message ancestry from newest to oldest.

        Message timestamps cannot identify a branch: edited turns leave sibling
        messages in the same chat. Following parent links is therefore required
        anywhere that resolves conversational media.
        """

        rows: list[Message] = []
        current_id = head_message_id
        visited: set[str] = set()
        while current_id and current_id not in visited and (limit is None or len(rows) < limit):
            visited.add(current_id)
            message = session.scalar(
                select(Message)
                .options(selectinload(Message.parts).selectinload(MessagePart.artifact))
                .where(Message.id == current_id, Message.chat_id == chat_id)
            )
            if not message:
                break
            rows.append(message)
            current_id = message.parent_id
        return rows

    @staticmethod
    def _pending_parent_step_id(
        session: Session,
        chat_id: str,
        parent_message_id: str | None,
    ) -> str | None:
        if not parent_message_id:
            return None
        parent = session.get(Message, parent_message_id)
        if (
            not parent
            or parent.chat_id != chat_id
            or parent.role != MessageRole.ASSISTANT.value
            or parent.status != MessageStatus.PENDING.value
        ):
            return None
        return session.scalar(
            select(Run.work_step_id)
            .join(Job, Job.run_id == Run.id)
            .where(
                Run.chat_id == chat_id,
                Run.assistant_message_id == parent_message_id,
                Run.work_step_id.is_not(None),
                Job.status.in_(
                    [
                        JobStatus.QUEUED.value,
                        JobStatus.RUNNING.value,
                        JobStatus.PAUSED.value,
                    ]
                ),
            )
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(1)
        )

    @classmethod
    def _latest_completed_context_head(
        cls,
        session: Session,
        chat_id: str,
        parent_message_id: str | None,
    ) -> str | None:
        return next(
            (
                message.id
                for message in cls._ancestor_messages(
                    session,
                    chat_id,
                    parent_message_id,
                )
                if message.status == MessageStatus.COMPLETE.value
            ),
            None,
        )

    @classmethod
    def _latest_image_context(
        cls,
        session: Session,
        chat_id: str,
        head_message_id: str | None,
    ) -> tuple[str | None, str | None]:
        for message in cls._ancestor_messages(session, chat_id, head_message_id):
            if (
                message.role != MessageRole.ASSISTANT.value
                or message.status != MessageStatus.COMPLETE.value
            ):
                continue
            run = session.scalar(
                select(Run).where(
                    Run.chat_id == chat_id,
                    Run.assistant_message_id == message.id,
                    Run.status == RunStatus.COMPLETE.value,
                    Run.operation != Operation.TEXT.value,
                )
            )
            if not run:
                continue
            image_parts = [
                part
                for part in message.parts
                if (
                    part.type == PartType.IMAGE.value
                    and part.artifact_id
                    and not part.metadata_json.get("preview")
                )
            ]
            if image_parts:
                image_parts.sort(key=lambda part: (part.position, part.id))
                prompt = run.standalone_prompt.strip() or None
                return image_parts[-1].artifact_id, prompt
        return None, None

    @classmethod
    def _has_prior_image(
        cls,
        session: Session,
        chat_id: str,
        head_message_id: str | None,
    ) -> bool:
        artifact_id, _prompt = cls._latest_image_context(
            session,
            chat_id,
            head_message_id,
        )
        return artifact_id is not None

    @classmethod
    def _latest_image(
        cls,
        session: Session,
        chat_id: str,
        head_message_id: str | None,
    ) -> str | None:
        artifact_id, _prompt = cls._latest_image_context(
            session,
            chat_id,
            head_message_id,
        )
        return artifact_id

    @classmethod
    def _latest_media_prompt(
        cls,
        session: Session,
        chat_id: str,
        head_message_id: str | None,
    ) -> str | None:
        for message in cls._ancestor_messages(session, chat_id, head_message_id):
            if (
                message.role != MessageRole.ASSISTANT.value
                or message.status != MessageStatus.COMPLETE.value
            ):
                continue
            run = session.scalar(
                select(Run).where(
                    Run.chat_id == chat_id,
                    Run.assistant_message_id == message.id,
                    Run.status == RunStatus.COMPLETE.value,
                    Run.operation != Operation.TEXT.value,
                )
            )
            if run and run.standalone_prompt.strip():
                return run.standalone_prompt.strip()
        return None

    @staticmethod
    def _generation_offer_for_message(message: Message | None) -> GenerationOffer | None:
        if (
            not message
            or message.role != MessageRole.ASSISTANT.value
            or message.status != MessageStatus.COMPLETE.value
            or not message.transcript_visible
        ):
            return None
        for part in message.parts:
            if part.type != PartType.GENERATION_METADATA.value:
                continue
            offer = generation_offer_from_metadata(part.metadata_json.get("generation_offer"))
            if offer:
                return offer
        return None

    @staticmethod
    def _routing_context(
        session: Session, chat: Chat, parent_message_id: str | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if chat.project_id:
            project = session.get(Project, chat.project_id)
            if project and project.instructions:
                messages.append({"role": "system", "content": project.instructions})
        rows = ConversationOrchestrator._ancestor_messages(
            session,
            chat.id,
            parent_message_id,
            limit=8,
        )
        for message in reversed(rows):
            content = ConversationOrchestrator._message_context_text(
                message,
                ConversationOrchestrator._message_input_artifacts(session, message),
            )
            if content:
                messages.append({"role": message.role, "content": content})
        return messages

    @classmethod
    def _profile_for_operation(
        cls,
        session: Session,
        chat: Chat,
        operation: Operation,
        prompt: str,
    ) -> tuple[ModelProfile | None, dict[str, Any]]:
        if operation == Operation.TEXT:
            selected_id = chat.active_chat_profile_id
        elif "image" in operation.value and "video" not in operation.value:
            selected_id = chat.active_image_profile_id
        else:
            selected_id = chat.active_video_profile_id
        role = cls._role_for_operation(operation)
        profiles = list(
            session.scalars(
                select(ModelProfile)
                .where(ModelProfile.role == role)
                .order_by(ModelProfile.updated_at.desc(), ModelProfile.id)
            ).all()
        )

        if selected_id and selected_id != AUTO_PROFILE_ID:
            selected = next((profile for profile in profiles if profile.id == selected_id), None)
            if selected:
                return selected, {
                    "mode": "explicit",
                    "profile_id": selected.id,
                    "profile_name": selected.name,
                }

        default = next((profile for profile in profiles if profile.is_default), None)
        if selected_id != AUTO_PROFILE_ID:
            return default, {
                "mode": "default",
                "profile_id": default.id if default else None,
                "profile_name": default.name if default else None,
            }

        installed = [
            profile
            for profile in profiles
            if profile.model_install_id
            and (install := session.get(ModelInstall, profile.model_install_id))
            and install.active
        ]
        candidates = installed or profiles
        prompt_text = prompt.casefold()
        prompt_terms = cls._selection_terms(prompt_text)
        ranked: list[tuple[int, ModelProfile, list[str]]] = []
        for profile in candidates:
            use_case = profile.use_case.strip().casefold()
            use_case_terms = cls._selection_terms(use_case)
            matches = sorted(prompt_terms & use_case_terms)
            score = len(matches) * 10
            if use_case and use_case in prompt_text:
                score += 25
            score += len(prompt_terms & cls._selection_terms(profile.name.casefold()))
            ranked.append((score, profile, matches))
        ranked.sort(
            key=lambda item: (
                item[0],
                item[1].is_default,
                item[1].updated_at,
                item[1].id,
            ),
            reverse=True,
        )
        if ranked:
            score, selected, matches = ranked[0]
        else:
            score, selected, matches = 0, default, []
        return selected, {
            "mode": "auto",
            "profile_id": selected.id if selected else None,
            "profile_name": selected.name if selected else None,
            "profile_use_case": selected.use_case if selected else "",
            "score": score,
            "matched_terms": matches,
            "fallback": score == 0,
        }

    def _profile_has_verified_vision(self, session: Session, profile: ModelProfile) -> bool:
        if not profile.model_install_id:
            return False
        install = session.get(ModelInstall, profile.model_install_id)
        if not install:
            return False
        evidence = current_capability_evidence(
            session,
            install,
            self.engines.settings,
            self.processes.runtimes,
        )
        return "image" in evidence_input_modalities(evidence)

    @staticmethod
    def _vision_profile_provenance(
        session: Session,
        profile_id: str | None,
    ) -> dict[str, Any] | None:
        profile = session.get(ModelProfile, profile_id) if profile_id else None
        install = (
            session.get(ModelInstall, profile.model_install_id)
            if profile and profile.model_install_id
            else None
        )
        source = (
            session.get(ModelSource, install.source_id) if install and install.source_id else None
        )
        if not profile or not install:
            return None
        return {
            "profile_id": profile.id,
            "profile_name": profile.name,
            "install_id": install.id,
            "engine": install.engine,
            "component_hashes": install.manifest_json.get("expected_sha256") or {},
            "source": (
                {
                    "provider": source.provider,
                    "remote_id": source.remote_id,
                    "revision": source.revision,
                }
                if source
                else None
            ),
        }

    def _vision_profile_for_chat(
        self,
        session: Session,
        chat: Chat,
        text_profile: ModelProfile | None,
    ) -> ModelProfile | None:
        if text_profile and self._profile_has_verified_vision(session, text_profile):
            return text_profile
        profiles = list(
            session.scalars(
                select(ModelProfile)
                .where(ModelProfile.role == "chat")
                .order_by(ModelProfile.updated_at.desc(), ModelProfile.id)
            ).all()
        )
        verified = [
            profile
            for profile in profiles
            if profile.model_install_id
            and (install := session.get(ModelInstall, profile.model_install_id))
            and install.active
            and self._profile_has_verified_vision(session, profile)
        ]
        selected_id = chat.active_vision_profile_id
        if selected_id and selected_id != AUTO_PROFILE_ID:
            return next((profile for profile in verified if profile.id == selected_id), None)
        if selected_id == AUTO_PROFILE_ID:
            return next((profile for profile in verified if profile.is_default), None) or (
                verified[0] if verified else None
            )
        return None

    @staticmethod
    def _selection_terms(value: str) -> set[str]:
        stop_words = {
            "about",
            "and",
            "are",
            "for",
            "from",
            "into",
            "model",
            "that",
            "the",
            "this",
            "use",
            "with",
        }
        return {
            SELECTION_TERM_ALIASES.get(term, term)
            for term in re.findall(r"[a-z0-9][a-z0-9+#.-]{2,}", value)
            if term not in stop_words
        }

    @classmethod
    def _default_preset(cls, session: Session, operation: Operation) -> GenerationPreset | None:
        return session.scalar(
            select(GenerationPreset)
            .where(
                GenerationPreset.role == cls._role_for_operation(operation),
                GenerationPreset.is_default.is_(True),
            )
            .order_by(GenerationPreset.updated_at.desc(), GenerationPreset.id)
            .limit(1)
        )

    @staticmethod
    def _scoped_generation_settings(
        owner: Project | Chat | None,
        role: str,
    ) -> dict[str, Any]:
        scoped = owner.generation_settings_json if owner else {}
        if not isinstance(scoped, dict):
            return {}
        settings = scoped.get(role)
        return dict(settings) if isinstance(settings, dict) else {}

    @staticmethod
    def _bound_preset(
        session: Session,
        owner: Project | Chat | None,
        role: str,
    ) -> GenerationPreset | None:
        bindings = owner.generation_preset_ids_json if owner else {}
        if not isinstance(bindings, dict):
            return None
        preset_id = bindings.get(role)
        if not isinstance(preset_id, str):
            return None
        preset = session.get(GenerationPreset, preset_id)
        return preset if preset and preset.role == role else None

    @staticmethod
    def _role_for_operation(operation: Operation) -> str:
        if operation == Operation.TEXT:
            return "chat"
        if "video" in operation.value:
            return "video"
        return "image"

    async def request_settings_for_operation(
        self,
        operation: Operation,
        values: dict[str, Any],
        *,
        input_schema: dict[str, Any] | None = None,
        engine: str | None = None,
    ) -> dict[str, Any]:
        role = self._role_for_operation(operation)
        fields = workflow_settings(
            await self.engines.settings_for_role(role, engine=engine),
            input_schema,
        )
        request_fields = [field for field in fields if field.scope != "load"]
        return compatible_stored_settings(values, request_fields)

    @staticmethod
    def _initial_output_parts(
        operation: Operation,
        ordinal: int,
        output_count: int,
    ) -> list[MessagePart]:
        progress_metadata: dict[str, Any] = {
            "progress": 0,
            "phase": "queued",
        }
        if output_count > 1:
            progress_metadata.update({"output_index": ordinal, "output_count": output_count})
        if operation == Operation.TEXT:
            progress_metadata["activity"] = "chat"
            return [
                MessagePart(position=0, type=PartType.TEXT.value, text=""),
                MessagePart(
                    position=1,
                    type=PartType.PROGRESS.value,
                    text="Queued",
                    metadata_json=progress_metadata,
                ),
            ]
        progress_metadata["indeterminate"] = True
        return [
            MessagePart(
                position=0,
                type=PartType.PROGRESS.value,
                text="Queued",
                metadata_json=progress_metadata,
            )
        ]

    @staticmethod
    def _model_provenance(
        session: Session,
        profile: ModelProfile | None,
    ) -> dict[str, Any] | None:
        if not profile or not profile.model_install_id:
            return None
        install = session.get(ModelInstall, profile.model_install_id)
        source = (
            session.get(ModelSource, install.source_id) if install and install.source_id else None
        )
        if not install:
            return None
        return {
            "profile_id": profile.id,
            "profile_name": profile.name,
            "profile_use_case": profile.use_case,
            "install_id": install.id,
            "engine": install.engine,
            "local_path": install.local_path,
            "size_bytes": install.size_bytes,
            "manifest": install.manifest_json,
            "source": (
                {
                    "provider": source.provider,
                    "remote_id": source.remote_id,
                    "revision": source.revision,
                    "metadata": source.metadata_json,
                }
                if source
                else None
            ),
        }

    @classmethod
    def _media_plan_estimate(
        cls,
        operation: Operation,
        settings: dict[str, Any],
        output_count: int,
    ) -> dict[str, int]:
        if "video" in operation.value:
            per_output = cls._video_estimate(settings)
            per_output_work = int(per_output["work_units"])
            per_output_bytes = int(per_output["estimated_output_bytes"]) + int(
                per_output["estimated_intermediate_bytes"]
            )
        else:
            width = max(1, int(settings.get("width", 1024)))
            height = max(1, int(settings.get("height", 1024)))
            steps = max(1, int(settings.get("steps", 30)))
            per_output_work = width * height * steps
            raw_bytes = width * height * 4
            per_output_bytes = max(1_000_000, raw_bytes * 3)
        return {
            "output_count": output_count,
            "work_units_per_output": per_output_work,
            "work_units": per_output_work * output_count,
            "estimated_bytes_per_output": per_output_bytes,
            "estimated_bytes": per_output_bytes * output_count,
        }

    @staticmethod
    def _video_estimate(settings: dict[str, Any]) -> dict[str, int | float]:
        width = int(settings.get("width", 768))
        height = int(settings.get("height", 432))
        frames = int(settings.get("frames", 49))
        fps = max(float(settings.get("fps", 24)), 1)
        steps = int(settings.get("steps", 30))
        raw_bytes = width * height * frames * 3
        return {
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "duration_seconds": round(frames / fps, 2),
            "work_units": width * height * frames * steps,
            "estimated_output_bytes": max(1_000_000, raw_bytes // 45),
            "estimated_intermediate_bytes": raw_bytes * 2,
        }

    @staticmethod
    def _image_edit_provenance(
        resolution: ImageEditStrengthResolution | None,
    ) -> dict[str, Any] | None:
        if resolution is None:
            return None
        return {
            "policy": "preserve_unrequested_details_v1",
            "default_change_strength_applied": resolution.default_applied,
            "strength": resolution.provenance(),
        }

    @staticmethod
    def _media_prompt(run: Run) -> str:
        edit = run.provenance_json.get("image_edit")
        if (
            run.operation != Operation.IMAGE_TO_IMAGE.value
            or not isinstance(edit, dict)
            or edit.get("policy") != "preserve_unrequested_details_v1"
        ):
            return run.standalone_prompt
        return (
            "Apply the requested edit visibly to the supplied image. Preserve areas "
            "that the edit does not affect. Keep each person's facial identity, hair, "
            "skin tone, body proportions, and pose unless the request explicitly changes "
            "them. Do not simply reproduce the source unchanged. Requested edit: "
            f"{run.standalone_prompt}"
        )

    @staticmethod
    def _job_kind(operation: Operation) -> JobKind:
        if operation == Operation.TEXT:
            return JobKind.CHAT
        if "video" in operation.value:
            return JobKind.VIDEO
        return JobKind.IMAGE

    def _workflow_for_operation(
        self,
        session: Session,
        operation: Operation,
        *,
        project_id: str | None = None,
        model_install_id: str | None = None,
    ) -> WorkflowRevision | None:
        if operation == Operation.TEXT:
            return None
        if project_id:
            project = session.get(Project, project_id)
            revision_id = None
            if project:
                revision_id = (
                    project.video_workflow_revision_id
                    if "video" in operation.value
                    else project.image_workflow_revision_id
                )
            if revision_id:
                revision = session.get(WorkflowRevision, revision_id)
                definition = (
                    session.get(WorkflowDefinition, revision.workflow_id) if revision else None
                )
                if revision and definition and definition.operation == operation.value:
                    return revision
        definitions = session.scalars(
            select(WorkflowDefinition)
            .where(WorkflowDefinition.operation == operation.value)
            .order_by(WorkflowDefinition.created_at.desc())
        ).all()
        generic: list[WorkflowRevision] = []
        for definition in definitions:
            if not definition.current_revision_id:
                continue
            revision = session.get(WorkflowRevision, definition.current_revision_id)
            if not revision or not self._workflow_matches_engine(revision):
                continue
            dependencies = revision.dependencies_json.get("model_install_ids")
            declared_installs = (
                {str(item) for item in dependencies} if isinstance(dependencies, list) else set()
            )
            if declared_installs:
                if model_install_id in declared_installs:
                    return revision
                continue
            generic.append(revision)
        return generic[0] if generic else None

    def _workflow_matches_engine(self, revision: WorkflowRevision) -> bool:
        engine = self.engines.settings.media_engine
        return revision.engine == engine and (engine == "mock" or bool(revision.api_graph_json))

    @staticmethod
    def _input_part_type(artifact: Artifact) -> str:
        media_type = artifact.media_type.casefold()
        if media_type.startswith("image/"):
            return PartType.IMAGE.value
        if media_type.startswith("video/"):
            return PartType.VIDEO.value
        return PartType.ATTACHMENT.value

    @staticmethod
    def input_artifact_ids_for_run(session: Session, run: Run) -> list[str]:
        """Return normalized turn inputs, with provenance fallback for old data."""

        user_message = session.scalar(
            select(Message)
            .options(selectinload(Message.parts))
            .where(Message.id == run.user_message_id, Message.chat_id == run.chat_id)
        )
        durable_ids = (
            [
                part.artifact_id
                for part in sorted(user_message.parts, key=lambda part: part.position)
                if (part.artifact_id and part.metadata_json.get("input_reference") is True)
            ]
            if user_message
            else []
        )
        provenance = run.provenance_json if isinstance(run.provenance_json, dict) else {}
        dependency_ids = provenance.get("resolved_dependency_artifact_ids")
        resolved_dependency_ids = (
            [value for value in dependency_ids if isinstance(value, str)]
            if isinstance(dependency_ids, list)
            else []
        )
        if durable_ids or resolved_dependency_ids:
            return list(dict.fromkeys([*durable_ids, *resolved_dependency_ids]))
        legacy_ids = provenance.get("input_artifact_ids")
        if not isinstance(legacy_ids, list):
            return []
        return list(
            dict.fromkeys(artifact_id for artifact_id in legacy_ids if isinstance(artifact_id, str))
        )

    @staticmethod
    def _resolve_step_inputs(session: Session, run: Run) -> None:
        if not run.work_step_id or not run.work_plan_id:
            return
        step = session.get(WorkStep, run.work_step_id)
        if not step:
            raise RuntimeError("The planned work step is missing.")
        text_inputs: list[dict[str, str]] = []
        artifact_ids: list[str] = []
        for binding in step.input_bindings_json:
            binding_type = binding.get("type")
            if binding_type not in {"step_output.text", "step_output.artifact"}:
                continue
            source_step_id = binding.get("source_step_id")
            if not isinstance(source_step_id, str):
                raise RuntimeError("A planned step input is missing its source.")
            source_step = session.get(WorkStep, source_step_id)
            if (
                not source_step
                or source_step.plan_id != run.work_plan_id
                or source_step.status != JobStatus.COMPLETE.value
                or not source_step.run_id
            ):
                raise RuntimeError("A required planned step did not complete successfully.")
            source_run = session.get(Run, source_step.run_id)
            source_message = (
                session.get(Message, source_run.assistant_message_id) if source_run else None
            )
            if (
                not source_run
                or source_run.chat_id != run.chat_id
                or not source_message
                or source_message.chat_id != run.chat_id
            ):
                raise RuntimeError("A planned step output crossed its chat boundary.")
            if binding_type == "step_output.text":
                text = "\n".join(
                    part.text
                    for part in sorted(source_message.parts, key=lambda value: value.position)
                    if part.type == PartType.TEXT.value and part.text
                ).strip()
                if not text:
                    raise RuntimeError("A required text step produced no usable text.")
                if len(text) > 50_000:
                    raise RuntimeError("A required text step exceeded the dependency budget.")
                text_inputs.append({"source_step_id": source_step_id, "text": text})
                continue
            binding_artifact_ids: list[str] = []
            for part in sorted(source_message.parts, key=lambda value: value.position):
                if (
                    not part.artifact_id
                    or part.metadata_json.get("preview")
                    or part.metadata_json.get("input_reference")
                ):
                    continue
                artifact = session.get(Artifact, part.artifact_id)
                if not artifact:
                    continue
                if run.operation in {
                    Operation.IMAGE_TO_IMAGE.value,
                    Operation.IMAGE_TO_VIDEO.value,
                } and not artifact.media_type.casefold().startswith("image/"):
                    continue
                if (
                    run.operation == Operation.TEXT.value
                    and not artifact.media_type.casefold().startswith(("image/", "video/"))
                ):
                    continue
                binding_artifact_ids.append(artifact.id)
            if not binding_artifact_ids:
                raise RuntimeError("A required media step produced no compatible artifact.")
            artifact_ids.extend(binding_artifact_ids)

        provenance = run.provenance_json if isinstance(run.provenance_json, dict) else {}
        compiled_step = provenance.get("compiled_step")
        compiled_prompt = compiled_step.get("prompt") if isinstance(compiled_step, dict) else None
        base_prompt: str = (
            compiled_prompt if isinstance(compiled_prompt, str) else run.standalone_prompt
        )
        if run.operation != Operation.TEXT.value and text_inputs:
            context = "\n\n".join(
                f"Output from {item['source_step_id']}:\n{item['text']}" for item in text_inputs
            )
            run.standalone_prompt = f"{base_prompt}\n\nUse this prior text as context:\n{context}"
        else:
            run.standalone_prompt = base_prompt
        run.provenance_json = {
            **provenance,
            "resolved_dependency_text": text_inputs,
            "resolved_dependency_artifact_ids": list(dict.fromkeys(artifact_ids)),
        }

    @classmethod
    def _message_input_artifacts(
        cls,
        session: Session,
        message: Message,
    ) -> list[Artifact]:
        """Load a user's explicit inputs, including legacy provenance-only runs."""

        if message.role != MessageRole.USER.value:
            return []
        direct_ids = [
            part.artifact_id
            for part in sorted(message.parts, key=lambda part: part.position)
            if part.artifact_id and part.metadata_json.get("input_reference") is True
        ]
        artifact_ids = list(dict.fromkeys(direct_ids))
        if not artifact_ids:
            run = session.scalar(
                select(Run)
                .where(Run.chat_id == message.chat_id, Run.user_message_id == message.id)
                .order_by(Run.created_at.desc(), Run.id.desc())
                .limit(1)
            )
            if run:
                artifact_ids = cls.input_artifact_ids_for_run(session, run)
        return [
            artifact
            for artifact_id in artifact_ids
            if (artifact := session.get(Artifact, artifact_id)) is not None
        ]

    @staticmethod
    def _context_messages(session: Session, run: Run) -> list[dict[str, str]]:
        messages, _ = ConversationOrchestrator._context_messages_with_sources(session, run)
        return messages

    @staticmethod
    def _context_messages_with_sources(
        session: Session,
        run: Run,
    ) -> tuple[list[dict[str, str]], list[str | None]]:
        chat = session.get(Chat, run.chat_id)
        if not chat:
            return [], []
        messages: list[dict[str, str]] = []
        source_message_ids: list[str | None] = []
        if chat.project_id:
            project = session.get(Project, chat.project_id)
            if project and project.instructions:
                messages.append({"role": "system", "content": project.instructions})
                source_message_ids.append(None)
        plan = session.get(WorkPlan, run.work_plan_id) if run.work_plan_id else None
        if plan:
            rows = [
                message
                for message in reversed(
                    ConversationOrchestrator._ancestor_messages(
                        session, run.chat_id, plan.context_head_message_id
                    )
                )
                if message.status == MessageStatus.COMPLETE.value
            ]
            user_message = session.scalar(
                select(Message)
                .options(selectinload(Message.parts))
                .where(Message.id == run.user_message_id, Message.chat_id == run.chat_id)
            )
            if user_message and all(message.id != user_message.id for message in rows):
                rows.append(user_message)
        else:
            rows = list(
                reversed(
                    ConversationOrchestrator._ancestor_messages(
                        session,
                        run.chat_id,
                        run.user_message_id,
                    )
                )
            )
        for message in rows:
            text = ConversationOrchestrator._message_context_text(
                message,
                ConversationOrchestrator._message_input_artifacts(session, message),
            )
            if text:
                messages.append({"role": message.role, "content": text})
                source_message_ids.append(message.id)
        provenance = run.provenance_json if isinstance(run.provenance_json, dict) else {}
        compiled_step = provenance.get("compiled_step")
        if run.operation == Operation.TEXT.value and isinstance(compiled_step, dict):
            dependency_text = provenance.get("resolved_dependency_text")
            if isinstance(dependency_text, list):
                for item in dependency_text:
                    if (
                        isinstance(item, dict)
                        and isinstance(item.get("text"), str)
                        and item["text"].strip()
                    ):
                        messages.append(
                            {
                                "role": MessageRole.ASSISTANT.value,
                                "content": item["text"].strip(),
                            }
                        )
                        source_message_ids.append(None)
            step_prompt = compiled_step.get("prompt")
            if isinstance(step_prompt, str) and step_prompt.strip():
                messages.append({"role": MessageRole.USER.value, "content": step_prompt.strip()})
                source_message_ids.append(None)
        return messages, source_message_ids

    @staticmethod
    def _message_context_text(
        message: Message,
        input_artifacts: list[Artifact] | None = None,
    ) -> str:
        text = "\n".join(part.text for part in message.parts if part.text).strip()
        attachment_lines: list[str] = []
        for artifact in input_artifacts or []:
            if artifact.media_type.casefold().startswith("image/"):
                kind = "image"
            elif artifact.media_type.casefold().startswith("video/"):
                kind = "video"
            else:
                kind = "file"
            name = " ".join((artifact.original_name or kind).split())[:240]
            description = artifact.metadata_json.get("semantic_description")
            detail = ""
            if isinstance(description, str) and description.strip():
                normalized = " ".join(description.split())
                if artifact.metadata_json.get("semantic_description_source") in {
                    None,
                    "generation_prompt",
                }:
                    detail = (
                        ". Generation request (visual contents not inspected): "
                        f"{normalized[:1_000]}"
                    )
                else:
                    detail = f". Description: {normalized[:1_000]}"
            attachment_lines.append(f"[Attached {kind}: {name}{detail}]")
        if text or attachment_lines:
            return "\n".join([value for value in (text, *attachment_lines) if value])

        prompt = ""
        for part in message.parts:
            if part.type != PartType.GENERATION_METADATA.value:
                continue
            provenance = part.metadata_json.get("provenance")
            routing = provenance.get("routing") if isinstance(provenance, dict) else None
            candidate = routing.get("standalone_prompt") if isinstance(routing, dict) else None
            if isinstance(candidate, str) and candidate.strip():
                prompt = " ".join(candidate.split())
                if len(prompt) > 1_000:
                    prompt = f"{prompt[:997]}..."
                break

        media: list[str] = []
        for part_type, singular in (
            (PartType.IMAGE.value, "image"),
            (PartType.VIDEO.value, "video"),
        ):
            count = sum(
                part.type == part_type and part.metadata_json.get("input_reference") is not True
                for part in message.parts
            )
            if count:
                media.append(singular if count == 1 else f"{count} {singular}s")
        if not media:
            return ""
        summary = f"Generated {' and '.join(media)}"
        return (
            f'{summary} requested with this prompt (visual contents not inspected): "{prompt}".'
            if prompt
            else f"{summary}; visual contents not inspected."
        )

    @staticmethod
    def _accepted_for_run(session: Session, run: Run) -> TurnAccepted:
        refreshed = session.scalar(select(Run).where(Run.id == run.id))
        if not refreshed:
            raise LookupError("run not found")
        user_message = session.scalar(
            select(Message)
            .options(
                selectinload(Message.parts).selectinload(MessagePart.artifact),
                selectinload(Message.response_revisions)
                .selectinload(ResponseRevision.parts)
                .selectinload(ResponseRevisionPart.artifact),
            )
            .where(Message.id == refreshed.user_message_id)
        )
        assistant_message = session.scalar(
            select(Message)
            .options(
                selectinload(Message.parts).selectinload(MessagePart.artifact),
                selectinload(Message.response_revisions)
                .selectinload(ResponseRevision.parts)
                .selectinload(ResponseRevisionPart.artifact),
            )
            .where(Message.id == refreshed.assistant_message_id)
        )
        if not user_message or not assistant_message:
            raise LookupError("run messages not found")
        return TurnAccepted(
            run=RunOut.model_validate(refreshed),
            user_message=MessageOut.model_validate(user_message),
            assistant_message=MessageOut.model_validate(assistant_message),
        )
