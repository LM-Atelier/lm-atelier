from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, object_session, selectinload

from .adapters.base import ChatRequest, MediaEvent, MediaRequest
from .artifacts import ArtifactStore
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
from .processes import ProcessSupervisor
from .profile_service import AUTO_PROFILE_ID
from .progress import completed_progress, update_job_progress
from .routing import ModalityRouter, RouteConfirmationRequired
from .scheduler import ResourceScheduler
from .schemas import MessageOut, RunOut, TurnAccepted, TurnRequest, WorkerStatus
from .settings_registry import (
    compatible_stored_settings,
    resolve_generation_settings,
    validate_settings,
    workflow_settings,
)

logger = logging.getLogger(__name__)

IDEMPOTENCY_CLAIM_WAIT_SECONDS = 120.0
MAX_VISION_IMAGES = 4
MAX_VISION_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VISION_TOTAL_BYTES = 32 * 1024 * 1024
MAX_PENDING_WORK_PER_CHAT = 32
PENDING_OUTPUT_REFERENCE = re.compile(
    r"\b(?:"
    r"(?:that|this|it|its)(?:\s+(?:image|video|answer|response|story|result|output))?"
    r"|(?:previous|last|above|earlier)\s+(?:image|video|answer|response|story|result|output)"
    r"|based\s+on\s+(?:that|this|it|the\s+(?:previous|last|above|earlier|story|response))"
    r")\b",
    re.IGNORECASE,
)
VISION_MEDIA_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}


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
    ) -> None:
        self.engines = engines
        self.artifacts = artifacts
        self.events = events
        self.scheduler = scheduler
        self.processes = processes
        self.router = ModalityRouter()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._chat_guards: dict[str, asyncio.Lock] = {}
        self._chat_planner_ready = asyncio.Event()
        self._chat_planner_ready.set()

    def recover_interrupted(self) -> None:
        queued: list[tuple[str, str]] = []
        with SessionLocal() as session:
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
    ) -> TurnAccepted:
        async with self.chat_guard(chat_id):
            return await self._create_turn(
                session,
                chat_id,
                request,
                use_explicit_parent=use_explicit_parent,
                replacement_message_id=replacement_message_id,
                source_action=source_action,
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
    ) -> TurnAccepted:
        chat = session.get(Chat, chat_id)
        if not chat:
            raise LookupError("chat not found")
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

        mode = request.mode or RoutingMode(chat.routing_mode)
        prior_image, prior_image_prompt = self._latest_image_context(
            session,
            chat.id,
            context_head_message_id,
        )
        has_prior_image = prior_image is not None
        routing_context = self._routing_context(session, chat, context_head_message_id)
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
        effective_preset = preset_layers[-1] if preset_layers else None
        if plan.operation != Operation.TEXT and effective_settings.get("seed") == -1:
            effective_settings["seed"] = secrets.randbelow(2_147_483_648)
        generation_estimate = (
            self._video_estimate(effective_settings) if "video" in plan.operation.value else None
        )
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
        if plan.operation == Operation.TEXT:
            initial_parts = [
                MessagePart(position=0, type=PartType.TEXT.value, text=""),
                MessagePart(
                    position=1,
                    type=PartType.PROGRESS.value,
                    text="Queued",
                    metadata_json={"activity": "chat", "progress": 0, "phase": "queued"},
                ),
            ]
        else:
            initial_parts = [
                MessagePart(
                    position=0,
                    type=PartType.PROGRESS.value,
                    text="Queued",
                    metadata_json={
                        "progress": 0,
                        "phase": "queued",
                        "indeterminate": True,
                    },
                )
            ]
        assistant_message = Message(
            chat_id=chat.id,
            parent_id=None,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.PENDING.value,
            transcript_visible=replacement_message is None,
            parts=initial_parts,
        )
        session.add_all([user_message, assistant_message])
        session.flush()
        assistant_message.parent_id = user_message.id
        if replacement_message is None:
            chat.active_head_message_id = assistant_message.id

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
            persistence_scope="durable",
            status=JobStatus.QUEUED.value,
            context_head_message_id=context_head_message_id,
            transcript_sequence=transcript_sequence,
            priority=10 if plan.operation == Operation.TEXT else 0,
            planner_version="legacy-turn-v1",
            failure_policy="stop_dependents",
            summary_json={
                "operation": plan.operation.value,
                "step_count": 1,
                "source_action": source_action,
                "user_message_id": user_message.id,
                "assistant_message_id": assistant_message.id,
                "dependency_step_ids": (
                    [pending_dependency_step_id] if references_pending_output else []
                ),
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
        work_step = WorkStep(
            plan=work_plan,
            ordinal=1,
            operation=plan.operation.value,
            status=JobStatus.QUEUED.value,
            prompt=plan.standalone_prompt,
            profile_id=profile_id,
            workflow_revision_id=workflow_revision.id if workflow_revision else None,
            settings_json=effective_settings,
            input_bindings_json=input_bindings,
            output_contract_json=[{"slot": "response", "type": output_type}],
            queue_class=queue_class,
        )
        session.add(work_plan)
        session.flush()
        if references_pending_output and pending_dependency_step_id:
            session.add(
                WorkStepDependency(
                    step_id=work_step.id,
                    depends_on_step_id=pending_dependency_step_id,
                )
            )
        run = Run(
            idempotency_key=request.idempotency_key,
            chat_id=chat.id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            work_plan_id=work_plan.id,
            work_step_id=work_step.id,
            operation=plan.operation.value,
            status=RunStatus.QUEUED.value,
            standalone_prompt=plan.standalone_prompt,
            profile_id=profile_id,
            workflow_revision_id=workflow_revision.id if workflow_revision else None,
            settings_json=effective_settings,
            provenance_json={
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
                "resolved_settings": effective_settings,
                "generation_estimate": generation_estimate,
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
            },
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
            queue_ticket=f"{transcript_sequence:020d}:{run.id}",
            enqueued_at=utcnow(),
            payload_json={"operation": plan.operation.value},
        )
        update_job_progress(
            job,
            stage="queued",
            queue_resource=queue_class,
            queue_position=0,
            queue_length=1,
            indeterminate=True,
        )
        session.add(job)
        if chat.title == "New chat":
            chat.title = request.text.strip().replace("\n", " ")[:72] or "New chat"
        session.commit()
        accepted = self._accepted_for_run(session, run)
        await self.events.publish(
            "work_plan.created",
            work_plan.id,
            {
                "plan_id": work_plan.id,
                "step_id": work_step.id,
                "run_id": run.id,
                "job_id": job.id,
                "chat_id": chat.id,
            },
        )
        self.start(job.id, run.id)
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
        with SessionLocal() as session:
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
        with SessionLocal() as session:
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
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            if not job or job.status in {
                JobStatus.COMPLETE.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            }:
                return False
            if job.run_id:
                run = session.get(Run, job.run_id)
                if run and run.operation == Operation.TEXT.value:
                    await self.engines.chat.cancel(run.id)
                elif run:
                    await self.engines.media.cancel(run.id)
            task = self._tasks.get(job_id)
            if task:
                task.cancel()
                cancelled_task = task
            self._mark_cancelled(session, job)
            session.commit()
        if cancelled_task:
            await asyncio.gather(cancelled_task, return_exceptions=True)
        await self.scheduler.publish_job(job_id)
        await self.events.publish("run.cancelled", job.run_id, {"job_id": job_id})
        return True

    async def close(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _execute(self, job_id: str, run_id: str) -> None:
        try:
            with SessionLocal() as session:
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
                with SessionLocal() as session:
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
                    if resume_chat_profile:
                        await self._resume_chat_worker(resume_chat_profile)
        except asyncio.CancelledError:
            with SessionLocal() as session:
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
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            if not run:
                return
            messages, request_settings, context_metadata = await self._prepare_chat_context(
                session, run
            )
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

        completed_assistant_id = assistant_id
        with SessionLocal() as session:
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
                text_part.text = accumulated.rstrip()
            else:
                self._replace_parts(
                    message,
                    [MessagePart(position=0, type=PartType.TEXT.value, text=accumulated.rstrip())],
                )
            context_metadata = dict(run.provenance_json.get("context", {}))
            if usage := completion_metadata.get("usage"):
                context_metadata["usage"] = usage
            text_output = accumulated.rstrip()
            self._complete(
                session,
                run,
                job,
                {"characters": len(accumulated), "context": context_metadata},
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
        with SessionLocal() as session:
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
        if self.engines.settings.chat_engine != "llama.cpp":
            return None
        with SessionLocal() as session:
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
        if self.engines.settings.chat_engine != "llama.cpp":
            return True
        if not self.processes.settings.llama_executable:
            return False
        if not self._chat_planner_ready.is_set():
            return False
        status = next(item for item in self.processes.statuses() if item.name == "chat")
        return status.state == "ready"

    async def _prepare_chat_context(
        self, session: Session, run: Run
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        messages = self._context_messages(session, run)
        capabilities = await self.engines.chat_capabilities()
        vision_metadata: dict[str, Any] = {
            "available": "image" in capabilities.input_modalities,
            "images_included": 0,
            "artifact_ids": [],
        }
        if vision_metadata["available"]:
            messages, vision_metadata = self._attach_visual_context(session, run, messages)
        profile = session.get(ModelProfile, run.profile_id) if run.profile_id else None
        context_limit = int(
            (profile.load_settings_json if profile else {}).get("context_length", 8192)
        )
        requested_output = int(run.settings_json.get("max_tokens", 1024))
        safety_tokens = min(128, max(32, context_limit // 100))
        maximum_output = max(1, context_limit - safety_tokens - 64)
        output_limit = min(requested_output, maximum_output)
        input_budget = max(64, context_limit - output_limit - safety_tokens)

        input_tokens = await self.engines.chat.count_tokens(messages)
        omitted = 0
        system_messages = 1 if messages and messages[0].get("role") == "system" else 0
        while input_tokens > input_budget and len(messages) > system_messages + 1:
            remove_count = 1
            if (
                messages[system_messages].get("role") == MessageRole.USER.value
                and len(messages) > system_messages + 2
                and messages[system_messages + 1].get("role") == MessageRole.ASSISTANT.value
            ):
                remove_count = 2
            del messages[system_messages : system_messages + remove_count]
            omitted += remove_count
            input_tokens = await self.engines.chat.count_tokens(messages)
        if input_tokens > input_budget:
            raise ValueError(
                "The current instructions and message exceed this profile's context window. "
                "Increase Context length, reduce Maximum output, or shorten the message."
            )

        request_settings = {**run.settings_json, "max_tokens": output_limit}
        metadata = {
            "policy": "preserve-system-and-newest",
            "context_limit": context_limit,
            "input_budget": input_budget,
            "input_tokens": input_tokens,
            "requested_output_tokens": requested_output,
            "output_limit": output_limit,
            "safety_tokens": safety_tokens,
            "messages_included": len(messages),
            "messages_omitted": omitted,
            "output_adjusted": output_limit != requested_output,
            "vision": vision_metadata,
        }
        return messages, request_settings, metadata

    def _attach_visual_context(
        self,
        session: Session,
        run: Run,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        candidates = self._visual_context_artifacts(session, run)
        user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == MessageRole.USER.value
            ),
            None,
        )
        if user_index is None:
            return messages, {
                "available": True,
                "images_included": 0,
                "artifact_ids": [],
                "bytes_included": 0,
                "images_skipped": len(candidates),
            }
        encoded: list[tuple[Artifact, str, str]] = []
        total_bytes = 0
        skipped = 0
        for artifact in candidates:
            if len(encoded) >= MAX_VISION_IMAGES:
                skipped += 1
                continue
            try:
                path, detected_type, _disposition = self.artifacts.delivery_metadata(artifact)
                size = path.stat().st_size
                if (
                    detected_type not in VISION_MEDIA_TYPES
                    or size > MAX_VISION_IMAGE_BYTES
                    or total_bytes + size > MAX_VISION_TOTAL_BYTES
                ):
                    skipped += 1
                    continue
                content = path.read_bytes()
                if (
                    len(content) != artifact.size_bytes
                    or hashlib.sha256(content).hexdigest() != artifact.sha256
                ):
                    skipped += 1
                    continue
            except (OSError, ValueError):
                skipped += 1
                continue
            encoded.append(
                (
                    artifact,
                    detected_type,
                    base64.b64encode(content).decode("ascii"),
                )
            )
            total_bytes += size

        if encoded:
            user_index = next(
                (
                    index
                    for index in range(len(messages) - 1, -1, -1)
                    if messages[index].get("role") == MessageRole.USER.value
                ),
                None,
            )
            if user_index is not None:
                text = messages[user_index].get("content", "")
                text = text if isinstance(text, str) else ""
                messages[user_index] = {
                    "role": MessageRole.USER.value,
                    "content": [
                        {"type": "text", "text": text},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{payload}",
                                    "detail": "auto",
                                },
                            }
                            for _artifact, media_type, payload in encoded
                        ],
                    ],
                }

        return messages, {
            "available": True,
            "images_included": len(encoded),
            "artifact_ids": [artifact.id for artifact, _media_type, _payload in encoded],
            "bytes_included": total_bytes,
            "images_skipped": skipped,
        }

    @classmethod
    def _visual_context_artifacts(cls, session: Session, run: Run) -> list[Artifact]:
        """Return explicit current inputs, then the newest prior branch visual."""

        current = session.get(Message, run.user_message_id)
        candidates: list[Artifact] = []
        for artifact in cls._message_input_artifacts(session, current) if current else []:
            if artifact.media_type.casefold().startswith("video/"):
                poster_id = artifact.metadata_json.get("poster_artifact_id")
                poster = session.get(Artifact, poster_id) if isinstance(poster_id, str) else None
                if poster:
                    candidates.append(poster)
                continue
            candidates.append(artifact)
        seen = {artifact.id for artifact in candidates}

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
                selected: Artifact | None = referenced
                if part.type == PartType.VIDEO.value:
                    poster_id = referenced.metadata_json.get("poster_artifact_id")
                    selected = (
                        session.get(Artifact, poster_id) if isinstance(poster_id, str) else None
                    )
                    if not selected:
                        continue
                if selected is None:
                    continue
                if (
                    part.type in {PartType.IMAGE.value, PartType.VIDEO.value}
                    and selected.id not in seen
                ):
                    candidates.append(selected)
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
            with SessionLocal() as session:
                profile = session.get(ModelProfile, profile_id)
                install = (
                    session.get(ModelInstall, profile.model_install_id)
                    if profile and profile.model_install_id
                    else None
                )
                if not profile or not install:
                    return
                await self.processes.load_chat(profile, install)
        except Exception:
            logger.exception("Could not reload chat profile %s after media handoff", profile_id)
        finally:
            self._chat_planner_ready.set()

    async def _execute_media(self, job_id: str, run_id: str) -> None:
        if self.engines.settings.media_engine == "comfyui":
            status = next(item for item in self.processes.statuses() if item.name == "media")
            if not status.running or status.state != "ready":
                with SessionLocal() as session:
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
                await self.processes.start_media()
        with SessionLocal() as session:
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
            request = MediaRequest(
                run_id=run.id,
                operation=run.operation,
                prompt=run.standalone_prompt,
                negative_prompt=str(run.settings_json.get("negative_prompt", "")) or None,
                input_paths=input_paths,
                workflow=workflow,
                parameters=run.settings_json,
            )
            assistant_id = run.assistant_message_id

        completed_assets = []
        preview_artifact_id: str | None = None
        async for event in self.engines.media.generate(request):
            if event.type in {"progress", "queued"}:
                with SessionLocal() as session:
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
                with SessionLocal() as session:
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
        with SessionLocal() as session:
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
                poster_artifact_id: str | None = None
                proxy_artifact_id: str | None = None
                if generated.kind == "video":
                    playback_artifact = artifact
                    proxy_result = await self.artifacts.browser_video_proxy(artifact)
                    if proxy_result:
                        proxy_content, proxy_media_type, proxy_name = proxy_result
                        proxy = self.artifacts.ingest_bytes(
                            session,
                            proxy_content,
                            kind=ArtifactKind.OTHER,
                            media_type=proxy_media_type,
                            original_name=proxy_name,
                            metadata={
                                "run_id": run.id,
                                "browser_proxy": True,
                                "proxy_for": artifact.id,
                            },
                        )
                        proxy_artifact_id = proxy.id
                        playback_artifact = proxy
                        artifact.metadata_json = {
                            **artifact.metadata_json,
                            "browser_proxy_artifact_id": proxy.id,
                        }
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
        with SessionLocal() as session:
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
                self._replace_parts(
                    message,
                    [
                        MessagePart(
                            position=0, type=PartType.ERROR.value, text="Generation cancelled"
                        )
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
                plan.status = status

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

    @staticmethod
    def _persist_streamed_text(message_id: str, text: str) -> None:
        with SessionLocal() as session:
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
        if durable_ids:
            return list(dict.fromkeys(durable_ids))
        provenance = run.provenance_json if isinstance(run.provenance_json, dict) else {}
        legacy_ids = provenance.get("input_artifact_ids")
        if not isinstance(legacy_ids, list):
            return []
        return list(
            dict.fromkeys(artifact_id for artifact_id in legacy_ids if isinstance(artifact_id, str))
        )

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
        chat = session.get(Chat, run.chat_id)
        if not chat:
            return []
        messages: list[dict[str, str]] = []
        if chat.project_id:
            project = session.get(Project, chat.project_id)
            if project and project.instructions:
                messages.append({"role": "system", "content": project.instructions})
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
        return messages

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
