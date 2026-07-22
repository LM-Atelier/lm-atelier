from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, object_session, selectinload

from .adapters.base import ChatRequest, MediaRequest
from .artifacts import ArtifactStore
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
    utcnow,
)
from .engines import EngineRegistry
from .events import EventBroker
from .models import (
    Artifact,
    Chat,
    Job,
    Message,
    MessagePart,
    ModelInstall,
    ModelProfile,
    Project,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
)
from .processes import ProcessSupervisor
from .routing import ModalityRouter
from .scheduler import ResourceScheduler
from .schemas import MessageOut, RunOut, TurnAccepted, TurnRequest
from .settings_registry import (
    CHAT_SETTINGS,
    IMAGE_SETTINGS,
    VIDEO_SETTINGS,
    defaults,
    resolve_settings,
    validate_settings,
)

logger = logging.getLogger(__name__)


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

    def recover_interrupted(self) -> None:
        with SessionLocal() as session:
            jobs = session.scalars(
                select(Job).where(Job.status.in_([JobStatus.RUNNING.value, JobStatus.QUEUED.value]))
            ).all()
            for job in jobs:
                job.status = JobStatus.INTERRUPTED.value
                job.phase = "interrupted by application restart"
                job.error = "The application restarted before this job completed."
                job.completed_at = utcnow()
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
                        message = session.get(Message, run.assistant_message_id)
                        if message:
                            message.status = MessageStatus.FAILED.value
                            self._replace_parts(
                                message,
                                [
                                    MessagePart(
                                        position=0,
                                        type=PartType.ERROR.value,
                                        text=job.error,
                                    )
                                ],
                            )
            session.commit()

    def create_turn(self, session: Session, chat_id: str, request: TurnRequest) -> TurnAccepted:
        if request.idempotency_key:
            existing = session.scalar(
                select(Run).where(Run.idempotency_key == request.idempotency_key)
            )
            if existing:
                return self._accepted_for_run(session, existing)

        chat = session.get(Chat, chat_id)
        if not chat:
            raise LookupError("chat not found")
        parent_message_id = request.parent_message_id
        if parent_message_id:
            parent = session.get(Message, parent_message_id)
            if not parent or parent.chat_id != chat_id:
                raise LookupError("parent message not found in this chat")
        else:
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
        for artifact_id in request.input_artifact_ids:
            if not session.get(Artifact, artifact_id):
                raise LookupError(f"input artifact not found: {artifact_id}")

        mode = request.mode or RoutingMode(chat.routing_mode)
        has_prior_image = self._has_prior_image(session, chat.id)
        plan = self.router.plan(
            text=request.text,
            mode=mode,
            input_artifact_ids=request.input_artifact_ids,
            has_prior_image=has_prior_image,
        )
        resolved_input_ids = list(request.input_artifact_ids)
        if plan.operation in {Operation.IMAGE_TO_IMAGE, Operation.IMAGE_TO_VIDEO}:
            if not resolved_input_ids:
                prior_image = self._latest_image(session, chat.id)
                if prior_image:
                    resolved_input_ids.append(prior_image)
            prior_prompt = self._latest_media_prompt(session, chat.id)
            if prior_prompt:
                plan.standalone_prompt = f"{prior_prompt}. Follow-up instruction: {request.text}"
        plan.input_artifact_ids = resolved_input_ids

        user_message = Message(
            chat_id=chat.id,
            parent_id=parent_message_id,
            role=MessageRole.USER.value,
            status=MessageStatus.COMPLETE.value,
            parts=[MessagePart(position=0, type=PartType.TEXT.value, text=request.text)],
        )
        if plan.operation == Operation.TEXT:
            initial_parts = [MessagePart(position=0, type=PartType.TEXT.value, text="")]
        else:
            initial_parts = [
                MessagePart(
                    position=0,
                    type=PartType.PROGRESS.value,
                    text="Queued",
                    metadata_json={"progress": 0, "phase": "queued"},
                )
            ]
        assistant_message = Message(
            chat_id=chat.id,
            parent_id=None,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.PENDING.value,
            parts=initial_parts,
        )
        session.add_all([user_message, assistant_message])
        session.flush()
        assistant_message.parent_id = user_message.id
        chat.active_head_message_id = assistant_message.id

        profile_id = self._profile_for_operation(chat, plan.operation)
        profile = session.get(ModelProfile, profile_id) if profile_id else None
        fields = self._fields_for_operation(plan.operation)
        effective_settings = resolve_settings(
            defaults(fields), profile.request_settings_json if profile else None, request.settings
        )
        effective_settings = validate_settings(effective_settings, fields)
        workflow_revision = self._workflow_for_operation(session, plan.operation)

        run = Run(
            idempotency_key=request.idempotency_key,
            chat_id=chat.id,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            operation=plan.operation.value,
            status=RunStatus.QUEUED.value,
            standalone_prompt=plan.standalone_prompt,
            profile_id=profile_id,
            workflow_revision_id=workflow_revision.id if workflow_revision else None,
            settings_json=effective_settings,
            provenance_json={
                "routing": plan.model_dump(mode="json"),
                "input_artifact_ids": resolved_input_ids,
            },
        )
        session.add(run)
        session.flush()
        job = Job(
            kind=self._job_kind(plan.operation).value,
            status=JobStatus.QUEUED.value,
            run_id=run.id,
            progress=0,
            phase="queued",
            payload_json={"operation": plan.operation.value},
        )
        session.add(job)
        if chat.title == "New chat":
            chat.title = request.text.strip().replace("\n", " ")[:72] or "New chat"
        session.commit()
        accepted = self._accepted_for_run(session, run)
        self.start(job.id, run.id)
        return accepted

    def start(self, job_id: str, run_id: str) -> None:
        if job_id in self._tasks and not self._tasks[job_id].done():
            return
        task = asyncio.create_task(self._execute(job_id, run_id), name=f"local-lm-{job_id}")
        self._tasks[job_id] = task
        task.add_done_callback(lambda finished: self._task_done(job_id, finished))

    def _task_done(self, job_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(job_id, None)
        if task.cancelled():
            return
        exception = task.exception()
        if exception:
            logger.exception(
                "Background job %s terminated unexpectedly", job_id, exc_info=exception
            )

    async def cancel(self, job_id: str) -> bool:
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
            self._mark_cancelled(session, job)
            session.commit()
        await self.events.publish("run.cancelled", job.run_id, {"job_id": job_id})
        return True

    async def _execute(self, job_id: str, run_id: str) -> None:
        try:
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                run = session.get(Run, run_id)
                if not job or not run:
                    return
                job.status = JobStatus.RUNNING.value
                job.phase = "starting"
                job.started_at = utcnow()
                job.attempt += 1
                run.status = RunStatus.RUNNING.value
                run.started_at = utcnow()
                session.commit()
            await self.events.publish("run.created", run_id, {"job_id": job_id})
            await self.events.publish(
                "plan.selected",
                run_id,
                {"operation": run.operation, "prompt": run.standalone_prompt},
            )
            resume_chat_profile = await self._prepare_device_handoff(run.operation)
            try:
                async with self.scheduler.lease("primary"):
                    if run.operation == Operation.TEXT.value:
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
            await self._fail(job_id, run_id, str(exc))

    async def _execute_chat(self, job_id: str, run_id: str) -> None:
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
            }
            session.commit()
            request = ChatRequest(
                run_id=run.id,
                messages=messages,
                settings=request_settings,
            )
            assistant_id = run.assistant_message_id

        accumulated = ""
        completion_metadata: dict[str, Any] = {}
        last_persisted_length = 0
        last_persisted_at = time.monotonic()
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
                    len(accumulated) - last_persisted_length >= 32
                    or now - last_persisted_at >= 0.25
                ):
                    self._persist_streamed_text(assistant_id, accumulated)
                    last_persisted_length = len(accumulated)
                    last_persisted_at = now
            elif event.type == "cancelled":
                return
            elif event.type == "error":
                raise RuntimeError(str(event.data.get("error", "chat engine failed")))
            elif event.type in {"usage", "complete"}:
                completion_metadata.update(event.data)

        with SessionLocal() as session:
            message = session.get(Message, assistant_id)
            run = session.get(Run, run_id)
            job = session.get(Job, job_id)
            if not message or not run or not job:
                return
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
            run.provenance_json = {
                **run.provenance_json,
                "context": context_metadata,
                "completion": completion_metadata,
            }
            metadata_part = next(
                (part for part in message.parts if part.type == PartType.GENERATION_METADATA.value),
                None,
            )
            metadata = {
                "run_id": run.id,
                "context": context_metadata,
                "completion": completion_metadata,
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
            self._complete(
                session,
                run,
                job,
                {"characters": len(accumulated), "context": context_metadata},
            )
            session.commit()
        await self.events.publish("run.completed", run_id, {"job_id": job_id})

    async def _prepare_chat_context(
        self, session: Session, run: Run
    ) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
        messages = self._context_messages(session, run)
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
        }
        return messages, request_settings, metadata

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
        await self.processes.stop("chat")
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

    async def _execute_media(self, job_id: str, run_id: str) -> None:
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            if not run:
                return
            input_paths: list[Path] = []
            for artifact_id in run.provenance_json.get("input_artifact_ids", []):
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
        async for event in self.engines.media.generate(request):
            if event.type in {"progress", "queued"}:
                with SessionLocal() as session:
                    job = session.get(Job, job_id)
                    message = session.get(Message, assistant_id)
                    if job and message:
                        job.progress = event.progress
                        job.phase = event.phase
                        self._replace_parts(
                            message,
                            [
                                MessagePart(
                                    position=0,
                                    type=PartType.PROGRESS.value,
                                    text=event.phase.title(),
                                    metadata_json={
                                        "progress": event.progress,
                                        "phase": event.phase,
                                    },
                                )
                            ],
                        )
                        session.commit()
                await self.events.publish(
                    "generation.progress",
                    run_id,
                    {"progress": event.progress, "phase": event.phase, "job_id": job_id},
                )
            elif event.type == "preview" and event.preview:
                await self.events.publish(
                    "generation.preview", run_id, {"job_id": job_id, "bytes": len(event.preview)}
                )
            elif event.type == "cancelled":
                return
            elif event.type == "complete":
                completed_assets.extend(event.assets)

        with SessionLocal() as session:
            message = session.get(Message, assistant_id)
            run = session.get(Run, run_id)
            job = session.get(Job, job_id)
            if not message or not run or not job:
                return
            parts: list[MessagePart] = []
            artifact_ids: list[str] = []
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
                        "settings": run.settings_json,
                    },
                )
                artifact_ids.append(artifact.id)
                parts.append(
                    MessagePart(
                        position=len(parts),
                        type=PartType.IMAGE.value
                        if generated.kind == "image"
                        else PartType.VIDEO.value,
                        artifact_id=artifact.id,
                        metadata_json={"media_type": artifact.media_type},
                    )
                )
            parts.append(
                MessagePart(
                    position=len(parts),
                    type=PartType.GENERATION_METADATA.value,
                    metadata_json={"run_id": run.id},
                )
            )
            self._replace_parts(message, parts)
            self._complete(session, run, job, {"artifact_ids": artifact_ids})
            session.commit()
        for artifact_id in artifact_ids:
            await self.events.publish(
                "artifact.ready", run_id, {"artifact_id": artifact_id, "job_id": job_id}
            )
        await self.events.publish("run.completed", run_id, {"job_id": job_id})

    async def _fail(self, job_id: str, run_id: str, error: str) -> None:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            if not job or not run:
                return
            now = utcnow()
            job.status = JobStatus.FAILED.value
            job.phase = "failed"
            job.error = error
            job.completed_at = now
            run.status = RunStatus.FAILED.value
            run.error = error
            run.completed_at = now
            message = session.get(Message, run.assistant_message_id)
            if message:
                message.status = MessageStatus.FAILED.value
                self._replace_parts(
                    message,
                    [MessagePart(position=0, type=PartType.ERROR.value, text=error)],
                )
            session.commit()
        await self.events.publish("run.failed", run_id, {"job_id": job_id, "error": error})

    def _complete(self, session: Session, run: Run, job: Job, result: dict[str, Any]) -> None:
        now = utcnow()
        run.status = RunStatus.COMPLETE.value
        run.completed_at = now
        if run.started_at:
            run.duration_ms = int((time.time() - run.started_at.timestamp()) * 1000)
        message = session.get(Message, run.assistant_message_id)
        if message:
            message.status = MessageStatus.COMPLETE.value
        job.status = JobStatus.COMPLETE.value
        job.progress = 1
        job.phase = "complete"
        job.result_json = result
        job.completed_at = now

    def _mark_cancelled(self, session: Session, job: Job) -> None:
        now = utcnow()
        job.status = JobStatus.CANCELLED.value
        job.phase = "cancelled"
        job.completed_at = now
        if not job.run_id:
            return
        run = session.get(Run, job.run_id)
        if not run:
            return
        run.status = RunStatus.CANCELLED.value
        run.completed_at = now
        message = session.get(Message, run.assistant_message_id)
        if message:
            message.status = MessageStatus.CANCELLED.value
            self._replace_parts(
                message,
                [MessagePart(position=0, type=PartType.ERROR.value, text="Generation cancelled")],
            )

    @staticmethod
    def _persist_streamed_text(message_id: str, text: str) -> None:
        with SessionLocal() as session:
            message = session.get(Message, message_id)
            if not message:
                return
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
    def _replace_parts(message: Message, parts: list[MessagePart]) -> None:
        session = object_session(message)
        message.parts.clear()
        if session is not None:
            # Flush orphan deletes before inserting replacement rows that reuse
            # the unique (message_id, position) values.
            session.flush()
        message.parts.extend(parts)

    @staticmethod
    def _has_prior_image(session: Session, chat_id: str) -> bool:
        return (
            session.scalar(
                select(MessagePart.id)
                .join(Message)
                .where(Message.chat_id == chat_id, MessagePart.type == PartType.IMAGE.value)
                .limit(1)
            )
            is not None
        )

    @staticmethod
    def _latest_image(session: Session, chat_id: str) -> str | None:
        return session.scalar(
            select(MessagePart.artifact_id)
            .join(Message)
            .where(
                Message.chat_id == chat_id,
                MessagePart.type == PartType.IMAGE.value,
                MessagePart.artifact_id.is_not(None),
            )
            .order_by(MessagePart.created_at.desc())
            .limit(1)
        )

    @staticmethod
    def _latest_media_prompt(session: Session, chat_id: str) -> str | None:
        return session.scalar(
            select(Run.standalone_prompt)
            .where(Run.chat_id == chat_id, Run.operation != Operation.TEXT.value)
            .order_by(Run.created_at.desc())
            .limit(1)
        )

    @staticmethod
    def _profile_for_operation(chat: Chat, operation: Operation) -> str | None:
        if operation == Operation.TEXT:
            return chat.active_chat_profile_id
        if "image" in operation.value and "video" not in operation.value:
            return chat.active_image_profile_id
        return chat.active_video_profile_id

    @staticmethod
    def _fields_for_operation(operation: Operation):  # type: ignore[no-untyped-def]
        if operation == Operation.TEXT:
            return CHAT_SETTINGS
        if "video" in operation.value:
            return VIDEO_SETTINGS
        return IMAGE_SETTINGS

    @staticmethod
    def _job_kind(operation: Operation) -> JobKind:
        if operation == Operation.TEXT:
            return JobKind.CHAT
        if "video" in operation.value:
            return JobKind.VIDEO
        return JobKind.IMAGE

    @staticmethod
    def _workflow_for_operation(session: Session, operation: Operation) -> WorkflowRevision | None:
        if operation == Operation.TEXT:
            return None
        definition = session.scalar(
            select(WorkflowDefinition)
            .where(WorkflowDefinition.operation == operation.value)
            .order_by(WorkflowDefinition.created_at)
        )
        if not definition or not definition.current_revision_id:
            return None
        return session.get(WorkflowRevision, definition.current_revision_id)

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
        rows: list[Message] = []
        current_id: str | None = run.user_message_id
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
            rows.append(message)
            current_id = message.parent_id
        rows.reverse()
        for message in rows:
            text = "\n".join(part.text for part in message.parts if part.text).strip()
            if text:
                messages.append({"role": message.role, "content": text})
        return messages

    @staticmethod
    def _accepted_for_run(session: Session, run: Run) -> TurnAccepted:
        refreshed = session.scalar(select(Run).where(Run.id == run.id))
        if not refreshed:
            raise LookupError("run not found")
        user_message = session.scalar(
            select(Message)
            .options(selectinload(Message.parts).selectinload(MessagePart.artifact))
            .where(Message.id == refreshed.user_message_id)
        )
        assistant_message = session.scalar(
            select(Message)
            .options(selectinload(Message.parts).selectinload(MessagePart.artifact))
            .where(Message.id == refreshed.assistant_message_id)
        )
        if not user_message or not assistant_message:
            raise LookupError("run messages not found")
        return TurnAccepted(
            run=RunOut.model_validate(refreshed),
            user_message=MessageOut.model_validate(user_message),
            assistant_message=MessageOut.model_validate(assistant_message),
        )
