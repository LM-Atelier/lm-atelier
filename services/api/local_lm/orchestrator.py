from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, object_session, selectinload

from .adapters.base import ChatRequest, MediaRequest
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
    Job,
    Message,
    MessagePart,
    ModelInstall,
    ModelProfile,
    ModelSource,
    Project,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
)
from .processes import ProcessSupervisor
from .routing import ModalityRouter, RouteConfirmationRequired
from .scheduler import ResourceScheduler
from .schemas import MessageOut, RunOut, TurnAccepted, TurnRequest, WorkerStatus
from .settings_registry import (
    CHAT_SETTINGS,
    IMAGE_SETTINGS,
    VIDEO_SETTINGS,
    defaults,
    resolve_settings,
    validate_settings,
)

logger = logging.getLogger(__name__)


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

    async def create_turn(
        self,
        session: Session,
        chat_id: str,
        request: TurnRequest,
        *,
        use_explicit_parent: bool = False,
    ) -> TurnAccepted:
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
        for artifact_id in request.input_artifact_ids:
            if not session.get(Artifact, artifact_id):
                raise LookupError(f"input artifact not found: {artifact_id}")

        mode = request.mode or RoutingMode(chat.routing_mode)
        has_prior_image = self._has_prior_image(session, chat.id)
        plan = await self.router.plan_with_model(
            adapter=self.engines.chat,
            text=request.text,
            mode=mode,
            input_artifact_ids=request.input_artifact_ids,
            has_prior_image=has_prior_image,
            conversation=self._routing_context(session, chat, parent_message_id),
        )
        if (
            mode == RoutingMode.AUTO
            and chat.confirm_uncertain_media
            and plan.operation != Operation.TEXT
            and plan.confidence < 0.8
            and not request.confirm_media
        ):
            raise RouteConfirmationRequired(plan)
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

        profile_id = self._profile_for_operation(chat, plan.operation)
        profile = session.get(ModelProfile, profile_id) if profile_id else None
        fields = self._fields_for_operation(plan.operation)
        request_fields = [field for field in fields if field.scope != "load"]
        request_settings = validate_settings(request.settings, request_fields)
        effective_settings = resolve_settings(
            defaults(fields),
            profile.load_settings_json if profile else None,
            profile.request_settings_json if profile else None,
            request_settings,
        )
        effective_settings = validate_settings(effective_settings, fields)
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

        workflow_revision = self._workflow_for_operation(
            session, plan.operation, project_id=chat.project_id
        )
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
                "model": model_provenance,
                "workflow": workflow_provenance,
                "resolved_settings": effective_settings,
                "generation_estimate": generation_estimate,
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
            detail = str(exc).strip() or f"Generation failed ({type(exc).__name__})"
            await self._fail(job_id, run_id, detail)

    async def _execute_chat(self, job_id: str, run_id: str) -> None:
        worker = await self._ensure_chat_worker(run_id)
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
                        len(accumulated) - last_persisted_length >= 32
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
            session.commit()
        await self.events.publish("run.completed", run_id, {"job_id": job_id})

    async def _ensure_chat_worker(self, run_id: str) -> WorkerStatus | None:
        if (
            self.engines.settings.chat_engine != "llama.cpp"
            or not self.processes.settings.llama_executable
        ):
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
                        preview_ids = self._temporary_preview_ids(message)
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
                        session.flush()
                        for artifact_id in preview_ids:
                            self.artifacts.delete_temporary_preview(session, artifact_id)
                        session.commit()
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
            session.commit()
            if preview_artifact_id and preview_artifact_id not in artifact_ids:
                self.artifacts.delete_temporary_preview(session, preview_artifact_id)
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
                preview_ids = self._temporary_preview_ids(message)
                message.status = MessageStatus.FAILED.value
                if run.operation == Operation.TEXT.value:
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
                session.flush()
                for artifact_id in preview_ids:
                    self.artifacts.delete_temporary_preview(session, artifact_id)
            session.commit()
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
            preview_ids = self._temporary_preview_ids(message)
            message.status = MessageStatus.CANCELLED.value
            if run.operation != Operation.TEXT.value:
                self._replace_parts(
                    message,
                    [
                        MessagePart(
                            position=0, type=PartType.ERROR.value, text="Generation cancelled"
                        )
                    ],
                )
            session.flush()
            for artifact_id in preview_ids:
                self.artifacts.delete_temporary_preview(session, artifact_id)

    @staticmethod
    def _temporary_preview_ids(message: Message) -> list[str]:
        return [
            part.artifact_id
            for part in message.parts
            if part.artifact_id and part.metadata_json.get("preview")
        ]

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
    def _routing_context(
        session: Session, chat: Chat, parent_message_id: str | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if chat.project_id:
            project = session.get(Project, chat.project_id)
            if project and project.instructions:
                messages.append({"role": "system", "content": project.instructions})
        rows: list[Message] = []
        current_id = parent_message_id
        visited: set[str] = set()
        while current_id and current_id not in visited and len(rows) < 8:
            visited.add(current_id)
            message = session.scalar(
                select(Message)
                .options(selectinload(Message.parts))
                .where(Message.id == current_id, Message.chat_id == chat.id)
            )
            if not message:
                break
            rows.append(message)
            current_id = message.parent_id
        for message in reversed(rows):
            content = "\n".join(part.text for part in message.parts if part.text).strip()
            if content:
                messages.append({"role": message.role, "content": content})
        return messages

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

    @classmethod
    def request_settings_for_operation(
        cls, operation: Operation, values: dict[str, Any]
    ) -> dict[str, Any]:
        request_keys = {
            field.key for field in cls._fields_for_operation(operation) if field.scope != "load"
        }
        return {key: value for key, value in values.items() if key in request_keys}

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

    @staticmethod
    def _workflow_for_operation(
        session: Session, operation: Operation, *, project_id: str | None = None
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
