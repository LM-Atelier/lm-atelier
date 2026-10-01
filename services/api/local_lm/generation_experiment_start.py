"""Start an accepted comparison: queue each choice's picture as independent work.

Starting resolves nothing again. Each picture runs exactly the choice that was
accepted: its model, its workflow revision and activation, its settings and
the seed it was given. What can be checked is checked first, and a choice that
would now run differently refuses the start before anything is written.

The pictures are two steps of one work plan with no dependency between them,
so a failure, stop or retry of one never touches the other. They run in one
hidden chat whose messages are never shown, searched, exported or used as
context, and whose context is frozen when the work is accepted.
"""

from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from . import generation_experiment_store as store
from .accepted_turn_context import AcceptedWorkflow, accepted_context, resolve_accepted_workflow
from .auxiliary_assets import LORA_GRAPH_TRANSFORM_VERSION, resolve_lora_stack
from .domain import (
    JobStatus,
    MessageRole,
    MessageStatus,
    Operation,
    RoutingMode,
    RunStatus,
    utcnow,
)
from .generation_experiment_preflight import ArmRefused, _refusal, _require_image_profile
from .generation_experiments_v1 import (
    EXPERIMENT_CHAT_SCOPE,
    ExperimentRefusalOut,
    GenerationExperimentRefusalCode,
    GenerationExperimentStart,
    GenerationExperimentState,
    ResourceEvidenceOut,
)
from .generation_retry import capture_retry_budget
from .models import (
    Chat,
    GenerationExperiment,
    GenerationExperimentArm,
    Job,
    Message,
    ModelProfile,
    Run,
    WorkPlan,
    WorkStep,
)
from .orchestrator import (
    ConversationOrchestrator,
    _queued_workflow_activation,
    _require_consistent_workflow_witness,
)
from .progress import update_job_progress
from .workflow_lora_admission import WorkflowLoraAdmissionError
from .workflow_node_dependencies import node_dependency_errors

if TYPE_CHECKING:
    from .config import Settings

Code = GenerationExperimentRefusalCode
OPERATION = Operation.TEXT_TO_IMAGE
PLANNER_VERSION = "generation-experiment-v1"
SOURCE_ACTION = "generation_experiment"
QUEUE_CLASS = "media_compute"


class StartRefused(Exception):
    """A start that queues nothing, with the code the route answers and what it carries."""

    def __init__(self, code: str, **extra: Any) -> None:
        super().__init__(code)
        self.code = code
        self.extra = extra


@dataclass
class StartOutcome:
    experiment_id: str
    created: bool
    work_plan_id: str | None = None
    dispatch: list[tuple[str, str]] = field(default_factory=list)
    announcement: dict[str, Any] = field(default_factory=dict)


def _check_arm(
    orchestrator: ConversationOrchestrator,
    session: Session,
    arm: GenerationExperimentArm,
) -> None:
    """Refuse a choice whose accepted identities no longer hold; read nothing into it."""

    snapshot: dict[str, Any] = arm.snapshot_json
    _require_image_profile(session, arm.profile_id)
    workflow = snapshot.get("workflow") or {}
    if workflow.get("engine") != orchestrator.engines.settings.media_engine:
        raise ArmRefused(Code.ARM_WORKFLOW_UNAVAILABLE)
    try:
        revision = resolve_accepted_workflow(session, AcceptedWorkflow.model_validate(workflow))
    except RuntimeError:
        raise ArmRefused(Code.ARM_WORKFLOW_UNTRUSTED) from None
    except ValueError:
        raise ArmRefused(Code.ARM_CHANGED) from None
    if revision is None:
        raise ArmRefused(Code.ARM_WORKFLOW_UNAVAILABLE)
    if node_dependency_errors(session, revision.dependencies_json):
        raise ArmRefused(Code.ARM_PACKAGE_MISSING)
    try:
        activation = _queued_workflow_activation(session, revision)
    except ValueError:
        raise ArmRefused(Code.ARM_ACTIVATION_NOT_READY) from None
    if activation != snapshot.get("activation"):
        raise ArmRefused(Code.ARM_CHANGED)
    lora = snapshot.get("lora") or {}
    loras = arm.effective_settings_json.get("loras")
    if lora.get("workflow") is None and loras:
        try:
            stack = resolve_lora_stack(session, revision, loras)
        except ValueError:
            raise ArmRefused(Code.ARM_LORA_REFUSED) from None
        if stack.provenance != lora.get("stack") or stack.graph_sha256 != lora.get("graph_sha256"):
            raise ArmRefused(Code.ARM_CHANGED)


def _admission(
    orchestrator: ConversationOrchestrator,
    settings: Settings,
    experiment: GenerationExperiment,
    payload: GenerationExperimentStart,
) -> dict[str, int]:
    """Check the resources the pictures need, as a turn's admission does."""

    totals = {"work_units": 0, "estimated_bytes": 0}
    for arm in experiment.arms:
        for trial in arm.trials:
            figures = ConversationOrchestrator._media_plan_estimate(
                OPERATION, {**arm.effective_settings_json, "seed": trial.seed}, 1
            )
            totals["work_units"] += int(figures["work_units"])
            totals["estimated_bytes"] += int(figures["estimated_bytes"])
    if (
        totals["work_units"] > settings.max_media_plan_work_units
        or totals["estimated_bytes"] > settings.max_media_plan_estimated_bytes
    ):
        raise StartRefused(
            "generation-experiment-refused",
            refusals=[_refusal(Code.EXPERIMENT_TOO_LARGE).model_dump(mode="json")],
        )
    if (
        settings.video_confirmation_work_units > 0
        and totals["work_units"] >= settings.video_confirmation_work_units
        and not payload.confirm_expensive
    ):
        raise StartRefused(
            "generation-experiment-confirmation-required",
            estimate=[
                ResourceEvidenceOut(
                    resource="work_units",
                    value=totals["work_units"],
                    unit="work_units",
                    kind="estimated",
                    source="admission_formula",
                    confidence="heuristic",
                ).model_dump(mode="json"),
                ResourceEvidenceOut(
                    resource="output_bytes",
                    value=totals["estimated_bytes"],
                    unit="bytes",
                    kind="estimated",
                    source="admission_formula",
                    confidence="heuristic",
                ).model_dump(mode="json"),
            ],
        )
    root = orchestrator.artifacts.root
    root.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(root).free
    if totals["estimated_bytes"] > available:
        raise StartRefused("generation-experiment-storage-insufficient")
    return {**totals, "available_bytes_at_admission": available}


def _auxiliary_assets(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """The run's LoRA and trigger-word record, built as a turn builds it, from the snapshot."""

    lora = snapshot.get("lora") or {}
    words = snapshot.get("trigger_words") or {
        "model_trigger_words_applied": [],
        "lora_trigger_words_applied": [],
        "trigger_words_applied": [],
    }
    if lora.get("graph_sha256"):
        return {
            "lora_stack": copy.deepcopy(lora.get("stack") or []),
            "selection": {"mode": "explicit"},
            "graph_transform_version": LORA_GRAPH_TRANSFORM_VERSION,
            "effective_graph_sha256": lora["graph_sha256"],
            **copy.deepcopy(words),
        }
    if words.get("trigger_words_applied"):
        return copy.deepcopy(words)
    return None


def _write_work(
    orchestrator: ConversationOrchestrator,
    session: Session,
    experiment: GenerationExperiment,
    payload: GenerationExperimentStart,
    admission: dict[str, int],
) -> tuple[WorkPlan, list[tuple[str, Job, Run, WorkStep]]]:
    """The hidden chat, its messages, one plan and one step, run and job per picture."""

    trials = [(arm, trial) for arm in experiment.arms for trial in arm.trials]
    count = len(trials)
    prompt = experiment.common_json["prompt"]
    chat = Chat(
        title="Generation comparison",
        scope=EXPERIMENT_CHAT_SCOPE,
        archived=True,
        project_id=None,
        routing_mode=RoutingMode.IMAGE.value,
        confirm_uncertain_media=False,
        generation_settings_json={},
        generation_preset_ids_json={},
        vision_settings_json={},
    )
    session.add(chat)
    session.flush()
    # The prompt travels in each run, never as a transcript message, so the
    # frozen context of each picture holds no conversation at all.
    user_message = Message(
        chat_id=chat.id,
        parent_id=None,
        role=MessageRole.USER.value,
        status=MessageStatus.COMPLETE.value,
        transcript_visible=False,
        parts=[],
    )
    session.add(user_message)
    session.flush()
    assistant_messages = [
        Message(
            chat_id=chat.id,
            parent_id=user_message.id,
            role=MessageRole.ASSISTANT.value,
            status=MessageStatus.PENDING.value,
            transcript_visible=False,
            parts=ConversationOrchestrator._initial_output_parts(OPERATION, ordinal, count),
        )
        for ordinal in range(1, count + 1)
    ]
    session.add_all(assistant_messages)
    session.flush()
    sequence = ConversationOrchestrator._next_transcript_sequence(session, chat)
    plan = WorkPlan(
        chat_id=chat.id,
        idempotency_key=payload.idempotency_key,
        source_action=SOURCE_ACTION,
        persistence_scope=orchestrator.persistence_scope,
        status=JobStatus.QUEUED.value,
        context_head_message_id=None,
        transcript_sequence=sequence,
        priority=0,
        planner_version=PLANNER_VERSION,
        failure_policy="continue_independent",
        summary_json={
            "operation": OPERATION.value,
            "routing_mode": RoutingMode.IMAGE.value,
            "step_count": count,
            "output_count": count,
            "source_action": SOURCE_ACTION,
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_messages[0].id,
            "assistant_message_ids": [message.id for message in assistant_messages],
            "dependency_step_ids": [],
            "media_plan_estimate": admission,
            "status_counts": {"queued": count},
            "generation_experiment_id": experiment.id,
        },
    )
    session.add(plan)
    session.flush()
    written: list[tuple[str, Job, Run, WorkStep]] = []
    for ordinal, ((arm, trial), message) in enumerate(
        zip(trials, assistant_messages, strict=True), start=1
    ):
        snapshot: dict[str, Any] = arm.snapshot_json
        settings = {**copy.deepcopy(arm.effective_settings_json), "seed": trial.seed}
        slot = f"arm-{arm.ordinal}"
        step = WorkStep(
            plan=plan,
            ordinal=ordinal,
            display_group=SOURCE_ACTION,
            operation=OPERATION.value,
            status=JobStatus.QUEUED.value,
            prompt=prompt,
            profile_id=arm.profile_id,
            workflow_revision_id=arm.workflow_revision_id,
            settings_json=copy.deepcopy(settings),
            input_bindings_json=[],
            output_contract_json=[
                {"slot": slot, "type": "image", "index": ordinal, "count": count}
            ],
            queue_class=QUEUE_CLASS,
        )
        session.add(step)
        session.flush()
        profile = session.get(ModelProfile, arm.profile_id)
        auxiliary = _auxiliary_assets(snapshot)
        lora_receipt = (snapshot.get("lora") or {}).get("workflow")
        provenance: dict[str, Any] = {
            "generation_experiment": {
                "version": 1,
                "experiment_id": experiment.id,
                "experiment_snapshot_sha256": experiment.snapshot_sha256,
                "arm_id": arm.id,
                "arm_ordinal": arm.ordinal,
                "arm_snapshot_sha256": arm.snapshot_sha256,
                "trial_id": trial.id,
                "trial_ordinal": trial.ordinal,
                "seed": trial.seed,
                "geometry": copy.deepcopy(snapshot.get("geometry")),
                "adaptations": copy.deepcopy(snapshot.get("adaptations") or []),
            },
            "model_selection": {
                "mode": SOURCE_ACTION,
                "profile_id": arm.profile_id,
                "workflow_revision_id": arm.workflow_revision_id,
                "compatibility_only": True,
            },
            "input_artifact_ids": [],
            "model": ConversationOrchestrator._model_provenance(session, profile),
            "preset": None,
            "preset_layers": [],
            "workflow": copy.deepcopy(snapshot.get("workflow_witness")),
            **({"workflow_lora": copy.deepcopy(lora_receipt)} if lora_receipt else {}),
            "resolved_settings": copy.deepcopy(settings),
            "generation_estimate": None,
            "video_length": None,
            "source_fit_request": None,
            "media_plan_estimate": ConversationOrchestrator._media_plan_estimate(
                OPERATION, settings, 1
            ),
            "media_output": {"index": ordinal, "count": count, "slot": slot},
            "image_edit": None,
            "auxiliary_assets": auxiliary,
        }
        run = Run(
            idempotency_key=payload.idempotency_key if ordinal == 1 else None,
            chat_id=chat.id,
            user_message_id=user_message.id,
            assistant_message_id=message.id,
            work_plan_id=plan.id,
            work_step_id=step.id,
            operation=OPERATION.value,
            status=RunStatus.QUEUED.value,
            standalone_prompt=prompt,
            profile_id=arm.profile_id,
            vision_profile_id=None,
            workflow_revision_id=arm.workflow_revision_id,
            settings_json=copy.deepcopy(settings),
            provenance_json=provenance,
        )
        _require_consistent_workflow_witness(step, run)
        # The automatic retries a turn's picture gets, frozen with the run as a
        # turn freezes them, so changing the setting later cannot extend them.
        run.provenance_json = {
            **run.provenance_json,
            "failure_retries": capture_retry_budget(session, run.operation),
        }
        session.add(run)
        session.flush()
        step.run_id = run.id
        job = Job(
            kind=ConversationOrchestrator._job_kind(OPERATION).value,
            status=JobStatus.QUEUED.value,
            run_id=run.id,
            work_plan_id=plan.id,
            work_step_id=step.id,
            progress=0,
            phase="queued",
            queue_resource=QUEUE_CLASS,
            queue_group="primary",
            queue_priority=plan.priority,
            queue_ticket=f"{sequence:020d}:{ordinal:04d}:{run.id}",
            enqueued_at=utcnow(),
            payload_json={
                "operation": OPERATION.value,
                "output_index": ordinal,
                "output_count": count,
                "generation_experiment_trial_id": trial.id,
            },
        )
        update_job_progress(
            job,
            stage="queued",
            queue_resource=QUEUE_CLASS,
            queue_position=ordinal - 1,
            queue_length=count,
            indeterminate=True,
        )
        session.add(job)
        session.flush()
        written.append((trial.id, job, run, step))
    plan.summary_json = {
        **plan.summary_json,
        "step_ids": [step.id for _, _, _, step in written],
        "run_ids": [run.id for _, _, run, _ in written],
        "job_ids": [job.id for _, job, _, _ in written],
    }
    return plan, written


def _freeze_and_bind(
    orchestrator: ConversationOrchestrator,
    session: Session,
    experiment: GenerationExperiment,
    written: list[tuple[str, Job, Run, WorkStep]],
) -> None:
    """Freeze each picture's context and require it to be the accepted choice."""

    arms = {trial.id: arm for arm in experiment.arms for trial in arm.trials}
    refusals: list[ExperimentRefusalOut] = []
    for trial_id, _job, run, _step in written:
        arm = arms[trial_id]
        snapshot: dict[str, Any] = arm.snapshot_json
        orchestrator._freeze_turn_context(session, run)
        context = accepted_context(session, run)
        if (
            context is None
            or context.profile is None
            or context.workflow is None
            or context.profile.model_dump(mode="json") != snapshot.get("profile")
            or context.workflow.model_dump(mode="json") != snapshot.get("workflow")
            or context.media_engine != (snapshot.get("workflow") or {}).get("engine")
        ):
            refusals.append(_refusal(Code.ARM_CHANGED, arm_ordinal=arm.ordinal))
            continue
        try:
            ConversationOrchestrator.preflight_workflow_lora_replay(session, run)
        except WorkflowLoraAdmissionError:
            refusals.append(_refusal(Code.ARM_LORA_REFUSED, arm_ordinal=arm.ordinal))
    if refusals:
        raise StartRefused(
            "generation-experiment-preflight-changed",
            refusals=[refusal.model_dump(mode="json") for refusal in refusals],
        )


def start_generation_experiment(
    orchestrator: ConversationOrchestrator,
    settings: Settings,
    session: Session,
    experiment_id: str,
    payload: GenerationExperimentStart,
) -> StartOutcome:
    """Queue the comparison's pictures in one transaction, or refuse and write nothing.

    Synchronous on purpose: nothing is awaited between taking the write lock and
    the commit. The caller announces the work and starts it after the commit.
    """

    if not orchestrator._admission_open:
        raise StartRefused("generation-experiment-unavailable")
    if session.in_transaction():
        session.rollback()
    try:
        session.execute(text("BEGIN IMMEDIATE"))
        experiment = session.get(GenerationExperiment, experiment_id, populate_existing=True)
        if experiment is None:
            raise StartRefused("generation-experiment-not-found")
        try:
            store.out(experiment)
        except store.GenerationExperimentRecordError:
            raise StartRefused("generation-experiment-record-invalid") from None
        if experiment.state == GenerationExperimentState.STARTED.value:
            if experiment.start_idempotency_key != payload.idempotency_key:
                raise StartRefused("generation-experiment-already-started")
            if experiment.start_request_sha256 != store.start_request_digest(payload):
                raise StartRefused("generation-experiment-idempotency-conflict")
            session.rollback()
            return StartOutcome(experiment_id, created=False)
        if payload.snapshot_sha256 != experiment.snapshot_sha256:
            raise StartRefused("generation-experiment-snapshot-changed")
        refusals: list[ExperimentRefusalOut] = []
        for arm in experiment.arms:
            try:
                _check_arm(orchestrator, session, arm)
            except ArmRefused as refused:
                refusals.append(
                    _refusal(
                        refused.code,
                        arm_ordinal=arm.ordinal,
                        setting=refused.setting,
                        alternative=refused.alternative,
                    )
                )
        if refusals:
            raise StartRefused(
                "generation-experiment-preflight-changed",
                refusals=[refusal.model_dump(mode="json") for refusal in refusals],
            )
        admission = _admission(orchestrator, settings, experiment, payload)
        plan, written = _write_work(orchestrator, session, experiment, payload, admission)
        _freeze_and_bind(orchestrator, session, experiment, written)
        store.link_started_work(
            experiment,
            payload,
            plan.id,
            {trial_id: (step.id, run.id) for trial_id, _job, run, step in written},
        )
        session.flush()
        outcome = StartOutcome(
            experiment_id,
            created=True,
            work_plan_id=plan.id,
            dispatch=[(job.id, run.id) for _, job, run, _ in written],
            announcement={
                "plan_id": plan.id,
                "step_id": written[0][3].id,
                "run_id": written[0][2].id,
                "job_id": written[0][1].id,
                "step_ids": [step.id for _, _, _, step in written],
                "run_ids": [run.id for _, _, run, _ in written],
                "job_ids": [job.id for _, job, _, _ in written],
                "chat_id": plan.chat_id,
            },
        )
        session.commit()
        return outcome
    except BaseException:
        session.rollback()
        raise
