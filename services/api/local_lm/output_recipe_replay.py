"""Whether a generation record could be generated again exactly here, and from what.

A record names everything by content - the workflow by its recomputed identity,
the model by its files' hashes, each LoRA and input by its file's hash - so
generating it again means finding, for each of those, the one local thing that
is exactly it. This works that out and nothing more: it reads the record and
this installation's rows, starts nothing, writes nothing and keeps nothing.

Exact means exact. A requirement that nothing here satisfies, or that more than
one thing here satisfies, is refused rather than guessed at; a model install
holding the recorded files and others besides is a different install; a
workflow whose stored identity no longer matches its own graph is not that
workflow. Every refusal is collected, so one answer says everything that stands
in the way, and none repeats the prompt or any setting value.

Two things the record itself says are missing do not stand in the way. A record
of a turn that froze no copy of its inputs was written from the same rows a run
reads, so the missing copy changes nothing here. A record made by the test
engine can be generated again by the test engine, which the engine check
already requires.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from .auxiliary_assets import (
    MAX_LORA_STACK_SIZE,
    resolve_lora_stack,
    revision_accepts_added_loras,
)
from .domain import Operation, operation_model_role
from .engines import EngineRegistry
from .lora_constraints import MAX_LORA_STRENGTH
from .model_planner import (
    WORKFLOW_ARTIFACT_CONTRACT_VERSION,
    revision_accepts_install,
    workflow_artifact_contract,
)
from .models import (
    Artifact,
    Chat,
    Message,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowRevision,
)
from .orchestrator import ConversationOrchestrator
from .output_recipe import describe_run
from .output_recipe_v1 import canonical_bytes
from .schemas import TurnRequest
from .settings_registry import workflow_settings
from .setup_verification import setup_verification_for_chat
from .upscale_workflows import effective_upscale_schema
from .workflow_selection import WorkflowFamilySelectionError, resolve_exact_workflow_revision

RefusalKind = Literal["record", "engine", "workflow", "model", "lora", "input"]

#: What a replay starts: generation from words, a video from a picture, and an
#: edit of a picture.
REPLAYABLE_OPERATIONS: Final = frozenset(
    {"text_to_image", "text_to_video", "image_to_video", "image_to_image"}
)
#: The sections an exact replay must reproduce, in the order a refusal names them.
REPLAYED_SECTIONS: Final = (
    "operation",
    "prompt",
    "seed",
    "settings",
    "inputs",
    "workflow",
    "model",
    "loras",
)
#: Removals that would hide from the comparison something that ran.
_UNCOMPARED_REMOVALS: Final = frozenset({"settings.mask", "settings.workflow_lora_overrides"})

#: What a record may say is missing and still be generated again exactly.
TOLERATED_MISSING: Final = frozenset({"frozen_snapshot_absent", "mock_engine"})
#: The most inputs one turn takes, as the turn request itself bounds them.
MAX_REPLAY_INPUTS: Final[int] = next(
    item.max_length
    for item in TurnRequest.model_fields["input_artifact_ids"].metadata
    if hasattr(item, "max_length")
)


def plan_output_recipe_replay(
    session: Session, record: dict[str, Any], *, media_engine: str
) -> dict[str, Any]:
    """Resolve each of a record's requirements to exactly one local thing, or refuse.

    `record` is a record already read strictly. The answer names local rows only
    when nothing refuses; it is for this installation's own client, never for a
    file that leaves it.
    """

    refusals: list[dict[str, Any]] = []
    operation = Operation(record["operation"])
    _check_record(record, operation, refusals)
    workflow = record["workflow"]
    output_engine = record["output"]["engine"]
    if (workflow is not None and workflow["engine"] != media_engine) or (
        output_engine is not None and output_engine != media_engine
    ):
        refusals.append(_refusal("replay-engine-differs", "engine"))

    revision, bound_profile = _workflow(session, record, operation, media_engine, refusals)
    profile = _model(session, record, operation, media_engine, revision, bound_profile, refusals)
    lora_asset_ids = _loras(session, record, revision, refusals)
    input_artifact_ids = _inputs(session, record, refusals)

    resolved = None
    if not refusals and revision is not None:
        resolved = {
            "mode": operation_model_role(operation),
            "workflow_revision_id": revision.id,
            "profile_id": profile.id if profile is not None else None,
            "lora_asset_ids": lora_asset_ids,
            "input_artifact_ids": input_artifact_ids,
        }
    return {
        "digest": record["digest"],
        "operation": record["operation"],
        "ready": resolved is not None,
        "refusals": refusals,
        "resolved": resolved,
    }


def _refusal(
    code: str, kind: RefusalKind, sha256: str | None = None, reasons: list[str] | None = None
) -> dict[str, Any]:
    return {"code": code, "kind": kind, "sha256": sha256, "reasons": reasons or []}


def _check_record(
    record: dict[str, Any], operation: Operation, refusals: list[dict[str, Any]]
) -> None:
    """Refuse what the record itself says it lacks, and shapes a turn cannot carry."""

    missing = set(record["reproducibility"]["missing"])
    # The format does not tie an empty section to the reason that names it, so a
    # record written elsewhere could leave one out; each is added here rather
    # than trusted to be listed.
    workflow = record["workflow"]
    if workflow is None:
        missing.add("workflow_unavailable")
    elif not workflow["verified"]:
        missing.add("workflow_unverified")
    if record["model"] is None or not record["model"]["files"]:
        missing.add("model_files_not_recorded")
    if not record["prompt"]["included"]:
        missing.add("prompt_omitted")
    if record["seed"]["value"] is None:
        missing.add("seed_not_recorded")
    missing -= TOLERATED_MISSING
    if missing:
        # Each name comes from the format's own closed list, so none echoes input.
        refusals.append(_refusal("replay-record-incomplete", "record", reasons=sorted(missing)))
    reasons: list[str] = []
    if workflow is not None and workflow["contract_version"] != WORKFLOW_ARTIFACT_CONTRACT_VERSION:
        reasons.append("workflow_contract_version")
    inputs = record["inputs"]
    if len(inputs) > MAX_REPLAY_INPUTS:
        reasons.append("too_many_inputs")
    loras = record["loras"]
    if len(loras) > MAX_LORA_STACK_SIZE:
        reasons.append("too_many_loras")
    if any(
        abs(lora[key]) > MAX_LORA_STRENGTH
        for lora in loras
        for key in ("model_strength", "clip_strength")
    ):
        reasons.append("lora_strength")
    if [lora["position"] for lora in loras] != list(range(len(loras))):
        reasons.append("lora_positions")
    hashes = [item["sha256"] for item in inputs]
    if len(set(hashes)) != len(hashes):
        # A turn takes each input once, so a repeated one would run as one.
        reasons.append("repeated_inputs")
    roles = [item["role"] for item in inputs]
    if operation in {Operation.TEXT_TO_IMAGE, Operation.TEXT_TO_VIDEO}:
        if roles:
            reasons.append("inputs_for_operation")
    elif not roles or roles[0] != "source":
        reasons.append("inputs_for_operation")
    if "mask" in roles:
        reasons.append("mask_input")
    positive = record["prompt"]["positive"]
    if (
        operation == Operation.IMAGE_TO_IMAGE
        and isinstance(positive, str)
        and (
            not positive.startswith(edit_prompt_preamble())
            or not positive.removeprefix(edit_prompt_preamble()).strip()
        )
    ):
        # Written by a version that worded its edits differently, or with no
        # request after the wording: either way there is no request to send.
        reasons.append("edit_prompt_wording")
    if reasons:
        refusals.append(_refusal("replay-record-unsupported", "record", reasons=reasons))


def _workflow(
    session: Session,
    record: dict[str, Any],
    operation: Operation,
    media_engine: str,
    refusals: list[dict[str, Any]],
) -> tuple[WorkflowRevision | None, ModelProfile | None]:
    """The one revision that executes exactly the recorded workflow, ready to run here."""

    workflow = record["workflow"]
    if workflow is None or not workflow["verified"]:
        # Named by the record-level refusal.
        return None, None
    identity = workflow["artifact_sha256"]
    exact = [
        revision
        for revision in session.scalars(
            select(WorkflowRevision)
            .where(WorkflowRevision.artifact_sha256 == identity)
            .order_by(WorkflowRevision.id)
        ).all()
        if _executes_exactly(session, revision, workflow)
    ]
    if not exact:
        refusals.append(_refusal("replay-workflow-missing", "workflow", identity))
        return None, None
    # Only a revision admission would run counts: a copy awaiting review, or one
    # with no ready activation, is not a second choice.
    role = operation_model_role(operation)
    runnable: list[tuple[WorkflowRevision, WorkflowActivation | None, ModelProfile | None]] = []
    for revision in exact:
        try:
            _, activation, bound = resolve_exact_workflow_revision(
                session,
                revision.id,
                capability="video" if role == "video" else "image",
                operation=operation,
                engine=media_engine,
            )
        except WorkflowFamilySelectionError:
            # Its own reason can name a family; the code alone says enough.
            continue
        runnable.append((revision, activation, bound))
    if not runnable:
        refusals.append(_refusal("replay-workflow-not-ready", "workflow", identity))
        return None, None
    if len(runnable) > 1:
        refusals.append(_refusal("replay-workflow-ambiguous", "workflow", identity))
        return None, None
    revision, activation, bound_profile = runnable[0]
    binding = activation.binding_sha256 if activation is not None else None
    if binding != workflow["binding_sha256"]:
        refusals.append(_refusal("replay-workflow-binding-differs", "workflow", identity))
        return None, None
    if binding is None and media_engine == "comfyui":
        # Without a binding nothing says which files the runtime mounted for it.
        refusals.append(_refusal("replay-workflow-unscoped", "workflow", identity))
        return None, None
    return revision, bound_profile


def _executes_exactly(
    session: Session, revision: WorkflowRevision, workflow: dict[str, Any]
) -> bool:
    """Whether this revision's own graph is the recorded workflow.

    The stored identity is never refreshed, so it is recomputed from the
    revision's columns rather than trusted.
    """

    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if (
        definition is None
        or definition.operation != workflow["operation"]
        or revision.engine != workflow["engine"]
        or revision.dependency_contract_sha256 != workflow["dependency_contract_sha256"]
        or not isinstance(revision.api_graph_json, dict)
        or not isinstance(revision.input_schema_json, dict)
    ):
        return False
    try:
        recomputed = workflow_artifact_contract(
            operation=definition.operation,
            engine=revision.engine,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=(
                revision.dependencies_json if isinstance(revision.dependencies_json, dict) else {}
            ),
        )
    except (TypeError, ValueError):
        return False
    return bool(recomputed == workflow["artifact_sha256"])


def _model(
    session: Session,
    record: dict[str, Any],
    operation: Operation,
    media_engine: str,
    revision: WorkflowRevision | None,
    bound_profile: ModelProfile | None,
    refusals: list[dict[str, Any]],
) -> ModelProfile | None:
    """The one profile whose install holds exactly the recorded model files."""

    model = record["model"]
    if model is None or not model["files"]:
        # Named by the record-level refusal.
        return None
    files = model["files"]
    role = operation_model_role(operation)
    candidates: list[ModelProfile] = []
    for profile in session.scalars(
        select(ModelProfile)
        .where(
            ModelProfile.role == role,
            ModelProfile.engine == media_engine,
            ModelProfile.model_install_id.is_not(None),
        )
        .order_by(ModelProfile.id)
    ).all():
        install = session.get(ModelInstall, profile.model_install_id)
        if (
            install is None
            or not install.active
            or install.engine != profile.engine
            or (install.manifest_json or {}).get("expected_sha256") != files
        ):
            continue
        if revision is not None and not revision_accepts_install(
            session,
            revision.dependencies_json if isinstance(revision.dependencies_json, dict) else {},
            install.id,
        ):
            continue
        if bound_profile is not None and profile.id != bound_profile.id:
            continue
        candidates.append(profile)
    if not candidates:
        refusals.append(_refusal("replay-model-missing", "model"))
        return None
    if len(candidates) > 1:
        refusals.append(_refusal("replay-model-ambiguous", "model"))
        return None
    return candidates[0]


def _loras(
    session: Session,
    record: dict[str, Any],
    revision: WorkflowRevision | None,
    refusals: list[dict[str, Any]],
) -> list[str]:
    """One active asset for each recorded LoRA, in recorded order, usable by the workflow."""

    loras = sorted(record["loras"], key=lambda item: item["position"])
    if not loras:
        return []
    assets = [
        asset
        for asset in session.scalars(
            select(ModelAssetInstall)
            .where(ModelAssetInstall.kind == "lora", ModelAssetInstall.active.is_(True))
            .order_by(ModelAssetInstall.id)
        ).all()
    ]
    asset_ids: list[str] = []
    for lora in loras:
        holders = [
            asset for asset in assets if (asset.manifest_json or {}).get("sha256") == lora["sha256"]
        ]
        if not holders:
            refusals.append(_refusal("replay-lora-missing", "lora", lora["sha256"]))
        elif len(holders) > 1:
            refusals.append(_refusal("replay-lora-ambiguous", "lora", lora["sha256"]))
        else:
            asset_ids.append(holders[0].id)
    if len(asset_ids) != len(loras) or revision is None:
        return []
    stack = [
        {
            "asset_id": asset_id,
            "model_strength": lora["model_strength"],
            "clip_strength": lora["clip_strength"],
            "enabled": lora["enabled"],
        }
        for asset_id, lora in zip(asset_ids, loras, strict=True)
    ]
    try:
        resolve_lora_stack(session, revision, stack)
    except ValueError:
        # Its message names assets and families; the code alone says enough.
        refusals.append(_refusal("replay-lora-unusable", "lora"))
        return []
    return asset_ids


def _inputs(session: Session, record: dict[str, Any], refusals: list[dict[str, Any]]) -> list[str]:
    """Each recorded input by its content identity, in recorded order."""

    identifiers: list[str] = []
    for item in record["inputs"]:
        identifier = f"sha256:{item['sha256']}"
        if session.get(Artifact, identifier) is None:
            refusals.append(_refusal("replay-input-missing", "input", item["sha256"]))
        else:
            identifiers.append(identifier)
    return identifiers


class ReplayDiffers(Exception):
    """What admission accepted is not exactly the record; names where it differs."""

    def __init__(self, sections: list[str]) -> None:
        super().__init__("The accepted generation differs from the record.")
        self.sections = sections


def edit_prompt_preamble() -> str:
    """The wording every picture edit's prompt starts with, written as a run writes it."""

    probe = Run(
        operation=Operation.IMAGE_TO_IMAGE.value,
        standalone_prompt="",
        provenance_json={"image_edit": {"policy": "preserve_unrequested_details_v1"}},
    )
    return ConversationOrchestrator._media_prompt(probe)


def chat_is_clean_for_replay(session: Session, chat: Chat) -> bool:
    """Whether nothing in the chat can change what a turn sent into it runs.

    Earlier pictures are attached and prompts rewritten from history, and chat
    and project settings sit under the turn's own, so a replay goes only into a
    chat with none of them.
    """

    has_message = session.scalar(select(Message.id).where(Message.chat_id == chat.id).limit(1))
    return (
        has_message is None
        and chat.project_id is None
        and not chat.generation_settings_json
        and not any(chat.generation_preset_ids_json.values())
        and setup_verification_for_chat(session, chat.id) is None
    )


def without_edit_check(chat: Chat) -> Callable[[], None]:
    """Turn a chat's after-the-fact check of edits off for the turn being accepted.

    The check makes a second, different result once the edit finishes, so a
    replayed edit runs without it. The run keeps what was frozen at acceptance;
    the returned function puts the chat's own setting back before the commit,
    so every later edit in the chat is checked as before.
    """

    kept = dict(chat.vision_settings_json or {})
    chat.vision_settings_json = {**kept, "verify_image_edits": False}

    def restore() -> None:
        chat.vision_settings_json = kept

    return restore


async def replay_turn_request(
    session: Session, engines: EngineRegistry, record: dict[str, Any], resolved: dict[str, Any]
) -> TurnRequest:
    """One explicit turn that asks for exactly what the record names.

    Every recorded setting the workflow takes as a request is sent, so nothing
    is left for a default to fill. A setting admission works out for itself -
    a video's frames from its length, say - is left for it to work out again,
    and the comparison afterwards holds it to the recorded value.
    """

    revision = session.get(WorkflowRevision, resolved["workflow_revision_id"])
    profile = session.get(ModelProfile, resolved["profile_id"]) if resolved["profile_id"] else None
    if revision is None:
        raise ReplayDiffers(["workflow"])
    # The fields admission offers this turn, worked out the same way it does.
    engine_fields = await engines.settings_for_role(
        resolved["mode"], engine=profile.engine if profile is not None else revision.engine
    )
    offered = {
        field.key
        for field in workflow_settings(
            engine_fields,
            effective_upscale_schema(revision.api_graph_json, revision.input_schema_json),
            accepts_added_loras=revision_accepts_added_loras(revision),
        )
        if field.scope != "load"
    }
    recorded = {**record["settings"]["unbound"], **record["settings"]["bound"]}
    settings: dict[str, Any] = {key: value for key, value in recorded.items() if key in offered}
    if "seed" in offered:
        settings["seed"] = record["seed"]["value"]
    if "negative_prompt" in offered:
        # Sent even when empty, so a default, profile or preset negative prompt
        # cannot step in where the record had none.
        settings["negative_prompt"] = record["prompt"]["negative"] or ""
    if "loras" in offered:
        # Sent even when empty, which also keeps automatic LoRAs out.
        loras = sorted(record["loras"], key=lambda item: item["position"])
        settings["loras"] = [
            {
                "asset_id": asset_id,
                "model_strength": lora["model_strength"],
                "clip_strength": lora["clip_strength"],
                "enabled": lora["enabled"],
            }
            for asset_id, lora in zip(resolved["lora_asset_ids"], loras, strict=True)
        ]
    text = record["prompt"]["positive"]
    if record["operation"] == Operation.IMAGE_TO_IMAGE.value:
        # A run adds this wording to every edit; the turn carries only the request.
        text = text.removeprefix(edit_prompt_preamble())
    return TurnRequest(
        text=text,
        mode=resolved["mode"],
        profile_id=resolved["profile_id"],
        workflow_revision_id=resolved["workflow_revision_id"],
        # Null on purpose: every preset layer stays out of the turn.
        preset_id=None,
        output_count=1,
        input_artifact_ids=resolved["input_artifact_ids"],
        settings=settings,
    )


def exact_replay_check(record: dict[str, Any]) -> Callable[[Session, Run], None]:
    """A check run inside the acceptance transaction, before anything is committed.

    It describes each accepted run the way its record would be written and
    raises ReplayDiffers unless that description equals the record, so a turn
    that admission changed in any way is never committed.
    """

    def check(session: Session, first: Run) -> None:
        session.flush()
        runs = session.scalars(select(Run).where(Run.work_plan_id == first.work_plan_id)).all()
        if len(runs) != 1:
            raise ReplayDiffers(["output_count"])
        description = describe_run(session, runs[0])
        differing = [
            name
            for name in REPLAYED_SECTIONS
            if canonical_bytes(_compared(name, description.sections[name]))
            != canonical_bytes(_compared(name, record[name]))
        ]
        if description.missing or description.removed & _UNCOMPARED_REMOVALS:
            differing.append("completeness")
        if differing:
            raise ReplayDiffers(differing)

    return check


def _compared(name: str, value: Any) -> Any:
    """The part of a section an exact replay must reproduce."""

    if name == "workflow" and isinstance(value, dict):
        # Where the graph was read from is not what ran.
        return {key: item for key, item in value.items() if key != "graph_source"}
    if name == "model" and isinstance(value, dict):
        # The files are the model; where they came from is not compared.
        return value["files"]
    return value
