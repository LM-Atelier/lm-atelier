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

import re
from collections.abc import Callable
from typing import Any, Final, Literal, NamedTuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from .auxiliary_assets import (
    MAX_LORA_STACK_SIZE,
    resolve_lora_stack,
    revision_accepts_added_loras,
)
from .domain import Operation, RunStatus, operation_model_role
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
from .output_origin import names_a_preview
from .output_recipe import describe_run
from .output_recipe_v1 import PURPOSES_VERSION, canonical_bytes
from .schemas import TurnRequest
from .settings_registry import workflow_settings
from .setup_verification import setup_verification_for_chat
from .studio_masks import MASK_SETTING_KEY
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
#: Where a replayed run keeps what it was generated again from.
REPLAY_RECEIPT_KEY: Final = "replay"
ReplayOutcome = Literal["pending", "identical", "different", "output_missing", "adapted"]
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
#: Where an adapted run keeps the record it came from and what was chosen in its place.
ADAPTATION_KEY: Final = "adaptation"
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
    _check_engine(record, media_engine, refusals)

    revision, bound_profile = _workflow(session, record, operation, media_engine, refusals)
    profile = _model(session, record, operation, media_engine, revision, bound_profile, refusals)
    lora_asset_ids: list[str | None] = list(_loras(session, record, revision, refusals))
    return _plan(session, record, operation, revision, profile, lora_asset_ids, refusals)


class AdaptationChoices(NamedTuple):
    """What the person chose to stand in for a record's requirements; None keeps the exact one."""

    workflow_revision_id: str | None
    profile_id: str | None
    #: By recorded position: an asset to use in its place, or None to leave it out.
    loras: dict[int, str | None]
    #: By recorded position among the inputs: a picture here to use in its place, or
    #: BUNDLED_INPUT for the copy the record's bundle carries.
    inputs: dict[int, str]


#: The input choice naming the copy a bundle carries for that position.
BUNDLED_INPUT: Final = "bundle"


class AdaptationChoiceInvalid(ValueError):
    """A choice that names no requirement of the record, or is not a local identifier."""


def adaptation_choices(
    *,
    workflow_revision_ids: list[str],
    profile_ids: list[str],
    loras: list[str],
    lora_count: int,
    inputs: list[str],
    input_count: int,
) -> AdaptationChoices:
    """Read the person's choices: each `lora` as `<position>:<asset id>` or `<position>:omit`,
    and each `input` as `<position>:sha256:<hex>`, a picture here in place of that input, or
    `<position>:bundle` for the copy of it the record's bundle carries."""

    if len(workflow_revision_ids) > 1 or len(profile_ids) > 1:
        raise AdaptationChoiceInvalid
    if not all(_local_id(value) for value in (*workflow_revision_ids, *profile_ids)):
        raise AdaptationChoiceInvalid
    chosen: dict[int, str | None] = {}
    for value in loras:
        position_text, separator, asset = value.partition(":")
        if not separator or not (asset == "omit" or _local_id(asset)):
            raise AdaptationChoiceInvalid
        position = _position(position_text)
        if position >= lora_count or position in chosen:
            raise AdaptationChoiceInvalid
        chosen[position] = None if asset == "omit" else asset
    pictures: dict[int, str] = {}
    for value in inputs:
        position_text, separator, artifact = value.partition(":")
        if not separator or not (artifact == BUNDLED_INPUT or _DIGEST.fullmatch(artifact)):
            raise AdaptationChoiceInvalid
        position = _position(position_text)
        if position >= input_count or position in pictures:
            raise AdaptationChoiceInvalid
        pictures[position] = artifact
    return AdaptationChoices(
        workflow_revision_ids[0] if workflow_revision_ids else None,
        profile_ids[0] if profile_ids else None,
        chosen,
        pictures,
    )


def _position(text: str) -> int:
    """A recorded position written plainly, without signs or leading zeros."""

    if (
        not text.isascii()
        or not text.isdecimal()
        or len(text) > 3
        or (len(text) > 1 and text.startswith("0"))
    ):
        raise AdaptationChoiceInvalid
    return int(text)


def plan_output_recipe_adaptation(
    session: Session, record: dict[str, Any], choices: AdaptationChoices, *, media_engine: str
) -> dict[str, Any]:
    """Resolve each requirement to the person's choice where there is one, else exactly, or refuse.

    The record itself must still hold what a turn sends - its prompt, seed and
    inputs - and name this engine; only the workflow, the model and each LoRA
    can be chosen.
    """

    refusals: list[dict[str, Any]] = []
    operation = Operation(record["operation"])
    _check_record(
        record,
        operation,
        refusals,
        workflow_chosen=choices.workflow_revision_id is not None,
        model_chosen=choices.profile_id is not None,
    )
    _check_engine(record, media_engine, refusals)
    if record["workflow"] is None and record["output"]["engine"] is None:
        # Nothing then names the engine it was made with, so it cannot be this one.
        refusals.append(_refusal("replay-engine-differs", "engine"))
    if choices.workflow_revision_id is None:
        revision, bound_profile = _workflow(session, record, operation, media_engine, refusals)
    else:
        revision, bound_profile = _chosen_workflow(
            session, choices.workflow_revision_id, operation, media_engine, refusals
        )
    if choices.profile_id is None:
        profile = _model(
            session, record, operation, media_engine, revision, bound_profile, refusals
        )
    else:
        profile = _chosen_model(
            session, choices.profile_id, operation, media_engine, revision, bound_profile, refusals
        )
    lora_asset_ids = _adapted_loras(session, record, revision, choices.loras, refusals)
    return _plan(
        session,
        record,
        operation,
        revision,
        profile,
        lora_asset_ids,
        refusals,
        chosen_inputs=choices.inputs,
    )


def adaptation_choice_list(choices: AdaptationChoices) -> list[dict[str, Any]]:
    """The choices as an adapted run keeps them, in a fixed order."""

    listed: list[dict[str, Any]] = []
    if choices.workflow_revision_id is not None:
        listed.append(
            {"requirement": "workflow", "position": None, "chosen": choices.workflow_revision_id}
        )
    if choices.profile_id is not None:
        listed.append({"requirement": "model", "position": None, "chosen": choices.profile_id})
    listed.extend(
        {"requirement": "lora", "position": position, "chosen": asset}
        for position, asset in sorted(choices.loras.items())
    )
    listed.extend(
        {"requirement": "input", "position": position, "chosen": artifact}
        for position, artifact in sorted(choices.inputs.items())
    )
    return listed


def _plan(
    session: Session,
    record: dict[str, Any],
    operation: Operation,
    revision: WorkflowRevision | None,
    profile: ModelProfile | None,
    lora_asset_ids: list[str | None],
    refusals: list[dict[str, Any]],
    *,
    chosen_inputs: dict[int, str] | None = None,
) -> dict[str, Any]:
    input_artifact_ids, mask_artifact_id, input_image_roles = _inputs(
        session, record, refusals, chosen_inputs or {}
    )
    resolved = None
    if not refusals and revision is not None:
        resolved = {
            "mode": operation_model_role(operation),
            "workflow_revision_id": revision.id,
            "profile_id": profile.id if profile is not None else None,
            "lora_asset_ids": lora_asset_ids,
            "input_artifact_ids": input_artifact_ids,
            "mask_artifact_id": mask_artifact_id,
            # Only a record that names purposes gives any, so a version 1 plan reads as before.
            **({"input_image_roles": input_image_roles} if input_image_roles is not None else {}),
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


def _check_engine(
    record: dict[str, Any], media_engine: str, refusals: list[dict[str, Any]]
) -> None:
    """Refuse a record made by another engine; the engine names the runtime, not a choice."""

    workflow = record["workflow"]
    output_engine = record["output"]["engine"]
    if (workflow is not None and workflow["engine"] != media_engine) or (
        output_engine is not None and output_engine != media_engine
    ):
        refusals.append(_refusal("replay-engine-differs", "engine"))


def _local_id(value: str) -> bool:
    return (
        0 < len(value) <= 64
        and value.isascii()
        and all(char.isalnum() or char in "_-" for char in value)
    )


def _chosen_workflow(
    session: Session,
    revision_id: str,
    operation: Operation,
    media_engine: str,
    refusals: list[dict[str, Any]],
) -> tuple[WorkflowRevision | None, ModelProfile | None]:
    """The chosen revision, when admission would run it for this operation."""

    role = operation_model_role(operation)
    try:
        revision, _activation, bound = resolve_exact_workflow_revision(
            session,
            revision_id,
            capability="video" if role == "video" else "image",
            operation=operation,
            engine=media_engine,
        )
    except WorkflowFamilySelectionError:
        refusals.append(_refusal("adaptation-workflow-unusable", "workflow"))
        return None, None
    return revision, bound


def _chosen_model(
    session: Session,
    profile_id: str,
    operation: Operation,
    media_engine: str,
    revision: WorkflowRevision | None,
    bound_profile: ModelProfile | None,
    refusals: list[dict[str, Any]],
) -> ModelProfile | None:
    """The chosen profile, when admission would run it for this operation in this workflow."""

    profile = session.get(ModelProfile, profile_id)
    install = (
        session.get(ModelInstall, profile.model_install_id)
        if profile is not None and profile.model_install_id
        else None
    )
    if (
        profile is None
        or profile.role != operation_model_role(operation)
        or profile.engine != media_engine
        or (
            profile.model_install_id is not None
            and (install is None or not install.active or install.engine != profile.engine)
        )
        # A workflow bound to its own model runs only that one, and one that
        # declares the models it takes runs no other.
        or (bound_profile is not None and profile.id != bound_profile.id)
        or (
            revision is not None
            and not revision_accepts_install(
                session,
                revision.dependencies_json if isinstance(revision.dependencies_json, dict) else {},
                profile.model_install_id,
            )
        )
    ):
        refusals.append(_refusal("adaptation-model-unusable", "model"))
        return None
    return profile


def _adapted_loras(
    session: Session,
    record: dict[str, Any],
    revision: WorkflowRevision | None,
    chosen: dict[int, str | None],
    refusals: list[dict[str, Any]],
) -> list[str | None]:
    """An asset for each recorded LoRA, in recorded order: the chosen one, None where left out,
    else the one that holds exactly its file."""

    loras = sorted(record["loras"], key=lambda item: item["position"])
    assets = _active_lora_assets(session)
    asset_ids: list[str | None] = []
    resolved = True
    for position, lora in enumerate(loras):
        if position in chosen:
            choice = chosen[position]
            asset = (
                next((item for item in assets if item.id == choice), None)
                if choice is not None
                else None
            )
            if choice is not None and asset is None:
                refusals.append(_refusal("adaptation-lora-unusable", "lora", lora["sha256"]))
                resolved = False
            asset_ids.append(choice)
            continue
        holders = [
            item for item in assets if (item.manifest_json or {}).get("sha256") == lora["sha256"]
        ]
        if len(holders) != 1:
            code = "replay-lora-missing" if not holders else "replay-lora-ambiguous"
            refusals.append(_refusal(code, "lora", lora["sha256"]))
            resolved = False
            asset_ids.append(None)
        else:
            asset_ids.append(holders[0].id)
    stack = [
        {
            "asset_id": asset_id,
            "model_strength": lora["model_strength"],
            "clip_strength": lora["clip_strength"],
            "enabled": lora["enabled"],
        }
        for asset_id, lora in zip(asset_ids, loras, strict=True)
        if asset_id is not None
    ]
    if not resolved or revision is None or not stack:
        return asset_ids
    try:
        resolve_lora_stack(session, revision, stack)
    except ValueError:
        refusals.append(_refusal("replay-lora-unusable", "lora"))
    return asset_ids


def _active_lora_assets(session: Session) -> list[ModelAssetInstall]:
    return list(
        session.scalars(
            select(ModelAssetInstall)
            .where(ModelAssetInstall.kind == "lora", ModelAssetInstall.active.is_(True))
            .order_by(ModelAssetInstall.id)
        ).all()
    )


def _check_record(
    record: dict[str, Any],
    operation: Operation,
    refusals: list[dict[str, Any]],
    *,
    workflow_chosen: bool = False,
    model_chosen: bool = False,
) -> None:
    """Refuse what the record itself says it lacks, and shapes a turn cannot carry.

    A workflow or model chosen to stand in for the record's own makes up for
    the record lacking it.
    """

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
    if workflow_chosen:
        missing -= {"workflow_unavailable", "workflow_unverified"}
    if model_chosen:
        missing.discard("model_files_not_recorded")
    if missing:
        # Each name comes from the format's own closed list, so none echoes input.
        refusals.append(_refusal("replay-record-incomplete", "record", reasons=sorted(missing)))
    reasons: list[str] = []
    if (
        not workflow_chosen
        and workflow is not None
        and workflow["contract_version"] != WORKFLOW_ARTIFACT_CONTRACT_VERSION
    ):
        reasons.append("workflow_contract_version")
    inputs = record["inputs"]
    # A selection travels as a setting, not as one of the turn's pictures.
    if len([item for item in inputs if item["role"] != "mask"]) > MAX_REPLAY_INPUTS:
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
    if record["version"] == PURPOSES_VERSION:
        # Pictures named by purpose: one made from words may read references,
        # and only a change may name the picture it changes. A change must name
        # it, or a turn would start from whatever picture it found instead.
        if operation in {Operation.TEXT_TO_IMAGE, Operation.TEXT_TO_VIDEO}:
            if "edit_source" in roles or "mask" in roles:
                reasons.append("inputs_for_operation")
        elif "edit_source" not in roles:
            reasons.append("inputs_for_operation")
    elif operation in {Operation.TEXT_TO_IMAGE, Operation.TEXT_TO_VIDEO}:
        if roles:
            reasons.append("inputs_for_operation")
    elif not roles or roles[0] != "source":
        reasons.append("inputs_for_operation")
    masks = [position for position, role in enumerate(roles) if role == "mask"]
    if masks and (
        operation != Operation.IMAGE_TO_IMAGE
        or masks != [len(roles) - 1]
        or "settings.mask" in record["removed"]
    ):
        # Only a selection given as its picture alone can be sent again, and a
        # record lists it after the pictures it was drawn over. How one was
        # feathered, turned round or blended back is not in the record.
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
    assets = _active_lora_assets(session)
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


def _inputs(
    session: Session,
    record: dict[str, Any],
    refusals: list[dict[str, Any]],
    chosen: dict[int, str],
) -> tuple[list[str], str | None, list[str] | None]:
    """Each recorded picture by its content identity, in recorded order, and the selection apart.

    A turn takes its selection as a setting rather than as one of its pictures.
    A picture chosen to stand in for one stands at its position, and takes that
    position's purpose; it must be a picture here, and no picture may then stand
    at two positions. Purposes come back only from a record that names them.
    """

    identifiers: list[str] = []
    purposes: list[str] = []
    mask: str | None = None
    for position, item in enumerate(record["inputs"]):
        if position in chosen:
            identifier = chosen[position]
            artifact = session.get(Artifact, identifier)
            if artifact is None or not str(artifact.media_type or "").startswith("image/"):
                refusals.append(_refusal("adaptation-input-unusable", "input", item["sha256"]))
                continue
        else:
            identifier = f"sha256:{item['sha256']}"
            if session.get(Artifact, identifier) is None:
                refusals.append(_refusal("replay-input-missing", "input", item["sha256"]))
                continue
        if item["role"] == "mask":
            mask = identifier
        else:
            identifiers.append(identifier)
            purposes.append(item["role"])
    if chosen and len({*identifiers, *([mask] if mask else [])}) != len(identifiers) + bool(mask):
        # A turn takes each picture once, so one given twice would run as one.
        refusals.append(_refusal("adaptation-input-unusable", "input"))
    return identifiers, mask, purposes if record["version"] == PURPOSES_VERSION else None


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

    turn, _left_out = await _record_turn_request(session, engines, record, resolved)
    return turn


async def adapted_turn_request(
    session: Session, engines: EngineRegistry, record: dict[str, Any], resolved: dict[str, Any]
) -> tuple[TurnRequest, list[str]]:
    """The record's turn against the chosen workflow and model, and the settings it could not send.

    A recorded setting the chosen workflow does not take - a seed or negative
    prompt among them - is left out of the turn and named.
    """

    return await _record_turn_request(session, engines, record, resolved)


async def _record_turn_request(
    session: Session, engines: EngineRegistry, record: dict[str, Any], resolved: dict[str, Any]
) -> tuple[TurnRequest, list[str]]:
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
        # A setting the workflow marks unavailable would be refused, not filled.
        if field.scope != "load" and field.available
    }
    recorded = {**record["settings"]["unbound"], **record["settings"]["bound"]}
    settings: dict[str, Any] = {key: value for key, value in recorded.items() if key in offered}
    left_out = sorted(key for key in recorded if key not in offered)
    if "seed" in offered:
        settings["seed"] = record["seed"]["value"]
    elif record["seed"]["value"] is not None:
        left_out.append("seed")
    if "negative_prompt" in offered:
        # Sent even when empty, so a default, profile or preset negative prompt
        # cannot step in where the record had none.
        settings["negative_prompt"] = record["prompt"]["negative"] or ""
    elif record["prompt"]["negative"]:
        left_out.append("negative_prompt")
    if "loras" in offered:
        # Sent even when empty, which also keeps automatic LoRAs out. One left
        # out by choice is not sent.
        loras = sorted(record["loras"], key=lambda item: item["position"])
        settings["loras"] = [
            {
                "asset_id": asset_id,
                "model_strength": lora["model_strength"],
                "clip_strength": lora["clip_strength"],
                "enabled": lora["enabled"],
            }
            for asset_id, lora in zip(resolved["lora_asset_ids"], loras, strict=True)
            if asset_id is not None
        ]
    if resolved["mask_artifact_id"] is not None:
        # The selection alone, as the record has it: nothing to feather or turn round.
        settings[MASK_SETTING_KEY] = {"artifact_id": resolved["mask_artifact_id"]}
    text = record["prompt"]["positive"]
    if record["operation"] == Operation.IMAGE_TO_IMAGE.value:
        # A run adds this wording to every edit; the turn carries only the request.
        text = text.removeprefix(edit_prompt_preamble())
    turn = TurnRequest(
        text=text,
        mode=resolved["mode"],
        profile_id=resolved["profile_id"],
        workflow_revision_id=resolved["workflow_revision_id"],
        # Null on purpose: every preset layer stays out of the turn.
        preset_id=None,
        output_count=1,
        input_artifact_ids=resolved["input_artifact_ids"],
        # The purposes the record names, so the replay is accepted with them again.
        input_image_roles=resolved.get("input_image_roles"),
        settings=settings,
    )
    return turn, left_out


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


def mark_adaptation(
    session: Session,
    run: Run,
    record: dict[str, Any],
    choices: list[dict[str, Any]],
    left_out: list[str],
) -> None:
    """Keep on an adapted run which record it came from, what stood in for what, and what differs.

    Inside the acceptance transaction, like the exact check; but a difference is
    the point of an adaptation, so it is recorded rather than refused.
    """

    session.flush()
    runs = session.scalars(select(Run).where(Run.work_plan_id == run.work_plan_id)).all()
    if len(runs) != 1:
        # A record is of one result, and so is a new version of it.
        raise ReplayDiffers(["output_count"])
    description = describe_run(session, run)
    differs = [
        name
        for name in REPLAYED_SECTIONS
        if canonical_bytes(_compared(name, description.sections[name]))
        != canonical_bytes(_compared(name, record[name]))
    ]
    # A setting the turn could not send but admission worked out to the
    # recorded value anyway - a video's frames from its length - was not lost.
    recorded = {
        **record["settings"]["unbound"],
        **record["settings"]["bound"],
        "seed": record["seed"]["value"],
        "negative_prompt": record["prompt"]["negative"] or "",
    }
    resolved = run.settings_json if isinstance(run.settings_json, dict) else {}
    lost = sorted(
        key for key in left_out if key not in resolved or resolved[key] != recorded.get(key)
    )
    run.provenance_json = {
        **run.provenance_json,
        ADAPTATION_KEY: {
            "record_digest": record["digest"],
            "choices": choices,
            "left_out_settings": lost,
            "differs": differs,
        },
    }


def _compared(name: str, value: Any) -> Any:
    """The part of a section an exact replay must reproduce."""

    if name == "workflow" and isinstance(value, dict):
        # Where the graph was read from is not what ran.
        return {key: item for key, item in value.items() if key != "graph_source"}
    if name == "model" and isinstance(value, dict):
        # The files are the model; where they came from is not compared.
        return value["files"]
    return value


def mark_replay(run: Run, record: dict[str, Any]) -> None:
    """Keep on a replayed run which record it is, and which output it should come out as."""

    run.provenance_json = {
        **run.provenance_json,
        REPLAY_RECEIPT_KEY: {
            "record_digest": record["digest"],
            "output_index": record["output"]["index"],
            "output_sha256": record["output"]["sha256"],
        },
    }


def replay_outcome(run: Run) -> dict[str, Any] | None:
    """Whether a replayed run's output came out as the record's did, or None for another run.

    Different bytes are reported, never refused: the record does not hold the
    executed graph or the runtime's version, so a matching run can still draw
    differently, and saying so is the point. A run made as a new version of a
    record answers that instead, with the parts of it that differ from the record.
    """

    if REPLAY_RECEIPT_KEY not in run.provenance_json:
        return _adapted(run.provenance_json.get(ADAPTATION_KEY))
    receipt = run.provenance_json.get(REPLAY_RECEIPT_KEY)
    if (
        not isinstance(receipt, dict)
        or not isinstance(receipt.get("record_digest"), str)
        or not _DIGEST.fullmatch(receipt["record_digest"])
        or not isinstance(receipt.get("output_sha256"), str)
        or not _HEX64.fullmatch(receipt["output_sha256"])
        or type(receipt.get("output_index")) is not int
        or receipt["output_index"] < 0
    ):
        return None
    state: ReplayOutcome
    if run.status in {RunStatus.FAILED.value, RunStatus.CANCELLED.value}:
        state = "output_missing"
    elif run.status != RunStatus.COMPLETE.value:
        state = "pending"
    else:
        outputs = run.provenance_json.get("outputs")
        kept = [
            item
            for item in (outputs if isinstance(outputs, list) else [])
            if isinstance(item, dict) and not names_a_preview(item.get("output_origin"))
        ]
        index = receipt["output_index"]
        if index >= len(kept):
            state = "output_missing"
        elif kept[index].get("artifact_id") == f"sha256:{receipt['output_sha256']}":
            state = "identical"
        else:
            state = "different"
    return {"state": state, "record_digest": receipt["record_digest"]}


def _adapted(receipt: object) -> dict[str, Any] | None:
    """What a run made as a new version of a record keeps about it, read strictly."""

    if (
        not isinstance(receipt, dict)
        or not isinstance(receipt.get("record_digest"), str)
        or not _DIGEST.fullmatch(receipt["record_digest"])
        or not isinstance(receipt.get("differs"), list)
        or not all(section in REPLAYED_SECTIONS for section in receipt["differs"])
    ):
        return None
    state: ReplayOutcome = "adapted"
    return {
        "state": state,
        "record_digest": receipt["record_digest"],
        "differs": list(receipt["differs"]),
    }
