"""Write the portable record of one generated output from what its run recorded.

A run that froze its inputs is described from that frozen copy. Any other run
is described from its own rows and the workflow revision it names, which is
also what dispatch read. Every field is chosen on purpose: nothing is copied
wholesale from a run's provenance, an artifact's metadata or a frozen snapshot,
because each of those also holds chat text, local identifiers and file paths. A
field that could carry one of those is checked, and dropped by name when it does.

A record is refused rather than written when anything this computer alone would
recognise - an identifier this module read, or a path - survives into the bytes.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import __version__
from .accepted_turn_context import AcceptedContext, accepted_context
from .artifacts import ArtifactStore
from .model_planner import WORKFLOW_ARTIFACT_CONTRACT_VERSION, workflow_artifact_contract
from .models import Artifact, Message, Run, WorkflowDefinition, WorkflowRevision
from .orchestrator import ConversationOrchestrator
from .output_origin import names_a_preview
from .output_recipe_v1 import (
    NOT_RECORDED,
    OPERATIONS,
    SCHEMA_ID,
    SCHEMA_VERSION,
    OutputRecipeFormatError,
    record_digest,
    seal_output_recipe,
)
from .project_portability import has_local_path
from .settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS, workflow_settings

#: An identifier minted by this application, such as a run or chat id.
_LOCAL_ID = re.compile(r"\b[a-z]+_[0-9a-f]{32}\b")
#: How the ComfyUI adapter names the inputs it stages for one run.
_STAGED_NAME = re.compile(r"lm-atelier-[a-z]+_[0-9a-f]{32}", re.IGNORECASE)
#: Paths written in text, matched only in shapes prose does not take: "a cat / a
#: dog" is not a path, "/home/someone/x.png" is. A rooted path needs at least two
#: segments, so a lone slash between words never matches.
_PROSE_PATHS = (
    re.compile(r"(?i)\bfile:"),
    re.compile(r"(?i)(?<![\w])[a-z]:[\\/]"),
    re.compile(r"(?<![:\w])[\\/]{2}[^\\/\s\"'<>]+[\\/]"),
    re.compile(r"(?<![\w:/.~])/[^\s/\\]+/\S"),
    re.compile(r"(?<![\w\\])\\[^\s\\/]+\\"),
    re.compile(r"(?<![\w])~[\\/]"),
    re.compile(r"(?<![\w.])\.\.[\\/]"),
)
_SHA256_ID = re.compile(r"sha256:([0-9a-f]{64})")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_PLACEHOLDER = re.compile(r"\$\{([^{}]+)\}")

#: Settings that have a section of their own, or that name local rows.
_SECTIONED_SETTINGS: Final = frozenset(
    {"seed", "negative_prompt", "loras", "mask", "workflow_lora_overrides"}
)
_SOURCE_OPERATIONS: Final = frozenset({"image_to_image", "image_to_video"})
_MIN_LOCAL_VALUE_LENGTH: Final = 12
#: The most room a record gives its prompt, in the bytes the record writes.
_MAX_PROMPT_BYTES: Final = 128 * 1024
_MAX_SETTING_TEXT: Final = 4096
#: The engine's own text settings - a sampler, a scheduler, a codec. They name
#: how the engine runs, never what was asked for; the negative prompt, the one
#: text setting that is the person's words, has a section of its own.
_ENGINE_TEXT_SETTINGS: Final = frozenset(
    field.key
    for field in (*IMAGE_SETTINGS, *VIDEO_SETTINGS)
    if field.type in {"string", "enum"} and field.key not in _SECTIONED_SETTINGS
)


class OutputRecipeUnavailable(Exception):
    """Why no record can be written for this output, as a code and a sentence."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class OutputRecipe:
    """A sealed record and the name it downloads under."""

    content: bytes
    digest: str
    file_name: str


@dataclass
class _Draft:
    """What one build has decided so far, and every local value it has seen."""

    removed: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    local_values: set[str] = field(default_factory=set)

    def saw(self, *values: object) -> None:
        # A short value such as a request key of "1" would match ordinary text,
        # and the pattern check below already covers every minted identifier.
        for value in values:
            if (
                isinstance(value, str)
                and len(value) >= _MIN_LOCAL_VALUE_LENGTH
                and not _SHA256_ID.fullmatch(value)
            ):
                self.local_values.add(value)


def build_output_recipe(
    session: Session,
    artifacts: ArtifactStore,
    *,
    run_id: str,
    artifact_id: str,
    include_prompts: bool,
) -> OutputRecipe:
    """Write the record of one output of one run, or say why there is none."""

    run = session.get(Run, run_id)
    if run is None or not isinstance(run.provenance_json, dict):
        raise OutputRecipeUnavailable(
            404, "output-recipe-run-not-found", "This generation no longer exists."
        )
    if run.operation not in OPERATIONS:
        raise OutputRecipeUnavailable(
            422, "output-recipe-not-media", "Only a generated picture or video has a record."
        )
    draft = _Draft()
    draft.saw(
        run.id,
        run.chat_id,
        run.user_message_id,
        run.assistant_message_id,
        run.work_plan_id,
        run.work_step_id,
        run.profile_id,
        run.vision_profile_id,
        run.workflow_revision_id,
        run.idempotency_key,
    )
    entry, index, count = _selected_output(run, artifact_id)
    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        raise OutputRecipeUnavailable(
            404, "output-recipe-output-not-found", "This generation has no such output."
        )
    try:
        _path, media_type, _disposition = artifacts.delivery_metadata(artifact)
    except (OSError, ValueError) as exc:
        raise OutputRecipeUnavailable(
            410, "output-recipe-output-unreadable", "This output's file is missing or changed."
        ) from exc

    sections, snapshot = _describe(session, run, include_prompts, draft)
    origin = entry.get("output_origin")
    origin = origin if isinstance(origin, dict) else {}
    engine = _plain_string(origin.get("engine"))
    if engine == "mock":
        draft.missing.add("mock_engine")
    if _finished_after_generation(entry, snapshot):
        draft.missing.add("finished_after_generation")

    payload: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "version": SCHEMA_VERSION,
        "exported_by": {"application": "LM Atelier", "version": __version__},
        "output": {
            "sha256": artifact.sha256,
            "size_bytes": artifact.size_bytes,
            "media_type": media_type,
            "kind": "video" if entry.get("kind") == "video" else "image",
            "index": index,
            "count": count,
            "engine": engine,
            "node_id": _plain_string(origin.get("node_id")),
            "collection": _plain_string(origin.get("collection")),
            "raster": _raster(artifact),
        },
        **sections,
        "not_recorded": sorted(NOT_RECORDED),
    }
    payload["removed"] = sorted(draft.removed)
    draft.missing.update(_missing_for_removed(draft.removed))
    payload["reproducibility"] = {
        "status": "incomplete" if draft.missing else "recorded",
        "missing": sorted(draft.missing),
    }
    try:
        content = seal_output_recipe(payload)
    except OutputRecipeFormatError as exc:
        raise OutputRecipeUnavailable(
            409, "output-recipe-invalid", "This output's record could not be written."
        ) from exc
    _refuse_local_values(content, payload, draft)
    return OutputRecipe(
        content=content,
        digest=record_digest(payload),
        file_name=f"generation-record-{artifact.sha256[:12]}.json",
    )


@dataclass(frozen=True)
class RunDescription:
    """What a run executes, in the record's own terms, apart from any one output.

    The sections are the record's operation, prompt, seed, settings, inputs,
    workflow, model and loras, built by the same functions, so a run described
    here and a record written from it agree field for field.
    """

    sections: dict[str, Any]
    removed: frozenset[str]
    missing: frozenset[str]


def describe_run(session: Session, run: Run) -> RunDescription:
    """Describe what a run executes, with its prompt, as its record would."""

    draft = _Draft()
    sections, _snapshot_used = _describe(session, run, True, draft)
    removed = frozenset(draft.removed)
    return RunDescription(
        sections=sections,
        removed=removed,
        missing=frozenset(draft.missing | _missing_for_removed(set(removed))),
    )


def _describe(
    session: Session, run: Run, include_prompts: bool, draft: _Draft
) -> tuple[dict[str, Any], AcceptedContext | None]:
    snapshot = _snapshot(session, run, draft)
    settings = _execution_settings(run, snapshot)
    graph, schema, workflow = _workflow(session, run, snapshot, draft)
    placeholders = set(_placeholders(graph)) if graph is not None else None
    choices = _setting_choices(run.operation, schema)
    sections = {
        "operation": run.operation,
        "prompt": _prompt(session, run, snapshot, settings, include_prompts, draft),
        "seed": _seed(settings, placeholders, draft),
        "settings": _settings(settings, placeholders, choices, include_prompts, draft),
        "inputs": _inputs(session, run, snapshot, settings, draft),
        "workflow": workflow,
        "model": _model(run, draft),
        "loras": _loras(run, snapshot, draft),
    }
    return sections, snapshot


def _selected_output(run: Run, artifact_id: str) -> tuple[dict[str, Any], int, int]:
    """The run's own entry for this output, and its place among the kept outputs."""

    outputs = run.provenance_json.get("outputs")
    kept = [
        item
        for item in (outputs if isinstance(outputs, list) else [])
        if isinstance(item, dict) and not names_a_preview(item.get("output_origin"))
    ]
    matches = [index for index, item in enumerate(kept) if item.get("artifact_id") == artifact_id]
    if not matches:
        raise OutputRecipeUnavailable(
            404, "output-recipe-output-not-found", "This generation has no such output."
        )
    if len(matches) > 1:
        raise OutputRecipeUnavailable(
            409,
            "output-recipe-output-ambiguous",
            "This generation made the same file more than once.",
        )
    return kept[matches[0]], matches[0], len(kept)


def _snapshot(session: Session, run: Run, draft: _Draft) -> AcceptedContext | None:
    """The run's frozen inputs, when it has valid ones."""

    try:
        snapshot = accepted_context(session, run)
    except ValueError:
        draft.missing.add("frozen_snapshot_unavailable")
        return None
    if snapshot is None:
        draft.missing.add("frozen_snapshot_absent")
        return None
    draft.saw(
        snapshot.run_id,
        snapshot.chat_id,
        snapshot.source_message_id,
        snapshot.source_run_id,
        snapshot.profile_id,
        snapshot.vision_profile_id,
        snapshot.workflow_revision_id,
    )
    if snapshot.dependencies:
        # Dispatch adds the text and pictures of other outputs to these, so
        # what this run used is only complete together with them.
        draft.missing.add("depends_on_other_outputs")
    return snapshot


def _execution_settings(run: Run, snapshot: AcceptedContext | None) -> dict[str, Any]:
    settings = snapshot.settings if snapshot is not None else run.settings_json
    return dict(settings) if isinstance(settings, dict) else {}


def _prompt(
    session: Session,
    run: Run,
    snapshot: AcceptedContext | None,
    settings: Mapping[str, Any],
    include_prompts: bool,
    draft: _Draft,
) -> dict[str, Any]:
    """The prompt the engine was given, when the person asked for it to be included."""

    def omitted(reason: str) -> dict[str, Any]:
        draft.missing.add("prompt_omitted")
        draft.removed.add("prompt")
        return {"included": False, "positive": None, "negative": None, "omitted_reason": reason}

    if not include_prompts:
        return omitted("chosen")
    if _prompt_was_removed(session, run, snapshot) or (
        snapshot is None and _chat_has_removed_earlier_content(session, run)
    ):
        # Without a frozen copy, nothing records which earlier messages this
        # prompt quotes - a follow-up repeats the last picture's prompt, and a
        # request about a passage carries the passage. Once anything earlier
        # in the chat was taken back, the prompt may still hold it.
        return omitted("removed_from_chat")
    if "frozen_snapshot_unavailable" in draft.missing:
        # The frozen inputs exist but cannot be read, which is also what a
        # removed source message looks like. The run's own prompt may quote
        # that message, so it is left out rather than guessed safe.
        return omitted("unavailable")
    # Exactly what dispatch sends: the frozen prompt when there is one,
    # otherwise the same helper dispatch itself calls on the run.
    try:
        positive = (
            snapshot.media_prompt
            if snapshot is not None
            else ConversationOrchestrator._media_prompt(run)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        # Older or imported runs can hold provenance the helper does not expect.
        return omitted("unavailable")
    negative = settings.get("negative_prompt")
    negative = negative if isinstance(negative, str) and negative.strip() else None
    if not isinstance(positive, str) or not positive.strip():
        return omitted("unavailable")
    if not _safe_prose(positive) or (negative is not None and not _safe_prose(negative)):
        return omitted("contains_local_reference")
    if _encoded_size(positive) + _encoded_size(negative or "") > _MAX_PROMPT_BYTES:
        return omitted("too_long")
    return {"included": True, "positive": positive, "negative": negative, "omitted_reason": None}


def _prompt_was_removed(session: Session, run: Run, snapshot: AcceptedContext | None) -> bool:
    """Whether the person took back the message this prompt came from."""

    message_ids: set[str | None] = {run.user_message_id}
    if snapshot is not None:
        message_ids.add(snapshot.source_message_id)
        message_ids.update(item.source_message_id for item in snapshot.messages)
    for message_id in message_ids:
        if not message_id:
            continue
        message = session.get(Message, message_id)
        if message is None or message.content_removed_at is not None:
            return True
    return False


def _chat_has_removed_earlier_content(session: Session, run: Run) -> bool:
    """Whether any message before this run's own was taken back from its chat."""

    asked = session.get(Message, run.user_message_id) if run.user_message_id else None
    if asked is None:
        return True
    return (
        session.scalar(
            select(Message.id)
            .where(
                Message.chat_id == run.chat_id,
                Message.content_removed_at.is_not(None),
                Message.created_at <= asked.created_at,
            )
            .limit(1)
        )
        is not None
    )


def _seed(
    settings: Mapping[str, Any], placeholders: set[str] | None, draft: _Draft
) -> dict[str, Any]:
    value = settings.get("seed")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        draft.missing.add("seed_not_recorded")
        return {"value": None, "binding": "not_recorded"}
    if placeholders is None:
        binding = "unknown"
    else:
        # A seed reaches the engine only through its placeholder; without one
        # the number the graph was saved with is what ran.
        binding = "bound" if "seed" in placeholders else "graph_literal"
    return {"value": value, "binding": binding}


def _settings(
    settings: Mapping[str, Any],
    placeholders: set[str] | None,
    choices: Mapping[str, frozenset[str]],
    include_prompts: bool,
    draft: _Draft,
) -> dict[str, Any]:
    """Plain setting values, split by whether the graph reads them."""

    bound: dict[str, Any] = {}
    unbound: dict[str, Any] = {}
    mask = settings.get("mask")
    if isinstance(mask, dict) and not _plain_selection(mask):
        # How a selection is applied - inverted, feathered, blended - changes
        # what ran. The selection is listed among the inputs; how it was used is
        # not yet written, so the record says so.
        draft.removed.add("settings.mask")
    if settings.get("workflow_lora_overrides"):
        draft.removed.add("settings.workflow_lora_overrides")
    for key in sorted(settings):
        if key in _SECTIONED_SETTINGS:
            continue
        value = settings[key]
        if not _safe_name(key):
            draft.removed.add("settings")
            continue
        if isinstance(value, str):
            # The engine's own text settings and a setting's fixed choices are
            # not anybody's words. Text a workflow declares for itself can carry
            # the description, so it follows the prompt's inclusion.
            technical = key in _ENGINE_TEXT_SETTINGS or value in choices.get(key, frozenset())
            if (
                not (technical or include_prompts)
                or not _safe_prose(value)
                or len(value) > _MAX_SETTING_TEXT
            ):
                draft.removed.add(f"settings.{key}")
                continue
        elif not _plain_value(value):
            draft.removed.add(f"settings.{key}")
            continue
        (bound if placeholders is not None and key in placeholders else unbound)[key] = value
    return {"bound": bound, "unbound": unbound}


def _plain_selection(mask: dict[str, Any]) -> bool:
    """Whether a selection is applied exactly as its picture alone would be.

    Image Studio always says how far to feather and whether to turn the
    selection round; with no feathering and no turning, that is the selection
    as it is.
    """

    return (
        set(mask) <= {"artifact_id", "feather_px", "invert"}
        and mask.get("feather_px", 0) == 0
        and type(mask.get("feather_px", 0)) is int
        and mask.get("invert", False) is False
    )


def _setting_choices(operation: str, schema: object) -> dict[str, frozenset[str]]:
    """The fixed text choices each setting offers, from the engine and the workflow."""

    base = VIDEO_SETTINGS if "video" in operation else IMAGE_SETTINGS
    try:
        fields = workflow_settings(base, schema if isinstance(schema, dict) else None)
    except Exception:
        # A schema that cannot be read offers no choices, so its text settings
        # follow the prompt's inclusion like any other text.
        fields = []
    return {
        field.key: frozenset(choice for choice in field.choices if isinstance(choice, str))
        for field in fields
        if field.choices
    }


def _inputs(
    session: Session,
    run: Run,
    snapshot: AcceptedContext | None,
    settings: Mapping[str, Any],
    draft: _Draft,
) -> list[dict[str, Any]]:
    """The pictures this run was given, in order, by hash and role."""

    if snapshot is not None:
        identities = list(snapshot.input_artifact_ids)
    else:
        # The same helper dispatch calls, so a step fed by an earlier step's
        # picture lists that picture as its input.
        identities = ConversationOrchestrator.input_artifact_ids_for_run(session, run)
        provenance = run.provenance_json
        if provenance.get("resolved_dependency_artifact_ids") or provenance.get(
            "resolved_dependency_text"
        ):
            draft.missing.add("depends_on_other_outputs")
    mask = settings.get("mask")
    mask_id = mask.get("artifact_id") if isinstance(mask, dict) else None
    ordered: list[tuple[str, str]] = []
    for position, identity in enumerate(dict.fromkeys(identities)):
        # A selection that was also one of the pictures stays listed as that
        # picture too, so the record never hides a picture the run was given.
        role = "source" if position == 0 and run.operation in _SOURCE_OPERATIONS else "input"
        ordered.append((identity, role))
    if isinstance(mask_id, str):
        ordered.append((mask_id, "mask"))
    entries: list[dict[str, Any]] = []
    for identity, role in ordered:
        match = _SHA256_ID.fullmatch(identity)
        if match is None:
            draft.removed.add("inputs")
            continue
        artifact = session.get(Artifact, identity)
        if artifact is None:
            draft.missing.add("input_unavailable")
        entries.append(
            {
                "sha256": match.group(1),
                "role": role,
                "size_bytes": artifact.size_bytes if artifact is not None else None,
                "media_type": _plain_string(artifact.media_type) if artifact is not None else None,
            }
        )
    return entries


def _workflow(
    session: Session,
    run: Run,
    snapshot: AcceptedContext | None,
    draft: _Draft,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """The graph and settings schema that ran, and the workflow's identity recomputed from them."""

    witness = run.provenance_json.get("workflow")
    witness = witness if isinstance(witness, dict) else {}
    draft.saw(witness.get("family_id"), witness.get("definition_id"), witness.get("revision_id"))
    activation = (
        snapshot.workflow_activation
        if snapshot is not None and snapshot.workflow_activation is not None
        else witness.get("activation")
    )
    binding = activation.get("binding_sha256") if isinstance(activation, dict) else None
    if isinstance(activation, dict):
        draft.saw(activation.get("id"))
    frozen = snapshot.workflow if snapshot is not None else None
    if frozen is not None:
        graph_source = "frozen"
        engine = frozen.engine
        graph = frozen.api_graph_json
        schema = frozen.input_schema_json
        dependencies = frozen.dependencies_json
        stored = frozen.artifact_sha256
        dependency_contract = frozen.dependency_contract_sha256
        draft.saw(frozen.id, frozen.workflow_id)
    else:
        revision = (
            session.get(WorkflowRevision, run.workflow_revision_id)
            if run.workflow_revision_id
            else None
        )
        if revision is None:
            draft.missing.add("workflow_unavailable")
            return None, None, None
        graph_source = "live"
        engine = revision.engine
        graph = revision.api_graph_json
        schema = revision.input_schema_json
        dependencies = revision.dependencies_json
        stored = revision.artifact_sha256
        dependency_contract = revision.dependency_contract_sha256
        draft.saw(revision.id, revision.workflow_id)
    definition_operation = _definition_operation(session, witness, frozen, run)
    if not isinstance(graph, dict) or not isinstance(schema, dict) or definition_operation is None:
        draft.missing.add("workflow_unavailable")
        return None, None, None
    try:
        recomputed = workflow_artifact_contract(
            operation=definition_operation,
            engine=engine,
            api_graph=graph,
            input_schema=schema,
            dependencies=dependencies if isinstance(dependencies, dict) else {},
        )
    except (TypeError, ValueError):
        draft.missing.add("workflow_unavailable")
        return None, None, None
    verified = stored == recomputed
    if not verified:
        draft.missing.add("workflow_unverified")
    return (
        graph,
        schema,
        {
            "engine": engine,
            "operation": definition_operation,
            "artifact_sha256": recomputed,
            "verified": verified,
            "graph_source": graph_source,
            "contract_version": WORKFLOW_ARTIFACT_CONTRACT_VERSION,
            "dependency_contract_sha256": _hex_or_none(dependency_contract),
            "binding_sha256": _hex_or_none(binding),
        },
    )


def _definition_operation(
    session: Session,
    witness: Mapping[str, Any],
    frozen: object,
    run: Run,
) -> str | None:
    """The workflow's own operation, which its identity covers, not the run's."""

    operation = witness.get("operation")
    if isinstance(operation, str) and operation in OPERATIONS:
        return operation
    definition_id = getattr(frozen, "workflow_id", None) or witness.get("definition_id")
    if not isinstance(definition_id, str) and run.workflow_revision_id:
        revision = session.get(WorkflowRevision, run.workflow_revision_id)
        definition_id = revision.workflow_id if revision is not None else None
    definition = session.get(WorkflowDefinition, definition_id) if definition_id else None
    if definition is None or definition.operation not in OPERATIONS:
        return None
    return definition.operation


def _model(run: Run, draft: _Draft) -> dict[str, Any] | None:
    """The model's files by hash and where it came from, as the run recorded them."""

    model = run.provenance_json.get("model")
    if not isinstance(model, dict):
        draft.missing.add("model_files_not_recorded")
        return None
    draft.saw(model.get("profile_id"), model.get("install_id"))
    manifest = model.get("manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    expected = manifest.get("expected_sha256")
    files: dict[str, str] = {}
    for name, digest in expected.items() if isinstance(expected, dict) else ():
        if isinstance(name, str) and _safe_file_name(name) and _HEX64.fullmatch(str(digest)):
            files[name] = str(digest)
        else:
            draft.removed.add("model.files")
    if not files:
        draft.missing.add("model_files_not_recorded")
    source = model.get("source")
    source = source if isinstance(source, dict) else {}
    result: dict[str, Any] = {"files": files}
    for key in ("provider", "remote_id", "revision"):
        result[key] = _identity_string(source.get(key), f"model.{key}", draft)
    result["content_rating"] = _identity_string(
        manifest.get("content_rating"), "model.content_rating", draft
    )
    return result


def _loras(run: Run, snapshot: AcceptedContext | None, draft: _Draft) -> list[dict[str, Any]]:
    """Each added LoRA by hash and strength; its local name and asset stay behind."""

    auxiliary = (
        snapshot.auxiliary_assets
        if snapshot is not None
        else run.provenance_json.get("auxiliary_assets")
    )
    stack = auxiliary.get("lora_stack") if isinstance(auxiliary, dict) else None
    entries: list[dict[str, Any]] = []
    for item in stack if isinstance(stack, list) else []:
        if not isinstance(item, dict):
            draft.missing.add("lora_identity_missing")
            continue
        draft.saw(item.get("asset_id"))
        if any(key in item for key in ("name", "comfy_name", "trigger_words")):
            draft.removed.add("loras.names")
        digest = item.get("sha256")
        strengths = (item.get("model_strength"), item.get("clip_strength"))
        position = item.get("position")
        if (
            not isinstance(digest, str)
            or not _HEX64.fullmatch(digest)
            or not all(_finite_number(strength) for strength in strengths)
            or isinstance(position, bool)
            or not isinstance(position, int)
        ):
            draft.missing.add("lora_identity_missing")
            continue
        entries.append(
            {
                "sha256": digest,
                "model_strength": strengths[0],
                "clip_strength": strengths[1],
                "enabled": item.get("enabled") is not False,
                "position": position,
            }
        )
    return entries


def _finished_after_generation(entry: Mapping[str, Any], snapshot: AcceptedContext | None) -> bool:
    """Whether the stored picture is the workflow's output after more work on it.

    A fitted source is padded or cropped before the engine sees it and put back
    afterwards, and a selection, relight or restore composites the result into
    the source. A replay of the workflow alone does not make those bytes.
    """

    finishing = {"region_edit", "relight", "source_restore", "source_fit_agreement"}
    return bool(finishing & set(entry)) or (
        snapshot is not None and snapshot.source_fit is not None
    )


def _missing_for_removed(removed: set[str]) -> set[str]:
    """What each kind of left-out field costs a replay."""

    missing: set[str] = set()
    for name in removed:
        if name == "settings" or name.startswith("settings."):
            missing.add("settings_removed")
        elif name == "inputs":
            missing.add("input_unavailable")
        elif name == "model.files":
            missing.add("model_files_not_recorded")
    return missing


def _encoded_size(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=True))


def _raster(artifact: Artifact) -> dict[str, int] | None:
    measurement = (artifact.metadata_json or {}).get("output_measurement")
    if not isinstance(measurement, dict):
        return None
    width, height = measurement.get("raster_width"), measurement.get("raster_height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return {"width": width, "height": height}
    return None


def _placeholders(value: object) -> Iterator[str]:
    """Every name a graph substitutes whole, which is the only way a setting reaches it."""

    if isinstance(value, dict):
        for item in value.values():
            yield from _placeholders(item)
    elif isinstance(value, list):
        for item in value:
            yield from _placeholders(item)
    elif isinstance(value, str):
        match = _PLACEHOLDER.fullmatch(value)
        if match is not None:
            yield match.group(1)


def _refuse_local_values(content: bytes, payload: Mapping[str, Any], draft: _Draft) -> None:
    """Refuse the record if anything only this computer would recognise is in it."""

    text = content.decode("ascii")
    leaked = (
        any(value in text for value in draft.local_values)
        or _LOCAL_ID.search(text) is not None
        or _STAGED_NAME.search(text) is not None
        or has_local_path(_without_prose(payload))
        or not all(_safe_prose(value) for value in _prose_values(payload))
    )
    if leaked:
        raise OutputRecipeUnavailable(
            409, "output-recipe-unsafe", "This output's record could not be written safely."
        )


def _without_prose(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The record without the fields that hold the person's own words."""

    rest = {key: value for key, value in payload.items() if key not in {"prompt", "settings"}}
    settings = payload.get("settings")
    if isinstance(settings, dict):
        rest["settings"] = {
            group: {key: value for key, value in values.items() if not isinstance(value, str)}
            for group, values in settings.items()
            if isinstance(values, dict)
        }
    return rest


def _prose_values(payload: Mapping[str, Any]) -> Iterator[str]:
    """The person's own words in a record: the prompt and the text settings."""

    prompt = payload.get("prompt")
    if isinstance(prompt, dict):
        yield from (value for value in prompt.values() if isinstance(value, str))
    settings = payload.get("settings")
    if isinstance(settings, dict):
        for values in settings.values():
            if isinstance(values, dict):
                yield from (value for value in values.values() if isinstance(value, str))


def _safe_prose(text: str) -> bool:
    return (
        _LOCAL_ID.search(text) is None
        and _STAGED_NAME.search(text) is None
        and not any(pattern.search(text) for pattern in _PROSE_PATHS)
    )


def _safe_name(name: str) -> bool:
    return bool(name) and len(name) <= 200 and _safe_prose(name) and not has_local_path(name)


def _safe_file_name(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return (
        _safe_name(name)
        and not name.startswith(("/", "\\"))
        and all(part not in {"", ".", ".."} for part in parts)
    )


def _identity_string(value: object, label: str, draft: _Draft) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value and _safe_name(value):
        return value
    draft.removed.add(label)
    return None


def _plain_string(value: object) -> str | None:
    if isinstance(value, str) and value and _safe_name(value):
        return value
    return None


def _plain_value(value: object) -> bool:
    return value is None or isinstance(value, bool) or _finite_number(value)


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value)


def _hex_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and _HEX64.fullmatch(value) else None
