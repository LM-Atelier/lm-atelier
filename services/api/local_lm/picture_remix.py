"""A remix of a picture made elsewhere: its settings checked against a workflow chosen here.

The settings a picture's own file carries (external_generation_metadata) are
plain claims. Nothing in them chooses what runs: the person picks a workflow
revision and a model profile by their local ids, and each claim is then judged
against the fields admission itself uses for that pair.

- supported: the chosen workflow takes this setting and this value as it is;
- unresolved: it takes the setting, but this value cannot be confirmed to mean
  the same here (a name with no vocabulary to check it against, a setting the
  workflow binds more than once, a size that cannot be proven);
- incompatible: it does not take the setting, or refuses the value;
- ignored: a remix never uses it (how many pictures to make; for a remix from
  words, how much to change a starting picture).

A remix starts from words, or from the picture itself (its edit role): the
picture, chosen from the Media Library only, is the starting image of a
workflow that changes a picture it is given, through a plain encode, sample
and decode. Then the file's strength is the strength this picture was made
with, never offered where it would redraw the whole picture, and its size is
kept rather than set.

A preview resolves the choice exactly as the remix turn would be accepted: in
a new chat, with no saved preset or settings recipe, against a chat that is
never stored, and it writes nothing. Only supported claims can be applied; the
rest stay listed with their reasons. A remix is queued only when it resolves to
the same digest, and committed only when the accepted run is the one previewed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, Final, Literal

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from .accepted_turn_context import accepted_context
from .auxiliary_assets import prompt_trigger_word_provenance
from .db import SessionLocal
from .domain import ArtifactKind, Operation
from .engines import EngineNotConfiguredError, EngineSchemaUnavailableError
from .external_generation_metadata import ExternalGenerationMetadata, RemixClaim
from .image_edit_strength import (
    STRENGTH_PARAMETER,
    EditSettingSource,
    EditStrengthMode,
    resolve_image_edit_strength,
)
from .models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    ModelInstall,
    ModelProfile,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
)
from .orchestrator import ConversationOrchestrator, _chosen_setting_layers, _resolve_output_loras
from .outpaint_workflows import OUTPAINT_SETTING_KEY, extends_by_nothing
from .routing import ModalityRouter
from .schemas import SettingField, TurnRequest
from .settings_registry import WORKFLOW_LORA_OVERRIDES_SETTING_KEY, validate_settings
from .source_fit_image import MAX_SOURCE_BYTES, MAX_SOURCE_PIXELS
from .source_fit_preview import source_fit_capability
from .workflow_edit_calibration import safe_workflow_edit_calibration
from .workflow_graph_settings_v1 import GRAPH_SETTING_INPUT_NAMES, workflow_graph_settings
from .workflow_node_dependencies import node_dependency_errors
from .workflow_output_geometry import (
    WorkflowOutputGeometryResult,
    match_source_output_geometry,
    prove_workflow_output_geometry,
    resolve_workflow_output_geometry,
)
from .workflow_selection import WorkflowFamilySelectionError, resolve_exact_workflow_revision
from .workflow_source_geometry import trace_source_crop_route

if TYPE_CHECKING:
    from .artifacts import ArtifactStore

PREVIEW_VERSION: Final = 3
OPERATION: Final = Operation.TEXT_TO_IMAGE

RemixRole = Literal["words", "edit"]
#: What each role makes: a picture from words, or a change to the picture itself.
ROLE_OPERATIONS: Final = {"words": Operation.TEXT_TO_IMAGE, "edit": Operation.IMAGE_TO_IMAGE}

ClaimState = Literal["supported", "unresolved", "incompatible", "ignored"]

#: The setting each claim would set. ``guidance`` is the CFG scale every
#: reader takes it from; there is no setting by that name.
CLAIM_SETTINGS: Final = {
    "negative_prompt": "negative_prompt",
    "seed": "seed",
    "steps": "steps",
    "guidance": "cfg",
    "sampler": "sampler",
    "scheduler": "scheduler",
    "width": "width",
    "height": "height",
}
#: Claims a remix from words never uses, and why.
IGNORED_CLAIMS: Final = {"denoise": "edit_only", "batch": "one_picture"}
#: Claims a remix starting from the picture never uses, and why.
_EDIT_IGNORED_CLAIMS: Final = {"batch": "one_picture"}
#: The rotation tags that turn a picture on its side when it is shown.
_QUARTER_TURNS: Final = frozenset({5, 6, 7, 8})
_ORIENTATION_TAG: Final = 0x0112
#: The size a picture-to-picture workflow's encoder keeps every side a multiple of.
_ENCODED_SIDE: Final = 8
#: Names that mean something only when the workflow says which names it knows.
_NAMED: Final = frozenset({"sampler", "scheduler"})
_SIZE: Final = ("width", "height")

#: Why a choice cannot be previewed or queued, with the one sentence said about it.
REFUSAL_MESSAGES: Final = {
    "remix-prompt-missing": "This picture's file holds no prompt to make a picture from.",
    "remix-workflow-unusable": "This workflow cannot make a picture from words here.",
    "remix-model-unusable": "This model cannot run with this workflow.",
    "remix-engine-unavailable": "The image engine is not available to check this choice.",
    "remix-several-pictures": (
        "This picture's prompt asks for more than one picture, and a remix makes one."
    ),
    "remix-workflow-makes-several": (
        "This workflow makes more than one picture at a time, and a remix makes one."
    ),
    "remix-picture-unusable": (
        "This picture cannot be started from: only a picture in the Media Library that can "
        "be read here can."
    ),
    "remix-edit-unusable": (
        "This workflow does not change a picture it is given in a way a remix can check."
    ),
}
#: Where a remixed run keeps which picture and choices it came from.
REMIX_KEY: Final = "remix"


class RemixChoiceInvalid(ValueError):
    """What to apply names a claim that cannot be applied."""


class RemixDiffers(Exception):
    """What admission accepted is not the remix that was previewed."""


class _Refused(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ClaimVerdict:
    """One claim, the setting it would set, and whether the chosen workflow takes it.

    ``value`` is the claim as the file states it; ``setting_value`` is what
    applying it would set, which differs only when a whole number written as a
    decimal is given to a setting that takes whole numbers.
    """

    key: str
    setting: str | None
    value: str | int | float
    source: str
    state: ClaimState
    reason: str | None
    setting_value: str | int | float


@dataclass(frozen=True)
class RemixSource:
    """The picture a remix starts from, as its stored bytes are."""

    width: int
    height: int
    #: Whether a rotation tag turns it on its side when shown, so the size an
    #: engine gets is not the size the picture is seen at.
    turned: bool


@dataclass(frozen=True)
class RemixStrength:
    """How much a remix starting from the picture changes it, and where that came from."""

    parameter: str
    mode: str
    value: float
    from_file: bool


@dataclass(frozen=True)
class RemixPreview:
    """What a remix with this workflow and model would run, or why it cannot.

    ``request_settings`` are the settings the remix turn sends; ``settings`` are
    what the run is accepted with, which a remix is refused unless it matches.
    """

    metadata: ExternalGenerationMetadata
    role: RemixRole
    workflow_revision_id: str
    profile_id: str
    #: The picture a remix starts from, for the edit role.
    source: RemixSource | None
    claims: tuple[ClaimVerdict, ...]
    #: The picture's shape at a size this workflow makes, offered when its exact
    #: size cannot be used: width and height.
    shape: tuple[int, int] | None
    applied: tuple[str, ...]
    refusals: tuple[str, ...]
    text: str | None
    request_settings: dict[str, Any]
    settings: dict[str, Any]
    #: Whether a seed is drawn when the remix is made, rather than one set here or
    #: the workflow's own fixed seed.
    seed_drawn: bool
    trigger_words: tuple[str, ...]
    #: The words the engine is given: for a change to a picture, a fixed request
    #: to keep what the words do not change, then the words.
    engine_prompt: str | None
    strength: RemixStrength | None
    review_digest: str | None

    @property
    def ready(self) -> bool:
        return not self.refusals

    @property
    def operation(self) -> Operation:
        return ROLE_OPERATIONS[self.role]


def read_remix_source(artifacts: ArtifactStore, artifact_id: str) -> RemixSource | None:
    """The picture's own size, when it is a picture the Media Library shows; otherwise none.

    Only the stored bytes are read; nothing is kept. A picture larger than a
    starting picture can be, or one that cannot be opened, is no starting picture.
    """

    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        entry = session.scalar(
            select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact_id)
        )
        if (
            artifact is None
            or artifact.kind != ArtifactKind.IMAGE.value
            or entry is None
            or entry.state != "visible"
            or entry.deleted_at is not None
            or entry.recovery_id is not None
        ):
            return None
        try:
            content = artifacts.verified_bytes(artifact, maximum_bytes=MAX_SOURCE_BYTES)
        except (OSError, ValueError):
            return None
    try:
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            # Measured before anything is decoded: reading the rotation tag can decode it all.
            if width < 1 or height < 1 or width * height > MAX_SOURCE_PIXELS:
                return None
            # Each frame of a moving picture would be changed into a picture of its own.
            if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
                return None
            orientation = image.getexif().get(_ORIENTATION_TAG)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    return RemixSource(width, height, orientation in _QUARTER_TURNS)


@dataclass(frozen=True)
class _EditTerms:
    """What judging a claim needs when the remix starts from the picture."""

    source: RemixSource
    prompt: str
    fields: Sequence[SettingField]
    schema: dict[str, Any] | None


def classify_claims(
    claims: Iterable[RemixClaim],
    fields: Sequence[SettingField],
    revision: WorkflowRevision,
    geometry: WorkflowOutputGeometryResult | None,
    edit: _EditTerms | None = None,
) -> tuple[ClaimVerdict, ...]:
    """Judge each claim against the fields the chosen workflow takes from a request."""

    by_key = {item.key: item for item in fields}
    several = _several_controls(revision)
    by_claim = {claim.key: claim for claim in claims}
    ignored = _EDIT_IGNORED_CLAIMS if edit is not None else IGNORED_CLAIMS
    verdicts: list[ClaimVerdict] = []
    for claim in by_claim.values():
        if claim.key == "prompt":
            verdicts.append(_verdict(claim, None, "supported", None))
        elif claim.key in ignored:
            verdicts.append(_verdict(claim, None, "ignored", ignored[claim.key]))
        elif claim.key in _SIZE:
            continue
        elif claim.key == "denoise" and edit is not None:
            verdicts.append(_strength_verdict(claim, by_key, several, edit))
        else:
            setting = CLAIM_SETTINGS[claim.key]
            state, reason, value = _setting_state(setting, claim.value, by_key, several)
            verdicts.append(_verdict(claim, setting, state, reason, value))
    if edit is not None:
        verdicts.extend(_kept_size_verdicts(by_claim, edit.source))
    else:
        verdicts.extend(_size_verdicts(by_claim, by_key, several, geometry))
    return tuple(verdicts)


def _strength_verdict(
    claim: RemixClaim,
    by_key: dict[str, SettingField],
    several: frozenset[str],
    edit: _EditTerms,
) -> ClaimVerdict:
    """The strength the picture was made with, judged as a change to the picture itself.

    It is offered only where a turn would take it as its own chosen strength,
    and never at the most a workflow allows: that redraws the whole picture.
    """

    calibration = safe_workflow_edit_calibration(edit.schema)
    parameter = calibration.parameter if calibration else STRENGTH_PARAMETER
    state, reason, value = _setting_state(parameter, claim.value, by_key, several)
    if state != "supported" or isinstance(value, str):
        return _verdict(claim, parameter, state, reason, value)
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        edit.prompt,
        edit.fields,
        {parameter: value},
        [(EditSettingSource.TURN, {parameter: value})],
        workflow_schema=edit.schema,
    )
    if resolution is None or resolution.mode != EditStrengthMode.MANUAL:
        return _verdict(claim, parameter, "incompatible", "value_refused", value)
    if float(value) >= resolution.maximum:
        return _verdict(claim, parameter, "incompatible", "replaces_picture", value)
    return _verdict(claim, parameter, "supported", None, value)


def _kept_size_verdicts(by_claim: dict[str, RemixClaim], source: RemixSource) -> list[ClaimVerdict]:
    """A remix starting from the picture keeps its size; the claim only says whether it is that."""

    present = [by_claim[key] for key in _SIZE if key in by_claim]
    if not present:
        return []
    if len(present) == 1:
        return [_verdict(present[0], None, "unresolved", "size_incomplete")]
    width, height = (by_claim[key].value for key in _SIZE)
    state: ClaimState
    reason: str | None
    if source.turned:
        # A turned picture is shown at one size and stored, and so changed, at another.
        state, reason = "unresolved", "size_unproven"
    elif (width, height) != (source.width, source.height):
        state, reason = "incompatible", "size_differs"
    elif width % _ENCODED_SIDE or height % _ENCODED_SIDE:
        # Encoding trims a side to its multiple, so the kept size cannot be promised.
        state, reason = "unresolved", "size_unproven"
    else:
        state, reason = "supported", None
    return [_verdict(claim, None, state, reason) for claim in present]


def _verdict(
    claim: RemixClaim,
    setting: str | None,
    state: ClaimState,
    reason: str | None,
    value: str | int | float | None = None,
) -> ClaimVerdict:
    return ClaimVerdict(
        claim.key,
        setting,
        claim.value,
        claim.source,
        state,
        reason,
        claim.value if value is None else value,
    )


def _several_controls(revision: WorkflowRevision) -> frozenset[str]:
    """The settings the workflow puts in more than one place, or in one of several.

    Read from what the workflow's own graph says: a setting its generated
    controls bind twice, or split into one control per node, or bind once while
    a copy elsewhere is fixed; for a workflow without generated controls, a
    setting's input in more than one node, whether it holds a placeholder, a
    fixed value or a link; and a ``${setting}`` placeholder written into more
    than one input.
    """

    try:
        marker = workflow_graph_settings(revision.input_schema_json or {})
    except ValueError:
        marker = None
    several: set[str] = set()
    # Counted apart: a generated control also writes its own placeholder into the graph.
    controls: dict[str, int] = {}
    placeholders: dict[str, int] = {}
    if marker is not None:
        for entry in [*marker["bindings"], *marker.get("fixed", [])]:
            if entry.get("reason") == "disconnected":
                # A node that reaches no output never runs, so it is no second place.
                continue
            name = GRAPH_SETTING_INPUT_NAMES.get(entry.get("declared_name", entry["input_name"]))
            if name is None:
                continue
            controls[name] = controls.get(name, 0) + 1
            if entry.get("parameter", name) != name:
                # Split into one control per node: each pass has its own.
                several.add(name)
    for input_name, value in _graph_inputs(revision):
        if marker is None and input_name in GRAPH_SETTING_INPUT_NAMES:
            canonical = GRAPH_SETTING_INPUT_NAMES[input_name]
            controls[canonical] = controls.get(canonical, 0) + 1
        placeholder = _placeholder(value)
        if placeholder is not None:
            placeholders[placeholder] = placeholders.get(placeholder, 0) + 1
    for counts in (controls, placeholders):
        several.update(name for name, count in counts.items() if count > 1)
    return frozenset(several)


def _graph_inputs(revision: WorkflowRevision) -> list[tuple[str, object]]:
    """Every input of the workflow's stored graph, by name, with what it holds."""

    graph = revision.api_graph_json if isinstance(revision.api_graph_json, dict) else {}
    found: list[tuple[str, object]] = []
    for node in graph.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict):
            found.extend((name, value) for name, value in inputs.items() if isinstance(name, str))
    return found


def _placeholder(value: object) -> str | None:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return value[2:-1]
    return None


def _makes_several(revision: WorkflowRevision, settings: dict[str, Any]) -> bool:
    """Whether the workflow makes more than one picture at once, whatever a remix asks.

    A picture count is one only when every one the graph holds is the number 1,
    or a setting this remix resolves to 1. One fixed higher, set by another
    node or not known here would make several.
    """

    for input_name, value in _graph_inputs(revision):
        if input_name != "batch_size":
            continue
        placeholder = _placeholder(value)
        count = settings.get(placeholder) if placeholder is not None else value
        if isinstance(count, bool) or count != 1:
            return True
    return False


def _setting_state(
    setting: str,
    value: str | int | float,
    by_key: dict[str, SettingField],
    several: frozenset[str],
) -> tuple[ClaimState, str | None, str | int | float]:
    if setting in several:
        # One value would have to be put in several places, or in one of them.
        return "unresolved", "several_controls", value
    field = by_key.get(setting)
    if field is None or not field.available:
        return "incompatible", "not_offered", value
    if field.type == "integer" and isinstance(value, float) and value.is_integer():
        # "7" and "7.0" are one number; the setting takes it as a whole number.
        value = int(value)
    try:
        validate_settings({setting: value}, [field])
    except (ValueError, TypeError, OverflowError):
        return "incompatible", "not_a_choice" if setting in _NAMED else "value_refused", value
    if setting in _NAMED and not field.choices:
        return "unresolved", "no_vocabulary", value
    return "supported", None, value


def _size_verdicts(
    by_claim: dict[str, RemixClaim],
    by_key: dict[str, SettingField],
    several: frozenset[str],
    geometry: WorkflowOutputGeometryResult | None,
) -> list[ClaimVerdict]:
    """Width and height are judged together: a size is one shape, not two numbers."""

    present = [by_claim[key] for key in _SIZE if key in by_claim]
    if not present:
        return []
    if len(present) == 1:
        return [_verdict(present[0], present[0].key, "unresolved", "size_incomplete")]
    state: ClaimState = "supported"
    reason: str | None = None
    values: dict[str, str | int | float] = {}
    for claim in present:
        each, why, values[claim.key] = _setting_state(claim.key, claim.value, by_key, several)
        if each != "supported" and state == "supported":
            state, reason = each, why
    if state == "supported":
        width, height = values["width"], values["height"]
        if geometry is not None and geometry.available:
            request = {"mode": "image", "size_mode": "exact", "width": width, "height": height}
            if resolve_workflow_output_geometry(geometry, request) is None:
                state, reason = "incompatible", "size_not_offered"
        elif not all(_on_step(by_key[key], values[key]) for key in _SIZE):
            state, reason = "unresolved", "size_unproven"
    return [_verdict(claim, claim.key, state, reason, values[claim.key]) for claim in present]


def _on_step(field: SettingField, value: object) -> bool:
    step = field.step or field.multiple_of
    if not step or not isinstance(value, int):
        return True
    return value % step == 0


def shape_at_workflow_size(
    verdicts: Sequence[ClaimVerdict],
    fields: Sequence[SettingField],
    geometry: WorkflowOutputGeometryResult | None,
) -> tuple[int, int] | None:
    """The picture's shape at a size the workflow makes, when its exact size cannot be used.

    The workflow's own proof of the sizes it makes picks the pair: the exact
    ratio, at the legal size nearest the one it makes by default. A shape it
    cannot make exactly gets nothing, and so does a pair its own fields would
    refuse, so what is offered is what admission takes.
    """

    by_claim = {verdict.key: verdict for verdict in verdicts if verdict.key in _SIZE}
    if geometry is None or set(by_claim) != set(_SIZE):
        return None
    if all(verdict.state == "supported" for verdict in by_claim.values()):
        return None
    width, height = (by_claim[key].setting_value for key in _SIZE)
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    resolution = match_source_output_geometry(geometry, width, height)
    if resolution is None:
        return None
    shape = (resolution.geometry.width, resolution.geometry.height)
    by_key = {item.key: item for item in fields}
    if any(
        _setting_state(key, value, by_key, frozenset())[0] != "supported"
        for key, value in zip(_SIZE, shape, strict=True)
    ):
        return None
    return shape


def applied_settings(
    verdicts: Sequence[ClaimVerdict],
    apply: Iterable[str],
    shape: tuple[int, int] | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """The claims to apply, checked, and the settings they set.

    Only a supported claim can be applied, and a size only whole. The picture's
    shape can be applied only where one is offered, and never with its own size.
    The prompt is always the words a remix is made from, so it is never named here.
    """

    chosen = tuple(sorted(set(apply)))
    by_key = {verdict.key: verdict for verdict in verdicts}
    settings: dict[str, Any] = {}
    if "shape" in chosen:
        if shape is None or any(key in chosen for key in _SIZE):
            raise RemixChoiceInvalid("shape")
        settings.update(zip(_SIZE, shape, strict=True))
    for key in chosen:
        if key == "shape":
            continue
        verdict = by_key.get(key)
        if verdict is None or verdict.setting is None or verdict.state != "supported":
            # The prompt has no setting: it is always the words, never applied by name.
            raise RemixChoiceInvalid(key)
        settings[verdict.setting] = verdict.setting_value
    if any(key in chosen for key in _SIZE) and not all(key in chosen for key in _SIZE):
        raise RemixChoiceInvalid("size")
    return chosen, settings


def _resolution_chat() -> Chat:
    """The chat a remix resolves against: a new chat's, with nothing saved, never stored."""

    return Chat(
        id="chat_picture_remix",
        title="",
        scope="standard",
        project_id=None,
        generation_settings_json={},
        generation_preset_ids_json={},
        vision_settings_json={},
    )


def _require_clean(session: Session) -> None:
    if session.new or session.dirty or session.deleted:
        raise RuntimeError("A remix is resolved only from a session with nothing pending.")


def _image_profile(session: Session, profile_id: str) -> ModelProfile:
    profile = session.get(ModelProfile, profile_id)
    if profile is None or profile.role != "image":
        raise _Refused("remix-model-unusable")
    if profile.model_install_id:
        install = session.get(ModelInstall, profile.model_install_id)
        if install is None or not install.active or install.engine != profile.engine:
            raise _Refused("remix-model-unusable")
    return profile


def _geometry(session: Session, revision: WorkflowRevision) -> WorkflowOutputGeometryResult | None:
    """The revision's own proof of the sizes it makes, from its stored graph only."""

    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if definition is None:
        return None
    return prove_workflow_output_geometry(
        workflow_id=definition.id,
        revision_id=revision.id,
        operation=definition.operation,
        engine=revision.engine,
        api_graph=revision.api_graph_json,
        input_schema=revision.input_schema_json,
        dependencies=revision.dependencies_json,
        artifact_sha256=revision.artifact_sha256,
        trusted=revision.trusted,
    )


def review_digest(
    artifact_id: str,
    metadata: ExternalGenerationMetadata,
    role: RemixRole,
    source: RemixSource | None,
    workflow_revision_id: str,
    profile_id: str,
    claims: Sequence[ClaimVerdict],
    shape: tuple[int, int] | None,
    applied: Sequence[str],
    text: str,
    settings: dict[str, Any],
    seed_drawn: bool,
    trigger_words: Sequence[str],
    engine_prompt: str,
    strength: RemixStrength | None,
) -> str:
    """One digest over everything a person reviews, so what queues is what they saw."""

    reviewed = {
        "version": PREVIEW_VERSION,
        "artifact_id": artifact_id,
        "metadata": {
            "dialect": metadata.dialect,
            "digest": metadata.digest,
            "parser_version": metadata.parser_version,
            "budget_version": metadata.budget_version,
        },
        "role": role,
        "source": ([source.width, source.height, source.turned] if source is not None else None),
        "workflow_revision_id": workflow_revision_id,
        "profile_id": profile_id,
        "claims": [[claim.key, claim.state, claim.reason] for claim in claims],
        "shape": list(shape) if shape is not None else None,
        "applied": list(applied),
        "text": text,
        "settings": settings,
        "seed_drawn": seed_drawn,
        "trigger_words": list(trigger_words),
        "engine_prompt": engine_prompt,
        "strength": (
            [strength.parameter, strength.mode, strength.value, strength.from_file]
            if strength is not None
            else None
        ),
    }
    encoded = json.dumps(
        reviewed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return "sha256:" + hashlib.sha256(encoded.encode("ascii")).hexdigest()


async def preview_remix(
    orchestrator: ConversationOrchestrator,
    session: Session,
    artifact_id: str,
    metadata: ExternalGenerationMetadata,
    workflow_revision_id: str,
    profile_id: str,
    apply: Iterable[str],
    role: RemixRole = "words",
    source: RemixSource | None = None,
) -> RemixPreview:
    """Resolve a remix with this workflow and model exactly as a turn would, writing nothing.

    A choice that cannot run is an answer, not an error: the preview names why
    with one fixed code. Applying a claim that cannot be applied raises
    RemixChoiceInvalid, since that is a request no preview could honour. A
    remix starting from the picture needs its source, read by read_remix_source.
    """

    _require_clean(session)
    prompt = next((claim.value for claim in metadata.claims if claim.key == "prompt"), None)
    try:
        if not isinstance(prompt, str):
            raise _Refused("remix-prompt-missing")
        # The words as a turn takes them: one picture's prompt is the text, stripped.
        prompt = prompt.strip()
        if ModalityRouter.requested_output_count(prompt) > 1:
            # A turn would make as many as the words ask for, each with rewritten words.
            raise _Refused("remix-several-pictures")
        with session.no_autoflush:
            preview = await _resolve(
                orchestrator,
                session,
                artifact_id,
                metadata,
                workflow_revision_id,
                profile_id,
                apply,
                prompt,
                role,
                source,
            )
    except _Refused as refused:
        preview = RemixPreview(
            metadata=metadata,
            role=role,
            workflow_revision_id=workflow_revision_id,
            profile_id=profile_id,
            source=source,
            claims=(),
            shape=None,
            applied=(),
            refusals=(refused.code,),
            text=None,
            request_settings={},
            settings={},
            seed_drawn=False,
            trigger_words=(),
            engine_prompt=None,
            strength=None,
            review_digest=None,
        )
    _require_clean(session)
    return preview


async def _resolve(
    orchestrator: ConversationOrchestrator,
    session: Session,
    artifact_id: str,
    metadata: ExternalGenerationMetadata,
    workflow_revision_id: str,
    profile_id: str,
    apply: Iterable[str],
    prompt: str,
    role: RemixRole,
    source: RemixSource | None,
) -> RemixPreview:
    operation = ROLE_OPERATIONS[role]
    edit = role == "edit"
    if edit and source is None:
        raise _Refused("remix-picture-unusable")
    unusable = "remix-edit-unusable" if edit else "remix-workflow-unusable"
    chat = _resolution_chat()
    profile = _image_profile(session, profile_id)
    try:
        revision, _activation, bound_profile = resolve_exact_workflow_revision(
            session,
            workflow_revision_id,
            capability="image",
            operation=operation,
            engine=orchestrator.engines.settings.media_engine,
        )
    except WorkflowFamilySelectionError:
        raise _Refused(unusable) from None
    if bound_profile is not None and bound_profile.id != profile.id:
        raise _Refused("remix-model-unusable")
    if node_dependency_errors(session, revision.dependencies_json):
        raise _Refused(unusable)
    if edit:
        choice = TurnRequest(
            text=prompt,
            mode="image",
            preset_id=None,
            profile_id=profile.id,
            workflow_revision_id=revision.id,
            input_artifact_ids=[artifact_id],
        )
    else:
        choice = TurnRequest(
            text=prompt,
            preset_id=None,
            profile_id=profile.id,
            workflow_revision_id=revision.id,
        )
    try:
        chosen_profile, _selection, chosen_revision = orchestrator._execution_for_turn(
            session, chat, operation, prompt, choice
        )
    except (LookupError, ValueError):
        raise _Refused("remix-model-unusable") from None
    if (
        chosen_revision is None
        or chosen_revision.id != revision.id
        or chosen_profile is None
        or chosen_profile.id != profile.id
    ):
        raise _Refused("remix-model-unusable")
    try:
        turn_workflow = await orchestrator._turn_workflow(
            session, operation, chosen_profile, chosen_revision, None
        )
    except (EngineSchemaUnavailableError, EngineNotConfiguredError):
        raise _Refused("remix-engine-unavailable") from None
    except ValueError:
        raise _Refused(unusable) from None
    if edit and not _changes_the_picture(session, chosen_revision):
        raise _Refused("remix-edit-unusable")
    fields = turn_workflow.fields
    request_fields = [item for item in fields if item.scope != "load"]
    # A change to a picture keeps its size; no size or shape is offered for it.
    geometry = None if edit else _geometry(session, chosen_revision)
    terms = (
        _EditTerms(source, prompt, fields, chosen_revision.input_schema_json)
        if edit and source is not None
        else None
    )
    claims = classify_claims(metadata.claims, request_fields, chosen_revision, geometry, terms)
    shape = shape_at_workflow_size(claims, request_fields, geometry)
    applied, settings = applied_settings(claims, apply, shape)
    request_settings = settings
    offered = {item.key for item in request_fields if item.available}
    # Said outright, so no default, preset or word-matched LoRA fills them in:
    # what runs is only what the picture's file and the person chose.
    if "negative_prompt" in offered and "negative_prompt" not in settings:
        settings["negative_prompt"] = ""
    if "loras" in offered:
        settings["loras"] = []
    if "batch_size" in offered:
        settings["batch_size"] = 1
    try:
        layers = orchestrator.resolve_turn_setting_layers(
            session,
            chat,
            operation,
            chosen_profile,
            choice.model_copy(update={"settings": settings}),
            fields,
            workflow_revision=chosen_revision,
        )
    except (ValueError, TypeError, OverflowError):
        raise _Refused(unusable) from None
    if layers.mask is not None or layers.relight is not None:
        raise _Refused(unusable)
    effective = copy.deepcopy(dict(layers.effective_settings))
    # As a turn does: a picture from words takes its strength's default, and a
    # change to a picture takes the strength chosen here or estimates one from
    # the words, from the same layers a turn reads.
    resolution = resolve_image_edit_strength(
        operation,
        prompt,
        fields,
        effective,
        _chosen_setting_layers(
            chosen_profile,
            layers.workflow_lora_layers,
            layers.request_settings,
            use_case_settings=layers.use_case_settings,
        ),
        workflow_schema=chosen_revision.input_schema_json,
    )
    strength = None
    if edit:
        if resolution is None:
            raise _Refused("remix-edit-unusable")
        strength = RemixStrength(
            resolution.parameter, resolution.mode.value, resolution.value, "denoise" in applied
        )
    if extends_by_nothing(effective.get(OUTPAINT_SETTING_KEY)):
        effective.pop(OUTPAINT_SETTING_KEY, None)
    if OUTPAINT_SETTING_KEY in effective:
        raise _Refused(unusable)
    try:
        lora_outcome, lora_resolution = _resolve_output_loras(
            session,
            chosen_revision,
            turn_workflow.activation,
            layers.workflow_lora_layers,
            effective.get("loras", []),
            admit=layers.workflow_lora_layers.relevant_to(chosen_revision),
        )
    except ValueError:
        raise _Refused(unusable) from None
    if lora_resolution is not None:
        effective["loras"] = lora_resolution.settings
    if lora_outcome is not None:
        effective[WORKFLOW_LORA_OVERRIDES_SETTING_KEY] = lora_outcome.setting
    if _makes_several(chosen_revision, effective):
        raise _Refused("remix-workflow-makes-several")
    seed_drawn = effective.get("seed") == -1
    if seed_drawn:
        # A random seed is drawn when the remix is queued; a saved one is kept.
        effective.pop("seed")
    trigger_words = tuple(
        prompt_trigger_word_provenance(
            orchestrator._model_provenance(session, chosen_profile),
            lora_resolution.provenance if lora_resolution else [],
            prompt,
        )["trigger_words_applied"]
    )
    engine_prompt = _engine_prompt(operation, prompt, trigger_words)
    return RemixPreview(
        metadata=metadata,
        role=role,
        workflow_revision_id=chosen_revision.id,
        profile_id=chosen_profile.id,
        source=source,
        claims=claims,
        shape=shape,
        applied=applied,
        refusals=(),
        text=prompt,
        request_settings=copy.deepcopy(request_settings),
        settings=effective,
        seed_drawn=seed_drawn,
        trigger_words=trigger_words,
        engine_prompt=engine_prompt,
        strength=strength,
        review_digest=review_digest(
            artifact_id,
            metadata,
            role,
            source,
            chosen_revision.id,
            chosen_profile.id,
            claims,
            shape,
            applied,
            prompt,
            effective,
            seed_drawn,
            trigger_words,
            engine_prompt,
            strength,
        ),
    )


def _changes_the_picture(session: Session, revision: WorkflowRevision) -> bool:
    """Whether the workflow changes the picture it is given, at a strength it takes by name.

    The picture must reach the one saved picture through a plain encode,
    sample and decode, so it is the picture changed and not a node left aside;
    and that sampler's strength must be the setting a remix chooses.
    """

    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if definition is None or "crop" not in source_fit_capability(definition, revision).modes:
        return False
    graph = revision.api_graph_json
    saves = [
        key
        for key, node in graph.items()
        if isinstance(node, dict) and node.get("class_type") == "SaveImage"
    ]
    route = trace_source_crop_route(graph, saves[0]) if len(saves) == 1 else None
    if route is None:
        return False
    calibration = safe_workflow_edit_calibration(revision.input_schema_json)
    parameter = calibration.parameter if calibration else STRENGTH_PARAMETER
    sampler = graph.get(route.sampler_node_id)
    inputs = sampler.get("inputs") if isinstance(sampler, dict) else None
    return isinstance(inputs, dict) and inputs.get("denoise") == "$" + "{" + parameter + "}"


def _engine_prompt(operation: Operation, prompt: str, trigger_words: Sequence[str]) -> str:
    """The words the engine is given, written the way an accepted run writes them."""

    probe = Run(
        operation=operation.value,
        standalone_prompt=prompt,
        provenance_json={
            **(
                {"image_edit": {"policy": "preserve_unrequested_details_v1"}}
                if operation == Operation.IMAGE_TO_IMAGE
                else {}
            ),
            "auxiliary_assets": {"trigger_words_applied": list(trigger_words)},
        },
    )
    return ConversationOrchestrator._media_prompt(probe)


def remix_receipt(artifact_id: str, preview: RemixPreview) -> dict[str, Any]:
    """What a remixed run keeps: which picture, reading and choices; never a claim's value.

    Applied values are already the run's own prompt and settings, and the
    unchanged picture can be read again; the digest says which reading it was.
    """

    metadata = preview.metadata
    return {
        "version": PREVIEW_VERSION,
        "source_artifact_id": artifact_id,
        "metadata": {
            "dialect": metadata.dialect,
            "digest": metadata.digest,
            "parser_version": metadata.parser_version,
            "budget_version": metadata.budget_version,
        },
        "role": preview.role,
        "operation": preview.operation.value,
        "choices": {
            "workflow_revision_id": preview.workflow_revision_id,
            "profile_id": preview.profile_id,
        },
        # Where the strength came from, never its value: the run itself holds that.
        "strength": (
            {
                "parameter": preview.strength.parameter,
                "mode": preview.strength.mode,
                "from_file": preview.strength.from_file,
            }
            if preview.strength is not None
            else None
        ),
        "claims": [
            {
                "key": claim.key,
                "state": claim.state,
                "applied": claim.key in preview.applied
                or (
                    preview.role == "edit" and claim.setting is None and claim.state == "supported"
                ),
            }
            for claim in preview.claims
        ],
        "shape_applied": "shape" in preview.applied,
        "review_digest": preview.review_digest,
    }


def remix_check(artifact_id: str, preview: RemixPreview) -> Callable[[Session, Run], None]:
    """A check run inside the acceptance transaction, before anything is committed.

    The run must be the one previewed: one picture, with exactly the settings
    the preview showed, save a seed drawn because none was set; the words the
    engine is given, the picture it starts from and how its strength was
    chosen as shown. Anything else raises RemixDiffers, and nothing is committed.
    """

    def check(session: Session, run: Run) -> None:
        session.flush()
        runs = session.scalars(select(Run).where(Run.work_plan_id == run.work_plan_id)).all()
        if len(runs) != 1:
            raise RemixDiffers("output_count")
        accepted = dict(run.settings_json) if isinstance(run.settings_json, dict) else {}
        if "seed" not in preview.settings:
            accepted.pop("seed", None)
        if accepted != preview.settings:
            raise RemixDiffers("settings")
        context = accepted_context(session, run)
        if context is None or context.media_prompt != preview.engine_prompt:
            raise RemixDiffers("prompt")
        # The picture it starts from is exactly the one previewed, and nothing else.
        expected = [artifact_id] if preview.role == "edit" else []
        if run.operation != preview.operation.value or context.input_artifact_ids != expected:
            raise RemixDiffers("source")
        if preview.strength is not None:
            edit = run.provenance_json.get("image_edit")
            recorded = edit.get("strength") if isinstance(edit, dict) else None
            if (
                not isinstance(recorded, dict)
                or recorded.get("mode") != preview.strength.mode
                or recorded.get("parameter") != preview.strength.parameter
            ):
                raise RemixDiffers("strength")
        run.provenance_json = {
            **run.provenance_json,
            REMIX_KEY: remix_receipt(artifact_id, preview),
        }

    return check
