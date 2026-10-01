"""Resolve both choices of a comparison exactly as a turn would, and accept nothing.

Each choice goes through the same steps a picture turn takes: the exact
workflow revision and model, the workflow's own settings, the setting layers
and the LoRA stack. Two things differ on purpose. Saved defaults and presets
never enter a choice, so what is compared is only what was chosen. And no LoRA
is ever added automatically, so a choice runs only the LoRAs it names.

Nothing is written. The resolution runs against a chat that is never stored,
and the session it reads must be as clean when it ends as when it began.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy.orm import Session

from .accepted_turn_context import (
    accepted_profile_provenance,
    capture_profile,
    capture_workflow,
)
from .auxiliary_assets import _family_key, prompt_trigger_word_provenance, workflow_model_family
from .config import Settings
from .default_output_shape import default_output_size
from .domain import Operation
from .engines import EngineNotConfiguredError, EngineSchemaUnavailableError
from .generation_experiments_v1 import (
    CONTRACT_VERSION,
    MAX_SEED,
    REFUSAL_MESSAGES,
    ArmPreflightOut,
    ExperimentArmRequest,
    ExperimentRefusalOut,
    GenerationExperimentPreflightOut,
    GenerationExperimentRefusalCode,
    GenerationExperimentRequest,
    PresetGeometry,
    RefusalAlternativeOut,
    ResourceEvidenceOut,
    SeedPolicyKind,
    canonical_sha256,
)
from .models import Chat, ModelInstall, ModelProfile
from .orchestrator import _resolve_output_loras, _workflow_execution_witness
from .outpaint_workflows import OUTPAINT_SETTING_KEY, extends_by_nothing
from .schemas import SettingField, TurnRequest
from .settings_registry import WORKFLOW_LORA_OVERRIDES_SETTING_KEY, validate_settings
from .studio_masks import MASK_SETTING_KEY
from .studio_relight import RELIGHT_SETTING_KEY
from .video_length import resolve_video_length_settings
from .workflow_node_dependencies import node_dependency_errors
from .workflow_selection import WorkflowFamilySelectionError, resolve_exact_workflow_revision

if TYPE_CHECKING:
    from .orchestrator import ConversationOrchestrator

Code = GenerationExperimentRefusalCode
OPERATION = Operation.TEXT_TO_IMAGE
EXPERIMENT_CHAT_SCOPE = "experiment"
# Shared by both choices, so a choice cannot set one for itself.
COMMON_SETTING_KEYS = frozenset({"negative_prompt", "seed", "width", "height", "batch_size"})
# Inputs a comparison from words does not take.
INPUT_SETTING_KEYS = frozenset({MASK_SETTING_KEY, RELIGHT_SETTING_KEY, OUTPAINT_SETTING_KEY})
_SETTING_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_REASONS: dict[str, GenerationExperimentRefusalCode] = {
    "revision_missing": Code.ARM_WORKFLOW_UNAVAILABLE,
    "engine_mismatch": Code.ARM_WORKFLOW_UNAVAILABLE,
    "revision_not_executable": Code.ARM_WORKFLOW_UNAVAILABLE,
    "revision_ignores_the_description": Code.ARM_WORKFLOW_UNAVAILABLE,
    "capability_mismatch": Code.ARM_WORKFLOW_UNAVAILABLE,
    "operation_mismatch": Code.ARM_OPERATION_MISMATCH,
    "revision_untrusted": Code.ARM_WORKFLOW_UNTRUSTED,
    "activation_not_ready": Code.ARM_ACTIVATION_NOT_READY,
    "profile_binding_missing": Code.ARM_MODEL_MISMATCH,
}


class ArmRefused(Exception):
    """One choice cannot run; carries the refusal it reports."""

    def __init__(
        self,
        code: GenerationExperimentRefusalCode,
        *,
        setting: str | None = None,
        alternative: RefusalAlternativeOut | None = None,
    ) -> None:
        super().__init__(code.value)
        self.code = code
        self.setting = setting
        self.alternative = alternative


@dataclass(frozen=True)
class ResolvedArm:
    """One choice as it would run, with the snapshot its digest is taken over."""

    ordinal: int
    label: str
    profile_id: str
    profile_name: str
    workflow_revision_id: str
    workflow_version: int
    workflow_activation_id: str | None
    model_family: str | None
    width: int
    height: int
    effective_settings: dict[str, Any]
    trigger_words_applied: tuple[str, ...]
    snapshot: dict[str, Any]
    snapshot_sha256: str


@dataclass(frozen=True)
class ExperimentResolution:
    request: GenerationExperimentRequest
    arms: tuple[ResolvedArm | None, ...]
    refusals: tuple[ExperimentRefusalOut, ...]
    seed_equivalence: Literal["same_family", "none"]
    confirmation_required: bool
    estimate: tuple[ResourceEvidenceOut, ...] = field(default=())
    preflight_sha256: str | None = None

    @property
    def compatible(self) -> bool:
        return not self.refusals

    def out(self) -> GenerationExperimentPreflightOut:
        arms = []
        for ordinal, (requested, resolved) in enumerate(
            zip(self.request.arms, self.arms, strict=True), start=1
        ):
            refused = resolved is None or any(
                refusal.arm_ordinal == ordinal for refusal in self.refusals
            )
            if resolved is None:
                arms.append(
                    ArmPreflightOut(
                        ordinal=ordinal,
                        label=requested.label,
                        outcome="refused",
                        profile_id=requested.profile_id,
                        workflow_revision_id=requested.workflow_revision_id,
                    )
                )
                continue
            arms.append(
                ArmPreflightOut(
                    ordinal=ordinal,
                    label=resolved.label,
                    outcome="refused" if refused else "compatible",
                    profile_id=resolved.profile_id,
                    profile_name=resolved.profile_name,
                    workflow_revision_id=resolved.workflow_revision_id,
                    workflow_version=resolved.workflow_version,
                    workflow_activation_id=resolved.workflow_activation_id,
                    model_family=resolved.model_family,
                    width=resolved.width,
                    height=resolved.height,
                    effective_settings=copy.deepcopy(resolved.effective_settings),
                    trigger_words_applied=list(resolved.trigger_words_applied),
                    snapshot_sha256=resolved.snapshot_sha256,
                )
            )
        return GenerationExperimentPreflightOut(
            outcome="compatible" if self.compatible else "refused",
            preflight_sha256=self.preflight_sha256,
            seed_equivalence=self.seed_equivalence,
            confirmation_required=self.confirmation_required,
            estimate=list(self.estimate),
            arms=arms,
            refusals=list(self.refusals),
        )


def _refusal(
    code: GenerationExperimentRefusalCode,
    *,
    arm_ordinal: int | None = None,
    setting: str | None = None,
    alternative: RefusalAlternativeOut | None = None,
) -> ExperimentRefusalOut:
    return ExperimentRefusalOut(
        code=code,
        arm_ordinal=arm_ordinal,
        setting=setting,
        alternative=alternative,
        message=REFUSAL_MESSAGES[code],
    )


def _setting_name(key: str) -> str | None:
    """A setting is named back only when it looks like one; anything else stays unnamed."""

    return key if _SETTING_NAME.fullmatch(key) else None


def _resolution_chat() -> Chat:
    """The chat a choice resolves against: no project, no defaults, never stored."""

    return Chat(
        id="chat_generation_experiment",
        title="",
        scope=EXPERIMENT_CHAT_SCOPE,
        project_id=None,
        generation_settings_json={},
        generation_preset_ids_json={},
        vision_settings_json={},
    )


def _require_clean(session: Session) -> None:
    if session.new or session.dirty or session.deleted:
        raise RuntimeError("A comparison is resolved only from a session with nothing pending.")


def _require_image_profile(session: Session, profile_id: str) -> ModelProfile:
    profile = session.get(ModelProfile, profile_id)
    if profile is None or profile.role != "image":
        raise ArmRefused(Code.ARM_PROFILE_UNAVAILABLE)
    if profile.model_install_id:
        install = session.get(ModelInstall, profile.model_install_id)
        if install is None or not install.active or install.engine != profile.engine:
            raise ArmRefused(Code.ARM_PROFILE_UNAVAILABLE)
    return profile


def _common_settings(
    session: Session,
    request: GenerationExperimentRequest,
    arm: ExperimentArmRequest,
    revision: Any,
    request_fields: list[SettingField],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The turn settings one choice runs with: its own, plus what both share.

    Returns them with the shape the choice was given, so a shape worked out
    from a preset is recorded as derived rather than asked for.
    """

    # A field the workflow declares but maps to no input it can edit is not a
    # setting this choice has: a value for it would change nothing it runs.
    by_key = {item.key: item for item in request_fields if item.available}
    for key in arm.settings:
        if key in COMMON_SETTING_KEYS:
            raise ArmRefused(Code.COMMON_INPUT_OVERRIDDEN, setting=key)
        if key in INPUT_SETTING_KEYS:
            raise ArmRefused(Code.ARM_INPUT_UNSUPPORTED, setting=key)
    for key, value in arm.settings.items():
        if key not in by_key and key != WORKFLOW_LORA_OVERRIDES_SETTING_KEY:
            raise ArmRefused(Code.ARM_SETTING_UNSUPPORTED, setting=_setting_name(key))
        try:
            validate_settings({key: value}, request_fields)
        except (ValueError, TypeError, OverflowError):
            raise ArmRefused(Code.ARM_SETTING_INVALID, setting=_setting_name(key)) from None
    settings: dict[str, Any] = copy.deepcopy(dict(arm.settings))
    if "negative_prompt" in by_key:
        # Always set, even when empty, so neither choice inherits a default the
        # other does not have.
        settings["negative_prompt"] = request.negative_prompt
    elif request.negative_prompt:
        raise ArmRefused(Code.ARM_SETTING_UNSUPPORTED, setting="negative_prompt")
    seed = by_key.get("seed")
    if (
        seed is None
        or seed.type != "integer"
        or seed.minimum is None
        or seed.minimum > 0
        or seed.maximum is None
        or seed.maximum < MAX_SEED
    ):
        raise ArmRefused(Code.ARM_SEED_UNSUPPORTED)
    settings["seed"] = 0
    geometry = request.geometry
    if isinstance(geometry, PresetGeometry):
        size = default_output_size(session, OPERATION, revision, geometry.preset_id, None)
        if size is None:
            raise ArmRefused(Code.ARM_GEOMETRY_UNREACHABLE)
        width, height, derived = size.width, size.height, True
    else:
        width, height, derived = geometry.width, geometry.height, False
    for key, value in (("width", width), ("height", height)):
        if key not in by_key:
            raise ArmRefused(Code.ARM_SETTING_UNSUPPORTED, setting=key)
        try:
            validate_settings({key: value}, request_fields)
        except (ValueError, TypeError, OverflowError):
            raise ArmRefused(Code.ARM_SETTING_INVALID, setting=key) from None
        settings[key] = value
    shape = {
        "intent": geometry.model_dump(mode="json"),
        "width": width,
        "height": height,
        "derived": derived,
    }
    return settings, shape


async def _resolve_arm(
    orchestrator: ConversationOrchestrator,
    session: Session,
    chat: Chat,
    request: GenerationExperimentRequest,
    arm: ExperimentArmRequest,
    ordinal: int,
) -> ResolvedArm:
    profile = _require_image_profile(session, arm.profile_id)
    try:
        revision, _activation, bound_profile = resolve_exact_workflow_revision(
            session,
            arm.workflow_revision_id,
            capability="image",
            operation=OPERATION,
            engine=orchestrator.engines.settings.media_engine,
        )
    except WorkflowFamilySelectionError as exc:
        raise ArmRefused(_REASONS.get(exc.reason, Code.ARM_WORKFLOW_UNAVAILABLE)) from None
    if bound_profile is not None and bound_profile.id != profile.id:
        raise ArmRefused(
            Code.ARM_MODEL_MISMATCH,
            alternative=RefusalAlternativeOut(profile_id=bound_profile.id),
        )
    if node_dependency_errors(session, revision.dependencies_json):
        raise ArmRefused(Code.ARM_PACKAGE_MISSING)
    choice = TurnRequest(
        text=request.prompt,
        preset_id=None,
        profile_id=profile.id,
        workflow_revision_id=revision.id,
    )
    try:
        chosen_profile, model_selection, chosen_revision = orchestrator._execution_for_turn(
            session, chat, OPERATION, request.prompt, choice
        )
    except LookupError:
        raise ArmRefused(Code.ARM_PROFILE_UNAVAILABLE) from None
    except ValueError:
        raise ArmRefused(Code.ARM_MODEL_MISMATCH) from None
    if (
        chosen_revision is None
        or chosen_revision.id != revision.id
        or chosen_profile is None
        or chosen_profile.id != profile.id
    ):
        raise ArmRefused(Code.ARM_MODEL_MISMATCH)
    model_selection = {**model_selection, "compatibility_only": True}
    try:
        turn_workflow = await orchestrator._turn_workflow(
            session, OPERATION, chosen_profile, chosen_revision, None
        )
    except (EngineSchemaUnavailableError, EngineNotConfiguredError):
        raise ArmRefused(Code.ARM_ENGINE_UNAVAILABLE) from None
    except ValueError:
        raise ArmRefused(Code.ARM_ACTIVATION_NOT_READY) from None
    activation = turn_workflow.activation
    fields = turn_workflow.fields
    request_fields = [item for item in fields if item.scope != "load"]
    settings, shape = _common_settings(session, request, arm, chosen_revision, request_fields)
    setting_request = choice.model_copy(update={"settings": settings})
    try:
        layers = orchestrator.resolve_turn_setting_layers(
            session,
            chat,
            OPERATION,
            chosen_profile,
            setting_request,
            fields,
            workflow_revision=chosen_revision,
        )
    except (ValueError, TypeError, OverflowError):
        raise ArmRefused(Code.ARM_SETTING_INVALID) from None
    if layers.mask is not None or layers.relight is not None:
        raise ArmRefused(Code.ARM_INPUT_UNSUPPORTED)
    effective, _video_length = resolve_video_length_settings(
        layers.effective_settings, chosen_revision.input_schema_json
    )
    # As for a turn: margins of nothing that nobody asked for are only the
    # workflow's declared default. Any others extend a picture this has none of.
    if extends_by_nothing(effective.get(OUTPAINT_SETTING_KEY)):
        effective.pop(OUTPAINT_SETTING_KEY, None)
    if OUTPAINT_SETTING_KEY in effective:
        raise ArmRefused(Code.ARM_INPUT_UNSUPPORTED, setting=OUTPAINT_SETTING_KEY)
    try:
        lora_outcome, lora_resolution = _resolve_output_loras(
            session,
            chosen_revision,
            activation,
            layers.workflow_lora_layers,
            effective.get("loras", []),
            admit=layers.workflow_lora_layers.relevant_to(chosen_revision),
        )
    except ValueError:
        raise ArmRefused(Code.ARM_LORA_REFUSED) from None
    if lora_resolution is not None:
        effective["loras"] = lora_resolution.settings
    if lora_outcome is not None:
        effective[WORKFLOW_LORA_OVERRIDES_SETTING_KEY] = lora_outcome.setting
    adaptations: list[dict[str, Any]] = []
    batch = effective.get("batch_size")
    if isinstance(batch, int) and batch > 1:
        adaptations.append({"setting": "batch_size", "from": batch, "to": 1})
        effective["batch_size"] = 1
    # The seed each picture runs with is chosen when the comparison is accepted.
    effective.pop("seed", None)
    witness = _workflow_execution_witness(session, chosen_revision, activation, model_selection)
    accepted = capture_profile(session, chosen_profile.id)
    workflow = capture_workflow(session, chosen_revision.id)
    model_family = workflow_model_family(session, chosen_revision)
    # The words a turn appends to its prompt for this model and these LoRAs,
    # decided here as a turn decides them when it is accepted, so what runs is
    # what was compared.
    trigger_words = prompt_trigger_word_provenance(
        orchestrator._model_provenance(session, chosen_profile),
        lora_resolution.provenance if lora_resolution else [],
        request.prompt,
    )
    snapshot = {
        "version": CONTRACT_VERSION,
        "operation": OPERATION.value,
        "profile": accepted.model_dump(mode="json") if accepted else None,
        "model_provenance": accepted_profile_provenance(accepted),
        "workflow": workflow.model_dump(mode="json") if workflow else None,
        "activation": copy.deepcopy(activation),
        "workflow_witness": copy.deepcopy(witness),
        "model_family": model_family,
        "requested_settings": copy.deepcopy(dict(arm.settings)),
        "effective_settings": copy.deepcopy(effective),
        "lora": {
            "stack": copy.deepcopy(lora_resolution.provenance) if lora_resolution else [],
            "graph_sha256": lora_resolution.graph_sha256 if lora_resolution else None,
            "workflow": copy.deepcopy(lora_outcome.receipt) if lora_outcome else None,
        },
        "trigger_words": copy.deepcopy(trigger_words),
        "geometry": shape,
        "adaptations": adaptations,
    }
    return ResolvedArm(
        ordinal=ordinal,
        label=arm.label,
        profile_id=chosen_profile.id,
        profile_name=chosen_profile.name,
        workflow_revision_id=chosen_revision.id,
        workflow_version=chosen_revision.version,
        workflow_activation_id=activation["id"] if activation else None,
        model_family=model_family,
        width=shape["width"],
        height=shape["height"],
        effective_settings=effective,
        trigger_words_applied=tuple(trigger_words["trigger_words_applied"]),
        snapshot=snapshot,
        snapshot_sha256=canonical_sha256(snapshot),
    )


def _same_family(first: str | None, second: str | None) -> bool:
    if not first or not second:
        return False
    # The one spelling every family comparison uses, so this check cannot
    # disagree with the one that admits a LoRA to the same workflow.
    return _family_key(first) == _family_key(second)


def _estimate(
    orchestrator: ConversationOrchestrator, arms: tuple[ResolvedArm, ...]
) -> tuple[int, int]:
    work = 0
    output_bytes = 0
    for arm in arms:
        figures = orchestrator._media_plan_estimate(OPERATION, arm.effective_settings, 1)
        work += int(figures["work_units"])
        output_bytes += int(figures["estimated_bytes"])
    return work, output_bytes


def common_inputs(request: GenerationExperimentRequest) -> dict[str, Any]:
    """What both choices share, in the form a comparison stores and digests it."""

    return {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "geometry": request.geometry.model_dump(mode="json"),
        "seed_policy": request.seed_policy.model_dump(mode="json"),
        "output_count": 1,
    }


def preflight_digest(common: dict[str, Any], arms: list[tuple[int, str, str]]) -> str:
    """One digest over the shared request and each choice's snapshot.

    Random seeds are drawn only when a comparison is accepted, so they are not
    part of it: a preflight and the create that follows it agree.
    """

    return canonical_sha256(
        {
            "contract_version": CONTRACT_VERSION,
            "operation": OPERATION.value,
            "common": common,
            "arms": [list(arm) for arm in arms],
        }
    )


async def resolve_generation_experiment(
    orchestrator: ConversationOrchestrator,
    settings: Settings,
    session: Session,
    request: GenerationExperimentRequest,
) -> ExperimentResolution:
    """Resolve both choices and every check between them, refusing all of it or none."""

    _require_clean(session)
    chat = _resolution_chat()
    resolved: list[ResolvedArm | None] = []
    refusals: list[ExperimentRefusalOut] = []
    with session.no_autoflush:
        for ordinal, arm in enumerate(request.arms, start=1):
            try:
                resolved.append(
                    await _resolve_arm(orchestrator, session, chat, request, arm, ordinal)
                )
            except ArmRefused as refused:
                resolved.append(None)
                refusals.append(
                    _refusal(
                        refused.code,
                        arm_ordinal=ordinal,
                        setting=refused.setting,
                        alternative=refused.alternative,
                    )
                )
        _require_clean(session)
    policy = request.seed_policy.kind
    seed_equivalence: Literal["same_family", "none"] = "none"
    confirmation_required = False
    estimate: tuple[ResourceEvidenceOut, ...] = ()
    arms = tuple(arm for arm in resolved if arm is not None)
    if len(arms) == len(request.arms):
        first, second = arms
        if policy == SeedPolicyKind.FIXED_NUMERIC:
            if _same_family(first.model_family, second.model_family):
                seed_equivalence = "same_family"
            else:
                refusals.append(
                    _refusal(
                        Code.SEED_FAMILY_UNPROVEN,
                        alternative=RefusalAlternativeOut(
                            seed_policy=SeedPolicyKind.SAME_RECORDED_NUMBER
                        ),
                    )
                )
        if (
            policy in {SeedPolicyKind.FIXED_NUMERIC, SeedPolicyKind.SAME_RECORDED_NUMBER}
            and first.snapshot_sha256 == second.snapshot_sha256
        ):
            refusals.append(_refusal(Code.ARMS_IDENTICAL))
        work, output_bytes = _estimate(orchestrator, arms)
        estimate = (
            ResourceEvidenceOut(
                resource="work_units",
                value=work,
                unit="work_units",
                kind="estimated",
                source="admission_formula",
                confidence="heuristic",
            ),
            ResourceEvidenceOut(
                resource="output_bytes",
                value=output_bytes,
                unit="bytes",
                kind="estimated",
                source="admission_formula",
                confidence="heuristic",
            ),
        )
        if (
            work > settings.max_media_plan_work_units
            or output_bytes > settings.max_media_plan_estimated_bytes
        ):
            refusals.append(_refusal(Code.EXPERIMENT_TOO_LARGE))
        confirmation_required = (
            settings.video_confirmation_work_units > 0
            and work >= settings.video_confirmation_work_units
        )
    if refusals:
        seed_equivalence = "none"
    preflight_sha256 = (
        preflight_digest(
            common_inputs(request),
            [(arm.ordinal, arm.label, arm.snapshot_sha256) for arm in arms],
        )
        if not refusals
        else None
    )
    return ExperimentResolution(
        request=request,
        arms=tuple(resolved),
        refusals=tuple(refusals),
        seed_equivalence=seed_equivalence,
        confirmation_required=confirmation_required,
        estimate=estimate,
        preflight_sha256=preflight_sha256,
    )
