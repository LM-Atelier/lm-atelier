"""Capture scoped recipes before asynchronous probes and admit exact revisions."""

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from .auxiliary_assets import revision_accepts_added_loras
from .domain import Operation
from .engines import EngineRegistry
from .models import WorkflowDefinition, WorkflowRevision
from .outpaint_workflows import OUTPAINT_SETTING_KEY
from .schemas import SettingField, TurnRequest
from .studio_masks import MASK_APPLY_BLEND, MASK_SETTING_KEY
from .upscale_workflows import UPSCALE_SETTING_KEY
from .workflow_selection import RevisionEligibility
from .workflow_use_case_preset_admission import (
    AdmittedWorkflowUseCasePreset,
    WorkflowUseCasePresetAdmissionError,
    admit_workflow_use_case_preset,
)
from .workflow_use_case_preset_provenance import WorkflowUseCasePresetSnapshot
from .workflow_use_case_preset_resolution import (
    ResolvedWorkflowUseCasePreset,
    resolve_workflow_use_case_preset,
)
from .workflow_use_cases_v1 import (
    SelectionApplication,
    WorkflowUseCaseInputs,
    classify_workflow_use_case,
)


@dataclass(frozen=True)
class InheritedWorkflowUseCasePreset:
    """Preserve an accepted recipe, including the absence of one on an older turn."""

    snapshot: WorkflowUseCasePresetSnapshot | None


@dataclass(frozen=True)
class WorkflowUseCaseExecution:
    preset: ResolvedWorkflowUseCasePreset
    facts: WorkflowUseCaseInputs
    fields: tuple[SettingField, ...]

    def admit(
        self,
        session: Session,
        revision: WorkflowRevision | None,
        *,
        fields: list[SettingField] | None = None,
    ) -> AdmittedWorkflowUseCasePreset:
        definition = session.get(WorkflowDefinition, revision.workflow_id) if revision else None
        return admit_workflow_use_case_preset(
            self.preset,
            self.facts,
            revision,
            definition,
            expected_revision_id=revision.id if revision else "",
            fields=self.fields if fields is None else fields,
            accepts_added_loras=revision is not None and revision_accepts_added_loras(revision),
        )

    def eligibility(self, session: Session) -> RevisionEligibility:
        def check(revision: WorkflowRevision | None) -> str | None:
            try:
                self.admit(session, revision)
            except WorkflowUseCasePresetAdmissionError as exc:
                return exc.code
            return None

        return check


def workflow_use_case_inputs(
    operation: Operation, request: TurnRequest, *, source_present: bool
) -> WorkflowUseCaseInputs:
    """Read canonical operation markers after the turn's source inputs are resolved."""
    mask = request.settings.get(MASK_SETTING_KEY)
    selection = SelectionApplication.NONE
    if mask is not None:
        selection = (
            SelectionApplication.BLEND
            if isinstance(mask, dict) and mask.get("apply") == MASK_APPLY_BLEND
            else SelectionApplication.WORKFLOW
        )
    return WorkflowUseCaseInputs(
        operation,
        source_present=source_present,
        selection=selection,
        extend=request.source_fit is not None or OUTPAINT_SETTING_KEY in request.settings,
        upscale=UPSCALE_SETTING_KEY in request.settings,
    )


async def prepare_workflow_use_case_execution(
    session_factory: Callable[[], Session],
    engines: EngineRegistry,
    facts: WorkflowUseCaseInputs,
    *,
    chat_id: str,
    inherited: InheritedWorkflowUseCasePreset | None = None,
) -> WorkflowUseCaseExecution | None:
    if inherited is not None and inherited.snapshot is None:
        return None
    requirements = classify_workflow_use_case(facts)
    if inherited is not None:
        assert inherited.snapshot is not None
        preset = inherited.snapshot.resolved_preset()
    else:
        with session_factory() as session:
            preset = resolve_workflow_use_case_preset(
                session, requirements.use_case, chat_id=chat_id
            )
    if preset.mode in {"unconfigured", "automatic"}:
        return None
    if preset.use_case != requirements.use_case:
        raise WorkflowUseCasePresetAdmissionError("workflow-use-case-preset-mismatch")
    fields = await engines.settings_for_role(requirements.capability)
    return WorkflowUseCaseExecution(preset, facts, tuple(fields))
