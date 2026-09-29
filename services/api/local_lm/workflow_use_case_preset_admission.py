"""Bind a recipe's input and setting requirements to one candidate revision."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

from .models import WorkflowDefinition, WorkflowRevision
from .schemas import SettingField
from .workflow_selection import (
    LegacyRevisionResolver,
    ResolvedWorkflowFamily,
    RevisionPreference,
    WorkflowSelectionMode,
    resolve_workflow_family,
)
from .workflow_use_case_preset_resolution import ResolvedWorkflowUseCasePreset
from .workflow_use_case_preset_settings import (
    SettingsRefusal,
    WorkflowUseCasePresetSettingsError,
    validate_workflow_use_case_preset_settings,
)
from .workflow_use_case_structure import (
    StructureRefusal,
    assess_workflow_use_case_structure,
)
from .workflow_use_cases_v1 import WorkflowUseCaseInputs, classify_workflow_use_case

AdmissionRefusal = (
    SettingsRefusal | StructureRefusal | Literal["workflow-use-case-preset-revision-required"]
)


class WorkflowUseCasePresetAdmissionError(ValueError):
    def __init__(self, code: AdmissionRefusal) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class AdmittedWorkflowUseCasePreset:
    workflow_revision_id: str
    preset: ResolvedWorkflowUseCasePreset
    upscale_kind: Literal["model", "resample"] | None


@dataclass(frozen=True)
class ResolvedWorkflowUseCasePresetFamily:
    selection: ResolvedWorkflowFamily
    admission: AdmittedWorkflowUseCasePreset


def resolve_workflow_use_case_preset_family(
    session: Session,
    preset: ResolvedWorkflowUseCasePreset,
    facts: WorkflowUseCaseInputs,
    *,
    mode: WorkflowSelectionMode,
    engine: str,
    fields: Iterable[SettingField],
    workflow_family_id: str | None = None,
    prompt: str = "",
    required_capabilities: Iterable[str] = (),
    legacy_revision_resolver: LegacyRevisionResolver | None = None,
    preferred_revision: RevisionPreference | None = None,
    accepts_added_loras: bool = False,
) -> ResolvedWorkflowUseCasePresetFamily:
    """Narrow the existing family choice before ranking, then admit its exact revision."""
    requirements = classify_workflow_use_case(facts)
    if preset.use_case != requirements.use_case:
        raise WorkflowUseCasePresetAdmissionError("workflow-use-case-preset-mismatch")
    engine_fields = tuple(fields)

    def admit(revision: WorkflowRevision | None) -> AdmittedWorkflowUseCasePreset:
        definition = session.get(WorkflowDefinition, revision.workflow_id) if revision else None
        return admit_workflow_use_case_preset(
            preset,
            facts,
            revision,
            definition,
            expected_revision_id=revision.id if revision else "",
            fields=engine_fields,
            accepts_added_loras=accepts_added_loras,
        )

    def eligibility(revision: WorkflowRevision | None) -> str | None:
        try:
            admit(revision)
        except WorkflowUseCasePresetAdmissionError as exc:
            return exc.code
        return None

    selection = resolve_workflow_family(
        session,
        capability=requirements.capability,
        operation=requirements.operation,
        mode=mode,
        workflow_family_id=workflow_family_id,
        prompt=prompt,
        engine=engine,
        required_capabilities=required_capabilities,
        legacy_revision_resolver=legacy_revision_resolver,
        preferred_revision=preferred_revision,
        revision_eligibility=eligibility,
    )
    revision = (
        session.get(WorkflowRevision, selection.workflow_revision_id)
        if selection.workflow_revision_id
        else None
    )
    return ResolvedWorkflowUseCasePresetFamily(selection, admit(revision))


def admit_workflow_use_case_preset(
    preset: ResolvedWorkflowUseCasePreset,
    facts: WorkflowUseCaseInputs,
    revision: WorkflowRevision | None,
    definition: WorkflowDefinition | None,
    *,
    expected_revision_id: str,
    fields: Iterable[SettingField],
    accepts_added_loras: bool = False,
) -> AdmittedWorkflowUseCasePreset:
    """Check one exact candidate after the caller's trust and readiness checks.

    Selection can use the fixed refusal code to exclude an automatic candidate
    or explain an incompatible explicit choice. Recheck the selected revision
    before recording the returned detached recipe in a run.
    """
    requirements = classify_workflow_use_case(facts)
    if revision is None or definition is None:
        raise WorkflowUseCasePresetAdmissionError("workflow-use-case-preset-revision-required")
    if (
        not expected_revision_id
        or revision.id != expected_revision_id
        or not definition.id
        or revision.workflow_id != definition.id
    ):
        raise WorkflowUseCasePresetAdmissionError("workflow-use-case-preset-revision-mismatch")
    if preset.use_case != requirements.use_case:
        raise WorkflowUseCasePresetAdmissionError("workflow-use-case-preset-mismatch")
    structure = assess_workflow_use_case_structure(
        facts,
        operation=definition.operation,
        engine=revision.engine,
        api_graph=revision.api_graph_json,
        input_schema=revision.input_schema_json,
    )
    if structure.reason is not None:
        raise WorkflowUseCasePresetAdmissionError(structure.reason)
    try:
        validated = validate_workflow_use_case_preset_settings(
            preset,
            revision,
            expected_revision_id=expected_revision_id,
            use_case=requirements.use_case,
            fields=fields,
            accepts_added_loras=accepts_added_loras,
        )
    except WorkflowUseCasePresetSettingsError as exc:
        raise WorkflowUseCasePresetAdmissionError(exc.code) from None
    return AdmittedWorkflowUseCasePreset(revision.id, validated.preset, structure.upscale_kind)
