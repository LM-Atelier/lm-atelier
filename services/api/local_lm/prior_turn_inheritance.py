"""Resolve edited operations from their source configurations without mutating defaults."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from .accepted_turn_context import AcceptedContext, accepted_context
from .domain import Operation
from .models import GenerationPreset, ModelProfile, Run
from .schemas import (
    PriorTurnEditConfiguration,
    PriorTurnEditRequest,
    PriorTurnEditSource,
    PriorTurnEditStepSource,
    TurnRequest,
)
from .turn_inheritance import TurnInheritance, inherited_edit_strength

if TYPE_CHECKING:
    from .orchestrator import ConversationOrchestrator


@dataclass
class SourceConfiguration:
    run_id: str
    role: str
    ordinal: int | None
    step_id: str | None
    values: PriorTurnEditConfiguration
    snapshot: AcceptedContext | None


@dataclass
class BoundConfiguration:
    source: SourceConfiguration
    inherit_profile: bool
    inherit_vision: bool
    inherit_workflow: bool
    inherit_preset: bool
    inherit_loras: bool


class PriorTurnInheritance:
    def __init__(
        self,
        orchestrator: ConversationOrchestrator,
        session: Session,
        source: PriorTurnEditSource,
        payload: PriorTurnEditRequest,
    ) -> None:
        self.orchestrator = orchestrator
        self.payload = payload
        self.selected_run_id = source.source_run_id
        self.sources: list[SourceConfiguration] = []
        self.bound: dict[int | None, BoundConfiguration] = {}
        self.consumed_steps: set[str] = set()
        configurations: list[PriorTurnEditSource | PriorTurnEditStepSource] = (
            list(source.steps) if source.steps else [source]
        )
        for config in configurations:
            run = session.get(Run, config.source_run_id)
            if run is None:
                raise ValueError("A source configuration is no longer available.")
            self.sources.append(
                SourceConfiguration(
                    run.id,
                    config.settings_role,
                    getattr(config, "ordinal", None),
                    getattr(config, "step_id", None),
                    config,
                    accepted_context(session, run),
                )
            )
        known = {item.step_id for item in self.sources if item.step_id is not None}
        if set(payload.step_overrides) - known:
            raise ValueError("An edited step does not belong to this source plan.")
        for item in self.sources:
            override = payload.step_overrides.get(item.step_id or "")
            if (
                override
                and override.workflow_selection
                and override.workflow_selection.selector_capability != item.role
            ):
                raise ValueError("An edited step workflow selection has a different role.")

    def _source(self, role: str, ordinal: int | None) -> SourceConfiguration | None:
        candidates = [item for item in self.sources if item.role == role]
        if ordinal is None:
            selected = next(
                (item for item in candidates if item.run_id == self.selected_run_id), None
            )
            return selected or (candidates[0] if len(candidates) == 1 else None)
        exact = next((item for item in candidates if item.ordinal == ordinal), None)
        if exact is not None:
            return exact
        return (
            candidates[0]
            if len(candidates) == 1 and candidates[0].step_id not in self.payload.step_overrides
            else None
        )

    async def resolve(
        self,
        session: Session,
        request: TurnRequest,
        operation: Operation,
        ordinal: int | None,
    ) -> tuple[TurnRequest, TurnInheritance]:
        role = self.orchestrator._role_for_operation(operation)
        source = self._source(role, ordinal)
        if source is None:
            return request, TurnInheritance()
        step_override = self.payload.step_overrides.get(source.step_id or "")
        if step_override is not None:
            role_override = request.role_overrides.get(role)
            fields = role_override.model_dump(exclude_unset=True) if role_override else {}
            fields.update(step_override.model_dump(exclude_unset=True))
            if role_override:
                fields["settings"] = {**role_override.settings, **step_override.settings}
            override = step_override.__class__.model_validate(fields)
            request = request.model_copy(
                update={"role_overrides": {**request.role_overrides, role: override}}
            ).for_role(role)
            self.consumed_steps.add(source.step_id or "")
        fields_set = set(request.model_fields_set)
        # Legacy global ordered choices apply only to their own role. A nested
        # role/step choice remains explicit and is validated by normal admission.
        if ordinal is not None:
            role_override = request.role_overrides.get(role)
            for key in ("profile_id", "preset_id"):
                selected_id = getattr(request, key)
                selected = (
                    (
                        session.get(ModelProfile, selected_id)
                        if key == "profile_id"
                        else session.get(GenerationPreset, selected_id)
                    )
                    if selected_id
                    else None
                )
                if (
                    selected is not None
                    and selected.role != role
                    and (role_override is None or key not in role_override.model_fields_set)
                ):
                    fields_set.discard(key)
        if "preset_id" not in fields_set:
            request = request.model_copy(update={"preset_id": None})
        workflow_override = bool({"workflow_selection", "workflow_revision_id"} & fields_set)
        inherited = BoundConfiguration(
            source=source,
            inherit_profile=not workflow_override and "profile_id" not in fields_set,
            inherit_vision=role == "chat" and "vision_profile_id" not in fields_set,
            inherit_workflow=not workflow_override,
            inherit_preset="preset_id" not in fields_set,
            inherit_loras=request.preset_id is None and "loras" not in request.settings,
        )
        values: dict[str, object] = {}
        if ordinal is None and "output_count" not in fields_set:
            values["output_count"] = source.values.output_count
        inherited_strength = (
            inherited_edit_strength(source.values.image_edit_strength, request, operation)
            if source.values.operation == operation.value
            else None
        )
        if inherited.inherit_profile:
            values["profile_id"] = source.values.profile_id
        if inherited.inherit_vision:
            values["vision_profile_id"] = source.values.vision_profile_id
        if inherited.inherit_workflow:
            values["workflow_revision_id"] = source.values.workflow_revision_id
            values["workflow_selection"] = None
        if inherited.inherit_preset:
            values["preset_id"] = None
        if request.preset_id is None or "preset_id" not in fields_set:
            baseline = await self.orchestrator.request_settings_for_operation(
                operation,
                source.values.resolved_settings,
                input_schema=source.values.workflow_schema,
                engine=source.values.profile_engine,
            )
            values["settings"] = {**baseline, **request.settings}
        self.bound[ordinal] = inherited
        context = source.snapshot
        return request.model_copy(update=values), TurnInheritance(
            profile=context.profile if context and inherited.inherit_profile else None,
            vision_profile=context.vision_profile if context and inherited.inherit_vision else None,
            workflow=context.workflow if context and inherited.inherit_workflow else None,
            image_edit_strength=inherited_strength,
        )

    def validate_consumed(self) -> None:
        if set(self.payload.step_overrides) - self.consumed_steps:
            raise ValueError(
                "The edited plan cannot apply a selected source step. Adjust its step settings."
            )

    def bind(self, session: Session, run: Run, ordinal: int | None) -> BoundConfiguration | None:
        bound = self.bound.get(ordinal)
        if bound is None:
            return None
        source = session.get(Run, bound.source.run_id)
        if source is None:
            raise ValueError("A source configuration is no longer available.")
        snapshot = bound.source.snapshot
        if bound.inherit_loras:
            before = (
                snapshot.auxiliary_assets
                if snapshot
                else source.provenance_json.get("auxiliary_assets") or {}
            )
            after = run.provenance_json.get("auxiliary_assets") or {}
            if (before.get("lora_stack") or []) != (after.get("lora_stack") or []):
                from .prior_turn_edits import EditRequestConflict

                raise EditRequestConflict(
                    "An inherited LoRA changed. Select it again before queuing this edit."
                )
        if bound.inherit_preset:
            run.provenance_json = {
                **run.provenance_json,
                "preset": copy.deepcopy(
                    snapshot.preset if snapshot else source.provenance_json.get("preset")
                ),
                "preset_layers": copy.deepcopy(
                    snapshot.preset_layers
                    if snapshot
                    else source.provenance_json.get("preset_layers") or []
                ),
            }
        return bound
