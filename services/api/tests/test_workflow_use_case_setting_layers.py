"""Keep admitted recipe settings below explicit choices in both turn paths."""

from copy import deepcopy
from typing import Any, Literal

import pytest
from fastapi import FastAPI

from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.image_edit_strength import resolve_image_edit_strength
from local_lm.models import Chat, GenerationPreset, ModelProfile, Project
from local_lm.orchestrator import _chosen_setting_layers
from local_lm.schemas import SettingField, TurnRequest
from local_lm.workflow_use_case_errors import workflow_use_case_error
from local_lm.workflow_use_case_preset_admission import AdmittedWorkflowUseCasePreset
from local_lm.workflow_use_case_preset_resolution import ResolvedWorkflowUseCasePreset
from local_lm.workflow_use_case_preset_settings import WorkflowUseCasePresetSettingsError
from local_lm.workflow_use_cases_v1 import WorkflowUseCase

_FIELDS = [
    SettingField(
        key=key, label=key, type="number", default=0.1, minimum=0, maximum=1, scope="workflow"
    )
    for key in ("quality", "denoise")
] + [
    SettingField(key="context_length", label="Context", type="integer", default=2048, scope="load"),
    SettingField(key="options", label="Options", type="object", default={}, scope="workflow"),
]


def _admission(
    settings: dict[str, Any],
    mode: Literal["preset", "automatic", "unconfigured"] = "preset",
) -> AdmittedWorkflowUseCasePreset:
    return AdmittedWorkflowUseCasePreset(
        "revision",
        ResolvedWorkflowUseCasePreset(
            WorkflowUseCase.IMAGE_EDIT,
            mode,
            "chat" if mode != "unconfigured" else None,
            "recipe" if mode == "preset" else None,
            "Example recipe" if mode == "preset" else None,
            deepcopy(settings),
        ),
        None,
    )


@pytest.mark.parametrize("ordered", [False, True], ids=["single", "ordered"])
@pytest.mark.parametrize("choice", ["inherit", "selected", "override", "none"])
@pytest.mark.parametrize("mode", ["preset", "automatic", "unconfigured"])
def test_recipe_layer_yields_to_turn_choices_without_changing_legacy_layers(
    app: FastAPI,
    ordered: bool,
    choice: str,
    mode: Literal["preset", "automatic", "unconfigured"],
) -> None:
    orchestrator = app.state.services.orchestrator
    recipe_values = {"quality": 0.7, "denoise": 0.47, "options": {"sizes": [256]}}
    admission = _admission(recipe_values if mode == "preset" else {}, mode)
    with SessionLocal() as session:
        profile = ModelProfile(
            name="Example profile",
            role="image",
            engine="mock",
            load_settings_json={"context_length": 4096},
            request_settings_json={"quality": 0.2},
        )
        presets = [
            GenerationPreset(
                name=f"Example layer {index}",
                role="image",
                settings_json={"quality": value},
                is_default=index == 0,
            )
            for index, value in enumerate((0.3, 0.35, 0.5, 0.8))
        ]
        session.add_all([profile, *presets])
        session.flush()
        project = Project(
            name="Example project",
            generation_preset_ids_json={"image": presets[1].id},
            generation_settings_json={"image": {"quality": 0.4}},
        )
        session.add(project)
        session.flush()
        chat = Chat(
            title="Example chat",
            project_id=project.id,
            generation_preset_ids_json={"image": presets[2].id},
            generation_settings_json={"image": {"quality": 0.6, "denoise": 0.6}},
        )
        session.add(chat)
        session.commit()
        payload: dict[str, Any] = {"text": "Change the square to blue."}
        if choice in {"selected", "override"}:
            payload["preset_id"] = presets[3].id
        elif choice == "none":
            payload["preset_id"] = None
        if choice == "override":
            payload["settings"] = {"quality": 0.9, "denoise": 0.61}
        request = TurnRequest.model_validate(payload)
        unchanged = deepcopy((chat.generation_settings_json, admission.preset.settings_json))
        legacy = orchestrator.resolve_turn_setting_layers(
            session,
            chat,
            Operation.IMAGE_TO_IMAGE,
            profile,
            request,
            _FIELDS,
            ordered=ordered,
        )
        prepared = orchestrator.resolve_turn_setting_layers(
            session,
            chat,
            Operation.IMAGE_TO_IMAGE,
            profile,
            request,
            _FIELDS,
            ordered=ordered,
            use_case_preset=admission,
        )
        expected = (
            0.9
            if choice == "override"
            else 0.8
            if choice == "selected"
            else (0.7 if mode == "preset" else 0.6)
        )
        assert prepared.effective_settings["quality"] == expected
        assert prepared.effective_settings["context_length"] == 4096
        assert prepared.request_settings == legacy.request_settings
        assert prepared.workflow_lora_layers == legacy.workflow_lora_layers
        assert [
            (scope, preset.id, preset.name, values)
            for scope, preset, values in prepared.preset_layers
        ] == [
            (scope, preset.id, preset.name, values)
            for scope, preset, values in legacy.preset_layers
        ]
        assert prepared.mask == legacy.mask and prepared.relight == legacy.relight
        assert prepared.use_case_settings == (recipe_values if mode == "preset" else {})
        if mode != "preset":
            assert prepared.effective_settings == legacy.effective_settings
        chosen = _chosen_setting_layers(
            profile,
            prepared.workflow_lora_layers,
            prepared.request_settings,
            use_case_settings=prepared.use_case_settings,
        )
        strength = resolve_image_edit_strength(
            Operation.IMAGE_TO_IMAGE,
            request.text,
            _FIELDS,
            prepared.effective_settings,
            chosen,
        )
        assert strength is not None
        if choice == "override":
            assert strength.value == 0.61 and strength.provenance()["source_scope"] == "turn"
        elif mode == "preset":
            assert strength.value == 0.47
            assert strength.provenance()["source_scope"] == "use_case_preset"
            assert strength.provenance()["mode"] == "manual"
        assert unchanged == (chat.generation_settings_json, admission.preset.settings_json)
        assert not session.dirty and not session.new and not session.deleted
        if mode == "preset":
            admission.preset.settings_json["options"]["sizes"].append(512)
            assert prepared.use_case_settings["options"] == {"sizes": [256]}


@pytest.mark.parametrize("values", [{"context_length": 8192}, {"unknown": 1}, {"quality": 2}])
def test_recipe_layer_refuses_values_incompatible_with_final_request_fields(
    app: FastAPI,
    values: dict[str, Any],
) -> None:
    with SessionLocal() as session:
        chat = Chat(title="Example chat")
        session.add(chat)
        session.commit()
        with pytest.raises(
            WorkflowUseCasePresetSettingsError,
            match="^workflow-use-case-preset-settings-invalid$",
        ) as refusal:
            app.state.services.orchestrator.resolve_turn_setting_layers(
                session,
                chat,
                Operation.IMAGE_TO_IMAGE,
                None,
                TurnRequest(text="Change the square to blue."),
                _FIELDS,
                use_case_preset=_admission(values),
            )
        explained = workflow_use_case_error(refusal.value)
        assert explained is not None
        assert explained[0] == "workflow-use-case-preset-settings-invalid"
        assert "Correct it in Manage recipes" in explained[1]
        assert not session.dirty and not session.new and not session.deleted
