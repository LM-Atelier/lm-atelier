from copy import deepcopy
from typing import Any

import pytest
from sqlalchemy import select
from test_workflow_loras import PRIVATE_LOADER_ID, REFERENCE, _seed_revision

from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.comfy_package_widgets import POWER_LORA_LOADER
from local_lm.comfy_workflow_compiler import compile_comfyui_ui_graph
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import WorkflowActivation
from local_lm.schemas import WorkflowLoraControlsOut
from local_lm.settings_registry import (
    IMAGE_SETTINGS,
    resolve_generation_settings,
    workflow_settings,
)
from local_lm.workflow_graph_settings import bind_compiled_workflow_settings
from local_lm.workflow_lora_composition import (
    compose_workflow_lora_graph,
    workflow_lora_composition_graph,
)
from local_lm.workflow_lora_execution import build_workflow_lora_execution_authority
from local_lm.workflow_lora_overrides import (
    WorkflowLoraOverrideLayer,
    parse_workflow_lora_overrides,
    resolve_workflow_lora_override_layers,
)
from local_lm.workflow_loras import workflow_lora_controls


@pytest.mark.parametrize("rgthree", [False, True])
@pytest.mark.parametrize("invalid_activation", [False, True])
def test_native_mapping_preserves_lora_controls_and_their_evidence_requirements(
    settings: Settings, rgthree: bool, invalid_activation: bool
) -> None:
    del settings
    with SessionLocal() as session:
        _, revision = _seed_revision(
            session, rgthree=rgthree, invalid_activation=invalid_activation
        )
        before = workflow_lora_controls(session, revision_id=revision.id)
        graph = deepcopy(revision.ui_graph_json)
        loader = graph["nodes"][1]
        loader["outputs"] = [
            {"name": "MODEL", "type": "MODEL", "links": [3]},
            {"name": "CLIP", "type": "CLIP", "links": []},
        ]
        graph["nodes"].append(
            {
                "id": "render",
                "type": "NeutralRender",
                "mode": 0,
                "inputs": [{"name": "model", "type": "MODEL", "link": 3}],
                "outputs": [],
                "widgets_values": [20],
            }
        )
        graph["links"].append([3, PRIVATE_LOADER_ID, 0, "render", 0, "MODEL"])
        info: dict[str, Any] = {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [["base.safetensors"]]}},
                "input_order": {"required": ["ckpt_name"]},
                "output": ["MODEL", "CLIP"],
            },
            "NeutralRender": {
                "input": {
                    "required": {"model": ["MODEL"], "steps": ["INT", {"min": 1, "max": 100}]}
                },
                "input_order": {"required": ["model", "steps"]},
                "output": [],
                "output_node": True,
            },
        }
        if rgthree:
            info[POWER_LORA_LOADER] = {
                "input": {"optional": {"model": ["MODEL"], "clip": ["CLIP"]}},
                "input_order": {"optional": ["model", "clip"]},
                "output": ["MODEL", "CLIP"],
            }
        else:
            info["LoraLoader"] = {
                "input": {
                    "required": {
                        "model": ["MODEL"],
                        "clip": ["CLIP"],
                        "lora_name": [[REFERENCE]],
                        "strength_model": ["FLOAT", {"min": -20, "max": 20}],
                        "strength_clip": ["FLOAT", {"min": -20, "max": 20}],
                    }
                },
                "input_order": {
                    "required": ["model", "clip", "lora_name", "strength_model", "strength_clip"]
                },
                "output": ["MODEL", "CLIP"],
            }
        compiled = compile_comfyui_ui_graph(graph, info)
        bound = bind_compiled_workflow_settings(compiled, compiled.api_graph, {})
        assert bound.api_graph["render"]["inputs"]["steps"] == "${steps}"
        assert (
            bound.api_graph[PRIVATE_LOADER_ID]["inputs"]
            == revision.api_graph_json[PRIVATE_LOADER_ID]["inputs"]
        )
        assert set(bound.input_schema["properties"]) == {"steps"}
        revision.ui_graph_json = graph
        revision.api_graph_json = bound.api_graph
        revision.input_schema_json = bound.input_schema
        session.commit()
        after = workflow_lora_controls(session, revision_id=revision.id)
        assert len(before.slots) == len(after.slots) == 1
        original, current = before.slots[0], after.slots[0]
        assert current.default_model_strength == original.default_model_strength
        assert current.default_clip_strength == original.default_clip_strength
        assert current.editability == original.editability
        assert current.read_only_reason == original.read_only_reason
        assert current.editable_fields == original.editable_fields
        if invalid_activation:
            assert current.editability == "detected_read_only"
            assert after.override_target is None
        else:
            assert current.editability == "editable"
            assert after.override_target is not None
            assert after.api_graph_sha256 != before.api_graph_sha256
            active = session.scalar(
                select(WorkflowActivation).where(
                    WorkflowActivation.workflow_revision_id == revision.id
                )
            )
            assert active is not None
            authority = build_workflow_lora_execution_authority(
                session, revision, activation_id=active.id
            )
            target = WorkflowLoraControlsOut.model_validate(after).model_dump(mode="json")[
                "override_target"
            ]
            assert isinstance(target, dict)
            target["overrides"] = [
                {
                    "slot_id": current.slot_id,
                    "loader_contract": current.loader_contract,
                    "loader_authority_sha256": current.loader_authority_sha256,
                    "changes": {"model_strength": 0.625},
                }
            ]
            overrides = parse_workflow_lora_overrides({"version": 1, "targets": [target]})
            resolution = resolve_workflow_lora_override_layers(
                catalog=authority.catalog, layers=(WorkflowLoraOverrideLayer("turn", overrides),)
            )
            composition = compose_workflow_lora_graph(
                session,
                revision,
                added_loras=[],
                workflow_activation_id=active.id,
                override_catalog=authority.catalog,
                slot_extraction=authority.slot_extraction,
                override_resolution=resolution,
            )
            fields = workflow_settings(IMAGE_SETTINGS, bound.input_schema)
            values = resolve_generation_settings(fields, turn_overrides={"steps": 37})
            dispatched = ComfyUIAdapter._compile(
                workflow_lora_composition_graph(composition), values
            )
            assert dispatched["render"]["inputs"]["steps"] == 37
            loader_inputs = dispatched[PRIVATE_LOADER_ID]["inputs"]
            if rgthree:
                assert loader_inputs["lora_1"]["strength"] == 0.625
                assert loader_inputs["lora_1"]["strengthTwo"] == original.default_clip_strength
            else:
                assert loader_inputs["strength_model"] == 0.625
                assert loader_inputs["strength_clip"] == original.default_clip_strength
            assert revision.api_graph_json == bound.api_graph
            assert revision.input_schema_json == bound.input_schema
