"""Edit workflows carry the LoRA stack: the item-22 slice-1 verification.

The owner asked for LoRAs on image edits. The orchestrator already resolves
the stack for every non-text operation, and the revision builder already
adds the `loras` schema wherever `detect_lora_extension` finds an insertion
point - so what needed proving is that a checkpoint-shaped image_to_image
template actually gets both. It does; these pin it.
"""

from __future__ import annotations

import pytest

from local_lm.auxiliary_assets import workflow_lora_extension
from local_lm.comfy_templates import ComfyTemplate, CompiledComfyTemplate
from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.downloads import DownloadManager
from local_lm.models import ModelInstall

pytestmark = pytest.mark.asyncio


async def test_a_checkpoint_edit_template_gains_the_lora_stack(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    # The standard checkpoint edit shape: loader feeding model and clip into
    # the sampler and prompt encoders - exactly what photo-edit workflows use.
    graph = {
        "loader": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
        "positive": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["loader", 1], "text": "edit instruction"},
        },
        "negative": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["loader", 1], "text": ""},
        },
        "sampler": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["loader", 0],
                "positive": ["positive", 0],
                "negative": ["negative", 0],
            },
        },
    }
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="image_checkpoint_edit_test",
            path=settings.data_dir / "checkpoint-edit-template.json",
            role="image",
            operation="image_to_image",
            score=1_000,
            sha256="8" * 64,
            dependencies=(),
        ),
        ui_graph={"nodes": []},
        api_graph=graph,
        input_schema={"type": "object", "properties": {}},
    )
    with SessionLocal() as session:
        install = ModelInstall(
            name="Checkpoint editor",
            role="image",
            engine="comfyui",
            local_path=str(settings.model_dir / "checkpoint-editor"),
            manifest_json={"family": "checkpoint-edit-test"},
            active=True,
        )
        session.add(install)
        session.flush()

        revision = DownloadManager._ensure_template_workflow(session, compiled, install)

        # The graph extension point was detected on the checkpoint shape...
        assert workflow_lora_extension(revision) == {
            "model": ["loader", 0],
            "clip": ["loader", 1],
        }
        # ...and the edit workflow's settings now offer the stack, which is
        # what surfaces the LoRA section in Image Settings for edit turns.
        loras = revision.input_schema_json["properties"]["loras"]
        assert loras["type"] == "array"
        assert loras["maxItems"] == 8
        from local_lm.models import WorkflowDefinition

        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        assert definition.operation == "image_to_image"
