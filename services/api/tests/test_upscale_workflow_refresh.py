"""Refresh older catalog declarations without changing retained workflow revisions."""

import copy

import pytest

from local_lm import downloads
from local_lm.comfy_templates import ComfyTemplate, CompiledComfyTemplate
from local_lm.config import Settings
from local_lm.db import SessionLocal, configure_database, init_db
from local_lm.downloads import DownloadManager
from local_lm.models import ModelInstall


@pytest.mark.parametrize("adjustable", [False, True])
@pytest.mark.parametrize("previous_compiler", [23, 24])
def test_installed_enlargement_declarations_refresh_once(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    adjustable: bool,
    previous_compiler: int,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    factor = {"type": "number", "default": 2.5, "minimum": 1, "maximum": 4}
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="image_enlargement",
            path=settings.data_dir / "enlargement.json",
            role="image",
            operation="image_to_image",
            score=100,
            sha256="a" * 64,
            dependencies=(),
        ),
        ui_graph={"nodes": []},
        api_graph={
            "source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
            "scale": {
                "class_type": "ImageScaleBy",
                "inputs": {
                    "image": ["source", 0],
                    "scale_by": "${upscale_factor}" if adjustable else 4,
                },
            },
            "output": {"class_type": "SaveImage", "inputs": {"images": ["scale", 0]}},
        },
        input_schema={
            "type": "object",
            "properties": {"upscale_factor": factor} if adjustable else {},
        },
    )
    with SessionLocal() as session:
        install = ModelInstall(
            name="Image enlargement",
            role="image",
            engine="comfyui",
            local_path=str(settings.model_dir / "enlargement"),
            manifest_json={},
            active=True,
        )
        session.add(install)
        session.flush()
        with monkeypatch.context() as previous:
            previous.setattr(downloads, "COMFY_TEMPLATE_COMPILER_VERSION", previous_compiler)
            old = DownloadManager._ensure_template_workflow(session, compiled, install)
        old.input_schema_json = {
            "type": "object",
            "properties": {
                "upscale_factor": {
                    "type": "number",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 8,
                    "x-lm-atelier-kind": "upscale",
                }
            },
        }
        original_schema = copy.deepcopy(old.input_schema_json)
        refreshed = DownloadManager._ensure_template_workflow(session, compiled, install)
        assert refreshed.id != old.id
        assert refreshed.version == old.version + 1
        assert old.input_schema_json == original_schema
        field = refreshed.input_schema_json["properties"]["upscale_factor"]
        if adjustable:
            assert field == {**factor, "x-lm-atelier-kind": "upscale"}
        else:
            assert field["readOnly"] is True
            assert "default" not in field
        assert DownloadManager._ensure_template_workflow(session, compiled, install) is refreshed
