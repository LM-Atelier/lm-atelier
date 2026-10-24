"""A checkpoint kept in a model library outside the application, run with its family.

A model library shared between tools holds its files in the runtime's own
folder layout, and importing it was meant to make them available. The import
recorded no folders, so the runtime was never told to load from the library,
nothing inside it could be adopted, and a checkpoint placed there stayed
invisible. A registered checkpoint was also never asked which family it was,
so a workflow built on one could neither refuse a LoRA made for another family
nor choose a LoRA automatically. These cases hold each step: the import names
the library's runtime folders to the runtime, adoption still reaches nothing
until it does, and the adopted checkpoint answers for its workflow's family.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy.orm import Session

from local_lm.auxiliary_assets import (
    checkpoint_lora_extension,
    resolve_lora_stack,
    select_automatic_lora_stack,
    workflow_model_family,
)
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ModelAssetInstall, ModelInstall, WorkflowDefinition, WorkflowRevision
from local_lm.processes import ProcessSupervisor

pytestmark = pytest.mark.asyncio

CHECKPOINT = "atelier-base.safetensors"
PAYLOAD = b"neutral weights for a test, not a model" * 64


def _safetensors(path: Path) -> None:
    """A minimal safetensors file a loader can read: one tensor covering the bytes after it."""

    header = {
        "weight": {"dtype": "F32", "shape": [len(PAYLOAD) // 4], "data_offsets": [0, len(PAYLOAD)]}
    }
    encoded = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + PAYLOAD)


def _link_dir(link: Path, target: Path) -> bool:
    """Point `link` at `target`: a junction on Windows, a symbolic link elsewhere."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


async def _import(client: AsyncClient, path: Path, *, engine: str = "comfyui") -> dict[str, Any]:
    response = await client.post(
        "/api/models/import",
        json={"name": "Model library", "role": "image", "engine": engine, "local_path": str(path)},
    )
    assert response.status_code == 201, response.text
    with SessionLocal() as session:
        install = session.get(ModelInstall, response.json()["id"])
        assert install is not None
        return dict(install.manifest_json)


def _served(settings: Settings) -> dict[str, dict[str, str]]:
    """What the runtime is told to load from, by base folder."""

    settings.prepare()
    destination = ProcessSupervisor(settings)._write_comfy_model_paths()
    published = json.loads(destination.read_text(encoding="utf-8"))
    return {
        str(Path(entry["base_path"])): {
            folder: relative for folder, relative in entry.items() if folder != "base_path"
        }
        for entry in published.values()
    }


def _graph(*, diffusion_model: bool = False, checkpoints: int = 1) -> dict[str, Any]:
    graph: dict[str, Any] = {
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
    }
    for index in range(checkpoints):
        node = "1" if index == 0 else f"1{index}"
        graph[node] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}}
    if diffusion_model:
        graph["4"] = {"class_type": "UNETLoader", "inputs": {"unet_name": CHECKPOINT}}
    return graph


def _hand_built(session: Session, graph: dict[str, Any]) -> WorkflowRevision:
    """A workflow as a person builds or imports it: where LoRAs go, and no model."""

    extension = checkpoint_lora_extension(_graph())
    assert extension
    definition = WorkflowDefinition(name="Hand built", operation="text_to_image")
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        api_graph_json=graph,
        input_schema_json={
            "type": "object",
            "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
        },
        dependencies_json={"extensions": {"lora": extension}},
        trusted=True,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    return revision


def _registered(
    session: Session,
    name: str,
    digest: str,
    *,
    kind: str = "checkpoint",
    comfy_name: str = CHECKPOINT,
    family: str | None = "sdxl",
    active: bool = True,
    verified: bool = True,
) -> ModelAssetInstall:
    asset = ModelAssetInstall(
        name=name,
        kind=kind,
        family=family,
        local_path=f"C:/library/{name}",
        size_bytes=4096,
        manifest_json={"adopted": True, "sha256": digest * 64, "comfy_name": comfy_name},
        active=active,
        verified_at=utcnow() if verified else None,
    )
    session.add(asset)
    session.flush()
    return asset


def _installed(
    session: Session, name: str, *, family: str | None, file: str = CHECKPOINT
) -> ModelInstall:
    manifest: dict[str, Any] = {"files": [file]}
    if family is not None:
        manifest["family"] = family
    install = ModelInstall(
        name=name,
        role="image",
        engine="comfyui",
        local_path=f"C:/managed/{name}",
        manifest_json=manifest,
        active=True,
    )
    session.add(install)
    session.flush()
    return install


def _lora(
    session: Session, name: str, family: str, digest: str, *, automatic: bool = False
) -> ModelAssetInstall:
    asset = ModelAssetInstall(
        name=name,
        kind="lora",
        family=family,
        local_path=f"C:/library/{name}",
        size_bytes=1024,
        manifest_json={
            "sha256": digest * 64,
            "comfy_name": f"{name}.safetensors",
            "metadata": {"network_type": "networks.lora"},
        },
        active=True,
        auto_apply=automatic,
        use_case="portrait" if automatic else "",
        verified_at=utcnow(),
    )
    session.add(asset)
    session.flush()
    return asset


def _stack(asset: ModelAssetInstall) -> list[object]:
    return [{"asset_id": asset.id, "model_strength": 1.0, "clip_strength": 1.0}]


async def test_an_imported_runtime_folder_is_named_to_the_runtime(
    client: AsyncClient, settings: Settings, tmp_path: Path
) -> None:
    folder = tmp_path / "library" / "checkpoints"
    _safetensors(folder / CHECKPOINT)

    manifest = await _import(client, folder)

    assert manifest["comfy_paths"] == {"checkpoints": "."}
    assert _served(settings)[str(folder.resolve())] == {"checkpoints": "."}


async def test_an_imported_library_exposes_only_the_runtime_folders_directly_inside_it(
    client: AsyncClient, settings: Settings, tmp_path: Path
) -> None:
    library = tmp_path / "library"
    _safetensors(library / "checkpoints" / CHECKPOINT)
    _safetensors(library / "loras" / "detail.safetensors")
    (library / "notes").mkdir()
    (library / "vae").write_bytes(b"a file named like a folder is not one")
    (library / "nested" / "text_encoders").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # The far side of a link is not what was named, so it is not exposed.
    assert _link_dir(library / "upscale_models", elsewhere)

    manifest = await _import(client, library)

    expected = {"checkpoints": "checkpoints", "loras": "loras"}
    assert manifest["comfy_paths"] == expected
    assert _served(settings)[str(library.resolve())] == expected


@pytest.mark.parametrize("case", ["another engine", "a single file"])
async def test_an_import_the_runtime_does_not_read_records_no_runtime_folder(
    client: AsyncClient, settings: Settings, tmp_path: Path, case: str
) -> None:
    folder = tmp_path / "library" / "checkpoints"
    _safetensors(folder / CHECKPOINT)

    if case == "another engine":
        manifest = await _import(client, folder, engine="mock")
    else:
        manifest = await _import(client, folder / CHECKPOINT)

    assert "comfy_paths" not in manifest
    assert str(folder.resolve()) not in _served(settings)


async def test_a_checkpoint_adopted_from_an_imported_library_answers_for_its_workflow(
    client: AsyncClient, settings: Settings, tmp_path: Path
) -> None:
    folder = tmp_path / "library" / "checkpoints"
    _safetensors(folder / CHECKPOINT)
    adopt = {"kind": "checkpoint", "comfy_name": CHECKPOINT, "family": "sdxl"}

    # Adoption reaches only what the runtime already reads, and nothing reads
    # the library until it is imported.
    before = await client.post("/api/model-assets", json=adopt)
    assert before.status_code == 409, before.text
    assert before.json()["code"] == "asset-runtime-unconfigured"

    await _import(client, folder)
    adopted = await client.post("/api/model-assets", json=adopt)

    assert adopted.status_code == 201, adopted.text
    body = adopted.json()
    assert (body["kind"], body["family"]) == ("checkpoint", "sdxl")
    assert _served(settings)[str(folder.resolve())] == {"checkpoints": "."}
    with SessionLocal() as session:
        asset = session.get(ModelAssetInstall, body["id"])
        assert asset is not None
        assert Path(asset.local_path) == folder.resolve()
        revision = _hand_built(session, _graph())
        foreign = _lora(session, "Foreign", "krea2", "a")
        matching = _lora(session, "Matching", "SDXL", "b")
        automatic = _lora(session, "Portrait", "sdxl", "c", automatic=True)
        _lora(session, "Elsewhere", "krea2", "d", automatic=True)

        assert workflow_model_family(session, revision) == "sdxl"
        with pytest.raises(ValueError, match="targets krea2, which is incompatible"):
            resolve_lora_stack(session, revision, _stack(foreign))
        accepted = resolve_lora_stack(session, revision, _stack(matching))
        selected = select_automatic_lora_stack(session, revision, "a detailed portrait")

    assert [item["asset_id"] for item in accepted.settings] == [matching.id]
    assert [item["asset_id"] for item in selected.settings] == [automatic.id]
    assert selected.provenance.get("skipped_reason") is None


async def test_a_graph_node_whose_class_is_not_text_is_not_a_loader(
    client: AsyncClient,
) -> None:
    del client
    graph = _graph()
    graph["9"] = {"class_type": ["CheckpointLoaderSimple"], "inputs": {}}
    with SessionLocal() as session:
        _registered(session, "Atelier Base", "1")
        revision = _hand_built(session, graph)

        assert workflow_model_family(session, revision) == "sdxl"


async def test_an_installed_model_holding_the_checkpoint_answers_before_a_registration(
    client: AsyncClient,
) -> None:
    del client
    with SessionLocal() as session:
        _installed(session, "Installed base", family="z-image", file=f"checkpoints/{CHECKPOINT}")
        _registered(session, "Atelier Base", "2")
        revision = _hand_built(session, _graph())

        assert workflow_model_family(session, revision) == "z-image"


@pytest.mark.parametrize(
    "case",
    [
        "no registered checkpoint answers to the name",
        "two registered checkpoints answer to the name",
        "only a diffusion model answers to the name",
        "the graph also has a diffusion-model loader",
        "the graph has two checkpoint loaders",
        "the checkpoint recorded no family",
        "the checkpoint is turned off",
        "the checkpoint was never verified",
        "an installed model without a family also holds the file",
        "two installed models also hold the file",
    ],
)
async def test_anything_short_of_one_exact_registered_checkpoint_leaves_the_family_unknown(
    client: AsyncClient, case: str
) -> None:
    del client
    graph = _graph()
    with SessionLocal() as session:
        if case == "no registered checkpoint answers to the name":
            _registered(session, "Other", "3", comfy_name="other.safetensors")
        elif case == "two registered checkpoints answer to the name":
            _registered(session, "First", "4")
            _registered(session, "Second", "5", family="krea2")
        elif case == "only a diffusion model answers to the name":
            _registered(session, "Atelier DiT", "6", kind="diffusion_model")
        elif case == "the graph also has a diffusion-model loader":
            _registered(session, "Atelier Base", "7")
            graph = _graph(diffusion_model=True)
        elif case == "the graph has two checkpoint loaders":
            _registered(session, "Atelier Base", "8")
            graph = _graph(checkpoints=2)
        elif case == "the checkpoint recorded no family":
            _registered(session, "Atelier Base", "9", family=None)
        elif case == "the checkpoint is turned off":
            _registered(session, "Atelier Base", "a", active=False)
        elif case == "the checkpoint was never verified":
            _registered(session, "Atelier Base", "b", verified=False)
        elif case == "an installed model without a family also holds the file":
            _registered(session, "Atelier Base", "c")
            _installed(session, "Unlabelled base", family=None)
        else:
            # Which of them the runtime loads is its own precedence to decide,
            # so a registration elsewhere does not settle the question.
            _registered(session, "Atelier Base", "d")
            _installed(session, "First base", family="sdxl")
            _installed(session, "Second base", family="sdxl")
        session.flush()
        revision = _hand_built(session, graph)
        foreign = _lora(session, "Foreign", "krea2", "f")

        assert workflow_model_family(session, revision) is None
        # Unknown is what it was before: the guard does not fire.
        accepted = resolve_lora_stack(session, revision, _stack(foreign))

    assert [item["asset_id"] for item in accepted.settings] == [foreign.id]
