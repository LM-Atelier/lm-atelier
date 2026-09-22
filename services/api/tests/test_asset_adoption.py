"""Registering a model file that is already where the runtime loads it."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
from httpx2 import AsyncClient
from sqlalchemy.orm import Session

from local_lm.asset_adoption import (
    AssetAdoptionError,
    measure_adoptable_file,
    read_safetensors_metadata,
    resolve_adoptable_path,
)
from local_lm.auxiliary_assets import _installed_lora_trigger_words
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import ModelAssetInstall, ModelInstall

PAYLOAD = b"neutral weights for a test, not a model" * 64


def _safetensors(path: Path, metadata: dict[str, str] | None = None) -> bytes:
    """A minimal safetensors file: a length-prefixed header, then bytes."""

    header: dict[str, object] = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode("utf-8")
    content = struct.pack("<Q", len(encoded)) + encoded + PAYLOAD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def _loras(settings: Settings) -> Path:
    assert settings.comfy_directory is not None
    folder = settings.comfy_directory / "models" / "loras"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@pytest.fixture
def runtime_settings(settings: Settings, tmp_path: Path) -> Settings:
    settings.comfy_directory = tmp_path / "comfy"
    return settings


def test_a_file_in_place_is_measured_rather_than_described(runtime_settings: Settings) -> None:
    folder = _loras(runtime_settings)
    content = _safetensors(folder / "slider.safetensors", {"modelspec.architecture": "krea2/lora"})

    measured = measure_adoptable_file([folder], "slider.safetensors")

    assert measured.sha256 == hashlib.sha256(content).hexdigest()
    assert measured.size_bytes == len(content)
    assert measured.comfy_name == "slider.safetensors"
    assert measured.declared_family == "krea2"


def test_a_file_that_names_no_trigger_word_says_so(runtime_settings: Settings) -> None:
    """A slider is driven by strength; an empty list is the honest answer."""

    folder = _loras(runtime_settings)
    _safetensors(folder / "slider.safetensors", {"modelspec.usage_hint": "no keyword needed"})

    measured = measure_adoptable_file([folder], "slider.safetensors")

    assert measured.trigger_words == []
    assert measured.metadata["modelspec.usage_hint"] == "no keyword needed"


def test_trigger_words_are_read_in_the_order_the_file_names_them(
    runtime_settings: Settings,
) -> None:
    folder = _loras(runtime_settings)
    _safetensors(folder / "styled.safetensors", {"modelspec.trigger_phrase": "one, two,  three"})

    assert measure_adoptable_file([folder], "styled.safetensors").trigger_words == [
        "one",
        "two",
        "three",
    ]


def test_a_training_tag_table_is_not_a_trigger_vocabulary(runtime_settings: Settings) -> None:
    """Trigger words reach the prompt, so a frequency table must not become one."""

    folder = _loras(runtime_settings)
    _safetensors(
        folder / "slider.safetensors",
        {"ss_tag_frequency": '{"img": {"a subject": 100, "another": 4}}'},
    )

    assert measure_adoptable_file([folder], "slider.safetensors").trigger_words == []


def test_a_declared_trigger_word_the_run_would_refuse_is_left_out(
    runtime_settings: Settings,
) -> None:
    """What adoption records has to be something a run can still load."""

    folder = _loras(runtime_settings)
    _safetensors(
        folder / "styled.safetensors",
        {"modelspec.trigger_phrase": "keep, " + "x" * 400 + ", keep, also"},
    )

    assert measure_adoptable_file([folder], "styled.safetensors").trigger_words == ["keep", "also"]


async def test_what_adoption_records_is_what_a_run_can_read(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    """The stack reader refuses malformed vocabularies; this one comes from it."""

    folder = _loras(runtime_settings)
    _safetensors(
        folder / "slider.safetensors",
        {"ss_tag_frequency": '{"img": {"a subject": 100}}', "modelspec.architecture": "krea2/lora"},
    )

    adopted = await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "slider.safetensors"}
    )

    assert adopted.status_code == 201, adopted.text
    with SessionLocal() as session:
        stored = session.get(ModelAssetInstall, adopted.json()["id"])
        assert stored is not None
        assert _installed_lora_trigger_words(stored.manifest_json["metadata"]) == []


def test_a_nested_name_is_kept_as_the_graph_will_pass_it(runtime_settings: Settings) -> None:
    folder = _loras(runtime_settings)
    _safetensors(folder / "packs" / "inner.safetensors", {})

    measured = measure_adoptable_file([folder], "packs/inner.safetensors")

    assert measured.comfy_name == "packs/inner.safetensors"
    assert measured.path.name == "inner.safetensors"


@pytest.mark.parametrize(
    "name",
    ["../escape.safetensors", "/absolute.safetensors", "packs/../../escape.safetensors", " "],
)
def test_a_name_that_leaves_the_folder_is_refused(runtime_settings: Settings, name: str) -> None:
    folder = _loras(runtime_settings)
    _safetensors(folder.parent / "escape.safetensors", {})

    with pytest.raises(AssetAdoptionError) as refusal:
        resolve_adoptable_path([folder], name)

    assert refusal.value.code == "asset-name-invalid"


def test_a_format_that_executes_on_load_is_refused(runtime_settings: Settings) -> None:
    folder = _loras(runtime_settings)
    (folder / "weights.ckpt").write_bytes(PAYLOAD)

    with pytest.raises(AssetAdoptionError) as refusal:
        resolve_adoptable_path([folder], "weights.ckpt")

    assert refusal.value.code == "asset-format-unsupported"


def test_a_file_that_is_not_there_says_which_folder(runtime_settings: Settings) -> None:
    folder = _loras(runtime_settings)

    with pytest.raises(AssetAdoptionError) as refusal:
        resolve_adoptable_path([folder], "absent.safetensors")

    assert refusal.value.code == "asset-file-missing"
    assert "folder" in refusal.value.detail


def test_an_oversized_header_is_not_read_into_memory(runtime_settings: Settings) -> None:
    """The length prefix is bounded before it is believed."""

    folder = _loras(runtime_settings)
    path = folder / "huge.safetensors"
    path.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")

    assert read_safetensors_metadata(path) == {}


def test_a_file_that_will_not_describe_itself_is_still_adoptable(
    runtime_settings: Settings,
) -> None:
    folder = _loras(runtime_settings)
    (folder / "bare.safetensors").write_bytes(b"not a header at all")

    measured = measure_adoptable_file([folder], "bare.safetensors")

    assert measured.metadata == {}
    assert measured.declared_family is None
    assert len(measured.sha256) == 64


async def test_adopting_a_lora_records_what_it_measured(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    folder = _loras(runtime_settings)
    content = _safetensors(
        folder / "slider.safetensors",
        {"modelspec.architecture": "krea2/lora", "modelspec.usage_hint": "no keyword needed"},
    )

    adopted = await client.post(
        "/api/model-assets",
        json={"kind": "lora", "comfy_name": "slider.safetensors", "family": "krea2"},
    )

    assert adopted.status_code == 201, adopted.text
    body = adopted.json()
    assert body["kind"] == "lora"
    assert body["family"] == "krea2"
    with SessionLocal() as session:
        stored = session.get(ModelAssetInstall, body["id"])
        assert stored is not None
        assert stored.active is True and stored.verified_at is not None
        assert stored.manifest_json["sha256"] == hashlib.sha256(content).hexdigest()
        assert stored.manifest_json["comfy_name"] == "slider.safetensors"
        assert stored.manifest_json["metadata"]["trigger_words"] == []
        assert stored.manifest_json["adopted"] is True


async def test_an_adopted_lora_is_offered_to_the_browser(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    """The list the LoRA control reads is the one that must show it."""

    folder = _loras(runtime_settings)
    _safetensors(folder / "slider.safetensors", {})

    await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "slider.safetensors"}
    )
    listed = (await client.get("/api/model-assets", params={"kind": "lora"})).json()

    assert [asset["name"] for asset in listed] == ["slider"]
    assert listed[0]["verified_at"] is not None


async def test_the_same_bytes_are_not_registered_twice(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    folder = _loras(runtime_settings)
    _safetensors(folder / "slider.safetensors", {})
    (folder / "copy.safetensors").write_bytes((folder / "slider.safetensors").read_bytes())

    first = await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "slider.safetensors"}
    )
    second = await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "copy.safetensors"}
    )

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["code"] == "asset-already-registered"


async def test_a_use_case_is_refused_where_nothing_would_read_it(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    """The two routes have to answer this the same way, and one did not.

    Editing an asset refuses a use case on anything but a LoRA, because
    automatic selection is what reads it and selects on kind. Registering one
    stored it for any kind, so a use case given at the door could never be
    corrected or cleared afterwards.
    """

    assert runtime_settings.comfy_directory is not None
    folder = runtime_settings.comfy_directory / "models" / "vae"
    folder.mkdir(parents=True, exist_ok=True)
    _safetensors(folder / "encoder.safetensors", {})

    refused = await client.post(
        "/api/model-assets",
        json={"kind": "vae", "comfy_name": "encoder.safetensors", "use_case": "product photos"},
    )

    assert refused.status_code == 422, refused.text
    assert refused.json()["code"] == "automatic-selection-lora-only"
    assert (await client.get("/api/model-assets")).json() == []


async def test_a_use_case_a_lora_is_registered_with_can_still_be_changed(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    """What one route accepts the other has to accept, or it cannot be undone."""

    folder = _loras(runtime_settings)
    _safetensors(folder / "slider.safetensors", {})

    adopted = await client.post(
        "/api/model-assets",
        json={
            "kind": "lora",
            "comfy_name": "slider.safetensors",
            "use_case": "watercolour landscapes",
        },
    )
    assert adopted.status_code == 201, adopted.text
    asset_id = adopted.json()["id"]

    changed = await client.patch(
        f"/api/model-assets/{asset_id}", json={"use_case": "product photography"}
    )

    assert changed.status_code == 200, changed.text
    assert changed.json()["use_case"] == "product photography"


async def test_an_empty_use_case_is_not_metadata_to_refuse(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    """The refusal is about a value that would be kept, not about the field."""

    assert runtime_settings.comfy_directory is not None
    folder = runtime_settings.comfy_directory / "models" / "vae"
    folder.mkdir(parents=True, exist_ok=True)
    _safetensors(folder / "encoder.safetensors", {})

    adopted = await client.post(
        "/api/model-assets",
        json={"kind": "vae", "comfy_name": "encoder.safetensors", "use_case": "   "},
    )

    assert adopted.status_code == 201, adopted.text
    assert adopted.json()["use_case"] == ""


async def test_a_kind_outside_the_vocabulary_never_reaches_the_handler(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    """The declared vocabulary refuses it before any file is looked for."""

    refused = await client.post(
        "/api/model-assets", json={"kind": "sandwich", "comfy_name": "slider.safetensors"}
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "request-validation-invalid"


async def test_a_kind_the_runtime_has_no_folder_for_is_refused(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    """In the vocabulary, but nothing in the runtime loads it from a folder."""

    refused = await client.post(
        "/api/model-assets", json={"kind": "gguf_model", "comfy_name": "slider.safetensors"}
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "asset-kind-unsupported"


async def test_a_missing_file_answers_not_found(
    client: AsyncClient, runtime_settings: Settings
) -> None:
    _loras(runtime_settings)

    refused = await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "absent.safetensors"}
    )

    assert refused.status_code == 404
    assert refused.json()["code"] == "asset-file-missing"


def _declared_folder(session: Session, tmp_path: Path, name: str) -> Path:
    """A folder an installed model tells the runtime to read LoRAs from."""

    base = tmp_path / name
    (base / "loras").mkdir(parents=True, exist_ok=True)
    session.add(
        ModelInstall(
            name=f"{name} assets",
            role="image",
            engine="comfyui",
            local_path=str(base),
            manifest_json={"comfy_paths": {"loras": "loras"}},
            active=True,
        )
    )
    session.flush()
    return base / "loras"


async def test_a_folder_an_installed_model_declares_is_searched(
    client: AsyncClient, runtime_settings: Settings, tmp_path: Path
) -> None:
    """The runtime reads several folders, and its own is only one of them."""

    _loras(runtime_settings)
    with SessionLocal() as session:
        declared = _declared_folder(session, tmp_path, "carried-over")
        session.commit()
    content = _safetensors(
        declared / "slider.safetensors", {"modelspec.architecture": "krea2/lora"}
    )

    adopted = await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "slider.safetensors"}
    )

    assert adopted.status_code == 201, adopted.text
    with SessionLocal() as session:
        stored = session.get(ModelAssetInstall, adopted.json()["id"])
        assert stored is not None
        assert stored.manifest_json["sha256"] == hashlib.sha256(content).hexdigest()
        assert Path(stored.local_path) == declared.resolve()


async def test_a_name_two_runtime_folders_answer_is_refused(
    client: AsyncClient, runtime_settings: Settings, tmp_path: Path
) -> None:
    """Which one the runtime loads is its choice, so neither digest is safe."""

    own = _loras(runtime_settings)
    with SessionLocal() as session:
        declared = _declared_folder(session, tmp_path, "carried-over")
        session.commit()
    _safetensors(own / "slider.safetensors", {"modelspec.usage_hint": "first"})
    _safetensors(declared / "slider.safetensors", {"modelspec.usage_hint": "second"})

    refused = await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "slider.safetensors"}
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "asset-name-ambiguous"


async def test_a_folder_the_runtime_never_reads_is_not_searched(
    client: AsyncClient, runtime_settings: Settings, tmp_path: Path
) -> None:
    """Adoption gives the runtime no reach it did not already have."""

    _loras(runtime_settings)
    elsewhere = tmp_path / "somewhere-else"
    _safetensors(elsewhere / "slider.safetensors", {})

    refused = await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "slider.safetensors"}
    )

    assert refused.status_code == 404
    assert refused.json()["code"] == "asset-file-missing"


async def test_a_registered_asset_makes_its_own_folder_searchable(
    client: AsyncClient, runtime_settings: Settings, tmp_path: Path
) -> None:
    """An asset's folder is named to the runtime, so a sibling is adoptable."""

    _loras(runtime_settings)
    beside = tmp_path / "already-registered"
    beside.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as session:
        session.add(
            ModelAssetInstall(
                name="Earlier",
                kind="lora",
                local_path=str(beside),
                size_bytes=1024,
                manifest_json={"sha256": "a" * 64, "comfy_name": "earlier.safetensors"},
                active=True,
                verified_at=utcnow(),
            )
        )
        session.commit()
    _safetensors(beside / "sibling.safetensors", {})

    adopted = await client.post(
        "/api/model-assets", json={"kind": "lora", "comfy_name": "sibling.safetensors"}
    )

    assert adopted.status_code == 201, adopted.text
