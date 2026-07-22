from __future__ import annotations

import re
from pathlib import Path

from httpx2 import AsyncClient

from local_lm.downloads import DownloadManager
from local_lm.processes import ProcessSupervisor
from local_lm.recipes import list_reference_recipes, recipe_download_request
from local_lm.settings_registry import (
    CHAT_SETTINGS,
    IMAGE_SETTINGS,
    VIDEO_SETTINGS,
    validate_settings,
)


def test_reference_recipes_are_immutable_safe_candidates() -> None:
    recipes = list_reference_recipes()
    assert {recipe.id for recipe in recipes} == {
        "qwen3-8b-q4-k-m",
        "flux1-schnell-fp8",
        "sd35-medium-fp8",
        "wan21-t2v-13b",
        "wan21-i2v-14b-480p-fp8",
    }
    blocked = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
    for recipe in recipes:
        assert re.fullmatch(r"[0-9a-f]{40}", recipe.revision)
        assert recipe.license_id in {"Apache-2.0", "stabilityai-ai-community"}
        assert recipe.status == "reference-candidate"
        assert recipe.certified is False
        assert recipe.files
        assert all(Path(file.path).suffix.lower() not in blocked for file in recipe.files)
        assert all(not Path(file.path).is_absolute() for file in recipe.files)


def test_recipe_downloads_pin_files_hashes_and_comfy_folders() -> None:
    for recipe in list_reference_recipes():
        request = recipe_download_request(recipe)
        assert request.revision == recipe.revision
        assert request.allow_patterns == [file.path for file in recipe.files]
        assert set(request.expected_sha256) == {file.path for file in recipe.files if file.sha256}
        if recipe.engine == "comfyui":
            assert request.comfy_paths


def test_recipe_defaults_match_the_public_setting_registry() -> None:
    fields = {"chat": CHAT_SETTINGS, "image": IMAGE_SETTINGS, "video": VIDEO_SETTINGS}
    for recipe in list_reference_recipes():
        assert (
            validate_settings(recipe.default_settings, fields[recipe.role])
            == recipe.default_settings
        )


def test_staging_activation_adds_files_to_an_existing_revision(tmp_path: Path) -> None:
    destination = tmp_path / "installed"
    destination.mkdir()
    (destination / "first.safetensors").write_bytes(b"first")
    staging = tmp_path / "staging"
    nested = staging / "split_files" / "vae"
    nested.mkdir(parents=True)
    (nested / "second.safetensors").write_bytes(b"second")

    DownloadManager._activate_staging(staging, destination)

    assert (destination / "first.safetensors").read_bytes() == b"first"
    assert (destination / "split_files" / "vae" / "second.safetensors").read_bytes() == b"second"
    assert not staging.exists()


def test_comfy_model_folder_validation_rejects_escapes() -> None:
    assert ProcessSupervisor._validated_comfy_paths(
        {"vae": "split_files/vae", "custom_nodes": ".", "checkpoints": "../outside"}
    ) == {"vae": "split_files/vae"}


async def test_reference_recipe_api_lists_pinned_metadata(client: AsyncClient) -> None:
    response = await client.get("/api/recipes")
    assert response.status_code == 200
    recipes = response.json()
    assert len(recipes) == 5
    assert all(len(recipe["revision"]) == 40 for recipe in recipes)
    assert all(recipe["certified"] is False for recipe in recipes)

    detail = await client.get("/api/recipes/qwen3-8b-q4-k-m")
    assert detail.status_code == 200
    assert detail.json()["files"][0]["path"] == "Qwen3-8B-Q4_K_M.gguf"

    missing = await client.get("/api/recipes/not-a-recipe")
    assert missing.status_code == 404
