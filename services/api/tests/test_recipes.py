from __future__ import annotations

import re
from pathlib import Path

from httpx2 import AsyncClient

from local_lm.catalog import HuggingFaceCatalog
from local_lm.downloads import DownloadManager
from local_lm.processes import ProcessSupervisor
from local_lm.recipes import get_reference_recipe, list_reference_recipes
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


def test_every_recipe_pins_a_checksum_for_every_file() -> None:
    """A recipe exists to promise specific bytes; an unpinned file cannot."""
    for recipe in list_reference_recipes():
        assert recipe.files
        assert all(file.sha256 for file in recipe.files), recipe.id


def _chat_recipe_catalog(recipe, monkeypatch, *, files=None):  # type: ignore[no-untyped-def]
    """Answer catalog lookups with exactly what the recipe pins, unless overridden."""

    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "model": {
                "remote_id": remote_id,
                "name": recipe.name,
                "compatibility": "supported",
                "formats": ["gguf"],
            },
            "revision": revision,
            "files": files
            if files is not None
            else [
                {"filename": file.path, "size": file.size_bytes or 1024, "sha256": file.sha256}
                for file in recipe.files
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    # Bounded header reads are an optimisation preflight treats as optional, and
    # the staged bytes are inspected for real before activation. Removing it keeps
    # the test off the network entirely.
    monkeypatch.delattr(HuggingFaceCatalog, "inspect_file_prefix", raising=False)


async def test_recipe_install_produces_a_plan_matching_its_pins(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """Recipes used to bypass the install plan entirely and could never be ready."""
    recipe = get_reference_recipe("qwen3-8b-q4-k-m")
    assert recipe
    _chat_recipe_catalog(recipe, monkeypatch)

    accepted = await client.post(f"/api/recipes/{recipe.id}/install")

    assert accepted.status_code == 202
    job = accepted.json()
    assert job["kind"] == "download"
    payload = job["payload_json"]
    # The download must be plan-driven; that is what enables staged inspection,
    # component manifests, the activation probe and the evidence write.
    assert payload["install_plan_id"]
    assert payload["remote_id"] == recipe.remote_id
    assert payload["revision"] == recipe.revision
    assert set(payload["allow_patterns"]) == {file.path for file in recipe.files}
    assert payload["expected_sha256"] == {
        file.path: file.sha256 for file in recipe.files if file.sha256
    }
    assert payload["recipe_id"] == recipe.id
    await client.post(f"/api/jobs/{job['id']}/cancel")


async def test_recipe_install_refuses_a_repository_that_drifted_from_its_pins(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """Preflight carries no hashes, so drift has to be caught before installing."""
    recipe = get_reference_recipe("qwen3-8b-q4-k-m")
    assert recipe
    _chat_recipe_catalog(
        recipe,
        monkeypatch,
        files=[
            {"filename": file.path, "size": file.size_bytes or 1024, "sha256": "f" * 64}
            for file in recipe.files
        ],
    )

    refused = await client.post(f"/api/recipes/{recipe.id}/install")

    assert refused.status_code == 422
    assert "different file contents" in refused.json()["detail"]


async def test_recipe_install_refuses_a_repository_missing_pinned_files(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    recipe = get_reference_recipe("qwen3-8b-q4-k-m")
    assert recipe
    _chat_recipe_catalog(
        recipe,
        monkeypatch,
        files=[{"filename": "something-else.gguf", "size": 1024, "sha256": "a" * 64}],
    )

    refused = await client.post(f"/api/recipes/{recipe.id}/install")

    # Preflight rejects this one before the pin check does, which is the better
    # error; what matters is that nothing installs.
    assert refused.status_code == 422
    assert "not present in this revision" in refused.json()["detail"]


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


async def test_recipe_install_reports_an_unreachable_catalog_as_unavailable(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """The shared operation raises a domain error; each route maps it itself."""

    async def unreachable(
        _catalog: HuggingFaceCatalog,
        _remote_id: str,
        _revision: str = "main",
        _requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        raise OSError("network is down")

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", unreachable)

    response = await client.post("/api/recipes/qwen3-8b-q4-k-m/install")

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]
