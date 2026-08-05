"""A recipe records what produced a result, not what is current when it is saved."""

from __future__ import annotations

from typing import Any

from local_lm.edit_recipes import capture_recipe


def _provenance(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "workflow": {"definition_id": "def-1", "revision_id": "rev-1", "version": 3},
        "model": {"profile_id": "profile-1", "profile_name": "Edit"},
        "resolved_settings": {"denoise": 0.6, "steps": 24, "seed": 991},
    }
    payload.update(overrides)
    return payload


def test_a_recipe_keeps_the_workflow_and_profile_that_ran() -> None:
    capture = capture_recipe(_provenance())

    assert capture.workflow_revision_id == "rev-1"
    assert capture.model_profile_id == "profile-1"


def test_the_seed_is_not_part_of_the_recipe() -> None:
    """A seed reproduces one picture, not the reason anyone liked it."""
    capture = capture_recipe(_provenance())

    assert capture.settings == {"denoise": 0.6, "steps": 24}


def test_a_masked_run_records_that_it_expects_a_selection() -> None:
    capture = capture_recipe(
        _provenance(resolved_settings={"denoise": 0.6, "mask": {"artifact_id": "sha256:x"}})
    )

    assert capture.mask_mode == "selection"
    # The selection was drawn on one picture and means nothing on another.
    assert "mask" not in capture.settings


def test_an_inverted_mask_is_a_different_recipe() -> None:
    capture = capture_recipe(
        _provenance(resolved_settings={"mask": {"artifact_id": "sha256:x", "invert": True}})
    )

    assert capture.mask_mode == "inverse"


def test_a_whole_image_edit_expects_no_selection() -> None:
    assert capture_recipe(_provenance()).mask_mode == "none"


def test_provenance_that_names_no_workflow_yields_a_recipe_that_admits_it() -> None:
    """Rather than one that names today's, which is how a recipe starts lying."""
    capture = capture_recipe(_provenance(workflow=None, model={}))

    assert capture.workflow_revision_id is None
    assert capture.model_profile_id is None


def test_malformed_provenance_never_raises_into_the_save() -> None:
    capture = capture_recipe({"workflow": "not a mapping", "resolved_settings": 7})

    assert capture.workflow_revision_id is None
    assert capture.settings == {}
    assert capture.mask_mode == "none"


async def test_saving_from_a_run_reads_that_run_rather_than_the_request(client) -> None:
    """Through the route: the payload's settings are ignored when a run is named."""
    from local_lm.db import SessionLocal
    from local_lm.models import Run

    source = (
        await client.post(
            "/api/artifacts",
            files={"file": ("recipe-source.png", b"source-image", "image/png")},
        )
    ).json()
    chat = (await client.post("/api/chats", json={"title": "Recipe"})).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "make it a watercolor",
            "mode": "image",
            "input_artifact_ids": [source["id"]],
        },
    )
    assert accepted.status_code == 202, accepted.text
    run_id = accepted.json()["run"]["id"]

    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        run.provenance_json = {
            **run.provenance_json,
            "workflow": {"revision_id": "rev-from-run"},
            "model": {"profile_id": "profile-from-run"},
            "resolved_settings": {"denoise": 0.42, "seed": 5},
        }
        session.commit()

    saved = await client.post(
        "/api/edit-templates",
        json={
            "name": "Watercolor",
            "instruction": "make it a watercolor",
            "settings_json": {"denoise": 0.99},
            "from_run_id": run_id,
        },
    )

    assert saved.status_code == 201, saved.text
    body = saved.json()
    assert body["workflow_revision_id"] == "rev-from-run"
    assert body["model_profile_id"] == "profile-from-run"
    assert body["settings_json"] == {"denoise": 0.42}


async def test_saving_from_a_run_that_no_longer_exists_refuses(client) -> None:
    response = await client.post(
        "/api/edit-templates",
        json={"name": "Gone", "instruction": "anything", "from_run_id": "run_missing"},
    )

    assert response.status_code == 404
