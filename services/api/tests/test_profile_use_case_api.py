from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import ModelProfile, WorkflowProfileCompatibility
from local_lm.workflow_compatibility import ensure_legacy_profile_workflow


async def _derived_profile(client: AsyncClient) -> tuple[str, str]:
    response = await client.get("/api/profiles?role=chat")
    assert response.status_code == 200
    profile_id = response.json()[0]["id"]
    with SessionLocal() as session:
        profile = session.get(ModelProfile, profile_id)
        assert profile is not None
        profile.use_case = "coding; technical answers"
        profile.use_case_derived = True
        ensure_legacy_profile_workflow(session, profile)
        session.commit()
        compatibility = session.scalar(
            select(WorkflowProfileCompatibility).where(
                WorkflowProfileCompatibility.model_profile_id == profile_id
            )
        )
        assert compatibility is not None
        return profile_id, compatibility.workflow_family_id


async def test_profile_and_family_responses_expose_derived_provenance(client: AsyncClient) -> None:
    profile_id, family_id = await _derived_profile(client)
    profiles = (await client.get("/api/profiles")).json()
    profile = next(item for item in profiles if item["id"] == profile_id)
    assert profile.get("use_case_derived") is True
    family = await client.get(f"/api/workflow-families/{family_id}")
    assert family.status_code == 200
    assert family.json()["use_case"] == profile["use_case"]
    assert family.json().get("use_case_derived") is True


@pytest.mark.parametrize("text", ["My chosen use case", "", "coding; technical answers"])
async def test_explicit_profile_text_becomes_manual_and_survives_reload(
    client: AsyncClient, text: str
) -> None:
    profile_id, family_id = await _derived_profile(client)
    updated = await client.patch(f"/api/profiles/{profile_id}", json={"use_case": text})
    assert updated.status_code == 200
    assert updated.json()["use_case"] == text
    assert updated.json().get("use_case_derived") is False
    with SessionLocal() as session:
        profile = session.get(ModelProfile, profile_id)
        assert profile is not None
        assert profile.use_case == text
        assert getattr(profile, "use_case_derived", None) is False
    family = (await client.get(f"/api/workflow-families/{family_id}")).json()
    assert family["use_case"] == text and family.get("use_case_derived") is False


async def test_profile_name_and_settings_reset_preserve_derived_text(client: AsyncClient) -> None:
    profile_id, family_id = await _derived_profile(client)
    renamed = await client.patch(f"/api/profiles/{profile_id}", json={"name": "Renamed profile"})
    assert renamed.status_code == 200
    assert renamed.json().get("use_case_derived") is True
    reset = await client.post(f"/api/profiles/{profile_id}/reset")
    assert reset.status_code == 200
    assert reset.json()["use_case"] == "coding; technical answers"
    assert reset.json().get("use_case_derived") is True
    family = await client.patch(
        f"/api/workflow-families/{family_id}", json={"name": "Renamed family"}
    )
    assert family.status_code == 200
    assert family.json().get("use_case_derived") is True


@pytest.mark.parametrize("text", ["Chosen from the family editor", ""])
async def test_family_text_edits_clear_compatibility_profile_provenance(
    client: AsyncClient, text: str
) -> None:
    profile_id, family_id = await _derived_profile(client)
    family = await client.patch(f"/api/workflow-families/{family_id}", json={"use_case": text})
    assert family.status_code == 200
    assert family.json()["use_case"] == text
    assert family.json().get("use_case_derived") is False
    profiles = (await client.get("/api/profiles")).json()
    profile = next(item for item in profiles if item["id"] == profile_id)
    assert profile["use_case"] == text and profile.get("use_case_derived") is False


async def test_clone_and_portable_bundle_preserve_derived_provenance(client: AsyncClient) -> None:
    profile_id, _ = await _derived_profile(client)
    cloned = await client.post(f"/api/profiles/{profile_id}/clone", json={"name": "Derived copy"})
    assert cloned.status_code == 201
    assert cloned.json()["use_case"] == "coding; technical answers"
    assert cloned.json().get("use_case_derived") is True
    exported = await client.get(f"/api/profiles/{profile_id}/export")
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle.get("use_case_derived") is True
    bundle["name"] = "Imported derived profile"
    imported = await client.post("/api/profiles/import", json=bundle)
    assert imported.status_code == 201
    assert imported.json()["use_case"] == bundle["use_case"]
    assert imported.json().get("use_case_derived") is True


async def test_old_portable_bundle_text_is_manual(client: AsyncClient) -> None:
    profile_id, _ = await _derived_profile(client)
    bundle = (await client.get(f"/api/profiles/{profile_id}/export")).json()
    bundle.pop("use_case_derived", None)
    bundle["name"] = "Old portable profile"
    imported = await client.post("/api/profiles/import", json=bundle)
    assert imported.status_code == 201
    assert imported.json()["use_case"] == bundle["use_case"]
    assert imported.json().get("use_case_derived") is False
