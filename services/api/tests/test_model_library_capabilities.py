from __future__ import annotations

from datetime import UTC, datetime

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event

from local_lm.adapters.contracts import ADAPTER_CONTRACT_VERSION
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.hardware import hardware_capability_class
from local_lm.model_planner import ACTIVATION_PROBE_VERSION, LAUNCH_CONTRACT_VERSION
from local_lm.models import ModelCapabilityEvidence, ModelInstall, ModelProfile


def _seed(settings: Settings, extra: int = 0) -> None:
    hardware = hardware_capability_class(settings)
    with SessionLocal() as session:
        for index in range(11 + extra):
            identity = f"capability-{index:04}"
            row = ModelInstall(
                id=identity,
                name=f"Straße %_ {index:04}" if index in {1, 3} else f"Capability {index:04}",
                role="image" if index == 9 else "chat",
                engine="mock",
                local_path=f"C:/neutral-fixture/{identity}",
                active=index != 6,
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
                manifest_json={
                    "expected_sha256": {"neutral.gguf": "a" * 64},
                    "input_modalities": ["text", "image"],
                },
            )
            session.add(row)
            session.flush()
            if index != 4:
                session.add(
                    ModelProfile(
                        id=f"profile-{identity}",
                        name=identity,
                        model_install_id=identity,
                        role="image" if index == 7 else row.role,
                        engine="other" if index == 8 else "mock",
                    )
                )
            if index != 0:
                session.add(
                    ModelCapabilityEvidence(
                        id=f"proof-{identity}",
                        model_install_id=identity,
                        evidence_key=identity,
                        component_hashes_json={"neutral.gguf": ("b" if index == 5 else "a") * 64},
                        runtime_build="neutral-runtime",
                        adapter_contract_version=ADAPTER_CONTRACT_VERSION,
                        launch_contract_version=LAUNCH_CONTRACT_VERSION,
                        workflow_contract_version=None,
                        hardware_class=hardware,
                        probe_version=ACTIVATION_PROBE_VERSION,
                        details_json={}
                        if index == 10
                        else {"input_modalities": ["text"] if index == 2 else ["text", "image"]},
                    )
                )
        session.commit()


@pytest.mark.parametrize(
    "capability, offset, expected",
    [
        ("vision", 0, "capability-0001"),
        ("vision", 1, "capability-0003"),
        ("text", 0, "capability-0002"),
        ("text", 1, "capability-0010"),
    ],
)
async def test_capability_filters_apply_before_model_pages(
    client: AsyncClient, settings: Settings, capability: str, offset: int, expected: str
) -> None:
    _seed(settings)
    response = await client.get(
        "/api/models", params={"chat_capability": capability, "limit": 1, "offset": offset}
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [expected]
    assert response.json()[0]["readiness"] == "ready"
    assert response.json()[0]["capability_evidence"]["model_install_id"] == expected


@pytest.mark.parametrize("search", ["STRASSE", "%_"])
async def test_capability_pages_combine_literal_search_before_offset(
    client: AsyncClient, settings: Settings, search: str
) -> None:
    _seed(settings)
    response = await client.get(
        "/api/models",
        params={"chat_capability": "vision", "search": search, "offset": 1, "limit": 1},
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["capability-0003"]


@pytest.mark.parametrize("index", [0, 4, 5, 6, 7, 8, 9])
async def test_capability_pages_require_current_proof_and_an_eligible_profile(
    client: AsyncClient, settings: Settings, index: int
) -> None:
    _seed(settings)
    response = await client.get(
        "/api/models",
        params={"chat_capability": "vision", "model_id": f"capability-{index:04}", "limit": 1},
    )
    assert response.status_code == 200
    assert response.json() == []


async def test_model_role_filter_precedes_paging(client: AsyncClient, settings: Settings) -> None:
    _seed(settings)
    response = await client.get("/api/models", params={"role": "image", "limit": 1})
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["capability-0009"]
    conflict = await client.get(
        "/api/models", params={"role": "image", "chat_capability": "vision", "limit": 1}
    )
    assert conflict.status_code == 200
    assert conflict.json() == []


async def test_model_capability_filter_rejects_unknown_values(client: AsyncClient) -> None:
    response = await client.get("/api/models", params={"chat_capability": "unknown"})
    assert response.status_code == 422


async def test_capability_page_stops_hydration_after_a_bounded_batch(
    client: AsyncClient, settings: Settings
) -> None:
    _seed(settings, extra=440)
    loaded: list[str] = []

    def track(row: ModelInstall, _context: object) -> None:
        loaded.append(row.id)

    event.listen(ModelInstall, "load", track)
    try:
        response = await client.get("/api/models", params={"chat_capability": "vision", "limit": 1})
    finally:
        event.remove(ModelInstall, "load", track)
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["capability-0001"]
    assert 1 <= len(loaded) <= 200


@pytest.mark.parametrize("offset, expected", [(0, "0001"), (1, "0003")])
async def test_profile_vision_filter_precedes_paging(
    client: AsyncClient, settings: Settings, offset: int, expected: str
) -> None:
    _seed(settings)
    response = await client.get(
        "/api/profiles",
        params={
            "role": "chat",
            "input_modality": "image",
            "search": "capability-",
            "limit": 1,
            "offset": offset,
        },
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [f"profile-capability-{expected}"]
    assert "image" in response.json()[0]["input_modalities"]


@pytest.mark.parametrize("index", [0, 2, 5, 6, 7, 8, 9, 10])
async def test_profile_vision_filter_keeps_only_current_eligible_evidence(
    client: AsyncClient, settings: Settings, index: int
) -> None:
    _seed(settings)
    response = await client.get(
        "/api/profiles",
        params={
            "role": "chat",
            "input_modality": "image",
            "profile_id": f"profile-capability-{index:04}",
            "limit": 1,
        },
    )
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("search", ["STRASSE", "%_"])
async def test_profile_vision_filter_combines_literal_search_and_offset(
    client: AsyncClient, settings: Settings, search: str
) -> None:
    _seed(settings)
    with SessionLocal() as session:
        for index in (1, 3):
            profile = session.get(ModelProfile, f"profile-capability-{index:04}")
            assert profile is not None
            profile.name = f"Straße %_ {index}"
        session.commit()
    response = await client.get(
        "/api/profiles",
        params={"input_modality": "image", "search": search, "limit": 1, "offset": 1},
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["profile-capability-0003"]


async def test_profile_vision_filter_hydrates_only_a_bounded_candidate_batch(
    client: AsyncClient, settings: Settings
) -> None:
    _seed(settings, extra=440)
    loaded: list[str] = []

    def track(row: ModelProfile, _context: object) -> None:
        loaded.append(row.id)

    event.listen(ModelProfile, "load", track)
    try:
        response = await client.get(
            "/api/profiles",
            params={"role": "chat", "input_modality": "image", "search": "capability-", "limit": 1},
        )
    finally:
        event.remove(ModelProfile, "load", track)
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["profile-capability-0001"]
    assert 1 <= len(loaded) <= 200


async def test_profile_input_filter_refuses_unknown_values(client: AsyncClient) -> None:
    response = await client.get("/api/profiles", params={"input_modality": "unknown"})
    assert response.status_code == 422
