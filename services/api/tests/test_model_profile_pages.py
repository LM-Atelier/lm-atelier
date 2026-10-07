from __future__ import annotations

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import ModelInstall, ModelProfile


def _profiles() -> None:
    with SessionLocal() as session:
        for install_id in ("page-install-a", "page-install-b"):
            session.add(
                ModelInstall(
                    id=install_id,
                    name="Profile paging fixture",
                    role="image",
                    engine="mock",
                    local_path="C:/neutral-fixture/profile-pages",
                )
            )
        session.flush()
        for index, install_id in enumerate(
            ("page-install-a", "page-install-b", "page-install-a", "page-install-b")
        ):
            session.add(
                ModelProfile(
                    id=f"page-profile-{index}",
                    name="Profile paging fixture",
                    role="image",
                    engine="mock",
                    model_install_id=install_id,
                )
            )
        session.add(
            ModelProfile(
                id="page-profile-engine",
                name="Profile paging fixture",
                role="image",
                engine="comfyui",
            )
        )
        session.commit()


@pytest.mark.parametrize("offset, expected", [(0, "page-profile-1"), (1, "page-profile-3")])
async def test_profile_install_filter_precedes_paging(
    client: AsyncClient, offset: int, expected: str
) -> None:
    _profiles()
    response = await client.get(
        "/api/profiles",
        params={
            "install_id": "page-install-b",
            "role": "image",
            "engine": "mock",
            "search": "Profile paging fixture",
            "limit": 1,
            "offset": offset,
        },
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [expected]


async def test_profile_engine_filter_precedes_paging(client: AsyncClient) -> None:
    _profiles()
    response = await client.get(
        "/api/profiles",
        params={"engine": "comfyui", "search": "Profile paging fixture", "limit": 1},
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["page-profile-engine"]


async def test_profile_install_filter_accepts_a_bounded_identity_batch(client: AsyncClient) -> None:
    _profiles()
    response = await client.get(
        "/api/profiles",
        params=[
            ("install_id", "page-install-b"),
            ("install_id", "page-install-a"),
            ("install_id", "page-install-b"),
            ("search", "Profile paging fixture"),
            ("limit", "200"),
        ],
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [f"page-profile-{index}" for index in range(4)]


async def test_profile_install_filter_refuses_an_oversized_identity_batch(
    client: AsyncClient,
) -> None:
    response = await client.get(
        "/api/profiles", params=[("install_id", f"install-{index}") for index in range(201)]
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, value", [("active", False), ("role", "video"), ("engine", "comfyui")]
)
async def test_profile_install_filter_keeps_incompatible_links_hidden(
    client: AsyncClient, field: str, value: str | bool
) -> None:
    _profiles()
    with SessionLocal() as session:
        install = session.get(ModelInstall, "page-install-b")
        assert install is not None
        setattr(install, field, value)
        session.commit()
    response = await client.get(
        "/api/profiles", params={"install_id": "page-install-b", "limit": 1}
    )
    assert response.status_code == 200
    assert response.json() == []
