"""A media turn records the model it ran on, whole, beside its settings."""

from __future__ import annotations

from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import ModelInstall, ModelProfile, ModelSource, Run


async def test_a_picture_turn_records_the_installed_model_and_where_it_came_from(
    app: FastAPI, client: AsyncClient
) -> None:
    with SessionLocal() as session:
        source = ModelSource(
            provider="huggingface",
            remote_id="example/neutral-picture-model",
            revision="0123abcd",
            metadata_json={"license": "example"},
        )
        session.add(source)
        session.flush()
        install = ModelInstall(
            source_id=source.id,
            name="Neutral picture model",
            role="image",
            engine="mock",
            local_path="models/neutral-picture-model",
            size_bytes=2048,
            manifest_json={"files": ["weights.bin"]},
        )
        session.add(install)
        session.flush()
        profile = ModelProfile(
            name="Neutral picture profile",
            role="image",
            engine="mock",
            use_case="Plain product shots",
            model_install_id=install.id,
        )
        session.add(profile)
        session.commit()
        expected = {
            "profile_id": profile.id,
            "profile_name": "Neutral picture profile",
            "profile_use_case": "Plain product shots",
            "install_id": install.id,
            "engine": "mock",
            "local_path": "models/neutral-picture-model",
            "size_bytes": 2048,
            "manifest": {"files": ["weights.bin"]},
            "source": {
                "provider": "huggingface",
                "remote_id": "example/neutral-picture-model",
                "revision": "0123abcd",
                "metadata": {"license": "example"},
            },
        }
        profile_id = profile.id
    chat = await client.post("/api/chats", json={"title": "Model record"})
    assert chat.status_code == 201, chat.text

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat.json()['id']}/turns",
            json={"text": "A plain gray square", "mode": "image", "profile_id": profile_id},
        )
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            run = session.get(Run, response.json()["run"]["id"])
            assert run is not None
            assert run.profile_id == profile_id
            assert run.provenance_json["model"] == expected
