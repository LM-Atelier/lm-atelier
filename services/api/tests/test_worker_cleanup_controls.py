from __future__ import annotations

from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.main import Services
from local_lm.models import ModelAssetInstall, ModelInstall, ModelProfile
from local_lm.schemas import WorkerStatus


def _report_cleanup(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    name: Literal["chat", "media"],
    profile_id: str | None = None,
) -> Services:
    services = cast(Services, app.state.services)
    statuses = services.processes.statuses()
    status = WorkerStatus.model_construct(
        name=name, state="stopping", managed=True, running=False, profile_id=profile_id
    )
    monkeypatch.setattr(
        services.processes,
        "statuses",
        lambda: [status if current.name == name else current for current in statuses],
    )
    return services


@pytest.mark.parametrize("worker", ["chat", "media"])
@pytest.mark.parametrize("target", ["models", "profiles"])
async def test_model_and_profile_deletion_waits_for_worker_cleanup(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    worker: Literal["chat", "media"],
    target: str,
) -> None:
    model_root = settings.model_dir / "cleanup-model"
    model_root.mkdir(parents=True)
    marker = model_root / "model.gguf"
    marker.write_bytes(b"neutral model marker")
    role = "chat" if worker == "chat" else "image"
    with SessionLocal() as session:
        install = ModelInstall(
            name="Cleanup model", role=role, engine="mock", local_path=str(model_root), active=True
        )
        session.add(install)
        session.flush()
        profile = ModelProfile(
            name="Cleanup profile", role=role, engine="mock", model_install_id=install.id
        )
        session.add(profile)
        session.commit()
        model_id, profile_id = install.id, profile.id
    _report_cleanup(app, monkeypatch, worker, profile_id)

    identifier = model_id if target == "models" else profile_id
    response = await client.delete(f"/api/{target}/{identifier}", params={"delete_profiles": True})

    assert response.status_code == 409
    expected = (
        "profile-loaded-by-worker"
        if target == "profiles"
        else "media-worker-running"
        if worker == "media"
        else "model-loaded-by-worker"
    )
    assert response.json()["code"] == expected
    assert marker.read_bytes() == b"neutral model marker"
    with SessionLocal() as session:
        assert session.get(ModelInstall, model_id) is not None
        assert session.get(ModelProfile, profile_id) is not None


@pytest.mark.parametrize("operation", ["change", "delete"])
async def test_auxiliary_model_changes_finish_worker_cleanup_first(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    model_root: Path = settings.model_dir / "cleanup-asset"
    model_root.mkdir(parents=True)
    marker = model_root / "neutral.safetensors"
    marker.write_bytes(b"neutral asset marker")
    with SessionLocal() as session:
        asset = ModelAssetInstall(
            name="Cleanup asset",
            kind="lora",
            family="sdxl",
            local_path=str(model_root),
            size_bytes=20,
            manifest_json={"sha256": "d" * 64, "comfy_name": marker.name},
            active=True,
            verified_at=utcnow(),
        )
        session.add(asset)
        session.commit()
        identifier = asset.id
    services = _report_cleanup(app, monkeypatch, "media")
    pending = AsyncMock(side_effect=RuntimeError("neutral cleanup still pending"))
    monkeypatch.setattr(
        services.processes, "start_media" if operation == "change" else "stop", pending
    )

    with pytest.raises(RuntimeError, match="neutral cleanup still pending"):
        if operation == "change":
            await client.patch(f"/api/model-assets/{identifier}", json={"active": False})
        else:
            await client.delete(f"/api/model-assets/{identifier}")

    assert pending.await_count >= 1
    assert marker.read_bytes() == b"neutral asset marker"
    with SessionLocal() as session:
        retained = session.get(ModelAssetInstall, identifier)
        assert retained is not None and retained.active
