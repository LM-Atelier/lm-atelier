"""Diagnostics say whether running work could keep the computer awake, and nothing more."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm import main as main_module
from local_lm.config import Settings
from local_lm.main import create_app
from local_lm.power_inhibition import PowerInhibitor, SleepJobKind

# A platform's refusal can name a system path; the bundle must not carry it.
REFUSAL = "refused by the platform at C:/neutral/marker/path"


class _Backend:
    name = "test"
    supported = True

    def __init__(self) -> None:
        self.refuse = False

    def acquire(self, reason: str) -> object:
        if self.refuse:
            raise OSError(REFUSAL)
        return object()

    def release(self, handle: object) -> None:
        return None


@pytest.fixture
def backend() -> _Backend:
    return _Backend()


@pytest.fixture
def app(settings: Settings, backend: _Backend, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(main_module, "default_power_backend", lambda: backend)
    return create_app(settings)


def _power(app: FastAPI) -> PowerInhibitor:
    power = app.state.services.power
    assert isinstance(power, PowerInhibitor)
    return power


async def _bundle(client: AsyncClient) -> tuple[dict[str, Any], bytes]:
    created = await client.post("/api/diagnostics")
    assert created.status_code == 201
    archive = await client.get(created.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        raw = bundle.read("diagnostics.json")
    return json.loads(raw), raw


async def test_the_bundle_says_whether_work_is_keeping_the_computer_awake(
    client: AsyncClient, app: FastAPI
) -> None:
    idle, _ = await _bundle(client)
    assert idle["sleep"] == {
        "supported": True,
        "backend": "test",
        "enabled": True,
        "active": False,
        "holder_count": 0,
        "kind_counts": {},
        "refused": False,
    }

    _power(app).acquire("attempt_a", SleepJobKind.DOWNLOAD)
    busy, _ = await _bundle(client)

    assert busy["sleep"]["active"] is True
    assert busy["sleep"]["holder_count"] == 1
    assert busy["sleep"]["kind_counts"] == {"download": 1}


async def test_a_refusal_is_reported_without_the_platforms_words(
    client: AsyncClient, app: FastAPI, backend: _Backend
) -> None:
    backend.refuse = True
    _power(app).acquire("attempt_a", SleepJobKind.GENERATION)

    payload, raw = await _bundle(client)

    assert payload["sleep"]["refused"] is True
    assert payload["sleep"]["active"] is False
    assert b"neutral/marker" not in raw
    assert b"refused by the platform" not in raw
