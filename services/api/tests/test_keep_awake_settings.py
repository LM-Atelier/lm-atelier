"""Keeping the computer awake while work runs is a setting a person can see and change."""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm import api as api_module
from local_lm import main as main_module
from local_lm.config import Settings
from local_lm.main import create_app
from local_lm.power_inhibition import PowerInhibitor, SleepJobKind, UnsupportedPowerBackend
from local_lm.runtime_config import (
    RuntimeConfigError,
    configure_persisted_runtime,
    runtime_config_path,
)

KEY = "LOCAL_LM_KEEP_AWAKE_DURING_WORK"


class _Backend:
    name = "test"
    supported = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def acquire(self, reason: str) -> object:
        self.calls.append("acquire")
        return object()

    def release(self, handle: object) -> None:
        self.calls.append("release")


@pytest.fixture
def backend() -> _Backend:
    return _Backend()


@pytest.fixture
def app(settings: Settings, backend: _Backend, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(main_module, "default_power_backend", lambda: backend)
    return create_app(settings)


@pytest.fixture(autouse=True)
def _forget_the_saved_value(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Saving also applies the value to this process's environment.
    monkeypatch.delenv(KEY, raising=False)
    yield


def _power(app: FastAPI) -> PowerInhibitor:
    power = app.state.services.power
    assert isinstance(power, PowerInhibitor)
    return power


async def test_the_status_shows_the_setting_and_the_work_holding_it(
    client: AsyncClient, app: FastAPI
) -> None:
    idle = await client.get("/api/settings/keep-awake")
    assert idle.json() == {"enabled": True, "supported": True, "active": False, "running_jobs": 0}

    _power(app).acquire("job_a", SleepJobKind.GENERATION)
    busy = await client.get("/api/settings/keep-awake")

    assert busy.json() == {"enabled": True, "supported": True, "active": True, "running_jobs": 1}


async def test_the_scheduler_and_the_setting_share_one_hold(app: FastAPI) -> None:
    assert app.state.services.scheduler._power is _power(app)


async def test_turning_it_off_lets_the_computer_sleep_without_touching_the_work(
    client: AsyncClient, app: FastAPI, backend: _Backend, settings: Settings
) -> None:
    _power(app).acquire("job_a", SleepJobKind.DOWNLOAD)

    off = await client.put("/api/settings/keep-awake", json={"enabled": False})

    assert off.status_code == 200
    assert off.json() == {"enabled": False, "supported": True, "active": False, "running_jobs": 1}
    assert backend.calls == ["acquire", "release"]
    assert settings.keep_awake_during_work is False
    saved = json.loads(runtime_config_path(settings.data_dir).read_text(encoding="utf-8"))
    assert saved[KEY] == "false"

    on = await client.put("/api/settings/keep-awake", json={"enabled": True})

    assert on.json() == {"enabled": True, "supported": True, "active": True, "running_jobs": 1}
    assert backend.calls == ["acquire", "release", "acquire"]


async def test_the_saved_choice_is_what_the_next_start_reads(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    await client.put("/api/settings/keep-awake", json={"enabled": False})
    monkeypatch.delenv(KEY)
    environment: dict[str, str] = {}

    configure_persisted_runtime(settings.data_dir, environment)
    monkeypatch.setenv(KEY, environment[KEY])

    assert Settings(data_dir=settings.data_dir).keep_awake_during_work is False


async def test_a_choice_that_cannot_be_saved_is_not_applied(
    client: AsyncClient, app: FastAPI, backend: _Backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    _power(app).acquire("job_a", SleepJobKind.GENERATION)

    def refusing(*args: object, **kwargs: object) -> None:
        raise RuntimeConfigError("refused")

    monkeypatch.setattr(api_module, "persist_runtime_values", refusing)
    response = await client.put("/api/settings/keep-awake", json={"enabled": False})

    assert response.status_code == 409
    assert response.json()["code"] == "keep-awake-setting-not-saved"
    assert _power(app).state().active is True and backend.calls == ["acquire"]


async def test_a_platform_without_a_way_to_stay_awake_says_so(
    client: AsyncClient, app: FastAPI
) -> None:
    app.state.services.power = PowerInhibitor(UnsupportedPowerBackend())
    _power(app).acquire("job_a", SleepJobKind.GENERATION)

    status = (await client.get("/api/settings/keep-awake")).json()

    assert status == {"enabled": True, "supported": False, "active": False, "running_jobs": 1}
