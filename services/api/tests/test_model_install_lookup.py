from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from local_lm import api
from local_lm.db import SessionLocal
from local_lm.models import ModelInstall


def _installs() -> tuple[list[str], str]:
    with SessionLocal() as session:
        rows = [
            ModelInstall(
                name=f"Checkpoint {index}",
                role="image",
                engine="mock",
                local_path=f"managed/checkpoint-{index}",
                compatibility="unsupported" if index == 1 else "likely",
                active=index != 3,
                updated_at=datetime(2026, 9, 10, tzinfo=UTC) + timedelta(seconds=index),
            )
            for index in range(4)
        ]
        session.add_all(rows)
        session.commit()
        return [row.id for row in rows[:3]], rows[3].id


@pytest.mark.parametrize("selection", ["selected", "missing", "inactive", "empty"])
async def test_exact_install_filter_precedes_loading_and_capability_checks(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
) -> None:
    active_ids, inactive_id = _installs()
    selected = {
        "selected": active_ids[0],
        "missing": "model_missing",
        "inactive": inactive_id,
        "empty": "",
    }[selection]
    expected = [active_ids[0]] if selection == "selected" else []
    loaded: list[str] = []
    probed: list[str] = []

    def loading(_session: Session, instance: object) -> None:
        if isinstance(instance, ModelInstall):
            loaded.append(instance.id)

    def evidence(*args: object, **_kwargs: object) -> None:
        assert isinstance(args[1], ModelInstall)
        probed.append(args[1].id)

    monkeypatch.setattr(api, "current_capability_evidence", evidence)
    event.listen(Session, "loaded_as_persistent", loading)
    try:
        response = await client.get("/api/models", params={"install_id": selected})
    finally:
        event.remove(Session, "loaded_as_persistent", loading)
    assert response.status_code == 200, response.json()
    assert loaded == expected, "unselected installed models were hydrated"
    assert probed == expected, "unselected installed models were checked"
    assert [row["id"] for row in response.json()] == expected


async def test_unfiltered_model_list_keeps_order_and_readiness(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_ids, _ = _installs()
    probed: list[str] = []

    def evidence(*args: object, **_kwargs: object) -> None:
        assert isinstance(args[1], ModelInstall)
        probed.append(args[1].id)

    monkeypatch.setattr(api, "current_capability_evidence", evidence)
    response = await client.get("/api/models")
    assert response.status_code == 200, response.json()
    rows = response.json()
    assert [row["id"] for row in rows] == list(reversed(active_ids))
    assert probed == list(reversed(active_ids))
    assert [row["readiness"] for row in rows] == ["unverified", "unsupported", "unverified"]
    assert all(row["capability_evidence"] is None for row in rows)
