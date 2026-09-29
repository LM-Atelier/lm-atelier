from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import event
from test_prior_turn_edit_presets import _preset_source

from local_lm import prior_turn_edits
from local_lm.db import SessionLocal
from local_lm.models import GenerationPreset, Message
from local_lm.schemas import PriorTurnEditRequest


@pytest.mark.parametrize("choice", ["inherit", "deleted", "explicit", "none", "override"])
async def test_edit_preparation_is_read_only_and_shared_with_admission(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    orchestrator = app.state.services.orchestrator
    async with app.state.services.scheduler.lease("primary"):
        source, presets = await _preset_source(client)
        if choice == "deleted":
            with SessionLocal() as session:
                preset = session.get(GenerationPreset, presets[0])
                assert preset is not None
                session.delete(preset)
                session.commit()
        values: dict[str, Any] = {"text": "Changed question", "idempotency_key": "prepared-edit"}
        if choice in {"explicit", "override"}:
            values["preset_id"] = presets[1]
        elif choice == "none":
            values["preset_id"] = None
        if choice == "override":
            values["settings"] = {"temperature": 0.41}
        payload = PriorTurnEditRequest.model_validate(values)
        prepare = prior_turn_edits.prepare_prior_turn_edit
        writes = []

        def observe(
            connection: Any,
            cursor: Any,
            statement: str,
            parameters: Any,
            context: Any,
            many: bool,
        ) -> None:
            if statement.lstrip().split(" ", 1)[0].upper() in {"INSERT", "UPDATE", "DELETE"}:
                writes.append(statement.split(" ", 1)[0])

        with SessionLocal() as session:
            connection = session.connection()
            event.listen(connection, "before_cursor_execute", observe)
            try:
                with session.no_autoflush:
                    prepared = await prepare(
                        orchestrator, session, source["user_message"]["id"], payload
                    )
                assert not session.new and not session.dirty and not session.deleted
                assert not writes
                request_values = prepared.request.model_dump(mode="json")
                assert prepared.prior.id == source["run"]["id"]
                assert prepared.request.parent_message_id == source["user_message"]["parent_id"]
                assert prepared.same_role
                assert prepared.source_inheritance is None
                if choice in {"inherit", "deleted", "none"}:
                    assert prepared.request.preset_id is None
                    assert prepared.request.settings["temperature"] == 0.23
                else:
                    assert prepared.request.preset_id == presets[1]
                assert prepared.inherit_preset == (choice in {"inherit", "deleted"})
            finally:
                event.remove(connection, "before_cursor_execute", observe)
        calls = []

        async def tracked(*args: Any, **kwargs: Any) -> Any:
            result = await prepare(*args, **kwargs)
            calls.append(result.request.model_dump(mode="json"))
            return result

        monkeypatch.setattr(prior_turn_edits, "prepare_prior_turn_edit", tracked)
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits", json=values
        )
        assert response.status_code == 202, response.text
        assert calls == [request_values]
        expected = 0.41 if choice == "override" else 0.61 if choice == "explicit" else 0.23
        assert response.json()["run"]["settings_json"]["temperature"] == expected


@pytest.mark.parametrize("change", ["stale", "hidden", "wrong_run", "during_await"])
async def test_edit_preparation_refuses_changed_or_unowned_source(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    orchestrator = app.state.services.orchestrator
    async with app.state.services.scheduler.lease("primary"):
        source, _ = await _preset_source(client)
        values: dict[str, Any] = {"text": "Changed question", "idempotency_key": "refused-edit"}
        if change == "stale":
            values["source_snapshot_sha256"] = "0" * 64
        if change == "wrong_run":
            values["source_run_id"] = "missing-run"
        if change == "hidden":
            with SessionLocal() as session:
                message = session.get(Message, source["user_message"]["id"])
                assert message is not None
                message.transcript_visible = False
                session.commit()
        if change == "during_await":
            original = orchestrator.request_settings_for_operation

            async def changed(*args: Any, **kwargs: Any) -> Any:
                result = await original(*args, **kwargs)
                with SessionLocal() as session:
                    message = session.get(Message, source["user_message"]["id"])
                    assert message is not None
                    message.parts[0].text = "A different neutral question"
                    session.commit()
                return result

            monkeypatch.setattr(orchestrator, "request_settings_for_operation", changed)
        prepare = prior_turn_edits.prepare_prior_turn_edit
        error = (
            LookupError
            if change in {"hidden", "wrong_run"}
            else prior_turn_edits.EditRequestConflict
        )
        with SessionLocal() as session, session.no_autoflush:
            with pytest.raises(error):
                await prepare(
                    orchestrator,
                    session,
                    source["user_message"]["id"],
                    PriorTurnEditRequest.model_validate(values),
                )
            assert not session.new and not session.dirty and not session.deleted
