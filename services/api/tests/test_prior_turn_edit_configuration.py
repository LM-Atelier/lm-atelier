from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import Chat, ModelInstall, ModelProfile, Run
from local_lm.schemas import WorkerStatus


@pytest.mark.parametrize(
    "selection",
    [
        "inherit",
        "explicit_profile",
        "partial_settings",
        "inherit_vision",
        "explicit_vision",
        "clear_vision",
    ],
)
async def test_edit_keeps_accepted_model_configuration_until_explicitly_reselected(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    orchestrator = app.state.services.orchestrator
    fields = await orchestrator.engines.settings_for_role("chat", engine="mock")
    vision_case = selection in {"inherit_vision", "explicit_vision", "clear_vision"}
    if vision_case:
        monkeypatch.setattr(
            orchestrator,
            "_profile_has_verified_vision",
            lambda _session, profile: profile.id == "profile-edit-configuration",
        )

    async def neutral_settings(_role: str, **_kwargs: Any) -> Any:
        return fields

    monkeypatch.setattr(orchestrator.engines, "settings_for_role", neutral_settings)
    monkeypatch.setattr(orchestrator.engines.settings, "chat_engine", "llama.cpp")
    chat = (await client.post("/api/chats", json={"title": "Inherited model configuration"})).json()
    original_load = {"context_length": 4096, "gpu_layers": 9}
    later_load = {"context_length": 32768, "gpu_layers": 1}
    with SessionLocal() as session:
        install = ModelInstall(
            id="model-edit-configuration",
            name="Synthetic model",
            role="chat",
            engine="llama.cpp",
            local_path="synthetic-model.gguf",
            active=True,
        )
        profile = ModelProfile(
            id="profile-edit-configuration",
            model_install_id=install.id,
            name="Original model configuration",
            role="chat",
            engine="llama.cpp",
            load_settings_json=original_load,
            request_settings_json={"temperature": 0.25},
        )
        session.add_all([install, profile])
        stored_chat = session.get(Chat, chat["id"])
        assert stored_chat is not None
        stored_chat.active_chat_profile_id = profile.id
        from local_lm.workflow_compatibility import mirror_legacy_chat_workflow_selections

        mirror_legacy_chat_workflow_selections(session, stored_chat, ["chat"])
        session.commit()

    async with app.state.services.scheduler.lease("primary"):
        initial = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": "Explain how a paper boat floats.", "mode": "text"},
        )
        assert initial.status_code == 202, initial.text
        first = await client.post(
            f"/api/messages/{initial.json()['user_message']['id']}/edits",
            json={"text": "Explain buoyancy.", "idempotency_key": "first-model-edit"},
        )
        assert first.status_code == 202, first.text
        from local_lm.accepted_turn_context import accepted_context

        with SessionLocal() as session:
            source = session.get(Run, first.json()["run"]["id"])
            assert source is not None
            frozen = accepted_context(session, source)
            assert frozen is not None and frozen.profile is not None
            assert frozen.profile.load_settings_json == original_load
            profile = session.get(ModelProfile, "profile-edit-configuration")
            assert profile is not None
            profile.load_settings_json = later_load
            profile.request_settings_json = {"temperature": 0.75, "top_p": 0.3}
            profile.name = "Later model configuration"
            session.commit()

        source_view = await client.get(
            f"/api/messages/{first.json()['user_message']['id']}/edit-source"
        )
        assert source_view.status_code == 200, source_view.text
        assert source_view.json()["profile_settings"] == {
            **original_load,
            "temperature": 0.25,
        }

        payload: dict[str, Any] = {
            "text": "Explain buoyancy with a short example.",
            "idempotency_key": "second-model-edit",
        }
        if selection == "explicit_profile":
            payload["profile_id"] = "profile-edit-configuration"
        elif selection == "partial_settings":
            payload["settings"] = {"max_tokens": 128}
        elif selection == "explicit_vision":
            payload["vision_profile_id"] = "profile-edit-configuration"
        elif selection == "clear_vision":
            payload["vision_profile_id"] = None
        second = await client.post(
            f"/api/messages/{first.json()['user_message']['id']}/edits", json=payload
        )
        assert second.status_code == 202, second.text
        with SessionLocal() as session:
            edited = session.get(Run, second.json()["run"]["id"])
            assert edited is not None
            current = accepted_context(session, edited)
            assert current is not None and current.profile is not None
            expected_load = later_load if selection == "explicit_profile" else original_load
            assert current.profile.load_settings_json == expected_load
            assert current.context_limit == expected_load["context_length"]
            if vision_case:
                assert frozen.vision_profile is not None
                if selection == "clear_vision":
                    assert current.vision_profile is None and current.vision_profile_id is None
                else:
                    from local_lm.accepted_turn_context import resolve_accepted_profile

                    assert current.vision_profile is not None
                    expected_vision = (
                        later_load if selection == "explicit_vision" else original_load
                    )
                    assert current.vision_profile.load_settings_json == expected_vision
                    resolved_vision, _, _ = resolve_accepted_profile(
                        session, current.vision_profile
                    )
                    assert resolved_vision.load_settings_json == expected_vision

            if selection == "partial_settings":
                assert current.settings["temperature"] == frozen.settings["temperature"]
                assert current.settings["max_tokens"] == 128
            untouched = session.get(Run, source.id)
            assert untouched is not None
            assert accepted_context(session, untouched) == frozen

        loaded = WorkerStatus(
            name="chat",
            state="ready",
            running=True,
            managed=True,
            profile_id="profile-edit-configuration",
        )
        loader = AsyncMock(return_value=loaded)
        monkeypatch.setattr(orchestrator.processes, "statuses", Mock(return_value=[]))
        monkeypatch.setattr(orchestrator.processes, "load_chat", loader)
        assert await orchestrator._ensure_chat_worker(second.json()["run"]["id"]) is loaded
        assert loader.await_args.args[0].load_settings_json == expected_load
        with SessionLocal() as session:
            live_profile = session.get(ModelProfile, "profile-edit-configuration")
            assert live_profile is not None and live_profile.load_settings_json == later_load
