"""Use the same setting layers for read-only preparation and real admission."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import event

from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.models import Chat, GenerationPreset, ModelProfile, Project
from local_lm.schemas import TurnRequest
from local_lm.settings_registry import workflow_settings


@pytest.mark.parametrize("ordered", [False, True], ids=["single", "ordered"])
@pytest.mark.parametrize(
    "choice,expected",
    [
        ("inherit", 0.6),
        ("selected", 0.7),
        ("none", 0.6),
        ("override", 0.8),
        ("load_override", None),
    ],
)
async def test_read_only_setting_layers_match_real_turn_admission(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
    expected: float | None,
    ordered: bool,
) -> None:
    orchestrator = app.state.services.orchestrator
    with SessionLocal() as session:
        profile = ModelProfile(
            name="Neutral settings profile",
            role="chat",
            engine="mock",
            load_settings_json={"context_length": 2048},
            request_settings_json={"temperature": 0.1},
        )
        presets = [
            GenerationPreset(
                name=f"Neutral layer {index}",
                role="chat",
                settings_json={"temperature": value},
                is_default=index == 0,
            )
            for index, value in enumerate((0.2, 0.3, 0.5, 0.7))
        ]
        session.add_all([profile, *presets])
        session.flush()
        project = Project(
            name="Neutral settings project",
            generation_preset_ids_json={"chat": presets[1].id},
            generation_settings_json={"chat": {"temperature": 0.4}},
        )
        session.add(project)
        session.flush()
        chat = Chat(
            title="Neutral setting layers",
            project_id=project.id,
            generation_preset_ids_json={"chat": presets[2].id},
            generation_settings_json={"chat": {"temperature": 0.6}},
        )
        session.add(chat)
        session.commit()
        chat_id, profile_id, selected_preset_id = chat.id, profile.id, presets[3].id

    payload: dict[str, Any] = {
        "text": "Describe a blue square.",
        "mode": "text",
        "profile_id": profile_id,
    }
    if choice in {"selected", "override"}:
        payload["preset_id"] = selected_preset_id
    elif choice == "none":
        payload["preset_id"] = None
    if choice == "override":
        payload["settings"] = {"temperature": 0.8}
    elif choice == "load_override":
        payload["settings"] = {"context_length": 4096}
    if ordered:
        payload.update(
            text="Write a short story about a paper boat, then create an image based on it, "
            "then animate the image into a video, then summarize the video",
            mode="auto",
            confirm_media=True,
            ordered_settings={"chat": payload.pop("settings", {})},
        )
    request = TurnRequest.model_validate(payload).for_role("chat", ordered=ordered)
    selection_options = {"ordered": True} if ordered else {}
    fields = workflow_settings(
        await orchestrator.engines.settings_for_role("chat", engine="mock"), None
    )
    # The shared method must exist and be used by normal admission, not just
    # by a separate preview implementation that agrees on these fixtures.
    resolve = orchestrator.resolve_turn_setting_layers
    with SessionLocal() as session, session.no_autoflush:
        read_chat = session.get(Chat, chat_id)
        read_profile = session.get(ModelProfile, profile_id)
        assert read_chat is not None and read_profile is not None
        before = deepcopy(read_chat.generation_settings_json)
        connection = session.connection()

        def read_only(
            _connection: Any, _cursor: Any, _sql: str, _parameters: Any, context: Any, _many: Any
        ) -> None:
            assert not (context.isinsert or context.isupdate or context.isdelete)

        event.listen(connection, "before_cursor_execute", read_only)
        try:
            if choice == "load_override":
                with pytest.raises(ValueError):
                    resolve(
                        session,
                        read_chat,
                        Operation.TEXT,
                        read_profile,
                        request,
                        fields,
                        **selection_options,
                    )
            else:
                prepared = resolve(
                    session,
                    read_chat,
                    Operation.TEXT,
                    read_profile,
                    request,
                    fields,
                    **selection_options,
                )
                assert prepared.effective_settings["temperature"] == expected
                assert prepared.effective_settings["context_length"] == 2048
                assert prepared.mask is None
                assert [scope for scope, _, _ in prepared.preset_layers] == (
                    ["turn"]
                    if choice in {"selected", "override"}
                    else []
                    if choice == "none"
                    else ["default", "project", "chat"]
                )
            assert read_chat.generation_settings_json == before
            assert not session.dirty and not session.new and not session.deleted
        finally:
            event.remove(connection, "before_cursor_execute", read_only)

    calls: list[dict[str, Any]] = []
    ordered_calls: list[tuple[Operation, bool, str | None]] = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = resolve(*args, **kwargs)
        calls.append(deepcopy(result.effective_settings))
        turn_preset = result.presets[3]
        ordered_calls.append(
            (args[2], kwargs.get("ordered", False), turn_preset.id if turn_preset else None)
        )
        return result

    monkeypatch.setattr(orchestrator, "resolve_turn_setting_layers", observe)
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(f"/api/chats/{chat_id}/turns", json=payload)
        if choice == "load_override":
            assert response.status_code == 422
            assert calls == []
        else:
            assert response.status_code == 202, response.text
            assert len(calls) == (4 if ordered else 1)
            if ordered:
                assert [operation for operation, _, _ in ordered_calls] == [
                    Operation.TEXT,
                    Operation.TEXT_TO_IMAGE,
                    Operation.IMAGE_TO_VIDEO,
                    Operation.TEXT,
                ]
                assert all(is_ordered for _, is_ordered, _ in ordered_calls)
                assert calls[3]["temperature"] == expected
                assert ordered_calls[1][2] is None and ordered_calls[2][2] is None
            assert calls[0]["temperature"] == expected
            assert response.json()["run"]["settings_json"]["temperature"] == expected
            assert response.json()["run"]["settings_json"]["context_length"] == 2048
