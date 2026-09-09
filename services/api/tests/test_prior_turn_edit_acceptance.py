from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, Job, Message, Project, Run, WorkPlan
from local_lm.orchestrator import ConversationOrchestrator


async def _turn(client: AsyncClient, chat_id: str, text: str) -> dict[str, Any]:
    response = await client.post(f"/api/chats/{chat_id}/turns", json={"text": text, "mode": "text"})
    assert response.status_code == 202
    return response.json()


def _source_state(run_id: str) -> dict[str, Any]:
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        plan = session.get(WorkPlan, run.work_plan_id)
        job = session.scalar(select(Job).where(Job.run_id == run.id))
        user = session.get(Message, run.user_message_id)
        assistant = session.get(Message, run.assistant_message_id)
        assert plan is not None and job is not None and user is not None and assistant is not None
        return deepcopy(
            {
                "run": (run.id, run.status, run.settings_json, run.standalone_prompt),
                "plan": (plan.id, plan.transcript_sequence, plan.priority, plan.source_action),
                "job": (job.id, job.status, job.queue_ticket),
                "user": (
                    user.id,
                    user.parent_id,
                    user.status,
                    [(p.type, p.text, p.artifact_id, p.position) for p in user.parts],
                ),
                "assistant": (
                    assistant.id,
                    assistant.parent_id,
                    assistant.status,
                    [(p.type, p.text, p.artifact_id, p.position) for p in assistant.parts],
                ),
            }
        )


async def test_queue_edit_preserves_original_descendants_and_active_branch(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Parallel versions"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, chat["id"], "Original question")
        later = await _turn(client, chat["id"], "Keep this later question")
        source_before = _source_state(source["run"]["id"])
        later_before = _source_state(later["run"]["id"])
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={"text": "Revised question", "idempotency_key": "edit-once"},
        )
        assert response.status_code == 202
        edited = response.json()
        assert edited["source_message_id"] == source["user_message"]["id"]
        assert edited["branch_activated"] is False
        assert edited["run"]["id"] not in {source["run"]["id"], later["run"]["id"]}
        assert _source_state(source["run"]["id"]) == source_before
        assert _source_state(later["run"]["id"]) == later_before
        with SessionLocal() as session:
            current = session.get(Chat, chat["id"])
            assert current is not None
            assert current.active_head_message_id == later["assistant_message"]["id"]
        new_state = _source_state(edited["run"]["id"])
        assert new_state["plan"][1] > later_before["plan"][1]
        assert new_state["plan"][2] == later_before["plan"][2]
        assert new_state["plan"][3] == "edit_and_branch"
        assert new_state["job"][2] not in {source_before["job"][2], later_before["job"][2]}
        assert new_state["user"][1] == source["user_message"]["parent_id"]


async def test_edit_replay_returns_same_work_when_queue_is_full(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import local_lm.orchestrator as orchestration

    monkeypatch.setattr(orchestration, "MAX_PENDING_WORK_PER_CHAT", 2)
    chat = (await client.post("/api/chats", json={"title": "Retry acceptance"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, chat["id"], "First version")
        url = f"/api/messages/{source['user_message']['id']}/edits"
        payload = {"text": "Second version", "idempotency_key": "stable-edit-request"}
        first = await client.post(url, json=payload)
        assert first.status_code == 202
        again = await client.post(url, json=payload)
        assert again.status_code == 202
        assert again.json()["run"]["id"] == first.json()["run"]["id"]
        assert again.json()["assistant_message"]["id"] == first.json()["assistant_message"]["id"]
        with SessionLocal() as session:
            assert (
                len(list(session.scalars(select(WorkPlan).where(WorkPlan.chat_id == chat["id"]))))
                == 2
            )


async def test_edit_idempotency_rejects_changed_source_or_payload(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Exact edit request"})).json()
    async with app.state.services.scheduler.lease("primary"):
        first = await _turn(client, chat["id"], "First question")
        second = await _turn(client, chat["id"], "Second question")
        payload = {"text": "One revision", "idempotency_key": "same-key"}
        url = f"/api/messages/{first['user_message']['id']}/edits"
        accepted = await client.post(url, json=payload)
        assert accepted.status_code == 202
        changed = await client.post(url, json={**payload, "text": "Different revision"})
        assert changed.status_code == 409
        other_source = await client.post(
            f"/api/messages/{second['user_message']['id']}/edits", json=payload
        )
        assert other_source.status_code == 409


@pytest.mark.parametrize(
    "replacement", [None, [], ["replacement"]], ids=["inherit", "remove", "replace"]
)
async def test_edit_attachment_field_preserves_three_distinct_intents(
    app: FastAPI,
    client: AsyncClient,
    replacement: list[str] | None,
) -> None:
    uploads = []
    for name in ("original", "replacement"):
        response = await client.post(
            "/api/artifacts",
            files={"file": (name + ".txt", (name + " neutral content").encode(), "text/plain")},
        )
        assert response.status_code == 201
        uploads.append(response.json()["id"])
    chat = (await client.post("/api/chats", json={"title": "Attachment intent"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source_response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Summarize this note",
                "mode": "text",
                "input_artifact_ids": [uploads[0]],
            },
        )
        assert source_response.status_code == 202
        source = source_response.json()
        payload: dict[str, Any] = {
            "text": "Summarize briefly",
            "idempotency_key": "attachment-edit",
        }
        if replacement is not None:
            payload["input_artifact_ids"] = [uploads[1]] if replacement else []
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits", json=payload
        )
        assert response.status_code == 202
        expected = [uploads[0]] if replacement is None else ([uploads[1]] if replacement else [])
        with SessionLocal() as session:
            run = session.get(Run, response.json()["run"]["id"])
            original = session.get(Run, source["run"]["id"])
            assert run is not None and original is not None
            assert ConversationOrchestrator.input_artifact_ids_for_run(session, run) == expected
            assert ConversationOrchestrator.input_artifact_ids_for_run(session, original) == [
                uploads[0]
            ]


async def test_edit_context_keeps_accepted_project_instructions(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    project = (
        await client.post(
            "/api/projects",
            json={"name": "Snapshot project", "instructions": "Use short paragraphs"},
        )
    ).json()
    chat = (
        await client.post(
            "/api/chats", json={"title": "Frozen context", "project_id": project["id"]}
        )
    ).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, chat["id"], "Explain the color wheel")
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={"text": "Explain primary colors", "idempotency_key": "frozen-context"},
        )
        assert response.status_code == 202
        with SessionLocal() as session:
            accepted = session.get(Run, response.json()["run"]["id"])
            assert accepted is not None
            before = ConversationOrchestrator._context_messages(session, accepted)
            assert {"role": "system", "content": "Use short paragraphs"} in before
            changed_project = session.get(Project, project["id"])
            assert changed_project is not None
            changed_project.instructions = "Use long numbered lists"
            session.commit()
        with SessionLocal() as session:
            persisted = session.get(Run, response.json()["run"]["id"])
            assert persisted is not None
            assert ConversationOrchestrator._context_messages(session, persisted) == before


async def test_legacy_branch_still_activates_its_new_head(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Legacy branch"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, chat["id"], "Original version")
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/branch",
            json={"text": "Legacy edited version"},
        )
        assert response.status_code == 202
        with SessionLocal() as session:
            current = session.get(Chat, chat["id"])
            assert current is not None
            assert current.active_head_message_id == response.json()["assistant_message"]["id"]


@pytest.mark.parametrize("ordered", [False, True], ids=["single", "ordered"])
async def test_nonactivating_acceptance_keeps_the_current_branch(
    app: FastAPI,
    client: AsyncClient,
    ordered: bool,
) -> None:
    from local_lm.schemas import TurnRequest

    chat = (await client.post("/api/chats", json={"title": "Keep the visible branch"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, chat["id"], "Keep this original")
        later = await _turn(client, chat["id"], "Keep this descendant")
        before = _source_state(source["run"]["id"])
        text = (
            "Write a short story about a paper boat, then create an image based on it, "
            "then animate the image into a video, then summarize the video"
            if ordered
            else "An alternate question"
        )
        turn = TurnRequest.model_validate(
            {
                "text": text,
                "mode": "auto" if ordered else "text",
                "parent_message_id": source["user_message"]["parent_id"],
                "idempotency_key": "nonactivating-acceptance",
                "confirm_media": True,
            }
        )
        with SessionLocal() as session:
            accepted = await app.state.services.orchestrator.create_turn(
                session,
                chat["id"],
                turn,
                use_explicit_parent=True,
                source_action="edit_and_branch",
                activate_branch=False,
            )
            plan = session.get(WorkPlan, accepted.run.work_plan_id)
            assert plan is not None
            assert len(plan.steps) == (4 if ordered else 1)
            assert plan.transcript_sequence > _source_state(later["run"]["id"])["plan"][1]
        with SessionLocal() as session:
            current = session.get(Chat, chat["id"])
            assert current is not None
            assert current.active_head_message_id == later["assistant_message"]["id"]
        assert _source_state(source["run"]["id"]) == before


async def test_accepted_request_replays_at_capacity_without_admitting_new_work(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import local_lm.orchestrator as orchestration

    monkeypatch.setattr(orchestration, "MAX_PENDING_WORK_PER_CHAT", 1)
    chat = (await client.post("/api/chats", json={"title": "Full queue replay"})).json()
    url = f"/api/chats/{chat['id']}/turns"
    payload = {"text": "Accepted once", "mode": "text", "idempotency_key": "same-request"}
    async with app.state.services.scheduler.lease("primary"):
        first = await client.post(url, json=payload)
        assert first.status_code == 202
        replay = await client.post(url, json=payload)
        assert replay.status_code == 202
        assert replay.json()["run"]["id"] == first.json()["run"]["id"]
        refused = await client.post(url, json={**payload, "idempotency_key": "new-request"})
        assert refused.status_code == 422
        with SessionLocal() as session:
            plans = list(session.scalars(select(WorkPlan).where(WorkPlan.chat_id == chat["id"])))
            assert len(plans) == 1


@pytest.mark.parametrize("status", ["queued", "running", "complete"])
async def test_edit_source_preloads_the_selected_turn_without_mutation(
    app: FastAPI, client: AsyncClient, status: str
) -> None:
    from local_lm.models import ReferenceSubject

    subject = (
        await client.post("/api/references", json={"name": "Neutral subject", "kind": "person"})
    ).json()
    upload = (
        await client.post(
            "/api/artifacts", files={"file": ("note.txt", b"Neutral source note", "text/plain")}
        )
    ).json()
    chat = (await client.post("/api/chats", json={"title": "Source initializer"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source_response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Summarize the neutral note",
                "mode": "text",
                "settings": {"temperature": 0.25},
                "input_artifact_ids": [upload["id"]],
                "references": [
                    {
                        "reference_subject_id": subject["id"],
                        "role": "subject",
                        "strength": 0.65,
                        "source": "picker",
                    }
                ],
            },
        )
        assert source_response.status_code == 202
        source = source_response.json()
        later = await _turn(client, chat["id"], "Keep the later question")
        with SessionLocal() as session:
            run = session.get(Run, source["run"]["id"])
            assert run is not None
            run.status = status
            job = session.scalar(select(Job).where(Job.run_id == run.id))
            assert job is not None
            job.status = status
            changed_subject = session.get(ReferenceSubject, subject["id"])
            assert changed_subject is not None
            changed_subject.name = "Later renamed subject"
            session.commit()
        before, later_before = _source_state(source["run"]["id"]), _source_state(later["run"]["id"])
        url = f"/api/messages/{source['user_message']['id']}/edit-source"
        response = await client.get(url, params={"source_run_id": source["run"]["id"]})
        assert response.status_code == 200, response.text
        editor = response.json()
        assert editor["source_user_message_id"] == source["user_message"]["id"]
        assert editor["source_run_id"] == source["run"]["id"]
        assert len(editor["source_snapshot_sha256"]) == 64
        assert editor["text"] == "Summarize the neutral note"
        assert editor["mode"] == "text"
        assert editor["settings"]["temperature"] == 0.25
        assert editor["input_artifact_ids"] == [upload["id"]]
        assert [item["id"] for item in editor["input_artifacts"]] == [upload["id"]]
        assert editor["references"][0]["subject_name"] == "Neutral subject"
        assert editor["references"][0]["strength"] == 0.65
        assert editor["workflow_selection"]["legacy_profile_id"] == source["run"]["profile_id"]
        assert all(
            "Keep the later question" not in item["content"] for item in editor["context_messages"]
        )
        edited = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={
                "text": "Summarize briefly",
                "idempotency_key": "from-initializer",
                "source_run_id": editor["source_run_id"],
            },
        )
        assert edited.status_code == 202, edited.text
        assert edited.json()["source_run_id"] == editor["source_run_id"]
        assert edited.json()["user_message"]["references"] == editor["references"]
        assert _source_state(source["run"]["id"]) == before
        assert _source_state(later["run"]["id"]) == later_before


async def test_edit_source_rejects_a_run_from_another_source(
    app: FastAPI, client: AsyncClient
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Source identity"})).json()
    async with app.state.services.scheduler.lease("primary"):
        first = await _turn(client, chat["id"], "First question")
        second = await _turn(client, chat["id"], "Second question")
        response = await client.get(
            f"/api/messages/{first['user_message']['id']}/edit-source",
            params={"source_run_id": second["run"]["id"]},
        )
        assert response.status_code == 404
        response = await client.post(
            f"/api/messages/{first['user_message']['id']}/edits",
            json={
                "text": "Wrong source",
                "idempotency_key": "wrong-source",
                "source_run_id": second["run"]["id"],
            },
        )
        assert response.status_code == 404


@pytest.mark.parametrize("ordered", [False, True], ids=["outputs", "ordered"])
async def test_edited_branch_head_contains_every_accepted_output(
    app: FastAPI, client: AsyncClient, ordered: bool
) -> None:
    from local_lm.models import WorkStep

    chat = (await client.post("/api/chats", json={"title": "Complete alternate branch"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, chat["id"], "Keep this source")
        payload = {
            "text": (
                "Write a short story about a paper boat, then create an image based on it, "
                "then animate the image into a video, then summarize the video"
            )
            if ordered
            else "Draw three blue circles",
            "mode": "auto" if ordered else "image",
            "confirm_media": True,
            "idempotency_key": "complete-edited-branch",
        }
        if not ordered:
            payload["output_count"] = 3
        response = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits", json=payload
        )
        assert response.status_code == 202, response.text
        result = response.json()
        from local_lm.accepted_turn_context import accepted_context

        with SessionLocal() as session:
            runs = list(
                session.scalars(
                    select(Run)
                    .join(WorkStep, WorkStep.id == Run.work_step_id)
                    .where(Run.work_plan_id == result["work_plan_id"])
                    .order_by(WorkStep.ordinal)
                )
            )
            assert len(runs) == (4 if ordered else 3)
            assert result["branch_head_message_id"] == runs[-1].assistant_message_id
            assert result["branch_head_message_id"] != result["assistant_message"]["id"]
            for run in runs:
                snapshot = accepted_context(session, run)
                assert snapshot is not None
                assert snapshot.source_message_id == source["user_message"]["id"]
                assert snapshot.source_run_id == source["run"]["id"]
            current = session.get(Chat, chat["id"])
            assert current is not None
            assert current.active_head_message_id == source["assistant_message"]["id"]
        replay = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits", json=payload
        )
        assert replay.status_code == 202
        assert replay.json()["branch_head_message_id"] == result["branch_head_message_id"]


@pytest.mark.parametrize("selection", ["inherit", "explicit", "ordered", "missing", "wrong_role"])
async def test_edit_model_selection_is_local_to_the_edited_turn(
    app: FastAPI, client: AsyncClient, selection: str
) -> None:
    from sqlalchemy import delete

    from local_lm.models import ChatWorkflowSelection, ModelProfile

    chat = (await client.post("/api/chats", json={"title": "Edit model choice"})).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, chat["id"], "Original model question")
        source_profile_id = source["run"]["profile_id"]
        assert source_profile_id is not None
        with SessionLocal() as session:
            alternate = ModelProfile(
                name="Later chat model",
                role="chat",
                engine="mock",
                load_settings_json={},
                request_settings_json={},
            )
            wrong_role = ModelProfile(
                name="Image-only choice",
                role="image",
                engine="mock",
                load_settings_json={},
                request_settings_json={},
            )
            session.add_all([alternate, wrong_role])
            session.flush()
            alternate_id, wrong_role_id = alternate.id, wrong_role.id
            current = session.get(Chat, chat["id"])
            assert current is not None
            current.active_chat_profile_id = alternate_id
            session.execute(
                delete(ChatWorkflowSelection).where(ChatWorkflowSelection.chat_id == chat["id"])
            )
            session.commit()
        payload: dict[str, Any] = {
            "text": "An edited question",
            "idempotency_key": "edit-model-choice",
        }
        if selection != "inherit":
            payload["profile_id"] = (
                alternate_id
                if selection in {"explicit", "ordered"}
                else wrong_role_id
                if selection == "wrong_role"
                else "profile_missing"
            )
        if selection == "ordered":
            payload.update(
                mode="auto",
                confirm_media=True,
                text=(
                    "Write a short story about a paper boat, then create an image based on it, "
                    "then animate the image into a video, then summarize the video"
                ),
            )
        result = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits", json=payload
        )
        if selection == "ordered" and result.status_code == 202:
            with SessionLocal() as session:
                steps = list(
                    session.scalars(
                        select(Run).where(Run.work_plan_id == result.json()["work_plan_id"])
                    )
                )
                assert len(steps) == 4
                assert all(
                    (run.profile_id == alternate_id) == (run.operation == "text") for run in steps
                )
        if selection in {"missing", "wrong_role"}:
            assert result.status_code == (404 if selection == "missing" else 422), result.text
            assert result.json()["code"] == (
                "turn-subject-not-found" if selection == "missing" else "turn-invalid"
            )
        else:
            assert result.status_code == 202, result.text
            assert result.json()["run"]["profile_id"] == (
                source_profile_id if selection == "inherit" else alternate_id
            )
        with SessionLocal() as session:
            current = session.get(Chat, chat["id"])
            original = session.get(Run, source["run"]["id"])
            assert current is not None and original is not None
            assert current.active_chat_profile_id == alternate_id
            assert original.profile_id == source_profile_id


@pytest.mark.parametrize("selection", ["inherit", "disable", "override", "missing", "unverified"])
async def test_edit_vision_choice_is_inherited_or_explicitly_disabled(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    from sqlalchemy import delete

    from local_lm.models import ChatWorkflowSelection, ModelInstall, ModelProfile

    chat = (await client.post("/api/chats", json={"title": "Edit vision choice"})).json()
    orchestrator = app.state.services.orchestrator
    async with app.state.services.scheduler.lease("primary"):
        source = await _turn(client, chat["id"], "Original vision choice")
        with SessionLocal() as session:
            profiles = []
            for name in ("Source vision", "Later vision"):
                install = ModelInstall(
                    name=name,
                    role="chat",
                    engine="mock",
                    local_path="C:/managed/neutral-vision",
                    manifest_json={},
                    active=True,
                )
                session.add(install)
                session.flush()
                profile = ModelProfile(
                    name=name,
                    role="chat",
                    engine="mock",
                    model_install_id=install.id,
                    load_settings_json={},
                    request_settings_json={},
                )
                session.add(profile)
                session.flush()
                profiles.append(profile.id)
            source_vision, later_vision = profiles
            original = session.get(Run, source["run"]["id"])
            current = session.get(Chat, chat["id"])
            assert original is not None and current is not None
            original.vision_profile_id = source_vision
            current.active_vision_profile_id = later_vision
            session.execute(
                delete(ChatWorkflowSelection).where(
                    ChatWorkflowSelection.chat_id == chat["id"],
                    ChatWorkflowSelection.selector_capability == "vision",
                )
            )
            session.commit()
        # Runtime verification is a separate authority; these two neutral profiles
        # stand for verified choices, while the ordinary text model is unverified.
        monkeypatch.setattr(
            orchestrator,
            "_profile_has_verified_vision",
            lambda session, profile: profile.id in profiles,
        )
        payload: dict[str, Any] = {
            "text": "An edited question",
            "idempotency_key": "edit-vision-choice",
        }
        if selection != "inherit":
            payload["vision_profile_id"] = {
                "disable": None,
                "override": later_vision,
                "missing": "profile_missing",
                "unverified": source["run"]["profile_id"],
            }[selection]
        result = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits", json=payload
        )
        if selection in {"missing", "unverified"}:
            assert result.status_code == (404 if selection == "missing" else 422), result.text
        else:
            assert result.status_code == 202, result.text
            assert (
                result.json()["run"]["vision_profile_id"]
                == {"inherit": source_vision, "disable": None, "override": later_vision}[selection]
            )
        with SessionLocal() as session:
            current = session.get(Chat, chat["id"])
            original = session.get(Run, source["run"]["id"])
            assert current is not None and original is not None
            assert current.active_vision_profile_id == later_vision
            assert original.vision_profile_id == source_vision
