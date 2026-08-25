"""Probe: regenerating a reply whose source item's content was removed.

The design requires a replay that needs removed payload to refuse with a typed
`source-content-removed`. This asks what actually happens today, once the
execute slice makes removal reachable.
"""

from __future__ import annotations

from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import (
    Chat,
    Message,
    MessagePart,
    ResponseRevision,
    Run,
)


async def test_probe_regenerate_after_source_content_removed(client: AsyncClient) -> None:
    with SessionLocal() as session:
        chat = Chat(id="chat_probe_regen", title="Probe")
        user = Message(id="msg_probe_user", chat=chat, role="user", status="complete")
        assistant = Message(
            id="msg_probe_assistant",
            chat=chat,
            parent_id=user.id,
            role="assistant",
            status="complete",
        )
        session.add_all([chat, user, assistant])
        session.flush()
        session.add(
            MessagePart(
                id="part_probe_user",
                message_id=user.id,
                position=0,
                type="text",
                text="the only copy of the prompt",
                metadata_json={},
            )
        )
        run = Run(
            id="run_probe",
            chat_id=chat.id,
            user_message_id=user.id,
            assistant_message_id=assistant.id,
            status="complete",
            operation="text",
            standalone_prompt="",
            provenance_json={"prompt_source": {"kind": "composer"}},
            settings_json={},
        )
        session.add(run)
        session.flush()
        revision = ResponseRevision(
            id="rev_probe",
            message_id=assistant.id,
            run_id=run.id,
            sequence=1,
            status="complete",
        )
        session.add(revision)
        session.flush()
        assistant.active_response_revision_id = revision.id
        session.commit()

    preview = await client.get("/api/messages/msg_probe_user/removal-impact")
    assert preview.status_code == 200
    removed = await client.post(
        "/api/messages/msg_probe_user/remove-content",
        json={
            "expected_message_id": "msg_probe_user",
            "expected_revision_id": preview.json()["message_revision_id"],
            "operation_key": "probe-regen",
        },
    )
    assert removed.status_code == 200

    try:
        regenerated = await client.post(
            "/api/messages/msg_probe_assistant/regenerate",
            json={"settings": {}},
        )
    except Exception as error:  # noqa: BLE001 - the probe is asking what escapes
        print("PROBE: regenerate raised", type(error).__name__)
        raise AssertionError("regenerate raised instead of refusing") from error
    print("PROBE: regenerate status", regenerated.status_code)
    print("PROBE: body", regenerated.text[:200])
    assert regenerated.status_code == 409, regenerated.status_code
