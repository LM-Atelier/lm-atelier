"""Removing one result of a batch generation leaves the batch alone.

The removal suite has seven tests and none of them involves a prompt expansion
batch, so the property that matters most here - that deleting one image of a
batch does not take its siblings or the batch with it - was asserted nowhere.

A batch is the case where that is easy to get wrong. Every result of one batch
shares a `WorkPlan`, and each carries an immutable `PromptExpansionItem` whose
rendered prompt and digest feed the batch's plan digest. A removal that reached
the item, or that widened along the shared plan, would silently change what the
batch says it produced.

The batch here is built by the STORE'S OWN `create_or_replay_expansion` rather
than by hand-written rows. A hand-built batch is a guess at the shape, and
that guess is easy to get wrong in both directions - inventing a column the
schema does not have, or omitting a NOT NULL it requires. Using the real
writer means the fixture cannot drift from the schema it is testing.
"""

from __future__ import annotations

from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import (
    Artifact,
    Chat,
    Message,
    MessagePart,
    PromptExpansionBatch,
    PromptExpansionItem,
    PromptTemplateDefinition,
    PromptTemplateRevision,
    ResponseRevision,
    Run,
    WorkPlan,
    WorkStep,
)
from local_lm.prompt_expansion import expand_prompt_template, parse_expansion_request
from local_lm.prompt_expansion_store import create_or_replay_expansion
from local_lm.prompt_templates import (
    parse_prompt_template_contract,
    prompt_template_contract_payload,
    prompt_template_contract_sha256,
)

CONTRACT = {
    "schema_version": 1,
    "operation": "text_to_image",
    "body": "a {{mood}} portrait",
    "slots": [
        {
            "name": "mood",
            "mode": "choice",
            "variation_scope": "item",
            "choices": ["calm", "bright"],
        }
    ],
    "resource_policy": {"mode": "inherited"},
}


def _request(message_id: str, revision_id: str, operation_key: str) -> dict[str, str]:
    return {
        "expected_message_id": message_id,
        "expected_revision_id": revision_id,
        "operation_key": operation_key,
    }


def _seed_two_result_batch() -> str:
    """One real batch of two, each item wired to its own assistant result."""

    with SessionLocal() as session:
        chat = Chat(id="chat_batch_remove", title="Batch chat")
        definition = PromptTemplateDefinition(
            id="ptdef_batch_remove", name="Batch template", description=""
        )
        session.add_all([chat, definition])
        session.flush()

        contract = parse_prompt_template_contract(CONTRACT)
        digest = prompt_template_contract_sha256(contract)
        revision = PromptTemplateRevision(
            id="ptrev_batch_remove",
            prompt_template_id=definition.id,
            version=1,
            schema_version=1,
            contract_json=prompt_template_contract_payload(contract),
            contract_sha256=digest,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id

        request = parse_expansion_request(
            {
                "definition_id": definition.id,
                "revision_id": revision.id,
                "contract_sha256": digest,
                "item_count": 2,
                "selection_seed": 412,
                "inputs": {},
            }
        )
        stored = create_or_replay_expansion(
            session,
            chat.id,
            "batch-remove-key",
            request,
            expand_prompt_template(contract, request),
            {"version": 1, "kind": "deterministic"},
        )
        batch_id = stored.batch.id

        prompt = Message(id="msg_batch_prompt", chat=chat, role="user", status="complete")
        results = [
            Message(
                id=f"msg_batch_{ordinal}",
                chat=chat,
                parent_id=prompt.id,
                role="assistant",
                status="complete",
            )
            for ordinal in (1, 2)
        ]
        session.add_all([prompt, *results])
        session.flush()

        plan = WorkPlan(
            id="plan_batch_remove",
            chat_id=chat.id,
            status="complete",
            transcript_sequence=1,
        )
        session.add(plan)
        session.flush()
        items = session.scalars(
            select(PromptExpansionItem)
            .where(PromptExpansionItem.batch_id == batch_id)
            .order_by(PromptExpansionItem.ordinal)
        ).all()
        assert len(items) == 2, f"the store wrote {len(items)} items, expected 2"

        for item, message in zip(items, results, strict=True):
            ordinal = item.ordinal
            step = WorkStep(
                id=f"step_batch_{ordinal}",
                plan_id=plan.id,
                ordinal=ordinal,
                operation="text_to_image",
                status="complete",
                prompt=item.reviewed_prompt,
                settings_json={},
                input_bindings_json=[],
                output_contract_json=[],
            )
            session.add(step)
            session.flush()
            run = Run(
                id=f"run_batch_{ordinal}",
                chat_id=chat.id,
                user_message_id=prompt.id,
                assistant_message_id=message.id,
                work_plan_id=plan.id,
                work_step_id=step.id,
                status="complete",
                standalone_prompt=item.reviewed_prompt,
                provenance_json={},
                settings_json={},
            )
            session.add(run)
            session.flush()
            response = ResponseRevision(
                id=f"rev_batch_{ordinal}",
                message_id=message.id,
                run_id=run.id,
                sequence=1,
                status="complete",
            )
            session.add(response)
            session.flush()
            message.active_response_revision_id = response.id
            artifact = Artifact(
                id=f"artifact_batch_{ordinal}",
                sha256=str(ordinal) * 64,
                kind="image",
                media_type="image/png",
                size_bytes=10,
                relative_path=f"artifacts/batch-{ordinal}.png",
                metadata_json={},
            )
            session.add(artifact)
            session.add(
                MessagePart(
                    id=f"part_batch_{ordinal}",
                    message_id=message.id,
                    position=0,
                    type="image",
                    artifact_id=artifact.id,
                    metadata_json={},
                )
            )
        session.commit()
    return batch_id


async def test_removing_one_batch_result_leaves_its_siblings_and_the_batch(
    client: AsyncClient,
) -> None:
    """Deleting one batch result leaves its siblings and the batch intact."""

    batch_id = _seed_two_result_batch()
    with SessionLocal() as session:
        before = [
            (item.ordinal, item.reviewed_prompt, item.reviewed_sha256)
            for item in session.scalars(
                select(PromptExpansionItem)
                .where(PromptExpansionItem.batch_id == batch_id)
                .order_by(PromptExpansionItem.ordinal)
            ).all()
        ]

    preview = await client.get("/api/messages/msg_batch_1/removal-impact")
    assert preview.status_code == 200, preview.text
    removed = await client.post(
        "/api/messages/msg_batch_1/remove-content",
        json=_request("msg_batch_1", preview.json()["message_revision_id"], "batch-remove-first"),
    )
    assert removed.status_code == 200, removed.text

    with SessionLocal() as session:
        first = session.get(Message, "msg_batch_1")
        second = session.get(Message, "msg_batch_2")
        assert first is not None and second is not None
        assert first.content_removed_at is not None, "the target was not removed"
        assert second.content_removed_at is None, "removal widened to the sibling"

        holders = {part.message_id for part in session.scalars(select(MessagePart)).all()}
        assert "msg_batch_2" in holders, "the sibling lost its image"
        assert "msg_batch_1" not in holders, "the target kept its image"

        assert session.get(PromptExpansionBatch, batch_id) is not None, (
            "removing one result deleted the whole batch"
        )
        after = [
            (item.ordinal, item.reviewed_prompt, item.reviewed_sha256)
            for item in session.scalars(
                select(PromptExpansionItem)
                .where(PromptExpansionItem.batch_id == batch_id)
                .order_by(PromptExpansionItem.ordinal)
            ).all()
        ]
        # The items are immutable evidence of what was planned. Removing a
        # RESULT must not edit the record of the prompt that produced it.
        assert after == before, "the immutable batch items changed"


async def test_a_batch_reads_back_consistently_after_one_result_is_removed(
    client: AsyncClient,
) -> None:
    """A removed result must not make the batch unreadable or inconsistent.

    Separate from the test above on purpose: that one proves the rows survive,
    this one proves the READ path still agrees with them. A batch that keeps
    its rows but can no longer be fetched is just as broken to the owner.
    """

    batch_id = _seed_two_result_batch()

    preview = await client.get("/api/messages/msg_batch_2/removal-impact")
    assert preview.status_code == 200, preview.text
    removed = await client.post(
        "/api/messages/msg_batch_2/remove-content",
        json=_request("msg_batch_2", preview.json()["message_revision_id"], "batch-remove-second"),
    )
    assert removed.status_code == 200, removed.text

    read = await client.get(f"/api/prompt-batches/{batch_id}")
    assert read.status_code == 200, read.text
    body = read.json()
    assert body["requested_count"] == 2
    assert [item["ordinal"] for item in body["items"]] == [1, 2]
    assert all(item["reviewed_prompt"] for item in body["items"])
