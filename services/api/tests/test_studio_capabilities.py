"""A tool says what it cannot do before the work, not after it."""

from __future__ import annotations

from typing import Any

from local_lm.studio_capabilities import tool_capabilities

MASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"mask": {"type": "object", "x-lm-atelier-kind": "mask"}},
}
PLAIN_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"denoise": {"type": "number"}}}


def _by_kind(schemas: list[dict[str, Any] | None]) -> dict[str, Any]:
    return {tool.kind: tool for tool in tool_capabilities(edit_input_schemas=schemas)}


def test_nothing_installed_leaves_every_tool_unavailable_with_a_reason() -> None:
    tools = _by_kind([])

    assert not any(tool.available for tool in tools.values())
    assert all(tool.reason for tool in tools.values())


def test_a_plain_editor_runs_instruct_but_not_a_selection() -> None:
    """The live case: an edit workflow that declares no mask input.

    Every selection tool was clickable and every masked apply was refused, so
    the refusal arrived after the drawing rather than before it.
    """
    tools = _by_kind([PLAIN_SCHEMA])

    assert tools["instruct"].available is True
    assert tools["brush"].available is False
    assert tools["brush"].workflow_class == "inpaint"
    assert "inpainting" in (tools["brush"].reason or "")


def test_one_mask_capable_workflow_enables_every_selection_tool() -> None:
    # Four ways to draw one mask: they stand or fall together.
    tools = _by_kind([PLAIN_SCHEMA, MASK_SCHEMA])

    assert all(tools[kind].available for kind in ("brush", "eraser", "rect", "lasso"))
    assert all(tools[kind].reason is None for kind in ("brush", "eraser", "rect", "lasso"))


def test_capability_follows_the_declaration_rather_than_the_shape() -> None:
    """A schema with a mask property that is not declared a mask is not one."""
    counterfeit: dict[str, Any] = {"type": "object", "properties": {"mask": {"type": "string"}}}

    assert _by_kind([counterfeit])["brush"].available is False


def test_a_workflow_without_a_schema_at_all_is_not_mask_capable() -> None:
    assert _by_kind([None])["brush"].available is False
    assert _by_kind([None])["instruct"].available is True


async def test_the_report_reads_what_is_installed_on_this_machine(client) -> None:
    """Through the route, because the surface asks the route and not the module."""
    from local_lm.db import SessionLocal
    from local_lm.models import WorkflowDefinition, WorkflowRevision

    before = (await client.get("/api/studio/capabilities")).json()["tools"]
    assert {tool["kind"]: tool["available"] for tool in before}["brush"] is False

    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Inpainter", operation="image_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            api_graph_json={"nodes": []},
            input_schema_json=MASK_SCHEMA,
            dependencies_json={},
            trusted=True,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()

    after = (await client.get("/api/studio/capabilities")).json()["tools"]

    assert {tool["kind"]: tool["available"] for tool in after}["brush"] is True
