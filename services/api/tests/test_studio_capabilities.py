"""A tool says what it cannot do before the work, not after it."""

from __future__ import annotations

from typing import Any

from httpx2 import AsyncClient

from local_lm.schemas import StudioToolCapability
from local_lm.studio_capabilities import TOOL_WORKFLOW_CLASSES, tool_capabilities

MASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"mask": {"type": "object", "x-lm-atelier-kind": "mask"}},
}
PLAIN_SCHEMA: dict[str, Any] = {"type": "object", "properties": {"denoise": {"type": "number"}}}
UPSCALE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"upscale_factor": {"type": "number", "x-lm-atelier-kind": "upscale"}},
}


def _by_kind(schemas: list[dict[str, Any] | None]) -> dict[str, Any]:
    return {tool.kind: tool for tool in tool_capabilities(edit_input_schemas=schemas)}


def test_nothing_installed_leaves_every_tool_unavailable_with_a_reason() -> None:
    tools = _by_kind([])
    needing = {kind: tool for kind, tool in tools.items() if tool.workflow_class != "local"}

    assert needing
    assert not any(tool.available for tool in needing.values())
    assert all(tool.reason for tool in needing.values())


def test_turning_and_flipping_need_nothing_installed() -> None:
    """The studio makes those edits itself, so nothing can be missing."""
    transform = _by_kind([])["transform"]

    assert transform.workflow_class == "local"
    assert transform.available is True
    assert transform.reason is None
    assert transform.workflow_revision_id is None


def test_enhance_waits_for_a_workflow_that_can_actually_enlarge() -> None:
    """Offering it over an ordinary editor would promise a size nothing delivers."""
    assert _by_kind([PLAIN_SCHEMA])["enhance"].available is False
    assert _by_kind([PLAIN_SCHEMA, UPSCALE_SCHEMA])["enhance"].available is True


def test_an_upscaler_does_not_enable_the_selection_tools() -> None:
    tools = _by_kind([UPSCALE_SCHEMA])

    assert tools["enhance"].available is True
    assert tools["brush"].available is False


def test_extend_waits_for_a_workflow_that_paints_past_the_edge() -> None:
    outpaint: dict[str, Any] = {
        "type": "object",
        "properties": {"outpaint_margins": {"type": "object", "x-lm-atelier-kind": "outpaint"}},
    }

    assert _by_kind([PLAIN_SCHEMA])["extend"].available is False
    assert _by_kind([outpaint])["extend"].available is True
    # Each class stands alone: an outpainter is not an upscaler.
    assert _by_kind([outpaint])["enhance"].available is False


def test_isolate_waits_for_a_workflow_that_says_it_cuts_a_subject_out() -> None:
    """An editor is not a matting workflow, and a matting workflow is not an editor's mask."""

    matting: dict[str, Any] = {
        "type": "object",
        "properties": {"matte": {"type": "boolean", "x-lm-atelier-kind": "matting"}},
    }

    assert _by_kind([PLAIN_SCHEMA])["isolate"].available is False
    assert _by_kind([matting])["isolate"].available is True
    # Each class stands alone, as with the outpainter above.
    assert _by_kind([matting])["brush"].available is False
    assert _by_kind([matting])["enhance"].available is False
    assert _by_kind([MASK_SCHEMA])["isolate"].available is False


def test_replacing_a_background_needs_a_cutout_and_an_editor() -> None:
    """The cutout finds the subject; an ordinary edit then redraws around it."""

    matting: dict[str, Any] = {
        "type": "object",
        "properties": {"matte": {"type": "boolean", "x-lm-atelier-kind": "matting"}},
    }
    alone = tool_capabilities(edit_input_schemas=[matting], matting_workflow_ids=["wfrev-matting"])
    background = next(tool for tool in alone if tool.kind == "background")
    assert background.available is False
    assert background.reason == "Install an image editing workflow to change a picture."
    assert _by_kind([PLAIN_SCHEMA])["background"].available is False
    assert "background removal" in (_by_kind([PLAIN_SCHEMA])["background"].reason or "")
    both = tool_capabilities(
        edit_input_schemas=[matting, PLAIN_SCHEMA], matting_workflow_ids=["wfrev-matting"]
    )
    ready = next(tool for tool in both if tool.kind == "background")
    assert ready.available is True and ready.reason is None
    # The cutout runs on the workflow the report names, as Isolate does.
    assert ready.workflow_revision_id == "wfrev-matting"


def test_replacing_a_subject_needs_a_cutout_and_an_editor_that_reads_a_second_picture() -> None:
    """The cutout finds the subject; the redraw takes the new one from a second picture."""

    matting: dict[str, Any] = {
        "type": "object",
        "properties": {"matte": {"type": "boolean", "x-lm-atelier-kind": "matting"}},
    }

    def subject(**kwargs: Any) -> Any:
        return next(tool for tool in tool_capabilities(**kwargs) if tool.kind == "subject")

    cutout_only = subject(edit_input_schemas=[matting, PLAIN_SCHEMA])
    assert cutout_only.available is False
    assert "takes a second picture" in (cutout_only.reason or "")
    editor_only = subject(edit_input_schemas=[PLAIN_SCHEMA], reference_workflow_ids=["wfrev-two"])
    assert editor_only.available is False
    assert "background removal" in (editor_only.reason or "")
    ready = subject(
        edit_input_schemas=[matting, PLAIN_SCHEMA],
        matting_workflow_ids=["wfrev-matting"],
        reference_workflow_ids=["wfrev-two", "wfrev-three"],
    )
    assert ready.available is True and ready.reason is None
    # Always the first that reads a second picture: the studio's own choice
    # may read only one.
    assert ready.workflow_revision_id == "wfrev-two"


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


def test_replacing_words_needs_an_edit_workflow_but_no_mask_input() -> None:
    """The selection is placed back after the edit, so the workflow never takes one."""
    assert _by_kind([PLAIN_SCHEMA])["text"].available is True
    assert _by_kind([PLAIN_SCHEMA])["text"].workflow_class == "image_to_image"
    assert _by_kind([])["text"].available is False
    assert "image editing workflow" in (_by_kind([])["text"].reason or "")


def test_one_mask_capable_workflow_enables_every_selection_tool() -> None:
    # Four ways to draw one mask: they stand or fall together.
    tools = _by_kind([PLAIN_SCHEMA, MASK_SCHEMA])

    selections = ("brush", "eraser", "rect", "lasso", "bucket", "wand")
    assert all(tools[kind].available for kind in selections)
    assert all(tools[kind].reason is None for kind in selections)


def test_capability_follows_the_declaration_rather_than_the_shape() -> None:
    """A schema with a mask property that is not declared a mask is not one."""
    counterfeit: dict[str, Any] = {"type": "object", "properties": {"mask": {"type": "string"}}}

    assert _by_kind([counterfeit])["brush"].available is False


def test_a_workflow_without_a_schema_at_all_is_not_mask_capable() -> None:
    assert _by_kind([None])["brush"].available is False
    assert _by_kind([None])["instruct"].available is True


async def test_the_report_reads_what_is_installed_on_this_machine(
    client: AsyncClient,
) -> None:
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


def test_studio_capability_kinds_match_the_tools_the_server_reports() -> None:
    schema = StudioToolCapability.model_json_schema()
    assert set(schema["properties"]["kind"].get("enum", [])) == set(TOOL_WORKFLOW_CLASSES)
