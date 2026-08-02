"""One-click edit templates: seeding, listing, and instruction rendering."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.edit_templates import (
    SEED_TEMPLATES,
    render_instruction,
    seed_edit_templates,
)
from local_lm.models import EditTemplate

pytestmark = pytest.mark.asyncio


async def test_seeded_templates_are_listed_ready_to_apply(client: AsyncClient) -> None:
    response = await client.get("/api/edit-templates")

    assert response.status_code == 200
    payload = response.json()
    names = [template["name"] for template in payload]
    assert "Watercolor painting" in names
    assert "Restore old photo" in names
    watercolor = next(t for t in payload if t["name"] == "Watercolor painting")
    assert watercolor["operation"] == "image_to_image"
    assert watercolor["builtin"] is True
    assert watercolor["content_rating"] == "general"
    assert watercolor["settings_json"]["strength"] == 0.55


async def test_seeding_twice_adds_nothing_and_respects_disabling(client: AsyncClient) -> None:
    """Disabling a built-in survives restart; re-seeding never duplicates."""

    with SessionLocal() as session:
        assert seed_edit_templates(session) == 0
        watercolor = session.query(EditTemplate).filter_by(name="Watercolor painting").one()
        watercolor.enabled = False
        session.commit()

    with SessionLocal() as session:
        assert seed_edit_templates(session) == 0
        session.commit()

    response = await client.get("/api/edit-templates")
    names = [template["name"] for template in response.json()]
    assert "Watercolor painting" not in names
    assert "Oil painting" in names


async def test_every_seed_keeps_the_source_grounded() -> None:
    """Style transforms must edit the picture, not regenerate its subject."""

    preservation_markers = ("Keep ", "Change nothing", "Do not alter")
    for seed in SEED_TEMPLATES:
        assert any(marker in seed.instruction for marker in preservation_markers), seed.name
        strength = seed.settings.get("strength")
        assert isinstance(strength, float) and strength <= 0.6, seed.name


async def test_instruction_rendering_bounds_and_splices_the_subject() -> None:
    template = EditTemplate(
        name="t",
        instruction="Transform this image into a watercolor painting.{subject}",
        operation="image_to_image",
    )
    assert render_instruction(template) == "Transform this image into a watercolor painting."
    assert render_instruction(template, "Focus on the harbor.") == (
        "Transform this image into a watercolor painting. Focus on the harbor."
    )
    # The subject extends the instruction; it cannot grow without bound.
    assert len(render_instruction(template, "x" * 10_000)) < 3_000

    without_slot = EditTemplate(
        name="t2",
        instruction="Colorize this photograph.",
        operation="image_to_image",
    )
    assert render_instruction(without_slot, "Keep the sky pale.") == (
        "Colorize this photograph. Keep the sky pale."
    )
