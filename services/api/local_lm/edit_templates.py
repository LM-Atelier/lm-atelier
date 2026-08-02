"""One-click edit templates: named instruction scaffolds over edit workflows.

A template is data. Applying one composes an ordinary image turn from its
instruction and settings, so the pixel pre-check, verification, retries, and
revision cycling behave exactly as they do for a hand-written edit. Nothing
here executes anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import EditTemplate

SUBJECT_SLOT = "{subject}"
MAX_SUBJECT_CHARACTERS = 2_000


@dataclass(frozen=True)
class SeedTemplate:
    name: str
    description: str
    instruction: str
    settings: dict[str, float | int | str]


# Every seed rides the existing image-to-image workflow; templates that need
# new workflows (upscaling, inpainting, background replacement) arrive with
# those workflows rather than promising them early. Strength values lean
# conservative: a template's first result should look like the source, edited,
# not a regeneration that happens to share a subject.
SEED_TEMPLATES: tuple[SeedTemplate, ...] = (
    SeedTemplate(
        "Watercolor painting",
        "Repaint the photo as a soft watercolor.",
        "Transform this image into a watercolor painting with soft washes, "
        "visible paper texture, and gentle color bleeding. Keep the composition "
        "and subjects exactly as they are.{subject}",
        {"strength": 0.55},
    ),
    SeedTemplate(
        "Oil painting",
        "Repaint the photo with oil-paint texture and brushwork.",
        "Transform this image into a classical oil painting with visible "
        "brushstrokes and rich, layered color. Keep the composition and "
        "subjects exactly as they are.{subject}",
        {"strength": 0.55},
    ),
    SeedTemplate(
        "Pencil sketch",
        "Redraw the photo as a graphite pencil sketch.",
        "Redraw this image as a detailed graphite pencil sketch with natural "
        "shading and clean linework on white paper. Keep the composition and "
        "subjects exactly as they are.{subject}",
        {"strength": 0.6},
    ),
    SeedTemplate(
        "Anime style",
        "Restyle the photo as anime artwork.",
        "Restyle this image as high-quality anime artwork with clean lines and "
        "cel shading. Keep the composition, poses, and subjects exactly as "
        "they are.{subject}",
        {"strength": 0.6},
    ),
    SeedTemplate(
        "Colorize",
        "Add realistic color to a black-and-white photo.",
        "Colorize this black-and-white photograph with realistic, historically "
        "plausible colors. Change nothing except adding color.{subject}",
        {"strength": 0.4},
    ),
    SeedTemplate(
        "Restore old photo",
        "Repair damage and fading in an old photograph.",
        "Restore this old photograph: repair scratches, tears, and spots, "
        "reduce fading, and recover natural contrast. Do not alter faces, "
        "clothing, or the scene itself.{subject}",
        {"strength": 0.35},
    ),
    SeedTemplate(
        "Golden hour relight",
        "Relight the scene with warm late-afternoon sun.",
        "Relight this image with warm golden-hour sunlight, long soft shadows, "
        "and a gentle glow. Keep every object and person exactly where and as "
        "they are.{subject}",
        {"strength": 0.45},
    ),
    SeedTemplate(
        "Remove text",
        "Remove visible text, captions, or watermarks.",
        "Remove all visible text, captions, logos, and watermarks from this "
        "image, filling the space with a natural continuation of the "
        "background. Change nothing else.{subject}",
        {"strength": 0.4},
    ),
)


def seed_edit_templates(session: Session) -> int:
    """Insert missing built-in templates; never touch user-saved rows.

    Matching is by name against builtin rows only, so renaming or disabling a
    built-in stays sticky across restarts and a user template with the same
    name as a future seed is left alone (the seed is skipped instead).
    """

    existing = {template.name for template in session.scalars(select(EditTemplate)).all()}
    added = 0
    for seed in SEED_TEMPLATES:
        if seed.name in existing:
            continue
        session.add(
            EditTemplate(
                name=seed.name,
                description=seed.description,
                instruction=seed.instruction,
                operation="image_to_image",
                settings_json=dict(seed.settings),
                trigger_words_json=[],
                content_rating="general",
                builtin=True,
                enabled=True,
            )
        )
        added += 1
    return added


def render_instruction(template: EditTemplate, subject: str = "") -> str:
    """Splice the optional user addition into the template's instruction.

    The subject is user text inside a fixed instruction: it extends what the
    edit should include, it cannot rewrite the instruction's frame. Bounded so
    a template application can never exceed an ordinary prompt's size.
    """

    addition = subject.strip()[:MAX_SUBJECT_CHARACTERS]
    rendered = template.instruction
    if SUBJECT_SLOT in rendered:
        return rendered.replace(SUBJECT_SLOT, f" {addition}" if addition else "", 1)
    return f"{rendered} {addition}".strip() if addition else rendered
