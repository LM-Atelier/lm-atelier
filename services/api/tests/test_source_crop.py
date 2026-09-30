"""Exact source viewports and real bounded crop pixels, without generation authority."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from fractions import Fraction
from io import BytesIO

import pytest
from PIL import Image
from sqlalchemy.orm import Session
from test_source_fit_image import artifact_session as artifact_session
from test_source_fit_image import ingest, png

from local_lm.artifacts import ArtifactStore
from local_lm.source_fit_image import prepare_source_fit_image


@pytest.mark.parametrize(
    "source,target,rectangle,scale",
    [
        ((400, 300), (1600, 900), (0, Fraction(75, 2), 400, 225), Fraction(4)),
        ((400, 300), (900, 1600), (Fraction(925, 8), 0, Fraction(675, 4), 300), Fraction(16, 3)),
        ((1600, 900), (400, 300), (200, 0, 1200, 900), Fraction(1, 3)),
        ((900, 1600), (1600, 900), (0, Fraction(4375, 8), 900, Fraction(2025, 4)), Fraction(16, 9)),
        ((3, 2), (16, 9), (0, Fraction(5, 32), 3, Fraction(27, 16)), Fraction(16, 3)),
        ((400, 300), (800, 600), (0, 0, 400, 300), Fraction(2)),
    ],
)
def test_crop_viewport_keeps_one_exact_uniform_scale(
    source: tuple[int, int],
    target: tuple[int, int],
    rectangle: tuple[Fraction | int, ...],
    scale: Fraction,
) -> None:
    from local_lm.source_crop import plan_source_crop

    plan = plan_source_crop(*source, *target)
    assert plan.rectangle == rectangle
    assert plan.scale == scale
    assert plan.source_size == source and plan.canvas_size == target
    assert plan.rectangle[2] * plan.scale == target[0]
    assert plan.rectangle[3] * plan.scale == target[1]


def test_crop_record_keeps_fractional_edges_without_rounding_to_whole_pixels() -> None:
    from local_lm.source_crop import plan_source_crop

    assert plan_source_crop(400, 300, 1600, 900).payload() == {
        "v": 1,
        "mode": "crop",
        "source": {"width": 400, "height": 300},
        "canvas": {"width": 1600, "height": 900},
        "rectangle": {
            "left": [0, 1],
            "top": [75, 2],
            "width": [400, 1],
            "height": [225, 1],
        },
        "sample_bounds": {"left": 0, "top": 37, "width": 400, "height": 226},
        "scale": [4, 1],
        "resampler": "lanczos",
        "coordinate_convention": "pixel-edges-v1",
    }


@pytest.mark.parametrize(
    "dimensions",
    [
        (True, 12, 16, 8),
        (16, 12, 16.0, 8),
        (16, 0, 16, 8),
        (16, 12, 0, 8),
        (16, 12, 1_000_001, 1),
        (16, 12, 4097, 4096),
        (4097, 4096, 16, 8),
    ],
)
def test_crop_preparation_refuses_invalid_or_unbounded_allocations(
    dimensions: tuple[object, ...],
) -> None:
    from local_lm.source_crop import plan_source_crop

    with pytest.raises(ValueError, match="source_crop_dimensions"):
        plan_source_crop(*dimensions)


def test_integer_crop_samples_exactly_the_recorded_source_rows(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_crop import prepare_source_crop

    raw = bytes(
        component
        for index in range(16 * 12)
        for component in (index % 256, (index * 37) % 256, 255 - index % 256)
    )
    with Image.frombytes("RGB", (16, 12), raw) as image:
        data = BytesIO()
        image.save(data, format="PNG")
    store, session = artifact_session
    artifact = ingest(artifact_session, data.getvalue())
    original = store.resolve(artifact).read_bytes()
    source = prepare_source_fit_image(store, artifact)
    before = set(store.root.rglob("*"))
    cropped = prepare_source_crop(source, 16, 8)
    assert cropped.plan.rectangle == (Fraction(0), Fraction(2), Fraction(16), Fraction(8))
    assert cropped.source is source
    assert cropped.sha256 == hashlib.sha256(cropped.content).hexdigest()
    with Image.open(BytesIO(cropped.content)) as image:
        image.load()
        assert image.size == (16, 8) and image.mode == "RGB"
        assert image.tobytes() == raw[2 * 16 * 3 : 10 * 16 * 3]
        assert not image.info
    assert store.resolve(artifact).read_bytes() == original
    assert set(store.root.rglob("*")) == before
    assert not session.new


def test_fractional_crop_of_an_oriented_source_retains_its_exact_source_binding(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_crop import prepare_source_crop

    store, _ = artifact_session
    artifact = ingest(artifact_session, png(orientation=6))
    source = prepare_source_fit_image(store, artifact)
    cropped = prepare_source_crop(source, 9, 16)
    assert cropped.source.source_artifact_id == artifact.id
    assert cropped.plan.source_size == (2, 3)
    assert cropped.plan.rectangle == (Fraction(5, 32), Fraction(0), Fraction(27, 16), Fraction(3))
    assert cropped.plan.scale == Fraction(16, 3)
    assert prepare_source_crop(source, 9, 16).content == cropped.content
    with Image.open(BytesIO(cropped.content)) as image:
        assert image.size == (9, 16) and image.mode == "RGB"
        assert not image.getexif() and not image.info


def test_crop_refuses_a_changed_verified_buffer_before_rendering(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_crop import prepare_source_crop

    store, _ = artifact_session
    artifact = ingest(artifact_session, png())
    source = prepare_source_fit_image(store, artifact)
    with pytest.raises(ValueError, match="source_crop_source_binding"):
        prepare_source_crop(replace(source, content=source.content + b"changed"), 16, 9)


def test_pixels_outside_sample_bounds_cannot_bleed_into_the_resized_crop(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    from local_lm.source_crop import prepare_source_crop

    blue, red = (0, 40, 200), (255, 0, 0)
    pixels = bytes(
        channel for y in range(8) for _x in range(8) for channel in (blue if 2 <= y < 6 else red)
    )
    with Image.frombytes("RGB", (8, 8), pixels) as image:
        output = BytesIO()
        image.save(output, format="PNG")
    store, _ = artifact_session
    source = prepare_source_fit_image(store, ingest(artifact_session, output.getvalue()))
    # The unchecked box alone lets the filter sample red beyond the desired crop.
    with (
        Image.open(BytesIO(source.content)) as original,
        original.resize((16, 8), Image.Resampling.LANCZOS, box=(0, 2, 8, 6)) as unchecked,
    ):
        assert unchecked.tobytes() != bytes(blue) * (16 * 8)
    cropped = prepare_source_crop(source, 16, 8)
    assert cropped.plan.sample_bounds == (0, 2, 8, 4)
    with Image.open(BytesIO(cropped.content)) as image:
        assert image.tobytes() == bytes(blue) * (16 * 8)
