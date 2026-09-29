"""The kept extension takes every source pixel from the prepared source."""

from __future__ import annotations

import hashlib
from io import BytesIO

import pytest
from PIL import Image, PngImagePlugin

from local_lm.output_origin import record_for, stated_origin
from local_lm.source_fit_image import PreparedSourceImage, SourceFitImageRecord
from local_lm.source_fit_recipe import SourceExtensionRecipe
from local_lm.source_fit_restore import SourceRestore, restore_source, restores

SOURCE_COLOUR = (200, 40, 10)
FILL_COLOUR = (30, 90, 160)


def png(image: Image.Image, **options: object) -> bytes:
    content = BytesIO()
    image.save(content, format="PNG", **options)
    return content.getvalue()


def restore(width: int = 20, height: int = 12, source: tuple[int, int] = (8, 6)) -> SourceRestore:
    with Image.new("RGB", source, SOURCE_COLOUR) as image:
        content = png(image)
    digest = hashlib.sha256(content).hexdigest()
    prepared = PreparedSourceImage(
        source_artifact_id="sha256:" + "1" * 64,
        source_sha256="1" * 64,
        sha256=digest,
        width=source[0],
        height=source[1],
        content=content,
    )
    recipe = SourceExtensionRecipe(
        image=SourceFitImageRecord(
            source_artifact_id="sha256:" + "1" * 64,
            prepared_artifact_id=f"sha256:{digest}",
            width=source[0],
            height=source[1],
        ),
        canvas_width=width,
        canvas_height=height,
        save_node_id="save",
    )
    return SourceRestore(recipe, prepared)


def extension(size: tuple[int, int] = (20, 12), **options: object) -> bytes:
    """A workflow picture whose source rectangle came back altered."""
    with Image.new("RGB", size, FILL_COLOUR) as image:
        return png(image, **options)


def pixels(content: bytes, box: tuple[int, int, int, int]) -> set[tuple[int, ...]]:
    with Image.open(BytesIO(content)) as image:
        raw = image.crop(box).tobytes()
        width = len(image.getbands())
    return {tuple(raw[index : index + width]) for index in range(0, len(raw), width)}


def test_an_altered_source_is_put_back_and_the_canvas_kept() -> None:
    restored = restore_source(restore(), extension())
    assert restored is not None
    # Margins centre an 8x6 source on 20x12: six on each side, three above and below.
    assert pixels(restored.content, (6, 3, 14, 9)) == {SOURCE_COLOUR}
    assert pixels(restored.content, (0, 0, 20, 3)) == {FILL_COLOUR}
    with Image.open(BytesIO(restored.content)) as image:
        assert image.size == (20, 12)
    assert (restored.record["left"], restored.record["top"]) == (6, 3)
    assert restored.record["result_resampler"] is None


def test_a_source_already_in_place_keeps_the_workflow_bytes() -> None:
    with Image.new("RGB", (20, 12), FILL_COLOUR) as image:
        with Image.new("RGB", (8, 6), SOURCE_COLOUR) as source:
            image.paste(source, (6, 3))
        content = png(image)
    assert restore_source(restore(), content) is None


def test_a_canvas_the_vae_rounded_down_is_resized_back_before_the_source() -> None:
    restored = restore_source(restore(), extension((16, 12)))
    assert restored is not None
    assert restored.record["result_width"] == 16
    assert restored.record["result_resampler"] == "lanczos"
    with Image.open(BytesIO(restored.content)) as image:
        assert image.size == (20, 12)
    assert pixels(restored.content, (6, 3, 14, 9)) == {SOURCE_COLOUR}


def test_transparency_outside_the_source_survives_the_restore() -> None:
    with Image.new("RGBA", (20, 12), (*FILL_COLOUR, 0)) as image:
        content = png(image)
    restored = restore_source(restore(), content)
    assert restored is not None
    assert pixels(restored.content, (6, 3, 14, 9)) == {(*SOURCE_COLOUR, 255)}
    assert pixels(restored.content, (0, 0, 20, 3)) == {(*FILL_COLOUR, 0)}


def gamma_png() -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add(b"gAMA", (45455).to_bytes(4, "big"))
    return extension(pnginfo=info)


def jpeg() -> bytes:
    content = BytesIO()
    with Image.new("RGB", (20, 12), FILL_COLOUR) as image:
        image.save(content, format="JPEG")
    return content.getvalue()


@pytest.mark.parametrize(
    "value,content",
    [
        pytest.param(restore(), extension((21, 12)), id="larger-than-the-canvas"),
        pytest.param(
            restore(100, 80, source=(64, 48)),
            extension((100, 16)),
            id="shorter-than-any-vae-rounding",
        ),
        pytest.param(restore(), extension()[:-8], id="unreadable"),
        pytest.param(restore(), jpeg(), id="jpeg"),
        pytest.param(restore(), gamma_png(), id="gamma"),
    ],
)
def test_a_picture_the_source_cannot_be_placed_on_exactly_is_left_alone(
    value: SourceRestore, content: bytes
) -> None:
    assert restore_source(value, content) is None


@pytest.mark.parametrize(
    "origin,expected",
    [
        (stated_origin("save", "output", "images"), True),
        (stated_origin("another-save", "output", "images"), False),
        (stated_origin("save", "temp", "images"), False),
        (None, False),
    ],
)
def test_only_the_recipes_own_saved_picture_is_restored(origin: object, expected: bool) -> None:
    assert restores(restore(), record_for(origin, "comfyui")) is expected
