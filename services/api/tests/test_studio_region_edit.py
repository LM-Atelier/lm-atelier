"""An edit placed back through its selection changes only what was selected."""

from __future__ import annotations

import hashlib
import io

import pytest
from PIL import Image, ImageDraw, ImageFilter

from local_lm import studio_region_edit
from local_lm.studio_masks import MaskSelection
from local_lm.studio_region_edit import (
    RegionEdit,
    RegionEditError,
    blend_through_selection,
    prepare_selection,
)

SOURCE_ID = f"sha256:{'b' * 64}"
MASK_ID = f"sha256:{'c' * 64}"
BOX = (16, 12, 40, 30)


def _pixel_channels(image: Image.Image, position: tuple[int, int]) -> tuple[int, ...]:
    pixel = image.getpixel(position)
    assert isinstance(pixel, tuple)
    return pixel


def _png(image: Image.Image, **options: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **options)
    return buffer.getvalue()


def _source(size: tuple[int, int] = (64, 48)) -> Image.Image:
    # A different colour in every pixel, so a misplaced or resampled pixel shows.
    picture = Image.new("RGB", size)
    picture.putdata(
        [
            ((x * 7) % 256, (y * 11) % 256, (x * y) % 256)
            for y in range(size[1])
            for x in range(size[0])
        ]
    )
    return picture


def _mask(
    size: tuple[int, int] = (64, 48),
    box: tuple[int, int, int, int] = BOX,
    blur: float = 0,
) -> bytes:
    alpha = Image.new("L", size, 0)
    scale = size[0] / 64
    ImageDraw.Draw(alpha).rectangle([round(value * scale) for value in box], fill=255)
    if blur:
        alpha = alpha.filter(ImageFilter.GaussianBlur(blur))
    # The browser's encoding: white, with the selection as alpha.
    mask = Image.new("RGBA", size, (255, 255, 255, 0))
    mask.putalpha(alpha)
    return _png(mask)


def _edit(
    source: bytes | None = None, mask: bytes | None = None, *, invert: bool = False
) -> RegionEdit:
    return RegionEdit(
        selection=MaskSelection(artifact_id=MASK_ID, feather_px=0, invert=invert, blend=True),
        source_artifact_id=SOURCE_ID,
        source=source if source is not None else _png(_source()),
        mask=mask if mask is not None else _mask(),
    )


def _pixels(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content))


def test_only_the_selected_pixels_take_the_edit() -> None:
    source = _source()
    # Returned at the model's own working size, as edit models do.
    edited = Image.new("RGB", (96, 72), (10, 200, 30))

    blend = blend_through_selection(_edit(_png(source)), _png(edited))

    out = _pixels(blend.content)
    assert blend.media_type == "image/png"
    assert out.format == "PNG"
    assert out.size == source.size
    for y in range(source.height):
        for x in range(source.width):
            inside = BOX[0] <= x <= BOX[2] and BOX[1] <= y <= BOX[3]
            expected = (10, 200, 30) if inside else source.getpixel((x, y))
            assert out.getpixel((x, y)) == expected, (x, y)


def test_a_feathered_edge_mixes_the_two_pictures() -> None:
    source = Image.new("RGB", (64, 48), (0, 0, 0))
    edited = Image.new("RGB", (64, 48), (255, 255, 255))

    out = _pixels(blend_through_selection(_edit(_png(source), _mask(blur=3)), _png(edited)).content)

    centre = out.getpixel((28, 21))
    edge = _pixel_channels(out, (16, 21))
    far = out.getpixel((2, 2))
    assert centre == (255, 255, 255)
    assert far == (0, 0, 0)
    assert 0 < edge[0] < 255


def test_an_inverted_selection_keeps_the_inside() -> None:
    source = _source()
    edited = Image.new("RGB", (64, 48), (10, 200, 30))

    out = _pixels(blend_through_selection(_edit(_png(source), invert=True), _png(edited)).content)

    assert out.getpixel((20, 20)) == source.getpixel((20, 20))
    assert out.getpixel((2, 2)) == (10, 200, 30)


def test_a_selection_drawn_smaller_is_resized_and_says_so() -> None:
    source = _source((128, 96))
    edited = Image.new("RGB", (128, 96), (10, 200, 30))

    blend = blend_through_selection(_edit(_png(source), _mask((64, 48))), _png(edited))

    geometry = blend.record["selection"]["geometry"]
    assert geometry["exact"] is False
    assert geometry["scale_x"] == 2.0
    assert geometry["resampler"] == "bilinear"
    out = _pixels(blend.content)
    assert out.getpixel((56, 42)) == (10, 200, 30)
    assert out.getpixel((4, 4)) == source.getpixel((4, 4))


def test_the_record_says_how_the_picture_was_made() -> None:
    result = _png(Image.new("RGB", (96, 72), (10, 200, 30)))

    record = blend_through_selection(_edit(), result).record

    assert record["mode"] == "blend"
    assert record["source_artifact_id"] == SOURCE_ID
    assert record["result_sha256"] == hashlib.sha256(result).hexdigest()
    assert (record["result_width"], record["result_height"]) == (96, 72)
    assert (record["width"], record["height"]) == (64, 48)
    assert record["result_resampler"] == "lanczos"
    selection = record["selection"]
    assert selection["artifact_id"] == MASK_ID
    assert selection["apply"] == "blend"
    expected_coverage = (25 * 19) / (64 * 48)
    assert selection["coverage"] == pytest.approx(expected_coverage)


def test_a_selection_on_a_different_shape_refuses_before_the_edit() -> None:
    with pytest.raises(RegionEditError) as raised:
        prepare_selection(_edit(mask=_mask((64, 64))))
    assert raised.value.code == "mask-aspect-mismatch"


def test_an_empty_selection_refuses_before_the_edit() -> None:
    empty = _png(Image.new("RGBA", (64, 48), (255, 255, 255, 0)))

    with pytest.raises(RegionEditError) as raised:
        prepare_selection(_edit(mask=empty))
    assert raised.value.code == "region-selection-empty"


def test_a_prepared_selection_reports_its_coverage() -> None:
    prepared = prepare_selection(_edit())

    assert prepared.geometry.is_exact is True
    assert prepared.coverage == pytest.approx((25 * 19) / (64 * 48))


def test_a_result_of_a_different_shape_cannot_be_placed_back() -> None:
    # Rounding to a patch size moves the shape by about one percent; that fits.
    rounded = _png(Image.new("RGB", (1184, 880), (10, 200, 30)))
    blend_through_selection(_edit(_png(_source((1024, 768))), _mask((1024, 768))), rounded)

    square = _png(Image.new("RGB", (64, 64), (10, 200, 30)))
    with pytest.raises(RegionEditError) as raised:
        blend_through_selection(_edit(), square)
    assert raised.value.code == "region-result-reshaped"


def test_the_source_is_used_the_way_up_it_was_seen() -> None:
    # Stored on its side with an orientation tag: the person saw, and selected
    # on, the upright 48 x 64 picture.
    stored = _source((64, 48))
    exif = Image.Exif()
    exif[0x0112] = 6
    source_bytes = _png(stored, exif=exif.tobytes())
    upright = Image.open(io.BytesIO(source_bytes))
    upright_pixels = upright.transpose(Image.Transpose.ROTATE_270)
    mask = _mask((48, 64), box=(0, 0, 64, 64))
    edited = Image.new("RGB", (48, 64), (10, 200, 30))
    half = Image.new("L", (48, 64), 0)
    ImageDraw.Draw(half).rectangle((0, 0, 47, 31), fill=255)
    top_half = Image.new("RGBA", (48, 64), (255, 255, 255, 0))
    top_half.putalpha(half)

    blend = blend_through_selection(_edit(source_bytes, _png(top_half)), _png(edited))

    out = _pixels(blend.content)
    assert out.size == (48, 64)
    assert blend.record["selection"]["geometry"]["orientation"] == 6
    assert out.getpixel((10, 10)) == (10, 200, 30)
    assert out.getpixel((10, 50)) == upright_pixels.getpixel((10, 50))
    assert prepare_selection(_edit(source_bytes, mask)).coverage > 0


def test_a_transparent_source_stays_transparent_where_it_was() -> None:
    source = Image.new("RGBA", (64, 48), (200, 20, 20, 0))
    ImageDraw.Draw(source).rectangle((20, 14, 36, 28), fill=(200, 20, 20, 255))
    edited = Image.new("RGB", (64, 48), (10, 200, 30))

    out = _pixels(blend_through_selection(_edit(_png(source)), _png(edited)).content)

    assert out.mode == "RGBA"
    assert out.getpixel((24, 20)) == (10, 200, 30, 255)
    assert _pixel_channels(out, (17, 13))[3] == 0
    assert out.getpixel((2, 2)) == (200, 20, 20, 0)


def test_the_source_colour_profile_is_kept() -> None:
    profile = b"neutral-colour-profile"
    source_bytes = _png(_source(), icc_profile=profile)

    blend = blend_through_selection(
        _edit(source_bytes), _png(Image.new("RGB", (64, 48), (1, 2, 3)))
    )

    assert _pixels(blend.content).info.get("icc_profile") == profile


def test_an_unreadable_result_refuses() -> None:
    with pytest.raises(RegionEditError) as raised:
        blend_through_selection(_edit(), b"not a picture")
    assert raised.value.code == "region-image-unreadable"


def test_a_picture_past_the_pixel_bound_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(studio_region_edit, "MAX_BLEND_PIXELS", 64 * 48 - 1)

    with pytest.raises(RegionEditError) as raised:
        prepare_selection(_edit())
    assert raised.value.code == "region-image-too-large"
