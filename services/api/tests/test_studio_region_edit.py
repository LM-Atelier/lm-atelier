"""An edit placed back through its selection changes only what was selected."""

from __future__ import annotations

import hashlib
import io
import random

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from local_lm import studio_region_edit
from local_lm.studio_masks import MaskSelection
from local_lm.studio_region_edit import (
    RegionEdit,
    RegionEditError,
    blend_through_selection,
    crop_selection,
    prepare_selection,
    stitch_selection,
    workflow_selection_mask,
)
from local_lm.workflow_source_geometry import SelectionRoute

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


def _opaque_mask(fill: tuple[int, int, int] = (0, 0, 0)) -> bytes:
    """A mask opaque everywhere, its selection drawn white over the rest."""
    mask = Image.new("RGBA", (64, 48), (*fill, 255))
    ImageDraw.Draw(mask).rectangle(BOX, fill=(255, 255, 255, 255))
    return _png(mask)


def _greyscale_mask() -> bytes:
    """A mask with no alpha at all, its selection drawn white on black."""
    mask = Image.new("L", (64, 48), 0)
    ImageDraw.Draw(mask).rectangle(BOX, fill=255)
    return _png(mask)


@pytest.mark.parametrize(
    "mask",
    [
        pytest.param(_mask(), id="in-alpha"),
        pytest.param(_opaque_mask(), id="white-on-opaque-black"),
        pytest.param(_greyscale_mask(), id="greyscale"),
    ],
)
def test_a_selection_drawn_in_transparency_or_in_brightness_selects_the_same_part(
    mask: bytes,
) -> None:
    """An opaque mask carries nothing in its alpha, so its brightness says what is selected."""
    source = _source()
    edited = Image.new("RGB", (64, 48), (10, 200, 30))

    out = _pixels(blend_through_selection(_edit(_png(source), mask), _png(edited)).content)

    for y in range(48):
        for x in range(64):
            inside = BOX[0] <= x <= BOX[2] and BOX[1] <= y <= BOX[3]
            expected = (10, 200, 30) if inside else source.getpixel((x, y))
            assert out.getpixel((x, y)) == expected, (x, y)


def test_an_opaque_mask_selects_all_of_a_white_picture_and_none_of_a_black_one() -> None:
    whole = _png(Image.new("RGBA", (64, 48), (255, 255, 255, 255)))
    assert prepare_selection(_edit(mask=whole)).coverage == 1
    with pytest.raises(RegionEditError) as raised:
        prepare_selection(_edit(mask=_png(Image.new("RGBA", (64, 48), (0, 0, 0, 255)))))
    assert raised.value.code == "region-selection-empty"


def _handed(edit: RegionEdit) -> Image.Image:
    content = workflow_selection_mask(edit)
    assert content is not None
    return _pixels(content)


@pytest.mark.parametrize("mask", [_mask(), _opaque_mask()], ids=["in-alpha", "opaque"])
@pytest.mark.parametrize("invert", [False, True], ids=["as-drawn", "turned-round"])
def test_a_workflow_is_handed_the_selection_in_its_brightness_and_its_alpha(
    mask: bytes, invert: bool
) -> None:
    """However it was drawn, a workflow reading either brightness or alpha finds the selection."""
    handed = _handed(_edit(mask=mask, invert=invert))

    assert handed.format == "PNG"
    assert handed.mode == "RGBA"
    assert handed.size == (64, 48)
    for y in range(48):
        for x in range(64):
            inside = BOX[0] <= x <= BOX[2] and BOX[1] <= y <= BOX[3]
            selected = 255 if inside != invert else 0
            assert handed.getpixel((x, y)) == (selected,) * 4, (x, y)


def test_a_workflow_selection_is_checked_against_the_upright_source() -> None:
    stored = _source((64, 48))
    exif = Image.Exif()
    exif[0x0112] = 6
    half = Image.new("L", (48, 64), 0)
    ImageDraw.Draw(half).rectangle((0, 0, 47, 31), fill=255)
    top_half = Image.new("RGBA", (48, 64), (255, 255, 255, 0))
    top_half.putalpha(half)

    handed = _handed(_edit(_png(stored, exif=exif.tobytes()), _png(top_half)))

    assert handed.size == (48, 64)
    assert handed.getchannel("A").getpixel((10, 10)) == 255
    assert handed.getchannel("A").getpixel((10, 50)) == 0
    # Drawn the stored way round, it is drawn on another picture than the one seen.
    with pytest.raises(RegionEditError):
        workflow_selection_mask(_edit(_png(stored, exif=exif.tobytes()), _mask((64, 48))))


def test_a_workflow_selection_reads_only_the_source_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A picture too large to blend can still be edited by a workflow through a selection."""
    monkeypatch.setattr(studio_region_edit, "MAX_BLEND_PIXELS", 100)
    with pytest.raises(RegionEditError):
        prepare_selection(_edit(mask=_opaque_mask()))

    assert workflow_selection_mask(_edit(mask=_opaque_mask())) is not None


def test_a_selection_too_large_or_unreadable_here_is_handed_as_drawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert workflow_selection_mask(_edit(b"not a picture", _opaque_mask())) is None
    assert workflow_selection_mask(_edit(mask=b"not a picture")) is None
    monkeypatch.setattr(studio_region_edit, "MAX_WORKFLOW_MASK_PIXELS", 100)
    assert workflow_selection_mask(_edit(mask=_opaque_mask())) is None


def test_a_workflow_selection_covering_nothing_refuses() -> None:
    with pytest.raises(RegionEditError) as raised:
        workflow_selection_mask(_edit(mask=_png(Image.new("RGBA", (64, 48), (0, 0, 0, 255)))))
    assert raised.value.code == "region-selection-empty"


#: A small selection on a larger picture: 40 x 30 pixels of a 512 x 384 picture.
SMALL = (200, 150, 239, 179)
#: A graph whose pictures all come back the crop's size; only pixels are tested here.
ROUTE = SelectionRoute(source_node_id="source", save_node_ids=frozenset({"save"}))
PAINTED = (10, 200, 30)


def _smooth_source() -> Image.Image:
    """A 512 x 384 picture of soft gradients, whose colours survive being resized."""
    picture = Image.new("RGB", (512, 384))
    picture.putdata(
        [(x // 2, y * 255 // 383, 128 + (x + y) // 16) for y in range(384) for x in range(512)]
    )
    return picture


def _grain(mean: tuple[int, int, int], deviation: float) -> Image.Image:
    """Fine grain around one colour, which enlarging and shrinking back softens."""
    rng = random.Random(7)
    picture = Image.new("RGB", (512, 384))
    picture.putdata(
        [
            (
                max(0, min(255, round(rng.gauss(mean[0], deviation)))),
                max(0, min(255, round(rng.gauss(mean[1], deviation)))),
                max(0, min(255, round(rng.gauss(mean[2], deviation)))),
            )
            for _ in range(512 * 384)
        ]
    )
    return picture


def _small_mask() -> bytes:
    alpha = Image.new("L", (512, 384), 0)
    ImageDraw.Draw(alpha).rectangle(SMALL, fill=255)
    mask = Image.new("RGBA", (512, 384), (255, 255, 255, 0))
    mask.putalpha(alpha)
    return _png(mask)


def test_a_selection_with_its_surroundings_past_half_the_picture_is_not_cropped() -> None:
    assert crop_selection(_edit(), ROUTE) is None


def test_a_small_selection_is_cropped_with_room_around_it_and_enlarged() -> None:
    crop = crop_selection(_edit(_png(_smooth_source()), _small_mask()), ROUTE)

    assert crop is not None
    assert crop.route == ROUTE
    # The selection's own bounds, widened by at least 32 pixels on each side,
    # then to the working size's shape.
    assert crop.box == (167, 118, 274, 212)
    # Enlarged toward a megapixel, in whole blocks of 64.
    assert crop.size == (1088, 960)
    picture = _pixels(crop.picture)
    mask = _pixels(crop.mask)
    assert picture.size == mask.size == crop.size
    assert picture.mode == "RGB"
    # The selection in its brightness and its alpha, as the whole picture's is handed.
    assert mask.getpixel((crop.size[0] // 2, crop.size[1] // 2)) == (255, 255, 255, 255)
    assert mask.getpixel((5, 5)) == (0, 0, 0, 0)


def test_a_picture_too_large_to_place_an_edit_back_into_is_not_cropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(studio_region_edit, "MAX_BLEND_PIXELS", 512 * 384 - 1)
    assert crop_selection(_edit(_png(_smooth_source()), _small_mask()), ROUTE) is None


def _round_drawing(crop: studio_region_edit.SelectionCrop) -> bytes:
    """A faithful workflow that draws a disc as tall as the selection it was handed."""
    given = _pixels(crop.picture).convert("RGB")
    selected = _pixels(crop.mask).getchannel("A").point(lambda value: 255 if value >= 128 else 0)
    bounds = selected.getbbox()
    assert bounds is not None
    middle_x, top, bottom = (bounds[0] + bounds[2]) / 2, bounds[1], bounds[3]
    radius = (bottom - top) / 2
    ImageDraw.Draw(given).ellipse((middle_x - radius, top, middle_x + radius, bottom), fill=PAINTED)
    return _png(given)


@pytest.mark.parametrize("selected", [(824, 738, 1223, 797), (874, 748, 1173, 787)])
def test_a_long_thin_selection_comes_back_in_its_own_shape(
    selected: tuple[int, int, int, int],
) -> None:
    """Rounding each side to whole blocks on its own would squash what the workflow draws."""
    source = Image.new("RGB", (2048, 1536), (128, 128, 128))
    alpha = Image.new("L", source.size, 0)
    ImageDraw.Draw(alpha).rectangle(selected, fill=255)
    drawn = Image.new("RGBA", source.size, (255, 255, 255, 0))
    drawn.putalpha(alpha)
    crop = crop_selection(_edit(_png(source), _png(drawn)), ROUTE)
    assert crop is not None
    across = crop.size[0] / (crop.box[2] - crop.box[0])
    down = crop.size[1] / (crop.box[3] - crop.box[1])
    assert abs(across / down - 1) <= 0.01

    out = _pixels(stitch_selection(crop, _round_drawing(crop)).content).convert("RGB")

    disc = Image.eval(out, lambda value: 255 if value < 60 else 0).getchannel("R").getbbox()
    assert disc is not None
    assert abs((disc[2] - disc[0]) - (disc[3] - disc[1])) <= 1, disc


def test_a_long_selection_whose_shape_does_not_fit_the_picture_is_edited_whole() -> None:
    """Across nearly the whole width, the crop cannot be widened to the working size's shape."""
    source = Image.new("RGB", (600, 2000), (128, 128, 128))
    alpha = Image.new("L", source.size, 0)
    ImageDraw.Draw(alpha).rectangle((10, 1000, 589, 1059), fill=255)
    drawn = Image.new("RGBA", source.size, (255, 255, 255, 0))
    drawn.putalpha(alpha)
    assert crop_selection(_edit(_png(source), _png(drawn)), ROUTE) is None


def test_a_source_that_does_not_read_whole_is_edited_whole() -> None:
    """A photo cut short still opens, so the whole picture is handed and its own checks decide."""
    buffer = io.BytesIO()
    _grain((90, 140, 70), 14).save(buffer, format="JPEG", quality=90)
    whole = buffer.getvalue()
    assert crop_selection(_edit(whole[: len(whole) // 2], _small_mask()), ROUTE) is None


def _faithful_edit(
    crop: studio_region_edit.SelectionCrop,
    painted: tuple[int, int, int],
    *,
    drift: tuple[int, int, int] = (0, 0, 0),
    shading: int = 0,
    light: int = 0,
) -> bytes:
    """What a faithful workflow returns: the crop it was given, painted where its mask selects.

    `drift` moves every pixel, as a model that shifts colour does; `shading`
    adds a gentle left-to-right ramp that averages to nothing; `light`
    brightens the right two fifths, as light cast by the edit would.
    """
    given = _pixels(crop.picture).convert("RGB")
    selected = _pixels(crop.mask).getchannel("A").point(lambda value: 255 if value >= 128 else 0)
    result = Image.composite(Image.new("RGB", given.size, painted), given, selected)
    if shading:
        width = given.width
        ramp = Image.new("L", (width, 1))
        ramp.putdata([round(shading * 2 * x / (width - 1)) for x in range(width)])
        ramp = ramp.resize(given.size, Image.Resampling.NEAREST)
        result = ImageChops.add(result, Image.merge("RGB", (ramp, ramp, ramp)), offset=-shading)
    if light:
        lit = Image.new("L", given.size, 0)
        ImageDraw.Draw(lit).rectangle(
            (given.width * 3 // 5, 0, given.width, given.height), fill=light
        )
        lit = ImageChops.subtract(lit, selected)
        result = ImageChops.add(result, Image.merge("RGB", (lit, lit, lit)))
    table = [min(255, max(0, value + offset)) for offset in drift for value in range(256)]
    return _png(result.point(table))


def _kept_outside_and_inside(
    source: Image.Image, content: bytes, painted: tuple[int, int, int], tolerance: int
) -> None:
    out = _pixels(content)
    assert out.size == (512, 384)
    for y in range(0, 384, 3):
        for x in range(0, 512, 3):
            if not (SMALL[0] <= x <= SMALL[2] and SMALL[1] <= y <= SMALL[3]):
                assert out.getpixel((x, y)) == source.getpixel((x, y)), (x, y)
    inside = _pixel_channels(out, (219, 164))
    assert all(abs(got - want) <= tolerance for got, want in zip(inside, painted, strict=True)), (
        inside
    )


def _specks(colour: tuple[int, int, int]) -> Image.Image:
    """A dark picture with a scattering of bright specks, as stars or dust are."""
    rng = random.Random(5)
    picture = Image.new("RGB", (512, 384), (8, 8, 8))
    for _ in range(2000):
        picture.putpixel((rng.randrange(512), rng.randrange(384)), colour)
    return picture


@pytest.mark.parametrize(
    "surroundings",
    [
        "grain",
        "dark-grain",
        "every-pixel-different",
        "shaded-wall",
        "white-specks",
        "red-specks",
        "light-cast",
    ],
)
def test_a_faithful_edit_on_fine_texture_is_kept_as_drawn(surroundings: str) -> None:
    """Resampling softens and clips fine texture, and light may fall around the edit.

    None of it is drift, and none of it moves the edit.
    """
    source, painted, shading, light = {
        "grain": (_grain((90, 140, 70), 14), (40, 40, 40), 0, 0),
        "dark-grain": (_grain((14, 14, 14), 4), (235, 225, 190), 0, 0),
        "every-pixel-different": (_source((512, 384)), PAINTED, 0, 0),
        "shaded-wall": (_grain((200, 200, 200), 0.6), (20, 20, 20), 4, 0),
        "white-specks": (_specks((255, 255, 255)), (200, 170, 150), 0, 0),
        "red-specks": (_specks((255, 0, 0)), (200, 170, 150), 0, 0),
        "light-cast": (_grain((90, 140, 70), 14), (230, 220, 120), 0, 15),
    }[surroundings]
    crop = crop_selection(_edit(_png(source), _small_mask()), ROUTE)
    assert crop is not None

    blend = stitch_selection(crop, _faithful_edit(crop, painted, shading=shading, light=light))

    _kept_outside_and_inside(source, blend.content, painted, tolerance=1)
    assert blend.record["colour_shift"] == [0, 0, 0]


def test_a_small_drift_around_the_selection_is_taken_out_of_the_edit() -> None:
    source = _grain((90, 140, 70), 14)
    crop = crop_selection(_edit(_png(source), _small_mask()), ROUTE)
    assert crop is not None

    blend = stitch_selection(crop, _faithful_edit(crop, PAINTED, drift=(5, -3, 4)))

    _kept_outside_and_inside(source, blend.content, PAINTED, tolerance=1)
    assert blend.record["mode"] == "crop"
    assert blend.record["box"] == list(crop.box)
    assert (blend.record["working_width"], blend.record["working_height"]) == crop.size
    assert blend.record["colour_shift"] == [-5, 3, -4]


def test_a_large_change_all_around_the_selection_is_taken_back_by_a_few_levels_only() -> None:
    """Past a few levels it may be the workflow's own, so only that much is taken back."""
    source = _smooth_source()
    crop = crop_selection(_edit(_png(source), _small_mask()), ROUTE)
    assert crop is not None

    blend = stitch_selection(crop, _faithful_edit(crop, PAINTED, drift=(30, 30, 30)))

    _kept_outside_and_inside(source, blend.content, (32, 222, 52), tolerance=1)
    assert blend.record["colour_shift"] == [-8, -8, -8]


def test_a_crop_that_comes_back_reshaped_cannot_be_placed_back() -> None:
    crop = crop_selection(_edit(_png(_smooth_source()), _small_mask()), ROUTE)
    assert crop is not None
    with pytest.raises(RegionEditError) as raised:
        stitch_selection(crop, _png(Image.new("RGB", (crop.size[0], crop.size[0]), (1, 2, 3))))
    assert raised.value.code == "region-result-reshaped"


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
