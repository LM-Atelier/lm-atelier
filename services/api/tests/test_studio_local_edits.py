"""The edits the studio makes without a model, each recorded as a step."""

from __future__ import annotations

import io
import math
from typing import Any

import pytest
from httpx2 import AsyncClient
from PIL import Image, ImageCms, ImageDraw

from local_lm.studio_adjustments import ColorAdjustments
from local_lm.studio_local_edits import (
    CanvasChange,
    CaptionOverlay,
    CropBox,
    LocalEditError,
    PerspectiveCorners,
    PictureSize,
    SelectionBlur,
    SelectionPaint,
    SelectionPixelate,
    render_local_edit,
)

RED, GREEN, BLUE, WHITE = (200, 0, 0), (0, 200, 0), (0, 0, 200), (250, 250, 250)


def _png(image: Image.Image, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **options)
    return buffer.getvalue()


def _tiles() -> Image.Image:
    """Three by two, with a different colour in each corner that matters."""

    picture = Image.new("RGB", (3, 2), WHITE)
    picture.putpixel((0, 0), RED)
    picture.putpixel((2, 0), GREEN)
    picture.putpixel((0, 1), BLUE)
    return picture


def _open(payload: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


@pytest.mark.parametrize(
    ("operation", "size", "red_at"),
    [
        # Top-left goes top-right when turned clockwise, and bottom-left the other way.
        ("rotate_clockwise", (2, 3), (1, 0)),
        ("rotate_counterclockwise", (2, 3), (0, 2)),
        ("flip_horizontal", (3, 2), (2, 0)),
        ("flip_vertical", (3, 2), (0, 1)),
    ],
)
def test_each_turn_and_flip_moves_every_pixel_exactly(
    operation: Any, size: tuple[int, int], red_at: tuple[int, int]
) -> None:
    result = _open(render_local_edit(_png(_tiles()), operation))

    assert result.format == "PNG"
    assert result.size == size
    assert result.getpixel(red_at) == RED
    # Nothing is resampled: the same colours, the same number of each.
    colours = result.convert("RGB").getcolors()
    assert colours is not None and sorted(colours) == sorted(_tiles().getcolors() or [])


def test_a_crop_keeps_exactly_the_box() -> None:
    result = _open(
        render_local_edit(_png(_tiles()), "crop", CropBox(left=1, top=0, width=2, height=1))
    )

    assert result.size == (2, 1)
    assert [result.getpixel((x, 0)) for x in range(2)] == [WHITE, GREEN]


@pytest.mark.parametrize(
    "box",
    [
        CropBox(left=0, top=0, width=4, height=1),
        CropBox(left=2, top=1, width=2, height=1),
        CropBox(left=0, top=0, width=0, height=1),
        CropBox(left=-1, top=0, width=1, height=1),
    ],
)
def test_a_crop_outside_the_picture_is_refused(box: CropBox) -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_tiles()), "crop", box)

    assert refused.value.code == "studio-crop-outside-picture"


def test_straightening_keeps_the_largest_box_of_the_pictures_own_shape() -> None:
    result = _open(
        render_local_edit(_png(Image.new("RGB", (400, 200), WHITE)), "straighten", straighten=10)
    )

    # Turned ten degrees, a 400 by 200 picture still covers a centered box of
    # its own shape 0.75 of its size, less two pixels a side: 292 by 146.
    assert result.size == (292, 146)


def test_a_leaning_line_stands_upright_once_straightened() -> None:
    picture = Image.new("L", (200, 200), 255)
    # From the bottom middle, leaning five degrees to the right toward the top.
    lean = math.tan(math.radians(5))
    ImageDraw.Draw(picture).line([(100, 190), (100 + 180 * lean, 10)], fill=0, width=3)

    result = _open(render_local_edit(_png(picture.convert("RGB")), "straighten", straighten=-5))

    width, height = result.size

    def middle_of_the_line(y: int) -> float:
        ink = [255 - result.getpixel((x, y))[0] for x in range(width)]
        return sum(x * amount for x, amount in enumerate(ink)) / sum(ink)

    # Over these rows the line leaned about nine pixels sideways, and turning
    # the wrong way would make that fifteen. Upright, its middle moves by about
    # one, which is how finely a drawn line three pixels wide lands.
    middles = [middle_of_the_line(y) for y in range(height // 4, 3 * height // 4)]
    assert max(middles) - min(middles) < 2


def test_straightening_leaves_no_empty_corner() -> None:
    solid = render_local_edit(_png(Image.new("RGB", (120, 80), WHITE)), "straighten", straighten=30)
    clear = render_local_edit(
        _png(Image.new("RGBA", (120, 80), (0, 0, 200, 255))), "straighten", straighten=-30
    )

    solid_picture = _open(solid)
    assert (
        min(
            solid_picture.getpixel((x, y))[0]
            for x in range(solid_picture.width)
            for y in range(solid_picture.height)
        )
        >= 249
    )
    clear_picture = _open(clear)
    assert clear_picture.mode == "RGBA"
    assert {
        clear_picture.getpixel((x, y))[3]
        for x in range(clear_picture.width)
        for y in range(clear_picture.height)
    } == {255}


@pytest.mark.parametrize("degrees", [None, 0])
def test_straightening_by_no_angle_is_refused(degrees: float | None) -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_tiles()), "straighten", straighten=degrees)

    assert refused.value.code == "studio-straighten-unchanged"


#: A card seen at an angle on a dark ground: its corners from the top left, clockwise.
CARD = ((30, 12), (96, 22), (100, 80), (18, 70))
CARD_COLOR = (230, 180, 40)


def _card_scene() -> Image.Image:
    scene = Image.new("RGB", (120, 90), (20, 20, 20))
    ImageDraw.Draw(scene).polygon(CARD, fill=CARD_COLOR)
    return scene


def test_a_card_seen_at_an_angle_fills_the_corrected_picture() -> None:
    result = _open(
        render_local_edit(_png(_card_scene()), "perspective", perspective=PerspectiveCorners(*CARD))
    )

    # The longer of each pair of opposite sides: the bottom, the square root of
    # 82 squared plus 10 squared, 82.6; and the left, of 12 and 58, 59.2.
    assert result.size == (83, 59)
    inside = {
        result.getpixel((x, y))
        for x in range(3, result.width - 3)
        for y in range(3, result.height - 3)
    }
    assert inside == {CARD_COLOR}


def test_each_corner_of_the_corrected_picture_shows_what_was_at_that_corner() -> None:
    scene = Image.new("RGB", (120, 90), WHITE)
    draw = ImageDraw.Draw(scene)
    for (x, y), color in zip(CARD, (RED, GREEN, BLUE, (0, 0, 0)), strict=True):
        draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=color)

    result = _open(
        render_local_edit(_png(scene), "perspective", perspective=PerspectiveCorners(*CARD))
    )

    right, bottom = result.width - 1, result.height - 1
    assert [
        result.getpixel(corner) for corner in ((0, 0), (right, 0), (right, bottom), (0, bottom))
    ] == [
        RED,
        GREEN,
        BLUE,
        (0, 0, 0),
    ]


def test_a_perspective_correction_keeps_transparency_and_hides_no_color_under_it() -> None:
    cutout = Image.new("RGBA", (60, 40), (255, 255, 255, 0))
    ImageDraw.Draw(cutout).rectangle((10, 8, 49, 31), fill=(200, 120, 40, 255))
    corners = PerspectiveCorners((5, 4), (55, 6), (54, 36), (4, 35))

    result = _open(render_local_edit(_png(cutout), "perspective", perspective=corners))

    assert result.mode == "RGBA"
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((result.width // 2, result.height // 2)) == (200, 120, 40, 255)
    # Along the cutout's edge the colors stay the cutout's own: the white under
    # the transparent pixels would otherwise lighten them.
    edge = [
        pixel
        for x in range(result.width)
        for y in range(result.height)
        if 64 <= (pixel := result.getpixel((x, y)))[3] < 255
    ]
    assert edge
    assert all(
        abs(red - 200) <= 10 and abs(green - 120) <= 10 and blue <= 55
        for red, green, blue, _ in edge
    )


@pytest.mark.parametrize(
    ("corners", "code"),
    [
        (None, "studio-perspective-missing"),
        # Where the corners start: the picture's own, which changes nothing.
        (((0, 0), (120, 0), (120, 90), (0, 90)), "studio-perspective-unchanged"),
        (((0, 0), (121, 0), (120, 90), (0, 90)), "studio-perspective-outside-picture"),
        # The top corners swapped, so the top and bottom sides cross.
        (((96, 22), (30, 12), (100, 80), (18, 70)), "studio-perspective-crossed"),
        # Every corner in its place but gone round the other way.
        (((30, 12), (18, 70), (100, 80), (96, 22)), "studio-perspective-crossed"),
        # One corner pulled inside the shape, so it is not four-sided any more.
        (((30, 12), (96, 22), (50, 30), (18, 70)), "studio-perspective-crossed"),
    ],
)
def test_corners_that_cannot_be_corrected_are_refused(
    corners: tuple[tuple[int, int], ...] | None, code: str
) -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(
            _png(_card_scene()),
            "perspective",
            perspective=PerspectiveCorners(*corners) if corners is not None else None,
        )

    assert refused.value.code == code


@pytest.mark.parametrize("size", [PictureSize(width=6, height=4), PictureSize(width=1, height=1)])
def test_a_resize_makes_exactly_the_size_asked_for(size: PictureSize) -> None:
    result = _open(render_local_edit(_png(_tiles()), "resize", size=size))

    assert result.format == "PNG"
    assert result.size == (size.width, size.height)
    assert result.mode == "RGB"


def test_a_resize_keeps_a_hidden_colour_from_bleeding_into_the_edge() -> None:
    """Under a transparent pixel there is still a colour, and nobody can see it."""
    picture = Image.new("RGBA", (2, 1))
    picture.putpixel((0, 0), (255, 0, 0, 0))
    picture.putpixel((1, 0), (0, 0, 255, 255))

    result = _open(render_local_edit(_png(picture), "resize", size=PictureSize(width=8, height=1)))

    assert result.mode == "RGBA"
    seen = [result.getpixel((x, 0)) for x in range(8)]
    assert any(0 < pixel[3] < 255 for pixel in seen)
    # Every pixel that shows at all shows blue: none of the hidden red.
    assert all(pixel[0] == 0 for pixel in seen if pixel[3] > 0)


@pytest.mark.parametrize(
    ("size", "code"),
    [
        (None, "studio-resize-missing"),
        (PictureSize(width=3, height=2), "studio-resize-unchanged"),
        (PictureSize(width=0, height=2), "studio-resize-empty"),
        (PictureSize(width=20_000, height=20_000), "studio-edit-too-large"),
    ],
)
def test_a_resize_that_cannot_be_made_is_refused(size: PictureSize | None, code: str) -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_tiles()), "resize", size=size)

    assert refused.value.code == code


def test_an_adjustment_changes_the_colors_and_nothing_else() -> None:
    result = _open(
        render_local_edit(_png(_tiles()), "adjust", adjustments=ColorAdjustments(saturation=-100))
    )

    assert result.size == (3, 2)
    # Every pixel is its own grey: the colors are gone and the light is kept.
    assert all(len(set(result.getpixel((x, y)))) == 1 for x in range(3) for y in range(2))
    assert result.getpixel((1, 1)) == (250, 250, 250)


@pytest.mark.parametrize("adjustments", [None, ColorAdjustments()])
def test_an_adjustment_that_moves_no_slider_is_refused(
    adjustments: ColorAdjustments | None,
) -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_tiles()), "adjust", adjustments=adjustments)

    assert refused.value.code == "studio-adjust-unchanged"


def _checkers(mode: str = "RGB") -> Image.Image:
    """Eight by four in alternating black and white, so any blur shows."""

    picture = Image.new("RGB", (8, 4))
    picture.putdata(
        [(255, 255, 255) if (x + y) % 2 else (0, 0, 0) for y in range(4) for x in range(8)]
    )
    return picture.convert(mode)


def _left_half(width: int = 8, height: int = 4) -> bytes:
    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (0, 0, width // 2, height))
    return _png(mask)


def test_a_blur_softens_the_marked_area_and_leaves_the_rest_exact() -> None:
    blur = SelectionBlur(mask=_left_half(), radius=2, mask_artifact_id="mask")

    result = _open(render_local_edit(_png(_checkers()), "blur", blur=blur))

    source = _checkers()
    marked = [result.getpixel((x, y)) for y in range(4) for x in range(4)]
    rest = [
        (result.getpixel((x, y)), source.getpixel((x, y))) for y in range(4) for x in range(4, 8)
    ]
    # Black and white are gone where it was marked: every pixel there is a grey.
    assert all(0 < pixel[0] < 255 for pixel in marked)
    assert all(after == before for after, before in rest)


def test_a_blur_keeps_transparency_and_hides_no_color_under_it() -> None:
    picture = Image.new("RGBA", (8, 4), (0, 0, 255, 255))
    for y in range(4):
        picture.putpixel((0, y), (255, 0, 0, 0))
    blur = SelectionBlur(mask=_left_half(), radius=2, mask_artifact_id="mask")

    result = _open(render_local_edit(_png(picture), "blur", blur=blur))

    assert result.mode == "RGBA"
    shown = [result.getpixel((x, y)) for y in range(4) for x in range(8)]
    # The red under the transparent column never shows up in its neighbours.
    assert all(pixel[0] == 0 for pixel in shown if pixel[3] > 0)


@pytest.mark.parametrize(
    ("mask", "code"),
    [
        (_png(Image.new("L", (8, 4), 0)), "region-selection-empty"),
        (b"not a picture", "region-image-unreadable"),
    ],
)
def test_a_blur_with_nothing_usable_marked_is_refused(mask: bytes, code: str) -> None:
    blur = SelectionBlur(mask=mask, radius=2, mask_artifact_id="mask")

    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_checkers()), "blur", blur=blur)

    assert refused.value.code == code


def _ramp() -> Image.Image:
    """Five by three, each pixel its own grey, so every block's mean is known."""

    picture = Image.new("RGB", (5, 3))
    for y in range(3):
        for x in range(5):
            shade = 10 * (x + 5 * y)
            picture.putpixel((x, y), (shade, shade, shade))
    return picture


def _all_marked(width: int = 5, height: int = 3) -> bytes:
    return _png(Image.new("L", (width, height), 255))


def test_a_pixelation_fills_each_block_with_its_mean() -> None:
    pixelate = SelectionPixelate(mask=_all_marked(), block=2, mask_artifact_id="mask")

    result = _open(render_local_edit(_png(_ramp()), "pixelate", pixelate=pixelate))

    rows = [[result.getpixel((x, y))[0] for x in range(5)] for y in range(3)]
    # Blocks start at the top-left corner. The last column and the last row are
    # blocks cut short by the edge, averaged over the pixels they have: 40 and
    # 90 make 65, 100 and 110 make 105, and the corner is 140 alone.
    assert rows == [
        [30, 30, 50, 50, 65],
        [30, 30, 50, 50, 65],
        [105, 105, 125, 125, 140],
    ]


def test_a_pixelation_changes_only_the_marked_area() -> None:
    marked_left = Image.new("L", (5, 3), 0)
    marked_left.paste(255, (0, 0, 4, 3))
    pixelate = SelectionPixelate(mask=_png(marked_left), block=2, mask_artifact_id="mask")

    result = _open(render_local_edit(_png(_ramp()), "pixelate", pixelate=pixelate))

    assert [result.getpixel((4, y)) for y in range(3)] == [
        _ramp().getpixel((4, y)) for y in range(3)
    ]
    assert result.getpixel((0, 0)) == (30, 30, 30)


def test_a_pixelation_keeps_transparency_and_hides_no_color_under_it() -> None:
    picture = Image.new("RGBA", (8, 4), (0, 0, 255, 255))
    for y in range(4):
        picture.putpixel((0, y), (255, 0, 0, 0))
    pixelate = SelectionPixelate(mask=_left_half(), block=2, mask_artifact_id="mask")

    result = _open(render_local_edit(_png(picture), "pixelate", pixelate=pixelate))

    assert result.mode == "RGBA"
    # The block over the transparent column is half covered, and blue only.
    assert result.getpixel((0, 0)) == (0, 0, 255, 128)
    shown = [result.getpixel((x, y)) for y in range(4) for x in range(8)]
    assert all(pixel[0] == 0 for pixel in shown if pixel[3] > 0)


def test_a_pixelation_without_a_usable_marked_area_is_refused() -> None:
    with pytest.raises(LocalEditError) as missing:
        render_local_edit(_png(_ramp()), "pixelate")
    empty = SelectionPixelate(
        mask=_png(Image.new("L", (5, 3), 0)), block=2, mask_artifact_id="mask"
    )
    with pytest.raises(LocalEditError) as unmarked:
        render_local_edit(_png(_ramp()), "pixelate", pixelate=empty)

    assert missing.value.code == "studio-pixelate-missing"
    assert unmarked.value.code == "region-selection-empty"


def _paint(color: tuple[int, int, int], opacity: int, mask: bytes | None = None) -> SelectionPaint:
    return SelectionPaint(
        mask=_left_half() if mask is None else mask,
        color=color,
        opacity=opacity,
        mask_artifact_id="mask",
    )


def test_a_paint_covers_the_marked_area_and_leaves_the_rest_exact() -> None:
    result = _open(render_local_edit(_png(_checkers()), "paint", paint=_paint((0, 0, 0), 100)))

    source = _checkers()
    assert all(result.getpixel((x, y)) == (0, 0, 0) for y in range(4) for x in range(4))
    assert all(
        result.getpixel((x, y)) == source.getpixel((x, y)) for y in range(4) for x in range(4, 8)
    )


def test_a_paint_at_half_opacity_lets_the_picture_show_through() -> None:
    white = Image.new("RGB", (8, 4), (255, 255, 255))

    result = _open(render_local_edit(_png(white), "paint", paint=_paint((255, 0, 0), 50)))

    red, green, blue = result.getpixel((0, 0))
    assert red == 255
    assert 120 <= green <= 135 and green == blue
    assert result.getpixel((7, 0)) == (255, 255, 255)


def test_paint_on_a_transparent_part_shows_as_paint() -> None:
    clear = Image.new("RGBA", (8, 4), (255, 0, 0, 0))

    result = _open(render_local_edit(_png(clear), "paint", paint=_paint((0, 0, 255), 100)))

    # Nothing of the red hidden under the transparency comes through.
    assert result.getpixel((0, 0)) == (0, 0, 255, 255)
    assert result.getpixel((7, 0))[3] == 0


def test_a_paint_with_nothing_marked_is_refused() -> None:
    with pytest.raises(LocalEditError) as missing:
        render_local_edit(_png(_checkers()), "paint")
    with pytest.raises(LocalEditError) as empty:
        render_local_edit(
            _png(_checkers()),
            "paint",
            paint=_paint((0, 0, 0), 100, _png(Image.new("L", (8, 4), 0))),
        )

    assert missing.value.code == "studio-paint-missing"
    assert empty.value.code == "region-selection-empty"


def _drawn(size: tuple[int, int] = (3, 2)) -> bytes:
    """What the browser draws: words on transparency, here one opaque and one faint pixel."""

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    overlay.putpixel((1, 0), (0, 0, 255, 255))
    overlay.putpixel((2, 1), (0, 0, 255, 128))
    return _png(overlay)


def test_drawn_words_are_laid_over_the_picture_as_drawn() -> None:
    caption = CaptionOverlay(overlay=_drawn(), overlay_artifact_id="words")

    result = _open(render_local_edit(_png(_tiles()), "caption", caption=caption))

    assert result.mode == "RGB"
    assert result.getpixel((1, 0)) == (0, 0, 255)
    # Half-covered: half the words' color over the picture beneath.
    red, green, blue = result.getpixel((2, 1))
    assert 120 <= red <= 130 and 120 <= green <= 130 and blue > 245
    # Where nothing was drawn the picture is exactly what it was.
    assert result.getpixel((0, 0)) == RED
    assert result.getpixel((2, 0)) == GREEN


def test_drawn_words_keep_a_transparent_picture_transparent_around_them() -> None:
    clear = Image.new("RGBA", (3, 2), (0, 0, 0, 0))
    caption = CaptionOverlay(overlay=_drawn(), overlay_artifact_id="words")

    result = _open(render_local_edit(_png(clear), "caption", caption=caption))

    assert result.mode == "RGBA"
    assert result.getpixel((1, 0)) == (0, 0, 255, 255)
    assert result.getpixel((0, 0))[3] == 0


@pytest.mark.parametrize(
    ("caption", "code"),
    [
        (None, "studio-caption-missing"),
        (
            CaptionOverlay(overlay=_drawn((4, 2)), overlay_artifact_id="w"),
            "studio-caption-size-mismatch",
        ),
        (
            CaptionOverlay(overlay=b"not a picture", overlay_artifact_id="w"),
            "studio-caption-unreadable",
        ),
    ],
)
def test_words_that_cannot_be_laid_down_are_refused(
    caption: CaptionOverlay | None, code: str
) -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_tiles()), "caption", caption=caption)

    assert refused.value.code == code


def test_a_blur_without_a_marked_area_is_refused() -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_checkers()), "blur")

    assert refused.value.code == "studio-blur-missing"


def test_a_larger_canvas_centers_the_picture_on_transparency() -> None:
    result = _open(render_local_edit(_png(_tiles()), "canvas", canvas=CanvasChange(7, 4)))

    assert result.size == (7, 4)
    assert result.mode == "RGBA"
    # Two columns and one row of new ground before the picture.
    assert result.getpixel((2, 1)) == (*RED, 255)
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((6, 3))[3] == 0


def test_an_odd_pixel_of_room_goes_after_the_picture() -> None:
    grown = _open(render_local_edit(_png(_tiles()), "canvas", canvas=CanvasChange(6, 2)))
    shrunk = _open(render_local_edit(_png(_tiles()), "canvas", canvas=CanvasChange(2, 2)))

    # Growing by three: one column before, two after.
    assert grown.getpixel((1, 0)) == (*RED, 255)
    # Shrinking by one: the column cut is the last one.
    assert shrunk.getpixel((0, 0)) == (*RED, 255)


def test_a_smaller_canvas_keeps_the_side_it_is_anchored_to() -> None:
    result = _open(
        render_local_edit(_png(_tiles()), "canvas", canvas=CanvasChange(2, 1, anchor="top_right"))
    )

    assert result.size == (2, 1)
    assert [result.getpixel((x, 0))[:3] for x in range(2)] == [WHITE, GREEN]


def test_a_filled_canvas_colors_only_the_new_ground() -> None:
    opaque = _open(
        render_local_edit(_png(_tiles()), "canvas", canvas=CanvasChange(5, 2, fill="black"))
    )
    cutout = Image.new("RGBA", (2, 1), (10, 20, 30, 255))
    cutout.putpixel((0, 0), (0, 0, 0, 0))
    kept = _open(render_local_edit(_png(cutout), "canvas", canvas=CanvasChange(4, 1, fill="white")))

    assert opaque.mode == "RGB"
    assert opaque.getpixel((0, 0)) == (0, 0, 0)
    assert opaque.getpixel((1, 0)) == RED
    # The picture's own transparency stays transparent; white fills only what is new.
    assert kept.getpixel((0, 0)) == (255, 255, 255, 255)
    assert kept.getpixel((1, 0))[3] == 0
    assert kept.getpixel((2, 0)) == (10, 20, 30, 255)


@pytest.mark.parametrize(
    ("canvas", "code"),
    [
        (None, "studio-canvas-missing"),
        (CanvasChange(3, 2), "studio-canvas-unchanged"),
        (CanvasChange(0, 2), "studio-canvas-empty"),
        (CanvasChange(20_000, 20_000), "studio-edit-too-large"),
    ],
)
def test_a_canvas_that_cannot_be_made_is_refused(canvas: CanvasChange | None, code: str) -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_tiles()), "canvas", canvas=canvas)

    assert refused.value.code == code


def test_a_picture_is_edited_as_it_is_seen_upright() -> None:
    """A camera's orientation tag is how the person saw it, so it is what they turned."""
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, shown turned a quarter clockwise
    stored = _png(_tiles(), exif=exif.tobytes())

    result = _open(render_local_edit(stored, "flip_horizontal"))

    assert result.size == (2, 3)
    # The tag is spent: the pixels are now the way up they were seen.
    assert result.getexif().get(0x0112) in (None, 1)


def test_transparency_survives_a_turn() -> None:
    picture = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    picture.putpixel((0, 0), (10, 20, 30, 255))

    result = _open(render_local_edit(_png(picture), "flip_horizontal"))

    assert result.mode == "RGBA"
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((1, 0)) == (10, 20, 30, 255)


def test_a_colour_profile_is_kept_only_where_it_still_describes_the_pixels() -> None:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    kept = _open(render_local_edit(_png(_tiles(), icc_profile=profile), "flip_vertical"))
    grey = Image.new("L", (2, 2), 90)
    # A grey picture's profile would misdescribe the RGB it is turned into.
    dropped = _open(render_local_edit(_png(grey, icc_profile=profile), "flip_vertical"))

    assert kept.info.get("icc_profile") == profile
    assert dropped.mode == "RGB"
    assert "icc_profile" not in dropped.info


def test_bytes_that_are_not_a_picture_are_refused() -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(b"not a picture", "rotate_clockwise")

    assert refused.value.code == "studio-edit-unreadable"


async def _upload(client: AsyncClient, name: str, content: bytes) -> str:
    response = await client.post("/api/artifacts", files={"file": (name, content, "image/png")})
    assert response.status_code == 201, response.text
    identifier: str = response.json()["id"]
    return identifier


async def _session_over(client: AsyncClient, artifact_id: str) -> str:
    opened = await client.post("/api/studio/sessions", json={"source_artifact_id": artifact_id})
    assert opened.status_code == 200, opened.text
    session_id: str = opened.json()["id"]
    return session_id


async def _content(client: AsyncClient, artifact_id: str) -> Image.Image:
    response = await client.get(f"/api/artifacts/{artifact_id}/content")
    assert response.status_code == 200
    return _open(response.content)


async def test_a_turn_becomes_the_next_step_after_the_picture_it_changed(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "rotate_clockwise"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    request, answer = body["messages"]
    assert request["role"] == "user"
    assert [part["type"] for part in request["parts"]] == ["text", "image"]
    assert request["parts"][0]["text"] == "Rotate right"
    # First among the request's pictures is the one changed: compare shows it.
    assert request["parts"][1]["artifact_id"] == source_id
    assert answer["role"] == "assistant"
    assert answer["status"] == "complete"
    assert answer["parent_id"] == request["id"]
    assert body["active_head_message_id"] == answer["id"]
    image, metadata = answer["parts"]
    assert image["type"] == "image"
    assert metadata["type"] == "generation_metadata"
    # Says what was done to which picture, and names no model, because none ran.
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {"operation": "rotate_clockwise", "source_artifact_id": source_id}
    }
    turned = await _content(client, image["artifact_id"])
    assert turned.size == (2, 3)
    assert turned.getpixel((1, 0)) == RED

    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.status_code == 200
    assert detail.json()["kind"] == "image"
    assert detail.json()["original_name"] == "tiles (rotated right).png"
    assert detail.json()["generation_identity"] is None
    library = await client.get("/api/artifact-library", params={"kind": "image"})
    assert image["artifact_id"] in {row["artifact_id"] for row in library.json()["items"]}


async def test_a_result_already_in_the_session_can_be_turned_again(client: AsyncClient) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)
    first = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "flip_horizontal"},
    )
    flipped = first.json()["messages"][-1]["parts"][0]["artifact_id"]

    second = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": flipped,
            "operation": "crop",
            "crop": {"left": 0, "top": 0, "width": 1, "height": 1},
        },
    )

    assert second.status_code == 200, second.text
    messages = second.json()["messages"]
    assert len(messages) == 4
    assert messages[2]["parent_id"] == messages[1]["id"]
    assert messages[2]["parts"][1]["artifact_id"] == flipped
    cropped = await _content(client, messages[3]["parts"][0]["artifact_id"])
    # The flip put green top-left; the crop kept only that pixel.
    assert cropped.size == (1, 1)
    assert cropped.getpixel((0, 0)) == GREEN


async def test_a_straightening_is_recorded_with_its_angle_and_method(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "wide.png", _png(Image.new("RGB", (400, 200), WHITE)))
    session_id = await _session_over(client, source_id)

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "straighten",
            "straighten": {"degrees": 10},
        },
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Straighten"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {
            "operation": "straighten",
            "source_artifact_id": source_id,
            "straighten": {"degrees": 10.0, "resampler": "bicubic"},
        }
    }
    straightened = await _content(client, image["artifact_id"])
    assert straightened.size == (292, 146)
    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.json()["original_name"] == "wide (straightened).png"


async def test_a_perspective_correction_is_recorded_with_its_corners_and_method(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "card.png", _png(_card_scene()))
    session_id = await _session_over(client, source_id)
    corners = {
        name: {"x": x, "y": y}
        for name, (x, y) in zip(
            ("top_left", "top_right", "bottom_right", "bottom_left"), CARD, strict=True
        )
    }

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "perspective",
            "perspective": corners,
        },
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Correct the perspective"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {
            "operation": "perspective",
            "source_artifact_id": source_id,
            "perspective": {"corners": corners, "resampler": "bicubic"},
        }
    }
    corrected = await _content(client, image["artifact_id"])
    assert corrected.size == (83, 59)
    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.json()["original_name"] == "card (perspective corrected).png"


async def test_a_perspective_correction_needs_its_corners_and_only_it_takes_them(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "card.png", _png(_card_scene()))
    session_id = await _session_over(client, source_id)
    corners = {
        "top_left": {"x": 1, "y": 1},
        "top_right": {"x": 100, "y": 1},
        "bottom_right": {"x": 100, "y": 80},
        "bottom_left": {"x": 1, "y": 80},
    }

    missing = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "perspective"},
    )
    stray = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "flip_vertical",
            "perspective": corners,
        },
    )
    outside = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "perspective",
            "perspective": {**corners, "bottom_right": {"x": 121, "y": 80}},
        },
    )

    assert missing.status_code == 422
    assert stray.status_code == 422
    assert outside.status_code == 422
    assert outside.json()["code"] == "studio-perspective-outside-picture"


async def test_a_resize_is_recorded_with_its_size_and_method(client: AsyncClient) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "resize",
            "size": {"width": 30, "height": 20},
        },
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Resize"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {
            "operation": "resize",
            "source_artifact_id": source_id,
            "size": {"width": 30, "height": 20},
            "resampler": "lanczos",
        }
    }
    resized = await _content(client, image["artifact_id"])
    assert resized.size == (30, 20)
    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.json()["original_name"] == "tiles (resized).png"


async def test_an_adjustment_is_recorded_with_where_each_slider_stood(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)
    sliders = {
        "brightness": 10,
        "contrast": -20,
        "saturation": 30,
        "warmth": -40,
        "tint": 25,
        "sharpness": 35,
    }

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "adjust", "adjustments": sliders},
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Adjust light and color"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {
            "operation": "adjust",
            "source_artifact_id": source_id,
            "adjustments": sliders,
        }
    }
    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.json()["original_name"] == "tiles (adjusted).png"


async def test_a_blur_is_recorded_with_its_radius_and_marked_area(client: AsyncClient) -> None:
    source_id = await _upload(client, "checkers.png", _png(_checkers()))
    mask_id = await _upload(client, "studio-selection.png", _left_half())
    session_id = await _session_over(client, source_id)

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "blur",
            "blur": {"mask_artifact_id": mask_id, "radius": 3},
        },
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Blur part of the picture"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {
            "operation": "blur",
            "source_artifact_id": source_id,
            "blur": {"radius": 3, "filter": "gaussian", "mask_artifact_id": mask_id},
        }
    }
    blurred = await _content(client, image["artifact_id"])
    assert blurred.getpixel((7, 0)) == _checkers().getpixel((7, 0))
    assert 0 < blurred.getpixel((0, 0))[0] < 255


async def test_a_pixelation_is_recorded_with_its_block_and_marked_area(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "ramp.png", _png(_ramp()))
    mask_id = await _upload(client, "studio-selection.png", _all_marked())
    session_id = await _session_over(client, source_id)

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "pixelate",
            "pixelate": {"mask_artifact_id": mask_id, "block": 2},
        },
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Pixelate part of the picture"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {
            "operation": "pixelate",
            "source_artifact_id": source_id,
            "pixelate": {"block": 2, "mask_artifact_id": mask_id},
        }
    }
    pixelated = await _content(client, image["artifact_id"])
    assert pixelated.getpixel((0, 0)) == (30, 30, 30)
    assert pixelated.getpixel((4, 2)) == (140, 140, 140)
    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.json()["original_name"] == "ramp (pixelated).png"


async def test_a_canvas_change_is_recorded_with_its_size_anchor_and_fill(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)
    canvas = {"width": 9, "height": 2, "anchor": "left", "fill": "white"}

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "canvas", "canvas": canvas},
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Change the canvas size"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {"operation": "canvas", "source_artifact_id": source_id, "canvas": canvas}
    }
    placed = await _content(client, image["artifact_id"])
    assert placed.size == (9, 2)
    assert placed.getpixel((0, 0)) == RED
    assert placed.getpixel((8, 1)) == (255, 255, 255)
    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.json()["original_name"] == "tiles (canvas changed).png"


async def test_a_paint_is_recorded_with_its_color_opacity_and_marked_area(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "checkers.png", _png(_checkers()))
    mask_id = await _upload(client, "studio-selection.png", _left_half())
    session_id = await _session_over(client, source_id)

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "paint",
            "paint": {"mask_artifact_id": mask_id, "color": "#112233", "opacity": 100},
        },
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Paint over part of the picture"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {
            "operation": "paint",
            "source_artifact_id": source_id,
            "paint": {"color": "#112233", "opacity": 100, "mask_artifact_id": mask_id},
        }
    }
    painted = await _content(client, image["artifact_id"])
    assert painted.getpixel((0, 0)) == (0x11, 0x22, 0x33)
    assert painted.getpixel((7, 0)) == _checkers().getpixel((7, 0))


async def test_added_words_are_recorded_with_the_drawn_overlay(client: AsyncClient) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    words_id = await _upload(client, "studio-words.png", _drawn())
    session_id = await _session_over(client, source_id)

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "caption",
            "caption": {"overlay_artifact_id": words_id},
        },
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Add text"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {
            "operation": "caption",
            "source_artifact_id": source_id,
            "caption": {"overlay_artifact_id": words_id},
        }
    }
    captioned = await _content(client, image["artifact_id"])
    assert captioned.getpixel((1, 0)) == (0, 0, 255)
    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.json()["original_name"] == "tiles (with text).png"


async def test_only_a_picture_in_the_session_can_be_edited_through_it(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    elsewhere = await _upload(client, "other.png", _png(Image.new("RGB", (2, 2), BLUE)))
    session_id = await _session_over(client, source_id)

    refused = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": elsewhere, "operation": "rotate_clockwise"},
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "studio-edit-source-not-in-session"
    after = await client.get(f"/api/studio/sessions/{session_id}")
    assert after.json()["messages"] == []


async def test_a_refused_edit_leaves_the_session_as_it_was(client: AsyncClient) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)

    outside = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "crop",
            "crop": {"left": 2, "top": 0, "width": 2, "height": 1},
        },
    )
    missing_box = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "crop"},
    )
    stray_box = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "flip_vertical",
            "crop": {"left": 0, "top": 0, "width": 1, "height": 1},
        },
    )
    missing_size = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "resize"},
    )
    stray_size = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "rotate_clockwise",
            "size": {"width": 4, "height": 4},
        },
    )
    unchanged = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "resize",
            "size": {"width": 3, "height": 2},
        },
    )
    missing_sliders = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "adjust"},
    )
    stray_sliders = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "flip_vertical",
            "adjustments": {"brightness": 10},
        },
    )
    too_far = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "adjust",
            "adjustments": {"brightness": 101},
        },
    )
    strange_color = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "paint",
            "paint": {"mask_artifact_id": source_id, "color": "red", "opacity": 50},
        },
    )
    no_words = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "caption",
            "caption": {"overlay_artifact_id": "sha256:" + "0" * 64},
        },
    )
    stray_canvas = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "flip_vertical",
            "canvas": {"width": 4, "height": 4},
        },
    )
    strange_anchor = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "canvas",
            "canvas": {"width": 4, "height": 4, "anchor": "middle"},
        },
    )
    no_mask = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "blur",
            "blur": {"mask_artifact_id": "sha256:" + "0" * 64, "radius": 3},
        },
    )
    stray_blur = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "flip_vertical",
            "blur": {"mask_artifact_id": source_id, "radius": 3},
        },
    )
    stray_pixelation = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "blur",
            "pixelate": {"mask_artifact_id": source_id, "block": 4},
        },
    )
    one_pixel_blocks = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "pixelate",
            "pixelate": {"mask_artifact_id": source_id, "block": 1},
        },
    )
    no_pixelation_mask = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "pixelate",
            "pixelate": {"mask_artifact_id": "sha256:" + "0" * 64, "block": 4},
        },
    )
    stray_angle = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "flip_vertical",
            "straighten": {"degrees": 5},
        },
    )
    too_steep = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "straighten",
            "straighten": {"degrees": 50},
        },
    )
    level = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "straighten",
            "straighten": {"degrees": 0},
        },
    )
    absent = await client.post(
        "/api/studio/sessions/absent/local-edits",
        json={"source_artifact_id": source_id, "operation": "flip_vertical"},
    )

    assert outside.status_code == 422
    assert outside.json()["code"] == "studio-crop-outside-picture"
    assert missing_box.status_code == 422
    assert stray_box.status_code == 422
    assert missing_size.status_code == 422
    assert stray_size.status_code == 422
    assert unchanged.status_code == 422
    assert unchanged.json()["code"] == "studio-resize-unchanged"
    assert missing_sliders.status_code == 422
    assert stray_sliders.status_code == 422
    assert too_far.status_code == 422
    assert no_mask.status_code == 422
    assert no_mask.json()["code"] == "studio-marked-area-missing"
    assert stray_blur.status_code == 422
    assert stray_angle.status_code == 422
    assert too_steep.status_code == 422
    assert level.status_code == 422
    assert level.json()["code"] == "studio-straighten-unchanged"
    assert stray_pixelation.status_code == 422
    assert one_pixel_blocks.status_code == 422
    assert no_pixelation_mask.status_code == 422
    assert no_pixelation_mask.json()["code"] == "studio-marked-area-missing"
    assert stray_canvas.status_code == 422
    assert no_words.status_code == 422
    assert no_words.json()["code"] == "studio-caption-missing"
    assert strange_color.status_code == 422
    assert strange_anchor.status_code == 422
    assert absent.status_code == 404
    assert absent.json()["code"] == "studio-session-not-found"
    after = await client.get(f"/api/studio/sessions/{session_id}")
    assert after.json()["messages"] == []
