"""An extension made on a canvas of whole blocks comes back as the canvas asked for."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import replace

import pytest
from PIL import Image
from test_outpaint_colour import (
    cast_by,
    comfy_extended,
    largest_difference,
    picture_of,
    placed,
    scene,
)

from local_lm.outpaint_finish import (
    ExtensionFinish,
    feathered_px,
    finish_extension,
    prepare_source,
    stated_feathering,
)
from local_lm.outpaint_workflows import working_pixels
from local_lm.workflow_source_geometry import SelectionRoute

SOURCE_ID = f"sha256:{'a' * 64}"
SIZE = (517, 383)
NEW = (51, 102, 204)
ROUTE = SelectionRoute(source_node_id="pad", save_node_ids=frozenset({"save"}))


def _png(image: Image.Image, **options: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **options)
    return buffer.getvalue()


def _source(size: tuple[int, int] = SIZE) -> Image.Image:
    # A different colour in every pixel, so a shifted or resampled pixel shows.
    picture = Image.new("RGB", size)
    picture.putdata(
        [
            ((x * 7) % 256, (y * 11) % 256, (x * y) % 256)
            for y in range(size[1])
            for x in range(size[0])
        ]
    )
    return picture


def _finish(requested: dict[str, int]) -> ExtensionFinish:
    working, grown = working_pixels(requested, *SIZE)
    prepared = prepare_source(_png(_source()), grown)
    assert prepared is not None
    return ExtensionFinish(
        source_artifact_id=SOURCE_ID,
        prepared=prepared[0],
        route=ROUTE,
        size=SIZE,
        requested=requested,
        working=working,
        grown=grown,
    )


def _made(finish: ExtensionFinish) -> Image.Image:
    """What a workflow returns: its canvas, the source it was handed in place and the rest new."""
    canvas = Image.new("RGB", finish.canvas, NEW)
    handed = Image.open(io.BytesIO(finish.prepared)).convert("RGB")
    canvas.paste(handed, (finish.working["left"], finish.working["top"]))
    return canvas


def test_an_extended_axis_is_padded_to_whole_blocks_and_another_grows_the_source() -> None:
    # Right only: 517 + 129 = 646 grows to 704 on the right; the height nobody
    # extended is not padded, and the source grows from 383 to 384 rows.
    assert working_pixels({"top": 0, "right": 129, "bottom": 0, "left": 0}, *SIZE) == (
        {"top": 0, "right": 187, "bottom": 0, "left": 0},
        {"right": 0, "bottom": 1},
    )
    # Left only takes the extra on the left (527 grows to 576), top only on the
    # top (423 to 448), and both sides put it on the right (537 to 576).
    assert working_pixels({"top": 0, "right": 0, "bottom": 0, "left": 10}, *SIZE)[0]["left"] == 59
    assert working_pixels({"top": 40, "right": 0, "bottom": 0, "left": 0}, *SIZE) == (
        {"top": 65, "right": 0, "bottom": 0, "left": 0},
        {"right": 59, "bottom": 0},
    )
    both, _ = working_pixels({"top": 0, "right": 10, "bottom": 0, "left": 10}, *SIZE)
    assert (both["left"], both["right"]) == (10, 49)
    # A canvas already in whole blocks is padded as asked.
    assert working_pixels({"top": 64, "right": 0, "bottom": 0, "left": 0}, 512, 384) == (
        {"top": 64, "right": 0, "bottom": 0, "left": 0},
        {"right": 0, "bottom": 0},
    )


def test_the_source_grows_by_repeating_its_last_column_and_row() -> None:
    source = _source()

    prepared = prepare_source(_png(source), {"right": 3, "bottom": 2})

    assert prepared is not None
    content, size = prepared
    assert size == SIZE
    handed = Image.open(io.BytesIO(content)).convert("RGB")
    assert handed.size == (520, 385)
    assert handed.crop((0, 0, *SIZE)).tobytes() == source.tobytes()
    for x in range(517, 520):
        assert handed.getpixel((x, 100)) == source.getpixel((516, 100))
    for y in range(383, 385):
        assert handed.getpixel((200, y)) == source.getpixel((200, 382))
    assert handed.getpixel((519, 384)) == source.getpixel((516, 382))


def test_a_turned_source_is_handed_upright() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6
    stored = _source((383, 517))

    prepared = prepare_source(_png(stored, exif=exif.tobytes()), {"right": 0, "bottom": 0})

    assert prepared is not None
    handed = Image.open(io.BytesIO(prepared[0]))
    assert prepared[1] == handed.size == (517, 383)
    assert handed.getexif().get(0x0112) is None


def _sixteen_bit() -> bytes:
    return _png(Image.new("I;16", SIZE, 32768))


def _animated() -> bytes:
    frames = [Image.new("RGB", SIZE, (60, 60, 60)), Image.new("RGB", SIZE, (200, 200, 200))]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])
    return buffer.getvalue()


def _mirrored() -> bytes:
    exif = Image.Exif()
    exif[0x0112] = 2
    return _png(_source(), exif=exif.tobytes())


def _unreadable() -> bytes:
    return b"not a picture"


@pytest.mark.parametrize(
    "stored",
    [_sixteen_bit, _animated, _mirrored, _unreadable],
    ids=["sixteen-bit", "animated", "mirrored", "unreadable"],
)
def test_a_source_comfyui_would_read_differently_is_not_replaced(
    stored: Callable[[], bytes],
) -> None:
    assert prepare_source(stored(), {"right": 0, "bottom": 1}) is None


@pytest.mark.parametrize(
    "requested",
    [
        {"top": 0, "right": 129, "bottom": 0, "left": 0},
        {"top": 40, "right": 0, "bottom": 0, "left": 10},
        {"top": 0, "right": 10, "bottom": 30, "left": 10},
    ],
    ids=["right", "top-and-left", "three-sides"],
)
def test_an_extension_is_cut_back_to_the_canvas_asked_for(requested: dict[str, int]) -> None:
    finish = _finish(requested)

    finished = finish_extension(finish, _png(_made(finish)))

    assert finished is not None
    out = Image.open(io.BytesIO(finished.content)).convert("RGB")
    width = SIZE[0] + requested["left"] + requested["right"]
    height = SIZE[1] + requested["top"] + requested["bottom"]
    assert out.size == (width, height)
    # The source sits exactly where it was asked to, and nothing it grew by is kept.
    left, top = requested["left"], requested["top"]
    assert out.crop((left, top, left + SIZE[0], top + SIZE[1])).tobytes() == _source().tobytes()
    assert finished.record["requested"] == requested
    assert finished.record["working"] == dict(finish.working)
    assert finished.record["grown"] == dict(finish.grown)
    assert (finished.record["width"], finished.record["height"]) == (width, height)
    assert (finished.record["result_width"], finished.record["result_height"]) == finish.canvas


@pytest.mark.parametrize("returned", ["asked-for size", "unreadable"])
def test_a_picture_that_is_not_the_canvas_the_workflow_ran_on_is_kept_as_made(
    returned: str,
) -> None:
    finish = _finish({"top": 0, "right": 129, "bottom": 0, "left": 0})
    result = (
        _png(Image.new("RGB", (646, 383), NEW))
        if returned == "asked-for size"
        else b"not a picture"
    )

    assert finish_extension(finish, result) is None


def _scene_finish(requested: dict[str, int]) -> ExtensionFinish:
    working, grown = working_pixels(requested, *SIZE)
    prepared = prepare_source(_png(picture_of(SIZE, scene)), grown)
    assert prepared is not None
    return ExtensionFinish(
        source_artifact_id=SOURCE_ID,
        prepared=prepared[0],
        route=ROUTE,
        size=SIZE,
        requested=requested,
        working=working,
        grown=grown,
        blended_saves=frozenset({"save"}),
        feathering=64,
    )


def _made_with_a_cast(finish: ExtensionFinish, cast: tuple[int, int, int]) -> bytes:
    """The working canvas as ComfyUI saves it, its decode the scene continued a few levels off."""
    handed = Image.open(io.BytesIO(finish.prepared)).convert("RGB")
    continued = placed(scene, finish.working)
    return _png(comfy_extended(handed, finish.working, 64, cast_by(cast, continued)))


def test_a_blended_extension_takes_the_colour_of_its_source() -> None:
    requested = {"top": 0, "right": 129, "bottom": 0, "left": 0}
    finish = _scene_finish(requested)
    cast = (12, -8, 16)

    finished = finish_extension(finish, _made_with_a_cast(finish, cast), blended=True)

    assert finished is not None
    out = Image.open(io.BytesIO(finished.content)).convert("RGB")
    assert out.size == (646, 383)
    assert largest_difference(out.crop((0, 0, *SIZE)), picture_of(SIZE, scene)) <= 1
    beside = picture_of((1, 383), lambda _x, y: scene(517, y))
    assert largest_difference(out.crop((517, 0, 518, 383)), beside) <= 1
    colour = finished.record["colour"]
    assert colour["applied"] is True and colour["feather_px"] == 64
    assert colour["sides"] == {"right": {"largest_shift": list(cast)}}


@pytest.mark.parametrize(
    "blended,feathering,reason",
    [(False, 64, "not_blended"), (True, None, "feather_unknown")],
    ids=["decoded-straight-to-its-save", "feather-not-stated"],
)
def test_an_extension_whose_decode_cannot_be_read_back_keeps_its_colour(
    blended: bool, feathering: int | None, reason: str
) -> None:
    requested = {"top": 0, "right": 129, "bottom": 0, "left": 0}
    finish = replace(_scene_finish(requested), feathering=feathering)
    made = _made_with_a_cast(finish, (12, -8, 16))

    finished = finish_extension(finish, made, blended=blended)

    assert finished is not None
    out = Image.open(io.BytesIO(finished.content)).convert("RGB")
    assert (
        out.tobytes()
        == Image.open(io.BytesIO(made)).convert("RGB").crop((0, 0, 646, 383)).tobytes()
    )
    assert finished.record["colour"] == {"applied": False, "reason": reason}


def test_a_picture_whose_colour_is_not_matched_is_cut_back_as_it_was_saved() -> None:
    finish = _finish({"top": 0, "right": 129, "bottom": 0, "left": 0})

    finished = finish_extension(finish, _png(_made(finish).convert("RGBA")))

    assert finished is not None
    assert Image.open(io.BytesIO(finished.content)).mode == "RGBA"


@pytest.mark.parametrize(
    "value,settings,stated",
    [
        (64, {}, 64),
        (0, {}, 0),
        ("$" + "{feathering}", {"feathering": 48}, 48),
        ("$" + "{feathering}", {}, None),
        ("$" + "{feathering}", {"feathering": 48.0}, None),
        (["feather", 0], {}, None),
        (64.0, {}, None),
        ("64", {}, None),
        (True, {}, None),
        (-1, {}, None),
        (None, {}, None),
    ],
)
def test_a_feather_is_known_only_as_a_whole_number_the_run_states(
    value: object, settings: dict[str, object], stated: int | None
) -> None:
    assert stated_feathering(value, settings) == stated


def test_comfyui_feathers_only_where_twice_the_feather_fits_the_picture_it_pads() -> None:
    assert feathered_px(64, (517, 384)) == 64
    assert feathered_px(64, (128, 384)) == 0
    assert feathered_px(64, (517, 128)) == 0
    assert feathered_px(0, (517, 384)) == 0
