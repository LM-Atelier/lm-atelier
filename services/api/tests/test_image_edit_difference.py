"""The pixel pre-check that a verifier cannot talk its way past."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from local_lm.image_edit_difference import (
    UNCHANGED_THRESHOLD,
    ChangedArea,
    compare_edit,
    compare_images,
    crop_changed_area,
)


def _encode(image: Image.Image, format_name: str = "PNG", **options: object) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=format_name, **options)
    return buffer.getvalue()


def _solid(colour: tuple[int, int, int], size: tuple[int, int] = (256, 256)) -> Image.Image:
    return Image.new("RGB", size, colour)


def test_an_unchanged_image_is_reported_as_unchanged() -> None:
    """The mug case: the edit ran, the picture did not change."""
    image = _solid((40, 90, 180))
    difference = compare_images(_encode(image), _encode(image))

    assert difference.comparable
    assert not difference.changed
    assert difference.mean_absolute_difference == 0.0


def test_re_encoding_alone_does_not_count_as_a_change() -> None:
    """A lossy round trip must not read as an edit."""
    image = _solid((40, 90, 180))
    difference = compare_images(
        _encode(image),
        _encode(image, "JPEG", quality=82),
    )

    assert difference.comparable
    assert not difference.changed
    assert difference.mean_absolute_difference < UNCHANGED_THRESHOLD


def test_a_recoloured_subject_counts_as_a_change() -> None:
    edited = _solid((40, 90, 180))
    edited.paste(_solid((200, 40, 60), (128, 128)), (40, 40))
    difference = compare_images(_encode(_solid((40, 90, 180))), _encode(edited))

    assert difference.changed
    assert difference.mean_absolute_difference > UNCHANGED_THRESHOLD


def test_a_hue_change_at_equal_brightness_counts_as_a_change() -> None:
    """A mug recoloured from blue to burgundy.

    These two colours have nearly identical luminance, so a greyscale
    comparison scored the edit below the threshold and called a real change
    "unchanged" - which is the same failure the vision verifier already made.
    """
    blue = (40, 90, 180)
    burgundy = (150, 30, 60)
    edited = _solid(blue)
    edited.paste(_solid(burgundy, (128, 128)), (40, 40))

    difference = compare_images(_encode(_solid(blue)), _encode(edited))

    assert difference.changed
    assert difference.mean_absolute_difference > UNCHANGED_THRESHOLD


def test_differing_output_sizes_still_compare() -> None:
    """Some workflows return a different resolution; that is not a failure."""
    difference = compare_images(
        _encode(_solid((30, 30, 30), (512, 512))),
        _encode(_solid((220, 220, 220), (256, 384))),
    )

    assert difference.comparable
    assert difference.changed


def test_an_unreadable_image_never_claims_nothing_changed() -> None:
    """Failing closed: an unreadable file must not stop a retry on a false
    certainty, so it reports incomparable and changed rather than unchanged."""
    difference = compare_images(b"not an image", _encode(_solid((0, 0, 0))))

    assert not difference.comparable
    assert difference.changed


def test_the_provenance_records_the_threshold_it_judged_against() -> None:
    difference = compare_images(_encode(_solid((10, 10, 10))), _encode(_solid((10, 10, 10))))
    provenance = difference.provenance()

    assert provenance["changed"] is False
    assert provenance["comparable"] is True
    assert provenance["threshold"] == UNCHANGED_THRESHOLD


def _mask(size: tuple[int, int], *boxes: tuple[int, int, int, int]) -> bytes:
    mask = Image.new("L", size, 0)
    for box in boxes:
        mask.paste(255, box)
    return _encode(mask)


GREY = (128, 128, 128)
RED = (255, 0, 0)


def _patched(size: tuple[int, int], box: tuple[int, int, int, int]) -> bytes:
    picture = _solid(GREY, size)
    picture.paste(RED, box)
    return _encode(picture)


def test_an_edit_check_calls_an_identical_or_re_encoded_result_unchanged() -> None:
    """Full-colour re-encoding stays under the threshold everywhere. Halved
    colour resolution smears a saturated hard edge across a few pixels, which
    one part can see; that errs toward "changed", so the model decides."""
    source = _solid((40, 90, 180), (400, 400))
    source.paste((200, 40, 60), (100, 100, 300, 300))
    gradient = Image.linear_gradient("L").resize((400, 400)).convert("RGB")

    identical = compare_edit(_encode(source), _encode(source))
    full_colour = compare_edit(_encode(source), _encode(source, "JPEG", quality=90, subsampling=0))
    smooth = compare_edit(_encode(gradient), _encode(gradient, "JPEG", quality=90))
    smeared = compare_edit(_encode(source), _encode(source, "JPEG", quality=90))

    assert identical.comparable and not identical.changed
    assert full_colour.comparable and not full_colour.changed
    assert smooth.comparable and not smooth.changed
    assert smeared.comparable and smeared.changed


def test_a_small_edit_is_a_change_however_little_of_the_picture_it_covers() -> None:
    """A sixteenth of the width, and a sixty-fourth of it: the whole-picture
    average of each is under the threshold, and each is a real change."""
    source = _encode(_solid(GREY, (128, 128)))

    for box in ((60, 60, 68, 68), (60, 60, 62, 62)):
        edit = compare_edit(source, _patched((128, 128), box))
        whole = compare_images(source, _patched((128, 128), box))

        assert not whole.changed, "the whole average should hide this edit"
        assert edit.comparable and edit.changed, box
        assert edit.largest_local_difference is not None
        assert edit.largest_local_difference > UNCHANGED_THRESHOLD


def test_a_pixel_the_selection_only_partly_covers_still_counts() -> None:
    """A 4x4 mask selecting its top-left 3x3 covers a quarter of the last
    column of a 3x3 picture, and that column is where the change is."""
    source = _solid((40, 90, 180), (3, 3))
    result = source.copy()
    for row in range(3):
        result.putpixel((2, row), (200, 40, 60))

    edit = compare_edit(_encode(source), _encode(result), mask=_mask((4, 4), (0, 0, 3, 3)))

    assert edit.comparable and edit.changed


def test_a_change_where_the_selection_excludes_is_not_a_change() -> None:
    """Inverted, the mask selects everything but its centre: a change only in
    that hole leaves the selection unchanged, and one in the ring does not."""
    mask = _mask((128, 128), (48, 48, 80, 80))
    source = _encode(_solid(GREY, (128, 128)))

    in_hole = compare_edit(source, _patched((128, 128), (48, 48, 80, 80)), mask=mask, invert=True)
    in_ring = compare_edit(source, _patched((128, 128), (8, 8, 16, 16)), mask=mask, invert=True)

    assert in_hole.comparable and not in_hole.changed
    assert in_ring.comparable and in_ring.changed


def test_a_change_between_two_selected_shapes_is_not_a_change() -> None:
    mask = _mask((128, 128), (10, 10, 30, 30), (90, 90, 110, 110))
    source = _encode(_solid(GREY, (128, 128)))

    between = compare_edit(source, _patched((128, 128), (55, 55, 70, 70)), mask=mask)
    inside = compare_edit(source, _patched((128, 128), (92, 92, 100, 100)), mask=mask)

    assert between.comparable and not between.changed
    assert inside.comparable and inside.changed


def test_a_result_at_another_size_compares_the_same_places() -> None:
    source = _solid(GREY, (200, 200))
    source.paste(RED, (20, 20, 60, 60))
    larger = source.resize((400, 400), Image.Resampling.NEAREST)
    moved = _solid(GREY, (400, 400))
    moved.paste(RED, (240, 240, 320, 320))

    same = compare_edit(_encode(source), _encode(larger), mask=_mask((200, 200), (0, 0, 200, 200)))
    different = compare_edit(_encode(source), _encode(moved))

    assert same.comparable and not same.changed
    assert different.comparable and different.changed


def test_an_edit_check_on_an_empty_or_unreadable_mask_or_image_proves_nothing() -> None:
    picture = _encode(_solid((40, 90, 180), (64, 64)))
    empty = _encode(Image.new("L", (64, 64), 0))
    everything = _encode(Image.new("L", (64, 64), 255))

    for difference in (
        compare_edit(picture, picture, mask=empty),
        compare_edit(picture, picture, mask=everything, invert=True),
        compare_edit(picture, picture, mask=b"not a mask"),
        compare_edit(b"not an image", picture),
        compare_edit(picture, b"not an image"),
    ):
        assert not difference.comparable and difference.changed


def test_an_edit_check_records_its_largest_local_difference() -> None:
    source = _encode(_solid(GREY, (128, 128)))
    provenance = compare_edit(source, _patched((128, 128), (60, 60, 68, 68))).provenance()

    assert provenance["changed"] is True
    assert provenance["threshold"] == UNCHANGED_THRESHOLD
    largest = provenance["largest_local_difference"]
    assert isinstance(largest, float) and largest > UNCHANGED_THRESHOLD


def test_changed_areas_are_counted_so_coverage_can_be_checked() -> None:
    """Two things changed in two places is two areas, not one bigger one.

    A review reading a bounded list of what changed needs to know whether
    everything that moved was named; the count of separate changed areas is
    what lets it ask.
    """

    source = _solid((200, 200, 200))
    one = source.copy()
    one.paste((20, 40, 200), (16, 16, 64, 64))
    two = one.copy()
    two.paste((240, 200, 20), (176, 176, 232, 232))
    touching = source.copy()
    touching.paste((20, 40, 200), (16, 16, 120, 64))

    single = compare_edit(_encode(source), _encode(one))
    double = compare_edit(_encode(source), _encode(two))
    spanning = compare_edit(_encode(source), _encode(touching))
    unchanged = compare_edit(_encode(source), _encode(source))

    assert single.changed_regions == 1
    assert double.changed_regions == 2
    # One thing wide enough to cross several parts of the grid is still one area.
    assert spanning.changed_regions == 1
    assert unchanged.changed_regions == 0
    assert single.provenance()["changed_regions"] == 1


def _patched_twice(
    size: tuple[int, int],
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bytes:
    picture = _solid(GREY, size)
    picture.paste(RED, first)
    picture.paste(RED, second)
    return _encode(picture)


def test_a_changed_area_says_where_it_is_not_only_that_it_is_there() -> None:
    """Counting told a reader that something unnamed moved; bounds say where."""

    source = _encode(_solid(GREY, (128, 128)))
    difference = compare_edit(source, _patched((128, 128), (32, 32, 96, 96)))

    assert difference.changed_areas is not None
    assert difference.changed_regions == len(difference.changed_areas)
    assert len(difference.changed_areas) == 1
    area = difference.changed_areas[0]
    # The patch is the middle half of the picture, so its box encloses that and
    # stays inside the picture.
    assert 0.0 <= area.left <= 0.3 and 0.0 <= area.top <= 0.3
    assert 0.7 <= area.right <= 1.0 and 0.7 <= area.bottom <= 1.0
    assert area.left < area.right and area.top < area.bottom
    assert difference.provenance()["changed_areas"] == [area.provenance()]


def test_two_separate_changes_are_two_areas_ordered_from_the_top_left() -> None:
    """Two things in two places are two boxes, and the order does not wander."""

    source = _encode(_solid(GREY, (128, 128)))
    difference = compare_edit(
        source, _patched_twice((128, 128), (8, 8, 24, 24), (96, 100, 120, 124))
    )

    assert difference.changed_areas is not None
    assert len(difference.changed_areas) == 2
    first, second = difference.changed_areas
    assert (first.top, first.left) < (second.top, second.left)
    assert first.bottom <= second.top or first.right <= second.left
    assert difference.changed_regions == 2


def test_one_subject_spanning_many_parts_is_still_one_area() -> None:
    """A wide change is one box, not one per part it touches."""

    source = _encode(_solid(GREY, (128, 128)))
    difference = compare_edit(source, _patched((128, 128), (4, 56, 124, 72)))

    assert difference.changed_areas is not None
    assert len(difference.changed_areas) == 1
    area = difference.changed_areas[0]
    assert area.right - area.left > area.bottom - area.top


def test_a_crop_covers_its_area_and_a_margin_around_it() -> None:
    """The grid is coarse, so the crop has to show more than the moved parts."""

    picture = _patched((128, 128), (32, 32, 96, 96))
    area = ChangedArea(left=0.25, top=0.25, right=0.75, bottom=0.75)

    tight, tight_cut = crop_changed_area(picture, area, margin=0.0)
    widened, widened_cut = crop_changed_area(picture, area, margin=1.0 / 32)

    with Image.open(io.BytesIO(tight)) as image:
        assert image.size == (64, 64)
    with Image.open(io.BytesIO(widened)) as image:
        assert image.size == (72, 72)
    # The extent reported is the one actually cut, measured back from those
    # pixels rather than from the box that was asked for.
    assert (widened_cut.right - widened_cut.left) * 128 == 72
    assert (tight_cut.right - tight_cut.left) * 128 == 64


def test_a_crop_against_an_edge_widens_inward_instead_of_falling_off() -> None:
    picture = _patched((128, 128), (0, 0, 32, 32))
    area = ChangedArea(left=0.0, top=0.0, right=0.25, bottom=0.25)

    cropped, cut = crop_changed_area(picture, area, margin=0.25)

    with Image.open(io.BytesIO(cropped)) as image:
        assert image.size == (64, 64), "the box is clamped to the picture, never past it"


def test_a_crop_of_one_grid_part_is_never_empty() -> None:
    """A single changed part is a thin box; a crop of nothing answers nothing."""

    picture = _patched((128, 128), (60, 60, 64, 64))
    area = ChangedArea(left=0.5, top=0.5, right=0.5, bottom=0.5)

    cropped, _cut = crop_changed_area(picture, area, margin=0.0)

    with Image.open(io.BytesIO(cropped)) as image:
        assert image.size[0] >= 1 and image.size[1] >= 1


def test_a_crop_shows_the_change_rather_than_the_whole_picture() -> None:
    source = _encode(_solid(GREY, (128, 128)))
    edited = _patched((128, 128), (80, 80, 112, 112))
    difference = compare_edit(source, edited)
    assert difference.changed_areas is not None
    area = difference.changed_areas[0]

    before, _before_cut = crop_changed_area(source, area, margin=1.0 / 32)
    after, _after_cut = crop_changed_area(edited, area, margin=1.0 / 32)

    assert compare_images(before, after).changed, "the crop must contain the edit"
    with Image.open(io.BytesIO(after)) as image:
        assert image.size[0] < 128 and image.size[1] < 128, "and less than everything else"


def test_a_negative_margin_is_refused_rather_than_shrinking_the_crop() -> None:
    picture = _patched((128, 128), (32, 32, 96, 96))
    area = ChangedArea(left=0.25, top=0.25, right=0.75, bottom=0.75)

    with pytest.raises(ValueError):
        crop_changed_area(picture, area, margin=-0.1)
