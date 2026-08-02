"""The pixel pre-check that a verifier cannot talk its way past."""

from __future__ import annotations

import io

from PIL import Image

from local_lm.image_edit_difference import UNCHANGED_THRESHOLD, compare_images


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
    """The owner's mug: blue recoloured burgundy.

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
