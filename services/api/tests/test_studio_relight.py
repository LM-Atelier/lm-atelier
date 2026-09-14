"""Relight is finished against its source by exact, checkable pixel arithmetic."""

from __future__ import annotations

import hashlib
import io
from typing import Any

import pytest
from PIL import Image, ImageStat

from local_lm.studio_region_edit import RegionEditError
from local_lm.studio_relight import (
    LIGHTING_ADAPTER,
    RelightContractError,
    RelightFinish,
    RelightSetting,
    finish_relight,
    is_lighting_adapter,
    parse_relight_setting,
    require_relight_turn,
    split_relight_setting,
    warmth_grade,
)

SOURCE_ID = f"sha256:{'b' * 64}"


def _png(image: Image.Image, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **options)
    return buffer.getvalue()


def _manifest(**overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "sha256": LIGHTING_ADAPTER.sha256,
        "comfy_name": LIGHTING_ADAPTER.filename,
        "expected_sha256": {LIGHTING_ADAPTER.filename: LIGHTING_ADAPTER.sha256},
    }
    manifest.update(overrides)
    return manifest


def _identity(**overrides: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "provider": LIGHTING_ADAPTER.provider,
        "remote_id": LIGHTING_ADAPTER.remote_id,
        "revision": LIGHTING_ADAPTER.revision,
        "manifest": _manifest(),
    }
    identity.update(overrides)
    return identity


def test_only_the_exact_adapter_file_is_the_lighting_adapter() -> None:
    assert is_lighting_adapter(**_identity()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "civitai"},
        {"remote_id": "someone-else/Qwen-Edit-2509-Multi-Angle-Lighting"},
        {"revision": "main"},
        {"manifest": _manifest(sha256="0" * 64)},
        {"manifest": _manifest(comfy_name="lighting.safetensors")},
        {"manifest": _manifest(expected_sha256={})},
        {"manifest": None},
    ],
)
def test_a_name_or_repository_alone_does_not_qualify(overrides: dict[str, Any]) -> None:
    assert is_lighting_adapter(**_identity(**overrides)) is False


def test_a_relight_request_is_kept_out_of_the_workflow_fields() -> None:
    relight, rest = split_relight_setting({"relight": {"direction": "left"}, "seed": 3})

    assert relight == {"direction": "left"}
    assert rest == {"seed": 3}


def test_a_valid_relight_request_normalizes() -> None:
    assert parse_relight_setting({}) is None
    setting = parse_relight_setting(
        {"relight": {"direction": "top", "intensity": 1, "kelvin": 4500}}
    )
    assert setting == RelightSetting(direction="top", intensity=1.0, kelvin=4500)
    assert setting.as_dict() == {"direction": "top", "intensity": 1.0, "kelvin": 4500}
    assert parse_relight_setting({"relight": {"direction": "left"}}) == RelightSetting(
        direction="left", intensity=1.0, kelvin=None
    )


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("left", "relight-setting-invalid"),
        ({"direction": "left", "angle": 30}, "relight-setting-invalid"),
        ({"direction": "bottom"}, "relight-direction-invalid"),
        ({"direction": "left", "intensity": 0.1}, "relight-intensity-invalid"),
        ({"direction": "left", "intensity": 1.5}, "relight-intensity-invalid"),
        ({"direction": "left", "intensity": True}, "relight-intensity-invalid"),
        ({"direction": "left", "intensity": float("nan")}, "relight-intensity-invalid"),
        ({"direction": "left", "kelvin": 2700}, "relight-warmth-invalid"),
        ({"direction": "left", "kelvin": 4500.5}, "relight-warmth-invalid"),
        ({"direction": "left", "kelvin": True}, "relight-warmth-invalid"),
    ],
)
def test_every_malformed_relight_request_has_its_own_code(raw: object, code: str) -> None:
    with pytest.raises(RelightContractError) as raised:
        parse_relight_setting({"relight": raw})
    assert raised.value.code == code


SETTING = RelightSetting(direction="left", intensity=0.5, kelvin=None)


def test_a_relight_turn_is_a_two_picture_edit_with_the_adapter_at_full_strength() -> None:
    require_relight_turn(
        SETTING,
        operation="image_to_image",
        source_count=2,
        loras=[{"asset_id": "asset_light"}],
        adapter_asset_ids=["asset_light"],
    )


@pytest.mark.parametrize(
    ("operation", "count", "loras", "code"),
    [
        ("image_to_image", 1, [{"asset_id": "asset_light"}], "relight-needs-source-and-map"),
        ("text_to_image", 2, [{"asset_id": "asset_light"}], "relight-needs-source-and-map"),
        ("image_to_image", 2, [], "relight-adapter-missing"),
        ("image_to_image", 2, [{"asset_id": "asset_other"}], "relight-adapter-missing"),
        (
            "image_to_image",
            2,
            [{"asset_id": "asset_light", "enabled": False}],
            "relight-adapter-missing",
        ),
        (
            "image_to_image",
            2,
            [{"asset_id": "asset_light", "model_strength": 0.6}],
            "relight-adapter-strength",
        ),
    ],
)
def test_a_relight_turn_that_cannot_relight_refuses(
    operation: str, count: int, loras: list[dict[str, Any]], code: str
) -> None:
    with pytest.raises(RelightContractError) as raised:
        require_relight_turn(
            SETTING,
            operation=operation,
            source_count=count,
            loras=loras,
            adapter_asset_ids=["asset_light"],
        )
    assert raised.value.code == code


def _finish(setting: RelightSetting, source: bytes) -> RelightFinish:
    return RelightFinish(setting=setting, source_artifact_id=SOURCE_ID, source=source)


def test_full_intensity_keeps_the_relit_picture() -> None:
    source = Image.new("RGB", (64, 48), (200, 180, 160))
    relit = Image.new("RGB", (96, 72), (40, 60, 80))

    finished = finish_relight(_finish(RelightSetting("left", 1.0, None), _png(source)), _png(relit))

    out = Image.open(io.BytesIO(finished.content))
    assert out.format == "PNG"
    assert out.size == (64, 48)
    assert out.getpixel((10, 10)) == (40, 60, 80)


def test_a_gentler_light_is_a_mix_toward_the_source() -> None:
    source = Image.new("RGB", (64, 48), (200, 180, 160))
    relit = Image.new("RGB", (64, 48), (40, 60, 80))

    finished = finish_relight(_finish(RelightSetting("left", 0.5, None), _png(source)), _png(relit))

    assert Image.open(io.BytesIO(finished.content)).getpixel((5, 5)) == (120, 120, 120)


def test_the_record_says_how_the_picture_was_finished() -> None:
    relit = _png(Image.new("RGB", (96, 72), (40, 60, 80)))

    record = finish_relight(
        _finish(RelightSetting("top", 0.75, 4500), _png(Image.new("RGB", (64, 48)))), relit
    ).record

    assert record == {
        "direction": "top",
        "intensity": 0.75,
        "kelvin": 4500,
        "source_artifact_id": SOURCE_ID,
        "mix": "pixel",
        "grade": "white_balance",
        "result_sha256": hashlib.sha256(relit).hexdigest(),
        "result_width": 96,
        "result_height": 72,
        "result_resampler": "lanczos",
        "width": 64,
        "height": 48,
    }


def test_neutral_warmth_changes_nothing_and_a_warm_light_warms() -> None:
    picture = Image.new("RGB", (8, 8), (128, 128, 128))

    assert warmth_grade(picture, 6500).getpixel((0, 0)) == (128, 128, 128)
    red, green, blue = warmth_grade(picture, 4500).getpixel((0, 0))
    assert red > 128 > blue
    cool_red, _, cool_blue = warmth_grade(picture, 7500).getpixel((0, 0))
    assert cool_blue > 128 > cool_red


def test_warmth_keeps_the_brightness() -> None:
    picture = Image.new("RGB", (8, 8), (120, 110, 100))
    before = ImageStat.Stat(picture.convert("L")).mean[0]

    after = ImageStat.Stat(warmth_grade(picture, 4500).convert("L")).mean[0]

    assert abs(after - before) <= 2


def test_a_transparent_source_keeps_its_transparency() -> None:
    source = Image.new("RGBA", (64, 48), (200, 20, 20, 0))
    source.putpixel((5, 5), (200, 20, 20, 255))

    out = Image.open(
        io.BytesIO(
            finish_relight(
                _finish(RelightSetting("right", 0.5, 7500), _png(source)),
                _png(Image.new("RGB", (64, 48), (10, 200, 30))),
            ).content
        )
    )

    assert out.mode == "RGBA"
    assert out.getpixel((5, 5))[3] == 255
    assert out.getpixel((30, 30))[3] == 0


def test_the_source_colour_profile_is_kept() -> None:
    profile = b"neutral-colour-profile"

    finished = finish_relight(
        _finish(SETTING, _png(Image.new("RGB", (64, 48)), icc_profile=profile)),
        _png(Image.new("RGB", (64, 48), (1, 2, 3))),
    )

    assert Image.open(io.BytesIO(finished.content)).info.get("icc_profile") == profile


def test_a_reframed_result_is_refused_rather_than_stretched() -> None:
    with pytest.raises(RegionEditError) as raised:
        finish_relight(
            _finish(SETTING, _png(Image.new("RGB", (64, 48)))),
            _png(Image.new("RGB", (64, 64))),
        )
    assert raised.value.code == "region-result-reshaped"
