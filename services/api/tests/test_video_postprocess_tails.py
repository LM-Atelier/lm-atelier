from local_lm.video_postprocess_tails import (
    ADMITTED_VIDEO_POSTPROCESS_TAILS,
    analyze_video_postprocess_tail,
)


def test_empty_catalogue_matches_nothing() -> None:
    assert ADMITTED_VIDEO_POSTPROCESS_TAILS == ()
    assert analyze_video_postprocess_tail(None) is None
    assert analyze_video_postprocess_tail({}) is None
    assert (
        analyze_video_postprocess_tail(
            {"1": {"class_type": "ImageUpscaleWithModel", "inputs": {}}},
        )
        is None
    )
    assert (
        analyze_video_postprocess_tail(
            {"nodes": [{"type": "RIFE VFI"}, {"type": "SaveVideo"}]},
        )
        is None
    )
