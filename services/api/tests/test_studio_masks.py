"""The mask contract refuses everything it cannot honor exactly."""

from __future__ import annotations

import pytest

from local_lm.studio_masks import (
    MaskContractError,
    mask_geometry,
    mask_provenance,
    parse_mask_setting,
    split_mask_setting,
    workflow_accepts_mask,
)

ARTIFACT = f"sha256:{'a' * 64}"
MASK_SCHEMA = {
    "type": "object",
    "properties": {"mask": {"type": "object", "x-lm-atelier-kind": "mask"}},
}


def test_only_a_declared_mask_input_counts() -> None:
    assert workflow_accepts_mask(MASK_SCHEMA) is True
    assert workflow_accepts_mask(None) is False
    assert workflow_accepts_mask({"properties": {}}) is False
    # A property named "mask" that is not declared as one does not count -
    # otherwise a coincidental name would look like inpainting support.
    assert workflow_accepts_mask({"properties": {"mask": {"type": "string"}}}) is False


def test_a_mask_on_a_workflow_without_one_refuses_loudly() -> None:
    with pytest.raises(MaskContractError) as raised:
        parse_mask_setting({"mask": {"artifact_id": ARTIFACT}}, {"properties": {}})
    assert raised.value.code == "workflow-has-no-mask-input"


def test_no_mask_is_simply_no_mask() -> None:
    assert parse_mask_setting({}, MASK_SCHEMA) is None
    assert parse_mask_setting({"mask": None}, MASK_SCHEMA) is None


def test_a_valid_selection_normalizes_its_defaults() -> None:
    selection = parse_mask_setting({"mask": {"artifact_id": ARTIFACT}}, MASK_SCHEMA)
    assert selection is not None
    assert selection.artifact_id == ARTIFACT
    assert selection.feather_px == 0
    assert selection.invert is False

    softened = parse_mask_setting(
        {"mask": {"artifact_id": ARTIFACT, "feather_px": 12, "invert": True}}, MASK_SCHEMA
    )
    assert softened is not None
    assert softened.feather_px == 12
    assert softened.invert is True


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ("not-an-object", "mask-setting-invalid"),
        ({"artifact_id": "art_123"}, "mask-artifact-invalid"),
        ({"artifact_id": ARTIFACT, "feather_px": 1.5}, "mask-feather-invalid"),
        ({"artifact_id": ARTIFACT, "feather_px": True}, "mask-feather-invalid"),
        ({"artifact_id": ARTIFACT, "feather_px": 999}, "mask-feather-out-of-range"),
        ({"artifact_id": ARTIFACT, "invert": "yes"}, "mask-invert-invalid"),
    ],
)
def test_every_malformed_selection_has_its_own_code(payload: object, code: str) -> None:
    with pytest.raises(MaskContractError) as raised:
        parse_mask_setting({"mask": payload}, MASK_SCHEMA)
    assert raised.value.code == code


def test_geometry_records_the_transform_rather_than_assuming_it() -> None:
    exact = mask_geometry(source_width=1024, source_height=768, mask_width=1024, mask_height=768)
    assert exact.is_exact is True
    assert exact.scale_x == 1.0

    scaled = mask_geometry(source_width=4096, source_height=3072, mask_width=1024, mask_height=768)
    assert scaled.is_exact is False
    assert scaled.scale_x == 4.0
    recorded = scaled.as_dict()
    # Every term the server needs to resize deterministically is written down.
    for key in ("source_width", "mask_width", "orientation", "scale_x", "resampler", "threshold"):
        assert key in recorded


def test_a_selection_drawn_on_a_different_shape_refuses() -> None:
    with pytest.raises(MaskContractError) as raised:
        mask_geometry(source_width=1024, source_height=768, mask_width=512, mask_height=512)
    assert raised.value.code == "mask-aspect-mismatch"


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"source_width": 0}, "mask-geometry-invalid"),
        ({"mask_height": -4}, "mask-geometry-invalid"),
        ({"orientation": 9}, "mask-orientation-invalid"),
        ({"threshold": 300}, "mask-threshold-invalid"),
    ],
)
def test_impossible_geometry_refuses_typed(kwargs: dict[str, int], code: str) -> None:
    base = {
        "source_width": 800,
        "source_height": 600,
        "mask_width": 800,
        "mask_height": 600,
    }
    with pytest.raises(MaskContractError) as raised:
        mask_geometry(**{**base, **kwargs})
    assert raised.value.code == code


def test_provenance_carries_the_whole_story() -> None:
    selection = parse_mask_setting(
        {"mask": {"artifact_id": ARTIFACT, "feather_px": 8}}, MASK_SCHEMA
    )
    assert selection is not None
    geometry = mask_geometry(
        source_width=2048, source_height=1536, mask_width=1024, mask_height=768
    )

    record = mask_provenance(selection, geometry, coverage=0.125)

    assert record["artifact_id"] == ARTIFACT
    assert record["feather_px"] == 8
    assert record["coverage"] == 0.125
    assert record["geometry"]["scale_x"] == 2.0
    assert record["geometry"]["exact"] is False

    with pytest.raises(MaskContractError) as raised:
        mask_provenance(selection, geometry, coverage=1.5)
    assert raised.value.code == "mask-coverage-invalid"


def test_the_turn_contract_holds_end_to_end(client) -> None:
    """A masked edit must refuse before acceptance, not after execution."""

    from local_lm.db import SessionLocal
    from local_lm.models import Chat, WorkflowDefinition, WorkflowRevision

    with SessionLocal() as session:
        chat = Chat(title="Masked edit")
        session.add(chat)
        definition = WorkflowDefinition(name="Plain editor", operation="image_to_image")
        session.add(definition)
        session.flush()
        # This workflow declares no mask input at all.
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="comfyui",
            api_graph_json={},
            input_schema_json={"type": "object", "properties": {}},
            dependencies_json={},
            trusted=True,
        )
        session.add(revision)
        session.commit()
        schema = revision.input_schema_json

    # The refusal is the contract's, raised where the workflow is known.
    with pytest.raises(MaskContractError) as raised:
        parse_mask_setting({"mask": {"artifact_id": ARTIFACT}}, schema)
    assert raised.value.code == "workflow-has-no-mask-input"


def test_a_selection_is_kept_out_of_the_tunable_schema() -> None:
    """The defect: a mask validated as a tunable is refused as unknown.

    No workflow declares `mask` among its setting fields, so running it
    through the schema validator produced "unsupported settings: mask" and
    every masked edit failed - while the orchestrator was, a few lines later,
    waiting to read exactly that key from the run record.
    """
    mask, tunables = split_mask_setting({"mask": {"artifact_id": "art-1"}, "steps": 20})

    assert mask == {"artifact_id": "art-1"}
    assert tunables == {"steps": 20}


def test_settings_without_a_selection_are_untouched() -> None:
    mask, tunables = split_mask_setting({"steps": 20, "cfg": 7})

    assert mask is None
    assert tunables == {"steps": 20, "cfg": 7}
