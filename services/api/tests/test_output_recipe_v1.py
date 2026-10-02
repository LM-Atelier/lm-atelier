"""The portable record format: one encoding, one digest, and nothing a reader would not write."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest

from local_lm.output_recipe_v1 import (
    OutputRecipeFormatError,
    open_output_recipe,
    record_digest,
    seal_output_recipe,
)


def _payload() -> dict[str, Any]:
    return {
        "schema": "lm-atelier-output-recipe-v1",
        "version": 1,
        "exported_by": {"application": "LM Atelier", "version": "9.9.9"},
        "output": {
            "sha256": "a" * 64,
            "size_bytes": 2048,
            "media_type": "image/png",
            "kind": "image",
            "index": 0,
            "count": 1,
            "engine": "comfyui",
            "node_id": "9",
            "collection": "images",
            "raster": {"width": 64, "height": 48},
        },
        "operation": "text_to_image",
        "prompt": {
            "included": True,
            "positive": "a red cube on a table",
            "negative": None,
            "omitted_reason": None,
        },
        "seed": {"value": 7, "binding": "bound"},
        "settings": {"bound": {"steps": 20}, "unbound": {}},
        "inputs": [],
        "workflow": None,
        "model": None,
        "loras": [],
        "removed": [],
        "not_recorded": ["executed_graph_sha256", "runtime_version"],
        "reproducibility": {
            "status": "incomplete",
            "missing": ["frozen_snapshot_absent", "workflow_unavailable"],
        },
    }


#: The same record written out by hand: sorted keys, no spaces, ASCII.
_UNSIGNED = (
    b'{"exported_by":{"application":"LM Atelier","version":"9.9.9"},"inputs":[],'
    b'"loras":[],"model":null,"not_recorded":["executed_graph_sha256","runtime_version"],'
    b'"operation":"text_to_image","output":{"collection":"images","count":1,'
    b'"engine":"comfyui","index":0,"kind":"image","media_type":"image/png","node_id":"9",'
    b'"raster":{"height":48,"width":64},"sha256":"' + b"a" * 64 + b'","size_bytes":2048},'
    b'"prompt":{"included":true,"negative":null,"omitted_reason":null,'
    b'"positive":"a red cube on a table"},"removed":[],"reproducibility":{"missing":'
    b'["frozen_snapshot_absent","workflow_unavailable"],"status":"incomplete"},'
    b'"schema":"lm-atelier-output-recipe-v1","seed":{"binding":"bound","value":7},'
    b'"settings":{"bound":{"steps":20},"unbound":{}},"version":1,"workflow":null}'
)
_DIGEST = "sha256:a94f81327a657c94cb8f655c5ab4b9c8ddb5a1ec7b7ed79e73b5475797d16ae5"


def test_a_record_has_exactly_one_encoding_and_a_pinned_digest() -> None:
    """The bytes and digest are written out here, not derived from the encoder.

    A test that compared the encoder with itself would pass whatever the
    encoding became, and every record already shared would stop verifying.
    """

    # The digest is the hash of the domain tag and the unsigned bytes.
    tagged = b"lm-atelier-output-recipe-v1\0" + _UNSIGNED
    assert "sha256:" + hashlib.sha256(tagged).hexdigest() == _DIGEST

    sealed = seal_output_recipe(_payload())

    assert sealed == b'{"digest":"' + _DIGEST.encode() + b'",' + _UNSIGNED[1:]
    assert record_digest(_payload()) == _DIGEST


def test_a_sealed_record_reads_back_as_written() -> None:
    sealed = seal_output_recipe(_payload())

    assert open_output_recipe(sealed) == {**_payload(), "digest": _DIGEST}


def test_one_changed_value_no_longer_matches_its_digest() -> None:
    sealed = seal_output_recipe(_payload())
    tampered = sealed.replace(b'"value":7', b'"value":8')

    assert tampered != sealed
    with pytest.raises(OutputRecipeFormatError, match="digest"):
        open_output_recipe(tampered)


def test_a_record_in_any_other_layout_is_refused() -> None:
    sealed = seal_output_recipe(_payload())

    with pytest.raises(OutputRecipeFormatError, match="canonical"):
        open_output_recipe(sealed.replace(b'"version":1', b'"version": 1'))


def test_a_repeated_key_is_refused_before_anything_else_is_read() -> None:
    sealed = seal_output_recipe(_payload())

    with pytest.raises(OutputRecipeFormatError, match="repeats"):
        open_output_recipe(sealed.replace(b'"version":1', b'"version":1,"version":1'))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unexpected",), "field"),
        (("output", "local_path"), "C:/somewhere"),
        (("prompt", "chat"), "words"),
    ],
)
def test_a_field_the_format_does_not_define_is_refused(path: tuple[str, ...], value: str) -> None:
    """Even correctly digested: a record carries only what version 1 names."""

    payload = _payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    signed = {**payload, "digest": record_digest(payload)}

    with pytest.raises(OutputRecipeFormatError, match="wrong fields"):
        open_output_recipe(_encode(signed))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("output", "kind", "audio"),
        ("seed", "binding", "guessed"),
        ("prompt", "omitted_reason", "secret"),
        ("reproducibility", "missing", ["no_such_reason"]),
    ],
)
def test_a_value_outside_its_vocabulary_is_refused(section: str, key: str, value: object) -> None:
    payload = _payload()
    payload[section][key] = value
    signed = {**payload, "digest": record_digest(payload)}

    with pytest.raises(OutputRecipeFormatError):
        open_output_recipe(_encode(signed))


def test_an_omitted_prompt_cannot_carry_its_text() -> None:
    payload = _payload()
    payload["prompt"] = {
        "included": False,
        "positive": "a red cube on a table",
        "negative": None,
        "omitted_reason": "chosen",
    }

    with pytest.raises(OutputRecipeFormatError, match="no text"):
        seal_output_recipe(payload)


def test_reproducibility_cannot_say_recorded_while_naming_what_is_missing() -> None:
    payload = _payload()
    payload["reproducibility"] = {"status": "recorded", "missing": ["workflow_unavailable"]}

    with pytest.raises(OutputRecipeFormatError, match="contradicts"):
        seal_output_recipe(payload)


def test_a_number_json_cannot_carry_is_refused() -> None:
    payload = _payload()
    payload["settings"]["bound"]["cfg"] = float("nan")

    with pytest.raises(OutputRecipeFormatError):
        seal_output_recipe(payload)


def test_a_record_is_sealed_once() -> None:
    payload = copy.deepcopy(_payload())
    payload["digest"] = _DIGEST

    with pytest.raises(OutputRecipeFormatError, match="once"):
        seal_output_recipe(payload)


def test_a_number_too_long_to_read_is_refused_as_a_malformed_record() -> None:
    with pytest.raises(OutputRecipeFormatError, match="canonical JSON"):
        open_output_recipe(b'{"version":' + b"7" * 5000 + b"}")


@pytest.mark.parametrize("version", [True, 1.0])
def test_a_version_that_only_compares_equal_to_one_is_refused(version: object) -> None:
    """True and 1.0 equal 1 in Python, and each has a canonical form of its own."""

    unsigned = {**_payload(), "version": version}
    content = _encode({**unsigned, "digest": record_digest(unsigned)})

    with pytest.raises(OutputRecipeFormatError, match="version 1"):
        open_output_recipe(content)


def _encode(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
