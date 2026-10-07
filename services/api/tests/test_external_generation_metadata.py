"""A picture's own settings text is read as plain claims, and nothing in it is followed."""

from __future__ import annotations

import json
import struct
import time
import zlib
from io import BytesIO

import pytest
from httpx import AsyncClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from sqlalchemy import func, select
from test_exif_text_metadata import ascii_tag, comment, exif_block, jpeg_with_exif, webp

from local_lm.artifacts import ArtifactStore
from local_lm.db import SessionLocal
from local_lm.external_generation_metadata import (
    BUDGET,
    ExternalGenerationMetadata,
    read_external_generation_metadata,
)
from local_lm.models import Artifact

_PROMPT = "a ceramic cup on a wooden table"
_NEGATIVE = "blurry, low contrast"
_SETTINGS_TEXT = (
    f"{_PROMPT}\n"
    f"Negative prompt: {_NEGATIVE}\n"
    "Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, Seed: 12345, "
    "Size: 512x768, Model hash: 0a1b2c3d4e, Model: neutral-model, "
    'Lora hashes: "neutral-detail: 0a1b2c3d4e5f, neutral-light: 5f4e3d2c1b0a", '
    "Denoising strength: 0.5, Version: 1.0"
)


def _png(*texts: tuple[str, str], compressed: tuple[str, ...] = ()) -> bytes:
    info = PngInfo()
    for keyword, text in texts:
        info.add_text(keyword, text, zip=keyword in compressed)
    output = BytesIO()
    Image.new("RGB", (8, 8), (90, 120, 150)).save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def _graph(**changes: object) -> dict[str, dict[str, object]]:
    graph: dict[str, dict[str, object]] = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 987654321012345678,
                "steps": 25,
                "cfg": 6.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "neutral.safetensors"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 896, "batch_size": 1},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": _PROMPT, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": _NEGATIVE, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }
    sampler_inputs = graph["3"]["inputs"]
    assert isinstance(sampler_inputs, dict)
    sampler_inputs.update(changes)
    return graph


def _claims(metadata: ExternalGenerationMetadata) -> list[tuple[str, object, str]]:
    return [(claim.key, claim.value, claim.source) for claim in metadata.claims]


def _ignored(metadata: ExternalGenerationMetadata) -> list[tuple[str, str]]:
    return [(item.name, item.reason) for item in metadata.ignored]


def test_a_settings_text_reads_as_claims_and_lists_what_it_does_not_use() -> None:
    metadata = read_external_generation_metadata(_png(("parameters", _SETTINGS_TEXT)))

    assert metadata.dialect == "parameters"
    assert metadata.parser_version == 2
    assert metadata.budget_version == BUDGET.version == 1
    assert _claims(metadata) == [
        ("prompt", _PROMPT, "prompt"),
        ("negative_prompt", _NEGATIVE, "Negative prompt"),
        ("steps", 20, "Steps"),
        ("sampler", "Euler a", "Sampler"),
        ("scheduler", "Karras", "Schedule type"),
        ("guidance", 7.0, "CFG scale"),
        ("seed", 12345, "Seed"),
        ("width", 512, "Size"),
        ("height", 768, "Size"),
        ("denoise", 0.5, "Denoising strength"),
    ]
    assert _ignored(metadata) == [
        ("Model hash", "names_a_file"),
        ("Model", "names_a_file"),
        ("Lora hashes", "names_a_file"),
        ("Version", "not_used"),
    ]
    assert metadata.warnings == ()
    assert metadata.digest is not None and metadata.digest.startswith("sha256:")


def test_the_same_text_reads_to_the_same_digest_and_other_text_to_another() -> None:
    first = read_external_generation_metadata(_png(("parameters", _SETTINGS_TEXT)))
    again = read_external_generation_metadata(_png(("parameters", _SETTINGS_TEXT)))
    other = read_external_generation_metadata(
        _png(("parameters", _SETTINGS_TEXT.replace("Seed: 12345", "Seed: 12346")))
    )

    assert first.digest == again.digest
    assert first.digest != other.digest


def test_a_settings_text_without_a_negative_prompt_or_with_windows_line_ends() -> None:
    plain = read_external_generation_metadata(
        _png(("parameters", f"{_PROMPT}\nSteps: 20, Seed: 7"))
    )
    windows = read_external_generation_metadata(
        _png(("parameters", _SETTINGS_TEXT.replace("\n", "\r\n")))
    )

    assert _claims(plain) == [
        ("prompt", _PROMPT, "prompt"),
        ("steps", 20, "Steps"),
        ("seed", 7, "Seed"),
    ]
    assert _claims(windows)[:2] == [
        ("prompt", _PROMPT, "prompt"),
        ("negative_prompt", _NEGATIVE, "Negative prompt"),
    ]


def test_values_that_are_not_their_kind_are_listed_as_malformed() -> None:
    text = f"{_PROMPT}\nSteps: twenty, CFG scale: 7, Seed: -1, Size: 512 by 768, Sampler: Euler"

    metadata = read_external_generation_metadata(_png(("parameters", text)))

    assert _claims(metadata) == [
        ("prompt", _PROMPT, "prompt"),
        ("guidance", 7.0, "CFG scale"),
        ("sampler", "Euler", "Sampler"),
    ]
    assert _ignored(metadata) == [
        ("Steps", "malformed"),
        ("Seed", "malformed"),
        ("Size", "malformed"),
    ]


def test_text_after_the_settings_line_is_listed_and_not_read() -> None:
    metadata = read_external_generation_metadata(
        _png(("parameters", f"{_PROMPT}\nSteps: 20\nTemplate: a neutral note"))
    )

    assert _claims(metadata) == [("prompt", _PROMPT, "prompt"), ("steps", 20, "Steps")]
    assert _ignored(metadata) == [("parameters", "trailing_text")]


def test_a_settings_text_without_a_settings_line_is_not_guessed_at() -> None:
    metadata = read_external_generation_metadata(_png(("parameters", _PROMPT)))

    assert metadata.dialect == "parameters"
    assert metadata.claims == ()
    assert _ignored(metadata) == [("parameters", "settings_unrecognized")]
    assert metadata.warnings == ("settings_unrecognized",)


def test_a_settings_line_past_its_length_is_listed_and_not_read() -> None:
    long_line = "Steps: 20, Note: " + "a" * (16 * 1024)
    no_colon = "Steps: 20, " + "a" * (16 * 1024 - 11)

    too_long = read_external_generation_metadata(_png(("parameters", f"{_PROMPT}\n{long_line}")))
    read_whole = read_external_generation_metadata(_png(("parameters", f"{_PROMPT}\n{no_colon}")))

    assert _claims(too_long) == [("prompt", _PROMPT, "prompt")]
    assert _ignored(too_long) == [("parameters", "too_long")]
    assert _claims(read_whole) == [("prompt", _PROMPT, "prompt"), ("steps", 20, "Steps")]


def test_a_comfyui_graph_reads_its_one_sampler_and_the_nodes_it_names() -> None:
    graph = json.dumps(_graph())
    metadata = read_external_generation_metadata(
        _png(("prompt", graph), ("workflow", '{"nodes": []}'))
    )

    assert metadata.dialect == "comfyui_prompt"
    assert _claims(metadata) == [
        # A seed past what a float holds exactly is read exactly.
        ("seed", 987654321012345678, "KSampler.seed"),
        ("steps", 25, "KSampler.steps"),
        ("guidance", 6.5, "KSampler.cfg"),
        ("sampler", "euler", "KSampler.sampler_name"),
        ("scheduler", "normal", "KSampler.scheduler"),
        ("denoise", 1.0, "KSampler.denoise"),
        ("prompt", _PROMPT, "KSampler.positive"),
        ("negative_prompt", _NEGATIVE, "KSampler.negative"),
        ("width", 1024, "EmptyLatentImage.width"),
        ("height", 896, "EmptyLatentImage.height"),
        ("batch", 1, "EmptyLatentImage.batch_size"),
    ]
    assert _ignored(metadata) == [
        ("KSampler.model", "not_used"),
        # Listed once though both prompt nodes have it.
        ("CLIPTextEncode.clip", "not_used"),
        ("CheckpointLoaderSimple", "names_a_file"),
        # A link to the loader is not itself a file name.
        ("VAEDecode", "not_used"),
        ("SaveImage", "not_used"),
        ("workflow", "workflow_graph"),
    ]
    assert metadata.warnings == ()


def test_a_value_from_another_node_is_listed_and_not_followed() -> None:
    graph = _graph(seed=["10", 0], positive=["11", 0])
    graph["10"] = {"class_type": "PrimitiveNode", "inputs": {"value": 4}}
    graph["11"] = {"class_type": "ConditioningCombine", "inputs": {}}

    metadata = read_external_generation_metadata(_png(("prompt", json.dumps(graph))))

    keys = [claim.key for claim in metadata.claims]
    assert "seed" not in keys and "prompt" not in keys
    assert ("KSampler.seed", "from_another_node") in _ignored(metadata)
    assert ("KSampler.positive", "not_plain_text") in _ignored(metadata)


def test_a_graph_with_no_sampler_or_several_claims_nothing() -> None:
    several = _graph()
    several["12"] = dict(several["3"])
    del_sampler = _graph()
    del del_sampler["3"]

    both = read_external_generation_metadata(_png(("prompt", json.dumps(several))))
    assert both.claims == ()
    assert both.warnings == ("several_samplers",)
    assert _ignored(both) == [("prompt", "several_samplers")]
    none = read_external_generation_metadata(_png(("prompt", json.dumps(del_sampler))))
    assert none.claims == ()
    assert none.warnings == ("no_sampler",)
    assert _ignored(none) == [("prompt", "no_sampler")]


@pytest.mark.parametrize(
    "graph_text",
    [
        "not json",
        '{"3": {"class_type": "KSampler", "class_type": "KSampler"}}',
        '{"3": {"class_type": "KSampler", "inputs": null}}',
        "[1, 2, 3]",
    ],
    ids=["not-json", "repeated-key", "inputs-not-an-object", "not-an-object"],
)
def test_a_malformed_graph_is_listed_and_claims_nothing(graph_text: str) -> None:
    metadata = read_external_generation_metadata(_png(("prompt", graph_text)))

    assert metadata.claims == ()
    assert ("prompt", "malformed") in _ignored(metadata)


def test_a_graph_past_the_parse_budget_is_not_read() -> None:
    deep: object = 0
    for _ in range(BUDGET.max_json_depth + 1):
        deep = [deep]
    wide = _graph()
    wide.update({str(index): {"inputs": {}} for index in range(100, 100 + BUDGET.max_json_nodes)})

    for graph in (_graph(steps=deep), wide):
        metadata = read_external_generation_metadata(_png(("prompt", json.dumps(graph))))
        assert metadata.claims == ()
        assert _ignored(metadata) == [("prompt", "too_complex")]


def test_values_of_the_wrong_kind_in_a_graph_are_malformed() -> None:
    graph = _graph(steps=True, cfg="7", seed=-3, sampler_name=12, denoise=1.5)

    metadata = read_external_generation_metadata(_png(("prompt", json.dumps(graph))))

    keys = {claim.key: claim.value for claim in metadata.claims}
    assert keys["guidance"] == 7.0
    for name in ("steps", "seed", "sampler_name", "denoise"):
        assert (f"KSampler.{name}", "malformed") in _ignored(metadata)


def test_settings_with_other_names_or_no_value_are_listed_not_dropped() -> None:
    text = (
        f"{_PROMPT}\nSteps: 20, Pad conds (v0): True, Kohya Hires.fix: False, Tiled, : 4, Seed: 7"
    )

    metadata = read_external_generation_metadata(_png(("parameters", text)))

    assert _claims(metadata) == [
        ("prompt", _PROMPT, "prompt"),
        ("steps", 20, "Steps"),
        ("seed", 7, "Seed"),
    ]
    assert _ignored(metadata) == [
        ("Pad conds (v0)", "not_used"),
        ("Kohya Hires.fix", "not_used"),
        # A setting written as its name alone, the way a switch is.
        ("Tiled", "not_used"),
        ("parameters", "malformed"),
    ]


@pytest.mark.parametrize(
    "middle",
    ['Model: cup"v2', 'Hires prompt: a 12" ceramic plate', 'Template: a\\"b', 'Note: "unclosed'],
    ids=["quote-inside", "quote-in-words", "escaped-quote", "unclosed-quote"],
)
def test_a_quote_inside_an_unquoted_value_is_part_of_that_value(middle: str) -> None:
    # The writer quotes a value only when it holds a comma, a colon or a line end.
    text = f"{_PROMPT}\nSteps: 20, {middle}, Seed: 42, Size: 512x512"

    metadata = read_external_generation_metadata(_png(("parameters", text)))

    assert _claims(metadata) == [
        ("prompt", _PROMPT, "prompt"),
        ("steps", 20, "Steps"),
        ("seed", 42, "Seed"),
        ("width", 512, "Size"),
        ("height", 512, "Size"),
    ]
    assert len(metadata.ignored) == 1


def test_a_quoted_value_keeps_its_commas_and_quotes() -> None:
    text = f'{_PROMPT}\nSteps: 20, Sampler: "Euler, \\"a\\"", Seed: 42'

    metadata = read_external_generation_metadata(_png(("parameters", text)))

    assert _claims(metadata) == [
        ("prompt", _PROMPT, "prompt"),
        ("steps", 20, "Steps"),
        ("sampler", 'Euler, "a"', "Sampler"),
        ("seed", 42, "Seed"),
    ]


def test_a_settings_line_of_whitespace_is_read_in_linear_time() -> None:
    # A pattern match retried from every offset of this run took over a second.
    line = "Steps: 20," + " " * (16 * 1024 - 10)
    payload = _png(("parameters", f"{_PROMPT}\n{line}"))

    started = time.process_time()
    metadata = read_external_generation_metadata(payload)
    spent = time.process_time() - started

    assert _claims(metadata) == [("prompt", _PROMPT, "prompt"), ("steps", 20, "Steps")]
    assert spent < 0.25


def test_a_no_break_space_or_a_wider_character_in_the_prompt_is_read() -> None:
    # Latin-1 text is stored as tEXt; anything wider as uncompressed iTXt.
    spaced = "a ceramic cup\xa0on a wooden table"
    dashed = "a ceramic cup \u2014 on a wooden table"
    latin = _png(("parameters", f"{spaced}\nSteps: 20, Seed: 7"))
    wide = _png(("parameters", f"{dashed}\nSteps: 20, Seed: 7"))
    assert b"iTXtparameters" in wide

    for payload, prompt in ((latin, spaced), (wide, dashed)):
        metadata = read_external_generation_metadata(payload)
        assert metadata.dialect == "parameters"
        assert _claims(metadata) == [
            ("prompt", prompt, "prompt"),
            ("steps", 20, "Steps"),
            ("seed", 7, "Seed"),
        ]


def test_a_chunk_holding_control_characters_is_listed_and_the_rest_read() -> None:
    metadata = read_external_generation_metadata(
        _png(("parameters", f"{_PROMPT}\nSteps: 20"), ("Comment", "neutral\x07note"))
    )

    assert _claims(metadata) == [("prompt", _PROMPT, "prompt"), ("steps", 20, "Steps")]
    assert _ignored(metadata) == [("Comment", "unsafe_text")]


def test_sampler_inputs_that_name_no_node_are_listed_as_malformed() -> None:
    graph = _graph(positive=_PROMPT, negative=["99", 0], latent_image={"width": 512})

    metadata = read_external_generation_metadata(_png(("prompt", json.dumps(graph))))

    keys = [claim.key for claim in metadata.claims]
    assert not {"prompt", "negative_prompt", "width"} & set(keys)
    for name in ("positive", "negative", "latent_image"):
        assert (f"KSampler.{name}", "malformed") in _ignored(metadata)


def test_other_inputs_of_the_nodes_read_are_listed_and_files_named_in_them() -> None:
    graph = _graph()
    graph["6"]["inputs"] = {"text": _PROMPT, "clip": ["4", 1], "lora_name": "cup.safetensors"}
    graph["5"]["inputs"] = {"width": 1024, "height": 896, "batch_size": 1, "tiling": 3}
    graph["11"] = {
        "class_type": "PowerLoraLoader",
        "inputs": {"slot_1": {"on": True, "file": "cup.safetensors"}, "model": ["4", 0]},
    }
    # Neither input's name says it names a file; the value inside does.
    graph["12"] = {"class_type": "LoraStack", "inputs": {"stack": [["cup.safetensors", 1.0]]}}

    metadata = read_external_generation_metadata(_png(("prompt", json.dumps(graph))))

    ignored = _ignored(metadata)
    assert ("CLIPTextEncode.lora_name", "names_a_file") in ignored
    assert ("EmptyLatentImage.tiling", "not_used") in ignored
    assert ("PowerLoraLoader", "names_a_file") in ignored
    assert ("LoraStack", "names_a_file") in ignored
    assert ("prompt", _PROMPT, "KSampler.positive") in _claims(metadata)


def test_windows_line_ends_in_a_graph_prompt_read_as_plain_ones() -> None:
    graph = _graph()
    graph["6"]["inputs"] = {"text": "a ceramic cup\r\non a wooden table", "clip": ["4", 1]}

    metadata = read_external_generation_metadata(_png(("prompt", json.dumps(graph))))

    assert ("prompt", "a ceramic cup\non a wooden table", "KSampler.positive") in _claims(metadata)


def test_a_float_written_out_in_full_is_read_as_its_number() -> None:
    text = (
        f"{_PROMPT}\n"
        "Steps: 20, CFG scale: 7.000000000000001, Denoising strength: 0.6000000000000001"
    )

    metadata = read_external_generation_metadata(_png(("parameters", text)))

    assert _claims(metadata) == [
        ("prompt", _PROMPT, "prompt"),
        ("steps", 20, "Steps"),
        ("guidance", 7.000000000000001, "CFG scale"),
        ("denoise", 0.6000000000000001, "Denoising strength"),
    ]


def test_an_empty_prompt_is_listed_as_empty_not_malformed() -> None:
    graph = _graph()
    graph["7"]["inputs"] = {"text": "", "clip": ["4", 1]}
    blank = f"{_PROMPT}\nNegative prompt: \nSteps: 20"

    from_graph = read_external_generation_metadata(_png(("prompt", json.dumps(graph))))
    from_text = read_external_generation_metadata(_png(("parameters", blank)))

    assert ("KSampler.negative", "empty") in _ignored(from_graph)
    assert ("Negative prompt", "empty") in _ignored(from_text)
    assert "negative_prompt" not in [claim.key for claim in from_graph.claims + from_text.claims]


def test_a_lone_surrogate_in_a_graph_leaves_out_only_that_value() -> None:
    # A JSON escape can carry half of a surrogate pair; the chunk stays ASCII.
    graph = _graph()
    positive = graph["6"]["inputs"]
    assert isinstance(positive, dict)
    positive["text"] = f"{_PROMPT} \ud83d"
    graph["10"] = {"class_type": "Note", "inputs": {"text": "\ud800"}}

    metadata = read_external_generation_metadata(_png(("prompt", json.dumps(graph))))

    keys = [claim.key for claim in metadata.claims]
    assert "prompt" not in keys
    assert {"seed", "steps", "negative_prompt"} <= set(keys)
    assert ("KSampler.positive", "unsafe_text") in _ignored(metadata)
    assert ("Note", "not_used") in _ignored(metadata)


def test_text_that_would_change_how_text_is_shown_is_left_out() -> None:
    # A right-to-left override and a soft hyphen are both format characters.
    graph = _graph()
    positive = graph["6"]["inputs"]
    assert isinstance(positive, dict)
    positive["text"] = "a ceramic cup ‮on a wooden table"
    hidden = read_external_generation_metadata(_png(("prompt", json.dumps(graph))))
    soft = read_external_generation_metadata(_png(("parameters", "a ceramic\xadcup\nSteps: 20")))

    assert "prompt" not in [claim.key for claim in hidden.claims]
    assert ("KSampler.positive", "unsafe_text") in _ignored(hidden)
    assert _claims(soft) == [("steps", 20, "Steps")]
    assert _ignored(soft) == [("prompt", "unsafe_text")]


def test_a_compressed_settings_chunk_is_listed_and_never_expanded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _png(("parameters", _SETTINGS_TEXT), compressed=("parameters",))

    def never(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("compressed text was expanded")

    monkeypatch.setattr(zlib, "decompress", never)
    metadata = read_external_generation_metadata(payload)

    assert metadata.dialect == "none"
    assert metadata.claims == ()
    assert _ignored(metadata) == [("parameters", "compressed")]
    assert metadata.warnings == ("no_settings_found",)


def test_a_repeated_settings_chunk_is_listed_and_the_first_one_read() -> None:
    other = _SETTINGS_TEXT.replace("Seed: 12345", "Seed: 99")
    metadata = read_external_generation_metadata(
        _png(("parameters", _SETTINGS_TEXT), ("parameters", other))
    )

    assert ("seed", 12345, "Seed") in _claims(metadata)
    assert ("parameters", "repeated") in _ignored(metadata)


def test_a_picture_without_settings_says_so() -> None:
    metadata = read_external_generation_metadata(_png(("Title", "A neutral fixture.")))

    assert metadata.dialect == "none"
    assert metadata.claims == ()
    assert _ignored(metadata) == [("Title", "not_settings")]
    assert metadata.warnings == ("no_settings_found",)


def test_a_file_that_is_not_a_png_is_not_read() -> None:
    metadata = read_external_generation_metadata(b"GIF89a" + b"\x00" * 32)

    assert metadata.dialect == "none"
    assert metadata.digest is None
    assert metadata.warnings == ("format_not_read",)


def _damaged_png() -> bytes:
    payload = bytearray(_png(("parameters", _SETTINGS_TEXT)))
    start = payload.index(b"tEXt")
    length = struct.unpack(">I", payload[start - 4 : start])[0]
    # Flip one bit of the text chunk's checksum.
    payload[start + 4 + length] ^= 0x01
    return bytes(payload)


def test_a_png_with_damaged_text_is_refused_whole() -> None:
    with pytest.raises(ValueError):
        read_external_generation_metadata(_damaged_png())


async def _uploaded(client: AsyncClient, content: bytes, media_type: str = "image/png") -> str:
    response = await client.post(
        "/api/artifacts", files={"file": ("picture.png", content, media_type)}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _settings(artifact_id: str) -> str:
    return f"/api/artifacts/{artifact_id}/generation-settings"


async def test_the_route_answers_with_the_claims_and_keeps_nothing(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", _SETTINGS_TEXT)))
    with SessionLocal() as session:
        artifacts_before = session.scalar(select(func.count()).select_from(Artifact))
        row = session.get(Artifact, artifact_id)
        assert row is not None
        metadata_before = dict(row.metadata_json)

    response = await client.get(_settings(artifact_id))

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["dialect"] == "parameters"
    assert body["parser_version"] == 2
    assert body["budget_version"] == 1
    assert body["digest"].startswith("sha256:")
    assert body["claims"][:2] == [
        {"key": "prompt", "value": _PROMPT, "source": "prompt"},
        {"key": "negative_prompt", "value": _NEGATIVE, "source": "Negative prompt"},
    ]
    assert {"name": "Model", "reason": "names_a_file"} in body["ignored"]
    assert body["warnings"] == []
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Artifact)) == artifacts_before
        row = session.get(Artifact, artifact_id)
        assert row is not None
        assert row.metadata_json == metadata_before


async def test_the_route_sends_a_seed_past_what_a_browser_reads_exactly_as_text(
    client: AsyncClient,
) -> None:
    artifact_id = await _uploaded(client, _png(("prompt", json.dumps(_graph()))))

    response = await client.get(_settings(artifact_id))

    assert response.status_code == 200, response.text
    values = {claim["key"]: claim["value"] for claim in response.json()["claims"]}
    assert values["seed"] == "987654321012345678"
    assert values["steps"] == 25
    assert values["guidance"] == 6.5


async def test_the_route_reads_a_prompt_with_a_no_break_space(client: AsyncClient) -> None:
    prompt = "a ceramic cup\xa0on a wooden table"
    artifact_id = await _uploaded(client, _png(("parameters", f"{prompt}\nSteps: 20")))

    response = await client.get(_settings(artifact_id))

    assert response.status_code == 200, response.text
    assert response.json()["claims"] == [
        {"key": "prompt", "value": prompt, "source": "prompt"},
        {"key": "steps", "value": 20, "source": "Steps"},
    ]


async def test_the_route_refuses_a_picture_whose_text_is_damaged(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _damaged_png())

    response = await client.get(_settings(artifact_id))

    assert response.status_code == 422
    assert response.json()["code"] == "generation-settings-unreadable"
    assert _NEGATIVE not in response.text


async def test_the_route_does_not_open_a_video(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id = await _uploaded(client, b"\x00\x00\x00\x18ftypmp42", "video/mp4")

    def never(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("the video was read")

    monkeypatch.setattr(ArtifactStore, "verified_bytes", never)
    response = await client.get(_settings(artifact_id))

    assert response.status_code == 200, response.text
    assert response.json()["warnings"] == ["format_not_read"]
    assert response.json()["claims"] == []


async def test_the_route_answers_not_found_for_an_unknown_picture(client: AsyncClient) -> None:
    response = await client.get(_settings("sha256:" + "0" * 64))

    assert response.status_code == 404
    assert response.json()["code"] == "artifact-not-found"


@pytest.mark.parametrize("container", ["jpeg", "webp"])
def test_a_jpeg_or_webp_reads_the_same_claims_as_a_png(container: str) -> None:
    graph = json.dumps(_graph())
    block = exif_block([ascii_tag(0x0110, "prompt:" + graph)])
    text_block = exif_block([], comment(_SETTINGS_TEXT))

    def wrap(found: bytes) -> bytes:
        return jpeg_with_exif(found) if container == "jpeg" else webp((b"EXIF", found))

    from_text = read_external_generation_metadata(wrap(text_block))
    from_graph = read_external_generation_metadata(wrap(block))
    png_text = read_external_generation_metadata(_png(("parameters", _SETTINGS_TEXT)))
    png_graph = read_external_generation_metadata(_png(("prompt", graph)))

    assert (from_text.dialect, _claims(from_text)) == ("parameters", _claims(png_text))
    assert _ignored(from_text) == _ignored(png_text)
    assert (from_graph.dialect, _claims(from_graph)) == ("comfyui_prompt", _claims(png_graph))
    assert from_text.digest is not None


def test_a_camera_comment_is_listed_and_not_read_as_settings() -> None:
    payload = jpeg_with_exif(exif_block([], comment("Taken from the harbour wall")))

    metadata = read_external_generation_metadata(payload)

    assert metadata.dialect == "none"
    assert metadata.claims == ()
    assert _ignored(metadata) == [("UserComment", "not_settings")]
    assert metadata.warnings == ("no_settings_found",)


def test_a_settings_comment_beside_a_workflow_graph_reads_the_comment() -> None:
    block = exif_block(
        [ascii_tag(0x010F, "workflow:" + json.dumps({"nodes": []}))], comment(_SETTINGS_TEXT)
    )

    metadata = read_external_generation_metadata(jpeg_with_exif(block))

    assert metadata.dialect == "parameters"
    assert ("workflow", "workflow_graph") in _ignored(metadata)


def test_a_damaged_jpeg_is_refused_and_nothing_is_returned() -> None:
    with pytest.raises(ValueError):
        read_external_generation_metadata(b"\xff\xd8\xff\xe1\xff\xf0Exif\x00\x00")


async def test_the_route_reads_a_jpeg_s_settings(client: AsyncClient) -> None:
    picture = BytesIO()
    exif = Image.Exif()
    order = "utf-16-be" if exif.tobytes()[6:8] == b"MM" else "utf-16-le"
    exif[0x8769] = {0x9286: b"UNICODE\x00" + _SETTINGS_TEXT.encode(order)}
    Image.new("RGB", (8, 8), (90, 120, 150)).save(picture, "JPEG", exif=exif.tobytes())
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("picture.jpg", picture.getvalue(), "image/jpeg")}
    )
    assert uploaded.status_code == 201, uploaded.text

    response = await client.get(_settings(uploaded.json()["id"]))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dialect"] == "parameters"
    assert body["claims"][:2] == [
        {"key": "prompt", "value": _PROMPT, "source": "prompt"},
        {"key": "negative_prompt", "value": _NEGATIVE, "source": "Negative prompt"},
    ]
    assert body["warnings"] == []


def test_a_comment_in_the_first_directory_or_a_capitalised_graph_reads_like_a_png() -> None:
    graph = json.dumps(_graph())
    comment_first = exif_block([ascii_tag(0x9286, _SETTINGS_TEXT)])
    capitalised = exif_block([ascii_tag(0x010F, "Prompt:" + graph)])

    from_comment = read_external_generation_metadata(jpeg_with_exif(comment_first))
    from_graph = read_external_generation_metadata(webp((b"EXIF", capitalised)))

    png_text = read_external_generation_metadata(_png(("parameters", _SETTINGS_TEXT)))
    png_graph = read_external_generation_metadata(_png(("prompt", graph)))
    assert (from_comment.dialect, _claims(from_comment)) == ("parameters", _claims(png_text))
    assert (from_graph.dialect, _claims(from_graph)) == ("comfyui_prompt", _claims(png_graph))
