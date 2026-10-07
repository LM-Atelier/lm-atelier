"""A workflow derived from one exact shipped template: the same edit with a third picture.

A neutral two-picture edit stands in for the shipped one: its instruction
node takes three pictures and the template links only two. The derived
workflow fills the third, changes nothing else, and exists only for those
exact bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_comfy_templates import _installed_templates, _registry

from local_lm import comfy_templates
from local_lm.comfy_templates import (
    ComfyTemplateRegistry,
    _Derivation,
    _derived_graph,
    _graph_sha256,
    _with_a_third_picture,
)
from local_lm.workflow_image_slots_v1 import SLOT_KEY

REVISION = "c" * 40
BASE = "image_two_picture_edit_example"
DERIVED = "image_two_picture_edit_example_three_pictures"


def _template() -> dict[str, Any]:
    """The first picture is encoded as the canvas; both condition the instruction."""

    def ui(
        node_id: int,
        kind: str,
        inputs: list[tuple[str, str, int | None]],
        outputs: list[tuple[str, list[int] | None]],
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "id": node_id,
            "type": kind,
            "pos": [node_id * 10, node_id * 10],
            "inputs": [{"name": name, "type": value, "link": link} for name, value, link in inputs],
            "outputs": [{"name": value, "type": value, "links": links} for value, links in outputs],
            "properties": {"cnr_id": "comfy-core", **extra.pop("properties", {})},
            **extra,
        }

    return {
        "last_node_id": 40,
        "last_link_id": 50,
        "nodes": [
            ui(
                1,
                "LoadImage",
                [],
                [("IMAGE", [1, 2]), ("MASK", None)],
                widgets_values=["a.png", "image"],
            ),
            ui(
                2,
                "LoadImage",
                [],
                [("IMAGE", [3]), ("MASK", None)],
                widgets_values=["b.png", "image"],
            ),
            ui(
                3,
                "CheckpointLoaderSimple",
                [],
                [("MODEL", [4]), ("CLIP", [5]), ("VAE", [6, 7])],
                properties={
                    "models": [
                        {
                            "directory": "checkpoints",
                            "name": "edit.safetensors",
                            "url": f"https://huggingface.co/owner/edit/resolve/{REVISION}/edit.safetensors",
                        }
                    ]
                },
                widgets_values=["edit.safetensors"],
            ),
            ui(4, "VAEEncode", [("pixels", "IMAGE", 1), ("vae", "VAE", 6)], [("LATENT", [8])]),
            ui(
                5,
                "ImageInstruction",
                [
                    ("clip", "CLIP", 5),
                    ("image1", "IMAGE", 2),
                    ("image2", "IMAGE", 3),
                    ("image3", "IMAGE", None),
                ],
                [("CONDITIONING", [9])],
            ),
            ui(
                6,
                "KSampler",
                [
                    ("model", "MODEL", 4),
                    ("positive", "CONDITIONING", 9),
                    ("latent_image", "LATENT", 8),
                ],
                [("LATENT", [10])],
            ),
            ui(7, "VAEDecode", [("samples", "LATENT", 10), ("vae", "VAE", 7)], [("IMAGE", [11])]),
            ui(8, "SaveImage", [("images", "IMAGE", 11)], [], widgets_values=["edited"]),
        ],
        "links": [
            [1, 1, 0, 4, 0, "IMAGE"],
            [2, 1, 0, 5, 1, "IMAGE"],
            [3, 2, 0, 5, 2, "IMAGE"],
            [4, 3, 0, 6, 0, "MODEL"],
            [5, 3, 1, 5, 0, "CLIP"],
            [6, 3, 2, 4, 1, "VAE"],
            [7, 3, 2, 7, 1, "VAE"],
            [8, 4, 0, 6, 2, "LATENT"],
            [9, 5, 0, 6, 1, "CONDITIONING"],
            [10, 6, 0, 7, 0, "LATENT"],
            [11, 7, 0, 8, 0, "IMAGE"],
        ],
    }


def _object_info() -> dict[str, Any]:
    def spec(required: dict[str, Any], optional: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "input": {"required": required, **({"optional": optional} if optional else {})},
            "input_order": {
                "required": list(required),
                **({"optional": list(optional)} if optional else {}),
            },
        }

    return {
        "LoadImage": spec({"image": [["a.png", "b.png"], {"image_upload": True}]}),
        "CheckpointLoaderSimple": spec({"ckpt_name": [["edit.safetensors"]]}),
        "VAEEncode": spec({"pixels": ["IMAGE"], "vae": ["VAE"]}),
        "ImageInstruction": spec(
            {"clip": ["CLIP"]}, {"image1": ["IMAGE"], "image2": ["IMAGE"], "image3": ["IMAGE"]}
        ),
        "KSampler": spec(
            {"model": ["MODEL"], "positive": ["CONDITIONING"], "latent_image": ["LATENT"]}
        ),
        "VAEDecode": spec({"samples": ["LATENT"], "vae": ["VAE"]}),
        "SaveImage": {**spec({"images": ["IMAGE"]}), "output_node": True},
    }


def _install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph: dict[str, Any]
) -> tuple[ComfyTemplateRegistry, str]:
    """Ship the template and register the derivation for exactly the bytes shipped."""

    registry = _registry(tmp_path)
    content = json.dumps(graph).encode("utf-8")
    (_installed_templates(registry) / f"{BASE}.json").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        comfy_templates,
        "_DERIVED_TEMPLATES",
        {DERIVED: _Derivation(BASE, digest, _with_a_third_picture)},
    )
    return registry, digest


def _compiled(registry: ComfyTemplateRegistry, template_id: str) -> Any:
    return registry.compile(
        template_id,
        "image",
        _object_info(),
        remote_id="owner/edit",
        revision=REVISION,
        selected_files=["edit.safetensors"],
    )


def test_the_derived_workflow_takes_a_third_picture_and_changes_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, digest = _install(tmp_path, monkeypatch, _template())

    offered = {item.id: item for item in registry.available("image")}
    base, derived = offered[BASE], offered[DERIVED]
    assert (base.sha256, base.derived_from) == (digest, "")
    assert derived.derived_from == digest and derived.sha256 not in {"", digest}
    assert (derived.dependencies, derived.operation, derived.path) == (
        base.dependencies,
        base.operation,
        base.path,
    )

    # One more LoadImage and one more link into the empty input; nothing else moves.
    graph = _derived_graph(derived)
    original = _template()
    added = graph["nodes"][-1]
    assert graph["nodes"][:-1] != original["nodes"]
    host = next(node for node in graph["nodes"] if node["id"] == 5)
    assert host["inputs"][3]["link"] == 51
    host["inputs"][3]["link"] = None
    assert graph["nodes"][:-1] == original["nodes"]
    assert graph["links"] == [*original["links"], [51, 41, 0, 5, 3, "IMAGE"]]
    assert (
        added["id"],
        added["type"],
        added["outputs"][0]["links"],
        added["outputs"][1]["links"],
    ) == (
        41,
        "LoadImage",
        [51],
        None,
    )
    assert (graph["last_node_id"], graph["last_link_id"]) == (41, 51)

    shipped = _compiled(registry, BASE)
    three = _compiled(registry, DERIVED)
    assert "image3" not in shipped.api_graph["5"]["inputs"]
    assert three.api_graph["5"]["inputs"]["image3"] == ["41", 0]
    assert [three.api_graph[node]["inputs"]["image"] for node in ("1", "2", "41")] == [
        "${input_image_0}",
        "${input_image_1}",
        "${input_image_2}",
    ]
    # Every picture is required, in order: the one edited, then two read alongside it.
    records = [
        three.input_schema["properties"][f"input_image_{index}"][SLOT_KEY] for index in range(3)
    ]
    assert [(record["node"], record["role"], record["required"]) for record in records] == [
        ("1", "edit_source", True),
        ("2", "reference", True),
        ("41", "reference", True),
    ]
    assert "input_image_2" not in shipped.input_schema["properties"]


def test_a_derived_workflow_is_installed_only_when_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _ = _install(tmp_path, monkeypatch, _template())

    ranked = registry.matches("owner/edit", "image")

    assert [item.id for item in ranked] == [BASE, DERIVED]
    assert ranked[0].score > ranked[1].score


def test_other_bytes_offer_no_derived_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _ = _install(tmp_path, monkeypatch, _template())
    changed = _template()
    changed["nodes"][-1]["widgets_values"] = ["renamed"]
    (_installed_templates(registry) / f"{BASE}.json").write_text(
        json.dumps(changed), encoding="utf-8"
    )

    offered = {item.id for item in registry.available("image")}
    assert BASE in offered and DERIVED not in offered
    with pytest.raises(ValueError, match="unavailable"):
        registry.get(DERIVED, "image")


def test_compiling_refuses_a_derived_workflow_whose_file_or_identity_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _ = _install(tmp_path, monkeypatch, _template())
    derived = registry.get(DERIVED, "image")

    with pytest.raises(ValueError, match="unavailable"):
        _derived_graph(replace(derived, sha256="0" * 64))
    with pytest.raises(ValueError, match="unavailable"):
        _derived_graph(replace(derived, derived_from="0" * 64))
    # The same graph in other bytes is still another file: only the shipped bytes count.
    derived.path.write_text(json.dumps(_template(), indent=1), encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable"):
        _derived_graph(derived)
    derived.path.write_text(json.dumps({**_template(), "extra": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable"):
        _derived_graph(derived)
    with pytest.raises(ValueError, match="unavailable"):
        _compiled(registry, DERIVED)


@pytest.mark.parametrize(
    "change", ["a third picture already", "the third input already linked", "no counters"]
)
def test_a_template_of_another_shape_is_refused(change: str) -> None:
    graph = _template()
    if change == "a third picture already":
        graph["nodes"].append({**graph["nodes"][1], "id": 9})
    elif change == "the third input already linked":
        graph["nodes"][4]["inputs"][3]["link"] = 3
    else:
        del graph["last_link_id"]

    with pytest.raises(ValueError, match="expected shape"):
        _with_a_third_picture(graph)


def test_the_derived_identity_is_the_derived_graph() -> None:
    graph = _with_a_third_picture(_template())

    assert _graph_sha256(graph) == _graph_sha256(json.loads(json.dumps(graph)))
    assert _graph_sha256(graph) != _graph_sha256(_template())
