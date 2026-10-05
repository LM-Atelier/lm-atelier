"""Image slot roles come from a workflow's structure or an authored declaration, never a guess."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from local_lm.comfy_templates import (
    _AUTHORED_IMAGE_SLOT_ROLES,
    COMFY_TEMPLATE_COMPILER_VERSION,
    _compile_ui_graph,
)
from local_lm.workflow_image_slots_v1 import (
    SLOT_KEY,
    ImageSlot,
    graph_roles,
    image_slots,
    verify_image_slots,
)

Link = tuple[str, int, str, int]


def node(
    kind: str, inputs: tuple[tuple[str, str], ...] = (), outputs: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "type": kind,
        "inputs": [{"name": name, "type": value} for name, value in inputs],
        "outputs": [{"name": value.lower(), "type": value} for value in outputs],
    }


LOAD = node("LoadImage", outputs=("IMAGE", "MASK"))
ENCODE = node("VAEEncode", (("pixels", "IMAGE"), ("vae", "VAE")), ("LATENT",))
SAMPLE = node(
    "KSampler",
    (
        ("model", "MODEL"),
        ("positive", "CONDITIONING"),
        ("negative", "CONDITIONING"),
        ("latent_image", "LATENT"),
    ),
    ("LATENT",),
)
WORDS = node("CLIPTextEncode", (("clip", "CLIP"),), ("CONDITIONING",))
PICTURES_AND_WORDS = node(
    "ImageInstruction",
    (("clip", "CLIP"), ("image1", "IMAGE"), ("image2", "IMAGE")),
    ("CONDITIONING",),
)
EMPTY = node("EmptyLatentImage", outputs=("LATENT",))
SIZE = node("GetImageSize", (("image", "IMAGE"),), ("INT", "INT", "INT"))
SIZED = node("EmptyLatentImage", (("width", "INT"), ("height", "INT")), ("LATENT",))
TO_MASK = node("ImageToMask", (("image", "IMAGE"),), ("MASK",))
NOISE_MASK = node("SetLatentNoiseMask", (("samples", "LATENT"), ("mask", "MASK")), ("LATENT",))
DECODE = node("VAEDecode", (("samples", "LATENT"), ("vae", "VAE")), ("IMAGE",))
# A save, as the template above declares it: with a socket nothing is linked to.
SAVE = node("SaveImage", (("images", "IMAGE"),), ("IMAGE",))
# The node types ComfyUI reports as outputs.
OUTPUTS = frozenset({"SaveImage"})


def saved(sampler: str) -> tuple[dict[str, dict[str, Any]], list[Link]]:
    """A sampler's result decoded and saved, which is what makes it part of the result."""

    return (
        {f"decode-{sampler}": DECODE, f"save-{sampler}": SAVE},
        [(sampler, 0, f"decode-{sampler}", 0), (f"decode-{sampler}", 0, f"save-{sampler}", 0)],
    )


def graph(
    nodes: dict[str, dict[str, Any]], links: list[Link], *samplers: str
) -> tuple[dict[str, dict[str, Any]], list[Link]]:
    for sampler in samplers:
        more_nodes, more_links = saved(sampler)
        nodes, links = {**nodes, **more_nodes}, [*links, *more_links]
    return nodes, links


def conditioning_only() -> tuple[dict[str, dict[str, Any]], list[Link]]:
    """Both pictures condition the words; the canvas starts empty."""

    return graph(
        {"1": LOAD, "5": LOAD, "6": PICTURES_AND_WORDS, "7": EMPTY, "3": SAMPLE},
        [("7", 0, "3", 3), ("1", 0, "6", 1), ("5", 0, "6", 2), ("6", 0, "3", 1)],
        "3",
    )


def edit_with_reference() -> tuple[dict[str, dict[str, Any]], list[Link]]:
    """The first picture is encoded as the canvas; both pictures also condition the words."""

    nodes = {"1": LOAD, "2": ENCODE, "3": SAMPLE, "5": LOAD, "6": PICTURES_AND_WORDS, "4": WORDS}
    links: list[Link] = [
        ("1", 0, "2", 0),
        ("2", 0, "3", 3),
        ("1", 0, "6", 1),
        ("5", 0, "6", 2),
        ("6", 0, "3", 1),
        ("4", 0, "3", 2),
    ]
    return graph(nodes, links, "3")


def test_a_picture_encoded_into_the_sampler_latent_is_the_one_edited() -> None:
    nodes, links = graph(
        {"1": LOAD, "2": ENCODE, "3": SAMPLE, "4": WORDS},
        [("1", 0, "2", 0), ("2", 0, "3", 3), ("4", 0, "3", 1)],
        "3",
    )

    assert graph_roles(nodes, links, ["1"], OUTPUTS) == ["edit_source"]


def test_a_picture_that_only_conditions_beside_one_canvas_is_a_reference() -> None:
    nodes, links = edit_with_reference()

    assert graph_roles(nodes, links, ["1", "5"], OUTPUTS) == ["edit_source", "reference"]
    # Slot order follows the bindings, not the roles.
    assert graph_roles(nodes, links, ["5", "1"], OUTPUTS) == ["reference", "edit_source"]


def test_pictures_that_only_condition_say_nothing_about_which_is_edited() -> None:
    nodes, links = conditioning_only()

    # However the first picture is wired, it is not shown to be the one edited.
    assert graph_roles(nodes, links, ["1", "5"], OUTPUTS) == ["unknown", "unknown"]


def test_only_the_picture_output_counts_never_the_mask() -> None:
    nodes, links = graph(
        {"1": LOAD, "2": ENCODE, "3": SAMPLE}, [("1", 1, "2", 0), ("2", 0, "3", 3)], "3"
    )

    assert graph_roles(nodes, links, ["1"], OUTPUTS) == ["unknown"]


def test_a_size_or_mask_taken_from_a_picture_does_not_make_it_the_canvas() -> None:
    # The picture only sets the size of an empty canvas.
    nodes, links = graph(
        {"1": LOAD, "2": SIZE, "7": SIZED, "3": SAMPLE, "4": WORDS},
        [("1", 0, "2", 0), ("2", 0, "7", 0), ("2", 1, "7", 1), ("7", 0, "3", 3), ("4", 0, "3", 1)],
        "3",
    )
    assert graph_roles(nodes, links, ["1"], OUTPUTS) == ["unknown"]

    # The picture becomes only the mask over another canvas.
    nodes, links = graph(
        {"1": LOAD, "2": TO_MASK, "7": EMPTY, "8": NOISE_MASK, "3": SAMPLE, "4": WORDS},
        [("1", 0, "2", 0), ("2", 0, "8", 1), ("7", 0, "8", 0), ("8", 0, "3", 3), ("4", 0, "3", 1)],
        "3",
    )
    assert graph_roles(nodes, links, ["1"], OUTPUTS) == ["unknown"]


def test_a_picture_counts_only_as_what_the_sampler_receives() -> None:
    # Wired straight into the canvas input as a picture, never encoded into a latent.
    nodes, links = graph(
        {"1": LOAD, "3": SAMPLE, "4": WORDS}, [("1", 0, "3", 3), ("4", 0, "3", 1)], "3"
    )
    assert graph_roles(nodes, links, ["1"], OUTPUTS) == ["unknown"]

    # Taken by a sampler as a picture, beside an encoded canvas: not shown to condition it.
    guided = node(
        "GuidedSampler",
        (("positive", "CONDITIONING"), ("latent_image", "LATENT"), ("guide", "IMAGE")),
        ("LATENT",),
    )
    nodes, links = graph(
        {"1": LOAD, "2": ENCODE, "3": guided, "4": WORDS, "5": LOAD},
        [("1", 0, "2", 0), ("2", 0, "3", 1), ("4", 0, "3", 0), ("5", 0, "3", 2)],
        "3",
    )
    assert graph_roles(nodes, links, ["1", "5"], OUTPUTS) == ["edit_source", "unknown"]


def test_two_canvases_leave_every_role_unknown() -> None:
    second_encode, second_sample = dict(ENCODE), dict(SAMPLE)
    nodes, links = graph(
        {"1": LOAD, "2": ENCODE, "3": SAMPLE, "5": LOAD, "8": second_encode, "9": second_sample},
        [("1", 0, "2", 0), ("2", 0, "3", 3), ("5", 0, "8", 0), ("8", 0, "9", 3)],
        "3",
        "9",
    )

    assert graph_roles(nodes, links, ["1", "5"], OUTPUTS) == ["unknown", "unknown"]


def test_a_picture_that_conditions_a_first_pass_is_not_the_canvas_of_the_second() -> None:
    nodes, links = edit_with_reference()
    # The first pass's result is the second pass's canvas, and the second is saved too.
    nodes, links = graph({**nodes, "9": dict(SAMPLE)}, [*links, ("3", 0, "9", 3)], "9")

    assert graph_roles(nodes, links, ["1", "5"], OUTPUTS) == ["edit_source", "reference"]


def test_conditioning_that_never_reaches_a_running_sampler_is_not_a_reference() -> None:
    nodes, links = graph(
        {"1": LOAD, "2": ENCODE, "3": SAMPLE, "4": WORDS, "5": LOAD, "6": PICTURES_AND_WORDS},
        # The second picture's encoder output goes nowhere.
        [("1", 0, "2", 0), ("2", 0, "3", 3), ("4", 0, "3", 1), ("5", 0, "6", 1)],
        "3",
    )

    assert graph_roles(nodes, links, ["1", "5"], OUTPUTS) == ["edit_source", "unknown"]


def test_a_sampler_whose_result_is_never_saved_counts_for_nothing() -> None:
    unused = dict(SAMPLE)
    nodes, links = graph(
        {"1": LOAD, "2": ENCODE, "8": unused, "7": EMPTY, "3": SAMPLE, "4": WORDS},
        # The picture is the canvas only of a sampler whose result goes nowhere.
        [("1", 0, "2", 0), ("2", 0, "8", 3), ("7", 0, "3", 3), ("4", 0, "3", 1)],
        "3",
    )

    assert graph_roles(nodes, links, ["1"], OUTPUTS) == ["unknown"]


def test_an_authored_declaration_names_every_slot_or_none() -> None:
    nodes, links = conditioning_only()
    bindings = [("1", "input_image_0"), ("5", "input_image_1")]

    declared = image_slots(nodes, links, bindings, ("edit_source", "reference"), OUTPUTS)
    assert declared == [
        ImageSlot("input_image_0", "1", "edit_source", "declared"),
        ImageSlot("input_image_1", "5", "reference", "declared"),
    ]
    # A declaration for some other graph applies to no slot.
    assert image_slots(nodes, links, bindings, ("edit_source",), OUTPUTS) == [
        ImageSlot("input_image_0", "1", "unknown", "none"),
        ImageSlot("input_image_1", "5", "unknown", "none"),
    ]
    assert image_slots(nodes, links, bindings, ("edit_source", "style"), OUTPUTS) == [
        ImageSlot("input_image_0", "1", "unknown", "none"),
        ImageSlot("input_image_1", "5", "unknown", "none"),
    ]
    # Without one, structure decides, and here it shows nothing.
    assert [slot.role for slot in image_slots(nodes, links, bindings, (), OUTPUTS)] == [
        "unknown",
        "unknown",
    ]


def test_graph_evidence_is_recorded_as_graph_and_unknown_as_none() -> None:
    nodes, links = edit_with_reference()

    assert image_slots(
        nodes, links, [("1", "input_image_0"), ("5", "input_image_1")], (), OUTPUTS
    ) == [
        ImageSlot("input_image_0", "1", "edit_source", "graph"),
        ImageSlot("input_image_1", "5", "reference", "graph"),
    ]


def compiled(*slots: ImageSlot) -> tuple[dict[str, Any], dict[str, Any]]:
    api_graph: dict[str, Any] = {
        slot.node: {"class_type": "LoadImage", "inputs": {"image": "${" + slot.name + "}"}}
        for slot in slots
    }
    api_graph["20"] = {"class_type": "KSampler", "inputs": {"seed": "${seed}"}}
    schema = {
        "properties": {
            **{slot.name: {"type": "string", SLOT_KEY: slot.as_schema()} for slot in slots},
            "prompt": {"type": "string"},
        }
    }
    return schema, api_graph


EDIT = ImageSlot("input_image_0", "10", "edit_source", "graph")
REFERENCE = ImageSlot("input_image_1", "11", "reference", "graph")


def test_records_bound_to_their_own_nodes_are_trusted() -> None:
    schema, api_graph = compiled(EDIT, REFERENCE)

    assert verify_image_slots(schema, api_graph) == {
        "input_image_0": EDIT,
        "input_image_1": REFERENCE,
    }
    single = ImageSlot("input_image", "10", "edit_source", "graph")
    assert verify_image_slots(*compiled(single)) == {"input_image": single}
    last = ImageSlot("input_image_63", "11", "reference", "graph")
    assert verify_image_slots(*compiled(EDIT, last)) == {
        "input_image_0": EDIT,
        "input_image_63": last,
    }


def _changed(change: str) -> tuple[dict[str, Any], dict[str, Any]]:
    schema, api_graph = compiled(EDIT, REFERENCE)
    records = schema["properties"]
    if change == "slots swapped in the graph":
        api_graph["10"]["inputs"]["image"], api_graph["11"]["inputs"]["image"] = (
            "${input_image_1}",
            "${input_image_0}",
        )
    elif change == "a record missing":
        del records["input_image_1"][SLOT_KEY]
    elif change == "a node missing":
        del api_graph["11"]
    elif change == "another class":
        api_graph["11"]["class_type"] = "LoadImageMask"
    elif change == "another version":
        records["input_image_1"][SLOT_KEY]["version"] = 2
    elif change == "a known role with no basis":
        records["input_image_1"][SLOT_KEY]["basis"] = "none"
    elif change == "an optional slot":
        records["input_image_1"][SLOT_KEY]["required"] = False
    elif change == "an extra field":
        records["input_image_1"][SLOT_KEY]["fallback"] = "input_image_0"
    elif change == "an unknown role":
        records["input_image_1"][SLOT_KEY]["role"] = "style"
    elif change == "an unrecorded slot":
        api_graph["12"] = {"class_type": "LoadImage", "inputs": {"image": "${input_image_2}"}}
    elif change == "a role that is not a string":
        records["input_image_1"][SLOT_KEY]["role"] = ["reference"]
    elif change == "a basis that is not a string":
        records["input_image_1"][SLOT_KEY]["basis"] = {"graph": True}
    elif change == "a version that is not an integer":
        records["input_image_1"][SLOT_KEY]["version"] = 1.0
    elif change == "a record that is not an object":
        records["input_image_1"][SLOT_KEY] = "reference"
    elif change == "two nodes on one slot":
        api_graph["12"] = {"class_type": "LoadImage", "inputs": {"image": "${input_image_1}"}}
    elif change in ("a slot number with a leading zero", "a slot number past the last"):
        # Bound and recorded alike, so only the name itself is wrong.
        name = "input_image_02" if "zero" in change else "input_image_64"
        api_graph["12"] = {"class_type": "LoadImage", "inputs": {"image": "${" + name + "}"}}
        records[name] = {
            "type": "string",
            SLOT_KEY: ImageSlot(name, "12", "reference", "graph").as_schema(),
        }
    elif change == "every picture on one node":
        api_graph["12"] = {"class_type": "LoadImage", "inputs": {"image": "${input_images}"}}
    return schema, api_graph


@pytest.mark.parametrize(
    "change",
    [
        "slots swapped in the graph",
        "a record missing",
        "a node missing",
        "another class",
        "another version",
        "a known role with no basis",
        "an optional slot",
        "an extra field",
        "an unknown role",
        "an unrecorded slot",
        "a role that is not a string",
        "a basis that is not a string",
        "a version that is not an integer",
        "a record that is not an object",
        "two nodes on one slot",
        "a slot number with a leading zero",
        "a slot number past the last",
        "every picture on one node",
    ],
)
def test_any_record_that_cannot_be_trusted_leaves_every_slot_unknown(change: str) -> None:
    assert verify_image_slots(*_changed(change)) == {}


def test_a_workflow_without_records_has_no_known_slot() -> None:
    schema, api_graph = compiled(EDIT, REFERENCE)
    for name in ("input_image_0", "input_image_1"):
        del schema["properties"][name][SLOT_KEY]

    assert verify_image_slots(schema, api_graph) == {}
    assert verify_image_slots({"properties": {}}, {}) == {}


def test_without_knowing_which_nodes_are_outputs_no_role_is_shown() -> None:
    nodes, links = edit_with_reference()

    assert graph_roles(nodes, links, ["1", "5"], frozenset()) == ["unknown", "unknown"]


def _ui_graph() -> dict[str, Any]:
    """Two pictures: the first encoded as the canvas, both conditioning the instruction."""

    def ui(
        node_id: int,
        kind: str,
        inputs: list[tuple[str, str, int]],
        outputs: list[tuple[str, list[int]]],
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "id": node_id,
            "type": kind,
            "inputs": [{"name": name, "type": value, "link": link} for name, value, link in inputs],
            "outputs": [{"name": value, "type": value, "links": links} for value, links in outputs],
            **extra,
        }

    return {
        "nodes": [
            ui(
                1,
                "LoadImage",
                [],
                [("IMAGE", [1, 2]), ("MASK", [])],
                widgets_values=["first.png", "image"],
            ),
            ui(
                2,
                "LoadImage",
                [],
                [("IMAGE", [3]), ("MASK", [])],
                widgets_values=["second.png", "image"],
            ),
            ui(3, "VAEEncode", [("pixels", "IMAGE", 1)], [("LATENT", [4])]),
            ui(
                4,
                "ImageInstruction",
                [("image1", "IMAGE", 2), ("image2", "IMAGE", 3)],
                [("CONDITIONING", [5])],
            ),
            ui(
                5,
                "KSampler",
                [("positive", "CONDITIONING", 5), ("latent_image", "LATENT", 4)],
                [("LATENT", [6])],
            ),
            ui(6, "VAEDecode", [("samples", "LATENT", 6)], [("IMAGE", [7])]),
            ui(7, "SaveImage", [("images", "IMAGE", 7)], [("IMAGE", [])]),
        ],
        "links": [
            [1, 1, 0, 3, 0, "IMAGE"],
            [2, 1, 0, 4, 0, "IMAGE"],
            [3, 2, 0, 4, 1, "IMAGE"],
            [4, 3, 0, 5, 1, "LATENT"],
            [5, 4, 0, 5, 0, "CONDITIONING"],
            [6, 5, 0, 6, 0, "LATENT"],
            [7, 6, 0, 7, 0, "IMAGE"],
        ],
    }


def _object_info(*, outputs: bool = True) -> dict[str, Any]:
    def spec(*names: tuple[str, str]) -> dict[str, Any]:
        return {
            "input": {"required": {name: [value] for name, value in names}},
            "input_order": {"required": [name for name, _ in names]},
        }

    info = {
        "LoadImage": {
            "input": {"required": {"image": [["first.png", "second.png"], {"image_upload": True}]}},
            "input_order": {"required": ["image"]},
        },
        "VAEEncode": spec(("pixels", "IMAGE")),
        "ImageInstruction": spec(("image1", "IMAGE"), ("image2", "IMAGE")),
        "KSampler": spec(("positive", "CONDITIONING"), ("latent_image", "LATENT")),
        "VAEDecode": spec(("samples", "LATENT")),
        "SaveImage": spec(("images", "IMAGE")),
    }
    if outputs:
        info["SaveImage"]["output_node"] = True
    return info


def _compiled_slots(
    object_info: dict[str, Any], declared: tuple[str, ...] = ()
) -> dict[str, tuple[str, str, str]]:
    api_graph, schema = _compile_ui_graph(
        _ui_graph(), object_info, operation="image_to_image", image_slot_roles=declared
    )
    assert api_graph["1"]["inputs"]["image"] == "${input_image_0}"
    assert api_graph["2"]["inputs"]["image"] == "${input_image_1}"
    return {
        name: (slot.node, slot.role, slot.basis)
        for name, slot in verify_image_slots(schema, api_graph).items()
    }


def test_a_compiled_workflow_records_each_picture_on_its_own_slot() -> None:
    assert _compiled_slots(_object_info()) == {
        "input_image_0": ("1", "edit_source", "graph"),
        "input_image_1": ("2", "reference", "graph"),
    }
    # An authored declaration is recorded as such, even where it reads the graph differently.
    assert _compiled_slots(_object_info(), ("reference", "edit_source")) == {
        "input_image_0": ("1", "reference", "declared"),
        "input_image_1": ("2", "edit_source", "declared"),
    }
    # Node information that names no output leaves the graph showing nothing.
    assert _compiled_slots(_object_info(outputs=False)) == {
        "input_image_0": ("1", "unknown", "none"),
        "input_image_1": ("2", "unknown", "none"),
    }


def test_a_changed_declaration_ships_with_a_new_compiler_version() -> None:
    """An installed workflow is reused while its template file and compiler version match.

    A declaration changes what the compiler records for that same file, so the
    declarations are part of what the version names: change one, and the
    version moves with it, or an install keeps the slots it was compiled with.
    """

    declarations = json.dumps(
        sorted([*key, list(roles)] for key, roles in _AUTHORED_IMAGE_SLOT_ROLES.items())
    )
    assert (
        COMFY_TEMPLATE_COMPILER_VERSION,
        hashlib.sha256(declarations.encode()).hexdigest()[:16],
    ) == (
        27,
        "1105c52739781ed1",
    )
