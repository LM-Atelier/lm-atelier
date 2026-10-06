from __future__ import annotations

import copy
from typing import Any

import pytest

from local_lm.comfy_lora_runtime_names import (
    bind_lora_runtime_names,
    has_nested_lora_names,
    lora_runtime_names_need_binding,
)


def _windows_binding(graph: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    return bind_lora_runtime_names(graph, info, runtime_platform="win32")


def graph(name: object, class_type: str = "LoraLoader") -> dict[str, Any]:
    return {
        "lora": {
            "class_type": class_type,
            "inputs": {
                "lora_name": name,
                "model": ["source", 0],
                "clip": ["source", 1],
                "strength_model": 0.0,
                "strength_clip": 0.75,
            },
        },
        "text": {"class_type": "CLIPTextEncode", "inputs": {"text": "A blue cube"}},
    }


def info(names: list[str], class_type: str = "LoraLoader") -> dict[str, Any]:
    return {class_type: {"input": {"required": {"lora_name": [names, {}]}}}}


@pytest.mark.parametrize("class_type", ["LoraLoader", "LoraLoaderModelOnly"])
@pytest.mark.parametrize(
    ("selected", "runtime"),
    [
        ("nested/adapter.safetensors", "nested\\adapter.safetensors"),
        ("nested\\adapter.safetensors", "nested/adapter.safetensors"),
    ],
)
def test_only_outgoing_filename_spelling_changes(
    class_type: str, selected: str, runtime: str
) -> None:
    stored = graph(selected, class_type)
    original = copy.deepcopy(stored)
    bound = _windows_binding(stored, info([runtime], class_type))
    assert bound["lora"]["inputs"]["lora_name"] == runtime
    expected = copy.deepcopy(original)
    expected["lora"]["inputs"]["lora_name"] = runtime
    assert bound == expected
    assert stored == original
    assert bound["lora"]["inputs"] is not stored["lora"]["inputs"]
    assert bound["text"] is stored["text"]


@pytest.mark.parametrize(
    "name",
    [
        "adapter.safetensors",
        "../adapter",
        "/nested/adapter",
        "C:\\nested\\adapter",
        "nested//adapter",
        ["source", 0],
        None,
    ],
)
def test_flat_invalid_and_linked_values_do_not_request_filename_metadata(name: object) -> None:
    stored = graph(name)
    assert not has_nested_lora_names(stored)
    assert _windows_binding(stored, info(["nested\\adapter.safetensors"])) == stored


def test_exact_choice_wins_without_rewriting_other_equivalent_choices() -> None:
    stored = graph("nested/adapter.safetensors")
    assert (
        _windows_binding(
            stored, info(["nested/adapter.safetensors", "nested\\adapter.safetensors"])
        )
        == stored
    )


@pytest.mark.parametrize(
    "choice", ["other\\adapter.safetensors", "Nested\\adapter.safetensors", "nested\\adapter.bin"]
)
def test_no_basename_case_or_extension_fallback(choice: str) -> None:
    stored = graph("nested/adapter.safetensors")
    assert _windows_binding(stored, info([choice])) == stored


def test_multiple_separator_equivalents_refuse_without_mutating_the_graph() -> None:
    stored = graph("nested/group/adapter.safetensors")
    before = copy.deepcopy(stored)
    with pytest.raises(ValueError, match="ambiguous LoRA filenames"):
        _windows_binding(
            stored, info(["nested\\group/adapter.safetensors", "nested/group\\adapter.safetensors"])
        )
    assert stored == before


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"LoraLoader": None},
        {"LoraLoader": {"input": {"required": {"lora_name": "STRING"}}}},
        {"LoraLoader": {"input": {"required": {"lora_name": [[1], {}]}}}},
    ],
)
def test_missing_or_malformed_choices_do_not_invent_a_filename(metadata: dict[str, Any]) -> None:
    stored = graph("nested/adapter.safetensors")
    assert _windows_binding(stored, metadata) == stored


def test_custom_nodes_and_unrelated_strings_are_not_rewritten() -> None:
    stored = graph("nested/adapter.safetensors", "CustomLoader")
    stored["text"]["inputs"]["text"] = "nested/adapter.safetensors"
    assert not has_nested_lora_names(stored)
    assert _windows_binding(stored, info(["nested\\adapter.safetensors"], "CustomLoader")) == stored


def test_malformed_nodes_are_ignored() -> None:
    stored: dict[str, Any] = {
        "a": None,
        "b": {"class_type": []},
        "c": {"class_type": "LoraLoader", "inputs": []},
    }
    assert not has_nested_lora_names(stored)
    assert _windows_binding(stored, {}) == stored


@pytest.mark.parametrize("platform", [None, "posix", "linux", "darwin", "unknown"])
def test_posix_or_unknown_platform_preserves_literal_backslash_choices(
    platform: str | None,
) -> None:
    stored = graph("nested/adapter.safetensors")
    choices = info(["nested\\adapter.safetensors"])
    assert lora_runtime_names_need_binding(stored, choices)
    assert bind_lora_runtime_names(stored, choices, runtime_platform=platform) == stored


@pytest.mark.parametrize("platform", ["win32", "nt", "Windows"])
def test_declared_windows_platform_allows_equivalent_spelling(platform: str) -> None:
    stored = graph("nested/adapter.safetensors")
    bound = bind_lora_runtime_names(
        stored, info(["nested\\adapter.safetensors"]), runtime_platform=platform
    )
    assert bound["lora"]["inputs"]["lora_name"] == "nested\\adapter.safetensors"


def test_exact_choices_need_no_platform_lookup() -> None:
    stored = graph("nested/adapter.safetensors")
    assert not lora_runtime_names_need_binding(stored, info(["nested/adapter.safetensors"]))
