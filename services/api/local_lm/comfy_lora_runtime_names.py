"""Use runtime spelling for nested filenames in the core LoRA loaders."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

_LOADERS = frozenset({"LoraLoader", "LoraLoaderModelOnly"})


def _nested_inputs(graph: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any], str]]:
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or class_type not in _LOADERS:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        name = inputs.get("lora_name")
        if not isinstance(name, str) or not any(separator in name for separator in ("/", "\\")):
            continue
        parts = name.replace("\\", "/").split("/")
        if any(part in {"", ".", ".."} or ":" in part for part in parts):
            continue
        yield node_id, node, name


def has_nested_lora_names(graph: dict[str, Any]) -> bool:
    return next(_nested_inputs(graph), None) is not None


def _choices(node: dict[str, Any], object_info: dict[str, Any]) -> list[str] | None:
    declaration = object_info.get(node["class_type"])
    if not isinstance(declaration, dict):
        return None
    inputs = declaration.get("input")
    required = inputs.get("required") if isinstance(inputs, dict) else None
    field = required.get("lora_name") if isinstance(required, dict) else None
    choices = field[0] if isinstance(field, list) and field else None
    if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
        return None
    return choices


def lora_runtime_names_need_binding(graph: dict[str, Any], object_info: dict[str, Any]) -> bool:
    for _node_id, node, name in _nested_inputs(graph):
        choices = _choices(node, object_info)
        if (
            choices is not None
            and name not in choices
            and any(item.replace("\\", "/") == name.replace("\\", "/") for item in choices)
        ):
            return True
    return False


def bind_lora_runtime_names(
    graph: dict[str, Any], object_info: dict[str, Any], *, runtime_platform: str | None = None
) -> dict[str, Any]:
    """Change only separators when one declared choice names the same file.

    Catalogue and workflow identities keep their portable spelling. The outgoing
    graph needs the runtime's exact dropdown value because validation compares
    strings before the loader opens a file. No path resolution or case folding
    participates in this match. Substitution requires declared Windows semantics;
    on POSIX a literal backslash can name a different file.
    """
    result = dict(graph)
    if runtime_platform not in {"win32", "nt", "Windows"}:
        return result
    for node_id, node, name in _nested_inputs(graph):
        choices = _choices(node, object_info)
        if choices is None:
            continue
        if name in choices:
            continue
        matches = {item for item in choices if item.replace("\\", "/") == name.replace("\\", "/")}
        if len(matches) > 1:
            raise ValueError("ComfyUI exposes ambiguous LoRA filenames")
        if matches:
            result[node_id] = {**node, "inputs": {**node["inputs"], "lora_name": matches.pop()}}
    return result
