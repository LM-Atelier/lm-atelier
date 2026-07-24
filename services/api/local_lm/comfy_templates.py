from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .config import Settings

_GENERIC_TOKENS = {
    "comfy",
    "comfyui",
    "diffusion",
    "huggingface",
    "image",
    "model",
    "models",
    "official",
    "org",
    "template",
    "text",
    "to",
    "video",
    "workflow",
}
_SPECIALTY_TOKENS = {
    "control",
    "controlnet",
    "edit",
    "fun",
    "i2v",
    "img2img",
    "inpaint",
    "lora",
    "redux",
    "union",
    "upscale",
}
_RUNTIME_PARAMETERS = {
    "batch_size": "batch_size",
    "cfg": "cfg",
    "denoise": "denoise",
    "fps": "fps",
    "frames": "frames",
    "height": "height",
    "sampler_name": "sampler",
    "scheduler": "scheduler",
    "seed": "seed",
    "steps": "steps",
    "width": "width",
}
_PRIMITIVE_WIDGET_TYPES = {"BOOLEAN", "FLOAT", "INT", "STRING"}
_CONTROL_AFTER_GENERATE = {"decrement", "fixed", "increment", "randomize"}


@dataclass(frozen=True)
class ComfyModelDependency:
    remote_id: str
    revision: str
    path: str
    directory: str
    name: str
    url: str


@dataclass(frozen=True)
class ComfyTemplate:
    id: str
    path: Path
    role: str
    operation: str
    score: int
    sha256: str
    dependencies: tuple[ComfyModelDependency, ...]

    @property
    def remote_id(self) -> str:
        return self.dependencies[0].remote_id

    @property
    def revision(self) -> str:
        return self.dependencies[0].revision

    @property
    def selected_files(self) -> list[str]:
        return [item.path for item in self.dependencies]

    @property
    def comfy_paths(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for dependency in self.dependencies:
            parent = str(Path(dependency.path).parent).replace("\\", "/")
            result[dependency.directory] = "." if parent == "." else parent
        return result


@dataclass(frozen=True)
class CompiledComfyTemplate:
    template: ComfyTemplate
    ui_graph: dict[str, Any]
    api_graph: dict[str, Any]
    input_schema: dict[str, Any]


class ComfyTemplateRegistry:
    """Discovers and compiles the workflow catalog shipped with ComfyUI."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def matches(self, remote_id: str, role: str) -> list[ComfyTemplate]:
        source_tokens = _tokens(remote_id)
        matches: list[ComfyTemplate] = []
        for path in self._template_files():
            template_role = _role_for_template(path.stem)
            if template_role != role:
                continue
            raw = _read_json(path)
            dependencies = tuple(_model_dependencies(raw))
            if not dependencies:
                continue
            repositories = {item.remote_id.lower() for item in dependencies}
            revisions = {item.revision for item in dependencies}
            if len(repositories) != 1 or len(revisions) != 1:
                continue
            if not _uses_only_core_nodes(raw):
                continue
            template_tokens = _tokens(path.stem)
            repository_tokens = _tokens(dependencies[0].remote_id)
            overlap = (template_tokens | repository_tokens) & source_tokens
            meaningful = overlap - _GENERIC_TOKENS
            if not meaningful:
                continue
            source_specialties = source_tokens & _SPECIALTY_TOKENS
            template_specialties = template_tokens & _SPECIALTY_TOKENS
            specialty_penalty = len(template_specialties - source_specialties) * 20
            score = len(meaningful) * 20 + len(overlap) * 2 - specialty_penalty
            matches.append(
                ComfyTemplate(
                    id=path.stem,
                    path=path,
                    role=role,
                    operation=_operation_for_template(path.stem, role),
                    score=score,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    dependencies=dependencies,
                )
            )
        return sorted(matches, key=lambda item: (-item.score, item.id))

    def get(self, template_id: str, role: str) -> ComfyTemplate:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", template_id):
            raise ValueError("invalid ComfyUI template identifier")
        for path in self._template_files():
            if path.stem != template_id:
                continue
            raw = _read_json(path)
            dependencies = tuple(_model_dependencies(raw))
            if (
                _role_for_template(path.stem) != role
                or not dependencies
                or len({item.remote_id.lower() for item in dependencies}) != 1
                or len({item.revision for item in dependencies}) != 1
                or not _uses_only_core_nodes(raw)
            ):
                break
            return ComfyTemplate(
                id=path.stem,
                path=path,
                role=role,
                operation=_operation_for_template(path.stem, role),
                score=0,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                dependencies=dependencies,
            )
        raise ValueError(f"ComfyUI template is unavailable: {template_id}")

    def compile(
        self,
        template_id: str,
        role: str,
        object_info: dict[str, Any],
    ) -> CompiledComfyTemplate:
        template = self.get(template_id, role)
        ui_graph = _read_json(template.path)
        api_graph, input_schema = _compile_ui_graph(ui_graph, object_info)
        output_nodes = [
            node_id
            for node_id, node in api_graph.items()
            if bool((object_info.get(str(node.get("class_type"))) or {}).get("output_node"))
        ]
        if not output_nodes:
            raise ValueError("ComfyUI template has no runnable output node")
        return CompiledComfyTemplate(
            template=template,
            ui_graph=ui_graph,
            api_graph=api_graph,
            input_schema=input_schema,
        )

    def validate_download(
        self,
        template_id: str,
        role: str,
        remote_id: str,
        selected_files: list[str],
        comfy_paths: dict[str, str],
    ) -> ComfyTemplate:
        template = self.get(template_id, role)
        if template.remote_id.lower() != remote_id.lower():
            raise ValueError("download repository does not match the ComfyUI template")
        if set(template.selected_files) != set(selected_files):
            raise ValueError("download files do not match the ComfyUI template")
        if template.comfy_paths != comfy_paths:
            raise ValueError("download model paths do not match the ComfyUI template")
        return template

    def _template_files(self) -> list[Path]:
        directory = self.settings.comfy_directory
        if not directory:
            return []
        root = directory.expanduser().resolve()
        candidates = [
            root
            / ".venv"
            / "Lib"
            / "site-packages"
            / "comfyui_workflow_templates_json"
            / "templates",
            root
            / "venv"
            / "Lib"
            / "site-packages"
            / "comfyui_workflow_templates_json"
            / "templates",
            root
            / ".venv"
            / "lib"
            / "python3"
            / "site-packages"
            / "comfyui_workflow_templates_json"
            / "templates",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return sorted(candidate.glob("*.json"))
        unix_matches = sorted(
            root.glob(
                ".venv/lib/python*/site-packages/comfyui_workflow_templates_json/templates/*.json"
            )
        )
        return unix_matches


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]+|\d+", value.lower())
        if len(token) > 1 or token == "z"
    }


def _role_for_template(template_id: str) -> str | None:
    if template_id.startswith("image_"):
        return "image"
    if template_id.startswith("video_"):
        return "video"
    return None


def _operation_for_template(template_id: str, role: str) -> str:
    tokens = _tokens(template_id)
    if role == "video":
        return "image_to_video" if {"i2v", "image2video"} & tokens else "text_to_video"
    return "image_to_image" if {"img2img", "image2image", "edit"} & tokens else "text_to_image"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"ComfyUI template must contain an object: {path.name}")
    return value


def _all_nodes(value: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [item for item in value.get("nodes", []) if isinstance(item, dict)]
    definitions = value.get("definitions")
    if isinstance(definitions, dict):
        for subgraph in definitions.get("subgraphs", []):
            if isinstance(subgraph, dict):
                nodes.extend(_all_nodes(subgraph))
    return nodes


def _uses_only_core_nodes(value: dict[str, Any]) -> bool:
    for node in _all_nodes(value):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        package = properties.get("cnr_id")
        if package not in {None, "", "comfy-core"}:
            return False
    return True


def _model_dependencies(value: dict[str, Any]) -> list[ComfyModelDependency]:
    dependencies: dict[tuple[str, str], ComfyModelDependency] = {}
    for node in _all_nodes(value):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        models = properties.get("models")
        if isinstance(models, dict):
            model_items = [models]
        elif isinstance(models, list):
            model_items = [item for item in models if isinstance(item, dict)]
        else:
            continue
        for model in model_items:
            url = str(model.get("url") or "")
            parsed = _parse_huggingface_url(url)
            if not parsed:
                continue
            remote_id, revision, path = parsed
            name = str(model.get("name") or Path(path).name)
            directory = str(model.get("directory") or Path(path).parent.name)
            dependency = ComfyModelDependency(
                remote_id=remote_id,
                revision=revision,
                path=path,
                directory=directory,
                name=name,
                url=url,
            )
            dependencies[(remote_id.lower(), path)] = dependency
    return sorted(dependencies.values(), key=lambda item: (item.directory, item.path))


def _parse_huggingface_url(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "huggingface.co":
        return None
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 6 or parts[2] != "resolve":
        return None
    owner, name, _, revision, *file_parts = parts
    if not owner or not name or not revision or not file_parts:
        return None
    if any(part in {"", ".", ".."} for part in file_parts):
        return None
    return f"{owner}/{name}", revision, "/".join(file_parts)


def _compile_ui_graph(
    ui_graph: dict[str, Any],
    object_info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    definitions = {
        str(item.get("id")): item
        for item in ((ui_graph.get("definitions") or {}).get("subgraphs") or [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    flat_nodes: dict[str, dict[str, Any]] = {}
    links: list[tuple[str, int, str, int]] = []
    group_inputs: dict[str, dict[int, list[tuple[str, int, str]]]] = {}
    group_outputs: dict[str, dict[int, tuple[str, int]]] = {}
    parameter_overrides: dict[tuple[str, str], str] = {}

    for node in ui_graph.get("nodes", []):
        if not isinstance(node, dict) or node.get("id") is None:
            continue
        node_id = str(node["id"])
        subgraph = definitions.get(str(node.get("type")))
        if not subgraph:
            flat_nodes[node_id] = node
            continue
        subgraph_input_targets: dict[int, list[tuple[str, int, str]]] = {}
        outputs: dict[int, tuple[str, int]] = {}
        subgraph_inputs = [item for item in subgraph.get("inputs", []) if isinstance(item, dict)]
        for link in subgraph.get("links", []):
            normalized = _normalize_link(link)
            if not normalized:
                continue
            origin, origin_slot, target, target_slot = normalized
            origin_key = f"{node_id}:{origin}"
            target_key = f"{node_id}:{target}"
            if origin == "-10":
                parameter = (
                    str(subgraph_inputs[origin_slot].get("name") or "")
                    if origin_slot < len(subgraph_inputs)
                    else ""
                )
                subgraph_input_targets.setdefault(origin_slot, []).append(
                    (target_key, target_slot, parameter)
                )
                continue
            if target == "-20":
                outputs[target_slot] = (origin_key, origin_slot)
                continue
            links.append((origin_key, origin_slot, target_key, target_slot))
        for inner in subgraph.get("nodes", []):
            if not isinstance(inner, dict) or inner.get("id") in {-10, -20, "-10", "-20"}:
                continue
            flat_nodes[f"{node_id}:{inner['id']}"] = inner
        group_inputs[node_id] = subgraph_input_targets
        group_outputs[node_id] = outputs
        for targets in subgraph_input_targets.values():
            for target_key, target_slot, parameter in targets:
                target_node = flat_nodes.get(target_key)
                if not target_node:
                    continue
                target_inputs = target_node.get("inputs") or []
                if target_slot >= len(target_inputs):
                    continue
                input_name = str(target_inputs[target_slot].get("name") or "")
                runtime_name = (
                    "prompt"
                    if parameter == "text"
                    else _runtime_parameter(parameter or input_name, target_node)
                )
                if runtime_name:
                    parameter_overrides[(target_key, input_name)] = runtime_name

    for raw_link in ui_graph.get("links", []):
        normalized = _normalize_link(raw_link)
        if not normalized:
            continue
        origin, origin_slot, target, target_slot = normalized
        resolved_origin = group_outputs.get(origin, {}).get(origin_slot, (origin, origin_slot))
        if target in group_inputs:
            for inner_target, inner_slot, _ in group_inputs[target].get(target_slot, []):
                links.append((*resolved_origin, inner_target, inner_slot))
        else:
            links.append((*resolved_origin, target, target_slot))

    linked_inputs: dict[tuple[str, str], list[Any]] = {}
    for origin, origin_slot, target, target_slot in links:
        target_node = flat_nodes.get(target)
        if not target_node:
            continue
        target_inputs = target_node.get("inputs") or []
        if target_slot >= len(target_inputs):
            continue
        name = str(target_inputs[target_slot].get("name") or "")
        if name:
            linked_inputs[(target, name)] = [origin, origin_slot]

    api_graph: dict[str, Any] = {}
    schema_properties: dict[str, Any] = {"prompt": {"type": "string"}}
    text_nodes = [
        node_id for node_id, node in flat_nodes.items() if str(node.get("type")) == "CLIPTextEncode"
    ]
    for node_id, node in flat_nodes.items():
        class_type = str(node.get("type") or "")
        node_info = object_info.get(class_type)
        if not isinstance(node_info, dict):
            if class_type not in {"MarkdownNote", "Note"}:
                raise ValueError(f"ComfyUI template requires missing node type {class_type}")
            continue
        if int(node.get("mode") or 0) in {2, 4}:
            continue
        inputs = _widget_values(node, node_info)
        for (target_id, input_name), connection in linked_inputs.items():
            if target_id == node_id:
                inputs[input_name] = connection
        for (target_id, input_name), runtime_name in parameter_overrides.items():
            if target_id == node_id:
                _bind_runtime_parameter(inputs, input_name, runtime_name, schema_properties)
        for input_name in list(inputs):
            runtime_name = _runtime_parameter(input_name, node)
            if runtime_name:
                _bind_runtime_parameter(inputs, input_name, runtime_name, schema_properties)
        if class_type == "CLIPTextEncode" and "text" in inputs:
            title = str(node.get("title") or "").lower()
            negative = "negative" in title
            if len(text_nodes) == 1 or not negative:
                inputs["text"] = "${prompt}"
            else:
                inputs["text"] = "${negative_prompt}"
                schema_properties.setdefault("negative_prompt", {"type": "string", "default": ""})
        api_graph[node_id] = {
            "inputs": inputs,
            "class_type": class_type,
            "_meta": {
                "title": str(node.get("title") or node_info.get("display_name") or class_type)
            },
        }
    return api_graph, {"type": "object", "properties": schema_properties}


def _normalize_link(value: Any) -> tuple[str, int, str, int] | None:
    if isinstance(value, dict):
        try:
            return (
                str(value["origin_id"]),
                int(value["origin_slot"]),
                str(value["target_id"]),
                int(value["target_slot"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, list) and len(value) >= 5:
        try:
            return str(value[1]), int(value[2]), str(value[3]), int(value[4])
        except (TypeError, ValueError):
            return None
    return None


def _widget_values(node: dict[str, Any], node_info: dict[str, Any]) -> dict[str, Any]:
    raw_values = node.get("widgets_values")
    values = list(raw_values) if isinstance(raw_values, list) else [raw_values]
    if raw_values is None:
        values = []
    result: dict[str, Any] = {}
    cursor = 0
    input_info = node_info.get("input") or {}
    input_order = node_info.get("input_order") or {}
    for section in ("required", "optional"):
        definitions = input_info.get(section) or {}
        names = input_order.get(section) or list(definitions)
        for name in names:
            spec = definitions.get(name)
            if not _is_widget_spec(spec):
                continue
            if cursor < len(values):
                result[str(name)] = values[cursor]
                cursor += 1
            else:
                default = _widget_default(spec)
                if default is not None:
                    result[str(name)] = default
            options = spec[1] if isinstance(spec, list) and len(spec) > 1 else {}
            if (
                isinstance(options, dict)
                and options.get("control_after_generate")
                and cursor < len(values)
                and str(values[cursor]).lower() in _CONTROL_AFTER_GENERATE
            ):
                cursor += 1
    return result


def _is_widget_spec(spec: Any) -> bool:
    if not isinstance(spec, list) or not spec:
        return False
    return isinstance(spec[0], list) or spec[0] in _PRIMITIVE_WIDGET_TYPES


def _widget_default(spec: Any) -> Any:
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0][0] if spec[0] else None
    if len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("default")
    return None


def _runtime_parameter(input_name: str, node: dict[str, Any]) -> str | None:
    if input_name == "text" and str(node.get("type")) == "CLIPTextEncode":
        return "prompt"
    return _RUNTIME_PARAMETERS.get(input_name)


def _bind_runtime_parameter(
    inputs: dict[str, Any],
    input_name: str,
    runtime_name: str,
    schema_properties: dict[str, Any],
) -> None:
    default = inputs.get(input_name)
    placeholder = f"${{{runtime_name}}}"
    inputs[input_name] = placeholder
    if default == placeholder and runtime_name in schema_properties:
        return
    property_schema: dict[str, Any]
    if runtime_name in {"batch_size", "frames", "height", "seed", "steps", "width"}:
        property_schema = {"type": "integer"}
    elif runtime_name in {"cfg", "denoise", "fps"}:
        property_schema = {"type": "number"}
    else:
        property_schema = {"type": "string"}
    if runtime_name == "seed":
        property_schema["default"] = -1
    elif runtime_name != "prompt" and default is not None and not isinstance(default, list):
        property_schema["default"] = default
    schema_properties[runtime_name] = property_schema
