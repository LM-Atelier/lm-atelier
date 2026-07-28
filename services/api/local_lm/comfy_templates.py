from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import unquote, urlparse

from .config import Settings
from .workflow_edit_calibration import (
    EDIT_CALIBRATION_SCHEMA_KEY,
    standard_edit_calibration,
)

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
_PRIMITIVE_WIDGET_TYPES = {"BOOLEAN", "COMBO", "FLOAT", "INT", "STRING"}
_CONTROL_AFTER_GENERATE = {"decrement", "fixed", "increment", "randomize"}
COMFY_TEMPLATE_COMPILER_VERSION = 5
DEFAULT_IMAGE_EDIT_DENOISE = 0.9
_ADAPTIVE_CHECKPOINT_PREFIX = "lma_image_checkpoint_v1_"
_ADAPTIVE_CHECKPOINT_PLACEHOLDER = "__LM_ATELIER_CHECKPOINT__"

# This is a deliberately narrow capability contract, not a claim that every
# safetensors file is a runnable checkpoint. It covers ComfyUI's standard
# single-checkpoint loader and is validated against the live runtime after the
# model is downloaded, before the install is activated.
_ADAPTIVE_CHECKPOINT_GRAPH: dict[str, Any] = {
    "nodes": [
        {
            "id": 1,
            "type": "CheckpointLoaderSimple",
            "inputs": [],
            "outputs": [
                {"name": "MODEL", "type": "MODEL", "links": [1]},
                {"name": "CLIP", "type": "CLIP", "links": [2, 3]},
                {"name": "VAE", "type": "VAE", "links": [9]},
            ],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [_ADAPTIVE_CHECKPOINT_PLACEHOLDER],
        },
        {
            "id": 2,
            "type": "CLIPTextEncode",
            "title": "Prompt",
            "inputs": [
                {"name": "clip", "type": "CLIP", "link": 2},
                {"name": "text", "type": "STRING", "widget": {"name": "text"}},
            ],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [4]}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [""],
        },
        {
            "id": 3,
            "type": "CLIPTextEncode",
            "title": "Negative Prompt",
            "inputs": [
                {"name": "clip", "type": "CLIP", "link": 3},
                {"name": "text", "type": "STRING", "widget": {"name": "text"}},
            ],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [5]}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [""],
        },
        {
            "id": 4,
            "type": "EmptyLatentImage",
            "inputs": [
                {"name": "width", "type": "INT", "widget": {"name": "width"}},
                {"name": "height", "type": "INT", "widget": {"name": "height"}},
                {
                    "name": "batch_size",
                    "type": "INT",
                    "widget": {"name": "batch_size"},
                },
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [6]}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [512, 512, 1],
        },
        {
            "id": 5,
            "type": "KSampler",
            "inputs": [
                {"name": "model", "type": "MODEL", "link": 1},
                {"name": "seed", "type": "INT", "widget": {"name": "seed"}},
                {"name": "steps", "type": "INT", "widget": {"name": "steps"}},
                {"name": "cfg", "type": "FLOAT", "widget": {"name": "cfg"}},
                {
                    "name": "sampler_name",
                    "type": "COMBO",
                    "widget": {"name": "sampler_name"},
                },
                {"name": "scheduler", "type": "COMBO", "widget": {"name": "scheduler"}},
                {"name": "positive", "type": "CONDITIONING", "link": 4},
                {"name": "negative", "type": "CONDITIONING", "link": 5},
                {"name": "latent_image", "type": "LATENT", "link": 6},
                {"name": "denoise", "type": "FLOAT", "widget": {"name": "denoise"}},
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7]}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [-1, "randomize", 20, 7.0, "euler", "normal", 1.0],
        },
        {
            "id": 6,
            "type": "VAEDecode",
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": 7},
                {"name": "vae", "type": "VAE", "link": 9},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [8]}],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": [],
        },
        {
            "id": 7,
            "type": "SaveImage",
            "inputs": [
                {"name": "images", "type": "IMAGE", "link": 8},
                {
                    "name": "filename_prefix",
                    "type": "STRING",
                    "widget": {"name": "filename_prefix"},
                },
            ],
            "outputs": [],
            "properties": {"cnr_id": "comfy-core"},
            "widgets_values": ["LM Atelier"],
        },
    ],
    "links": [
        [1, 1, 0, 5, 0, "MODEL"],
        [2, 1, 1, 2, 0, "CLIP"],
        [3, 1, 1, 3, 0, "CLIP"],
        [4, 2, 0, 5, 6, "CONDITIONING"],
        [5, 3, 0, 5, 7, "CONDITIONING"],
        [6, 4, 0, 5, 8, "LATENT"],
        [7, 5, 0, 6, 0, "LATENT"],
        [8, 6, 0, 7, 0, "IMAGE"],
        [9, 1, 2, 6, 1, "VAE"],
    ],
}


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
    runtime_adaptive: bool = False

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
            parent = str(PurePosixPath(dependency.path).parent)
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
        return [
            replace(template, score=1_000)
            for template in self.available(role)
            if template.remote_id.casefold() == remote_id.casefold()
        ]

    def available(self, role: str) -> list[ComfyTemplate]:
        matches: list[ComfyTemplate] = []
        for path in self._template_files():
            try:
                raw = _read_json(path)
            except (OSError, ValueError):
                continue
            template_role = _role_for_template(path.stem, raw)
            if template_role != role:
                continue
            dependencies = tuple(_model_dependencies(raw))
            if not dependencies:
                continue
            repositories = {item.remote_id.lower() for item in dependencies}
            revisions = {item.revision for item in dependencies}
            if len(repositories) != 1 or len(revisions) != 1:
                continue
            if not _uses_only_core_nodes(raw):
                continue
            matches.append(
                ComfyTemplate(
                    id=path.stem,
                    path=path,
                    role=role,
                    operation=_operation_for_template(path.stem, role),
                    score=0,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    dependencies=dependencies,
                )
            )
        return sorted(matches, key=lambda item: item.id)

    def adaptive_checkpoint(
        self,
        remote_id: str,
        revision: str,
        selected_files: list[str],
        role: str,
        *,
        comfy_paths: dict[str, str] | None = None,
    ) -> ComfyTemplate | None:
        """Return the narrow standard-checkpoint contract when it is safe to try.

        Architecture compatibility cannot be proven from a Hub filename. The
        returned workflow therefore remains provisional until ComfyUI loads and
        validates the downloaded checkpoint.
        """

        if role != "image" or len(selected_files) != 1:
            return None
        selected = PurePosixPath(selected_files[0])
        if (
            selected.is_absolute()
            or ".." in selected.parts
            or selected.suffix.casefold() != ".safetensors"
            or not selected.name
        ):
            return None
        parent = str(selected.parent)
        expected_paths = {"checkpoints": "." if parent == "." else parent}
        if comfy_paths is not None and comfy_paths != expected_paths:
            return None
        binding = {
            "contract": 1,
            "remote_id": remote_id.casefold(),
            "revision": revision,
            "selected_file": str(selected),
        }
        binding_json = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        identifier = (
            _ADAPTIVE_CHECKPOINT_PREFIX + hashlib.sha256(binding_json.encode()).hexdigest()[:20]
        )
        template_hash = hashlib.sha256(
            json.dumps(
                {
                    "binding": binding,
                    "graph": _ADAPTIVE_CHECKPOINT_GRAPH,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        dependency = ComfyModelDependency(
            remote_id=remote_id,
            revision=revision,
            path=str(selected),
            directory="checkpoints",
            name=selected.name,
            url="",
        )
        return ComfyTemplate(
            id=identifier,
            path=Path(identifier),
            role=role,
            operation="text_to_image",
            score=100,
            sha256=template_hash,
            dependencies=(dependency,),
            runtime_adaptive=True,
        )

    def get(
        self,
        template_id: str,
        role: str,
        *,
        remote_id: str | None = None,
        revision: str | None = None,
        selected_files: list[str] | None = None,
        comfy_paths: dict[str, str] | None = None,
    ) -> ComfyTemplate:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", template_id):
            raise ValueError("invalid ComfyUI template identifier")
        if template_id.startswith(_ADAPTIVE_CHECKPOINT_PREFIX):
            if remote_id is None or revision is None or selected_files is None:
                raise ValueError("adaptive ComfyUI template requires a pinned model binding")
            adaptive = self.adaptive_checkpoint(
                remote_id,
                revision,
                selected_files,
                role,
                comfy_paths=comfy_paths,
            )
            if adaptive and adaptive.id == template_id:
                return adaptive
            raise ValueError("adaptive ComfyUI template binding does not match the download")
        for path in self._template_files():
            if path.stem != template_id:
                continue
            raw = _read_json(path)
            dependencies = tuple(_model_dependencies(raw))
            if (
                _role_for_template(path.stem, raw) != role
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
        *,
        validate_model_choices: bool = True,
        remote_id: str | None = None,
        revision: str | None = None,
        selected_files: list[str] | None = None,
        comfy_paths: dict[str, str] | None = None,
    ) -> CompiledComfyTemplate:
        template = self.get(
            template_id,
            role,
            remote_id=remote_id,
            revision=revision,
            selected_files=selected_files,
            comfy_paths=comfy_paths,
        )
        if template.runtime_adaptive:
            ui_graph = deepcopy(_ADAPTIVE_CHECKPOINT_GRAPH)
            ui_graph["nodes"][0]["widgets_values"] = [template.dependencies[0].name]
        else:
            ui_graph = _read_json(template.path)
        api_graph, input_schema = _compile_ui_graph(
            ui_graph,
            object_info,
            validate_model_choices=validate_model_choices,
        )
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
        *,
        revision: str = "main",
    ) -> ComfyTemplate:
        template = self.get(
            template_id,
            role,
            remote_id=remote_id,
            revision=revision,
            selected_files=selected_files,
            comfy_paths=comfy_paths,
        )
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


def _role_for_template(template_id: str, value: dict[str, Any] | None = None) -> str | None:
    if template_id.startswith("image_"):
        return "image"
    if template_id.startswith("video_"):
        return "video"
    if value:
        node_types = {str(node.get("type") or "").casefold() for node in _all_nodes(value)}
        if any(
            "video" in node_type or node_type in {"saveanimatedwebp", "vhs_videocombine"}
            for node_type in node_types
        ):
            return "video"
        if node_types & {"previewimage", "saveimage"}:
            return "image"
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


def derive_image_to_image(
    compiled: CompiledComfyTemplate,
    object_info: dict[str, Any],
) -> CompiledComfyTemplate | None:
    """Derive a standard whole-image edit graph when the runtime supports it."""
    if (
        getattr(compiled.template, "role", None) != "image"
        or getattr(compiled.template, "operation", None) != "text_to_image"
        or not all(isinstance(object_info.get(name), dict) for name in ("LoadImage", "VAEEncode"))
    ):
        return None
    graph = deepcopy(compiled.api_graph)
    empty_latents = {
        node_id
        for node_id, node in graph.items()
        if re.fullmatch(
            r"Empty[A-Za-z0-9_]*LatentImage",
            str(node.get("class_type") or ""),
        )
    }
    latent_inputs: list[tuple[str, str]] = []
    for node_id, node in graph.items():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        connection = inputs.get("latent_image")
        if (
            isinstance(connection, list)
            and len(connection) == 2
            and str(connection[0]) in empty_latents
        ):
            latent_inputs.append((node_id, "latent_image"))
    vae_connection = next(
        (
            inputs["vae"]
            for node in graph.values()
            if str(node.get("class_type") or "") == "VAEDecode"
            and isinstance((inputs := node.get("inputs")), dict)
            and isinstance(inputs.get("vae"), list)
            and len(inputs["vae"]) == 2
        ),
        None,
    )
    if not latent_inputs or vae_connection is None:
        return None
    load_id = _available_node_id(graph, "lma-load-image")
    encode_id = _available_node_id(graph, "lma-vae-encode")
    graph[load_id] = {
        "inputs": {"image": "${input_image}"},
        "class_type": "LoadImage",
        "_meta": {"title": "Edit source image"},
    }
    graph[encode_id] = {
        "inputs": {"pixels": [load_id, 0], "vae": deepcopy(vae_connection)},
        "class_type": "VAEEncode",
        "_meta": {"title": "Encode edit source"},
    }
    for node_id, input_name in latent_inputs:
        graph[node_id]["inputs"][input_name] = [encode_id, 0]
    schema = deepcopy(compiled.input_schema)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name in ("width", "height", "batch_size"):
            properties.pop(name, None)
        denoise = properties.get("denoise")
        if isinstance(denoise, dict):
            raw_minimum = denoise.get("minimum", 0.0)
            raw_maximum = denoise.get("maximum", 1.0)
            minimum = (
                float(raw_minimum)
                if isinstance(raw_minimum, int | float)
                and not isinstance(raw_minimum, bool)
                and math.isfinite(raw_minimum)
                else 0.0
            )
            maximum = (
                float(raw_maximum)
                if isinstance(raw_maximum, int | float)
                and not isinstance(raw_maximum, bool)
                and math.isfinite(raw_maximum)
                else 1.0
            )
            if minimum >= maximum:
                minimum, maximum = 0.0, 1.0
            denoise.update(
                {
                    "type": "number",
                    "title": "Edit strength",
                    "description": (
                        "Higher values make the requested change more visible; "
                        "lower values preserve more of the source."
                    ),
                    "default": min(max(DEFAULT_IMAGE_EDIT_DENOISE, minimum), maximum),
                    "minimum": minimum,
                    "maximum": maximum,
                    "x-lm-atelier-visibility": "basic",
                }
            )
            steps = properties.get("steps")
            steps_parameter = (
                "steps"
                if isinstance(steps, dict)
                and steps.get("type") in {"integer", "number"}
                and any(key in steps for key in ("default", "const"))
                else None
            )
            schema[EDIT_CALIBRATION_SCHEMA_KEY] = standard_edit_calibration(
                parameter="denoise",
                minimum=minimum,
                maximum=maximum,
                steps_parameter=steps_parameter,
            )
    contract = {
        "base": compiled.template.sha256,
        "operation": "image_to_image",
        "transform": "load-image-vae-encode-v5",
    }
    derived_hash = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    template = replace(
        compiled.template,
        id=f"{compiled.template.id}_image_to_image",
        operation="image_to_image",
        sha256=derived_hash,
    )
    return CompiledComfyTemplate(
        template=template,
        ui_graph=deepcopy(compiled.ui_graph),
        api_graph=graph,
        input_schema=schema,
    )


def _parse_huggingface_url(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc.lower() != "huggingface.co":
        return None
    parts = [unquote(part) for part in parsed.path.strip("/").split("/")]
    if len(parts) < 5 or parts[2] != "resolve":
        return None
    owner, name, _, revision, *file_parts = parts
    if not owner or not name or not revision or not file_parts:
        return None
    if any(part in {"", ".", ".."} for part in file_parts):
        return None
    return f"{owner}/{name}", revision, "/".join(file_parts)


def _available_node_id(graph: dict[str, Any], preferred: str) -> str:
    if preferred not in graph:
        return preferred
    suffix = 2
    while f"{preferred}-{suffix}" in graph:
        suffix += 1
    return f"{preferred}-{suffix}"


def _compile_ui_graph(
    ui_graph: dict[str, Any],
    object_info: dict[str, Any],
    *,
    validate_model_choices: bool = True,
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
        inputs = _widget_values(
            node,
            node_info,
            validate_model_choices=validate_model_choices,
        )
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


def _widget_values(
    node: dict[str, Any],
    node_info: dict[str, Any],
    *,
    validate_model_choices: bool = True,
) -> dict[str, Any]:
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
            spec = cast(list[Any], spec)
            if cursor < len(values):
                selected = values[cursor]
                choices = (
                    spec[0]
                    if isinstance(spec[0], list)
                    else (
                        spec[1].get("options")
                        if len(spec) > 1 and isinstance(spec[1], dict) and spec[0] == "COMBO"
                        else None
                    )
                )
                if (
                    validate_model_choices
                    and isinstance(choices, list)
                    and choices
                    and selected not in choices
                ):
                    raise ValueError(
                        f"ComfyUI does not advertise the template value for {name}: {selected}"
                    )
                result[str(name)] = selected
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
        options = spec[1]
        if "default" in options:
            return options["default"]
        choices = options.get("options")
        if spec[0] == "COMBO" and isinstance(choices, list) and choices:
            return choices[0]
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
