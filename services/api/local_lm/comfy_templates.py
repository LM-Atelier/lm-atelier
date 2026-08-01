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
    "noise_seed": "seed",
    "sampler_name": "sampler",
    "scheduler": "scheduler",
    "seed": "seed",
    "steps": "steps",
    "width": "width",
}
_PRIMITIVE_WIDGET_TYPES = {"BOOLEAN", "COMBO", "FLOAT", "INT", "STRING"}
_CONTROL_AFTER_GENERATE = {"decrement", "fixed", "increment", "randomize"}
COMFY_TEMPLATE_COMPILER_VERSION = 17
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
    published_date: str | None = None
    general_purpose: bool = True
    performance_hints: tuple[str, ...] = ()

    @property
    def remote_id(self) -> str:
        return self.dependencies[0].remote_id

    @property
    def revision(self) -> str:
        return self.dependencies[0].revision

    @property
    def preference_score(self) -> int:
        date_score = 0
        if self.published_date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.published_date):
            date_score = int(self.published_date.replace("-", ""))
        performance_score = (
            1_000_000 if _NATIVE_FAST_HINTS & set(self.performance_hints) else 0
        ) + (2_000_000 if "kv-cache" in self.performance_hints else 0)
        return (100_000_000 if self.general_purpose else 0) + performance_score + date_score

    @property
    def selected_files(self) -> list[str]:
        return [item.path for item in self.dependencies]

    @property
    def component_folders(self) -> dict[str, str]:
        return {item.path: item.directory for item in self.dependencies}

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


@dataclass(frozen=True)
class WorkflowAccelerationRecipe:
    kind: str
    mode: str
    step_schedules: tuple[tuple[int, int], ...]
    switched_inputs: tuple[str, ...]
    asset_kinds: tuple[str, ...]


class ComfyTemplateRegistry:
    """Discovers and compiles the workflow catalog shipped with ComfyUI."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def matches(self, remote_id: str, role: str) -> list[ComfyTemplate]:
        matches = [
            replace(template, score=1_000 + template.preference_score)
            for template in self.available(role)
            if template.remote_id.casefold() == remote_id.casefold()
        ]
        return sorted(matches, key=lambda item: (-item.score, item.id))

    def available(self, role: str) -> list[ComfyTemplate]:
        matches: list[ComfyTemplate] = []
        paths = self._template_files()
        metadata = _template_index_metadata(paths)
        for path in paths:
            try:
                raw = _read_json(path)
            except (OSError, ValueError):
                continue
            template_role = _role_for_template(path.stem, raw)
            if template_role != role:
                continue
            operation = _operation_for_template(path.stem, role)
            if operation is None:
                continue
            dependencies = tuple(_model_dependencies(raw))
            if not dependencies:
                continue
            if not _supports_dependency_bundle(dependencies):
                continue
            if not _uses_only_core_nodes(raw):
                continue
            template_metadata = metadata.get(path.stem, {})
            matches.append(
                ComfyTemplate(
                    id=path.stem,
                    path=path,
                    role=role,
                    operation=operation,
                    score=0,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    dependencies=dependencies,
                    published_date=_metadata_string(template_metadata, "date"),
                    general_purpose=_is_general_purpose_template(
                        template_metadata,
                        dependencies,
                    ),
                    performance_hints=_template_performance_hints(template_metadata),
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
            operation = _operation_for_template(path.stem, role)
            if (
                _role_for_template(path.stem, raw) != role
                or operation is None
                or not dependencies
                or not _supports_dependency_bundle(dependencies)
                or not _uses_only_core_nodes(raw)
            ):
                break
            template_metadata = _template_index_metadata(self._template_files()).get(path.stem, {})
            return ComfyTemplate(
                id=path.stem,
                path=path,
                role=role,
                operation=operation,
                score=0,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                dependencies=dependencies,
                published_date=_metadata_string(template_metadata, "date"),
                general_purpose=_is_general_purpose_template(
                    template_metadata,
                    dependencies,
                ),
                performance_hints=_template_performance_hints(template_metadata),
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
        acceleration_recipes = _resolve_declared_workflow_acceleration(
            ui_graph,
            operation=template.operation,
        )
        api_graph, input_schema = _compile_ui_graph(
            ui_graph,
            object_info,
            operation=template.operation,
            validate_model_choices=validate_model_choices,
        )
        if acceleration_recipes:
            input_schema["x-lm-atelier-workflow-acceleration"] = _workflow_acceleration_provenance(
                acceleration_recipes
            )
        performance = _workflow_performance_provenance(
            template,
            ui_graph,
            api_graph,
            input_schema,
            object_info,
            acceleration_recipes,
        )
        if performance:
            bounds = _apply_workflow_performance_bounds(input_schema, performance)
            if bounds:
                performance["setting_bounds"] = bounds
            input_schema["x-lm-atelier-workflow-performance"] = performance
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
        file_sources: dict[str, Any] | None = None,
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
        sources = file_sources or {}
        expected_companions = {
            dependency.path: dependency
            for dependency in template.dependencies
            if dependency.remote_id.casefold() != remote_id.casefold()
        }
        if set(sources) != set(expected_companions):
            raise ValueError("download companion sources do not match the ComfyUI template")
        for destination, dependency in expected_companions.items():
            source = sources[destination]
            source_remote_id = str(getattr(source, "remote_id", "") or "")
            source_revision = str(getattr(source, "revision", "") or "")
            source_filename = str(getattr(source, "filename", "") or "")
            if (
                source_remote_id.casefold() != dependency.remote_id.casefold()
                or source_filename != dependency.path
                or (dependency.revision != "main" and source_revision != dependency.revision)
            ):
                raise ValueError("download companion source binding changed")
        return template

    def _template_files(self) -> list[Path]:
        for candidate in self._template_directories():
            if candidate.is_dir():
                return sorted(candidate.glob("*.json"))
        return []

    def _template_directories(self) -> list[Path]:
        """Template directories to try, most authoritative first."""
        ordered: list[Path] = []
        seen: set[Path] = set()
        for site_packages in self._site_packages_roots():
            candidate = site_packages / "comfyui_workflow_templates_json" / "templates"
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered

    def _site_packages_roots(self) -> list[Path]:
        """Interpreter site-packages directories that may ship the templates.

        The templates install alongside the interpreter that runs ComfyUI, which
        is not necessarily inside the ComfyUI directory. The official Windows
        portable release keeps ``python_embeded`` beside ``ComfyUI``, while a
        checkout built by hand usually keeps its virtual environment within.
        Prefer the configured executable, because that is the interpreter that
        actually executes the graph.
        """
        roots: list[Path] = []
        executable = self.settings.comfy_executable
        if executable:
            binary = executable.expanduser().resolve().parent
            roots.append(binary / "Lib" / "site-packages")
            roots.append(binary.parent / "Lib" / "site-packages")
            roots.extend(sorted(binary.parent.glob("lib/python*/site-packages")))
        directory = self.settings.comfy_directory
        if directory:
            root = directory.expanduser().resolve()
            for name in (".venv", "venv"):
                roots.append(root / name / "Lib" / "site-packages")
                roots.extend(sorted((root / name).glob("lib/python*/site-packages")))
            roots.append(root.parent / "python_embeded" / "Lib" / "site-packages")
        return roots


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z]+|\d+", value.lower())
        if len(token) > 1 or token == "z"
    }


def _template_index_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    index_path = next((path for path in paths if path.name == "index.json"), None)
    if index_path is None:
        return {}
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    result: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str) and name:
                result[name] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw)
    return result


def _metadata_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) and value else None


def _template_performance_hints(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Return conservative performance semantics declared by the template author."""

    values = [metadata.get("title"), metadata.get("description")]
    tags = metadata.get("tags")
    if isinstance(tags, list):
        values.extend(tags)
    tokens = set().union(*(_tokens(value) for value in values if isinstance(value, str)))
    hints: list[str] = [
        hint
        for hint in (
            "distilled",
            "hyper",
            "lcm",
            "lightning",
            "schnell",
            "tcd",
            "turbo",
        )
        if hint in tokens
    ]
    if {"few", "step"} <= tokens:
        hints.append("few-step")
    if {"low", "step"} <= tokens:
        hints.append("low-step")
    cache_terms = {"cache", "cached", "caching"}
    if tokens & cache_terms and ("kv" in tokens or {"key", "value"} <= tokens):
        hints.append("kv-cache")
    return tuple(hints)


def _metadata_marks_specialized_lora(metadata: dict[str, Any]) -> bool:
    values = [metadata.get("title"), metadata.get("description")]
    tags = metadata.get("tags")
    if isinstance(tags, list):
        values.extend(tags)
    return any(isinstance(value, str) and "lora" in value.casefold() for value in values)


def _is_general_purpose_template(
    metadata: dict[str, Any],
    dependencies: tuple[ComfyModelDependency, ...],
) -> bool:
    lora_count = sum(dependency.directory == "loras" for dependency in dependencies)
    return not _metadata_marks_specialized_lora(metadata) and lora_count <= 1


def _supports_dependency_bundle(
    dependencies: tuple[ComfyModelDependency, ...],
) -> bool:
    paths: set[str] = set()
    repository_revisions: dict[str, set[str]] = {}
    folder_parents: dict[str, str] = {}
    for dependency in dependencies:
        path = PurePosixPath(dependency.path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
        ):
            return False
        identity = path.as_posix().casefold()
        if identity in paths:
            return False
        paths.add(identity)
        repository_revisions.setdefault(dependency.remote_id.casefold(), set()).add(
            dependency.revision
        )
        parent = str(path.parent)
        parent = "." if parent == "." else parent
        existing = folder_parents.setdefault(dependency.directory, parent)
        if existing != parent:
            return False
    return all(len(revisions) == 1 for revisions in repository_revisions.values())


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


def _operation_markers(template_id: str) -> set[str]:
    # Not _tokens: that helper splits letter and digit runs apart and drops
    # single characters, so "i2v", "s2v", and "img2img" could never appear in
    # its output and every Wan variant classified as text-to-video.
    return set(re.findall(r"[a-z0-9]+", template_id.lower()))


def _operation_for_template(template_id: str, role: str) -> str | None:
    """Map a template id to a supported operation, or None for none.

    None means the workflow needs an input LM Atelier cannot provide - an
    audio- or speech-driven video template offered for a text prompt would
    install a workflow whose required input never arrives.
    """

    markers = _operation_markers(template_id)
    if role == "video":
        if {"i2v", "image2video", "img2video"} & markers:
            return "image_to_video"
        if {"s2v", "a2v", "speech2video", "audio2video"} & markers:
            return None
        return "text_to_video"
    return (
        "image_to_image" if {"img2img", "image2image", "i2i", "edit"} & markers else "text_to_image"
    )


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
    folder_priority = {
        "checkpoints": 0,
        "diffusion_models": 1,
        "unet": 1,
    }
    return sorted(
        dependencies.values(),
        key=lambda item: (
            folder_priority.get(item.directory, 10),
            item.directory,
            item.path,
            item.remote_id.casefold(),
        ),
    )


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


_ACCELERATION_CONTROL_KINDS = {
    "accelerated": "acceleration",
    "acceleration": "acceleration",
    "distilled": "distilled",
    "fast": "fast",
    "few_step": "few-step",
    "hyper": "hyper",
    "hyper_sd": "hyper",
    "lcm": "lcm",
    "lightning": "lightning",
    "low_step": "low-step",
    "schnell": "schnell",
    "tcd": "tcd",
    "turbo": "turbo",
}
_ACCELERATION_CONTROL_AFFIXES = {"enable", "enabled", "mode", "use"}
_ACCELERATION_RECIPE_INPUTS = {
    "cfg",
    "denoise",
    "end_at_step",
    "guidance",
    "guidance_scale",
    "model",
    "num_steps",
    "sampler",
    "sampler_name",
    "sampling_steps",
    "scheduler",
    "shift",
    "sigmas",
    "start_at_step",
    "steps",
    "total_steps",
}
_ACCELERATION_STEP_INPUTS = {
    "num_steps",
    "sampling_steps",
    "steps",
    "total_steps",
}
_ACCELERATION_MODEL_INPUTS = {"model"}
_ACCELERATION_ASSET_DIRECTORIES = {
    "checkpoints",
    "diffusion_models",
    "loras",
    "unet",
    "unets",
}


_MAX_ACCELERATION_GRAPHS = 256


def _candidate_workflow_graphs(
    value: dict[str, Any],
) -> tuple[dict[str, Any], ...] | None:
    candidates: list[dict[str, Any]] = []
    pending = [value]
    visited: set[int] = set()
    while pending:
        graph = pending.pop()
        identity = id(graph)
        if identity in visited:
            continue
        visited.add(identity)
        if len(candidates) >= _MAX_ACCELERATION_GRAPHS:
            return None
        candidates.append(graph)
        definitions = graph.get("definitions")
        if not isinstance(definitions, dict):
            continue
        children = [item for item in definitions.get("subgraphs", []) if isinstance(item, dict)]
        pending.extend(reversed(children))
    return tuple(candidates)


def _resolve_declared_workflow_acceleration(
    ui_graph: dict[str, Any],
    *,
    operation: str,
) -> tuple[WorkflowAccelerationRecipe, ...]:
    """Enable complete author-declared accelerated media-workflow branches."""

    if operation not in {
        "image_to_image",
        "image_to_video",
        "text_to_image",
        "text_to_video",
    }:
        return ()
    candidates = _candidate_workflow_graphs(ui_graph)
    if candidates is None:
        return ()
    resolved: list[WorkflowAccelerationRecipe] = []
    for graph in candidates:
        nodes = {
            str(node["id"]): node
            for node in graph.get("nodes", [])
            if isinstance(node, dict) and node.get("id") is not None
        }
        links = [
            normalized
            for item in graph.get("links", [])
            if (normalized := _normalize_link(item)) is not None
        ]
        graph_inputs = [item for item in graph.get("inputs", []) if isinstance(item, dict)]
        declared_controls: dict[str, str] = {}
        for origin, origin_slot, target, target_slot in links:
            if origin != "-10" or not 0 <= origin_slot < len(graph_inputs):
                continue
            definition = graph_inputs[origin_slot]
            kind = _acceleration_control_kind(
                str(definition.get("label") or definition.get("name") or "")
            )
            target_node = nodes.get(target)
            target_inputs = target_node.get("inputs", []) if target_node else []
            if (
                kind is not None
                and str(definition.get("type") or "").upper() == "BOOLEAN"
                and str((target_node or {}).get("type") or "") == "PrimitiveBoolean"
                and 0 <= target_slot < len(target_inputs)
                and str(target_inputs[target_slot].get("type") or "").upper() == "BOOLEAN"
            ):
                declared_controls[target] = kind
        graph_recipes: list[tuple[dict[str, Any], WorkflowAccelerationRecipe]] = []
        for control_id, kind in declared_controls.items():
            switches: list[dict[str, Any]] = []
            for origin, _, target, target_slot in links:
                target_node = nodes.get(target)
                target_inputs = target_node.get("inputs", []) if target_node else []
                if (
                    origin == control_id
                    and target_node
                    and str(target_node.get("type") or "") == "ComfySwitchNode"
                    and 0 <= target_slot < len(target_inputs)
                    and str(target_inputs[target_slot].get("name") or "") == "switch"
                ):
                    switches.append(target_node)
            recipe = _declared_acceleration_recipe(kind, switches, nodes, links)
            if recipe is None:
                continue
            graph_recipes.append((nodes[control_id], recipe))
        if len(graph_recipes) != 1:
            continue
        control, recipe = graph_recipes[0]
        raw_values = control.get("widgets_values")
        values = list(raw_values) if isinstance(raw_values, list) else []
        if values:
            values[0] = True
        else:
            values = [True]
        control["widgets_values"] = values
        resolved.append(recipe)
    return tuple(resolved)


def _acceleration_control_kind(label: str) -> str | None:
    tokens = [
        token
        for token in re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_").split("_")
        if token
    ]
    semantic = "_".join(token for token in tokens if token not in _ACCELERATION_CONTROL_AFFIXES)
    return _ACCELERATION_CONTROL_KINDS.get(semantic)


def _declared_acceleration_recipe(
    kind: str,
    switches: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
    links: list[tuple[str, int, str, int]],
) -> WorkflowAccelerationRecipe | None:
    step_schedules: list[tuple[int, int]] = []
    switched_inputs: set[str] = set()
    asset_kinds: set[str] = set()

    for switch in switches:
        output_inputs = _switch_output_input_names(switch, nodes, links)
        if not output_inputs or not output_inputs <= _ACCELERATION_RECIPE_INPUTS:
            return None
        recipe_inputs = output_inputs & _ACCELERATION_RECIPE_INPUTS
        switched_inputs.update(recipe_inputs)
        if recipe_inputs & _ACCELERATION_STEP_INPUTS:
            schedule = _switch_step_schedule(switch, nodes, links)
            if schedule is None:
                return None
            step_schedules.append(schedule)
        if recipe_inputs & _ACCELERATION_MODEL_INPUTS:
            asset_kind = _switch_bundled_acceleration_asset(switch, nodes, links)
            if asset_kind is None:
                return None
            asset_kinds.add(asset_kind)

    if not step_schedules:
        return None
    mode = (
        "bundled-four-step"
        if asset_kinds == {"bundled-lora"}
        and all(fast_steps == 4 for fast_steps, _ in step_schedules)
        else "declared-low-step"
    )
    return WorkflowAccelerationRecipe(
        kind=kind,
        mode=mode,
        step_schedules=tuple(sorted(step_schedules)),
        switched_inputs=tuple(sorted(switched_inputs)),
        asset_kinds=tuple(sorted(asset_kinds)),
    )


def _workflow_acceleration_provenance(
    recipes: tuple[WorkflowAccelerationRecipe, ...],
) -> dict[str, Any]:
    resolved: list[dict[str, Any]] = []
    for recipe in recipes:
        item: dict[str, Any] = {
            "kind": recipe.kind,
            "mode": recipe.mode,
            "step_schedules": [
                {"steps": steps, "baseline_steps": baseline_steps}
                for steps, baseline_steps in recipe.step_schedules
            ],
            "switched_inputs": list(recipe.switched_inputs),
        }
        if recipe.asset_kinds:
            item["asset_kinds"] = list(recipe.asset_kinds)
        resolved.append(item)
    result: dict[str, Any] = {
        "version": 1,
        "mode": (recipes[0].mode if len(recipes) == 1 else "declared-accelerated-branches"),
        "recipe_count": len(recipes),
        "recipes": resolved,
    }
    all_schedules = [schedule for recipe in recipes for schedule in recipe.step_schedules]
    fast_steps = {schedule[0] for schedule in all_schedules}
    baseline_steps = {schedule[1] for schedule in all_schedules}
    if len(fast_steps) == 1:
        result["steps"] = fast_steps.pop()
    if len(baseline_steps) == 1:
        result["baseline_steps"] = baseline_steps.pop()
    return result


_NATIVE_FAST_HINTS = {
    "distilled",
    "few-step",
    "hyper",
    "low-step",
    "lcm",
    "lightning",
    "schnell",
    "tcd",
    "turbo",
}
_MAX_NATIVE_FAST_STEPS = 12


def _workflow_performance_provenance(
    template: ComfyTemplate,
    ui_graph: dict[str, Any],
    api_graph: dict[str, Any],
    input_schema: dict[str, Any],
    object_info: dict[str, Any],
    acceleration_recipes: tuple[WorkflowAccelerationRecipe, ...],
) -> dict[str, Any] | None:
    """Describe verified authored fast paths without changing workflow semantics."""

    if template.operation not in {
        "image_to_image",
        "image_to_video",
        "text_to_image",
        "text_to_video",
    }:
        return None
    signals: list[dict[str, Any]] = []
    hints = set(template.performance_hints)
    techniques = sorted(hints & _NATIVE_FAST_HINTS)
    step_defaults = _authored_step_defaults(ui_graph, object_info)
    properties = input_schema.get("properties")
    step_schema = properties.get("steps") if isinstance(properties, dict) else None
    schema_default = step_schema.get("default") if isinstance(step_schema, dict) else None
    authored_fast_steps = (
        schema_default
        if len(step_defaults) == 1
        and isinstance(schema_default, int)
        and not isinstance(schema_default, bool)
        and schema_default in step_defaults
        and 1 <= schema_default <= _MAX_NATIVE_FAST_STEPS
        else None
    )
    if techniques and authored_fast_steps is not None:
        signals.append(
            {
                "kind": "native-low-step",
                "techniques": techniques,
                "steps": authored_fast_steps,
                "source": "template-metadata-and-authored-graph",
            }
        )

    executed_nodes = _executed_workflow_nodes(api_graph, object_info)
    cache_nodes = sorted(
        {
            str(node.get("class_type"))
            for node_id, node in api_graph.items()
            if node_id in executed_nodes
            and _is_model_cache_node(str(node.get("class_type") or ""), object_info)
        }
    )
    if cache_nodes:
        cache_signal: dict[str, Any] = {
            "kind": "model-cache",
            "technique": "kv-cache",
            "node_types": cache_nodes,
            "source": "executed-runtime-node-contract",
        }
        if authored_fast_steps is not None:
            cache_signal["steps"] = authored_fast_steps
        signals.append(cache_signal)

    if acceleration_recipes:
        acceleration = _workflow_acceleration_provenance(acceleration_recipes)
        signal: dict[str, Any] = {
            "kind": "declared-acceleration",
            "mode": acceleration["mode"],
            "recipe_count": acceleration["recipe_count"],
            "source": "author-declared-graph-branch",
        }
        if "steps" in acceleration:
            signal["steps"] = acceleration["steps"]
        signals.append(signal)
    if not signals:
        return None
    return {
        "version": 1,
        "signals": signals,
        "native_optimized": any(
            signal["kind"] in {"native-low-step", "model-cache"} for signal in signals
        ),
    }


def _apply_workflow_performance_bounds(
    input_schema: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    properties = input_schema.get("properties")
    step_schema = properties.get("steps") if isinstance(properties, dict) else None
    signals = performance.get("signals")
    if not isinstance(step_schema, dict) or not isinstance(signals, list):
        return {}
    steps = {
        value
        for signal in signals
        if isinstance(signal, dict)
        and isinstance((value := signal.get("steps")), int)
        and not isinstance(value, bool)
        and value > 0
    }
    if len(steps) != 1:
        return {}
    authored_steps = steps.pop()
    if step_schema.get("default") != authored_steps:
        return {}
    recommended_maximum = min(
        _MAX_NATIVE_FAST_STEPS,
        max(authored_steps, authored_steps * 2),
    )
    existing_maximum = step_schema.get("maximum")
    if isinstance(existing_maximum, int | float) and not isinstance(existing_maximum, bool):
        recommended_maximum = min(recommended_maximum, int(existing_maximum))
    if recommended_maximum < authored_steps:
        return {}
    step_schema["maximum"] = recommended_maximum
    return {
        "steps": {
            "default": authored_steps,
            "maximum": recommended_maximum,
        }
    }


def _authored_step_defaults(
    ui_graph: dict[str, Any],
    object_info: dict[str, Any],
) -> set[int]:
    candidates = _candidate_workflow_graphs(ui_graph)
    if candidates is None:
        return set()
    values: set[int] = set()
    for graph in candidates:
        for node in graph.get("nodes", []):
            if not isinstance(node, dict) or int(node.get("mode") or 0) in {2, 4}:
                continue
            node_info = object_info.get(str(node.get("type") or ""))
            if not isinstance(node_info, dict):
                continue
            inputs = _widget_values(
                node,
                node_info,
                validate_model_choices=False,
            )
            value = inputs.get("steps")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                values.add(value)
    return values


def _executed_workflow_nodes(
    api_graph: dict[str, Any],
    object_info: dict[str, Any],
) -> set[str]:
    pending = [
        node_id
        for node_id, node in api_graph.items()
        if bool((object_info.get(str(node.get("class_type") or "")) or {}).get("output_node"))
    ]
    executed = set(pending)
    while pending:
        node = api_graph.get(pending.pop())
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            if not isinstance(value, list) or len(value) != 2:
                continue
            origin = str(value[0])
            if origin in api_graph and origin not in executed:
                executed.add(origin)
                pending.append(origin)
    return executed


def _is_model_cache_node(
    class_type: str,
    object_info: dict[str, Any],
) -> bool:
    node_info = object_info.get(class_type)
    if not isinstance(node_info, dict):
        return False
    semantic_text = " ".join(
        str(value or "")
        for value in (
            class_type,
            node_info.get("display_name"),
            node_info.get("description"),
        )
    )
    semantic_text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", semantic_text)
    semantic_text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", semantic_text)
    tokens = _tokens(semantic_text)
    if not (
        tokens & {"cache", "cached", "caching"} and ("kv" in tokens or {"key", "value"} <= tokens)
    ):
        return False
    input_info = node_info.get("input")
    model_input = False
    if isinstance(input_info, dict):
        for section in ("required", "optional"):
            definitions = input_info.get(section)
            if not isinstance(definitions, dict):
                continue
            for spec in definitions.values():
                if isinstance(spec, list) and spec and spec[0] == "MODEL":
                    model_input = True
    outputs = node_info.get("output")
    model_output = isinstance(outputs, list) and "MODEL" in outputs
    return model_input and model_output


def _switch_input_origin(
    switch: dict[str, Any],
    input_name: str,
    links: list[tuple[str, int, str, int]],
) -> str | None:
    switch_id = str(switch.get("id"))
    slot = next(
        (
            index
            for index, item in enumerate(switch.get("inputs", []))
            if isinstance(item, dict) and str(item.get("name") or "") == input_name
        ),
        None,
    )
    if slot is None:
        return None
    return next(
        (
            origin
            for origin, _, target, target_slot in links
            if target == switch_id and target_slot == slot
        ),
        None,
    )


def _primitive_number(node: dict[str, Any] | None) -> float | None:
    if not node or str(node.get("type") or "") not in {"PrimitiveInt", "PrimitiveFloat"}:
        return None
    values = node.get("widgets_values")
    if (
        not isinstance(values, list)
        or not values
        or isinstance(values[0], bool)
        or not isinstance(values[0], int | float)
    ):
        return None
    return float(values[0])


def _switch_output_input_names(
    switch: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    links: list[tuple[str, int, str, int]],
) -> set[str]:
    switch_id = str(switch.get("id"))
    result: set[str] = set()
    for origin, _, target, target_slot in links:
        target_node = nodes.get(target)
        target_inputs = target_node.get("inputs", []) if target_node else []
        if origin == switch_id and 0 <= target_slot < len(target_inputs):
            name = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(target_inputs[target_slot].get("name") or "").casefold(),
            ).strip("_")
            if name:
                result.add(name)
    return result


def _switch_step_schedule(
    switch: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    links: list[tuple[str, int, str, int]],
) -> tuple[int, int] | None:
    false_value = _primitive_number(
        nodes.get(_switch_input_origin(switch, "on_false", links) or "")
    )
    true_value = _primitive_number(nodes.get(_switch_input_origin(switch, "on_true", links) or ""))
    if (
        true_value is None
        or false_value is None
        or not true_value.is_integer()
        or not false_value.is_integer()
    ):
        return None
    fast_steps = int(true_value)
    baseline_steps = int(false_value)
    if fast_steps < 1 or baseline_steps <= fast_steps:
        return None
    return fast_steps, baseline_steps


def _switch_bundled_acceleration_asset(
    switch: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    links: list[tuple[str, int, str, int]],
) -> str | None:
    true_node = nodes.get(_switch_input_origin(switch, "on_true", links) or "")
    if not true_node:
        return None
    node_type = str(true_node.get("type") or "")
    if "loader" not in node_type.casefold():
        return None
    properties = true_node.get("properties")
    models = properties.get("models") if isinstance(properties, dict) else None
    if isinstance(models, dict):
        model_items = [models]
    elif isinstance(models, list):
        model_items = [item for item in models if isinstance(item, dict)]
    else:
        model_items = []
    values = true_node.get("widgets_values")
    selected_values = (
        {str(value) for value in values if isinstance(value, str)}
        if isinstance(values, list)
        else set()
    )
    bundled = [
        item
        for item in model_items
        if str(item.get("directory") or "").casefold() in _ACCELERATION_ASSET_DIRECTORIES
        and str(item.get("name") or "") in selected_values
        and _parse_huggingface_url(str(item.get("url") or "")) is not None
    ]
    if not bundled:
        return None
    return (
        "bundled-lora"
        if node_type.casefold().startswith("loraloader")
        or all(str(item.get("directory") or "").casefold() == "loras" for item in bundled)
        else "bundled-model"
    )


def _compile_ui_graph(
    ui_graph: dict[str, Any],
    object_info: dict[str, Any],
    *,
    operation: str,
    validate_model_choices: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    definitions = {
        str(item.get("id")): item
        for item in ((ui_graph.get("definitions") or {}).get("subgraphs") or [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    flat_nodes: dict[str, dict[str, Any]] = {}
    links: list[tuple[str, int, str, int]] = []
    group_inputs: dict[str, dict[int, list[tuple[str, int, str, str]]]] = {}
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
        subgraph_input_targets: dict[int, list[tuple[str, int, str, str]]] = {}
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
                parameter_definition = (
                    subgraph_inputs[origin_slot] if origin_slot < len(subgraph_inputs) else {}
                )
                parameter = str(parameter_definition.get("name") or "")
                parameter_label = str(parameter_definition.get("label") or "")
                subgraph_input_targets.setdefault(origin_slot, []).append(
                    (target_key, target_slot, parameter, parameter_label)
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
            for target_key, target_slot, parameter, parameter_label in targets:
                target_node = flat_nodes.get(target_key)
                if not target_node:
                    continue
                target_inputs = target_node.get("inputs") or []
                if target_slot >= len(target_inputs):
                    continue
                input_name = str(target_inputs[target_slot].get("name") or "")
                input_type = str(target_inputs[target_slot].get("type") or "")
                runtime_name = _subgraph_runtime_parameter(
                    parameter,
                    parameter_label,
                    input_name,
                    input_type,
                    target_node,
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
            for inner_target, inner_slot, _, _ in group_inputs[target].get(target_slot, []):
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
    source_nodes = [
        node_id
        for node_id, node in flat_nodes.items()
        if operation in {"image_to_image", "image_to_video"}
        and str(node.get("type")) == "LoadImage"
    ]
    native_image_conditioning = operation == "image_to_image" and _source_reaches_conditioning(
        flat_nodes,
        links,
        source_nodes,
    )
    source_indices = {node_id: index for index, node_id in enumerate(source_nodes)}
    for node_id, node in flat_nodes.items():
        class_type = str(node.get("type") or "")
        if class_type == "SaveImageAdvanced" and isinstance(object_info.get("SaveImage"), dict):
            class_type = "SaveImage"
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
            runtime_input_names={"image"} if node_id in source_indices else set(),
        )
        for (target_id, input_name), connection in linked_inputs.items():
            if target_id == node_id:
                inputs[input_name] = connection
        overridden_inputs: set[str] = set()
        for (target_id, input_name), runtime_name in parameter_overrides.items():
            if target_id == node_id and (target_id, input_name) not in linked_inputs:
                if runtime_name == "denoise" and native_image_conditioning:
                    schema_properties.setdefault("denoise", {"readOnly": True})
                else:
                    _bind_runtime_parameter(inputs, input_name, runtime_name, schema_properties)
                overridden_inputs.add(input_name)
        for input_name in list(inputs):
            if input_name in overridden_inputs:
                continue
            if (node_id, input_name) in linked_inputs:
                runtime_name = _runtime_parameter(input_name, node)
                if runtime_name and runtime_name not in schema_properties:
                    schema_properties[runtime_name] = {"readOnly": True}
                continue
            runtime_name = _runtime_parameter(input_name, node)
            if runtime_name:
                if runtime_name == "denoise" and native_image_conditioning:
                    schema_properties.setdefault("denoise", {"readOnly": True})
                else:
                    _bind_runtime_parameter(inputs, input_name, runtime_name, schema_properties)
        source_index = source_indices.get(node_id)
        if source_index is not None and "image" in inputs:
            runtime_name = (
                "input_image" if len(source_nodes) == 1 else f"input_image_{source_index}"
            )
            _bind_runtime_parameter(inputs, "image", runtime_name, schema_properties)
            schema_properties[runtime_name] = {"type": "string"}
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
    for runtime_name in sorted(set(_RUNTIME_PARAMETERS.values()) | {"negative_prompt"}):
        schema_properties.setdefault(runtime_name, {"readOnly": True})
    return api_graph, {"type": "object", "properties": schema_properties}


def _source_reaches_conditioning(
    nodes: dict[str, dict[str, Any]],
    links: list[tuple[str, int, str, int]],
    source_nodes: list[str],
) -> bool:
    """Detect architecture-native image conditioning without model-name rules."""

    if not source_nodes:
        return False
    targets_by_origin: dict[str, list[str]] = {}
    for origin, _, target, _ in links:
        targets_by_origin.setdefault(origin, []).append(target)
    pending = list(source_nodes)
    visited = set(source_nodes)
    while pending:
        origin = pending.pop()
        for target in targets_by_origin.get(origin, []):
            if target in visited:
                continue
            visited.add(target)
            node = nodes.get(target)
            if not node:
                continue
            if any(
                isinstance(output, dict) and str(output.get("type") or "") == "CONDITIONING"
                for output in node.get("outputs") or []
            ):
                return True
            pending.append(target)
    return False


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
    runtime_input_names: set[str] | None = None,
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
                    and str(name) not in (runtime_input_names or set())
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
    if input_name in {"prompt", "text"} and _is_conditioning_text_input(
        input_name,
        node,
    ):
        title = str(node.get("title") or "").casefold()
        return "negative_prompt" if "negative" in title else "prompt"
    return _RUNTIME_PARAMETERS.get(input_name)


def _subgraph_runtime_parameter(
    parameter: str,
    parameter_label: str,
    input_name: str,
    input_type: str,
    node: dict[str, Any],
) -> str | None:
    aliases = {
        "negative": "negative_prompt",
        "negative_prompt": "negative_prompt",
        "negative_text": "negative_prompt",
        "positive": "prompt",
        "positive_prompt": "prompt",
        "positive_text": "prompt",
        "prompt": "prompt",
        "text": "prompt",
    }
    if input_type == "STRING":
        for value in (parameter_label, parameter):
            normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
            if runtime_name := aliases.get(normalized):
                return runtime_name
    return _runtime_parameter(parameter or input_name, node) or _runtime_parameter(
        input_name,
        node,
    )


def _is_conditioning_text_input(input_name: str, node: dict[str, Any]) -> bool:
    if input_name not in {"prompt", "text"}:
        return False
    return any(
        isinstance(output, dict) and str(output.get("type") or "") == "CONDITIONING"
        for output in node.get("outputs") or []
    )


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
