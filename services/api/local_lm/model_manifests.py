from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .capability_packs import architecture_family_contracts

MAX_METADATA_BYTES = 1024 * 1024
MAX_WEIGHT_HEADER_BYTES = 16 * 1024 * 1024
MAX_METADATA_NODES = 100_000
MAX_GGUF_FIELDS = 4_096
MAX_GGUF_ARRAY_ITEMS = 100_000

_GGUF_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


class ModelManifestError(ValueError):
    """Declarative model metadata is malformed, unsafe, or exceeds a bound."""


@dataclass(frozen=True)
class InspectedComponent:
    path: str
    kind: str
    target_folder: str
    architecture: str | None = None
    family: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelManifestInspection:
    architecture: str | None
    family: str | None
    components: tuple[InspectedComponent, ...]
    metadata_files: tuple[str, ...]


def inspect_repository_metadata(
    files: Mapping[str, bytes],
    selected_paths: list[str],
    *,
    role: str,
) -> ModelManifestInspection:
    """Inspect bounded data-only metadata without importing repository code."""

    metadata: dict[str, dict[str, Any]] = {}
    components: list[InspectedComponent] = []
    for raw_path, content in files.items():
        path = _safe_path(raw_path)
        name = path.name.casefold()
        if name.endswith(".json"):
            metadata[path.as_posix()] = _bounded_json_object(content)
        elif name.endswith(".safetensors"):
            components.append(_inspect_safetensors(path, content))
        elif name.endswith(".gguf"):
            components.append(_inspect_gguf(path, content))

    architecture, family = _repository_identity(metadata, components, role)
    selected = {_safe_path(value).as_posix() for value in selected_paths}
    by_path = {component.path: component for component in components}
    resolved: list[InspectedComponent] = []
    for selected_path in sorted(selected):
        component = by_path.get(selected_path)
        if component:
            resolved.append(
                InspectedComponent(
                    path=component.path,
                    kind=component.kind,
                    target_folder=component.target_folder,
                    architecture=component.architecture or architecture,
                    family=component.family or family,
                    metadata=component.metadata,
                )
            )
            continue
        resolved.append(
            InspectedComponent(
                path=selected_path,
                kind=_component_kind_from_path(selected_path, role),
                target_folder=_target_folder(selected_path, role),
                architecture=architecture,
                family=family,
            )
        )
    return ModelManifestInspection(
        architecture=architecture,
        family=family,
        components=tuple(resolved),
        metadata_files=tuple(sorted(metadata)),
    )


def _safe_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        not value
        or len(value) > 1_000
        or path.is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise ModelManifestError("model metadata path is unsafe")
    return path


def _bounded_json_object(content: bytes) -> dict[str, Any]:
    if len(content) > MAX_METADATA_BYTES:
        raise ModelManifestError("model metadata JSON exceeds the size limit")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ModelManifestError("model metadata JSON is malformed") from exc
    if not isinstance(value, dict):
        raise ModelManifestError("model metadata JSON must contain an object")
    if _count_nodes(value) > MAX_METADATA_NODES:
        raise ModelManifestError("model metadata JSON is too complex")
    return value


def _count_nodes(value: Any) -> int:
    count = 0
    pending = [value]
    while pending:
        current = pending.pop()
        count += 1
        if count > MAX_METADATA_NODES:
            return count
        if isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return count


def _inspect_safetensors(path: PurePosixPath, content: bytes) -> InspectedComponent:
    if len(content) < 8:
        raise ModelManifestError("safetensors header is truncated")
    header_size = int.from_bytes(content[:8], "little")
    if header_size < 2 or header_size > MAX_WEIGHT_HEADER_BYTES:
        raise ModelManifestError("safetensors header exceeds the size limit")
    if len(content) < 8 + header_size:
        raise ModelManifestError("safetensors header is truncated")
    header = _bounded_json_object_with_limit(
        content[8 : 8 + header_size],
        MAX_WEIGHT_HEADER_BYTES,
    )
    tensor_names = sorted(str(key) for key in header if key != "__metadata__")
    if len(tensor_names) > MAX_METADATA_NODES:
        raise ModelManifestError("safetensors tensor index is too large")
    raw_metadata = header.get("__metadata__") or {}
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    kind = _safetensors_kind(tensor_names, metadata)
    family = _safetensors_family(tensor_names, metadata)
    return InspectedComponent(
        path=path.as_posix(),
        kind=kind,
        target_folder=_target_folder_for_kind(kind),
        family=family,
        metadata={
            "tensor_count": len(tensor_names),
            "metadata_keys": sorted(str(key)[:200] for key in metadata)[:128],
        },
    )


def _bounded_json_object_with_limit(content: bytes, limit: int) -> dict[str, Any]:
    if len(content) > limit:
        raise ModelManifestError("weight metadata exceeds the size limit")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ModelManifestError("weight metadata is malformed") from exc
    if not isinstance(value, dict) or _count_nodes(value) > MAX_METADATA_NODES:
        raise ModelManifestError("weight metadata is too complex")
    return value


def _safetensors_kind(tensor_names: list[str], metadata: Mapping[str, Any]) -> str:
    lowered = [name.casefold() for name in tensor_names]
    metadata_text = " ".join(f"{key}={value}" for key, value in metadata.items()).casefold()
    if "lora" in metadata_text or any(
        name.startswith(("lora_", "lycoris_")) or ".lora_" in name for name in lowered
    ):
        return "lora"
    has_diffusion = any(
        name.startswith(("model.diffusion_model.", "diffusion_model.", "transformer."))
        or ".double_blocks." in name
        for name in lowered
    )
    has_conditioning = any(
        name.startswith(("conditioner.", "cond_stage_model.", "text_encoder.")) for name in lowered
    )
    has_vae = any(name.startswith(("first_stage_model.", "vae.")) for name in lowered)
    if has_diffusion and (has_conditioning or has_vae):
        return "checkpoint"
    if has_diffusion:
        return "diffusion_model"
    if has_vae:
        return "vae"
    if any("text_model." in name or name.startswith("text_encoder.") for name in lowered):
        return "text_encoder"
    if any("vision_model." in name for name in lowered):
        return "clip_vision"
    return "unknown_safetensors"


def _safetensors_family(
    tensor_names: list[str],
    metadata: Mapping[str, Any],
) -> str | None:
    lowered = [name.casefold() for name in tensor_names]
    metadata_text = " ".join(str(value) for value in metadata.values()).casefold()
    if "sdxl" in metadata_text or any("conditioner.embedders.1." in name for name in lowered):
        return "sdxl"
    if "flux" in metadata_text or any("double_blocks." in name for name in lowered):
        return "flux"
    if "stable-diffusion" in metadata_text or any(
        name.startswith("cond_stage_model.") for name in lowered
    ):
        return "stable-diffusion"
    return None


def _inspect_gguf(path: PurePosixPath, content: bytes) -> InspectedComponent:
    if len(content) > MAX_WEIGHT_HEADER_BYTES:
        content = content[:MAX_WEIGHT_HEADER_BYTES]
    cursor = _BinaryCursor(content)
    if cursor.read(4) != b"GGUF":
        raise ModelManifestError("GGUF header has an invalid magic value")
    version = cursor.scalar("<I")
    if version not in {2, 3}:
        raise ModelManifestError("GGUF version is unsupported")
    tensor_count = cursor.scalar("<Q")
    field_count = cursor.scalar("<Q")
    if field_count > MAX_GGUF_FIELDS:
        raise ModelManifestError("GGUF metadata has too many fields")
    fields: dict[str, Any] = {}
    for _ in range(field_count):
        key = cursor.string()
        value_type = cursor.scalar("<I")
        value = _read_gguf_value(cursor, value_type)
        if key in {
            "general.architecture",
            "general.name",
            "general.type",
            "clip.projector_type",
        }:
            fields[key] = value
    architecture = _printable_metadata(fields.get("general.architecture"))
    projector = "mmproj" in path.name.casefold() or "clip.projector_type" in fields
    return InspectedComponent(
        path=path.as_posix(),
        kind="projector" if projector else "gguf_model",
        target_folder="projectors" if projector else "models",
        architecture=architecture,
        family=architecture,
        metadata={
            "gguf_version": version,
            "tensor_count": tensor_count,
            "general_type": _printable_metadata(fields.get("general.type")),
        },
    )


class _BinaryCursor:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.position = 0

    def read(self, length: int) -> bytes:
        if length < 0 or self.position + length > len(self.content):
            raise ModelManifestError("weight header is truncated")
        result = self.content[self.position : self.position + length]
        self.position += length
        return result

    def scalar(self, format_string: str) -> Any:
        size = struct.calcsize(format_string)
        return struct.unpack(format_string, self.read(size))[0]

    def string(self) -> str:
        length = self.scalar("<Q")
        if length > MAX_METADATA_BYTES:
            raise ModelManifestError("weight metadata string exceeds the size limit")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModelManifestError("weight metadata string is invalid") from exc


def _read_gguf_value(cursor: _BinaryCursor, value_type: int) -> Any:
    if value_type in _GGUF_SCALAR_FORMATS:
        return cursor.scalar(_GGUF_SCALAR_FORMATS[value_type])
    if value_type == 8:
        return cursor.string()
    if value_type == 9:
        item_type = cursor.scalar("<I")
        item_count = cursor.scalar("<Q")
        if item_count > MAX_GGUF_ARRAY_ITEMS:
            raise ModelManifestError("GGUF metadata array exceeds the item limit")
        values = [_read_gguf_value(cursor, item_type) for _ in range(item_count)]
        return values[:128]
    raise ModelManifestError("GGUF metadata uses an unknown value type")


def _repository_identity(
    metadata: Mapping[str, Mapping[str, Any]],
    components: list[InspectedComponent],
    role: str,
) -> tuple[str | None, str | None]:
    candidates: list[str] = []
    for document in metadata.values():
        for key in ("model_type", "_class_name", "architecture"):
            value = document.get(key)
            if isinstance(value, str):
                candidates.append(value)
        architectures = document.get("architectures")
        if isinstance(architectures, list):
            candidates.extend(str(value) for value in architectures if isinstance(value, str))
    candidates.extend(
        value
        for component in components
        for value in (component.architecture, component.family)
        if value
    )
    architecture = candidates[0][:200] if candidates else None
    normalized = " ".join(candidates).casefold()
    family = _family_from_architecture(normalized, role)
    if not family:
        family = next((component.family for component in components if component.family), None)
    return architecture, family


def _family_from_architecture(value: str, role: str) -> str | None:
    for contract in architecture_family_contracts():
        if role in contract.roles and any(
            marker in value for marker in contract.architecture_markers
        ):
            return contract.id
    return "gguf" if role == "chat" and "gguf" in value else None


def _component_kind_from_path(path: str, role: str) -> str:
    name = PurePosixPath(path).name.casefold()
    if name.endswith(".gguf"):
        return "projector" if "mmproj" in name else "gguf_model"
    if name.endswith(".safetensors"):
        return "checkpoint" if role in {"image", "video"} else "weights"
    return "metadata"


def _target_folder(path: str, role: str) -> str:
    parts = PurePosixPath(path).parts
    known = {
        "checkpoints",
        "diffusion_models",
        "text_encoders",
        "vae",
        "clip_vision",
        "loras",
        "controlnet",
        "upscale_models",
        "embeddings",
    }
    folder = next((part for part in parts[:-1] if part in known), None)
    if folder:
        return folder
    return "models" if role == "chat" else "checkpoints"


def _target_folder_for_kind(kind: str) -> str:
    return {
        "checkpoint": "checkpoints",
        "diffusion_model": "diffusion_models",
        "text_encoder": "text_encoders",
        "vae": "vae",
        "clip_vision": "clip_vision",
        "lora": "loras",
    }.get(kind, "checkpoints")


def _printable_metadata(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 200:
        return None
    if any(character < " " for character in value):
        return None
    return value
