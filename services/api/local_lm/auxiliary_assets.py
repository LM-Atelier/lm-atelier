from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ModelAssetInstall, ModelInstall, WorkflowRevision

LORA_GRAPH_TRANSFORM_VERSION = "lora-graph-v1"
LORA_AUTO_SELECTION_VERSION = "lora-use-case-v1"
MAX_LORA_STACK_SIZE = 8
MAX_LORA_STRENGTH = 4.0
SUPPORTED_AUXILIARY_KINDS = {
    "lora",
    "vae",
    "controlnet",
    "upscaler",
    "embedding",
    "ip_adapter",
}
COMFY_AUXILIARY_FOLDERS = {
    "lora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscaler": "upscale_models",
    "embedding": "embeddings",
    "ip_adapter": "ipadapter",
}


@dataclass(frozen=True)
class ResolvedLoraStack:
    settings: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    graph_sha256: str


@dataclass(frozen=True)
class AutomaticLoraSelection:
    settings: list[dict[str, Any]]
    provenance: dict[str, Any]


_LORA_MATCH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "apply",
    "art",
    "for",
    "image",
    "images",
    "in",
    "look",
    "make",
    "of",
    "on",
    "or",
    "photo",
    "picture",
    "style",
    "the",
    "to",
    "use",
    "with",
}


def select_automatic_lora_stack(
    session: Session,
    revision: WorkflowRevision,
    prompt: str,
) -> AutomaticLoraSelection:
    """Select a small deterministic LoRA stack from user-authored use cases."""

    if not workflow_lora_extension(revision):
        return AutomaticLoraSelection([], _automatic_selection_provenance([]))
    prompt_text = _normalized_match_text(prompt)
    prompt_terms = set(_meaningful_terms(prompt_text))
    if not prompt_terms:
        return AutomaticLoraSelection([], _automatic_selection_provenance([]))
    base_families = _workflow_families(session, revision)
    assets = session.scalars(
        select(ModelAssetInstall).where(
            ModelAssetInstall.kind == "lora",
            ModelAssetInstall.active.is_(True),
            ModelAssetInstall.auto_apply.is_(True),
            ModelAssetInstall.verified_at.is_not(None),
        )
    ).all()
    ranked: list[tuple[int, float, int, str, str, ModelAssetInstall, list[str], str]] = []
    for asset in assets:
        family = asset.family.casefold() if asset.family else None
        if base_families and (not family or family not in base_families):
            continue
        comfy_name = asset.manifest_json.get("comfy_name")
        sha256 = asset.manifest_json.get("sha256")
        if (
            not isinstance(comfy_name, str)
            or not comfy_name
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            continue
        use_case_text = _normalized_match_text(asset.use_case)
        use_case_terms = set(_meaningful_terms(use_case_text))
        if not use_case_terms:
            continue
        matched_terms = sorted(use_case_terms & prompt_terms)
        exact = bool(use_case_text and f" {use_case_text} " in f" {prompt_text} ")
        coverage = len(matched_terms) / len(use_case_terms)
        if not exact and (
            (len(use_case_terms) == 1 and not matched_terms)
            or (len(use_case_terms) > 1 and (len(matched_terms) < 2 or coverage < 0.5))
        ):
            continue
        match_type = "exact use case" if exact else "shared use-case terms"
        ranked.append(
            (
                -int(exact),
                -coverage,
                -len(matched_terms),
                asset.name.casefold(),
                asset.id,
                asset,
                matched_terms,
                match_type,
            )
        )
    ranked.sort(key=lambda item: item[:5])
    selected: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for _, _, _, _, _, asset, matched_terms, match_type in ranked[:MAX_LORA_STACK_SIZE]:
        setting = {
            "asset_id": asset.id,
            "model_strength": asset.default_model_strength,
            "clip_strength": asset.default_clip_strength,
            "enabled": True,
        }
        selected.append(setting)
        provenance.append(
            {
                **setting,
                "name": asset.name,
                "use_case": asset.use_case,
                "matched_terms": matched_terms,
                "reason": match_type,
            }
        )
    return AutomaticLoraSelection(selected, _automatic_selection_provenance(provenance))


def _automatic_selection_provenance(selected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": "automatic",
        "selector_version": LORA_AUTO_SELECTION_VERSION,
        "selected": selected,
    }


def _normalized_match_text(value: str) -> str:
    return " ".join(
        "".join(character.casefold() if character.isalnum() else " " for character in value).split()
    )


def _meaningful_terms(value: str) -> list[str]:
    return [term for term in value.split() if len(term) >= 3 and term not in _LORA_MATCH_STOP_WORDS]


def checkpoint_lora_extension(graph: dict[str, Any]) -> dict[str, list[Any]] | None:
    """Return the one unambiguous core-checkpoint LoRA insertion point."""

    matches = [
        str(node_id)
        for node_id, node in graph.items()
        if isinstance(node, dict) and node.get("class_type") == "CheckpointLoaderSimple"
    ]
    if len(matches) != 1:
        return None
    model = [matches[0], 0]
    clip = [matches[0], 1]
    if not _graph_contains_link(graph, model) or not _graph_contains_link(graph, clip):
        return None
    return {
        "model": model,
        "clip": clip,
    }


def workflow_lora_extension(revision: WorkflowRevision) -> dict[str, list[Any]] | None:
    extensions = revision.dependencies_json.get("extensions")
    raw = extensions.get("lora") if isinstance(extensions, dict) else None
    if not isinstance(raw, dict):
        return None
    model = raw.get("model")
    clip = raw.get("clip")
    if not _valid_link(model) or not _valid_link(clip):
        return None
    assert isinstance(model, list)
    assert isinstance(clip, list)
    return {"model": list(model), "clip": list(clip)}


def validate_lora_workflow_contract(
    graph: dict[str, Any],
    input_schema: dict[str, Any],
    dependencies: dict[str, Any],
) -> None:
    properties = input_schema.get("properties")
    lora_schema = properties.get("loras") if isinstance(properties, dict) else None
    extensions = dependencies.get("extensions")
    raw_extension = extensions.get("lora") if isinstance(extensions, dict) else None
    if lora_schema is None and raw_extension is None:
        return
    if not isinstance(lora_schema, dict) or not isinstance(raw_extension, dict):
        raise ValueError(
            "A workflow must declare its LoRA setting and graph extension point together."
        )
    max_items = lora_schema.get("maxItems")
    if (
        lora_schema.get("type") != "array"
        or not isinstance(max_items, int)
        or isinstance(max_items, bool)
        or max_items < 1
        or max_items > MAX_LORA_STACK_SIZE
    ):
        raise ValueError(
            f"The LoRA workflow input must be an array capped at {MAX_LORA_STACK_SIZE} items."
        )
    model = raw_extension.get("model")
    clip = raw_extension.get("clip")
    if not _valid_link(model) or not _valid_link(clip):
        raise ValueError("The LoRA workflow extension has invalid model or CLIP links.")
    assert isinstance(model, list)
    assert isinstance(clip, list)
    if model[0] not in graph or clip[0] not in graph:
        raise ValueError("The LoRA workflow extension references a missing graph node.")
    if any(str(node_id).startswith("lma_lora_") for node_id in graph):
        raise ValueError("The workflow uses a reserved LM Atelier LoRA node identifier.")
    if not _graph_contains_link(graph, model) or not _graph_contains_link(graph, clip):
        raise ValueError("The LoRA workflow extension must feed both model and CLIP graph inputs.")


def resolve_lora_stack(
    session: Session,
    revision: WorkflowRevision,
    value: object,
) -> ResolvedLoraStack:
    if not isinstance(value, list):
        raise ValueError("LoRA stack must be a list.")
    if len(value) > MAX_LORA_STACK_SIZE:
        raise ValueError(f"A LoRA stack can contain at most {MAX_LORA_STACK_SIZE} assets.")
    if not value:
        graph_hash = _graph_hash(revision.api_graph_json)
        return ResolvedLoraStack([], [], graph_hash)
    validate_lora_workflow_contract(
        revision.api_graph_json,
        revision.input_schema_json,
        revision.dependencies_json,
    )
    extension = workflow_lora_extension(revision)
    if not extension:
        raise ValueError("The selected workflow does not provide a LoRA extension point.")

    base_families = _workflow_families(session, revision)
    seen: set[str] = set()
    settings: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    transform_items: list[dict[str, Any]] = []
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict) or not isinstance(raw.get("asset_id"), str):
            raise ValueError(
                "A saved LoRA selection is unavailable. Choose an installed LoRA again."
            )
        allowed = {"asset_id", "model_strength", "clip_strength", "enabled"}
        if set(raw) - allowed:
            raise ValueError(f"LoRA item {index} contains unsupported fields.")
        asset_id = raw["asset_id"]
        if asset_id in seen:
            raise ValueError("A LoRA stack cannot contain the same asset twice.")
        seen.add(asset_id)
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"LoRA item {index} has an invalid enabled value.")
        model_strength = _strength(raw.get("model_strength", 1.0), index, "model")
        clip_strength = _strength(raw.get("clip_strength", 1.0), index, "CLIP")
        asset = session.get(ModelAssetInstall, asset_id)
        if not asset or asset.kind != "lora" or not asset.active or not asset.verified_at:
            raise ValueError(
                "A selected LoRA is unavailable or no longer verified. Choose an installed LoRA."
            )
        if asset.family and base_families and asset.family.casefold() not in base_families:
            raise ValueError(
                f"{asset.name} targets {asset.family}, which is incompatible with this workflow."
            )
        comfy_name = asset.manifest_json.get("comfy_name")
        sha256 = asset.manifest_json.get("sha256")
        if (
            not isinstance(comfy_name, str)
            or not comfy_name
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ValueError(f"{asset.name} has incomplete verification metadata.")
        normalized = {
            "asset_id": asset.id,
            "model_strength": model_strength,
            "clip_strength": clip_strength,
            "enabled": enabled,
        }
        metadata = asset.manifest_json.get("metadata")
        trigger_words = (
            [
                word[:200]
                for word in metadata.get("trigger_words", [])[:100]
                if isinstance(word, str) and word
            ]
            if isinstance(metadata, dict) and isinstance(metadata.get("trigger_words"), list)
            else []
        )
        settings.append(normalized)
        provenance.append(
            {
                **normalized,
                "position": index - 1,
                "name": asset.name,
                "family": asset.family,
                "sha256": sha256,
                "comfy_name": comfy_name,
                "trigger_words": trigger_words,
            }
        )
        if enabled:
            transform_items.append(
                {
                    "comfy_name": comfy_name,
                    "model_strength": model_strength,
                    "clip_strength": clip_strength,
                }
            )
    transformed = transform_lora_graph(revision.api_graph_json, extension, transform_items)
    return ResolvedLoraStack(settings, provenance, _graph_hash(transformed))


def transform_lora_graph(
    graph: dict[str, Any],
    extension: dict[str, list[Any]],
    stack: list[dict[str, Any]],
) -> dict[str, Any]:
    transformed = copy.deepcopy(graph)
    if not stack:
        return transformed
    model_source = list(extension["model"])
    clip_source = list(extension["clip"])
    inserted_ids: list[str] = []
    for index, item in enumerate(stack, start=1):
        node_id = f"lma_lora_{index:03d}"
        if node_id in transformed:
            raise ValueError("The workflow reserves an LM Atelier LoRA node identifier.")
        transformed[node_id] = {
            "class_type": "LoraLoader",
            "_meta": {"title": f"LM Atelier LoRA {index}"},
            "inputs": {
                "model": model_source,
                "clip": clip_source,
                "lora_name": item["comfy_name"],
                "strength_model": item["model_strength"],
                "strength_clip": item["clip_strength"],
            },
        }
        inserted_ids.append(node_id)
        model_source = [node_id, 0]
        clip_source = [node_id, 1]

    original_model = extension["model"]
    original_clip = extension["clip"]
    for node_id, node in transformed.items():
        if node_id in inserted_ids or not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            node["inputs"] = _replace_links(
                inputs,
                original_model,
                model_source,
                original_clip,
                clip_source,
            )
    return transformed


def _replace_links(
    value: Any,
    original_model: list[Any],
    model_source: list[Any],
    original_clip: list[Any],
    clip_source: list[Any],
) -> Any:
    if value == original_model:
        return list(model_source)
    if value == original_clip:
        return list(clip_source)
    if isinstance(value, dict):
        return {
            key: _replace_links(item, original_model, model_source, original_clip, clip_source)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_links(item, original_model, model_source, original_clip, clip_source)
            for item in value
        ]
    return value


def _strength(value: object, index: int, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"LoRA item {index} has an invalid {label} strength.")
    result = float(value)
    if not math.isfinite(result) or abs(result) > MAX_LORA_STRENGTH:
        raise ValueError(
            f"LoRA item {index} {label} strength must be between "
            f"-{MAX_LORA_STRENGTH:g} and {MAX_LORA_STRENGTH:g}."
        )
    return result


def _workflow_families(session: Session, revision: WorkflowRevision) -> set[str]:
    raw_ids = revision.dependencies_json.get("model_install_ids")
    install_ids = (
        [item for item in raw_ids if isinstance(item, str)] if isinstance(raw_ids, list) else []
    )
    families: set[str] = set()
    for install_id in install_ids:
        install = session.get(ModelInstall, install_id)
        family = install.manifest_json.get("family") if install else None
        if isinstance(family, str) and family:
            families.add(family.casefold())
    return families


def _valid_link(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and value[1] >= 0
    )


def _graph_contains_link(value: object, link: list[Any]) -> bool:
    if value == link:
        return True
    if isinstance(value, dict):
        return any(_graph_contains_link(child, link) for child in value.values())
    if isinstance(value, list):
        return any(_graph_contains_link(child, link) for child in value)
    return False


def _graph_hash(graph: dict[str, Any]) -> str:
    encoded = json.dumps(
        graph,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
