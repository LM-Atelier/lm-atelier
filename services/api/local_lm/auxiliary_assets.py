from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    WorkflowDependencyBinding,
    WorkflowRevision,
)

LORA_GRAPH_TRANSFORM_VERSION = "lora-graph-v2"
LORA_AUTO_SELECTION_VERSION = "lora-use-case-v1"
MAX_LORA_STACK_SIZE = 8
MAX_LORA_STRENGTH = 4.0
AUXILIARY_ASSET_KINDS: frozenset[str] = frozenset(
    {
        "lora",
        "vae",
        "controlnet",
        "upscaler",
        "embedding",
        "ip_adapter",
    }
)
_MODEL_SAMPLER_CLASS_TYPES = {
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
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
    *,
    workflow_activation_id: str | None = None,
) -> AutomaticLoraSelection:
    """Select a small deterministic LoRA stack from user-authored use cases."""

    if not workflow_lora_extension(revision):
        return AutomaticLoraSelection([], _automatic_selection_provenance([]))
    prompt_text = _normalized_match_text(prompt)
    prompt_terms = set(_meaningful_terms(prompt_text))
    if not prompt_terms:
        return AutomaticLoraSelection([], _automatic_selection_provenance([]))
    base_families = _workflow_families(
        session,
        revision,
        workflow_activation_id=workflow_activation_id,
    )
    if not base_families:
        # Nothing is known about what architecture this workflow runs, and an
        # adapter trained for another one does not refuse to load - it quietly
        # degrades the image while provenance reports that it applied. The
        # compatibility test below can only skip an asset when the families are
        # known, so an unknown family used to mean no test at all: every
        # auto-apply LoRA was eligible for every workflow. Choosing nothing is
        # the only safe reading of "we cannot tell".
        #
        # This is why a workflow declaring where LoRAs go is not, on its own,
        # enough to receive them automatically. It also has to say what it runs.
        return AutomaticLoraSelection(
            [],
            _automatic_selection_provenance(
                [],
                skipped_reason="workflow_architecture_unknown",
            ),
        )
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
        if not family or family not in base_families:
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


def _automatic_selection_provenance(
    selected: list[dict[str, Any]],
    *,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "mode": "automatic",
        "selector_version": LORA_AUTO_SELECTION_VERSION,
        "selected": selected,
    }
    if skipped_reason:
        provenance["skipped_reason"] = skipped_reason
    return provenance


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


def model_only_lora_extension(graph: dict[str, Any]) -> dict[str, Any] | None:
    """Return the one model link shared by all supported core sampler nodes."""

    links = _sampler_model_links(graph)
    distinct = {tuple(link) for link in links}
    if len(distinct) != 1:
        return None
    model = list(next(iter(distinct)))
    if model[0] not in graph:
        return None
    return {
        "mode": "model_only",
        "model": model,
    }


def detect_lora_extension(graph: dict[str, Any]) -> dict[str, Any] | None:
    """Detect a backward-compatible checkpoint or split-model insertion point."""

    return checkpoint_lora_extension(graph) or model_only_lora_extension(graph)


def workflow_lora_extension(revision: WorkflowRevision) -> dict[str, Any] | None:
    extensions = revision.dependencies_json.get("extensions")
    raw = extensions.get("lora") if isinstance(extensions, dict) else None
    if not isinstance(raw, dict):
        return None
    model = raw.get("model")
    mode = raw.get("mode")
    if mode == "model_only":
        if set(raw) != {"mode", "model"} or not _valid_link(model):
            return None
        assert isinstance(model, list)
        return {"mode": "model_only", "model": list(model)}
    if mode is not None or set(raw) != {"model", "clip"}:
        return None
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
    mode = raw_extension.get("mode")
    model = raw_extension.get("model")
    clip = raw_extension.get("clip")
    if mode == "model_only":
        if set(raw_extension) != {"mode", "model"} or not _valid_link(model):
            raise ValueError("The model-only LoRA workflow extension is invalid.")
        assert isinstance(model, list)
        if model[0] not in graph:
            raise ValueError("The LoRA workflow extension references a missing graph node.")
        sampler_links = _sampler_model_links(graph)
        if not sampler_links or {tuple(link) for link in sampler_links} != {tuple(model)}:
            raise ValueError(
                "The model-only LoRA workflow extension must feed every supported sampler."
            )
    else:
        if mode is not None or set(raw_extension) != {"model", "clip"}:
            raise ValueError("The LoRA workflow extension has an unsupported shape.")
        if not _valid_link(model) or not _valid_link(clip):
            raise ValueError("The LoRA workflow extension has invalid model or CLIP links.")
        assert isinstance(model, list)
        assert isinstance(clip, list)
        if model[0] not in graph or clip[0] not in graph:
            raise ValueError("The LoRA workflow extension references a missing graph node.")
    if any(str(node_id).startswith("lma_lora_") for node_id in graph):
        raise ValueError("The workflow uses a reserved LM Atelier LoRA node identifier.")
    if mode == "model_only":
        if not _graph_contains_link(graph, model):
            raise ValueError("The model-only LoRA workflow extension must feed a graph input.")
    else:
        assert isinstance(clip, list)
        if not _graph_contains_link(graph, model) or not _graph_contains_link(graph, clip):
            raise ValueError(
                "The LoRA workflow extension must feed both model and CLIP graph inputs."
            )


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


def trigger_words_to_apply(provenance: list[dict[str, Any]], prompt: str) -> list[str]:
    """Trigger words the prompt still needs, from enabled stack items.

    A word the prompt already carries, in any casing, is not repeated - the
    model hearing it twice helps nothing and the transcript reads doubled.
    Order follows the stack so provenance and prompt agree.
    """

    lowered = prompt.casefold()
    applied: list[str] = []
    seen: set[str] = set()
    for item in provenance:
        if not item.get("enabled"):
            continue
        words = item.get("trigger_words")
        if not isinstance(words, list):
            continue
        for word in words:
            if not isinstance(word, str):
                continue
            cleaned = word.strip()
            key = cleaned.casefold()
            if not cleaned or key in lowered or key in seen:
                continue
            seen.add(key)
            applied.append(cleaned)
    return applied


def prompt_trigger_word_provenance(
    model: dict[str, Any] | None,
    lora_provenance: list[dict[str, Any]],
    prompt: str,
) -> dict[str, list[str]]:
    """Freeze model and LoRA trigger words into one transparent prompt snapshot."""

    manifest = model.get("manifest") if isinstance(model, dict) else None
    source = model.get("source") if isinstance(model, dict) else None
    source_metadata = source.get("metadata") if isinstance(source, dict) else None
    declared: list[str] = []
    for container, key in (
        (manifest, "trigger_words"),
        (manifest, "trained_words"),
        (source_metadata, "trained_words"),
        (source_metadata, "trigger_words"),
    ):
        values = container.get(key) if isinstance(container, dict) else None
        if isinstance(values, list):
            declared.extend(value[:200] for value in values[:100] if isinstance(value, str))
        if len(declared) >= 100:
            declared = declared[:100]
            break

    model_applied = trigger_words_to_apply(
        [{"enabled": True, "trigger_words": declared}],
        prompt,
    )
    prompt_with_model_words = f"{prompt}, {', '.join(model_applied)}" if model_applied else prompt
    lora_applied = trigger_words_to_apply(lora_provenance, prompt_with_model_words)
    return {
        "model_trigger_words_applied": model_applied,
        "lora_trigger_words_applied": lora_applied,
        "trigger_words_applied": [*model_applied, *lora_applied],
    }


def transform_lora_graph(
    graph: dict[str, Any],
    extension: dict[str, Any],
    stack: list[dict[str, Any]],
) -> dict[str, Any]:
    transformed = copy.deepcopy(graph)
    if not stack:
        return transformed
    model_source = list(extension["model"])
    model_only = extension.get("mode") == "model_only"
    clip_source = None if model_only else list(extension["clip"])
    inserted_ids: list[str] = []
    for index, item in enumerate(stack, start=1):
        node_id = f"lma_lora_{index:03d}"
        if node_id in transformed:
            raise ValueError("The workflow reserves an LM Atelier LoRA node identifier.")
        loader_inputs = {
            "model": model_source,
            "lora_name": item["comfy_name"],
            "strength_model": item["model_strength"],
        }
        if not model_only:
            loader_inputs.update(
                {
                    "clip": clip_source,
                    "strength_clip": item["clip_strength"],
                }
            )
        transformed[node_id] = {
            "class_type": "LoraLoaderModelOnly" if model_only else "LoraLoader",
            "_meta": {"title": f"LM Atelier LoRA {index}"},
            "inputs": loader_inputs,
        }
        inserted_ids.append(node_id)
        model_source = [node_id, 0]
        if not model_only:
            clip_source = [node_id, 1]

    original_model = extension["model"]
    original_clip = None if model_only else extension["clip"]
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
    original_clip: list[Any] | None,
    clip_source: list[Any] | None,
) -> Any:
    if value == original_model:
        return list(model_source)
    if original_clip is not None and value == original_clip:
        assert clip_source is not None
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


def _workflow_families(
    session: Session,
    revision: WorkflowRevision,
    *,
    workflow_activation_id: str | None = None,
) -> set[str]:
    """Return one verified architecture family or fail closed.

    An activation is the authoritative local binding for a portable workflow.
    Legacy install IDs are only a fallback for revisions without a typed
    activation. Every declared model binding must resolve and agree: an empty,
    partial, or mixed-family answer cannot safely authorize an automatic LoRA.
    """

    install_ids: list[str] = []
    if workflow_activation_id:
        bindings = session.scalars(
            select(WorkflowDependencyBinding).where(
                WorkflowDependencyBinding.workflow_activation_id == workflow_activation_id,
                WorkflowDependencyBinding.workflow_revision_id == revision.id,
            )
        ).all()
        for binding in bindings:
            if binding.model_install_id:
                install_ids.append(binding.model_install_id)
            elif binding.model_profile_id:
                profile = session.get(ModelProfile, binding.model_profile_id)
                if profile is None or not profile.model_install_id:
                    return set()
                install_ids.append(profile.model_install_id)
        if not install_ids:
            return set()
    else:
        raw_ids = revision.dependencies_json.get("model_install_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            return set()
        if any(not isinstance(item, str) or not item for item in raw_ids):
            return set()
        install_ids = list(raw_ids)

    families: set[str] = set()
    for install_id in install_ids:
        install = session.get(ModelInstall, install_id)
        family = install.manifest_json.get("family") if install else None
        if not isinstance(family, str) or not family.strip():
            return set()
        families.add(family.strip().casefold())
    return families if len(families) == 1 else set()


def _valid_link(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and value[1] >= 0
    )


def _sampler_model_links(graph: dict[str, Any]) -> list[list[Any]]:
    links: list[list[Any]] = []
    for node in graph.values():
        if not isinstance(node, dict) or node.get("class_type") not in _MODEL_SAMPLER_CLASS_TYPES:
            continue
        inputs = node.get("inputs")
        model = inputs.get("model") if isinstance(inputs, dict) else None
        if not _valid_link(model):
            return []
        assert isinstance(model, list)
        links.append(list(model))
    return links


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
