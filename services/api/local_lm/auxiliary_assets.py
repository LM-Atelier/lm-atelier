from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from .comfy_workflow_packages import (
    MAX_UI_GRAPH_DEPTH,
    MAX_UI_GRAPH_VALUES,
    WorkflowPackageError,
    validate_bounded_workflow_json,
)
from .lora_constraints import MAX_LORA_STRENGTH
from .models import (
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
    WorkflowActivation,
    WorkflowDependencyBinding,
    WorkflowRevision,
)

LORA_GRAPH_TRANSFORM_VERSION = "lora-graph-v2"
LORA_AUTO_SELECTION_VERSION = "lora-use-case-v1"
MAX_LORA_STACK_SIZE = 8
MAX_LORA_TRIGGER_WORDS = 100
MAX_LORA_TRIGGER_WORD_LENGTH = 200
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


@dataclass(frozen=True, slots=True)
class ResolvedLoraGraph:
    """One Added-LoRA resolution and its fresh detached API graph.

    The dataclass contract is immutable, while the graph field is deliberately
    a mutable, caller-owned JSON payload. Every resolution receives a fresh
    graph so mutating it cannot affect the workflow revision, the supplied
    base graph, or another resolution.
    """

    settings: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    graph: dict[str, Any]
    graph_sha256: str


@dataclass(frozen=True)
class AutomaticLoraSelection:
    settings: list[dict[str, Any]]
    provenance: dict[str, Any]


def _invalid_lora_trigger_words() -> ValueError:
    return ValueError("A selected LoRA has invalid trigger-word metadata.")


def _installed_lora_trigger_words(metadata: object) -> list[str]:
    """Return one bounded trigger-word vocabulary from stored LoRA metadata."""

    if metadata is None:
        return []
    if not isinstance(metadata, dict):
        raise _invalid_lora_trigger_words()

    words: list[str] = []
    seen: set[str] = set()
    for key in ("trigger_words", "trained_words"):
        if key not in metadata:
            continue
        declared = metadata[key]
        if not isinstance(declared, list) or len(declared) > MAX_LORA_TRIGGER_WORDS:
            raise _invalid_lora_trigger_words()
        for value in declared:
            if not isinstance(value, str):
                raise _invalid_lora_trigger_words()
            word = value.strip()
            if not word or len(word) > MAX_LORA_TRIGGER_WORD_LENGTH:
                raise _invalid_lora_trigger_words()
            folded = word.casefold()
            if folded in seen:
                continue
            if len(words) == MAX_LORA_TRIGGER_WORDS:
                raise _invalid_lora_trigger_words()
            seen.add(folded)
            words.append(word)
    return words


def lora_trigger_words(asset: ModelAssetInstall) -> list[str]:
    """Return every trigger word a LoRA carries: the file's own, then the typed ones.

    One list, because a run applies a word the same way whichever place it came
    from. The two stay apart where they are stored: the manifest holds only
    what the file declared, and what a person typed lives beside it.
    """

    words = _installed_lora_trigger_words(asset.manifest_json.get("metadata"))
    typed = asset.typed_trigger_words
    if typed is None:
        return words
    if not isinstance(typed, list):
        raise _invalid_lora_trigger_words()
    seen = {word.casefold() for word in words}
    for value in typed:
        if not isinstance(value, str):
            raise _invalid_lora_trigger_words()
        word = value.strip()
        if not word or len(word) > MAX_LORA_TRIGGER_WORD_LENGTH:
            raise _invalid_lora_trigger_words()
        folded = word.casefold()
        if folded in seen:
            continue
        if len(words) == MAX_LORA_TRIGGER_WORDS:
            raise _invalid_lora_trigger_words()
        seen.add(folded)
        words.append(word)
    return words


def normalize_typed_trigger_words(values: list[str], asset: ModelAssetInstall) -> list[str]:
    """Return the trigger words a person submitted for a LoRA, cleaned and bounded.

    Blank entries are dropped and repeats collapse in any casing, keeping the
    first. The bound counts the words the file declares as well, since both are
    applied together. Errors never repeat a submitted word.
    """

    typed: list[str] = []
    seen: set[str] = set()
    for value in values:
        word = value.strip()
        if not word:
            continue
        if len(word) > MAX_LORA_TRIGGER_WORD_LENGTH:
            raise ValueError(
                f"A trigger word can be at most {MAX_LORA_TRIGGER_WORD_LENGTH} characters."
            )
        folded = word.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        typed.append(word)
    measured = _installed_lora_trigger_words(asset.manifest_json.get("metadata"))
    combined = {word.casefold() for word in measured} | seen
    if len(combined) > MAX_LORA_TRIGGER_WORDS:
        raise ValueError(
            f"A LoRA can have at most {MAX_LORA_TRIGGER_WORDS} trigger words, "
            "counting the ones its file declares."
        )
    return typed


@dataclass(frozen=True, slots=True)
class _NormalizedLoraStack:
    settings: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    transform_items: list[dict[str, Any]]


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
    base_keys = {_family_key(family) for family in base_families}
    ranked: list[tuple[int, float, int, str, str, ModelAssetInstall, list[str], str]] = []
    for asset in assets:
        family = _family_key(asset.family) if asset.family else ""
        if not family or family not in base_keys:
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


def derived_lora_extension(graph: dict[str, Any]) -> dict[str, Any] | None:
    """The insertion point a graph offers, when it offers exactly one.

    A revision records where LoRAs go at the moment it is compiled here. A
    revision that arrived some other way - imported, or carried across from an
    older installation - can be missing that record while its graph plainly
    offers the same single point, and nothing can add it afterwards: the record
    is written when a revision is built, and no route rewrites a built one. So
    a workflow somebody runs every day could refuse every LoRA permanently over
    a line of missing metadata rather than anything about its graph.

    Reading the graph gives the same answer through the same functions that
    write that record in the first place, under the same conditions: one
    unambiguous point, feeding every sampler the graph samples with, on a graph
    that has not already reserved the identifiers an insertion uses.
    """

    extension = detect_lora_extension(graph)
    if extension is None:
        return None
    if any(str(node_id).startswith("lma_lora_") for node_id in graph):
        return None
    links = [extension["model"]]
    if extension.get("mode") != "model_only":
        links.append(extension["clip"])
    if not all(_graph_contains_link(graph, link) for link in links):
        return None
    return extension


def workflow_lora_extension(revision: WorkflowRevision) -> dict[str, Any] | None:
    dependencies = revision.dependencies_json
    extensions = dependencies.get("extensions") if isinstance(dependencies, dict) else None
    if extensions is not None and not isinstance(extensions, dict):
        return None
    raw = extensions.get("lora") if isinstance(extensions, dict) else None
    if raw is not None:
        return _declared_lora_extension(raw)
    graph = revision.api_graph_json
    return derived_lora_extension(graph) if isinstance(graph, dict) else None


def _declared_lora_extension(raw: object) -> dict[str, Any] | None:
    """Read a recorded insertion point exactly, or refuse it.

    A revision that says where its LoRAs go and says it wrongly is not a
    revision that says nothing. Measuring its graph instead would paper over
    the damage rather than report it, so a malformed record still refuses.
    """

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
    stack = _lora_stack_items(value)
    if not stack:
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

    normalized = _normalize_lora_stack(session, revision, stack)
    transformed = transform_lora_graph(
        revision.api_graph_json,
        extension,
        normalized.transform_items,
    )
    return ResolvedLoraStack(
        normalized.settings,
        normalized.provenance,
        _graph_hash(transformed),
    )


def resolve_lora_stack_against_graph(
    session: Session,
    revision: WorkflowRevision,
    value: object,
    *,
    base_api_graph: object,
    workflow_activation_id: str | None = None,
) -> ResolvedLoraGraph:
    """Apply the existing Added-LoRA transform to one caller-owned base graph.

    The supplied graph may already contain authorized workflow-native scalar
    edits. This function validates the revision's declared Added insertion
    boundary against that exact detached graph, then performs only the existing
    Added transform. It grants no authority to create or change the base graph.
    """

    base_graph = _detached_exact_api_graph(base_api_graph)
    stack = _lora_stack_items(value)
    validate_lora_workflow_contract(
        base_graph,
        revision.input_schema_json,
        revision.dependencies_json,
    )
    if not stack:
        return ResolvedLoraGraph([], [], base_graph, _graph_hash(base_graph))

    extension = workflow_lora_extension(revision)
    if not extension:
        raise ValueError("The selected workflow does not provide a LoRA extension point.")
    normalized = _normalize_lora_stack(
        session,
        revision,
        stack,
        workflow_activation_id=workflow_activation_id,
    )
    transformed = transform_lora_graph(
        base_graph,
        extension,
        normalized.transform_items,
    )
    return ResolvedLoraGraph(
        normalized.settings,
        normalized.provenance,
        transformed,
        _graph_hash(transformed),
    )


def _lora_stack_items(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("LoRA stack must be a list.")
    if len(value) > MAX_LORA_STACK_SIZE:
        raise ValueError(f"A LoRA stack can contain at most {MAX_LORA_STACK_SIZE} assets.")
    return value


def _normalize_lora_stack(
    session: Session,
    revision: WorkflowRevision,
    value: list[object],
    *,
    workflow_activation_id: str | None = None,
) -> _NormalizedLoraStack:
    base_families = _workflow_families(
        session,
        revision,
        workflow_activation_id=workflow_activation_id,
    )
    if workflow_activation_id is not None and not base_families:
        raise ValueError("The workflow activation does not identify exactly one model family.")
    base_keys = {_family_key(family) for family in base_families}
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
        if asset.family and base_keys and _family_key(asset.family) not in base_keys:
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
        trigger_words = lora_trigger_words(asset)
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
    return _NormalizedLoraStack(settings, provenance, transform_items)


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


def _detached_exact_api_graph(value: object) -> dict[str, Any]:
    _validate_exact_json(value)
    try:
        validate_bounded_workflow_json(value)
    except WorkflowPackageError as exc:
        raise ValueError("The supplied LoRA base graph is not bounded canonical JSON.") from exc
    if type(value) is not dict:
        raise ValueError("The supplied LoRA base graph must be an exact built-in JSON object.")
    exact_graph = cast(dict[str, Any], value)
    for node_id, node in exact_graph.items():
        if type(node_id) is not str or not node_id:
            raise ValueError("The supplied LoRA base graph has an invalid node identifier.")
        if node_id.startswith("lma_lora_"):
            raise ValueError("The workflow reserves an LM Atelier LoRA node identifier.")
        if (
            type(node) is not dict
            or type(node.get("class_type")) is not str
            or not node["class_type"]
            or type(node.get("inputs")) is not dict
        ):
            raise ValueError("The supplied LoRA base graph is not a valid ComfyUI API graph.")
    try:
        detached = copy.deepcopy(exact_graph)
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError("The supplied LoRA base graph could not be detached.") from exc
    return detached


def _validate_exact_json(value: object) -> None:
    stack = [(value, 1)]
    seen_containers: set[int] = set()
    values = 0
    while stack:
        current, depth = stack.pop()
        values += 1
        if values > MAX_UI_GRAPH_VALUES or depth > MAX_UI_GRAPH_DEPTH:
            raise ValueError("The supplied LoRA base graph is not bounded canonical JSON.")
        current_type = type(current)
        if current_type is dict:
            current_dict = cast(dict[object, object], current)
            identity = id(current)
            if identity in seen_containers:
                raise ValueError(
                    "The supplied LoRA base graph must not share or cycle JSON containers."
                )
            seen_containers.add(identity)
            for key, child in current_dict.items():
                if type(key) is not str:
                    raise ValueError(
                        "The supplied LoRA base graph must use exact built-in JSON keys."
                    )
                stack.append((child, depth + 1))
        elif current_type is list:
            current_list = cast(list[object], current)
            identity = id(current)
            if identity in seen_containers:
                raise ValueError(
                    "The supplied LoRA base graph must not share or cycle JSON containers."
                )
            seen_containers.add(identity)
            stack.extend((child, depth + 1) for child in current_list)
        elif current is not None and current_type not in {str, int, float, bool}:
            raise ValueError(
                "The supplied LoRA base graph must contain only exact built-in JSON values."
            )


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


def workflow_model_family(session: Session, revision: WorkflowRevision) -> str | None:
    """The one model family a workflow revision runs, or None when that is not known.

    A revision with a dependency contract answers only through its current ready
    activation, the same binding automatic LoRA selection reads; one without a
    contract falls back to its legacy install ids, or, declaring none, to the
    installed model its checkpoint loader names or, with no checkpoint loader,
    the registered diffusion model its model loader names. Anything partial,
    mixed or invalid is unknown rather than a guess.
    """

    activation_id: str | None = None
    if revision.dependency_contract_sha256 is not None:
        activation = session.scalar(
            select(WorkflowActivation).where(
                WorkflowActivation.workflow_revision_id == revision.id,
                WorkflowActivation.is_active.is_(True),
                WorkflowActivation.state == "ready",
            )
        )
        if activation is None:
            return None
        activation_id = activation.id
    try:
        families = _workflow_families(session, revision, workflow_activation_id=activation_id)
    except ValueError:
        return None
    return next(iter(families)) if len(families) == 1 else None


def _workflow_families(
    session: Session,
    revision: WorkflowRevision,
    *,
    workflow_activation_id: str | None = None,
) -> set[str]:
    """Return one verified architecture family or fail closed.

    An activation is the authoritative local binding for a portable workflow.
    Legacy install IDs are only a fallback for revisions without a typed
    activation, and a revision without a contract that declares none answers
    through the one installed model its checkpoint loader names or, with no
    checkpoint loader, the one registered diffusion model its model loader
    names. Every model binding must resolve and agree: an empty, partial, or
    mixed-family answer cannot safely authorize an automatic LoRA. Bindings
    agree when their families differ only in case and punctuation, and the
    answer is then one of their spellings.
    """

    install_ids: list[str] = []
    if workflow_activation_id is not None:
        if (
            type(workflow_activation_id) is not str
            or not workflow_activation_id.startswith("wfact_")
            or len(workflow_activation_id) > 40
            or len(workflow_activation_id) == len("wfact_")
            or any(
                not character.isascii() or (not character.isalnum() and character not in {"_", "-"})
                for character in workflow_activation_id
            )
        ):
            raise ValueError("Workflow activation identity is invalid.")
        activation = session.get(WorkflowActivation, workflow_activation_id)
        if activation is None:
            raise ValueError("Workflow activation is unavailable.")
        if activation.workflow_revision_id != revision.id:
            raise ValueError(
                "Workflow activation does not belong to the selected workflow revision."
            )
        bindings = session.scalars(
            select(WorkflowDependencyBinding).where(
                WorkflowDependencyBinding.workflow_activation_id == activation.id,
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
        if revision.dependency_contract_sha256 is None and (raw_ids is None or raw_ids == []):
            # A hand-built or imported workflow declares no model, but its
            # graph still names the checkpoint, or the diffusion model, it
            # loads. A declaration that is present and malformed stays unknown
            # rather than being read past.
            raw_ids = _checkpoint_install_ids(session, revision)
            if not raw_ids:
                family = _diffusion_model_family(session, revision)
                return {family} if family is not None else set()
        if not isinstance(raw_ids, list) or not raw_ids:
            return set()
        if any(not isinstance(item, str) or not item for item in raw_ids):
            return set()
        install_ids = list(raw_ids)

    families: set[str] = set()
    for install_id in install_ids:
        install = session.get(ModelInstall, install_id)
        family = install.manifest_json.get("family") if install else None
        if not isinstance(family, str) or not _family_key(family):
            return set()
        families.add(family.strip().casefold())
    keys = {_family_key(family) for family in families}
    return {min(families)} if len(keys) == 1 else set()


def _checkpoint_install_ids(session: Session, revision: WorkflowRevision) -> list[str]:
    """The installed model a workflow's one checkpoint loader names, as its install id.

    The family then comes from what was recorded about that model when it was
    installed, not from anything the graph claims. One loader naming a file
    that exactly one installed model holds is the only answer: no loader,
    several, a name that is not text, or a file no model or several models
    hold, leaves the family unknown, which is what it was before.
    """

    graph = revision.api_graph_json
    if not isinstance(graph, dict):
        return []
    loaders = [
        node
        for node in graph.values()
        if isinstance(node, dict) and node.get("class_type") == "CheckpointLoaderSimple"
    ]
    if len(loaders) != 1:
        return []
    inputs = loaders[0].get("inputs")
    named = inputs.get("ckpt_name") if isinstance(inputs, dict) else None
    if not isinstance(named, str) or not named.strip():
        return []
    wanted = _file_name(named)
    holders: list[str] = []
    for install in session.scalars(select(ModelInstall)).all():
        files = install.manifest_json.get("files")
        if isinstance(files, list) and any(
            isinstance(entry, str) and _file_name(entry) == wanted for entry in files
        ):
            holders.append(install.id)
    return holders if len(holders) == 1 else []


def _file_name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _diffusion_model_family(session: Session, revision: WorkflowRevision) -> str | None:
    """The family of the one registered diffusion model a graph's model loader names.

    A graph that builds its model from a diffusion-model loader, rather than a
    checkpoint, names the file by the name the runtime loads it under. A
    registered diffusion model records exactly that name, beside the digest of
    the file it measured and the family recorded for it. One such loader in a
    graph with no checkpoint loader, naming a file exactly one registered
    diffusion model answers to, is the only answer, and that model has to be
    active, verified and carry a family. Anything else leaves the family
    unknown, which is what it was.
    """

    graph = revision.api_graph_json
    if not isinstance(graph, dict):
        return None
    nodes = [node for node in graph.values() if isinstance(node, dict)]
    if any(node.get("class_type") == "CheckpointLoaderSimple" for node in nodes):
        return None
    loaders = [node for node in nodes if node.get("class_type") == "UNETLoader"]
    if len(loaders) != 1:
        return None
    inputs = loaders[0].get("inputs")
    named = inputs.get("unet_name") if isinstance(inputs, dict) else None
    if not isinstance(named, str) or not named.strip():
        return None
    wanted = _loader_name(named)
    holders = [
        asset
        for asset in session.scalars(
            select(ModelAssetInstall).where(ModelAssetInstall.kind == "diffusion_model")
        ).all()
        if isinstance(asset.manifest_json.get("comfy_name"), str)
        and _loader_name(asset.manifest_json["comfy_name"]) == wanted
    ]
    if len(holders) != 1:
        return None
    asset = holders[0]
    if not asset.active or asset.verified_at is None or not asset.family:
        return None
    return asset.family.strip().casefold() if _family_key(asset.family) else None


def _loader_name(name: str) -> str:
    return name.replace("\\", "/").casefold()


def _family_key(family: str) -> str:
    """The spelling one base model family is compared by.

    Files and installs record a family as each of them spells it: one base
    model arrives as krea2 and as Krea-2, another as z-image-turbo and as
    zimage-turbo. Compared as written, a LoRA made for the model a workflow
    runs would be refused as made for another one. Case and everything but
    letters and digits are set aside, so only a difference in the name itself
    makes a different family.
    """

    return "".join(character for character in family.casefold() if character.isalnum())


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
