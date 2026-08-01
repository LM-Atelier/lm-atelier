param([Parameter(Mandatory = $true)][string]$TargetPath)

$encoding = [System.Text.UTF8Encoding]::new($false)
$content = [System.IO.File]::ReadAllText($TargetPath, $encoding)

function Replace-Exact {
    param([string]$Old, [string]$New)
    if (-not $script:content.Contains($Old)) {
        throw "Expected source block was not found."
    }
    $script:content = $script:content.Replace($Old, $New)
}

Replace-Exact 'LORA_GRAPH_TRANSFORM_VERSION = "lora-graph-v1"' 'LORA_GRAPH_TRANSFORM_VERSION = "lora-graph-v2"'

Replace-Exact @'
COMFY_AUXILIARY_FOLDERS = {
    "lora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscaler": "upscale_models",
    "embedding": "embeddings",
    "ip_adapter": "ipadapter",
}
'@ @'
COMFY_AUXILIARY_FOLDERS = {
    "lora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscaler": "upscale_models",
    "embedding": "embeddings",
    "ip_adapter": "ipadapter",
}
_MODEL_SAMPLER_CLASS_TYPES = {
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
}
'@

$start = $content.IndexOf('def checkpoint_lora_extension(')
$end = $content.IndexOf('def resolve_lora_stack(', $start)
if ($start -lt 0 -or $end -lt 0) { throw "LoRA contract function block was not found." }
$contractBlock = @'
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
    elif not _graph_contains_link(graph, model) or not _graph_contains_link(graph, clip):
        raise ValueError("The LoRA workflow extension must feed both model and CLIP graph inputs.")


'@
$content = $content.Substring(0, $start) + $contractBlock + $content.Substring($end)

$start = $content.IndexOf('def transform_lora_graph(')
$end = $content.IndexOf('def _strength(', $start)
if ($start -lt 0 -or $end -lt 0) { throw "LoRA transform function block was not found." }
$transformBlock = @'
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
        inputs = {
            "model": model_source,
            "lora_name": item["comfy_name"],
            "strength_model": item["model_strength"],
        }
        if not model_only:
            inputs.update(
                {
                    "clip": clip_source,
                    "strength_clip": item["clip_strength"],
                }
            )
        transformed[node_id] = {
            "class_type": "LoraLoaderModelOnly" if model_only else "LoraLoader",
            "_meta": {"title": f"LM Atelier LoRA {index}"},
            "inputs": inputs,
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


'@
$content = $content.Substring(0, $start) + $transformBlock + $content.Substring($end)

Replace-Exact @'
def _graph_contains_link(value: object, link: list[Any]) -> bool:
'@ @'
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
'@

[System.IO.File]::WriteAllText($TargetPath, $content, $encoding)
