param([Parameter(Mandatory = $true)][string]$TargetPath)

$encoding = [System.Text.UTF8Encoding]::new($false)
$content = [System.IO.File]::ReadAllText($TargetPath, $encoding)

if (-not $content.Contains("    checkpoint_lora_extension,`n")) {
    throw "Expected test import was not found."
}
$content = $content.Replace(
    "    checkpoint_lora_extension,`n",
    "    checkpoint_lora_extension,`n    detect_lora_extension,`n"
)
$content = $content.Replace(
    'assert LORA_GRAPH_TRANSFORM_VERSION == "lora-graph-v1"',
    'assert LORA_GRAPH_TRANSFORM_VERSION == "lora-graph-v2"'
)

$tests = @'


def test_model_only_lora_extension_is_detected_and_transformed() -> None:
    graph = {
        "161": {"class_type": "UNETLoader", "inputs": {}},
        "145": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["161", 0]},
        },
        "152": {"class_type": "CFGNorm", "inputs": {"model": ["145", 0]}},
        "153": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["152", 0], "lora_name": "lightning.safetensors"},
        },
        "163": {
            "class_type": "ComfySwitchNode",
            "inputs": {"on_true": ["153", 0], "on_false": ["152", 0]},
        },
        "169": {"class_type": "KSampler", "inputs": {"model": ["163", 0]}},
    }
    extension = detect_lora_extension(graph)
    assert extension == {"mode": "model_only", "model": ["163", 0]}
    schema = {
        "type": "object",
        "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
    }
    dependencies = {"extensions": {"lora": extension}}
    validate_lora_workflow_contract(graph, schema, dependencies)

    transformed = transform_lora_graph(
        graph,
        extension,
        [
            {
                "comfy_name": "detail.safetensors",
                "model_strength": 0.8,
                "clip_strength": 0.6,
            },
            {
                "comfy_name": "style.safetensors",
                "model_strength": 1.1,
                "clip_strength": 0.9,
            },
        ],
    )

    assert transformed["lma_lora_001"] == {
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "LM Atelier LoRA 1"},
        "inputs": {
            "model": ["163", 0],
            "lora_name": "detail.safetensors",
            "strength_model": 0.8,
        },
    }
    assert transformed["lma_lora_002"]["inputs"]["model"] == ["lma_lora_001", 0]
    assert transformed["169"]["inputs"]["model"] == ["lma_lora_002", 0]
    assert transformed["163"] == graph["163"]


def test_model_only_lora_extension_fails_closed_for_multiple_model_paths() -> None:
    graph = {
        "base": {"class_type": "UNETLoader", "inputs": {}},
        "refiner": {"class_type": "UNETLoader", "inputs": {}},
        "first": {"class_type": "KSampler", "inputs": {"model": ["base", 0]}},
        "second": {
            "class_type": "KSamplerAdvanced",
            "inputs": {"model": ["refiner", 0]},
        },
    }
    assert detect_lora_extension(graph) is None
    schema = {
        "type": "object",
        "properties": {"loras": {"type": "array", "default": [], "maxItems": 8}},
    }
    dependencies = {
        "extensions": {"lora": {"mode": "model_only", "model": ["base", 0]}}
    }
    with pytest.raises(ValueError, match="every supported sampler"):
        validate_lora_workflow_contract(graph, schema, dependencies)
'@

$content = $content.TrimEnd("`r", "`n") + $tests + "`r`n"
[System.IO.File]::WriteAllText($TargetPath, $content, $encoding)
