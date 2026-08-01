param([Parameter(Mandatory = $true)][string]$TargetPath)

$encoding = [System.Text.UTF8Encoding]::new($false)
$content = [System.IO.File]::ReadAllText($TargetPath, $encoding)
$test = @'


async def test_template_workflow_exposes_model_only_loras(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    graph = {
        "model": {"class_type": "UNETLoader", "inputs": {}},
        "sampling": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["model", 0]},
        },
        "switch": {
            "class_type": "ComfySwitchNode",
            "inputs": {
                "on_true": ["sampling", 0],
                "on_false": ["model", 0],
            },
        },
        "sampler": {"class_type": "KSampler", "inputs": {"model": ["switch", 0]}},
    }
    compiled = CompiledComfyTemplate(
        template=ComfyTemplate(
            id="image_split_model_lora_test",
            path=settings.data_dir / "split-model-template.json",
            role="image",
            operation="image_to_image",
            score=1_000,
            sha256="9" * 64,
            dependencies=(),
        ),
        ui_graph={"nodes": []},
        api_graph=graph,
        input_schema={"type": "object", "properties": {}},
    )
    with SessionLocal() as session:
        install = ModelInstall(
            name="Split model editor",
            role="image",
            engine="comfyui",
            local_path=str(settings.model_dir / "split-model-editor"),
            manifest_json={"family": "split-model-test"},
            active=True,
        )
        session.add(install)
        session.flush()

        revision = DownloadManager._ensure_template_workflow(session, compiled, install)

        assert revision.input_schema_json["properties"]["loras"] == {
            "type": "array",
            "title": "LoRAs",
            "description": "Optional verified LoRAs applied in order.",
            "default": [],
            "maxItems": 8,
        }
        assert revision.dependencies_json["extensions"]["lora"] == {
            "mode": "model_only",
            "model": ["switch", 0],
        }
'@

$content = $content.TrimEnd("`r", "`n") + $test + "`r`n"
[System.IO.File]::WriteAllText($TargetPath, $content, $encoding)
