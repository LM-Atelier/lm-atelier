from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from local_lm import downloads
from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.domain import CompatibilityLevel
from local_lm.downloads import DownloadManager
from local_lm.preflight import (
    _automatic_selection,
    _automatic_vllm_selection,
    assess_catalog_install,
)
from local_lm.schemas import (
    CatalogDetail,
    CatalogModel,
    CatalogPage,
    CatalogPreflightRequest,
    DownloadRequest,
    SystemInfo,
)


class Sibling:
    def __init__(self, name: str, size: int, sha256: str | None = None) -> None:
        self.rfilename = name
        self.size = size
        self.lfs = {"sha256": sha256} if sha256 else None


def test_gguf_catalog_entry_is_likely_compatible() -> None:
    level, reasons = HuggingFaceCatalog._compatibility(
        requested_role="chat",
        pipeline_tag="text-generation",
        tags=["gguf"],
        filenames=["model-q4.gguf"],
    )
    assert level == CompatibilityLevel.LIKELY
    assert "GGUF" in reasons[0]


def test_modelopt_catalog_entry_names_its_required_runtime() -> None:
    model = HuggingFaceCatalog._normalize(
        {
            "id": "nvidia/Qwen3.6-27B-NVFP4",
            "pipeline_tag": "text-generation",
            "tags": ["safetensors", "ModelOpt", "NVFP4"],
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "hf_quant_config.json"},
                {"rfilename": "model-00001-of-00003.safetensors"},
            ],
        },
        "chat",
    )

    assert model.compatibility == "advanced_import"
    assert model.compatibility_reasons == ["requires the managed vLLM ModelOpt runtime"]
    assert model.required_runtime == "vllm"


def test_vllm_snapshot_selection_keeps_required_data_files_only() -> None:
    files = [
        {"filename": "model-00001-of-00003.safetensors"},
        {"filename": "model-00002-of-00003.safetensors"},
        {"filename": "model-00003-of-00003.safetensors"},
        {"filename": "model.safetensors.index.json"},
        {"filename": "config.json"},
        {"filename": "hf_quant_config.json"},
        {"filename": "tokenizer.json"},
        {"filename": "processor_config.json"},
        {"filename": "custom_modeling.py"},
        {"filename": "README.md"},
    ]

    selected = _automatic_vllm_selection(files)

    assert selected == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
        "config.json",
        "hf_quant_config.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "tokenizer.json",
    ]


def test_vllm_snapshot_selection_requires_modelopt_metadata() -> None:
    assert (
        _automatic_vllm_selection(
            [
                {"filename": "model.safetensors"},
                {"filename": "config.json"},
            ]
        )
        == []
    )


def test_default_chat_download_selects_smallest_gguf() -> None:
    request = DownloadRequest(
        remote_id="owner/model",
        role="chat",
        engine="llama.cpp",
    )
    files = DownloadManager._select_files(
        request,
        [Sibling("large.gguf", 20), Sibling("small.gguf", 10), Sibling("README.md", 1)],
    )
    assert files == ["small.gguf"]


def test_default_chat_download_includes_matching_multimodal_projector() -> None:
    request = DownloadRequest(
        remote_id="owner/model",
        role="chat",
        engine="llama.cpp",
    )
    files = DownloadManager._select_files(
        request,
        [
            Sibling("vision-model-4B-Q4_K_M.gguf", 20, "a" * 64),
            Sibling("vision-model-8B-Q4_K_M.gguf", 40, "b" * 64),
            Sibling("mmproj-vision-model-4B-f32.gguf", 8, "c" * 64),
            Sibling("mmproj-vision-model-4B-f16.gguf", 4, "d" * 64),
            Sibling("mmproj-vision-model-8B-f16.gguf", 6, "e" * 64),
        ],
    )

    assert files == [
        "vision-model-4B-Q4_K_M.gguf",
        "mmproj-vision-model-4B-f16.gguf",
    ]


def test_qwen36_chat_download_selects_main_model_and_vision_projector_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    # Selection ranks only candidates that fit in system memory, so pin the
    # host RAM to keep the expectation machine-independent.
    monkeypatch.setattr(
        downloads,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(total=64 * gib)),
    )
    request = DownloadRequest(
        remote_id="ggml-org/Qwen3.6-27B-GGUF",
        role="chat",
        engine="llama.cpp",
    )

    files = DownloadManager._select_files(
        request,
        [
            Sibling("Qwen3.6-27B-BF16.gguf", 53 * gib, "a" * 64),
            Sibling("Qwen3.6-27B-Q4_K_M.gguf", 19 * gib, "b" * 64),
            Sibling("Qwen3.6-27B-Q8_0.gguf", 28 * gib, "c" * 64),
            Sibling("dflash-Qwen3.6-27B-Q8_0.gguf", 2 * gib, "d" * 64),
            Sibling("mtp-Qwen3.6-27B-Q4_0.gguf", 2 * gib, "e" * 64),
            Sibling("mmproj-Qwen3.6-27B-BF16.gguf", 1 * gib, "f" * 64),
            Sibling("mmproj-Qwen3.6-27B-Q8_0.gguf", 700 * 1024**2, "1" * 64),
        ],
    )

    assert files == [
        "Qwen3.6-27B-Q4_K_M.gguf",
        "mmproj-Qwen3.6-27B-BF16.gguf",
    ]


def test_chat_download_selects_every_shard_from_one_quantization() -> None:
    request = DownloadRequest(
        remote_id="owner/model",
        role="chat",
        engine="llama.cpp",
    )
    siblings = [
        Sibling("model-Q2_K-00001-of-00002.gguf", 4, "a" * 64),
        Sibling("model-Q2_K-00002-of-00002.gguf", 4, "b" * 64),
        Sibling("model-Q4_K_M-00001-of-00002.gguf", 6, "c" * 64),
        Sibling("model-Q4_K_M-00002-of-00002.gguf", 6, "d" * 64),
    ]

    assert DownloadManager._select_files(request, siblings) == [
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
    ]


def test_chat_download_rejects_an_incomplete_split_gguf() -> None:
    request = DownloadRequest(
        remote_id="owner/model",
        role="chat",
        engine="llama.cpp",
    )

    with pytest.raises(ValueError, match="incomplete; missing shard"):
        DownloadManager._select_files(
            request,
            [Sibling("model-Q4_K_M-00001-of-00002.gguf", 6, "a" * 64)],
        )


def test_chat_download_rejects_duplicate_split_shard_metadata() -> None:
    request = DownloadRequest(
        remote_id="owner/model",
        role="chat",
        engine="llama.cpp",
    )
    siblings = [
        Sibling("model-Q4_K_M-00001-of-00002.gguf", 6, "a" * 64),
        Sibling("model-Q4_K_M-00001-of-00002.gguf", 6, "a" * 64),
        Sibling("model-Q4_K_M-00002-of-00002.gguf", 6, "b" * 64),
    ]

    with pytest.raises(ValueError, match="duplicate shard"):
        DownloadManager._select_files(request, siblings)


def test_explicit_chat_download_rejects_mixed_quantization_sets() -> None:
    filenames = [
        "model-Q4_K_M-00001-of-00002.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
        "model-Q5_K_M-00001-of-00002.gguf",
        "model-Q5_K_M-00002-of-00002.gguf",
    ]
    request = DownloadRequest(
        remote_id="owner/model",
        role="chat",
        engine="llama.cpp",
        allow_patterns=["*.gguf"],
    )
    siblings = [
        Sibling(filename, index + 1, f"{index + 1:064x}")
        for index, filename in enumerate(filenames)
    ]

    with pytest.raises(ValueError, match="mix quantizations or shard sets"):
        DownloadManager._select_files(request, siblings)


def test_automatic_chat_selection_falls_back_to_smallest_when_none_fit_memory() -> None:
    gib = 1024**3
    files = {
        "model-Q4_K_M.gguf": {
            "filename": "model-Q4_K_M.gguf",
            "size": 5 * gib,
        },
        "model-Q2_K.gguf": {
            "filename": "model-Q2_K.gguf",
            "size": 2 * gib,
        },
    }
    assert _automatic_selection(files, "chat", 2 * gib) == ["model-Q2_K.gguf"]


def test_chat_preflight_selects_and_hashes_multimodal_projector(tmp_path: Path) -> None:
    detail = CatalogDetail(
        model=CatalogModel(
            remote_id="owner/vision-model",
            name="vision-model",
            compatibility="likely",
        ),
        revision="a" * 40,
        files=[
            {
                "filename": "vision-model-4B-Q4_K_M.gguf",
                "size": 10,
                "sha256": "a" * 64,
            },
            {
                "filename": "mmproj-vision-model-4B-f16.gguf",
                "size": 3,
                "sha256": "b" * 64,
            },
        ],
    )
    system = SystemInfo.model_construct(
        memory_total_bytes=16 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        devices=[],
    )

    result = assess_catalog_install(
        detail,
        CatalogPreflightRequest(role="chat", engine="llama.cpp"),
        Settings(data_dir=tmp_path),
        system,
    )

    assert result.selected_files == [
        "vision-model-4B-Q4_K_M.gguf",
        "mmproj-vision-model-4B-f16.gguf",
    ]
    assert result.expected_sha256 == {
        "vision-model-4B-Q4_K_M.gguf": "a" * 64,
        "mmproj-vision-model-4B-f16.gguf": "b" * 64,
    }
    assert result.download_bytes == 13
    assert result.can_install is True


def test_chat_preflight_preserves_external_projector_provenance(tmp_path: Path) -> None:
    destination = "companions/author/model/mmproj-vision-model-4B-f16.gguf"
    detail = CatalogDetail(
        model=CatalogModel(
            remote_id="converter/vision-model-gguf",
            name="vision-model-gguf",
            compatibility="likely",
        ),
        revision="a" * 40,
        files=[
            {
                "filename": "vision-model-4B-Q4_K_M.gguf",
                "size": 10,
                "sha256": "a" * 64,
            },
            {
                "filename": destination,
                "size": 3,
                "sha256": "b" * 64,
                "source_remote_id": "author/vision-model",
                "source_revision": "c" * 40,
                "source_filename": "mmproj-vision-model-4B-f16.gguf",
            },
        ],
    )
    system = SystemInfo.model_construct(
        memory_total_bytes=16 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        devices=[],
    )

    result = assess_catalog_install(
        detail,
        CatalogPreflightRequest(role="chat", engine="llama.cpp"),
        Settings(data_dir=tmp_path),
        system,
    )

    assert result.selected_files == [
        "vision-model-4B-Q4_K_M.gguf",
        destination,
    ]
    assert result.file_sources[destination].model_dump() == {
        "remote_id": "author/vision-model",
        "revision": "c" * 40,
        "filename": "mmproj-vision-model-4B-f16.gguf",
        "size_bytes": 3,
        "sha256": "b" * 64,
    }


async def test_catalog_discovers_a_strongly_matched_external_projector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    candidate = CatalogModel(
        remote_id="HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive",
        name="Qwen3.6-27B-Uncensored-HauhauCS-Aggressive",
        pipeline_tag="image-text-to-text",
        tags=["vision", "multimodal"],
        downloads=100,
        compatibility="likely",
    )

    async def search(**_kwargs: object) -> CatalogPage:
        return CatalogPage(items=[candidate])

    async def inspect(
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict[str, object]:
        del revision, requested_role
        assert remote_id == candidate.remote_id
        return {
            "revision": "c" * 40,
            "files": [
                {
                    "filename": ("mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-f16.gguf"),
                    "size": 927_606_976,
                    "sha256": "d" * 64,
                }
            ],
        }

    monkeypatch.setattr(catalog, "search", search)
    monkeypatch.setattr(catalog, "inspect", inspect)
    try:
        result = await catalog.discover_vision_projector(
            ("SummonGovernance/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-NVFP4-MTP-GGUF"),
            [("Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-NVFP4-MTP-Q4_K_P.gguf")],
        )
    finally:
        await catalog.close()

    assert result == {
        "filename": (
            "companions/HauhauCS/Qwen3.6-27B-Uncensored-HauhauCS-Aggressive/"
            "mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-f16.gguf"
        ),
        "size": 927_606_976,
        "sha256": "d" * 64,
        "source_remote_id": candidate.remote_id,
        "source_revision": "c" * 40,
        "source_filename": ("mmproj-Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-f16.gguf"),
    }


@pytest.mark.parametrize(
    ("role", "engine", "filename"),
    [
        ("chat", "llama.cpp", "model-Q4_K_M.gguf"),
        ("image", "comfyui", "model.safetensors"),
        ("video", "comfyui", "model.safetensors"),
    ],
)
def test_preflight_pins_the_catalog_resolved_revision_for_every_role(
    tmp_path: Path,
    role: str,
    engine: str,
    filename: str,
) -> None:
    resolved_revision = "a" * 40
    detail = CatalogDetail(
        model=CatalogModel(
            remote_id="owner/model",
            name="model",
            compatibility="likely",
        ),
        revision=resolved_revision,
        files=[
            {
                "filename": filename,
                "size": 1024,
                "sha256": "b" * 64,
            }
        ],
    )
    request = CatalogPreflightRequest(
        revision="main",
        role=role,  # type: ignore[arg-type]
        engine=engine,
    )
    system = SystemInfo.model_construct(
        memory_total_bytes=16 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        devices=[],
    )

    result = assess_catalog_install(
        detail,
        request,
        Settings(data_dir=tmp_path / role),
        system,
    )

    revision_check = next(check for check in result.checks if check.id == "revision")
    assert result.revision == resolved_revision
    assert revision_check.status == "pass"
    assert resolved_revision in revision_check.detail


def test_explicit_download_patterns_are_honored() -> None:
    request = DownloadRequest(
        remote_id="owner/model",
        role="image",
        engine="comfyui",
        allow_patterns=["*.safetensors"],
    )
    files = DownloadManager._select_files(
        request,
        [Sibling("model.safetensors", 10), Sibling("unsafe.ckpt", 9)],
    )
    assert files == ["model.safetensors"]


def test_automatic_comfy_paths_cover_checkpoint_and_component_layouts() -> None:
    assert DownloadManager._automatic_comfy_paths(["model.safetensors"]) == {"checkpoints": "."}
    assert DownloadManager._automatic_comfy_paths(
        [
            "split_files/diffusion_models/model.safetensors",
            "split_files/text_encoders/encoder.safetensors",
            "split_files/vae/vae.safetensors",
        ]
    ) == {
        "diffusion_models": "split_files/diffusion_models",
        "text_encoders": "split_files/text_encoders",
        "vae": "split_files/vae",
    }


def _write_safetensors_header(path: Path, tensor_names: list[str]) -> None:
    header = {
        name: {
            "dtype": "F16",
            "shape": [1],
            "data_offsets": [index * 2, index * 2 + 2],
        }
        for index, name in enumerate(tensor_names)
    }
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0" * (len(header) * 2))


def test_adaptive_checkpoint_probe_accepts_a_complete_standard_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.safetensors"
    _write_safetensors_header(
        checkpoint,
        [
            "model.diffusion_model.input_blocks.0.weight",
            "first_stage_model.encoder.conv_in.weight",
            "conditioner.embedders.0.transformer.weight",
        ],
    )

    DownloadManager._validate_standard_checkpoint_safetensors(checkpoint)


def test_adaptive_checkpoint_probe_rejects_a_single_non_checkpoint_safetensors_repo(
    tmp_path: Path,
) -> None:
    lora = tmp_path / "model.safetensors"
    _write_safetensors_header(
        lora,
        [
            "lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight",
            "lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight",
        ],
    )

    with pytest.raises(ValueError, match="not a complete standard checkpoint"):
        DownloadManager._validate_standard_checkpoint_safetensors(lora)


def test_pickle_compatible_download_is_blocked() -> None:
    request = DownloadRequest(
        remote_id="owner/model",
        role="image",
        engine="comfyui",
        allow_patterns=["*.bin"],
    )
    with pytest.raises(ValueError, match="blocked"):
        DownloadManager._select_files(request, [Sibling("pytorch_model.bin", 10)])


def test_catalog_normalizes_filterable_model_metadata() -> None:
    model = HuggingFaceCatalog._normalize(
        {
            "id": "owner/Model-8B-GGUF",
            "pipeline_tag": "text-generation",
            "tags": ["gguf", "q4_k_m", "license:apache-2.0"],
            "config": {"model_type": "qwen3"},
            "siblings": [
                {
                    "rfilename": "Model-8B-Q4_K_M.gguf",
                    "size": 5_000_000_000,
                }
            ],
        },
        "chat",
    )
    assert model.formats == ["gguf"]
    assert model.quantizations == ["q4_k_m"]
    assert model.parameter_count == 8_000_000_000
    assert model.license_id == "apache-2.0"
    assert model.architecture == "qwen3"
    assert model.total_size_bytes == 5_000_000_000


def test_catalog_normalizes_modern_comfy_quantization_names() -> None:
    model = HuggingFaceCatalog._normalize(
        {
            "id": "owner/modern-image-model",
            "tags": ["nvfp4", "license:apache-2.0"],
            "siblings": [
                {"rfilename": "diffusion/model_int8_convrot.safetensors"},
                {"rfilename": "diffusion/model-mxfp8.safetensors"},
                {"rfilename": "diffusion/model_convrot-w4a4.safetensors"},
            ],
        },
        "image",
    )

    assert model.quantizations == ["int8_convrot", "mxfp8", "nvfp4", "w4a4_convrot"]


def test_catalog_filters_and_rejects_external_cursors() -> None:
    model = HuggingFaceCatalog._normalize(
        {
            "id": "owner/model-8B",
            "tags": ["gguf", "q4_k_m", "license:mit"],
            "siblings": [{"rfilename": "model.gguf", "size": 5_000_000_000}],
        },
        "chat",
    )
    assert HuggingFaceCatalog._filter_items(
        [model],
        compatibility="likely",
        file_format="gguf",
        quantization="Q4_K_M",
        license_id="MIT",
        gated="open",
        architecture=None,
        min_parameters=None,
        max_parameters=None,
        max_size_bytes=None,
    ) == [model]
    assert (
        HuggingFaceCatalog._filter_items(
            [model],
            compatibility=None,
            file_format=None,
            quantization=None,
            license_id=None,
            gated=None,
            architecture=None,
            min_parameters=None,
            max_parameters=7_000_000_000,
            max_size_bytes=None,
        )
        == []
    )
    assert (
        HuggingFaceCatalog._filter_items(
            [model],
            compatibility=None,
            file_format="safetensors",
            quantization=None,
            license_id=None,
            gated=None,
            architecture=None,
            min_parameters=None,
            max_parameters=None,
            max_size_bytes=None,
        )
        == []
    )
    with pytest.raises(ValueError, match="huggingface.co"):
        HuggingFaceCatalog._validated_cursor("https://example.com/api/models?cursor=secret")

    old_model = model.model_copy(update={"last_modified": model.created_at})
    assert (
        HuggingFaceCatalog._filter_items(
            [old_model],
            compatibility=None,
            file_format=None,
            quantization=None,
            license_id=None,
            gated=None,
            architecture=None,
            updated_within_days=30,
        )
        == []
    )


async def test_catalog_uses_hugging_face_trending_order_and_update_age(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    )
    try:
        await catalog.search(role="chat", sort="trending", updated_within_days=30)
    finally:
        await catalog.close()

    assert requests[0].url.params["sort"] == "trendingScore"
    assert requests[0].url.params["direction"] == "-1"


async def test_catalog_uses_filtered_saved_results_during_an_outage(tmp_path) -> None:
    online = [True]

    def handler(_request: httpx.Request) -> httpx.Response:
        if not online[0]:
            raise httpx.ConnectError("offline")
        return httpx.Response(
            200,
            json=[
                {
                    "id": "owner/Model-8B-GGUF",
                    "pipeline_tag": "text-generation",
                    "tags": ["gguf", "license:mit"],
                }
            ],
        )

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    )
    try:
        live = await catalog.search(role="chat", sort="trending")
        online[0] = False
        saved = await catalog.search(role="chat", sort="trending", license_id="mit")
    finally:
        await catalog.close()

    assert live.stale is False
    assert [item.remote_id for item in saved.items] == ["owner/Model-8B-GGUF"]
    assert saved.stale is True
    assert saved.next_cursor is None


async def test_catalog_detail_requests_live_blob_metadata(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "owner/Model-8B-GGUF",
                "sha": "resolved-commit",
                "pipeline_tag": "text-generation",
                "tags": ["gguf", "q4_k_m"],
                "siblings": [
                    {
                        "rfilename": "Model-8B-Q4_K_M.gguf",
                        "size": 5_000_000_000,
                        "lfs": {"sha256": "a" * 64},
                    }
                ],
            },
        )

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    )
    try:
        detail = await catalog.inspect("owner/Model-8B-GGUF", "main", "chat")
    finally:
        await catalog.close()

    assert len(requests) == 1
    assert requests[0].url.params["revision"] == "main"
    assert requests[0].url.params["blobs"] == "true"
    assert "files_metadata" not in requests[0].url.params
    assert detail["revision"] == "resolved-commit"
    assert detail["files"] == [
        {
            "filename": "Model-8B-Q4_K_M.gguf",
            "size": 5_000_000_000,
            "sha256": "a" * 64,
        }
    ]


async def test_catalog_file_prefix_is_bounded_and_cached_by_revision(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(206, content=b'{"model_type":"qwen3"}')

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    )
    try:
        first = await catalog.inspect_file_prefix(
            "owner/model",
            "immutable-sha",
            "configs/config.json",
            max_bytes=128,
        )
        second = await catalog.inspect_file_prefix(
            "owner/model",
            "immutable-sha",
            "configs/config.json",
            max_bytes=128,
        )
    finally:
        await catalog.close()

    assert first == second == b'{"model_type":"qwen3"}'
    assert len(requests) == 1
    assert requests[0].headers["range"] == "bytes=0-127"
    assert requests[0].url.path == "/owner/model/resolve/immutable-sha/configs/config.json"


async def test_catalog_file_prefix_rejects_an_oversized_response(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"12345")

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ValueError, match="inspection limit"):
            await catalog.inspect_file_prefix(
                "owner/model",
                "immutable-sha",
                "config.json",
                max_bytes=4,
            )
    finally:
        await catalog.close()


async def test_catalog_file_prefix_rejects_unsafe_paths_without_a_request(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ValueError, match="safe relative path"):
            await catalog.inspect_file_prefix(
                "owner/model",
                "immutable-sha",
                "../config.json",
                max_bytes=128,
            )
    finally:
        await catalog.close()

    assert requests == []


@pytest.mark.parametrize(
    "remote_id",
    [
        "../api",
        "owner/model/extra",
        "owner/model?blobs=false",
        "owner/model#fragment",
        "owner%2fother/model",
        "owner/",
    ],
)
async def test_catalog_detail_rejects_unsafe_remote_ids_without_a_request(
    tmp_path: Path,
    remote_id: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(ValueError, match="owner/model"):
            await catalog.inspect(remote_id)
    finally:
        await catalog.close()

    assert requests == []


async def test_catalog_search_ignores_entries_with_unsafe_remote_ids(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"id": "owner/model", "tags": ["gguf"]},
                {"id": "owner/model?private=true", "tags": ["gguf"]},
                {"id": "../api/models", "tags": ["gguf"]},
            ],
        )

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    )
    try:
        page = await catalog.search(role="chat")
    finally:
        await catalog.close()

    assert [item.remote_id for item in page.items] == ["owner/model"]


async def test_maximum_size_filter_hydrates_live_file_sizes(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/models":
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "owner/Small-1B-GGUF",
                        "sha": "small-commit",
                        "tags": ["gguf"],
                        "siblings": [{"rfilename": "small.gguf"}],
                    },
                    {
                        "id": "owner/Large-2B-GGUF",
                        "sha": "large-commit",
                        "tags": ["gguf"],
                        "siblings": [{"rfilename": "large.gguf"}],
                    },
                ],
            )
        size = 5_000_000_000 if request.url.path.endswith("Small-1B-GGUF") else 15_000_000_000
        name = "small.gguf" if size < 10_000_000_000 else "large.gguf"
        return httpx.Response(
            200,
            json={
                "id": request.url.path.removeprefix("/api/models/"),
                "tags": ["gguf"],
                "siblings": [{"rfilename": name, "size": size}],
            },
        )

    catalog = HuggingFaceCatalog(Settings(data_dir=tmp_path))
    await catalog.close()
    catalog._client = httpx.AsyncClient(
        base_url="https://huggingface.co",
        transport=httpx.MockTransport(handler),
    )
    try:
        page = await catalog.search(role="chat", max_size_bytes=10_000_000_000)
    finally:
        await catalog.close()

    assert [item.remote_id for item in page.items] == ["owner/Small-1B-GGUF"]
    assert page.items[0].total_size_bytes == 5_000_000_000
    detail_requests = [request for request in requests if request.url.path != "/api/models"]
    assert len(detail_requests) == 2
    assert {request.url.params["revision"] for request in detail_requests} == {
        "small-commit",
        "large-commit",
    }
    assert all(request.url.params["blobs"] == "true" for request in detail_requests)
