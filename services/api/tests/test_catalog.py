from __future__ import annotations

import httpx
import pytest

from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.domain import CompatibilityLevel
from local_lm.downloads import DownloadManager
from local_lm.preflight import _automatic_selection
from local_lm.schemas import DownloadRequest


class Sibling:
    def __init__(self, name: str, size: int) -> None:
        self.rfilename = name
        self.size = size


def test_gguf_catalog_entry_is_likely_compatible() -> None:
    level, reasons = HuggingFaceCatalog._compatibility(
        requested_role="chat",
        pipeline_tag="text-generation",
        tags=["gguf"],
        filenames=["model-q4.gguf"],
    )
    assert level == CompatibilityLevel.LIKELY
    assert "GGUF" in reasons[0]


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


async def test_catalog_detail_requests_live_blob_metadata(tmp_path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "owner/Model-8B-GGUF",
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
    assert detail["files"] == [
        {
            "filename": "Model-8B-Q4_K_M.gguf",
            "size": 5_000_000_000,
            "sha256": "a" * 64,
        }
    ]


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
