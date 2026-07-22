from __future__ import annotations

import pytest

from local_lm.catalog import HuggingFaceCatalog
from local_lm.domain import CompatibilityLevel
from local_lm.downloads import DownloadManager
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
            "id": "owner/model",
            "tags": ["gguf", "q4_k_m", "license:mit"],
            "siblings": [{"rfilename": "model.gguf"}],
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
    ) == [model]
    assert (
        HuggingFaceCatalog._filter_items(
            [model],
            compatibility=None,
            file_format="safetensors",
            quantization=None,
            license_id=None,
            gated=None,
            architecture=None,
        )
        == []
    )
    with pytest.raises(ValueError, match="huggingface.co"):
        HuggingFaceCatalog._validated_cursor("https://example.com/api/models?cursor=secret")
