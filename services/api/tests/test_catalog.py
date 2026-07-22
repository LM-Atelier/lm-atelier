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
