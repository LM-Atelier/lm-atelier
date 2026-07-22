from __future__ import annotations

from pathlib import Path

import pytest

from local_lm.processes import ProcessSupervisor


def test_llama_arguments_are_explicit_and_shell_free() -> None:
    arguments = ProcessSupervisor._llama_load_arguments(
        {
            "context_length": 16_384,
            "gpu_layers": 40,
            "flash_attention": True,
            "mmap": False,
            "mlock": True,
            "tensor_split": "3,1",
        }
    )
    assert arguments == [
        "--ctx-size",
        "16384",
        "--n-gpu-layers",
        "40",
        "--tensor-split",
        "3,1",
        "--flash-attn",
        "on",
        "--no-mmap",
        "--mlock",
    ]


def test_gguf_resolution_rejects_ambiguous_installs(tmp_path: Path) -> None:
    (tmp_path / "a.gguf").touch()
    (tmp_path / "b.gguf").touch()
    with pytest.raises(ValueError, match="exactly one"):
        ProcessSupervisor._gguf_path(tmp_path, {})


def test_chat_memory_estimate_includes_model_and_context_overhead() -> None:
    gib = 1024**3
    assert ProcessSupervisor._estimate_chat_memory(5 * gib, {"context_length": 8192}) == 6 * gib
    assert ProcessSupervisor._estimate_chat_memory(5 * gib, {"context_length": 2048}) == (
        5 * gib + 512 * 1024**2
    )
