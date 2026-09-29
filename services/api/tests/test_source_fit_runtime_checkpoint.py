"""Opt-in real sampler/VAE controls with an untrained neutral checkpoint."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.mark.parametrize("text_encoder", [False, True])
def test_real_checkpoint_sampling_preserves_source(
    tmp_path: Path,
    text_encoder: bool,
    record_property: Callable[[str, object], None],
) -> None:
    executable = os.environ.get("LM_ATELIER_TEST_COMFY_EXECUTABLE")
    directory = os.environ.get("LM_ATELIER_TEST_COMFY_DIRECTORY")
    if not executable or not directory:
        pytest.skip("An isolated CPU ComfyUI runtime is required")
    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(
        [
            executable,
            str(Path(__file__).with_name("source_fit_checkpoint_fixture.py")),
            "--runtime",
            directory,
            "--output",
            str(tmp_path / "checkpoint"),
            *(["--text-encoder"] if text_encoder else []),
        ],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.splitlines()[-1])
    assert report["model_class"] == "SD15"
    execution = report["execution"]
    assert execution["source_equal"] is True
    assert execution["extension_changes_with_seed"] is True
    assert execution["inverted_mask_control"] is True
    assert execution["saved_pngs"] == 2
    assert execution["text_encoder"] is text_encoder
    assert execution["text_tensors"] > 0 if text_encoder else execution["text_tensors"] == 0
    for key, value in execution.items():
        record_property(key, value)
