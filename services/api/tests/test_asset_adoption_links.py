"""Adoption measures a model file in its own folder, never through a link."""

from __future__ import annotations

import json
import os
import struct
import subprocess
from pathlib import Path

import pytest

from local_lm.asset_adoption import AssetAdoptionError, measure_adoptable_file

PAYLOAD = b"A neutral fixture." * 64


def _safetensors(path: Path) -> bytes:
    header = {
        "weight": {
            "dtype": "F32",
            "shape": [len(PAYLOAD) // 4],
            "data_offsets": [0, len(PAYLOAD)],
        },
        "__metadata__": {"modelspec.architecture": "fixture/lora"},
    }
    encoded = json.dumps(header).encode("utf-8")
    content = struct.pack("<Q", len(encoded)) + encoded + PAYLOAD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def _make_link_dir(link: Path, target: Path) -> bool:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def test_a_linked_model_file_is_not_measured(tmp_path: Path) -> None:
    folder = tmp_path / "loras"
    folder.mkdir()
    real = folder / "other.safetensors"
    content = _safetensors(real)
    linked = folder / "slider.safetensors"
    try:
        linked.symlink_to(real)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    with pytest.raises(AssetAdoptionError) as refused:
        measure_adoptable_file([folder], "slider.safetensors")

    assert refused.value.code == "asset-file-unreadable"
    assert real.read_bytes() == content
    assert linked.is_symlink()


def test_a_linked_model_folder_is_not_measured(tmp_path: Path) -> None:
    real = tmp_path / "real"
    content = _safetensors(real / "slider.safetensors")
    folder = tmp_path / "loras"
    if not _make_link_dir(folder, real):
        pytest.skip("directory links are unavailable")

    with pytest.raises(AssetAdoptionError) as refused:
        measure_adoptable_file([folder], "slider.safetensors")

    assert refused.value.code == "asset-file-unreadable"
    assert (real / "slider.safetensors").read_bytes() == content


def test_a_link_in_one_folder_does_not_select_another_folders_file(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    inside = first / "inside.safetensors"
    _safetensors(inside)
    linked = first / "slider.safetensors"
    try:
        linked.symlink_to(inside)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    second = tmp_path / "second"
    other = _safetensors(second / "slider.safetensors")

    with pytest.raises(AssetAdoptionError) as refused:
        measure_adoptable_file([first, second], "slider.safetensors")

    assert refused.value.code == "asset-file-unreadable"
    assert second.joinpath("slider.safetensors").read_bytes() == other
