from __future__ import annotations

import subprocess

import pytest

import local_lm.hardware as hardware


@pytest.mark.parametrize(
    ("total", "free", "expected_total", "expected_free"),
    [
        ("24576", "12288", 24576, 12288),
        ("24576", "0", 24576, 0),
        ("N/A", "N/A", None, None),
        ("24576", "N/A", 24576, None),
        ("N/A", "12288", None, 12288),
        ("[Not Supported]", "12288", None, 12288),
        ("24576", "", 24576, None),
        ("-1", "-1", None, None),
        ("nan", "inf", None, None),
        ("24576", "24577", 24576, None),
        ("99999999999999999999999", "1", None, 1),
    ],
)
def test_unavailable_gpu_memory_does_not_hide_other_device_facts(
    monkeypatch: pytest.MonkeyPatch,
    total: str,
    free: str,
    expected_total: int | None,
    expected_free: int | None,
) -> None:
    def execute(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"0, Example GPU, {total}, {free}, 600.0\n1, Second GPU, 8192, 4096, 600.0\n",
        )

    monkeypatch.setattr("local_lm.hardware.shutil.which", lambda name: "nvidia-smi")
    monkeypatch.setattr("local_lm.hardware.subprocess.run", execute)

    devices = hardware._nvidia_devices()

    assert len(devices) == 2
    first, second = devices
    assert (first.id, first.name, first.kind, first.backend) == (
        "cuda:0",
        "Example GPU",
        "gpu",
        "cuda",
    )
    assert first.details == {"driver": "600.0"}
    assert first.total_memory_bytes == (
        expected_total * 1024 * 1024 if expected_total is not None else None
    )
    assert first.available_memory_bytes == (
        expected_free * 1024 * 1024 if expected_free is not None else None
    )
    assert second.total_memory_bytes == 8192 * 1024 * 1024
    assert second.available_memory_bytes == 4096 * 1024 * 1024
