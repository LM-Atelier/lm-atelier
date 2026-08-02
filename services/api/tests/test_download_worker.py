from __future__ import annotations

import io
import json
from typing import Any

import pytest

from local_lm import download_worker


def _stdin(payload: dict[str, Any]) -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode()))


def test_download_worker_keeps_legacy_huggingface_payload_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def download(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "C:/models/model.gguf"

    output = io.StringIO()
    monkeypatch.setattr(download_worker, "hf_hub_download", download)
    monkeypatch.setattr(
        download_worker.sys,
        "stdin",
        _stdin(
            {
                "repo_id": "owner/model",
                "filename": "model.gguf",
                "revision": "a" * 40,
                "local_dir": "C:/staging",
                "token": "secret",
            }
        ),
    )
    monkeypatch.setattr(download_worker.sys, "stdout", output)

    assert download_worker.main() == 0
    assert json.loads(output.getvalue()) == {"path": "C:/models/model.gguf"}
    assert captured["repo_id"] == "owner/model"
    assert captured["token"] == "secret"


def test_download_worker_rejects_unimplemented_transfer_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(download_worker.sys, "stdin", _stdin({"kind": "https"}))

    with pytest.raises(ValueError, match="unsupported download worker kind: https"):
        download_worker.main()
