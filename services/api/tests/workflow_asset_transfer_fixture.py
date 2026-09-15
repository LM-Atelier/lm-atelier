"""Serve a constructed tensor through the real download worker's HTTP client."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pytest
    from fastapi import FastAPI

_REVISION = "d" * 40
AssetKind = Literal["lora", "background_removal"]


@dataclass(frozen=True)
class ConstructedAssetTransfer:
    plan_id: str
    content: bytes
    receipt: Path
    repository: str
    filename: str
    tensor_name: str
    runtime_folder: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def selection(self) -> dict[str, str]:
        return {
            "reference_filename": self.filename,
            "install_plan_id": self.plan_id,
            "artifact_path": self.filename,
        }

    def assert_transferred(self) -> None:
        recorded = json.loads(self.receipt.read_text(encoding="utf-8"))
        assert recorded["pid"] != os.getpid()
        assert recorded["requests"] == ["HEAD", "GET"]


def configure_asset_transfer(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    directory: Path,
    *,
    asset_kind: AssetKind = "lora",
) -> ConstructedAssetTransfer:
    from local_lm import downloads
    from local_lm.db import SessionLocal
    from local_lm.model_manifests import inspect_repository_metadata
    from local_lm.model_planner import persist_install_plan, resolve_install_plan

    repository = (
        "synthetic/neutral-detail" if asset_kind == "lora" else "synthetic/background-removal"
    )
    filename = "detail.safetensors" if asset_kind == "lora" else "birefnet.safetensors"
    tensor_name = (
        "lora_unet_block.lora_down.weight" if asset_kind == "lora" else "encoder.blocks.0.weight"
    )
    runtime_folder = "loras" if asset_kind == "lora" else "background_removal"
    header = json.dumps(
        {tensor_name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}},
        separators=(",", ":"),
    ).encode()
    header += b" " * (-len(header) % 8)
    content = len(header).to_bytes(8, "little") + header + struct.pack("<e", 0.5)
    digest = hashlib.sha256(content).hexdigest()
    planned = resolve_install_plan(
        remote_id=repository,
        revision=_REVISION,
        role="image",
        engine="comfyui",
        selected_files=[{"filename": filename, "size": len(content), "sha256": digest}],
        inspection=inspect_repository_metadata({filename: content}, [filename], role="image"),
        comfy_paths={"loras": "."} if asset_kind == "lora" else None,
        auxiliary_kind="lora" if asset_kind == "lora" else None,
        workflow_reference_kind=(
            "background_removal" if asset_kind == "background_removal" else None
        ),
    )
    with SessionLocal() as session:
        plan = persist_install_plan(session, planned)
        session.commit()
        plan_id = plan.id
    source = directory / "constructed-asset.bin"
    source.write_bytes(content)
    receipt = directory / "constructed-transfer.json"
    monkeypatch.setattr(
        downloads,
        "download_worker_command",
        lambda: [
            sys.executable,
            str(Path(__file__).resolve()),
            str(source),
            str(receipt),
            repository,
            _REVISION,
            filename,
        ],
    )
    monkeypatch.setattr(
        app.state.services.downloads,
        "_api",
        SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename=filename, size=len(content), lfs={"sha256": digest})
                ],
                sha=_REVISION,
                pipeline_tag=None,
                tags=[asset_kind],
                gated=False,
            )
        ),
    )
    return ConstructedAssetTransfer(
        plan_id,
        content,
        receipt,
        repository,
        filename,
        tensor_name,
        runtime_folder,
    )


def _worker_main(
    source: Path,
    receipt: Path,
    repository: str,
    revision: str,
    filename: str,
) -> int:
    # Isolate the client cache and implicit credentials before importing the SDK.
    os.environ["HF_HOME"] = str(source.parent / "constructed-hub-cache")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import httpx
    from huggingface_hub import set_client_factory

    from local_lm import download_worker

    content = source.read_bytes()
    requests: list[str] = []

    def serve(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "huggingface.co"
        assert request.url.path == f"/{repository}/resolve/{revision}/{filename}"
        assert request.method in {"HEAD", "GET"}
        assert "authorization" not in request.headers
        requests.append(request.method)
        receipt.write_text(json.dumps({"pid": os.getpid(), "requests": requests}), encoding="utf-8")
        return httpx.Response(
            200,
            headers={
                "etag": '"' + hashlib.sha256(content).hexdigest() + '"',
                "x-repo-commit": revision,
                "content-length": str(len(content)),
            },
            stream=httpx.ByteStream(content if request.method == "GET" else b""),
        )

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        set_client_factory(lambda: client)
        return download_worker.main()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(
        _worker_main(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
            sys.argv[3],
            sys.argv[4],
            sys.argv[5],
        )
    )
