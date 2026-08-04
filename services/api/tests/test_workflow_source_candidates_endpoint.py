"""Analysis surfaces the sources an author recorded, without acting on them."""

from __future__ import annotations

from typing import Any

import pytest
from httpx2 import AsyncClient

pytestmark = pytest.mark.asyncio

PORTRAIT = "https://civitai.com/models/1662740/portrait?modelVersionId=3075606"
QWEN_VAE = (
    "https://huggingface.co/Comfy-Org/Qwen/resolve/main/split_files/vae/qwen_image_vae.safetensors"
)


def _graph() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "LoraLoader",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": ["portrait_finish.safetensors", 1.0, 1.0],
            },
            {
                "id": 2,
                "type": "VAELoader",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": ["qwen_image_vae.safetensors"],
            },
            {
                "id": 3,
                "type": "Note",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                # The realistic pair: one note names the model by its display
                # name, the other happens to name the file.
                "widgets_values": [
                    f"Portrait Finish\n{PORTRAIT}\n\nVAE qwen_image_vae.safetensors {QWEN_VAE}"
                ],
            },
            {
                "id": 4,
                "type": "Note",
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [
                    "Mirror: https://files.example.test/portrait_finish.safetensors"
                ],
            },
        ],
        "links": [],
    }


async def test_a_named_file_carries_its_source_and_the_rest_are_offered(
    client: AsyncClient,
) -> None:
    response = await client.post("/api/workflows/packages/analyze", json={"ui_graph": _graph()})

    assert response.status_code == 200
    body = response.json()
    assets = {asset["filename"]: asset for asset in body["asset_references"]}

    # The note names this exact file, so its source binds to it.
    vae = assets["qwen_image_vae.safetensors"]["source_candidates"]
    assert [candidate["remote_id"] for candidate in vae] == ["Comfy-Org/Qwen"]
    assert vae[0]["filename"] == "split_files/vae/qwen_image_vae.safetensors"

    # The LoRA is named only by its display name, so nothing is bound to it...
    assert assets["portrait_finish.safetensors"]["source_candidates"] == []
    # ...but the link is still offered rather than discarded.
    assert [candidate["remote_id"] for candidate in body["source_candidates"]] == ["3075606"]


async def test_a_host_no_registered_source_serves_never_appears(client: AsyncClient) -> None:
    body = (
        await client.post("/api/workflows/packages/analyze", json={"ui_graph": _graph()})
    ).json()

    offered = [candidate["url"] for candidate in body["source_candidates"]]
    for asset in body["asset_references"]:
        offered.extend(candidate["url"] for candidate in asset["source_candidates"])

    # The mirror link points at a host no catalog source serves. It names a
    # real filename, which is exactly why dropping it matters: a plausible
    # direct URL is the one thing this path must never hand onward.
    assert not any("example.test" in url for url in offered)
    assert all(
        url.startswith(("https://civitai.com/", "https://huggingface.co/")) for url in offered
    )
