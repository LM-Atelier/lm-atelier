"""Explicit image bindings preserve the selected source and references."""

from pathlib import Path
from typing import Any

import pytest

from local_lm.adapters.base import MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter


def graph_for(*slots: str) -> dict[str, Any]:
    return {
        str(index): {"class_type": "LoadImage", "inputs": {"image": "${" + slot + "}"}}
        for index, slot in enumerate(slots)
    }


def request_for(
    graph: dict[str, Any], bindings: dict[str, tuple[int, ...]] | None, count: int = 2
) -> MediaRequest:
    return MediaRequest(
        run_id="garden-inputs",
        operation="image_to_image",
        prompt="Arrange the garden",
        negative_prompt=None,
        input_paths=[Path(f"picture-{index}.png") for index in range(count)],
        workflow=graph,
        parameters={"input_image_3": "unselected.png", "input_images": ["unselected.png"]},
        input_image_bindings=bindings,
    )


@pytest.mark.parametrize(
    ("slots", "bindings", "count", "expected"),
    [
        (
            ("input_image_0", "input_image_3"),
            {"input_image_0": (1,), "input_image_3": (0,)},
            2,
            ["uploaded-1.png", "uploaded-0.png"],
        ),
        (
            ("input_image", "input_image_1"),
            {"input_image": (1,), "input_image_1": (0,)},
            2,
            ["uploaded-1.png", "uploaded-0.png"],
        ),
        (
            ("input_image", "input_images"),
            {"input_image": (1,), "input_images": (2, 0)},
            3,
            ["uploaded-1.png", ["uploaded-2.png", "uploaded-0.png"]],
        ),
        (
            ("input_image_0", "input_image_3"),
            {"input_image_0": (0,), "input_image_3": (0,)},
            1,
            ["uploaded-0.png", "uploaded-0.png"],
        ),
    ],
)
async def test_compiled_images_follow_explicit_bindings(
    monkeypatch: pytest.MonkeyPatch,
    slots: tuple[str, ...],
    bindings: dict[str, tuple[int, ...]],
    count: int,
    expected: list[Any],
) -> None:
    adapter = ComfyUIAdapter("http://comfy.test")

    async def upload_inputs(_request: MediaRequest) -> list[str]:
        return [f"uploaded-{index}.png" for index in range(count)]

    async def upload_mask(_request: MediaRequest) -> str | None:
        return None

    monkeypatch.setattr(adapter, "_upload_inputs", upload_inputs)
    monkeypatch.setattr(adapter, "_upload_mask", upload_mask)
    graph = graph_for(*slots)
    try:
        parameters = await adapter._request_parameters(request_for(graph, bindings, count))
        compiled = adapter._compile(graph, parameters)
        assert [node["inputs"]["image"] for node in compiled.values()] == expected
        assert "unselected.png" not in str(compiled)
    finally:
        await adapter.close()


@pytest.mark.parametrize(
    ("slots", "bindings", "count"),
    [
        (("input_image_0", "input_image_3"), {"input_image_0": (0,)}, 1),
        (("input_image",), {"input_image": (0,), "input_image_1": (1,)}, 2),
        (("input_image",), {"input_image": (0,)}, 2),
        (("input_image",), {"input_image": (2,)}, 2),
        (("input_image",), {"input_image": (-1,)}, 1),
        (("input_image",), {"input_image": (True,)}, 2),
        (("input_image",), {"input_image": ()}, 1),
        (("input_image",), {"input_image": (0, 1)}, 2),
        (("input_images",), {"input_images": ()}, 1),
        (("input_image_64",), {"input_image_64": (0,)}, 1),
        (("input_image_01",), {"input_image_01": (0,)}, 1),
    ],
)
async def test_incomplete_image_bindings_refuse_before_any_upload(
    monkeypatch: pytest.MonkeyPatch,
    slots: tuple[str, ...],
    bindings: dict[str, tuple[int, ...]],
    count: int,
) -> None:
    adapter = ComfyUIAdapter("http://comfy.test")
    calls: list[str] = []

    async def upload_inputs(_request: MediaRequest) -> list[str]:
        calls.append("inputs")
        return [f"uploaded-{index}.png" for index in range(count)]

    async def upload_mask(_request: MediaRequest) -> str | None:
        calls.append("mask")
        return None

    monkeypatch.setattr(adapter, "_upload_inputs", upload_inputs)
    monkeypatch.setattr(adapter, "_upload_mask", upload_mask)
    try:
        with pytest.raises(ValueError, match="image bindings"):
            await adapter._request_parameters(request_for(graph_for(*slots), bindings, count))
        assert calls == []
    finally:
        await adapter.close()


async def test_image_bindings_cannot_change_during_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ComfyUIAdapter("http://comfy.test")
    bindings: dict[str, tuple[int, ...]] = {"input_image_0": (1,), "input_image_1": (0,)}

    async def upload_inputs(_request: MediaRequest) -> list[str]:
        bindings["input_image_0"] = (0,)
        return ["uploaded-0.png", "uploaded-1.png"]

    async def upload_mask(_request: MediaRequest) -> str | None:
        return None

    monkeypatch.setattr(adapter, "_upload_inputs", upload_inputs)
    monkeypatch.setattr(adapter, "_upload_mask", upload_mask)
    graph = graph_for("input_image_0", "input_image_1")
    try:
        parameters = await adapter._request_parameters(request_for(graph, bindings))
        compiled = adapter._compile(graph, parameters)
        assert [node["inputs"]["image"] for node in compiled.values()] == [
            "uploaded-1.png",
            "uploaded-0.png",
        ]
    finally:
        await adapter.close()


async def test_unclassified_inputs_keep_the_existing_last_image_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ComfyUIAdapter("http://comfy.test")

    async def upload_inputs(_request: MediaRequest) -> list[str]:
        return ["uploaded-0.png", "uploaded-1.png"]

    async def upload_mask(_request: MediaRequest) -> str | None:
        return None

    monkeypatch.setattr(adapter, "_upload_inputs", upload_inputs)
    monkeypatch.setattr(adapter, "_upload_mask", upload_mask)
    graph = graph_for("input_image_0", "input_image_3")
    # This is the unchanged request shape used by older accepted turns.
    request = MediaRequest(
        run_id="garden-legacy",
        operation="image_to_image",
        prompt="Arrange the garden",
        negative_prompt=None,
        input_paths=[Path("first.png"), Path("second.png")],
        workflow=graph,
        parameters={},
    )
    try:
        parameters = await adapter._request_parameters(request)
        compiled = adapter._compile(graph, parameters)
        assert [node["inputs"]["image"] for node in compiled.values()] == [
            "uploaded-0.png",
            "uploaded-1.png",
        ]
    finally:
        await adapter.close()
