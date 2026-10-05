"""Reference admission accounts for the actual numbered-slot fallback."""

from pathlib import Path

import pytest

from local_lm.adapters.base import MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.media_references import exceeds_capacity, reference_capacity


@pytest.mark.parametrize(
    ("indices", "first", "capacity"),
    [
        ((2,), False, 1),
        ((1, 2), False, 1),
        ((0, 2), False, 2),
        ((0, 1, 3), False, 3),
        ((3,), True, 2),
        ((1, 3), True, 3),
        ((0, 3), False, 2),
        ((0, 1, 2), False, 3),
        ((0, 1), True, 2),
    ],
)
async def test_reference_capacity_matches_every_consumed_attachment(
    monkeypatch: pytest.MonkeyPatch,
    indices: tuple[int, ...],
    first: bool,
    capacity: int,
) -> None:
    placeholders = [f"${{input_image_{index}}}" for index in indices]
    if first:
        placeholders.append("${input_image}")
    graph = {
        str(index): {"class_type": "LoadImage", "inputs": {"image": value}}
        for index, value in enumerate(placeholders)
    }
    adapter = ComfyUIAdapter("http://comfy.test")
    try:
        for supplied in range(1, max(indices) + 3):
            uploaded = [f"conditioning-{index}.png" for index in range(supplied)]

            async def upload_inputs(
                _request: MediaRequest, values: list[str] = uploaded
            ) -> list[str]:
                return values

            async def upload_mask(_request: MediaRequest) -> str | None:
                return None

            monkeypatch.setattr(adapter, "_upload_inputs", upload_inputs)
            monkeypatch.setattr(adapter, "_upload_mask", upload_mask)
            request = MediaRequest(
                run_id="conditioning-slots",
                operation="image_to_image",
                prompt="Combine the garden layouts",
                negative_prompt=None,
                input_paths=[Path(value) for value in uploaded],
                workflow=graph,
                parameters={},
            )
            parameters = await adapter._request_parameters(request)
            compiled = adapter._compile(graph, parameters)
            consumed = {node["inputs"]["image"] for node in compiled.values()}
            all_used = consumed == set(uploaded)
            assert all_used == (supplied <= capacity)
            admitted = exceeds_capacity(graph, supplied) is None
            assert admitted == all_used
            if not admitted:
                assert exceeds_capacity(graph, supplied) == capacity
        assert reference_capacity(graph) == capacity
    finally:
        await adapter.close()


@pytest.mark.parametrize("supplied", [1, 2, 3, 5])
def test_a_whole_list_remains_eligible_alongside_sparse_slots(supplied: int) -> None:
    graph = {
        "first": {"inputs": {"image": "${input_image_3}"}},
        "all": {"inputs": {"images": "${input_images}"}},
    }
    assert exceeds_capacity(graph, supplied) is None
