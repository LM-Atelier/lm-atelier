"""Malformed picture purposes receive a fixed refusal when a record is imported."""

import pytest
from httpx2 import AsyncClient
from test_output_recipe_v1 import _payload

from local_lm.output_recipe_v1 import (
    OutputRecipeFormatError,
    canonical_bytes,
    open_output_recipe,
    record_digest,
)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("role", [[], {}], ids=["list", "object"])
async def test_a_picture_purpose_that_is_not_a_name_is_refused_without_an_internal_error(
    client: AsyncClient, version: int, role: object
) -> None:
    payload = _payload()
    payload["version"] = version
    payload["inputs"] = [
        {"sha256": "b" * 64, "role": role, "size_bytes": 10, "media_type": "image/png"}
    ]
    content = canonical_bytes({**payload, "digest": record_digest(payload)})
    with pytest.raises(OutputRecipeFormatError, match="unknown input role"):
        open_output_recipe(content)
    for endpoint in ("/api/output-recipes/check", "/api/output-recipes/replay-plan"):
        response = await client.post(
            endpoint, content=content, headers={"content-type": "application/octet-stream"}
        )
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "output-recipe-unreadable"
