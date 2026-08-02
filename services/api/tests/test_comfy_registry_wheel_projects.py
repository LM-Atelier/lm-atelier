from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import local_lm.comfy_registry_wheel_projects as project_module
from local_lm.comfy_registry_wheel_projects import (
    ComfyRegistryWheelProjectClient,
    ComfyRegistryWheelProjectError,
)

_MEDIA_TYPE = "application/vnd.pypi.simple.v1+json"


def _project(name: str) -> bytes:
    return json.dumps(
        {
            "meta": {"api-version": "1.4"},
            "name": name,
            "files": [],
        }
    ).encode()


def _response(
    status: int,
    *,
    content: bytes = b"",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    resolved = {"content-type": _MEDIA_TYPE}
    if headers:
        resolved.update(headers)
    return httpx.Response(status, stream=httpx.ByteStream(content), headers=resolved)


async def _fetch(
    names: Any,
    handler: Any,
) -> dict[str, object]:
    client = ComfyRegistryWheelProjectClient(transport=httpx.MockTransport(handler))
    try:
        return await client.fetch(names)
    finally:
        await client.close()


def _assert_error(code: str, raised: pytest.ExceptionInfo[ComfyRegistryWheelProjectError]) -> None:
    assert raised.value.code == code


async def test_fetches_canonical_projects_in_stable_order_with_strict_headers() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        name = request.url.path.split("/")[2]
        return _response(200, content=_project(name))

    documents = await _fetch(["zeta", "alpha-package"], handler)

    assert list(documents) == ["alpha-package", "zeta"]
    assert [str(request.url) for request in requests] == [
        "https://pypi.org/simple/alpha-package/",
        "https://pypi.org/simple/zeta/",
    ]
    assert all(request.headers["accept"] == _MEDIA_TYPE for request in requests)
    assert all(request.headers["accept-encoding"] == "identity" for request in requests)


async def test_empty_request_does_not_reach_the_network() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("empty project request must not reach the network")

    assert await _fetch([], handler) == {}


@pytest.mark.parametrize(
    ("names", "code"),
    [
        ("package", "invalid_project_names"),
        (["Not_Canonical"], "invalid_project_name"),
        (["bad/name"], "invalid_project_name"),
        (["package", "package"], "duplicate_project_name"),
    ],
)
async def test_invalid_project_name_sets_fail_before_network(names: Any, code: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid names must not reach the network")

    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(names, handler)
    _assert_error(code, raised)


async def test_project_count_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(project_module, "MAX_REGISTRY_WHEEL_PROJECTS", 1)

    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(
            ["alpha", "beta"],
            lambda _request: _response(200, content=_project("unused")),
        )
    _assert_error("invalid_project_names", raised)


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (302, "project_http_error"),
        (404, "project_not_found"),
        (500, "project_http_error"),
    ],
)
async def test_redirect_missing_and_error_responses_fail_closed(
    status: int,
    code: str,
) -> None:
    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(["package"], lambda _request: _response(status))
    _assert_error(code, raised)


async def test_rate_limit_reports_bounded_numeric_retry_delay() -> None:
    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(
            ["package"],
            lambda _request: _response(429, headers={"retry-after": "120"}),
        )

    _assert_error("project_rate_limited", raised)
    assert raised.value.retry_after_seconds == 120


async def test_transport_failure_uses_stable_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(["package"], handler)
    _assert_error("project_network_error", raised)
    assert "connection failed" not in str(raised.value)


@pytest.mark.parametrize(
    ("headers", "code"),
    [
        ({"content-type": "text/html"}, "invalid_project_content_type"),
        ({"content-encoding": "gzip"}, "encoded_project_metadata"),
        ({"content-length": "invalid"}, "invalid_project_content_length"),
        ({"content-length": "-1"}, "invalid_project_content_length"),
    ],
)
async def test_response_transport_contract_is_enforced(
    headers: dict[str, str],
    code: str,
) -> None:
    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(
            ["package"],
            lambda _request: _response(200, content=_project("package"), headers=headers),
        )
    _assert_error(code, raised)


async def test_declared_and_streamed_project_size_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project_module, "MAX_REGISTRY_WHEEL_PROJECT_BYTES", 10)
    body = _project("package")

    with pytest.raises(ComfyRegistryWheelProjectError) as declared:
        await _fetch(
            ["package"],
            lambda _request: _response(
                200,
                content=b"",
                headers={"content-length": str(len(body))},
            ),
        )
    _assert_error("project_metadata_too_large", declared)

    with pytest.raises(ComfyRegistryWheelProjectError) as streamed:
        await _fetch(
            ["package"],
            lambda _request: _response(
                200,
                content=body,
                headers={"content-length": "1"},
            ),
        )
    _assert_error("project_metadata_too_large", streamed)


async def test_mismatched_content_length_fails_closed() -> None:
    body = _project("package")
    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(
            ["package"],
            lambda _request: _response(
                200,
                content=body,
                headers={"content-length": str(len(body) + 1)},
            ),
        )
    _assert_error("project_size_mismatch", raised)


async def test_aggregate_project_size_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _project("alpha")
    monkeypatch.setattr(
        project_module,
        "MAX_REGISTRY_WHEEL_PROJECT_TOTAL_BYTES",
        len(body) * 2 - 1,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.split("/")[2]
        return _response(200, content=_project(name))

    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(["alpha", "bravo"], handler)
    _assert_error("project_metadata_total_too_large", raised)


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"{not-json", "invalid_project_json"),
        (b'{"name":"package","name":"other"}', "invalid_project_json"),
        (b"[]", "invalid_project_document"),
        (_project("other"), "project_identity_mismatch"),
        (b'{"name": 4}', "project_identity_mismatch"),
    ],
)
async def test_invalid_json_shape_duplicates_and_identity_fail_closed(
    body: bytes,
    code: str,
) -> None:
    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(["package"], lambda _request: _response(200, content=body))
    _assert_error(code, raised)
