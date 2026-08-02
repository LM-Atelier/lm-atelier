from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

import httpx
from packaging.utils import canonicalize_name

from .comfy_registry_wheel_artifacts import MAX_PYPI_PROJECT_DOCUMENT_BYTES
from .network import shared_tls_context

MAX_REGISTRY_WHEEL_PROJECTS = 256
MAX_REGISTRY_WHEEL_PROJECT_BYTES = MAX_PYPI_PROJECT_DOCUMENT_BYTES
MAX_REGISTRY_WHEEL_PROJECT_TOTAL_BYTES = 64 * 1024 * 1024
PROJECT_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_PROJECT_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\Z")
_SIMPLE_JSON_MEDIA_TYPE = "application/vnd.pypi.simple.v1+json"


class ComfyRegistryWheelProjectError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retry_after_seconds = retry_after_seconds


class _DuplicateJsonKey(ValueError):
    pass


class ComfyRegistryWheelProjectClient:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(
            headers={
                "accept": _SIMPLE_JSON_MEDIA_TYPE,
                "accept-encoding": "identity",
                "user-agent": "local-lm/0.1",
            },
            timeout=httpx.Timeout(30, read=120),
            follow_redirects=False,
            verify=shared_tls_context(),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch(
        self,
        project_names: Sequence[str],
    ) -> dict[str, object]:
        names = _project_names(project_names)
        documents: dict[str, object] = {}
        total_bytes = 0
        for name in names:
            try:
                document, downloaded = await self._fetch_one(name)
            except ComfyRegistryWheelProjectError:
                raise
            except httpx.HTTPError as exc:
                raise ComfyRegistryWheelProjectError(
                    "project_network_error",
                    f"PyPI project metadata for {name} could not be retrieved",
                ) from exc
            total_bytes += downloaded
            if total_bytes > MAX_REGISTRY_WHEEL_PROJECT_TOTAL_BYTES:
                raise ComfyRegistryWheelProjectError(
                    "project_metadata_total_too_large",
                    "PyPI project metadata exceeds the aggregate size limit",
                )
            documents[name] = document
        return documents

    async def _fetch_one(self, name: str) -> tuple[object, int]:
        url = f"https://pypi.org/simple/{name}/"
        async with self._client.stream("GET", url) as response:
            if response.status_code == 404:
                raise ComfyRegistryWheelProjectError(
                    "project_not_found", f"PyPI project {name} does not exist"
                )
            if response.status_code == 429:
                raise ComfyRegistryWheelProjectError(
                    "project_rate_limited",
                    "PyPI project metadata is temporarily rate limited",
                    retry_after_seconds=_retry_after(response),
                )
            if response.status_code != 200:
                raise ComfyRegistryWheelProjectError(
                    "project_http_error",
                    f"PyPI project metadata returned HTTP {response.status_code}",
                )
            if _media_type(response) != _SIMPLE_JSON_MEDIA_TYPE:
                raise ComfyRegistryWheelProjectError(
                    "invalid_project_content_type",
                    "PyPI project metadata used an unsupported content type",
                )
            encoding = response.headers.get("content-encoding", "identity").strip().lower()
            if encoding not in {"", "identity"}:
                raise ComfyRegistryWheelProjectError(
                    "encoded_project_metadata",
                    "PyPI project metadata used unsupported content encoding",
                )
            expected = _content_length(response)
            body = bytearray()
            async for chunk in response.aiter_bytes(PROJECT_DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > MAX_REGISTRY_WHEEL_PROJECT_BYTES:
                    raise ComfyRegistryWheelProjectError(
                        "project_metadata_too_large",
                        f"PyPI project metadata for {name} exceeds the size limit",
                    )
            if expected is not None and len(body) != expected:
                raise ComfyRegistryWheelProjectError(
                    "project_size_mismatch",
                    f"PyPI project metadata size for {name} does not match Content-Length",
                )
        return _document(name, bytes(body)), len(body)


def _project_names(value: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) > MAX_REGISTRY_WHEEL_PROJECTS
    ):
        raise ComfyRegistryWheelProjectError(
            "invalid_project_names", "PyPI project names must be a bounded array"
        )
    names: set[str] = set()
    for item in value:
        if (
            not isinstance(item, str)
            or len(item) > 200
            or canonicalize_name(item) != item
            or _PROJECT_NAME.fullmatch(item) is None
        ):
            raise ComfyRegistryWheelProjectError(
                "invalid_project_name", "PyPI project name is not canonical"
            )
        if item in names:
            raise ComfyRegistryWheelProjectError(
                "duplicate_project_name", f"PyPI project {item} was requested more than once"
            )
        names.add(item)
    return tuple(sorted(names))


def _media_type(response: httpx.Response) -> str:
    raw: str = response.headers.get("content-type", "")
    return raw.split(";", 1)[0].strip().lower()


def _content_length(response: httpx.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None:
        return None
    try:
        size = int(raw)
    except ValueError as exc:
        raise ComfyRegistryWheelProjectError(
            "invalid_project_content_length",
            "PyPI project metadata returned an invalid Content-Length",
        ) from exc
    if size < 0:
        raise ComfyRegistryWheelProjectError(
            "invalid_project_content_length",
            "PyPI project metadata returned an invalid Content-Length",
        )
    if size > MAX_REGISTRY_WHEEL_PROJECT_BYTES:
        raise ComfyRegistryWheelProjectError(
            "project_metadata_too_large",
            "PyPI project metadata exceeds the size limit",
        )
    return size


def _retry_after(response: httpx.Response) -> int | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = int(raw)
    except ValueError:
        return None
    return seconds if 0 <= seconds <= 24 * 60 * 60 else None


def _document(name: str, body: bytes) -> object:
    try:
        text = body.decode("utf-8")
        document = json.loads(text, object_pairs_hook=_unique_object)
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKey) as exc:
        raise ComfyRegistryWheelProjectError(
            "invalid_project_json", f"PyPI project metadata for {name} is invalid JSON"
        ) from exc
    if not isinstance(document, dict):
        raise ComfyRegistryWheelProjectError(
            "invalid_project_document", f"PyPI project metadata for {name} is not an object"
        )
    project_name = document.get("name")
    if not isinstance(project_name, str) or canonicalize_name(project_name) != name:
        raise ComfyRegistryWheelProjectError(
            "project_identity_mismatch",
            f"PyPI project metadata identity does not match {name}",
        )
    return document


def _unique_object(pairs: list[tuple[str, object]]) -> Mapping[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result
