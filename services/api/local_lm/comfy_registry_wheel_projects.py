from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence

import httpx
from packaging.utils import canonicalize_name

from .comfy_registry_package_indexes import (
    MAX_SIMPLE_HTML_PAGE_BYTES,
    REMOTE_READ_BUDGET_EXCEEDED,
    SIMPLE_HTML_MEDIA_TYPES,
    UNREAD_WHEEL_KEY,
    ComfyRegistryPackageIndexError,
    RemoteReadBudget,
    project_index_url,
    read_remote_wheel_metadata,
    served_by_nvidia_index,
    simple_index_files,
    wheel_url_allowed,
)
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
                document, downloaded = await self._fetch_one(
                    name, MAX_REGISTRY_WHEEL_PROJECT_TOTAL_BYTES - total_bytes
                )
            except ComfyRegistryWheelProjectError:
                raise
            except ComfyRegistryPackageIndexError as exc:
                raise ComfyRegistryWheelProjectError(exc.code, str(exc)) from exc
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

    async def _fetch_one(self, name: str, remaining_bytes: int) -> tuple[object, int]:
        url = project_index_url(name)
        simple_html = served_by_nvidia_index(name)
        headers = {"accept": "text/html"} if simple_html else None
        maximum = MAX_SIMPLE_HTML_PAGE_BYTES if simple_html else MAX_REGISTRY_WHEEL_PROJECT_BYTES
        async with self._client.stream("GET", url, headers=headers) as response:
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
            if (
                _media_type(response) not in SIMPLE_HTML_MEDIA_TYPES
                if simple_html
                else _media_type(response) != _SIMPLE_JSON_MEDIA_TYPE
            ):
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
            expected = _content_length(response, maximum)
            body = bytearray()
            async for chunk in response.aiter_bytes(PROJECT_DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                body.extend(chunk)
                if len(body) > maximum:
                    raise ComfyRegistryWheelProjectError(
                        "project_metadata_too_large",
                        f"PyPI project metadata for {name} exceeds the size limit",
                    )
            if expected is not None and len(body) != expected:
                raise ComfyRegistryWheelProjectError(
                    "project_size_mismatch",
                    f"PyPI project metadata size for {name} does not match Content-Length",
                )
        if simple_html:
            return await self._simple_html_document(
                name, url, bytes(body), RemoteReadBudget(remaining_bytes - len(body))
            )
        return _document(name, bytes(body)), len(body)

    async def _simple_html_document(
        self,
        name: str,
        url: str,
        body: bytes,
        budget: RemoteReadBudget,
    ) -> tuple[object, int]:
        # Shape the page like the JSON index, reading each wheel's size and core
        # metadata from the wheel itself. A link outside this project's index is
        # never requested, and it and any wheel that cannot be read keep the reason
        # instead, which refuses resolution only if that is the wheel selected. A
        # transport failure still refuses the project: it is likely to recur for
        # every wheel, and the preparation can simply be retried.
        files: list[object] = []
        for item in simple_index_files(url, body):
            record: dict[str, object] = {
                "filename": item.filename,
                "url": item.url,
                "hashes": {} if item.sha256 is None else {"sha256": item.sha256},
                "yanked": item.yanked,
            }
            if item.requires_python is not None:
                record["requires-python"] = item.requires_python
            files.append(record)
            if not item.filename.endswith(".whl") or item.yanked:
                continue
            if not wheel_url_allowed(name, item.url, item.filename):
                record[UNREAD_WHEEL_KEY] = {
                    "code": "invalid_wheel_url",
                    "message": f"Package index link for {item.filename} is not allowlisted",
                }
                continue
            try:
                remote = await read_remote_wheel_metadata(
                    self._client, item.url, item.filename, budget
                )
            except ComfyRegistryPackageIndexError as exc:
                if exc.code == REMOTE_READ_BUDGET_EXCEEDED:
                    raise
                record[UNREAD_WHEEL_KEY] = {"code": exc.code, "message": str(exc)}
                continue
            record["size"] = remote.size_bytes
            record["core-metadata"] = {"sha256": hashlib.sha256(remote.metadata).hexdigest()}
        return {"meta": {"api-version": "1.0"}, "name": name, "files": files}, (
            len(body) + budget.spent
        )


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


def _content_length(response: httpx.Response, maximum: int) -> int | None:
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
    if size > maximum:
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
