from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, urlparse

import httpx

from .comfy_workflow_packages import WorkflowPackageRequirement
from .network import shared_tls_context

MAX_REGISTRY_PACKAGES = 64
MAX_REGISTRY_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REGISTRY_DEPENDENCIES = 256
MAX_REGISTRY_TEXT_CHARACTERS = 1_000
MAX_REGISTRY_RETRY_AFTER_SECONDS = 30.0

_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_PACKAGE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")

InstallKind = Literal["already_installed", "git_commit", "registry_archive"]


@dataclass(frozen=True)
class ComfyNodeResolution:
    package_id: str
    declared_version: str | None
    node_types: tuple[str, ...]
    install_kind: InstallKind | None = None
    repository_url: str | None = None
    registry_record_id: str | None = None
    download_url: str | None = None
    pip_dependencies: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error_code: str | None = None

    @property
    def resolved(self) -> bool:
        return self.error_code is None


@dataclass(frozen=True)
class ComfyRegistryResolution:
    packages: tuple[ComfyNodeResolution, ...]

    @property
    def metadata_resolved(self) -> bool:
        return all(package.resolved for package in self.packages)


class ComfyRegistryClient:
    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url="https://api.comfy.org",
            headers={
                "accept": "application/json",
                "user-agent": "local-lm/0.1",
            },
            timeout=30,
            follow_redirects=False,
            verify=shared_tls_context(),
            transport=transport,
        )
        self._request_lock = asyncio.Semaphore(1)
        self._sleep = sleep

    async def close(self) -> None:
        await self._client.aclose()

    async def resolve(
        self,
        requirements: Sequence[WorkflowPackageRequirement],
    ) -> ComfyRegistryResolution:
        if len(requirements) > MAX_REGISTRY_PACKAGES:
            raise ValueError("workflow requires too many custom node packages")
        packages = []
        for requirement in requirements:
            packages.append(await self._resolve(requirement))
        return ComfyRegistryResolution(tuple(packages))

    async def _resolve(
        self,
        requirement: WorkflowPackageRequirement,
    ) -> ComfyNodeResolution:
        package_id = requirement.package_id
        if not _PACKAGE_ID.fullmatch(package_id):
            return self._error(requirement, "invalid_package_id")
        if len(requirement.versions) != 1:
            code = (
                "unversioned_custom_node_package"
                if not requirement.versions
                else "conflicting_custom_node_versions"
            )
            return self._error(requirement, code)
        declared_version = requirement.versions[0]
        is_commit = _COMMIT.fullmatch(declared_version) is not None
        is_registry_version = _SEMANTIC_VERSION.fullmatch(declared_version) is not None
        if not is_commit and not is_registry_version:
            return self._error(requirement, "unsupported_package_version")
        if requirement.locally_resolved:
            return ComfyNodeResolution(
                package_id,
                declared_version.lower() if is_commit else declared_version,
                requirement.node_types,
                install_kind="already_installed",
            )
        node = await self._node(package_id)
        if node is None:
            return self._error(requirement, "registry_package_not_found")
        status = node.get("status")
        if status not in (None, "", "NodeStatusActive"):
            return self._error(requirement, "registry_package_inactive")
        try:
            repository = _github_repository(node.get("repository"))
        except ValueError:
            return self._error(requirement, "unsupported_package_repository")

        if is_commit:
            return ComfyNodeResolution(
                package_id,
                declared_version.lower(),
                requirement.node_types,
                install_kind="git_commit",
                repository_url=repository,
                warnings=(
                    "source_review_required",
                    "dependency_manifest_review_required",
                ),
            )
        version = await self._version(package_id, declared_version)
        if version is None:
            return self._error(requirement, "registry_version_not_found")
        if version.get("node_id") != package_id or version.get("version") != declared_version:
            return self._error(requirement, "registry_identity_mismatch")
        if version.get("status") != "NodeVersionStatusActive":
            return self._error(requirement, "registry_version_inactive")
        if _sequence(version.get("tags_admin"), "registry admin tags"):
            return self._error(requirement, "registry_security_warning")
        try:
            record_id = _text(version.get("id"), "registry version id")
            download_url = _download_url(version.get("downloadUrl"))
            dependencies = _dependencies(version.get("dependencies"))
        except ValueError:
            return self._error(requirement, "invalid_registry_metadata")
        warnings = ("deprecated_version",) if version.get("deprecated") is True else ()
        return ComfyNodeResolution(
            package_id,
            declared_version,
            requirement.node_types,
            install_kind="registry_archive",
            repository_url=repository,
            registry_record_id=record_id,
            download_url=download_url,
            pip_dependencies=dependencies,
            warnings=warnings,
        )

    async def _node(self, package_id: str) -> Mapping[str, object] | None:
        payload = await self._request_json(
            "/nodes",
            params={"node_id": package_id, "limit": 2},
        )
        if not isinstance(payload, Mapping):
            raise ValueError("Comfy Registry returned a non-object node response")
        values = payload.get("nodes")
        if not isinstance(values, list):
            raise ValueError("Comfy Registry returned an invalid node collection")
        matches = [
            value
            for value in values
            if isinstance(value, Mapping) and value.get("id") == package_id
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("Comfy Registry returned duplicate package identities")
        return matches[0]

    async def _version(
        self,
        package_id: str,
        version: str,
    ) -> Mapping[str, object] | None:
        try:
            payload = await self._request_json(
                f"/nodes/{quote(package_id, safe='')}/versions/{quote(version, safe='')}"
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        if not isinstance(payload, Mapping):
            raise ValueError("Comfy Registry returned a non-object version response")
        return payload

    async def _request_json(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> object:
        async with self._request_lock:
            for attempt in range(2):
                retry_delay: float | None = None
                async with self._client.stream("GET", url, params=params) as response:
                    if response.status_code == 429 and attempt == 0:
                        retry_delay = _retry_after(response)
                        if retry_delay is None:
                            response.raise_for_status()
                    else:
                        response.raise_for_status()
                    if retry_delay is None:
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > MAX_REGISTRY_RESPONSE_BYTES:
                                raise ValueError("Comfy Registry response exceeds the size limit")
                        return json.loads(body)
                await self._sleep(retry_delay)
        raise RuntimeError("Comfy Registry request retry loop exhausted")

    @staticmethod
    def _error(
        requirement: WorkflowPackageRequirement,
        code: str,
    ) -> ComfyNodeResolution:
        version = requirement.versions[0] if len(requirement.versions) == 1 else None
        return ComfyNodeResolution(
            requirement.package_id,
            version,
            requirement.node_types,
            error_code=code,
        )


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "").strip()
    try:
        seconds = float(value)
    except ValueError:
        return None
    if 0 <= seconds <= MAX_REGISTRY_RETRY_AFTER_SECONDS:
        return seconds
    return None


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_REGISTRY_TEXT_CHARACTERS
        or any(character < " " or character == "" for character in value)
    ):
        raise ValueError(f"invalid {name}")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError(f"invalid {name}")
    return value


def _dependencies(value: object) -> tuple[str, ...]:
    values = _sequence(value, "registry dependencies")
    if len(values) > MAX_REGISTRY_DEPENDENCIES:
        raise ValueError("too many registry dependencies")
    return tuple(_text(item, "registry dependency") for item in values)


def _github_repository(value: object) -> str:
    source = _text(value, "package repository")
    parsed = urlparse(source)
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or len(parts) != 2
    ):
        raise ValueError("unsupported package repository")
    repository = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", parts[0]) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", repository
    ):
        raise ValueError("invalid package repository")
    return f"https://github.com/{parts[0]}/{repository}.git"


def _download_url(value: object) -> str:
    url = _text(value, "registry download URL")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "cdn.comfy.org"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise ValueError("untrusted registry download URL")
    return url
