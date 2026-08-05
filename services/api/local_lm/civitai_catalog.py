from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import ValidationError

from .catalog_cache import CatalogCachePolicy, CatalogCacheStore
from .config import Settings
from .network import shared_tls_context
from .schemas import CatalogModel, CatalogPage, ContentRating

_ITEM_ID = re.compile(r"^[1-9][0-9]{0,11}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_EDIT_DECLARATION = re.compile(
    r"(?<![a-z0-9])(?:image[-_ ]?edit(?:ing)?|instruction[-_ ]?edit|inpaint(?:ing)?)(?![a-z0-9])",
    re.I,
)
_SAFE_MODEL_TYPES = {"Checkpoint", "LORA"}
_SORTS = {
    "trending": "Highest Rated",
    "downloads": "Most Downloaded",
    "likes": "Highest Rated",
    "newest": "Newest",
    "updated": "Newest",
    "compatible": "Most Downloaded",
}
_MAX_RETRY_AFTER_SECONDS = 30.0
_MAX_METADATA_VALUES = 128
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_GENERAL_LEVEL_MASK = 1 | 2
_MATURE_LEVEL_MASK = 4 | 8 | 16 | 32


class CivitaiCatalog:
    source_id = "civitai"
    display_name = "CivitAI"
    web_origin = "https://civitai.com"

    def __init__(
        self,
        settings: Settings,
        *,
        token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        cache_policy: CatalogCachePolicy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        headers = {
            "accept": "application/json",
            "user-agent": "local-lm/0.1",
        }
        if token:
            headers["authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            base_url="https://civitai.com",
            headers=headers,
            timeout=30,
            follow_redirects=False,
            verify=shared_tls_context(),
            transport=transport,
        )
        self._cache = CatalogCacheStore(
            settings.catalog_cache_dir / self.source_id,
            cache_policy,
        )
        self._request_lock = asyncio.Semaphore(1)
        self._sleep = sleep

    def set_token(self, token: str | None) -> None:
        if token:
            self._client.headers["authorization"] = f"Bearer {token}"
        else:
            self._client.headers.pop("authorization", None)

    @staticmethod
    def validate_item_id(item_id: str) -> bool:
        return bool(_ITEM_ID.fullmatch(item_id))

    async def search(
        self,
        *,
        query: str = "",
        role: str | None = None,
        sort: str = "trending",
        limit: int = 30,
        cursor: str | None = None,
        compatibility: str | None = None,
        file_format: str | None = None,
        quantization: str | None = None,
        license_id: str | None = None,
        gated: str | None = None,
        architecture: str | None = None,
        min_parameters: int | None = None,
        max_parameters: int | None = None,
        max_size_bytes: int | None = None,
        updated_within_days: int | None = None,
    ) -> CatalogPage:
        if role in {"chat", "video"}:
            return CatalogPage(items=[])
        params: dict[str, Any] = {
            "query": query or None,
            "sort": _SORTS.get(sort, _SORTS["trending"]),
            "limit": max(1, min(limit, 100)),
            "nsfw": "false",
            "primaryFileOnly": "true",
        }
        if role == "image":
            params["types"] = "Checkpoint"
        elif role == "lora":
            params["types"] = "LORA"
        url = self._validated_cursor(cursor) if cursor else "/api/v1/models"
        cache_path = self._cache_path(
            "search",
            url,
            None if cursor else params,
            role,
            compatibility,
            file_format,
            quantization,
            license_id,
            gated,
            architecture,
            min_parameters,
            max_parameters,
            max_size_bytes,
            updated_within_days,
        )
        cached = self._read_page_cache(
            cache_path,
            max_age_seconds=self._cache.policy.fresh_seconds,
        )
        if cached is not None:
            return cached.model_copy(update={"stale": False})
        try:
            payload = await self._request_json(url, params=None if cursor else params)
            items = [
                self._normalize(item, role, version=version)
                for item in self._items(payload)
                if self._is_general_item(item)
                for version in self._versions(item)
                if self._is_general_version(version)
            ]
            items = self._filter_items(
                items,
                compatibility=compatibility,
                file_format=file_format,
                quantization=quantization,
                license_id=license_id,
                gated=gated,
                architecture=architecture,
                min_parameters=min_parameters,
                max_parameters=max_parameters,
                max_size_bytes=max_size_bytes,
                updated_within_days=updated_within_days,
            )
            if sort == "compatible":
                rank = {"likely": 0, "advanced_import": 1, "unsupported": 2}
                items.sort(
                    key=lambda item: (
                        rank.get(item.compatibility, 3),
                        -(item.downloads or 0),
                    )
                )
            result = CatalogPage(
                items=items,
                next_cursor=self._next_cursor(payload),
            )
            self._cache.write_text(cache_path, result.model_dump_json())
            return result
        except (httpx.HTTPError, ValueError, ValidationError) as error:
            if not self._is_transient_error(error):
                raise
            stale = self._read_page_cache(
                cache_path,
                max_age_seconds=self._cache.policy.stale_seconds,
            )
            if stale is None:
                raise
            return stale.model_copy(update={"stale": True})

    async def inspect(
        self,
        item_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict[str, Any]:
        if not self.validate_item_id(item_id):
            raise ValueError("CivitAI item id must be a positive decimal integer")
        if revision != "main" and not self.validate_item_id(revision):
            raise ValueError("CivitAI revision must be main or a positive version id")
        cache_path = self._cache_path("detail", item_id, revision, requested_role)
        cached = self._read_detail_cache(
            cache_path,
            max_age_seconds=self._cache.policy.fresh_seconds,
        )
        if cached is not None:
            return cached
        try:
            if revision in {"main", item_id}:
                version = await self._request_json(f"/api/v1/model-versions/{item_id}")
                if str(version.get("id") or "") != item_id:
                    raise ValueError("CivitAI returned a different model version")
                version_model = self._mapping(version.get("model"))
                if version_model.get("nsfw") is not False or not self._is_general_version(version):
                    raise ValueError("CivitAI item is not available in the general catalog")
                model_id = str(version.get("modelId") or "")
                if not self.validate_item_id(model_id):
                    raise ValueError("CivitAI model version has no valid parent model")
                payload = await self._request_json(f"/api/v1/models/{model_id}")
            else:
                payload = await self._request_json(f"/api/v1/models/{item_id}")
                version = self._select_version(payload, revision)
            if not self._is_general_item(payload) or not self._is_general_version(version):
                raise ValueError("CivitAI item is not available in the general catalog")
            resolved_revision = str(version.get("id") or "")
            detail = {
                "model": self._normalize(payload, requested_role, version=version).model_dump(
                    mode="json"
                ),
                "revision": resolved_revision,
                "files": self._normalize_files(payload, version),
            }
            self._cache.write_text(
                cache_path,
                json.dumps(detail, sort_keys=True, separators=(",", ":")),
            )
            return detail
        except (httpx.HTTPError, ValueError) as error:
            if not self._is_transient_error(error):
                raise
            stale = self._read_detail_cache(
                cache_path,
                max_age_seconds=self._cache.policy.stale_seconds,
            )
            if stale is None:
                raise
            return stale

    async def inspect_file_prefix(
        self,
        item_id: str,
        revision: str,
        filename: str,
        *,
        max_bytes: int,
    ) -> bytes:
        raise NotImplementedError(
            "CivitAI file inspection is unavailable before verified transfer support"
        )

    async def versions(self, model_id: str) -> dict[str, Any]:
        """General-audience versions of one model, in the provider's order.

        Light summaries for staleness comparison only; installing a candidate
        goes back through `inspect` and the full verified pipeline, so nothing
        here needs files or compatibility. Mature and unrated versions are
        absent, not marked - the general catalog never names what it excludes.
        """
        if not self.validate_item_id(model_id):
            raise ValueError("CivitAI model id must be a positive decimal integer")
        cache_path = self._cache_path("versions", model_id)
        cached = self._read_detail_cache(
            cache_path,
            max_age_seconds=self._cache.policy.fresh_seconds,
        )
        if cached is not None:
            return cached
        try:
            payload = await self._request_json(f"/api/v1/models/{model_id}")
            if not self._is_general_item(payload):
                raise ValueError("CivitAI item is not available in the general catalog")
            summary = {
                "model_id": str(payload.get("id") or ""),
                "model_name": str(payload.get("name") or "") or None,
                "versions": [
                    {
                        "version_id": str(version.get("id") or ""),
                        "version_name": str(version.get("name") or "") or None,
                        "published_at": str(version.get("publishedAt") or "") or None,
                        "base_model": str(version.get("baseModel") or "") or None,
                        "changelog": str(version.get("description") or "")[:4096] or None,
                        # The chooser has to say what a version costs before
                        # someone picks it. The payload already carries the
                        # files, so this needs no extra call.
                        "size_bytes": sum(
                            self._size_bytes(file.get("sizeKB"))
                            for file in version.get("files") or []
                            if isinstance(file, dict)
                        ),
                    }
                    for version in self._versions(payload)
                    if self._is_general_version(version)
                ],
            }
            self._cache.write_text(
                cache_path,
                json.dumps(summary, sort_keys=True, separators=(",", ":")),
            )
            return summary
        except (httpx.HTTPError, ValueError) as error:
            if not self._is_transient_error(error):
                raise
            stale = self._read_detail_cache(
                cache_path,
                max_age_seconds=self._cache.policy.stale_seconds,
            )
            if stale is None:
                raise
            return stale

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async with self._request_lock:
            for attempt in range(2):
                retry_delay: float | None = None
                async with self._client.stream("GET", url, params=params) as response:
                    if response.status_code == 429 and attempt == 0:
                        delay = self._retry_after_seconds(response)
                        if delay is not None and delay <= _MAX_RETRY_AFTER_SECONDS:
                            retry_delay = delay
                        else:
                            response.raise_for_status()
                    else:
                        response.raise_for_status()
                    if retry_delay is None:
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > _MAX_RESPONSE_BYTES:
                                raise ValueError("CivitAI response exceeds the size limit")
                        payload = json.loads(body)
                if retry_delay is not None:
                    await self._sleep(retry_delay)
                    continue
                if not isinstance(payload, dict):
                    raise ValueError("CivitAI returned a non-object response")
                return payload
        raise RuntimeError("CivitAI request retry loop exhausted")

    @classmethod
    def _normalize(
        cls,
        item: dict[str, Any],
        requested_role: str | None,
        *,
        version: dict[str, Any] | None = None,
    ) -> CatalogModel:
        if version is None:
            raise ValueError("CivitAI catalog cards require an explicit model version")
        selected = version
        files = [value for value in selected.get("files") or [] if isinstance(value, dict)]
        filenames = [str(value.get("name") or "") for value in files]
        formats = sorted(
            {
                Path(filename).suffix.lower().lstrip(".")
                for filename in filenames
                if Path(filename).suffix
            }
        )
        tags = cls._string_list(item.get("tags"))
        base_models = cls._string_list(item.get("baseModels"))
        base_model = str(selected.get("baseModel") or "") or next(iter(base_models), "")
        if base_model and base_model not in tags:
            tags.append(base_model)
        model_type = str(item.get("type") or "")
        if model_type and model_type not in tags:
            tags.append(model_type)
        compatibility, reasons = cls._compatibility(
            model_type=model_type,
            files=files,
            requested_role=requested_role,
        )
        sizes = [cls._size_bytes(value.get("sizeKB")) for value in files]
        creator = cls._mapping(item.get("creator"))
        stats = cls._mapping(selected.get("stats")) or cls._mapping(item.get("stats"))
        published = cls._datetime(selected.get("publishedAt"))
        updated = cls._datetime(selected.get("updatedAt")) or published
        version_id = str(selected.get("id") or "")
        display_id = version_id or "model version"
        model_name = str(item.get("name") or f"CivitAI {display_id}")
        version_name = str(selected.get("name") or "").strip()
        name = (
            f"{model_name} - {version_name}"
            if version_name and version_name != model_name
            else model_name
        )
        return CatalogModel(
            provider=cls.source_id,
            remote_id=version_id,
            name=name,
            # The card is a version; this is the model it belongs to. Carrying
            # both keeps the installable identity exact while letting the
            # library list versions under one parent.
            parent_model_id=str(item.get("id") or "") or None,
            parent_model_name=model_name,
            author=str(creator.get("username") or "") or None,
            pipeline_tag="text-to-image" if model_type == "Checkpoint" else "lora",
            tags=tags,
            downloads=cls._integer(stats.get("downloadCount")),
            likes=cls._integer(stats.get("thumbsUpCount") or stats.get("favoriteCount")),
            created_at=published,
            last_modified=updated,
            gated=str(selected.get("availability") or item.get("availability") or "").lower()
            not in {"", "public"},
            private=False,
            architecture=base_model or None,
            formats=formats,
            quantizations=cls._quantizations(files),
            license_id=None,
            total_size_bytes=sum(sizes) or None,
            compatibility=compatibility,
            compatibility_reasons=reasons,
            required_runtime=("comfyui" if requested_role in {None, "image", "lora"} else None),
            content_rating=cls._content_rating(selected),
        )

    @classmethod
    def _normalize_files(
        cls,
        item: dict[str, Any],
        version: dict[str, Any],
    ) -> list[dict[str, Any]]:
        trained_words = cls._string_list(version.get("trainedWords"))
        tags = cls._string_list(item.get("tags"))
        base_models = cls._string_list(item.get("baseModels"))
        base_model = str(version.get("baseModel") or "") or next(iter(base_models), "")
        declared_values = [
            *tags,
            *base_models,
            *trained_words,
            base_model,
            str(item.get("name") or ""),
            str(version.get("name") or ""),
            str(item.get("description") or "")[:4096],
            str(version.get("description") or "")[:4096],
        ]
        metadata = {
            "provider": cls.source_id,
            "source_model_id": str(item.get("id") or ""),
            "source_version_id": str(version.get("id") or ""),
            "version_name": str(version.get("name") or "") or None,
            "published_at": str(version.get("publishedAt") or "") or None,
            "model_type": str(item.get("type") or ""),
            "base_model": base_model or None,
            "base_models": base_models,
            "tags": tags,
            "trained_words": trained_words,
            "edit_tailored": (
                "declared"
                if any(_EDIT_DECLARATION.search(value) for value in declared_values)
                else "unknown"
            ),
            "content_rating": cls._content_rating(version),
            "permissions": {
                "allow_commercial_use": item.get("allowCommercialUse"),
                "allow_derivatives": item.get("allowDerivatives"),
                "allow_different_license": item.get("allowDifferentLicense"),
                "allow_no_credit": item.get("allowNoCredit"),
            },
        }
        result: list[dict[str, Any]] = []
        for value in version.get("files") or []:
            if not isinstance(value, dict):
                continue
            hashes = cls._mapping(value.get("hashes"))
            file_metadata = cls._mapping(value.get("metadata"))
            sha256 = str(hashes.get("SHA256") or hashes.get("sha256") or "")
            result.append(
                {
                    "filename": str(value.get("name") or ""),
                    "size": cls._size_bytes(value.get("sizeKB")),
                    "sha256": sha256.lower() if _SHA256.fullmatch(sha256) else None,
                    "source_file_id": str(value.get("id") or ""),
                    "source_version_id": str(version.get("id") or ""),
                    "source_file_type": str(value.get("type") or ""),
                    "source_file_precision": str(file_metadata.get("fp") or ""),
                    "format": str(file_metadata.get("format") or ""),
                    "pickle_scan_result": value.get("pickleScanResult"),
                    "virus_scan_result": value.get("virusScanResult"),
                    "metadata": metadata,
                }
            )
        return result

    @staticmethod
    def _compatibility(
        *,
        model_type: str,
        files: list[dict[str, Any]],
        requested_role: str | None,
    ) -> tuple[str, list[str]]:
        if requested_role not in {None, "image", "lora"}:
            return "unsupported", ["CivitAI general catalog supports image assets only"]
        if requested_role == "image" and model_type != "Checkpoint":
            return "unsupported", ["CivitAI image search requires a checkpoint"]
        if requested_role == "lora" and model_type != "LORA":
            return "unsupported", ["CivitAI LoRA search requires a LoRA asset"]
        if model_type not in _SAFE_MODEL_TYPES:
            model_label = model_type or "asset"
            return "unsupported", [f"CivitAI {model_label} is not supported"]
        safe_files = [
            value
            for value in files
            if str(value.get("name") or "").lower().endswith(".safetensors")
            and str(value.get("pickleScanResult") or "").lower() == "success"
            and str(value.get("virusScanResult") or "").lower() == "success"
            and _SHA256.fullmatch(str((value.get("hashes") or {}).get("SHA256") or ""))
        ]
        if not safe_files:
            return "unsupported", ["no scan-cleared, hash-pinned safetensors file"]
        if model_type == "LORA":
            return "advanced_import", ["requires verified CivitAI LoRA installation"]
        return "advanced_import", ["requires verified CivitAI checkpoint installation"]

    @staticmethod
    def _filter_items(
        items: list[CatalogModel],
        *,
        compatibility: str | None,
        file_format: str | None,
        quantization: str | None,
        license_id: str | None,
        gated: str | None,
        architecture: str | None,
        min_parameters: int | None,
        max_parameters: int | None,
        max_size_bytes: int | None,
        updated_within_days: int | None,
    ) -> list[CatalogModel]:
        updated_after = (
            datetime.now(UTC).timestamp() - updated_within_days * 86400
            if updated_within_days is not None
            else None
        )
        return [
            item
            for item in items
            if (not compatibility or item.compatibility == compatibility)
            and (not file_format or file_format.lower() in item.formats)
            and (not quantization or quantization.lower().replace("-", "_") in item.quantizations)
            and (not license_id or (item.license_id or "").lower() == license_id.lower())
            and (gated not in {"gated", "open"} or bool(item.gated) == (gated == "gated"))
            and (not architecture or architecture.lower() in (item.architecture or "").lower())
            and min_parameters is None
            and max_parameters is None
            and (
                max_size_bytes is None
                or (item.total_size_bytes is not None and item.total_size_bytes <= max_size_bytes)
            )
            and (
                updated_after is None
                or (
                    item.last_modified is not None
                    and item.last_modified.timestamp() >= updated_after
                )
            )
        ]

    @staticmethod
    def _is_general_item(item: dict[str, Any]) -> bool:
        return item.get("nsfw") is False and CivitaiCatalog.validate_item_id(
            str(item.get("id") or "")
        )

    @classmethod
    def _is_general_version(cls, version: dict[str, Any]) -> bool:
        return cls.validate_item_id(str(version.get("id") or "")) and (
            cls._content_rating(version) == "general"
        )

    @staticmethod
    def _content_rating(version: dict[str, Any]) -> ContentRating:
        raw_level = version.get("nsfwLevel")
        if isinstance(raw_level, bool) or not isinstance(raw_level, (int, str)):
            return "unknown"
        try:
            level = int(raw_level)
        except (TypeError, ValueError, OverflowError):
            return "unknown"
        if level <= 0 or level & ~(_GENERAL_LEVEL_MASK | _MATURE_LEVEL_MASK):
            return "unknown"
        if level & _MATURE_LEVEL_MASK:
            return "mature"
        if level & _GENERAL_LEVEL_MASK:
            return "general"
        return "unknown"

    @staticmethod
    def _items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [value for value in payload.get("items") or [] if isinstance(value, dict)]

    @staticmethod
    def _versions(item: dict[str, Any]) -> list[dict[str, Any]]:
        return [value for value in item.get("modelVersions") or [] if isinstance(value, dict)]

    @classmethod
    def _select_version(
        cls,
        item: dict[str, Any],
        revision: str,
    ) -> dict[str, Any]:
        if revision == "main":
            raise ValueError("CivitAI requires an explicit model version")
        selected = next(
            (value for value in cls._versions(item) if str(value.get("id") or "") == revision),
            None,
        )
        if selected is None:
            raise ValueError("CivitAI model version was not found")
        return selected

    def _next_cursor(self, payload: dict[str, Any]) -> str | None:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return None
        candidate = str(metadata.get("nextPage") or "")
        if not candidate:
            return None
        try:
            return self._validated_cursor(candidate)
        except ValueError:
            return None

    @staticmethod
    def _validated_cursor(cursor: str | None) -> str:
        if not cursor or len(cursor) > 2048:
            raise ValueError("CivitAI catalog cursor is invalid")
        parsed = urlparse(cursor)
        query = parse_qs(parsed.query, keep_blank_values=True)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "civitai.com"
            or parsed.port not in {None, 443}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != "/api/v1/models"
            or parsed.fragment
            or query.get("nsfw") != ["false"]
            or query.get("primaryFileOnly") != ["true"]
        ):
            raise ValueError("CivitAI catalog cursor is invalid")
        return cursor

    def _cache_path(self, *parts: Any) -> Path:
        payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
        return self._cache.path(hashlib.sha256(payload.encode()).hexdigest())

    def _read_page_cache(
        self,
        path: Path,
        *,
        max_age_seconds: float,
    ) -> CatalogPage | None:
        text = self._cache.read_text(path, max_age_seconds=max_age_seconds)
        if text is None:
            return None
        try:
            return CatalogPage.model_validate_json(text)
        except ValidationError:
            return None

    def _read_detail_cache(
        self,
        path: Path,
        *,
        max_age_seconds: float,
    ) -> dict[str, Any] | None:
        text = self._cache.read_text(path, max_age_seconds=max_age_seconds)
        if text is None:
            return None
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        value = response.headers.get("retry-after", "").strip()
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                target = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if target.tzinfo is None:
                target = target.replace(tzinfo=UTC)
            return max(0.0, (target - datetime.now(UTC)).total_seconds())

    @staticmethod
    def _is_transient_error(error: Exception) -> bool:
        if isinstance(error, httpx.RequestError):
            return True
        return isinstance(error, httpx.HTTPStatusError) and (
            error.response.status_code == 429 or error.response.status_code >= 500
        )

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            text for text in (str(item).strip() for item in value[:_MAX_METADATA_VALUES]) if text
        ]

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _size_bytes(value: Any) -> int:
        try:
            return max(0, int(float(value) * 1024))
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _integer(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    @staticmethod
    def _quantizations(files: list[dict[str, Any]]) -> list[str]:
        values: set[str] = set()
        for value in files:
            metadata = value.get("metadata")
            if not isinstance(metadata, dict):
                continue
            for key in ("fp", "precision"):
                item = str(metadata.get(key) or "").strip().lower().replace("-", "_")
                if item:
                    values.add(item)
        return sorted(values)
