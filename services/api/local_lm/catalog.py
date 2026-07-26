from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import quote, urlparse

import httpx

from .config import Settings
from .domain import CompatibilityLevel
from .schemas import CatalogModel, CatalogPage

SORTS = {
    "trending": "trendingScore",
    "downloads": "downloads",
    "likes": "likes",
    "newest": "createdAt",
    "updated": "lastModified",
}

_QUANTIZATION = re.compile(r"^(?:q\d(?:_[a-z0-9]+)*|i?q\d(?:_[a-z0-9]+)*|fp\d+|bf16)$", re.I)
_PARAMETERS = re.compile(r"(?:^|[-_ ])(\d+(?:\.\d+)?)\s*([bmk])(?:$|[-_ ])", re.I)
_REMOTE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CACHE_VERSION = 3


class HuggingFaceCatalog:
    def __init__(self, settings: Settings) -> None:
        headers = {"user-agent": "local-lm/0.1"}
        if settings.hf_token:
            headers["authorization"] = f"Bearer {settings.hf_token}"
        self._client = httpx.AsyncClient(
            base_url="https://huggingface.co", headers=headers, timeout=30, follow_redirects=True
        )
        self._cache_dir = settings.catalog_cache_dir

    def set_token(self, token: str | None) -> None:
        if token:
            self._client.headers["authorization"] = f"Bearer {token}"
        else:
            self._client.headers.pop("authorization", None)

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
        params: dict[str, Any] = {
            "search": query or None,
            "sort": SORTS.get(sort, SORTS["trending"]),
            "direction": -1,
            "limit": max(1, min(limit, 100)),
            "full": "true",
            "config": "true",
        }
        if role:
            params["pipeline_tag"] = self._pipeline_tag(role)
        url = self._validated_cursor(cursor) if cursor else "/api/models"
        cache = self._cache_path(
            "search",
            query,
            role,
            sort,
            limit,
            cursor,
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
        fallback_cache = self._cache_path(
            "search-fallback",
            query,
            role,
            sort,
            limit,
        )
        try:
            response = await self._client.get(url, params=None if cursor else params)
            response.raise_for_status()
            payload = response.json()
            raw_items = [
                item
                for item in payload
                if isinstance(item, dict)
                and self._valid_remote_id(str(item.get("id") or item.get("modelId") or ""))
            ]
            if max_size_bytes is not None:
                raw_items = await self._hydrate_file_sizes(raw_items, role)
            items = [self._normalize(item, role) for item in raw_items]
            next_cursor = self._next_link(response)
            if cursor is None:
                self._write_cache(
                    fallback_cache,
                    CatalogPage(items=items, next_cursor=next_cursor).model_dump_json(),
                )
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
                rank = {"tested": 0, "likely": 1, "advanced_import": 2, "unsupported": 3}
                items.sort(
                    key=lambda item: (
                        rank.get(item.compatibility, 4),
                        -(item.downloads or 0),
                    )
                )
            result = CatalogPage(items=items, next_cursor=next_cursor)
            self._write_cache(cache, result.model_dump_json())
            return result
        except (httpx.HTTPError, ValueError):
            candidates = [(cache, False)]
            if cursor is None:
                candidates.append((fallback_cache, True))
            for candidate, is_fallback in candidates:
                cached = self._read_page_cache(candidate)
                if cached is None:
                    continue
                items = self._filter_items(
                    cached.items,
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
                    rank = {"tested": 0, "likely": 1, "advanced_import": 2, "unsupported": 3}
                    items.sort(
                        key=lambda item: (
                            rank.get(item.compatibility, 4),
                            -(item.downloads or 0),
                        )
                    )
                return cached.model_copy(
                    update={
                        "items": items,
                        "next_cursor": None if is_fallback else cached.next_cursor,
                        "stale": True,
                    }
                )
            raise

    async def inspect(
        self, remote_id: str, revision: str = "main", requested_role: str | None = None
    ) -> dict[str, Any]:
        if not self._valid_remote_id(remote_id):
            raise ValueError("remote_id must be in owner/model form")
        cache = self._cache_path("detail", remote_id, revision, requested_role)
        try:
            response = await self._client.get(
                f"/api/models/{remote_id}",
                params={"revision": revision, "blobs": "true"},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            if cache.is_file():
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if not isinstance(cached, dict):
                    raise ValueError("cached catalog detail must be an object") from None
                return cast(dict[str, Any], cached)
            raise
        siblings = []
        for sibling in payload.get("siblings") or []:
            lfs = sibling.get("lfs") or {}
            siblings.append(
                {
                    "filename": sibling.get("rfilename"),
                    "size": sibling.get("size") or lfs.get("size"),
                    "sha256": (lfs.get("sha256") or lfs.get("oid") or "").removeprefix("sha256:")
                    or None,
                }
            )
        model = self._normalize(payload, requested_role)
        result = {
            "model": model.model_dump(mode="json"),
            "revision": str(payload.get("sha") or revision),
            "files": siblings,
        }
        self._write_cache(cache, json.dumps(result, default=str))
        return result

    async def inspect_file_prefix(
        self,
        remote_id: str,
        revision: str,
        filename: str,
        *,
        max_bytes: int,
    ) -> bytes:
        """Fetch and cache a bounded immutable file prefix for static inspection."""

        if not self._valid_remote_id(remote_id):
            raise ValueError("remote_id must be in owner/model form")
        if max_bytes < 1 or max_bytes > 16 * 1024 * 1024 + 8:
            raise ValueError("metadata prefix size is outside the supported bounds")
        path = PurePosixPath(filename.replace("\\", "/"))
        if (
            not filename
            or len(filename) > 1_000
            or path.is_absolute()
            or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
        ):
            raise ValueError("metadata filename must be a safe relative path")
        if not revision or len(revision) > 200 or any(character < " " for character in revision):
            raise ValueError("metadata revision is invalid")
        cache = self._binary_cache_path(
            "file-prefix",
            remote_id,
            revision,
            path.as_posix(),
            max_bytes,
        )
        if cache.is_file():
            content = cache.read_bytes()
            if len(content) <= max_bytes:
                return content
        encoded_path = "/".join(quote(part, safe="") for part in path.parts)
        encoded_revision = quote(revision, safe="")
        content_buffer = bytearray()
        async with self._client.stream(
            "GET",
            f"/{remote_id}/resolve/{encoded_revision}/{encoded_path}",
            headers={"range": f"bytes=0-{max_bytes - 1}"},
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                content_buffer.extend(chunk)
                if len(content_buffer) > max_bytes:
                    raise ValueError("model metadata exceeds the inspection limit")
        result = bytes(content_buffer)
        self._write_binary_cache(cache, result)
        return result

    async def close(self) -> None:
        await self._client.aclose()

    async def _hydrate_file_sizes(
        self, items: list[dict[str, Any]], requested_role: str | None
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(6)

        async def hydrate(item: dict[str, Any]) -> dict[str, Any]:
            remote_id = str(item.get("id") or item.get("modelId") or "")
            if not remote_id:
                return item
            revision = str(item.get("sha") or "main")
            try:
                async with semaphore:
                    detail = await self.inspect(remote_id, revision, requested_role)
            except (httpx.HTTPError, ValueError):
                return item
            files = detail.get("files")
            if not isinstance(files, list):
                return item
            siblings = [
                {
                    "rfilename": file.get("filename"),
                    "size": file.get("size"),
                    "lfs": {
                        "oid": f"sha256:{file['sha256']}" if file.get("sha256") else None,
                    },
                }
                for file in files
                if isinstance(file, dict)
            ]
            return {**item, "siblings": siblings}

        return list(await asyncio.gather(*(hydrate(item) for item in items)))

    def _cache_path(self, *parts: object) -> Path:
        key = hashlib.sha256(
            json.dumps((_CACHE_VERSION, parts), sort_keys=True, default=str).encode()
        ).hexdigest()
        return self._cache_dir / f"{key}.json"

    def _binary_cache_path(self, *parts: object) -> Path:
        return self._cache_path(*parts).with_suffix(".bin")

    @staticmethod
    def _write_cache(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _write_binary_cache(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(content)
        os.replace(temporary, path)

    @staticmethod
    def _read_page_cache(path: Path) -> CatalogPage | None:
        if not path.is_file():
            return None
        try:
            return CatalogPage.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _pipeline_tag(role: str) -> str | None:
        return {
            "chat": "text-generation",
            "image": "text-to-image",
            "video": "text-to-video",
        }.get(role)

    @staticmethod
    def _next_link(response: httpx.Response) -> str | None:
        for link in str(response.headers.get("link", "")).split(","):
            if 'rel="next"' not in link:
                continue
            start = link.find("<")
            end = link.find(">", start + 1)
            if start >= 0 and end > start:
                return str(link[start + 1 : end])
        return None

    @staticmethod
    def _validated_cursor(cursor: str) -> str:
        parsed = urlparse(cursor)
        if parsed.scheme not in {"https", ""}:
            raise ValueError("catalog cursor must use HTTPS")
        if parsed.netloc and parsed.netloc != "huggingface.co":
            raise ValueError("catalog cursor must stay on huggingface.co")
        if parsed.path != "/api/models":
            raise ValueError("catalog cursor path is invalid")
        return cursor

    @staticmethod
    def _valid_remote_id(remote_id: str) -> bool:
        return bool(_REMOTE_ID.fullmatch(remote_id))

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
        min_parameters: int | None = None,
        max_parameters: int | None = None,
        max_size_bytes: int | None = None,
        updated_within_days: int | None = None,
    ) -> list[CatalogModel]:
        updated_after = (
            datetime.now(UTC) - timedelta(days=updated_within_days)
            if updated_within_days is not None
            else None
        )
        return [
            item
            for item in items
            if (not compatibility or item.compatibility == compatibility)
            and (not file_format or file_format.lower() in item.formats)
            and (
                not quantization
                or quantization.lower() in {value.lower() for value in item.quantizations}
            )
            and (not license_id or (item.license_id or "").lower() == license_id.lower())
            and (gated not in {"gated", "open"} or bool(item.gated) == (gated == "gated"))
            and (not architecture or architecture.lower() in (item.architecture or "").lower())
            and (
                min_parameters is None
                or (item.parameter_count is not None and item.parameter_count >= min_parameters)
            )
            and (
                max_parameters is None
                or (item.parameter_count is not None and item.parameter_count <= max_parameters)
            )
            and (
                max_size_bytes is None
                or (item.total_size_bytes is not None and item.total_size_bytes <= max_size_bytes)
            )
            and (
                updated_after is None
                or (item.last_modified is not None and item.last_modified >= updated_after)
            )
        ]

    @classmethod
    def _normalize(cls, item: dict[str, Any], requested_role: str | None) -> CatalogModel:
        tags = [str(tag) for tag in item.get("tags") or []]
        siblings = item.get("siblings") or []
        filenames = [str(sibling.get("rfilename", "")) for sibling in siblings]
        formats = sorted(
            {
                Path(filename).suffix.lower().lstrip(".")
                for filename in filenames
                if Path(filename).suffix
            }
        )
        quantizations = sorted(
            {tag.lower() for tag in tags if _QUANTIZATION.fullmatch(tag)}
            | {
                match.group(0).lower()
                for filename in filenames
                for match in re.finditer(
                    r"(?<![a-z0-9])(?:q\d(?:_[a-z0-9]+)*|fp\d+|bf16)(?![a-z0-9])",
                    filename,
                    re.I,
                )
            }
        )
        license_id = next(
            (tag.split(":", 1)[1] for tag in tags if tag.lower().startswith("license:")), None
        )
        architecture = (item.get("config") or {}).get("model_type") or next(
            (tag.split(":", 1)[1] for tag in tags if tag.lower().startswith("base_model:")),
            None,
        )
        parameter_count = cls._parameter_count(tags, remote_id=str(item.get("id") or ""))
        sizes = [
            int(sibling.get("size") or (sibling.get("lfs") or {}).get("size") or 0)
            for sibling in siblings
        ]
        compatibility, reasons = cls._compatibility(
            requested_role=requested_role,
            pipeline_tag=item.get("pipeline_tag"),
            tags=tags,
            filenames=filenames,
        )
        required_runtime = cls._required_runtime(
            requested_role=requested_role,
            tags=tags,
            filenames=filenames,
        )
        remote_id = str(item.get("id") or item.get("modelId") or "")
        return CatalogModel(
            remote_id=remote_id,
            name=remote_id.rsplit("/", 1)[-1],
            author=item.get("author") or (remote_id.split("/", 1)[0] if "/" in remote_id else None),
            pipeline_tag=item.get("pipeline_tag"),
            tags=tags,
            downloads=item.get("downloads"),
            likes=item.get("likes"),
            trending_score=item.get("trendingScore"),
            created_at=cls._datetime(item.get("createdAt")),
            last_modified=cls._datetime(item.get("lastModified")),
            gated=item.get("gated"),
            private=bool(item.get("private", False)),
            library_name=item.get("library_name"),
            architecture=str(architecture) if architecture else None,
            formats=formats,
            quantizations=quantizations,
            parameter_count=parameter_count,
            license_id=license_id,
            total_size_bytes=sum(sizes) or None,
            compatibility=compatibility.value,
            compatibility_reasons=reasons,
            required_runtime=required_runtime,
        )

    @staticmethod
    def _parameter_count(tags: list[str], remote_id: str) -> int | None:
        for value in [*tags, remote_id.rsplit("/", 1)[-1]]:
            match = _PARAMETERS.search(value)
            if not match:
                continue
            multiplier = {"b": 1_000_000_000, "m": 1_000_000, "k": 1_000}[match.group(2).lower()]
            return int(float(match.group(1)) * multiplier)
        return None

    @staticmethod
    def _datetime(value: Any) -> datetime | None:
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _required_runtime(
        *,
        requested_role: str | None,
        tags: list[str],
        filenames: list[str],
    ) -> str | None:
        if requested_role != "chat":
            return None
        lower_files = [name.lower() for name in filenames]
        lower_tags = {tag.lower() for tag in tags}
        if any(name.endswith(".gguf") for name in lower_files) or "gguf" in lower_tags:
            return "llama.cpp"
        if HuggingFaceCatalog._is_modelopt_snapshot(lower_tags, lower_files):
            return "vllm"
        return None

    @staticmethod
    def _is_modelopt_snapshot(lower_tags: set[str], lower_files: list[str]) -> bool:
        has_safe_weights = any(name.endswith(".safetensors") for name in lower_files)
        has_quant_config = any(
            PurePosixPath(name).name == "hf_quant_config.json" for name in lower_files
        )
        has_modelopt_tag = bool(lower_tags.intersection({"modelopt", "nvidia modelopt", "nvfp4"}))
        return has_safe_weights and (has_quant_config or has_modelopt_tag)

    @staticmethod
    def _compatibility(
        *,
        requested_role: str | None,
        pipeline_tag: str | None,
        tags: list[str],
        filenames: list[str],
    ) -> tuple[CompatibilityLevel, list[str]]:
        lower_files = [name.lower() for name in filenames]
        lower_tags = {tag.lower() for tag in tags}
        if any(name.endswith((".bin", ".pt", ".pth", ".ckpt")) for name in lower_files):
            unsafe_note = "contains pickle-compatible weights; blocked by default"
        else:
            unsafe_note = ""
        if requested_role == "chat":
            if any(name.endswith(".gguf") for name in lower_files) or "gguf" in lower_tags:
                reasons = ["GGUF artifact detected"]
                if unsafe_note:
                    reasons.append(unsafe_note)
                return CompatibilityLevel.LIKELY, reasons
            if HuggingFaceCatalog._is_modelopt_snapshot(lower_tags, lower_files):
                return CompatibilityLevel.ADVANCED, ["requires the managed vLLM ModelOpt runtime"]
            return CompatibilityLevel.ADVANCED, ["no GGUF artifact detected"]
        if requested_role == "image":
            if pipeline_tag in {"text-to-image", "image-to-image"}:
                return CompatibilityLevel.LIKELY, ["image pipeline metadata detected"]
            return CompatibilityLevel.ADVANCED, ["requires a verified ComfyUI recipe"]
        if requested_role == "video":
            if pipeline_tag in {"text-to-video", "image-to-video"}:
                return CompatibilityLevel.ADVANCED, ["video pipeline requires a verified workflow"]
            return CompatibilityLevel.ADVANCED, ["requires a verified ComfyUI recipe"]
        if requested_role == "lora":
            has_safetensors = any(name.endswith(".safetensors") for name in lower_files)
            has_lora_marker = (
                any("lora" in name for name in lower_files)
                or "lora" in lower_tags
                or "adapter" in lower_tags
            )
            if has_safetensors and has_lora_marker and not unsafe_note:
                return CompatibilityLevel.LIKELY, ["data-only LoRA candidate"]
            return CompatibilityLevel.ADVANCED, ["requires safetensors LoRA verification"]
        return CompatibilityLevel.ADVANCED, ["select a model role for compatibility guidance"]
