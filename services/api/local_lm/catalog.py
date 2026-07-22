from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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


class HuggingFaceCatalog:
    def __init__(self, settings: Settings) -> None:
        headers = {"user-agent": "local-lm/0.1"}
        if settings.hf_token:
            headers["authorization"] = f"Bearer {settings.hf_token}"
        self._client = httpx.AsyncClient(
            base_url="https://huggingface.co", headers=headers, timeout=30, follow_redirects=True
        )
        self._cache_dir = settings.catalog_cache_dir

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
        )
        try:
            response = await self._client.get(url, params=None if cursor else params)
            response.raise_for_status()
            payload = response.json()
            items = [self._normalize(item, role) for item in payload if isinstance(item, dict)]
            items = self._filter_items(
                items,
                compatibility=compatibility,
                file_format=file_format,
                quantization=quantization,
                license_id=license_id,
                gated=gated,
                architecture=architecture,
            )
            if sort == "compatible":
                rank = {"tested": 0, "likely": 1, "advanced_import": 2, "unsupported": 3}
                items.sort(
                    key=lambda item: (
                        rank.get(item.compatibility, 4),
                        -(item.downloads or 0),
                    )
                )
            result = CatalogPage(items=items, next_cursor=self._next_link(response))
            self._write_cache(cache, result.model_dump_json())
            return result
        except (httpx.HTTPError, ValueError):
            if cache.is_file():
                return CatalogPage.model_validate_json(cache.read_text(encoding="utf-8"))
            raise

    async def inspect(self, remote_id: str, revision: str = "main") -> dict[str, Any]:
        cache = self._cache_path("detail", remote_id, revision)
        try:
            response = await self._client.get(
                f"/api/models/{remote_id}",
                params={"revision": revision, "files_metadata": "true"},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            if cache.is_file():
                return json.loads(cache.read_text(encoding="utf-8"))
            raise
        siblings = []
        for sibling in payload.get("siblings") or []:
            lfs = sibling.get("lfs") or {}
            siblings.append(
                {
                    "filename": sibling.get("rfilename"),
                    "size": sibling.get("size") or lfs.get("size"),
                    "sha256": (lfs.get("oid") or "").removeprefix("sha256:") or None,
                }
            )
        model = self._normalize(payload, None)
        result = {"model": model.model_dump(mode="json"), "files": siblings}
        self._write_cache(cache, json.dumps(result, default=str))
        return result

    async def close(self) -> None:
        await self._client.aclose()

    def _cache_path(self, *parts: object) -> Path:
        key = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()
        return self._cache_dir / f"{key}.json"

    @staticmethod
    def _write_cache(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _pipeline_tag(role: str) -> str | None:
        return {
            "chat": "text-generation",
            "image": "text-to-image",
            "video": "text-to-video",
        }.get(role)

    @staticmethod
    def _next_link(response: httpx.Response) -> str | None:
        for link in response.headers.get("link", "").split(","):
            if 'rel="next"' not in link:
                continue
            start = link.find("<")
            end = link.find(">", start + 1)
            if start >= 0 and end > start:
                return link[start + 1 : end]
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
    def _filter_items(
        items: list[CatalogModel],
        *,
        compatibility: str | None,
        file_format: str | None,
        quantization: str | None,
        license_id: str | None,
        gated: str | None,
        architecture: str | None,
    ) -> list[CatalogModel]:
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
            return CompatibilityLevel.ADVANCED, ["no GGUF artifact detected"]
        if requested_role == "image":
            if pipeline_tag in {"text-to-image", "image-to-image"}:
                return CompatibilityLevel.LIKELY, ["image pipeline metadata detected"]
            return CompatibilityLevel.ADVANCED, ["requires a verified ComfyUI recipe"]
        if requested_role == "video":
            if pipeline_tag in {"text-to-video", "image-to-video"}:
                return CompatibilityLevel.ADVANCED, ["video pipeline requires a verified workflow"]
            return CompatibilityLevel.ADVANCED, ["requires a verified ComfyUI recipe"]
        return CompatibilityLevel.ADVANCED, ["select a model role for compatibility guidance"]
