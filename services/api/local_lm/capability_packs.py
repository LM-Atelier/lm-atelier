from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

MAX_CAPABILITY_PACK_BYTES = 256 * 1024
MAX_CAPABILITY_FAMILIES = 256
MAX_MARKERS_PER_FAMILY = 64
_PACK_DIRECTORY = Path(__file__).with_name("capability_packs")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")


class CapabilityPackError(ValueError):
    """A data-only capability pack failed its integrity or schema checks."""


@dataclass(frozen=True)
class ArchitectureFamilyContract:
    id: str
    roles: tuple[str, ...]
    architecture_markers: tuple[str, ...]


def _read_bounded_json(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    if len(content) > MAX_CAPABILITY_PACK_BYTES:
        raise CapabilityPackError("capability pack exceeds the size limit")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CapabilityPackError("capability pack JSON is malformed") from exc
    if not isinstance(value, dict):
        raise CapabilityPackError("capability pack must contain an object")
    return value


@lru_cache(maxsize=1)
def architecture_family_contracts() -> tuple[ArchitectureFamilyContract, ...]:
    lock = _read_bounded_json(_PACK_DIRECTORY / "capability-packs.lock.json")
    packs = lock.get("packs")
    if lock.get("version") != 1 or not isinstance(packs, dict):
        raise CapabilityPackError("capability pack lock is invalid")
    filename = "architecture-families-v1.json"
    expected_hash = packs.get(filename)
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise CapabilityPackError("capability pack is not hash-pinned")
    path = _PACK_DIRECTORY / filename
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise CapabilityPackError("capability pack integrity check failed")
    payload = _read_bounded_json(path)
    raw_families = payload.get("families")
    if payload.get("version") != 1 or not isinstance(raw_families, list):
        raise CapabilityPackError("capability pack schema is unsupported")
    if len(raw_families) > MAX_CAPABILITY_FAMILIES:
        raise CapabilityPackError("capability pack declares too many families")
    contracts: list[ArchitectureFamilyContract] = []
    seen: set[str] = set()
    for item in raw_families:
        if not isinstance(item, dict):
            raise CapabilityPackError("capability family entry is invalid")
        family_id = item.get("id")
        roles = item.get("roles")
        markers = item.get("architecture_markers")
        if (
            not isinstance(family_id, str)
            or not _SAFE_ID.fullmatch(family_id)
            or family_id in seen
            or not isinstance(roles, list)
            or not roles
            or any(role not in {"chat", "image", "video"} for role in roles)
            or not isinstance(markers, list)
            or not markers
            or len(markers) > MAX_MARKERS_PER_FAMILY
            or any(
                not isinstance(marker, str)
                or not marker
                or len(marker) > 200
                or marker != marker.casefold()
                for marker in markers
            )
        ):
            raise CapabilityPackError("capability family entry is invalid")
        seen.add(family_id)
        contracts.append(
            ArchitectureFamilyContract(
                id=family_id,
                roles=tuple(dict.fromkeys(str(role) for role in roles)),
                architecture_markers=tuple(dict.fromkeys(str(marker) for marker in markers)),
            )
        )
    return tuple(contracts)
