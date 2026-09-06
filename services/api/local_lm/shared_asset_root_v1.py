"""Explicit Shared Asset Library location selection, without filesystem access.

The recommendation sits outside desktop profile data. Selecting a path never
attaches a library: ownership, filesystem identity and compatibility probes
must still pass before any shared-store operation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, NoReturn

from .desktop import default_data_dir

SCHEMA_ID: Final = "lm-atelier-shared-asset-root-v1"
SCHEMA_VERSION: Final = 1
INVALID_SHARED_ROOT: Final = "shared asset library root is invalid"
STORE_LEAF: Final = "shared-assets-v1"


class SharedAssetRootError(ValueError):
    """Fixed non-echoing refusal for an unusable shared-asset root."""


def _invalid() -> NoReturn:
    raise SharedAssetRootError(INVALID_SHARED_ROOT)


def default_shared_asset_root() -> Path:
    """Recommend an external per-user location; do not discover or create it."""

    leaf = STORE_LEAF if sys.platform == "win32" else f"lm-atelier-{STORE_LEAF}"
    return default_data_dir().parent / leaf


def _absolute_local_path(value: Path) -> Path:
    if not isinstance(value, Path):
        _invalid()
    try:
        chosen = value.expanduser()
    except (OSError, RuntimeError, ValueError):
        _invalid()
    text = str(chosen)
    if (
        not text
        or "\x00" in text
        or not chosen.is_absolute()
        or text.startswith("\\\\")
        or chosen.as_posix().startswith("//")
        or ".." in chosen.parts
    ):
        _invalid()
    return chosen


def _refuse_overlap(protected: Path, chosen: Path) -> None:
    if chosen == protected or chosen in protected.parents or protected in chosen.parents:
        _invalid()


def resolve_shared_asset_root(
    *,
    profile_data_dir: Path,
    explicit: Path | None = None,
    protected_roots: tuple[Path, ...] = (),
) -> Path | None:
    """Select an explicitly configured location, or None when sharing is off.

    Callers supply every other known profile, install and uninstall-purge root.
    This lexical check does not follow links or inspect the selected volume;
    attachment must separately verify stable directory identity and suitability.
    """

    if explicit is None:
        return None
    chosen = _absolute_local_path(explicit)
    profile = _absolute_local_path(profile_data_dir)
    desktop = _absolute_local_path(default_data_dir())
    for protected in (profile, desktop, *protected_roots):
        _refuse_overlap(_absolute_local_path(protected), chosen)
    return chosen
