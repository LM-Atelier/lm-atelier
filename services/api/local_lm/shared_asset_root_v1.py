"""Pure Shared Asset Library root resolution (item 58, first slice).

Desktop launches of every LM Atelier-based application share one default
root. An explicit per-app root isolates that app. Isolated profile data
dirs never discover the real desktop library. No publish, claim, lease,
or runtime rewrite lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NoReturn

from .desktop import default_data_dir

SCHEMA_ID: Final = "lm-atelier-shared-asset-root-v1"
SCHEMA_VERSION: Final = 1
INVALID_SHARED_ROOT: Final = "shared asset library root is invalid"
SHARED_LEAF: Final = "shared-assets-v1"


class SharedAssetRootError(ValueError):
    """Fixed non-echoing refusal for an unusable shared-asset root."""


def _invalid() -> NoReturn:
    raise SharedAssetRootError(INVALID_SHARED_ROOT)


def default_shared_asset_root() -> Path:
    """Return the default root shared by public, Red, and desktop development."""

    data = default_data_dir()
    if data.name == "data":
        return data.parent / SHARED_LEAF
    return data.parent / f"{data.name}-{SHARED_LEAF}"


def _is_unc(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or text.startswith("//") or path.as_posix().startswith("//")


def _refuse_nested(left: Path, right: Path) -> None:
    if left == right or left in right.parents or right in left.parents:
        _invalid()


def resolve_shared_asset_root(
    *,
    profile_data_dir: Path,
    explicit: Path | None = None,
) -> Path | None:
    """Resolve the shared root for one profile, or None when sharing is off.

    Explicit overrides isolate that application. Isolated data dirs (tests,
    source `data/` cwd) stay None unless an explicit root is supplied.
    """

    if not isinstance(profile_data_dir, Path):
        _invalid()
    try:
        profile = profile_data_dir.expanduser()
    except (OSError, RuntimeError, ValueError):
        _invalid()
    if not str(profile) or _is_unc(profile):
        _invalid()

    if explicit is not None:
        if not isinstance(explicit, Path):
            _invalid()
        try:
            chosen = explicit.expanduser()
        except (OSError, RuntimeError, ValueError):
            _invalid()
        if not str(chosen) or _is_unc(chosen) or not chosen.is_absolute():
            _invalid()
        _refuse_nested(profile, chosen)
        return chosen

    desktop = default_data_dir()
    if profile.resolve() != desktop.resolve():
        return None
    shared = default_shared_asset_root()
    _refuse_nested(desktop, shared)
    if _is_unc(shared):
        _invalid()
    return shared
