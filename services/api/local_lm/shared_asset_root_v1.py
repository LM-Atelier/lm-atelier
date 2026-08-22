"""Pure Shared Asset Library root resolution (item 58, first slice).

The default library folder lives inside the desktop application data
directory. Compatible desktop profiles share that canonical folder. An
explicit per-app root isolates that app. Test and source-checkout data dirs
never discover the real desktop library. No publish, claim, lease, or
runtime rewrite lives here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final, NoReturn

from .desktop import default_data_dir

SCHEMA_ID: Final = "lm-atelier-shared-asset-root-v1"
SCHEMA_VERSION: Final = 1
INVALID_SHARED_ROOT: Final = "shared asset library root is invalid"
STORE_LEAF: Final = "packages"


class SharedAssetRootError(ValueError):
    """Fixed non-echoing refusal for an unusable shared-asset root."""


def _invalid() -> NoReturn:
    raise SharedAssetRootError(INVALID_SHARED_ROOT)


def default_shared_asset_root() -> Path:
    """Return the library folder inside the desktop application data directory."""

    return default_data_dir() / STORE_LEAF


def _is_unc(path: Path) -> bool:
    text = str(path)
    return text.startswith("\\\\") or text.startswith("//") or path.as_posix().startswith("//")


def _refuse_covering_profile(profile: Path, chosen: Path) -> None:
    if chosen == profile or chosen in profile.parents:
        _invalid()


def _source_tree_root() -> Path | None:
    here = Path(__file__).resolve()
    try:
        root = here.parents[3]
    except IndexError:
        return None
    if (root / "services" / "api" / "local_lm").is_dir():
        return root
    return None


def _is_under(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _is_test_or_dev_profile(profile: Path, desktop: Path) -> bool:
    """True for test/dev roots that must not discover the desktop library."""

    try:
        profile_resolved = profile.resolve()
        desktop_resolved = desktop.resolve()
    except (OSError, RuntimeError, ValueError):
        _invalid()
    if profile_resolved == desktop_resolved:
        return False
    if not profile.is_absolute() and not os.path.isabs(str(profile)):
        return True
    if any(part.lower().startswith("pytest") for part in profile_resolved.parts):
        return True
    tmp = Path(tempfile.gettempdir()).resolve()
    if _is_under(profile_resolved, tmp):
        return True
    source_root = _source_tree_root()
    return source_root is not None and _is_under(profile_resolved, source_root)


def resolve_shared_asset_root(
    *,
    profile_data_dir: Path,
    explicit: Path | None = None,
) -> Path | None:
    """Resolve the library root for one profile, or None when sharing is off.

    Explicit overrides isolate that application. Test and source-checkout
    data dirs stay None unless an explicit root is supplied. Other desktop
    profiles use the canonical folder inside the application data directory.
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
        _refuse_covering_profile(profile, chosen)
        return chosen

    desktop = default_data_dir()
    if _is_test_or_dev_profile(profile, desktop):
        return None
    shared = default_shared_asset_root()
    _refuse_covering_profile(desktop, shared)
    if _is_unc(shared):
        _invalid()
    return shared
