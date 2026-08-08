"""Canonical managed paths for prepared ComfyUI Registry packages."""

from __future__ import annotations

from pathlib import Path


def registry_wheel_environment_root(registry_root: Path) -> Path:
    """Return the one managed root for verified Registry wheel environments."""

    return registry_root / "registry-wheel-environments"
