from __future__ import annotations

from pathlib import Path

from local_lm.comfy_registry_paths import registry_wheel_environment_root


def test_registry_wheel_environment_root_has_one_canonical_location() -> None:
    registry_root = Path("managed-registry")

    assert registry_wheel_environment_root(registry_root) == (
        registry_root / "registry-wheel-environments"
    )
