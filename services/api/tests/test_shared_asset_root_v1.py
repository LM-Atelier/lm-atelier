from __future__ import annotations

import sys
from pathlib import Path

import pytest

from local_lm.desktop import default_data_dir
from local_lm.shared_asset_root_v1 import (
    INVALID_SHARED_ROOT,
    SHARED_LEAF,
    SharedAssetRootError,
    default_shared_asset_root,
    resolve_shared_asset_root,
)


def test_windows_desktop_default_is_sibling_of_data_dir(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    data = default_data_dir()
    shared = default_shared_asset_root()
    assert shared.parent == data.parent
    assert shared.name == SHARED_LEAF
    assert shared != data
    assert data not in shared.parents
    assert shared not in data.parents
    resolved = resolve_shared_asset_root(profile_data_dir=data)
    assert resolved == shared


def test_linux_desktop_default_stays_outside_the_profile_data_dir(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/xdg-data")
    data = default_data_dir()
    shared = default_shared_asset_root()
    assert shared.parent == data.parent
    assert SHARED_LEAF in shared.name
    assert shared != data
    assert resolve_shared_asset_root(profile_data_dir=data) == shared


def test_isolated_profile_does_not_discover_the_desktop_library(
    monkeypatch,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    isolated = tmp_path / "data"
    assert resolve_shared_asset_root(profile_data_dir=isolated) is None
    assert isolated != default_data_dir()


def test_explicit_root_isolates_an_app_even_for_isolated_profiles(tmp_path: Path) -> None:
    explicit = tmp_path / "custom-library"
    isolated = tmp_path / "data"
    resolved = resolve_shared_asset_root(profile_data_dir=isolated, explicit=explicit)
    assert resolved == explicit


def test_two_desktop_profiles_share_the_same_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    data = default_data_dir()
    first = resolve_shared_asset_root(profile_data_dir=data)
    second = resolve_shared_asset_root(profile_data_dir=data)
    assert first is not None
    assert first == second == default_shared_asset_root()


def test_refuses_unc_and_nested_roots(tmp_path: Path) -> None:
    isolated = tmp_path / "data"
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(
            profile_data_dir=isolated,
            explicit=Path(r"\\server\share\library"),
        )
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(
            profile_data_dir=isolated,
            explicit=isolated / "shared",
        )
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(profile_data_dir=isolated, explicit=Path("relative-lib"))
