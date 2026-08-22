from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from local_lm.desktop import default_data_dir
from local_lm.shared_asset_root_v1 import (
    INVALID_SHARED_ROOT,
    STORE_LEAF,
    SharedAssetRootError,
    default_shared_asset_root,
    resolve_shared_asset_root,
)


def test_windows_desktop_default_is_inside_the_data_dir(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    data = default_data_dir()
    shared = default_shared_asset_root()
    assert shared == data / STORE_LEAF
    assert shared.parent == data
    assert shared.name == "packages"
    resolved = resolve_shared_asset_root(profile_data_dir=data)
    assert resolved == shared


def test_linux_desktop_default_is_inside_the_data_dir(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/xdg-data")
    data = default_data_dir()
    shared = default_shared_asset_root()
    assert shared == data / STORE_LEAF
    assert shared.parent == data
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


def test_explicit_child_inside_an_isolated_profile_is_allowed(tmp_path: Path) -> None:
    isolated = tmp_path / "data"
    explicit = isolated / STORE_LEAF
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
    assert first.parent == data


def test_another_absolute_app_data_dir_uses_the_canonical_folder(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    data = default_data_dir()
    other = Path(r"C:\Users\Tester\AppData\Local") / "OtherApp" / "data"
    resolved = resolve_shared_asset_root(profile_data_dir=other)
    assert resolved == data / STORE_LEAF
    assert resolved == default_shared_asset_root()


def test_linux_other_absolute_app_data_dir_uses_the_canonical_folder(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/home/tester/.local/share")
    data = default_data_dir()
    other = Path("/opt/other-app/data")
    resolved = resolve_shared_asset_root(profile_data_dir=other)
    assert resolved == data / STORE_LEAF


def test_relative_source_data_dir_does_not_discover_the_desktop_library(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    assert resolve_shared_asset_root(profile_data_dir=Path("data")) is None


def test_pytest_named_ci_profile_does_not_discover_the_desktop_library(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: r"C:\Users\Tester\AppData\Local\Temp")
    ci = Path(r"C:\ci\pytest-artifacts\run")
    assert resolve_shared_asset_root(profile_data_dir=ci) is None


def test_linux_pytest_named_ci_profile_does_not_discover_the_desktop_library(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/home/tester/.local/share")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/tmp")
    ci = Path("/opt/pytest-ci/data")
    assert resolve_shared_asset_root(profile_data_dir=ci) is None


def test_source_tree_data_dir_does_not_discover_the_desktop_library(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Tester\AppData\Local")
    source_data = Path(__file__).resolve().parents[1] / "data"
    assert source_data != default_data_dir()
    assert resolve_shared_asset_root(profile_data_dir=source_data) is None


def test_refuses_unc_relative_and_covering_roots(tmp_path: Path) -> None:
    isolated = tmp_path / "data"
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(
            profile_data_dir=isolated,
            explicit=Path(r"\\server\share\library"),
        )
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(profile_data_dir=isolated, explicit=isolated)
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(profile_data_dir=isolated, explicit=isolated.parent)
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(profile_data_dir=isolated, explicit=Path("relative-lib"))
