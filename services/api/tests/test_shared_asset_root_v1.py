from __future__ import annotations

import sys
from pathlib import Path

import pytest

from local_lm.desktop import default_data_dir
from local_lm.shared_asset_root_v1 import (
    INVALID_SHARED_ROOT,
    SharedAssetRootError,
    default_shared_asset_root,
    resolve_shared_asset_root,
)


@pytest.fixture(autouse=True)
def _desktop_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "desktop-local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "desktop-xdg"))


@pytest.mark.parametrize(
    ("platform", "leaf"), [("win32", "shared-assets-v1"), ("linux", "lm-atelier-shared-assets-v1")]
)
def test_recommended_library_is_outside_desktop_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, leaf: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    data = default_data_dir()
    shared = default_shared_asset_root()
    assert shared == data.parent / leaf
    assert data not in shared.parents
    assert shared not in data.parents
    assert not shared.exists()


def test_desktop_and_isolated_profiles_require_explicit_selection(tmp_path: Path) -> None:
    assert resolve_shared_asset_root(profile_data_dir=default_data_dir()) is None
    assert resolve_shared_asset_root(profile_data_dir=tmp_path / "profile") is None


def test_data_directory_override_never_enables_implicit_sharing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = default_data_dir()
    monkeypatch.setenv("LOCAL_LM_DATA_DIR", str(data))
    assert resolve_shared_asset_root(profile_data_dir=data) is None


def test_explicit_root_isolates_an_app_even_for_isolated_profiles(tmp_path: Path) -> None:
    explicit = tmp_path / "custom-library"
    isolated = tmp_path / "data"
    assert resolve_shared_asset_root(profile_data_dir=isolated, explicit=explicit) == explicit
    assert not explicit.exists()


@pytest.mark.parametrize("location", ["same", "parent", "child", "normalized_child"])
def test_explicit_library_refuses_profile_overlap(tmp_path: Path, location: str) -> None:
    profile = tmp_path / "profile"
    chosen = {
        "same": profile,
        "parent": tmp_path,
        "child": profile / "packages",
        "normalized_child": tmp_path / "elsewhere" / ".." / "profile" / "packages",
    }[location]
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(profile_data_dir=profile, explicit=chosen)


@pytest.mark.parametrize("location", ["same", "parent", "child"])
def test_explicit_library_refuses_other_protected_roots(tmp_path: Path, location: str) -> None:
    install = tmp_path / "install"
    chosen = {"same": install, "parent": tmp_path, "child": install / "packages"}[location]
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(
            profile_data_dir=tmp_path / "profile",
            explicit=chosen,
            protected_roots=(install,),
        )


def test_isolated_profile_cannot_adopt_the_desktop_data_tree() -> None:
    data = default_data_dir()
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(
            profile_data_dir=data.parent / "separate-profile", explicit=data / "packages"
        )


def test_recommended_root_can_be_explicitly_selected() -> None:
    selected = default_shared_asset_root()
    assert (
        resolve_shared_asset_root(profile_data_dir=default_data_dir(), explicit=selected)
        == selected
    )


@pytest.mark.parametrize("value", ["relative-lib", r"\\server\share\library", "//server/share"])
def test_refuses_relative_and_network_roots(tmp_path: Path, value: str) -> None:
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(profile_data_dir=tmp_path / "profile", explicit=Path(value))


def test_explicit_selection_requires_an_absolute_profile_path(tmp_path: Path) -> None:
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(
            profile_data_dir=Path("relative-profile"), explicit=tmp_path / "library"
        )


def test_relative_recommendation_cannot_be_selected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "local_lm.shared_asset_root_v1.default_data_dir", lambda: Path("relative/profile")
    )
    recommendation = default_shared_asset_root()
    assert not recommendation.is_absolute()
    with pytest.raises(SharedAssetRootError, match=INVALID_SHARED_ROOT):
        resolve_shared_asset_root(profile_data_dir=tmp_path / "profile", explicit=recommendation)
