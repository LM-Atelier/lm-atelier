from __future__ import annotations

import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_lm.filesystem_links import LinkInspectionFailure, is_link_or_reparse


def test_regular_file_and_symbolic_link_are_distinguished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regular = tmp_path / "regular"
    regular.write_text("content", encoding="utf-8")
    assert not is_link_or_reparse(
        regular,
        missing="raise",
        unreadable="raise",
    )

    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0),
    )
    assert is_link_or_reparse(
        tmp_path / "synthetic-link",
        missing="raise",
        unreadable="raise",
    )


def test_windows_reparse_attribute_is_recognized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400),
    )
    assert is_link_or_reparse(
        tmp_path / "synthetic-junction",
        missing="raise",
        unreadable="raise",
    )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("assume_link", True), ("assume_regular", False)],
)
def test_missing_path_policy_is_explicit(
    tmp_path: Path,
    policy: LinkInspectionFailure,
    expected: bool,
) -> None:
    assert (
        is_link_or_reparse(
            tmp_path / "missing",
            missing=policy,
            unreadable="raise",
        )
        is expected
    )


def test_missing_path_can_propagate(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        is_link_or_reparse(
            tmp_path / "missing",
            missing="raise",
            unreadable="raise",
        )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("assume_link", True), ("assume_regular", False)],
)
def test_unreadable_path_policy_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: LinkInspectionFailure,
    expected: bool,
) -> None:
    def unavailable(_path: Path) -> None:
        raise PermissionError("unavailable")

    monkeypatch.setattr(Path, "lstat", unavailable)
    assert (
        is_link_or_reparse(
            tmp_path / "unreadable",
            missing="raise",
            unreadable=policy,
        )
        is expected
    )


def test_unreadable_path_can_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_path: Path) -> None:
        raise PermissionError("unavailable")

    monkeypatch.setattr(Path, "lstat", unavailable)
    with pytest.raises(PermissionError, match="unavailable"):
        is_link_or_reparse(
            tmp_path / "unreadable",
            missing="raise",
            unreadable="raise",
        )
