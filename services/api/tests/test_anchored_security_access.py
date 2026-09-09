from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from local_lm import filesystem_links as links


def _owner_query_status(handle: int) -> int:
    windows: Any = ctypes
    security = windows.WinDLL("advapi32", use_last_error=True)
    kernel = windows.WinDLL("kernel32", use_last_error=True)
    query = security.GetSecurityInfo
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_ulong,
        *([ctypes.POINTER(ctypes.c_void_p)] * 5),
    ]
    query.restype = ctypes.c_ulong
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    owner = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    try:
        result = int(
            query(
                ctypes.c_void_p(handle),
                1,
                1,
                ctypes.byref(owner),
                None,
                None,
                None,
                ctypes.byref(descriptor),
            )
        )
        if result == 0:
            assert owner.value is not None and descriptor.value is not None
        return result
    finally:
        if descriptor.value is not None:
            assert kernel.LocalFree(descriptor) is None


@pytest.mark.skipif(os.name != "nt", reason="native Windows access rights")
@pytest.mark.parametrize("create", [False, True])
@pytest.mark.parametrize("read_security", [False, True])
def test_security_access_is_explicit_and_only_requested_for_the_held_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, create: bool, read_security: bool
) -> None:
    root = tmp_path / "selected"
    if not create:
        root.mkdir()
    api = links._windows_api()
    native_open = api.ntdll.NtCreateFile
    masks: list[int] = []

    def record(*args: Any) -> int:
        masks.append(int(args[1].value))
        return int(native_open(*args))

    monkeypatch.setattr(api.ntdll, "NtCreateFile", record)
    monkeypatch.setattr(links, "_windows_api", lambda: api)
    options = {"read_security": True} if read_security else {}
    with links.AnchoredDirectory(root, create=create, **options) as anchor:
        held = anchor.handle
        assert held is not None
        anchor.path = tmp_path / "not-the-selected-directory"
        result = _owner_query_status(held)
        assert masks
        assert all(mask & 0x00020000 == 0 for mask in masks[:-1])
        assert bool(masks[-1] & 0x00020000) == read_security
        assert result == (0 if read_security else 5)
        assert anchor.handle == held
    assert root.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="native Windows volume root")
def test_a_volume_root_is_itself_the_security_query_leaf(tmp_path: Path) -> None:
    with links.AnchoredDirectory(Path(tmp_path.anchor), read_security=True) as anchor:
        assert anchor.handle is not None
        assert _owner_query_status(anchor.handle) == 0


@pytest.mark.parametrize("create", [False, True])
def test_posix_owner_metadata_remains_available_through_the_held_descriptor(
    tmp_path: Path, create: bool
) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX descriptor ownership")
    else:
        root = tmp_path / "selected"
        if not create:
            root.mkdir()
        with links.AnchoredDirectory(root, create=create, read_security=True) as anchor:
            descriptor = anchor.descriptor
            assert descriptor is not None
            anchor.path = tmp_path / "not-the-selected-directory"
            assert os.fstat(descriptor).st_uid == os.geteuid()
            assert anchor.descriptor == descriptor
        assert root.is_dir()
