from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from local_lm import filesystem_links as links


@pytest.fixture
def owned_directory(tmp_path: Path) -> Path:
    selected = tmp_path / "owned"
    if sys.platform == "win32":
        # Elevated tokens can default new objects to a group owner. These
        # tests require a user-owned, accessible directory, so create only
        # this temporary fixture with explicit owner and current-user access.
        windows: Any = ctypes
        security = windows.WinDLL("advapi32", use_last_error=True)
        query = security.GetTokenInformation
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        query.restype = ctypes.c_int
        needed = ctypes.c_ulong()
        assert not query(ctypes.c_void_p(-6), 1, None, 0, ctypes.byref(needed))
        assert windows.get_last_error() == 122
        buffer = ctypes.create_string_buffer(needed.value)
        assert query(ctypes.c_void_p(-6), 1, buffer, len(buffer), ctypes.byref(needed))
        user = ctypes.c_void_p.from_buffer(buffer)
        assert user.value is not None
        kernel = windows.WinDLL("kernel32", use_last_error=True)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree.restype = ctypes.c_void_p
        convert_sid = security.ConvertSidToStringSidW
        convert_sid.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]
        convert_sid.restype = ctypes.c_int
        convert_descriptor = security.ConvertStringSecurityDescriptorToSecurityDescriptorW
        convert_descriptor.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_ulong),
        ]
        convert_descriptor.restype = ctypes.c_int

        class SecurityAttributes(ctypes.Structure):
            _fields_ = (
                ("Length", ctypes.c_ulong),
                ("Descriptor", ctypes.c_void_p),
                ("Inherit", ctypes.c_int),
            )

        kernel.CreateDirectoryW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(SecurityAttributes)]
        kernel.CreateDirectoryW.restype = ctypes.c_int
        sid_text = ctypes.c_wchar_p()
        descriptor = ctypes.c_void_p()
        try:
            assert convert_sid(user, ctypes.byref(sid_text))
            assert sid_text.value is not None
            sddl = f"O:{sid_text.value}D:P(A;;FA;;;{sid_text.value})"
            assert convert_descriptor(sddl, 1, ctypes.byref(descriptor), None)
            attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor.value, 0)
            assert kernel.CreateDirectoryW(str(selected), ctypes.byref(attributes))
        finally:
            if descriptor.value:
                kernel.LocalFree(descriptor)
            if sid_text.value:
                kernel.LocalFree(ctypes.cast(sid_text, ctypes.c_void_p))
    else:
        selected.mkdir()
    return selected


def test_owner_comparison_uses_the_held_directory(owned_directory: Path) -> None:
    with links.AnchoredDirectory(owned_directory, read_security=True) as anchor:
        anchor.path = owned_directory / "absent"
        assert links.directory_owned_by_current_user(anchor) is True
        assert links.directory_owned_by_current_user(anchor) is True


def test_closed_directory_has_no_ownership_verdict(tmp_path: Path) -> None:
    anchor = links.AnchoredDirectory(tmp_path, read_security=True)
    anchor.close()
    with pytest.raises(links.AnchoredDirectoryError):
        links.directory_owned_by_current_user(anchor)


def test_posix_compares_the_effective_uid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX effective uid")
    else:
        effective_uid = os.geteuid()
        with links.AnchoredDirectory(tmp_path) as anchor:
            monkeypatch.setattr(os, "geteuid", lambda: effective_uid + 1)
            assert links.directory_owned_by_current_user(anchor) is False


def test_posix_refuses_failed_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX descriptor metadata")
    else:
        with links.AnchoredDirectory(tmp_path) as anchor:

            def fail(_descriptor: int) -> os.stat_result:
                raise OSError("constructed unavailable descriptor")

            monkeypatch.setattr(os, "fstat", fail)
            with pytest.raises(links.AnchoredDirectoryError):
                links.directory_owned_by_current_user(anchor)


@pytest.mark.skipif(os.name != "nt", reason="native Windows security")
def test_windows_owner_query_requires_security_access(tmp_path: Path) -> None:
    with (
        links.AnchoredDirectory(tmp_path) as anchor,
        pytest.raises(links.AnchoredDirectoryError),
    ):
        links.directory_owned_by_current_user(anchor)


@pytest.mark.skipif(os.name != "nt", reason="native Windows security")
@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "security",
        "token",
        "null-owner",
        "invalid-owner",
        "short-token",
        "different-owner",
        "outside-token-sid",
        "truncated-token-sid",
    ],
)
def test_windows_ownership_queries_and_releases_the_exact_descriptor(
    owned_directory: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    api = links._windows_ownership_api()
    native_query = api.security.GetSecurityInfo
    native_token = api.security.GetTokenInformation
    native_free = api.kernel.LocalFree
    queried: list[int] = []
    allocated: list[int] = []
    freed: list[int] = []
    tokens: list[int] = []
    # A valid well-known SID (Everyone), never a desktop account or file mutation.
    foreign_sid = ctypes.create_string_buffer(bytes.fromhex("010100000000000100000000"))
    invalid_sid = ctypes.create_string_buffer(68)

    def query(*args: Any) -> int:
        queried.append(int(args[0].value))
        result = int(native_query(*args))
        pointer = ctypes.cast(args[7], ctypes.POINTER(ctypes.c_void_p))[0]
        if pointer is not None:
            allocated.append(pointer)
        if failure == "security":
            return 5
        if failure in {"null-owner", "invalid-owner", "different-owner"}:
            owner = ctypes.cast(args[3], ctypes.POINTER(ctypes.c_void_p))
            owner[0] = (
                None
                if failure == "null-owner"
                else ctypes.addressof(invalid_sid if failure == "invalid-owner" else foreign_sid)
            )
        return result

    def token(*args: Any) -> int:
        tokens.append(int(args[0].value))
        assert args[1] == 1  # TokenUser, not the token's default owner or groups.
        result = int(native_token(*args))
        if failure == "token":
            return 0
        if result:
            returned = ctypes.cast(args[4], ctypes.POINTER(ctypes.c_ulong))
            if failure == "short-token":
                returned[0] = 1
            elif failure == "outside-token-sid":
                # A separately allocated valid SID stays alive even in the
                # missing-guard control, but is outside the token buffer.
                api.TokenUser.from_buffer(args[2]).Sid = ctypes.addressof(foreign_sid)
            elif failure == "truncated-token-sid":
                # Retain the real allocation and native SID. Only the reported
                # extent omits its subauthorities, so a missing-guard control
                # remains within allocated memory.
                sid = api.TokenUser.from_buffer(args[2]).Sid
                assert sid is not None
                assert ctypes.c_ubyte.from_address(sid + 1).value > 0
                returned[0] = sid - ctypes.addressof(args[2]) + 8
        return result

    def free(pointer: Any) -> Any:
        freed.append(int(pointer.value))
        return native_free(pointer)

    monkeypatch.setattr(api.security, "GetSecurityInfo", query)
    monkeypatch.setattr(api.security, "GetTokenInformation", token)
    monkeypatch.setattr(api.kernel, "LocalFree", free)
    monkeypatch.setattr(links, "_windows_ownership_api", lambda: api)
    with links.AnchoredDirectory(owned_directory, read_security=True) as anchor:
        held = anchor.handle
        anchor.path = owned_directory / "absent"
        if failure in {"none", "different-owner"}:
            assert links.directory_owned_by_current_user(anchor) is (failure == "none")
        else:
            with pytest.raises(links.AnchoredDirectoryError):
                links.directory_owned_by_current_user(anchor)
        assert queried == [held]
        assert len(allocated) == 1
        assert freed == allocated
        assert all(value == ctypes.c_void_p(-6).value for value in tokens)
        assert anchor.handle == held
