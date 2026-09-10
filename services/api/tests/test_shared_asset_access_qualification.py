from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from directory_security import create_owned_directory

from local_lm import filesystem_links as links
from local_lm import shared_asset_contract_v1 as contract


def test_private_access_uses_the_held_directory(tmp_path: Path) -> None:
    root = create_owned_directory(tmp_path / "private")
    with links.AnchoredDirectory(root, read_security=True) as anchor:
        anchor.path = root / "absent"
        assert links.directory_private_to_current_user(anchor) is True


def test_closed_directory_has_no_private_access_verdict(tmp_path: Path) -> None:
    root = create_owned_directory(tmp_path / "private")
    anchor = links.AnchoredDirectory(root, read_security=True)
    anchor.close()
    with pytest.raises(links.AnchoredDirectoryError):
        links.directory_private_to_current_user(anchor)


@pytest.mark.parametrize(
    ("mode", "private"),
    [
        (0o700, True),
        (0o500, True),
        (0o750, False),
        (0o710, False),
        (0o704, False),
        (0o770, False),
    ],
)
def test_posix_private_access_requires_no_group_or_other_permissions(
    tmp_path: Path, mode: int, private: bool
) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX permissions")
    root = create_owned_directory(tmp_path / "private")
    root.chmod(mode)
    try:
        with links.AnchoredDirectory(root) as anchor:
            assert links.directory_private_to_current_user(anchor) is private
    finally:
        root.chmod(0o700)


def test_posix_private_access_requires_the_effective_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX ownership")
    else:
        effective = os.geteuid()
        root = create_owned_directory(tmp_path / "private")
        with links.AnchoredDirectory(root) as anchor:
            monkeypatch.setattr(os, "geteuid", lambda: effective + 1)
            assert links.directory_private_to_current_user(anchor) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DACL")
@pytest.mark.parametrize(
    ("extra_aces", "private"),
    [
        ("", True),
        ("(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)", True),
        ("(A;;FR;;;WD)", False),
        ("(A;;FW;;;WD)", False),
        ("(A;;FA;;;AU)", False),
        ("(A;OICIIO;FR;;;WD)", False),
    ],
)
def test_windows_private_access_checks_all_allow_entries(
    tmp_path: Path, extra_aces: str, private: bool
) -> None:
    root = create_owned_directory(tmp_path / "private", extra_aces=extra_aces)
    with links.AnchoredDirectory(root, read_security=True) as anchor:
        assert links.directory_owned_by_current_user(anchor) is True
        assert links.directory_private_to_current_user(anchor) is private


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DACL")
def test_windows_null_dacl_is_not_private(tmp_path: Path) -> None:
    root = create_owned_directory(tmp_path / "private", null_dacl=True)
    with links.AnchoredDirectory(root, read_security=True) as anchor:
        assert links.directory_private_to_current_user(anchor) is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows READ_CONTROL")
def test_private_access_refuses_a_handle_without_security_access(tmp_path: Path) -> None:
    root = create_owned_directory(tmp_path / "private")
    with links.AnchoredDirectory(root) as anchor, pytest.raises(links.AnchoredDirectoryError):
        links.directory_private_to_current_user(anchor)


def unsafe_root(tmp_path: Path) -> Path:
    root = create_owned_directory(tmp_path / "shared", extra_aces="(A;;FR;;;WD)")
    if sys.platform != "win32":
        root.chmod(0o755)
    return root


def test_root_probe_refuses_public_access_before_writing_probe_files(tmp_path: Path) -> None:
    root = unsafe_root(tmp_path)
    before = sorted(path.name for path in root.iterdir())
    report = contract.probe_store_root(root=root, minimum_free_bytes=0)
    assert report.usable is False
    assert report.private_access is False
    assert sorted(path.name for path in root.iterdir()) == before


@pytest.mark.parametrize("min_writer", [1, 2])
def test_read_only_negotiation_never_accepts_public_root_access(
    tmp_path: Path, min_writer: int
) -> None:
    root = unsafe_root(tmp_path)
    contract.initialize_store_identity(root=root)
    record = root / contract.IDENTITY_LEAF
    data = json.loads(record.read_text())
    data["min_writer_version"] = min_writer
    record.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(contract.SharedAssetContractError, match=contract.INVALID_STORE):
        contract.store_access_mode(root=root)


def test_private_root_still_supports_read_write_and_newer_writer_read_only(tmp_path: Path) -> None:
    root = create_owned_directory(tmp_path / "private")
    contract.initialize_store_identity(root=root)
    report = contract.probe_store_root(root=root, minimum_free_bytes=0)
    assert report.private_access is True
    assert report.usable is True
    assert contract.store_access_mode(root=root) == "read_write"
    record = root / contract.IDENTITY_LEAF
    data = json.loads(record.read_text())
    data["min_writer_version"] = 2
    record.write_text(json.dumps(data), encoding="utf-8")
    assert contract.store_access_mode(root=root) == "read_only"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows descriptor lifetime")
@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "security",
        "different-owner",
        "invalid-acl",
        "ace-query",
        "outside-ace",
        "short-ace",
        "oversized-ace",
        "unaligned-ace",
        "truncated-sid",
        "invalid-sid",
        "unsupported-ace",
    ],
)
def test_windows_access_refuses_incomplete_evidence_and_releases_the_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    root = create_owned_directory(tmp_path / "private")
    api = links._windows_ownership_api()
    native_query = api.security.GetSecurityInfo
    native_ace = api.security.GetAce
    native_valid_acl = api.security.IsValidAcl
    native_free = api.kernel.LocalFree
    queried: list[int] = []
    allocated: list[int] = []
    freed: list[int] = []
    foreign_sid = ctypes.create_string_buffer(bytes.fromhex("010100000000000100000000"))
    outside_ace = ctypes.create_string_buffer(128)

    def query(*args: Any) -> int:
        queried.append(int(args[0].value))
        assert args[2] == 5  # Owner and DACL from one held-descriptor snapshot.
        result = int(native_query(*args))
        pointer = ctypes.cast(args[7], ctypes.POINTER(ctypes.c_void_p))[0]
        assert pointer is not None
        allocated.append(pointer)
        if failure == "security":
            return 5
        if failure == "different-owner":
            ctypes.cast(args[3], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.addressof(foreign_sid)
        return result

    def valid_acl(pointer: Any) -> int:
        result = int(native_valid_acl(pointer))
        return 0 if failure == "invalid-acl" else result

    def ace(*args: Any) -> int:
        result = int(native_ace(*args))
        if not result:
            return result
        pointer = ctypes.cast(args[2], ctypes.POINTER(ctypes.c_void_p))
        address = pointer[0]
        assert address is not None
        size = ctypes.c_ushort.from_address(address + 2)
        if failure == "ace-query":
            return 0
        if failure == "outside-ace":
            assert size.value < len(outside_ace)
            ctypes.memmove(outside_ace, address, size.value)
            pointer[0] = ctypes.addressof(outside_ace)
        elif failure == "short-ace":
            size.value = 12
        elif failure == "oversized-ace":
            size.value = 65532
        elif failure == "unaligned-ace":
            size.value -= 1
        elif failure == "truncated-sid":
            # The allocation remains intact, but its declared ACE ends before
            # the real SID's subauthorities. No invalid memory in a control.
            size.value = 16
        elif failure == "invalid-sid":
            ctypes.c_ubyte.from_address(address + 8).value = 0
        elif failure == "unsupported-ace":
            ctypes.c_ubyte.from_address(address).value = 9
        return result

    def free(pointer: Any) -> Any:
        freed.append(int(pointer.value))
        return native_free(pointer)

    monkeypatch.setattr(api.security, "GetSecurityInfo", query)
    monkeypatch.setattr(api.security, "IsValidAcl", valid_acl)
    monkeypatch.setattr(api.security, "GetAce", ace)
    monkeypatch.setattr(api.kernel, "LocalFree", free)
    monkeypatch.setattr(links, "_windows_ownership_api", lambda: api)
    with links.AnchoredDirectory(root, read_security=True) as anchor:
        held = anchor.handle
        anchor.path = root / "absent"
        if failure in {"none", "different-owner", "unsupported-ace"}:
            assert links.directory_private_to_current_user(anchor) is (failure == "none")
        else:
            with pytest.raises(links.AnchoredDirectoryError):
                links.directory_private_to_current_user(anchor)
        assert queried == [held]
        assert len(allocated) == 1
        assert freed == allocated
        assert anchor.handle == held


def test_unavailable_security_metadata_refuses_without_write_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = create_owned_directory(tmp_path / "private")
    contract.initialize_store_identity(root=root)

    def unavailable(_anchor: links.AnchoredDirectory) -> bool:
        raise links.AnchoredDirectoryError(links.CONTAINMENT_REFUSED)

    monkeypatch.setattr(contract, "directory_private_to_current_user", unavailable)
    before = sorted(path.name for path in root.iterdir())
    report = contract.probe_store_root(root=root, minimum_free_bytes=0)
    assert report.directory and report.no_reparse_points
    assert not report.private_access and not report.usable
    with pytest.raises(contract.SharedAssetContractError, match=contract.INVALID_STORE):
        contract.store_access_mode(root=root)
    assert sorted(path.name for path in root.iterdir()) == before
