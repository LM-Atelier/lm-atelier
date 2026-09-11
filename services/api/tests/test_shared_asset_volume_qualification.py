from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from directory_security import create_owned_directory

from local_lm import filesystem_links as links
from local_lm import shared_asset_contract_v1 as contract

windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows volume qualification")


def _reported_device(
    monkeypatch: pytest.MonkeyPatch, device_type: int, characteristics: int
) -> list[links.AnchoredDirectory]:
    queried: list[links.AnchoredDirectory] = []

    def query(anchor: links.AnchoredDirectory) -> links.DirectoryDeviceInformation:
        assert anchor.handle is not None
        queried.append(anchor)
        return links.DirectoryDeviceInformation(device_type, characteristics)

    # The unchanged parent has no import of this existing primitive. Keeping
    # setup valid there lets the parent fail on acceptance, not collection.
    monkeypatch.setattr(contract, "directory_device_information", query, raising=False)
    return queried


def _write_probe_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def rename(anchor: links.AnchoredDirectory) -> bool:
        calls.append("rename")
        return True

    def create(anchor: links.AnchoredDirectory) -> bool:
        calls.append("create")
        return True

    def space(anchor: links.AnchoredDirectory, minimum: int) -> bool:
        calls.append("space")
        return True

    monkeypatch.setattr(contract, "_probe_atomic_rename", rename)
    monkeypatch.setattr(contract, "_probe_exclusive_create", create)
    monkeypatch.setattr(contract, "_probe_free_space", space)
    return calls


@windows
@pytest.mark.parametrize(
    ("device_type", "characteristics"),
    [
        (0, 0),  # Missing device type is not a positive disk report.
        (2, 0),  # Optical.
        (8, 0),  # A different device type is not FILE_DEVICE_DISK.
        (0xFFFFFFFF, 0),
        (7, 0x1),  # Removable media.
        (7, 0x4),  # Floppy.
        (7, 0x8),  # Write-once.
        (7, 0x10),  # Remote.
        (7, 0x40),  # Virtual.
        (7, 0x1000),  # Terminal Services.
        (7, 0x2000),  # WebDAV.
        (7, 0x80000000),  # Unknown flags remain unsupported.
        (7, 0x4021),  # Portable hardware does not excuse removable media.
    ],
)
def test_ineligible_volume_prevents_every_write_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_type: int,
    characteristics: int,
) -> None:
    root = create_owned_directory(tmp_path / "store")
    queried = _reported_device(monkeypatch, device_type, characteristics)
    writes = _write_probe_calls(monkeypatch)

    report = contract.probe_store_root(root=root, minimum_free_bytes=0)

    assert not report.usable
    assert writes == []
    assert getattr(report, "windows_volume", None) == "ineligible"
    assert report.private_access
    assert len(queried) == 1
    assert list(root.iterdir()) == []
    with pytest.raises(contract.SharedAssetContractError, match=contract.INVALID_STORE):
        contract.require_usable_root(root=root, minimum_free_bytes=0)


@windows
@pytest.mark.parametrize("characteristics", [0, 0x2, 0x20, 0x100, 0x4000, 0x20000, 0x24122])
def test_understood_disk_report_qualifies_independently_of_write_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, characteristics: int
) -> None:
    root = create_owned_directory(tmp_path / "store")
    _reported_device(monkeypatch, 7, characteristics)
    _write_probe_calls(monkeypatch)
    # This check is about volume eligibility; a failed write still makes the
    # battery unusable even on an eligible (including read-only) device.
    monkeypatch.setattr(contract, "_probe_atomic_rename", lambda anchor: False)

    report = contract.probe_store_root(root=root, minimum_free_bytes=0)

    assert getattr(report, "windows_volume", None) == "eligible"
    assert not report.usable
    assert report.private_access
    assert not report.atomic_rename


@windows
@pytest.mark.parametrize("min_writer", [1, 2])
def test_volume_refusal_precedes_identity_negotiation_and_write_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, min_writer: int
) -> None:
    root = create_owned_directory(tmp_path / "store")
    contract.initialize_store_identity(root=root)
    identity = root / contract.IDENTITY_LEAF
    data = json.loads(identity.read_text())
    data["min_writer_version"] = min_writer
    identity.write_text(json.dumps(data), encoding="utf-8")
    before = identity.read_bytes()
    _reported_device(monkeypatch, 7, 0x10)
    writes = _write_probe_calls(monkeypatch)

    with pytest.raises(contract.SharedAssetContractError, match=contract.INVALID_STORE):
        contract.store_access_mode(root=root)

    assert writes == []
    assert identity.read_bytes() == before


@windows
def test_query_failure_refuses_instead_of_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = create_owned_directory(tmp_path / "store")
    contract.initialize_store_identity(root=root)
    calls = _write_probe_calls(monkeypatch)

    def refuse(anchor: links.AnchoredDirectory) -> links.DirectoryDeviceInformation:
        raise links.AnchoredDirectoryError("constructed unsupported query")

    monkeypatch.setattr(contract, "directory_device_information", refuse, raising=False)
    report = contract.probe_store_root(root=root, minimum_free_bytes=0)
    assert not report.usable
    assert getattr(report, "windows_volume", None) == "ineligible"
    assert calls == []
    with pytest.raises(contract.SharedAssetContractError, match=contract.INVALID_STORE):
        contract.store_access_mode(root=root)


@windows
def test_volume_query_keeps_the_security_checked_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = create_owned_directory(tmp_path / "store")
    contract.initialize_store_identity(root=root)
    seen: list[links.AnchoredDirectory] = []
    real_access = links.directory_private_to_current_user

    def check_access(anchor: links.AnchoredDirectory) -> bool:
        result = real_access(anchor)
        seen.append(anchor)
        # This label no longer resolves. Qualification must consume the held
        # object rather than opening its label again.
        anchor.path = root / "absent"
        return result

    monkeypatch.setattr(contract, "directory_private_to_current_user", check_access)
    queried = _reported_device(monkeypatch, 7, 0)
    assert contract.store_access_mode(root=root) == "read_write"
    assert queried
    assert seen
    assert all(anchor is seen[0] for anchor in [*seen, *queried])


@windows
def test_newer_writer_can_use_an_eligible_disk_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = create_owned_directory(tmp_path / "store")
    contract.initialize_store_identity(root=root)
    identity = root / contract.IDENTITY_LEAF
    data = json.loads(identity.read_text())
    data["min_writer_version"] = 2
    identity.write_text(json.dumps(data), encoding="utf-8")
    queried = _reported_device(monkeypatch, 7, 0x2)
    writes = _write_probe_calls(monkeypatch)
    assert contract.store_access_mode(root=root) == "read_only"
    assert len(queried) == 1
    assert writes == []


def test_unanchored_root_does_not_claim_a_volume_check(tmp_path: Path) -> None:
    report = contract.probe_store_root(root=tmp_path / "absent", minimum_free_bytes=0)
    assert not report.usable
    assert getattr(report, "windows_volume", None) == "not_checked"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX qualification scope")
def test_posix_does_not_claim_windows_volume_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = create_owned_directory(tmp_path / "store")
    contract.initialize_store_identity(root=root)

    def unexpected(anchor: links.AnchoredDirectory) -> links.DirectoryDeviceInformation:
        pytest.fail("Windows device query called on POSIX")

    monkeypatch.setattr(contract, "directory_device_information", unexpected, raising=False)
    report = contract.probe_store_root(root=root, minimum_free_bytes=0)
    assert getattr(report, "windows_volume", None) == "not_applicable"
    assert report.usable
    assert contract.store_access_mode(root=root) == "read_write"


@windows
def test_later_volume_query_failure_is_not_downgraded_to_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = create_owned_directory(tmp_path / "store")
    contract.initialize_store_identity(root=root)
    queries = 0
    writes = _write_probe_calls(monkeypatch)

    def query(anchor: links.AnchoredDirectory) -> links.DirectoryDeviceInformation:
        nonlocal queries
        queries += 1
        if queries > 1:
            raise links.AnchoredDirectoryError("constructed later query failure")
        return links.DirectoryDeviceInformation(7, 0)

    monkeypatch.setattr(contract, "directory_device_information", query, raising=False)
    with pytest.raises(contract.SharedAssetContractError, match=contract.INVALID_STORE):
        contract.store_access_mode(root=root)
    assert queries == 2
    assert writes == []
