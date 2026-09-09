from __future__ import annotations

import ctypes
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from local_lm import filesystem_links
from local_lm.filesystem_links import AnchoredDirectory, AnchoredDirectoryError


@pytest.mark.skipif(os.name != "nt", reason="native Windows device query")
def test_device_information_uses_the_held_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "held"
    root.mkdir()
    with AnchoredDirectory(root) as anchor:
        held = anchor.handle
        assert held is not None
        api = filesystem_links._windows_api()
        native_query = api.ntdll.NtQueryVolumeInformationFile
        calls: list[tuple[int, int, int]] = []

        def query(*args: Any) -> int:
            calls.append((int(args[0].value), int(args[3].value), int(args[4].value)))
            return int(native_query(*args))

        monkeypatch.setattr(api.ntdll, "NtQueryVolumeInformationFile", query)
        monkeypatch.setattr(filesystem_links, "_windows_api", lambda: api)
        anchor.path = tmp_path / "not-the-held-directory"
        result = filesystem_links.directory_device_information(anchor)

        assert calls == [(held, 8, 4)]
        assert result.device_type > 0
        assert 0 <= result.characteristics <= 0xFFFFFFFF
        assert anchor.handle == held


def _query_reply(
    api: Any, *, device_type: int, characteristics: int, status: int, returned: int
) -> Callable[..., int]:
    def query(
        _handle: Any, block: Any, information: Any, length: Any, information_class: Any
    ) -> int:
        assert length.value == 8
        assert information_class.value == 4
        result = ctypes.cast(information, ctypes.POINTER(api.FileFsDeviceInformation)).contents
        result.DeviceType = device_type
        result.Characteristics = characteristics
        status_block = ctypes.cast(block, ctypes.POINTER(api.IoStatusBlock)).contents
        status_block.Information = returned
        return status

    return query


@pytest.mark.skipif(os.name != "nt", reason="native Windows structure layout")
@pytest.mark.parametrize(("device_type", "characteristics"), [(7, 0), (2, 0x80000011)])
def test_device_information_preserves_the_reported_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_type: int, characteristics: int
) -> None:
    with AnchoredDirectory(tmp_path) as anchor:
        api = filesystem_links._windows_api()
        monkeypatch.setattr(
            api.ntdll,
            "NtQueryVolumeInformationFile",
            _query_reply(
                api,
                device_type=device_type,
                characteristics=characteristics,
                status=0,
                returned=8,
            ),
        )
        monkeypatch.setattr(filesystem_links, "_windows_api", lambda: api)
        result = filesystem_links.directory_device_information(anchor)
        assert (result.device_type, result.characteristics) == (device_type, characteristics)


@pytest.mark.skipif(os.name != "nt", reason="native Windows query result")
@pytest.mark.parametrize(
    ("status", "returned"), [(0xC0000022, 8), (0x103, 8), (0, 0), (0, 7), (0, 9)]
)
def test_an_unsuccessful_or_incomplete_device_query_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int, returned: int
) -> None:
    with AnchoredDirectory(tmp_path) as anchor:
        api = filesystem_links._windows_api()
        monkeypatch.setattr(
            api.ntdll,
            "NtQueryVolumeInformationFile",
            _query_reply(api, device_type=7, characteristics=0, status=status, returned=returned),
        )
        monkeypatch.setattr(filesystem_links, "_windows_api", lambda: api)
        with pytest.raises(AnchoredDirectoryError):
            filesystem_links.directory_device_information(anchor)


def test_a_closed_directory_is_refused_before_any_device_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor = AnchoredDirectory(tmp_path)
    anchor.close()

    def unexpected_query() -> None:
        raise AssertionError("a closed directory cannot query a device")

    monkeypatch.setattr(filesystem_links, "_windows_api", unexpected_query)
    with pytest.raises(AnchoredDirectoryError):
        filesystem_links.directory_device_information(anchor)


@pytest.mark.skipif(os.name == "nt", reason="POSIX capability refusal")
def test_posix_does_not_guess_windows_device_information(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_query() -> None:
        raise AssertionError("POSIX cannot use a Windows device query")

    monkeypatch.setattr(filesystem_links, "_windows_api", unexpected_query)
    with AnchoredDirectory(tmp_path) as anchor, pytest.raises(AnchoredDirectoryError):
        filesystem_links.directory_device_information(anchor)
