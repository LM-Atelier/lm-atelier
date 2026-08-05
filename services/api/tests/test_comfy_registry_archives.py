from __future__ import annotations

import hashlib
import stat
import zipfile
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_lm import comfy_registry_archives as registry_archives
from local_lm.comfy_registry_archives import (
    MAX_ARCHIVE_ENTRIES,
    ComfyRegistryArchiveError,
    _safe_member_path,
    stage_comfy_registry_archive,
    verify_staged_comfy_registry_archive,
)


def write_archive(
    path: Path,
    entries: Iterable[tuple[str | zipfile.ZipInfo, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> Path:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return path


def test_archive_is_staged_inertly_with_a_deterministic_report(tmp_path: Path) -> None:
    source = write_archive(
        tmp_path / "node.zip",
        [
            ("node/__init__.py", b"NODE_CLASS_MAPPINGS = {}\n"),
            ("node/requirements.txt", b"pillow>=12\n"),
            ("node/install.py", b"raise RuntimeError('must not execute')\n"),
            ("node/prestartup_script.py", b"raise RuntimeError('must not execute')\n"),
            ("node/native.pyd", b"not-a-real-extension"),
        ],
    )
    destination = tmp_path / "staged"

    report = stage_comfy_registry_archive(source, destination)

    assert (destination / "node" / "__init__.py").read_bytes().startswith(b"NODE_CLASS")
    assert report.archive_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report.entry_count == 5
    assert report.file_count == 5
    assert report.python_file_count == 3
    assert report.dependency_manifests == ("node/requirements.txt",)
    assert report.install_scripts == ("node/install.py",)
    assert report.startup_hooks == ("node/prestartup_script.py",)
    assert report.native_files == ("node/native.pyd",)
    assert report.top_level_entries == ("node",)
    assert report.review_required
    verify_staged_comfy_registry_archive(
        destination,
        expected_manifest_sha256=report.manifest_sha256,
        expected_file_count=report.file_count,
        expected_expanded_bytes=report.expanded_bytes,
    )

    (destination / "node" / "__init__.py").write_bytes(b"changed")
    with pytest.raises(ComfyRegistryArchiveError, match="contents have changed"):
        verify_staged_comfy_registry_archive(
            destination,
            expected_manifest_sha256=report.manifest_sha256,
            expected_file_count=report.file_count,
            expected_expanded_bytes=report.expanded_bytes,
        )


def test_orphaned_python_cache_invalidates_reviewed_source(tmp_path: Path) -> None:
    source = write_archive(tmp_path / "node.zip", [("node/main.py", b"value = 1\n")])
    destination = tmp_path / "staged"
    report = stage_comfy_registry_archive(source, destination)
    cache = destination / "node" / "__pycache__" / "other.cpython-313.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"unreviewed bytecode")

    with pytest.raises(ComfyRegistryArchiveError, match="contents have changed"):
        verify_staged_comfy_registry_archive(
            destination,
            expected_manifest_sha256=report.manifest_sha256,
            expected_file_count=report.file_count,
            expected_expanded_bytes=report.expanded_bytes,
        )


def test_runtime_python_cache_does_not_invalidate_reviewed_source(tmp_path: Path) -> None:
    source = write_archive(
        tmp_path / "node.zip",
        [("node/prestartup_script.py", b"NODE_CLASS_MAPPINGS = {}\n")],
    )
    destination = tmp_path / "staged"
    report = stage_comfy_registry_archive(source, destination)
    cache = destination / "node" / "__pycache__" / "prestartup_script.cpython-313.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"runtime bytecode")

    verify_staged_comfy_registry_archive(
        destination,
        expected_manifest_sha256=report.manifest_sha256,
        expected_file_count=report.file_count,
        expected_expanded_bytes=report.expanded_bytes,
    )

    (cache.parent / "unexpected.txt").write_text("not runtime bytecode", encoding="utf-8")
    with pytest.raises(ComfyRegistryArchiveError, match="contents have changed"):
        verify_staged_comfy_registry_archive(
            destination,
            expected_manifest_sha256=report.manifest_sha256,
            expected_file_count=report.file_count,
            expected_expanded_bytes=report.expanded_bytes,
        )


def test_manifest_hash_is_independent_of_zip_entry_order(tmp_path: Path) -> None:
    files = [("node/a.py", b"a"), ("node/b.py", b"b")]
    first = write_archive(tmp_path / "first.zip", files)
    second = write_archive(tmp_path / "second.zip", reversed(files))

    first_report = stage_comfy_registry_archive(first, tmp_path / "first")
    second_report = stage_comfy_registry_archive(second, tmp_path / "second")

    assert first_report.manifest_sha256 == second_report.manifest_sha256


def test_expected_archive_hash_is_enforced_before_extraction(tmp_path: Path) -> None:
    source = write_archive(tmp_path / "node.zip", [("node.py", b"pass\n")])
    destination = tmp_path / "staged"

    with pytest.raises(ComfyRegistryArchiveError, match="hash does not match"):
        stage_comfy_registry_archive(source, destination, expected_sha256="0" * 64)

    assert not destination.exists()


@pytest.mark.parametrize("value", ["", "abc", "g" * 64])
def test_invalid_expected_hash_is_rejected(tmp_path: Path, value: str) -> None:
    source = write_archive(tmp_path / "node.zip", [("node.py", b"pass\n")])
    with pytest.raises(ComfyRegistryArchiveError, match="invalid expected archive hash"):
        stage_comfy_registry_archive(
            source,
            tmp_path / "staged",
            expected_sha256=value,
        )


@pytest.mark.parametrize(
    "name",
    [
        "../node.py",
        "/node.py",
        "C:/node.py",
        "node//file.py",
        "node/./file.py",
        "node/../file.py",
        "node/CON.py",
        "node/trailing. ",
        "node/bad:name.py",
    ],
)
def test_unsafe_paths_are_rejected_and_staging_is_removed(
    tmp_path: Path,
    name: str,
) -> None:
    source = write_archive(tmp_path / "node.zip", [(name, b"pass\n")])
    destination = tmp_path / "staged"

    with pytest.raises(ComfyRegistryArchiveError, match="unsafe path"):
        stage_comfy_registry_archive(source, destination)

    assert not destination.exists()


def test_backslash_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ComfyRegistryArchiveError, match="unsafe path"):
        _safe_member_path("node\\file.py")


def test_nul_truncated_zip_path_is_rejected(tmp_path: Path) -> None:
    source = write_archive(tmp_path / "node.zip", [("node/\x00bad.py", b"pass\n")])
    with pytest.raises(ComfyRegistryArchiveError, match="invalid directory"):
        stage_comfy_registry_archive(source, tmp_path / "staged")


@pytest.mark.parametrize(
    "names",
    [
        ("Node.py", "node.py"),
        ("caf\N{LATIN SMALL LETTER E WITH ACUTE}.py", "cafe\N{COMBINING ACUTE ACCENT}.py"),
    ],
)
def test_filesystem_equivalent_duplicate_paths_are_rejected(
    tmp_path: Path,
    names: tuple[str, str],
) -> None:
    source = write_archive(
        tmp_path / "node.zip",
        [(names[0], b"first"), (names[1], b"second")],
    )
    with pytest.raises(ComfyRegistryArchiveError, match="duplicate paths"):
        stage_comfy_registry_archive(source, tmp_path / "staged")


def test_file_directory_collisions_are_rejected(tmp_path: Path) -> None:
    source = write_archive(
        tmp_path / "node.zip",
        [("node", b"file"), ("node/module.py", b"pass\n")],
    )
    with pytest.raises(ComfyRegistryArchiveError, match="file-directory collision"):
        stage_comfy_registry_archive(source, tmp_path / "staged")


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600])
def test_links_and_special_files_are_rejected(tmp_path: Path, mode: int) -> None:
    info = zipfile.ZipInfo("node/unsafe")
    info.create_system = 3
    info.external_attr = mode << 16
    source = write_archive(tmp_path / "node.zip", [(info, b"target")])

    with pytest.raises(ComfyRegistryArchiveError, match="links|special files"):
        stage_comfy_registry_archive(source, tmp_path / "staged")


def test_unsupported_compression_is_rejected(tmp_path: Path) -> None:
    source = write_archive(
        tmp_path / "node.zip",
        [("node.py", b"pass\n")],
        compression=zipfile.ZIP_BZIP2,
    )
    with pytest.raises(ComfyRegistryArchiveError, match="unsupported compression"):
        stage_comfy_registry_archive(source, tmp_path / "staged")


def test_compressed_archive_size_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_archive(tmp_path / "node.zip", [("node.py", b"pass\n")])
    monkeypatch.setattr(registry_archives, "MAX_ARCHIVE_BYTES", 8)

    with pytest.raises(ComfyRegistryArchiveError, match="archive exceeds the size limit"):
        stage_comfy_registry_archive(source, tmp_path / "staged")


def test_per_file_and_expanded_sizes_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_archive(tmp_path / "node.zip", [("node.py", b"123456789")])
    monkeypatch.setattr(registry_archives, "MAX_ARCHIVE_FILE_BYTES", 8)
    monkeypatch.setattr(registry_archives, "MAX_ARCHIVE_EXPANDED_BYTES", 8)

    with pytest.raises(ComfyRegistryArchiveError, match="file is too large"):
        stage_comfy_registry_archive(source, tmp_path / "staged")


def test_insufficient_disk_space_is_rejected_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = write_archive(tmp_path / "node.zip", [("node.py", b"123456789")])
    monkeypatch.setattr(
        registry_archives.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=8),
    )
    destination = tmp_path / "staged"

    with pytest.raises(ComfyRegistryArchiveError, match="insufficient disk space"):
        stage_comfy_registry_archive(source, destination)
    assert not destination.exists()


def test_entry_count_is_bounded(tmp_path: Path) -> None:
    source = write_archive(
        tmp_path / "node.zip",
        ((f"node/{index}.txt", b"") for index in range(MAX_ARCHIVE_ENTRIES + 1)),
        compression=zipfile.ZIP_STORED,
    )
    with pytest.raises(ComfyRegistryArchiveError, match="too many entries"):
        stage_comfy_registry_archive(source, tmp_path / "staged")


def test_empty_archive_and_existing_destination_are_rejected(tmp_path: Path) -> None:
    empty = write_archive(tmp_path / "empty.zip", [])
    with pytest.raises(ComfyRegistryArchiveError, match="archive is empty"):
        stage_comfy_registry_archive(empty, tmp_path / "empty-stage")

    source = write_archive(tmp_path / "node.zip", [("node.py", b"pass\n")])
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(ComfyRegistryArchiveError, match="already exists"):
        stage_comfy_registry_archive(source, destination)


def test_explicit_directory_after_implicit_parent_is_supported(tmp_path: Path) -> None:
    source = write_archive(
        tmp_path / "node.zip",
        [("node/module.py", b"pass\n"), ("node/", b"")],
    )
    report = stage_comfy_registry_archive(source, tmp_path / "staged")
    assert report.file_count == 1
