from __future__ import annotations

import hashlib
import io
import re
import shutil
import stat
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .filesystem_links import is_link_or_reparse

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 4_096
MAX_ARCHIVE_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_FILE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_PATH_CHARACTERS = 1_024
MAX_ARCHIVE_COMPONENT_CHARACTERS = 255
MAX_RUNTIME_FILE_COUNT = 256
MAX_RUNTIME_FILE_BYTES = 1024 * 1024
MAX_RUNTIME_FILES_BYTES = 8 * 1024 * 1024

_ARCHIVE_HASH = re.compile(r"^[0-9a-fA-F]{64}$")
_NATIVE_SUFFIXES = {".dll", ".dylib", ".exe", ".pyd", ".so"}
_RESERVED_WINDOWS_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_SUPPORTED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_RUNTIME_DATA_SUFFIXES = {
    ".cfg",
    ".conf",
    ".csv",
    ".example",
    ".ini",
    ".json",
    ".md",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


class ComfyRegistryArchiveError(ValueError):
    pass


@dataclass(frozen=True)
class ComfyRegistryArchiveReport:
    archive_sha256: str
    manifest_sha256: str
    entry_count: int
    file_count: int
    expanded_bytes: int
    python_file_count: int
    dependency_manifests: tuple[str, ...]
    install_scripts: tuple[str, ...]
    startup_hooks: tuple[str, ...]
    native_files: tuple[str, ...]
    top_level_entries: tuple[str, ...]
    review_required: bool = True


@dataclass(frozen=True)
class ComfyRegistryRuntimeFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class _ValidatedEntry:
    info: zipfile.ZipInfo
    path: PurePosixPath
    key: str


def stage_comfy_registry_archive(
    archive_path: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    strip_single_root: bool = False,
) -> ComfyRegistryArchiveReport:
    """Validate and extract an immutable registry archive without executing it."""
    if expected_sha256 is not None and not _ARCHIVE_HASH.fullmatch(expected_sha256):
        raise ComfyRegistryArchiveError("invalid expected archive hash")
    if not isinstance(strip_single_root, bool):
        raise ComfyRegistryArchiveError("invalid archive root normalization")
    if destination.exists() or destination.is_symlink():
        raise ComfyRegistryArchiveError("archive staging destination already exists")
    if not destination.parent.is_dir():
        raise ComfyRegistryArchiveError("archive staging parent does not exist")

    payload = _read_bounded_archive(archive_path)
    archive_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and archive_sha256 != expected_sha256.lower():
        raise ComfyRegistryArchiveError("registry archive hash does not match")

    destination.mkdir()
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = _validate_entries(archive)
            if strip_single_root:
                entries = _without_single_archive_root(entries)
            expanded_bytes = sum(
                entry.info.file_size for entry in entries if not entry.info.is_dir()
            )
            if shutil.disk_usage(destination.parent).free < expanded_bytes:
                raise ComfyRegistryArchiveError(
                    "insufficient disk space for Comfy Registry archive"
                )
            report = _extract_entries(archive, entries, destination, archive_sha256)
    except ComfyRegistryArchiveError:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        shutil.rmtree(destination, ignore_errors=True)
        raise ComfyRegistryArchiveError("invalid Comfy Registry archive") from exc
    return report


def _without_single_archive_root(
    entries: tuple[_ValidatedEntry, ...],
) -> tuple[_ValidatedEntry, ...]:
    """Remove GitHub's synthetic repository wrapper from an inert archive."""
    roots = {entry.path.parts[0] for entry in entries}
    if len(roots) != 1:
        raise ComfyRegistryArchiveError("commit archive does not have one repository root")
    stripped: list[_ValidatedEntry] = []
    for entry in entries:
        parts = entry.path.parts[1:]
        if not parts:
            if not entry.info.is_dir():
                raise ComfyRegistryArchiveError("commit archive repository root is not a directory")
            continue
        path = PurePosixPath(*parts)
        key = "/".join(unicodedata.normalize("NFC", part).casefold() for part in parts)
        stripped.append(_ValidatedEntry(entry.info, path, key))
    if not stripped or not any(not entry.info.is_dir() for entry in stripped):
        raise ComfyRegistryArchiveError("commit archive repository root is empty")
    return tuple(stripped)


def verify_staged_comfy_registry_archive(
    destination: Path,
    *,
    expected_manifest_sha256: str,
    expected_file_count: int,
    expected_expanded_bytes: int,
    runtime_files: tuple[ComfyRegistryRuntimeFile, ...] = (),
) -> None:
    """Re-hash inert staged node code before it is allowed into a launch."""
    if (
        not _ARCHIVE_HASH.fullmatch(expected_manifest_sha256)
        or isinstance(expected_file_count, bool)
        or not isinstance(expected_file_count, int)
        or expected_file_count < 0
        or isinstance(expected_expanded_bytes, bool)
        or not isinstance(expected_expanded_bytes, int)
        or expected_expanded_bytes < 0
    ):
        raise ComfyRegistryArchiveError("invalid staged Registry archive identity")
    expected_runtime = _validated_runtime_files(runtime_files)
    observed = snapshot_staged_comfy_registry_files(destination)
    runtime_by_path = {item.path: item for item in expected_runtime}
    files: list[tuple[str, int, str]] = []
    expanded_bytes = 0
    seen_runtime: set[str] = set()
    for item in observed:
        runtime = runtime_by_path.get(item.path)
        if runtime is not None:
            if item != runtime:
                raise ComfyRegistryArchiveError("staged Registry runtime file has changed")
            seen_runtime.add(item.path)
            continue
        files.append((item.path, item.size, item.sha256))
        expanded_bytes += item.size
    if seen_runtime != set(runtime_by_path):
        raise ComfyRegistryArchiveError("staged Registry runtime file is missing")
    manifest = "".join(
        f"{path}{chr(0)}{size}{chr(0)}{digest}{chr(10)}" for path, size, digest in sorted(files)
    )
    if (
        len(files) != expected_file_count
        or expanded_bytes != expected_expanded_bytes
        or hashlib.sha256(manifest.encode()).hexdigest() != expected_manifest_sha256.lower()
    ):
        raise ComfyRegistryArchiveError("staged Registry archive contents have changed")


def snapshot_staged_comfy_registry_files(
    destination: Path,
) -> tuple[ComfyRegistryRuntimeFile, ...]:
    """Capture exact non-cache file identities without trusting their contents."""
    if _is_link_or_reparse(destination) or not destination.is_dir():
        raise ComfyRegistryArchiveError("staged Registry archive is missing or unsafe")
    files: list[ComfyRegistryRuntimeFile] = []
    expanded_bytes = 0
    try:
        for path in sorted(destination.rglob("*")):
            if _is_link_or_reparse(path):
                raise ComfyRegistryArchiveError("staged Registry archive contains a link")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ComfyRegistryArchiveError("staged Registry archive contains a special file")
            relative_path = path.relative_to(destination)
            if _is_runtime_python_cache(destination, relative_path):
                continue
            size = path.stat().st_size
            expanded_bytes += size
            if (
                len(files) >= MAX_ARCHIVE_ENTRIES + MAX_RUNTIME_FILE_COUNT
                or expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES + MAX_RUNTIME_FILES_BYTES
            ):
                raise ComfyRegistryArchiveError("staged Registry archive exceeds its limits")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            files.append(
                ComfyRegistryRuntimeFile(
                    relative_path.as_posix(),
                    size,
                    digest.hexdigest(),
                )
            )
    except OSError as exc:
        raise ComfyRegistryArchiveError("could not verify staged Registry archive") from exc
    return tuple(sorted(files, key=lambda item: item.path))


def capture_staged_comfy_registry_runtime_files(
    destination: Path,
    *,
    before_start: tuple[ComfyRegistryRuntimeFile, ...],
    expected_manifest_sha256: str,
    expected_file_count: int,
    expected_expanded_bytes: int,
    runtime_files: tuple[ComfyRegistryRuntimeFile, ...] = (),
) -> tuple[ComfyRegistryRuntimeFile, ...]:
    """Record only bounded inert files added by one successful trusted startup."""
    before = _validated_snapshot(before_start)
    existing_runtime = _validated_runtime_files(runtime_files)
    after = snapshot_staged_comfy_registry_files(destination)
    after_by_path = {item.path: item for item in after}
    for item in before:
        if after_by_path.get(item.path) != item:
            raise ComfyRegistryArchiveError("staged Registry files changed during startup")
    before_paths = {item.path for item in before}
    additions = tuple(item for item in after if item.path not in before_paths)
    combined = _validated_runtime_files(
        tuple(sorted((*existing_runtime, *additions), key=lambda item: item.path)),
        destination=destination,
    )
    verify_staged_comfy_registry_archive(
        destination,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_file_count=expected_file_count,
        expected_expanded_bytes=expected_expanded_bytes,
        runtime_files=combined,
    )
    return combined


def parse_comfy_registry_runtime_files(value: object) -> tuple[ComfyRegistryRuntimeFile, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ComfyRegistryArchiveError("invalid staged Registry runtime evidence")
    files: list[ComfyRegistryRuntimeFile] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ComfyRegistryArchiveError("invalid staged Registry runtime evidence")
        files.append(ComfyRegistryRuntimeFile(item["path"], item["size"], item["sha256"]))
    return _validated_runtime_files(tuple(files))


def comfy_registry_runtime_files_json(
    files: tuple[ComfyRegistryRuntimeFile, ...],
) -> list[dict[str, object]]:
    return [
        {"path": item.path, "size": item.size, "sha256": item.sha256}
        for item in _validated_runtime_files(files)
    ]


def _validated_snapshot(
    files: tuple[ComfyRegistryRuntimeFile, ...],
) -> tuple[ComfyRegistryRuntimeFile, ...]:
    if not isinstance(files, tuple):
        raise ComfyRegistryArchiveError("invalid staged Registry startup snapshot")
    result = tuple(_validated_file_identity(item) for item in files)
    if not _identities_are_sorted_and_unique(result):
        raise ComfyRegistryArchiveError("invalid staged Registry startup snapshot")
    return result


def _validated_runtime_files(
    files: tuple[ComfyRegistryRuntimeFile, ...],
    *,
    destination: Path | None = None,
) -> tuple[ComfyRegistryRuntimeFile, ...]:
    if not isinstance(files, tuple) or len(files) > MAX_RUNTIME_FILE_COUNT:
        raise ComfyRegistryArchiveError("invalid staged Registry runtime evidence")
    result: list[ComfyRegistryRuntimeFile] = []
    total = 0
    for value in files:
        item = _validated_file_identity(value)
        suffix = PurePosixPath(item.path).suffix.casefold()
        if suffix not in _RUNTIME_DATA_SUFFIXES or item.size > MAX_RUNTIME_FILE_BYTES:
            raise ComfyRegistryArchiveError("Registry runtime file is not bounded inert data")
        total += item.size
        if total > MAX_RUNTIME_FILES_BYTES:
            raise ComfyRegistryArchiveError("Registry runtime files exceed their size limit")
        if destination is not None:
            try:
                payload = destination.joinpath(*PurePosixPath(item.path).parts).read_bytes()
                payload.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ComfyRegistryArchiveError(
                    "Registry runtime file is not bounded UTF-8 data"
                ) from exc
        result.append(item)
    if not _identities_are_sorted_and_unique(tuple(result)):
        raise ComfyRegistryArchiveError("invalid staged Registry runtime evidence")
    return tuple(result)


def _validated_file_identity(value: object) -> ComfyRegistryRuntimeFile:
    if not isinstance(value, ComfyRegistryRuntimeFile):
        raise ComfyRegistryArchiveError("invalid staged Registry file identity")
    try:
        path, _ = _safe_member_path(value.path)
    except ComfyRegistryArchiveError as exc:
        raise ComfyRegistryArchiveError("invalid staged Registry file identity") from exc
    if (
        path.as_posix() != value.path
        or isinstance(value.size, bool)
        or not isinstance(value.size, int)
        or value.size < 0
        or not isinstance(value.sha256, str)
        or not _ARCHIVE_HASH.fullmatch(value.sha256)
    ):
        raise ComfyRegistryArchiveError("invalid staged Registry file identity")
    return ComfyRegistryRuntimeFile(value.path, value.size, value.sha256.lower())


def _identities_are_sorted_and_unique(
    files: tuple[ComfyRegistryRuntimeFile, ...],
) -> bool:
    paths = tuple(item.path for item in files)
    keys = tuple(_safe_member_path(item.path)[1] for item in files)
    return paths == tuple(sorted(paths)) and len(keys) == len(set(keys))


def _is_runtime_python_cache(destination: Path, path: Path) -> bool:
    """Recognize only bytecode Python writes beside reviewed source at import."""

    if (
        path.suffix.casefold() != ".pyc"
        or len(path.parts) < 2
        or path.parts[-2].casefold() != "__pycache__"
    ):
        return False
    source_name, separator, implementation_tag = path.name.partition(".")
    if not separator or not source_name or not implementation_tag.casefold().startswith("cpython-"):
        return False
    source = destination.joinpath(*path.parts[:-2], f"{source_name}.py")
    return source.is_file() and not _is_link_or_reparse(source)


def _read_bounded_archive(path: Path) -> bytes:
    try:
        with path.open("rb") as source:
            payload = source.read(MAX_ARCHIVE_BYTES + 1)
    except OSError as exc:
        raise ComfyRegistryArchiveError("could not read Comfy Registry archive") from exc
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ComfyRegistryArchiveError("Comfy Registry archive exceeds the size limit")
    return payload


def _validate_entries(archive: zipfile.ZipFile) -> tuple[_ValidatedEntry, ...]:
    infos = archive.infolist()
    if not infos:
        raise ComfyRegistryArchiveError("Comfy Registry archive is empty")
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise ComfyRegistryArchiveError("Comfy Registry archive has too many entries")

    entries: list[_ValidatedEntry] = []
    kinds: dict[str, str] = {}
    expanded_bytes = 0
    for info in infos:
        path, key = _safe_member_path(info.filename)
        if key in kinds:
            raise ComfyRegistryArchiveError("Comfy Registry archive has duplicate paths")
        is_directory = info.is_dir()
        kinds[key] = "directory" if is_directory else "file"
        _validate_entry_type(info, is_directory)
        if not is_directory:
            if info.file_size > MAX_ARCHIVE_FILE_BYTES:
                raise ComfyRegistryArchiveError("Comfy Registry archive file is too large")
            expanded_bytes += info.file_size
            if expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
                raise ComfyRegistryArchiveError(
                    "Comfy Registry archive expands beyond the size limit"
                )
        entries.append(_ValidatedEntry(info, path, key))

    for entry in entries:
        parts = entry.key.split("/")
        for index in range(1, len(parts)):
            if kinds.get("/".join(parts[:index])) == "file":
                raise ComfyRegistryArchiveError(
                    "Comfy Registry archive has a file-directory collision"
                )
        if not entry.info.is_dir() and kinds.get(entry.key) == "directory":
            raise ComfyRegistryArchiveError("Comfy Registry archive has a file-directory collision")
    return tuple(entries)


def _safe_member_path(value: str) -> tuple[PurePosixPath, str]:
    if (
        not value
        or "\x00" in value
        or chr(92) in value
        or value.startswith("/")
        or len(value) > MAX_ARCHIVE_PATH_CHARACTERS
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise ComfyRegistryArchiveError("Comfy Registry archive contains an unsafe path")
    normalized = value[:-1] if value.endswith("/") else value
    raw_parts = normalized.split("/")
    if not normalized or any(part in {"", ".", ".."} for part in raw_parts):
        raise ComfyRegistryArchiveError("Comfy Registry archive contains an unsafe path")
    for part in raw_parts:
        if (
            len(part) > MAX_ARCHIVE_COMPONENT_CHARACTERS
            or part.rstrip(" .") != part
            or ":" in part
            or any(ord(character) < 32 or ord(character) == 127 for character in part)
            or part.split(".", 1)[0].upper() in _RESERVED_WINDOWS_NAMES
        ):
            raise ComfyRegistryArchiveError("Comfy Registry archive contains an unsafe path")
    path = PurePosixPath(*raw_parts)
    key = "/".join(unicodedata.normalize("NFC", part).casefold() for part in path.parts)
    return path, key


def _validate_entry_type(info: zipfile.ZipInfo, is_directory: bool) -> None:
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise ComfyRegistryArchiveError("Comfy Registry archive cannot contain links")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ComfyRegistryArchiveError("Comfy Registry archive cannot contain special files")
    if info.flag_bits & 0x1:
        raise ComfyRegistryArchiveError("Comfy Registry archive cannot be encrypted")
    if info.compress_type not in _SUPPORTED_COMPRESSION:
        raise ComfyRegistryArchiveError("Comfy Registry archive uses unsupported compression")
    if is_directory and info.file_size:
        raise ComfyRegistryArchiveError("Comfy Registry archive has an invalid directory")


def _extract_entries(
    archive: zipfile.ZipFile,
    entries: tuple[_ValidatedEntry, ...],
    destination: Path,
    archive_sha256: str,
) -> ComfyRegistryArchiveReport:
    file_manifest: list[tuple[str, int, str]] = []
    dependency_manifests: list[str] = []
    install_scripts: list[str] = []
    startup_hooks: list[str] = []
    native_files: list[str] = []
    python_file_count = 0
    expanded_bytes = 0

    for entry in entries:
        target = destination.joinpath(*entry.path.parts)
        if entry.info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        written = 0
        with archive.open(entry.info) as source, target.open("xb") as output:
            while chunk := source.read(1024 * 1024):
                written += len(chunk)
                if written > entry.info.file_size or written > MAX_ARCHIVE_FILE_BYTES:
                    raise ComfyRegistryArchiveError(
                        "Comfy Registry archive entry exceeds its declared size"
                    )
                digest.update(chunk)
                output.write(chunk)
        if written != entry.info.file_size:
            raise ComfyRegistryArchiveError(
                "Comfy Registry archive entry size does not match metadata"
            )
        relative = entry.path.as_posix()
        suffix = entry.path.suffix.lower()
        lower_name = entry.path.name.lower()
        expanded_bytes += written
        python_file_count += suffix == ".py"
        if lower_name in {"pyproject.toml", "requirements.txt", "setup.cfg"} or (
            lower_name.startswith("requirements-") and lower_name.endswith(".txt")
        ):
            dependency_manifests.append(relative)
        if lower_name in {"install.py", "setup.py"}:
            install_scripts.append(relative)
        if lower_name == "prestartup_script.py":
            startup_hooks.append(relative)
        if suffix in _NATIVE_SUFFIXES:
            native_files.append(relative)
        file_manifest.append((relative, written, digest.hexdigest()))

    manifest = "".join(
        f"{path}\x00{size}\x00{digest}\n" for path, size, digest in sorted(file_manifest)
    )
    return ComfyRegistryArchiveReport(
        archive_sha256=archive_sha256,
        manifest_sha256=hashlib.sha256(manifest.encode()).hexdigest(),
        entry_count=len(entries),
        file_count=len(file_manifest),
        expanded_bytes=expanded_bytes,
        python_file_count=python_file_count,
        dependency_manifests=tuple(sorted(dependency_manifests)),
        install_scripts=tuple(sorted(install_scripts)),
        startup_hooks=tuple(sorted(startup_hooks)),
        native_files=tuple(sorted(native_files)),
        top_level_entries=tuple(sorted({entry.path.parts[0] for entry in entries})),
    )


def _is_link_or_reparse(path: Path) -> bool:
    return is_link_or_reparse(
        path,
        missing="assume_link",
        unreadable="assume_link",
    )
