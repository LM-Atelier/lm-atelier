"""Choose the package index for each Python project and read what a static index omits.

Wheels come from the Python Package Index, except for projects NVIDIA publishes
only on its own index. Routing is by exact project name and never falls back
from one index to the other, so a same-named upload elsewhere is never used.

NVIDIA's index is a static HTML page that lists each wheel's SHA-256 but not its
size or core metadata, and serves no separate metadata files. Both are read from
the wheel itself with a few bounded range requests; the downloader later checks
that the verified wheel holds the same METADATA.
"""

from __future__ import annotations

import html
import re
import struct
import zlib
from dataclasses import dataclass
from urllib.parse import unquote, urljoin, urlparse

import httpx

NVIDIA_PACKAGE_INDEX_HOST = "pypi.nvidia.com"
NVIDIA_PACKAGE_INDEX_PROJECTS = frozenset({"nvidia-vfx"})
PYPI_FILES_HOST = "files.pythonhosted.org"
SIMPLE_HTML_MEDIA_TYPES = frozenset({"text/html", "application/vnd.pypi.simple.v1+html"})
MAX_SIMPLE_HTML_FILES = 256
MAX_SIMPLE_HTML_PAGE_BYTES = 128 * 1024
# A listed wheel whose size and metadata could not be read keeps its reason under
# this key instead; it refuses resolution only if it is the wheel selected.
UNREAD_WHEEL_KEY = "_unread"
MAX_WHEEL_URL_CHARACTERS = 2_000
MAX_REMOTE_WHEEL_BYTES = 0xFFFFFFFF
MAX_REMOTE_WHEEL_ENTRIES = 0xFFFF - 1
MAX_REMOTE_WHEEL_DIRECTORY_BYTES = 4 * 1024 * 1024
MAX_REMOTE_WHEEL_METADATA_BYTES = 1024 * 1024
# The end record is 22 bytes and may be followed by a comment of up to 0xFFFF bytes.
REMOTE_WHEEL_TAIL_BYTES = 22 + 0xFFFF
REMOTE_READ_BUDGET_EXCEEDED = "project_metadata_total_too_large"
_CENTRAL_ENTRY = struct.Struct("<4sHHHHHHIIIHHHHHII")
_END_OF_DIRECTORY = struct.Struct("<4sHHHHIIH")
_LOCAL_ENTRY = struct.Struct("<4sHHHHHIIIHH")
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_END_SIGNATURE = b"PK\x05\x06"
_LOCAL_SIGNATURE = b"PK\x03\x04"
_CONTENT_RANGE = re.compile(r"bytes ([0-9]{1,12})-([0-9]{1,12})/([0-9]{1,12})\Z")
_SHA256_FRAGMENT = re.compile(r"sha256=([0-9a-fA-F]{64})\Z")
# Every regular expression below matches without backtracking across tokens, so
# reading a page takes time linear in its size whatever the page holds.
_HTML_TOKEN = re.compile(r"<[^<>]*>|[^<]+")
_TAG_NAME = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_ATTRIBUTE = re.compile(r"""\s*([^\s"'=<>/`]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""")
_UNICODE_PATH_EXTRA = 0x7075


class ComfyRegistryPackageIndexError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SimpleIndexFile:
    filename: str
    url: str
    sha256: str | None
    requires_python: str | None
    yanked: bool | str


@dataclass(frozen=True)
class RemoteWheelMetadata:
    size_bytes: int
    metadata: bytes
    downloaded_bytes: int


class RemoteReadBudget:
    """Bytes a run of range reads may still request, charged before each request is sent."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.spent = 0

    def charge(self, size: int) -> None:
        if self.spent + size > self.limit:
            raise ComfyRegistryPackageIndexError(
                REMOTE_READ_BUDGET_EXCEEDED, "Package index reads exceed the aggregate size limit"
            )
        self.spent += size


def served_by_nvidia_index(name: str) -> bool:
    return name in NVIDIA_PACKAGE_INDEX_PROJECTS


def project_index_url(name: str) -> str:
    if served_by_nvidia_index(name):
        return f"https://{NVIDIA_PACKAGE_INDEX_HOST}/{name}/"
    return f"https://pypi.org/simple/{name}/"


def wheel_url_allowed(name: str, url: str, filename: str) -> bool:
    """Return whether a wheel URL points into the one index that serves this project."""
    if len(url) > MAX_WHEEL_URL_CHARACTERS or any(not "!" <= character <= "~" for character in url):
        return False
    parsed = urlparse(url)
    decoded = [unquote(segment) for segment in parsed.path.split("/") if segment]
    try:
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or port
        or parsed.query
        or parsed.fragment
        or not decoded
        or decoded[-1] != filename
        or any(
            segment in {".", ".."} or "/" in segment or chr(92) in segment for segment in decoded
        )
    ):
        return False
    if served_by_nvidia_index(name):
        return (
            parsed.hostname == NVIDIA_PACKAGE_INDEX_HOST
            and parsed.path.count("/") == 2
            and decoded == [name, filename]
        )
    return parsed.hostname == PYPI_FILES_HOST and parsed.path.startswith("/packages/")


def simple_index_files(page_url: str, body: bytes) -> tuple[SimpleIndexFile, ...]:
    """Read the file links of one project page in the simple HTML index format."""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ComfyRegistryPackageIndexError(
            "invalid_project_html", "Package index page is not UTF-8"
        ) from exc
    files: list[SimpleIndexFile] = []
    for attributes, label in _anchors(text):
        href = attributes.get("href")
        if not isinstance(href, str) or not href:
            raise ComfyRegistryPackageIndexError(
                "invalid_project_html", "Package index link has no address"
            )
        parsed = urlparse(urljoin(page_url, href))
        match = _SHA256_FRAGMENT.fullmatch(parsed.fragment)
        files.append(
            SimpleIndexFile(
                label.strip(),
                parsed._replace(fragment="").geturl(),
                match.group(1).lower() if match else None,
                attributes.get("data-requires-python"),
                _yank_state(attributes),
            )
        )
    return tuple(files)


async def read_remote_wheel_metadata(
    client: httpx.AsyncClient,
    url: str,
    filename: str,
    budget: RemoteReadBudget,
) -> RemoteWheelMetadata:
    """Read a remote wheel's size and its one METADATA member without downloading the wheel."""
    tail_start, total, tail = await _read_range(
        client,
        url,
        filename,
        f"bytes=-{REMOTE_WHEEL_TAIL_BYTES}",
        REMOTE_WHEEL_TAIL_BYTES,
        budget,
    )
    if tail_start != max(0, total - REMOTE_WHEEL_TAIL_BYTES) or total > MAX_REMOTE_WHEEL_BYTES:
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_range", f"Package server returned the wrong part of {filename}"
        )
    downloaded = len(tail)
    directory_offset, directory_size, entries = _end_of_directory(tail, tail_start, filename)
    if directory_offset >= tail_start:
        start = directory_offset - tail_start
        directory = tail[start : start + directory_size]
    else:
        directory = await _read_known_range(
            client, url, filename, directory_offset, directory_size, total, budget
        )
        downloaded += len(directory)
    entry = _metadata_entry(directory, directory_offset, entries, filename)
    window_end = min(
        directory_offset, entry.local_offset + _LOCAL_ENTRY.size + 2 * 0xFFFF + entry.compressed
    )
    if entry.local_offset >= tail_start:
        start = entry.local_offset - tail_start
        window = tail[start : window_end - tail_start]
    else:
        window = await _read_known_range(
            client,
            url,
            filename,
            entry.local_offset,
            window_end - entry.local_offset,
            total,
            budget,
        )
        downloaded += len(window)
    return RemoteWheelMetadata(total, _member_bytes(window, entry, filename), downloaded)


def _yank_state(attributes: dict[str, str | None]) -> bool | str:
    # A yank marker without a reason still yanks the file.
    if "data-yanked" not in attributes:
        return False
    return attributes["data-yanked"] or True


def _anchors(text: str) -> list[tuple[dict[str, str | None], str]]:
    # A simple index page is a list of links, so this reads only tags, links and their
    # text, and refuses a page whose markup it cannot account for byte by byte.
    anchors: list[tuple[dict[str, str | None], str]] = []
    open_anchor: dict[str, str | None] | None = None
    label: list[str] = []
    position = 0
    for match in _HTML_TOKEN.finditer(text):
        if match.start() != position:
            raise _malformed_page()
        position = match.end()
        token = match.group(0)
        if not token.startswith("<"):
            if open_anchor is not None:
                label.append(html.unescape(token))
            continue
        tag = _tag(token)
        if tag is None or tag[1] != "a":
            continue
        closing, _, rest = tag
        if closing:
            if open_anchor is not None:
                anchors.append((open_anchor, "".join(label)))
                open_anchor = None
            continue
        if open_anchor is not None:
            raise _malformed_page()
        if len(anchors) >= MAX_SIMPLE_HTML_FILES:
            raise ComfyRegistryPackageIndexError(
                "too_many_project_files", "Package index page lists too many files"
            )
        open_anchor = _attributes(rest)
        label = []
    if position != len(text) or open_anchor is not None:
        raise _malformed_page()
    return anchors


def _tag(token: str) -> tuple[bool, str, str] | None:
    inner = token[1:-1]
    if inner.startswith(("!", "?")):
        return None
    closing = inner.startswith("/")
    if closing:
        inner = inner[1:]
    match = _TAG_NAME.match(inner)
    if match is None:
        raise _malformed_page()
    return closing, match.group(0).lower(), inner[match.end() :]


def _attributes(rest: str) -> dict[str, str | None]:
    attributes: dict[str, str | None] = {}
    position = 0
    while (match := _ATTRIBUTE.match(rest, position)) is not None:
        name = match.group(1).lower()
        if name in attributes:
            raise _malformed_page()
        value = next((group for group in match.group(2, 3, 4) if group is not None), None)
        attributes[name] = None if value is None else html.unescape(value)
        position = match.end()
    if rest[position:].strip():
        raise _malformed_page()
    return attributes


def _malformed_page() -> ComfyRegistryPackageIndexError:
    return ComfyRegistryPackageIndexError(
        "invalid_project_html", "Package index page has a malformed link"
    )


@dataclass(frozen=True)
class _MetadataEntry:
    name: bytes
    method: int
    crc32: int
    compressed: int
    size: int
    local_offset: int


async def _read_known_range(
    client: httpx.AsyncClient,
    url: str,
    filename: str,
    first: int,
    length: int,
    total: int,
    budget: RemoteReadBudget,
) -> bytes:
    last = first + length - 1
    start, size, content = await _read_range(
        client, url, filename, f"bytes={first}-{last}", length, budget
    )
    if start != first or size != total or len(content) != length:
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_range", f"Package server returned the wrong part of {filename}"
        )
    return content


async def _read_range(
    client: httpx.AsyncClient,
    url: str,
    filename: str,
    requested: str,
    maximum: int,
    budget: RemoteReadBudget,
) -> tuple[int, int, bytes]:
    budget.charge(maximum)
    try:
        request = client.build_request(
            "GET", url, headers={"range": requested, "accept": "application/octet-stream"}
        )
    except httpx.InvalidURL as exc:
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_url", f"Package index address for {filename} is invalid"
        ) from exc
    response = await client.send(request, stream=True)
    try:
        if response.status_code != 206:
            raise ComfyRegistryPackageIndexError(
                "wheel_range_unavailable",
                f"Package server did not return part of {filename} (HTTP {response.status_code})",
            )
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in {"", "identity"}:
            raise ComfyRegistryPackageIndexError(
                "encoded_wheel_range", "Package server used an unsupported content encoding"
            )
        match = _CONTENT_RANGE.fullmatch(response.headers.get("content-range", ""))
        if match is None:
            raise ComfyRegistryPackageIndexError(
                "invalid_wheel_range", f"Package server returned the wrong part of {filename}"
            )
        start, end, total = (int(value) for value in match.groups())
        if start > end or end != min(total - 1, start + maximum - 1) or total < 1:
            raise ComfyRegistryPackageIndexError(
                "invalid_wheel_range", f"Package server returned the wrong part of {filename}"
            )
        content = bytearray()
        async for chunk in response.aiter_bytes(REMOTE_WHEEL_TAIL_BYTES):
            content.extend(chunk)
            if len(content) > end - start + 1:
                break
    finally:
        await response.aclose()
    if len(content) != end - start + 1:
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_range", f"Package server returned the wrong part of {filename}"
        )
    return start, total, bytes(content)


def _end_of_directory(tail: bytes, tail_start: int, filename: str) -> tuple[int, int, int]:
    position = tail.rfind(
        _END_SIGNATURE, 0, max(0, len(tail) - _END_OF_DIRECTORY.size + len(_END_SIGNATURE))
    )
    if position < 0:
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_archive", f"Wheel {filename} is not a readable archive"
        )
    (
        _signature,
        disk,
        directory_disk,
        disk_entries,
        entries,
        directory_size,
        directory_offset,
        comment_size,
    ) = _END_OF_DIRECTORY.unpack_from(tail, position)
    if (
        position + _END_OF_DIRECTORY.size + comment_size != len(tail)
        or disk != 0
        or directory_disk != 0
        or disk_entries != entries
        or not 0 < entries <= MAX_REMOTE_WHEEL_ENTRIES
        or not entries * _CENTRAL_ENTRY.size <= directory_size <= MAX_REMOTE_WHEEL_DIRECTORY_BYTES
        or directory_offset + directory_size != tail_start + position
    ):
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_archive", f"Wheel {filename} is not a readable archive"
        )
    return directory_offset, directory_size, entries


def _metadata_entry(
    directory: bytes, directory_offset: int, entries: int, filename: str
) -> _MetadataEntry:
    found: list[_MetadataEntry] = []
    position = 0
    count = 0
    while position < len(directory):
        if position + _CENTRAL_ENTRY.size > len(directory):
            break
        (
            signature,
            _made_by,
            _needed,
            flags,
            method,
            _time,
            _date,
            crc32,
            compressed,
            size,
            name_size,
            extra_size,
            comment_size,
            _disk,
            _internal,
            _external,
            local_offset,
        ) = _CENTRAL_ENTRY.unpack_from(directory, position)
        if signature != _CENTRAL_SIGNATURE:
            break
        name_start = position + _CENTRAL_ENTRY.size
        name = directory[name_start : name_start + name_size]
        extra = directory[name_start + name_size : name_start + name_size + extra_size]
        position += _CENTRAL_ENTRY.size + name_size + extra_size + comment_size
        count += 1
        # zipfile could not open the downloaded wheel with an undecodable name, and it
        # renames a member from a Unicode Path field, so its METADATA would not be the
        # one read here: refuse both before the download.
        if (flags & 0x800 and not _valid_utf8(name)) or not _plain_extra(extra):
            raise ComfyRegistryPackageIndexError(
                "invalid_wheel_archive", f"Wheel {filename} is not a readable archive"
            )
        parts = name.split(b"/")
        if len(parts) == 2 and parts[0].endswith(b".dist-info") and parts[1] == b"METADATA":
            if (
                flags & 1
                or method not in {0, 8}
                or size > MAX_REMOTE_WHEEL_METADATA_BYTES
                or compressed > MAX_REMOTE_WHEEL_METADATA_BYTES + 0xFFFF
                or local_offset + _LOCAL_ENTRY.size + name_size + compressed > directory_offset
            ):
                raise ComfyRegistryPackageIndexError(
                    "invalid_wheel_metadata", f"Wheel {filename} has unreadable core metadata"
                )
            found.append(_MetadataEntry(name, method, crc32, compressed, size, local_offset))
    if position != len(directory) or count != entries:
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_archive", f"Wheel {filename} is not a readable archive"
        )
    if len(found) != 1:
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_metadata", f"Wheel {filename} must hold exactly one METADATA file"
        )
    return found[0]


def _plain_extra(extra: bytes) -> bool:
    position = 0
    while position + 4 <= len(extra):
        kind, size = struct.unpack_from("<HH", extra, position)
        position += 4 + size
        if kind == _UNICODE_PATH_EXTRA or position > len(extra):
            return False
    return True


def _valid_utf8(value: bytes) -> bool:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _member_bytes(window: bytes, entry: _MetadataEntry, filename: str) -> bytes:
    if len(window) < _LOCAL_ENTRY.size:
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_metadata", f"Wheel {filename} has unreadable core metadata"
        )
    (
        signature,
        _needed,
        flags,
        method,
        _time,
        _date,
        _crc32,
        _compressed,
        _size,
        name_size,
        extra_size,
    ) = _LOCAL_ENTRY.unpack_from(window, 0)
    start = _LOCAL_ENTRY.size + name_size + extra_size
    data = window[start : start + entry.compressed]
    if (
        signature != _LOCAL_SIGNATURE
        or flags & 1
        or method != entry.method
        or window[_LOCAL_ENTRY.size : _LOCAL_ENTRY.size + name_size] != entry.name
        or len(data) != entry.compressed
    ):
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_metadata", f"Wheel {filename} has unreadable core metadata"
        )
    if entry.method == 0:
        content = data
    else:
        inflater = zlib.decompressobj(-15)
        try:
            content = inflater.decompress(data, entry.size + 1)
        except zlib.error as exc:
            raise ComfyRegistryPackageIndexError(
                "invalid_wheel_metadata", f"Wheel {filename} has unreadable core metadata"
            ) from exc
        if not inflater.eof or inflater.unconsumed_tail or inflater.unused_data:
            raise ComfyRegistryPackageIndexError(
                "invalid_wheel_metadata", f"Wheel {filename} has unreadable core metadata"
            )
    if len(content) != entry.size or zlib.crc32(content) != entry.crc32:
        raise ComfyRegistryPackageIndexError(
            "invalid_wheel_metadata", f"Wheel {filename} has unreadable core metadata"
        )
    return content
