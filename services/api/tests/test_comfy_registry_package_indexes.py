from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
import zlib
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest
from packaging.markers import default_environment

import local_lm.comfy_registry_wheel_projects as project_module
from local_lm.comfy_registry_closure_driver import (
    ComfyRegistryWheelClosureDriverError,
    ComfyRegistryWheelMetadataClient,
)
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_package_indexes import (
    ComfyRegistryPackageIndexError,
    RemoteReadBudget,
    project_index_url,
    read_remote_wheel_metadata,
    served_by_nvidia_index,
    simple_index_files,
    wheel_url_allowed,
)
from local_lm.comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactError,
    ComfyRegistryWheelArtifactManifest,
    resolve_comfy_registry_wheel_artifacts,
)
from local_lm.comfy_registry_wheel_downloads import (
    ComfyRegistryWheelDownloader,
    ComfyRegistryWheelDownloadError,
    ComfyRegistryWheelStageReport,
)
from local_lm.comfy_registry_wheel_projects import (
    ComfyRegistryWheelProjectClient,
    ComfyRegistryWheelProjectError,
)

_NAME = "nvidia-vfx"
_FILENAME = "nvidia_vfx-0.2.0.0-py3-none-any.whl"
_PAGE = "https://pypi.nvidia.com/nvidia-vfx/"
_WHEEL_URL = f"{_PAGE}{_FILENAME}"
_METADATA = (
    b"Metadata-Version: 2.1\nName: nvidia-vfx\nVersion: 0.2.0.0\nRequires-Python: >=3.10\n\n"
)
_METADATA_NAME = b"nvidia_vfx-0.2.0.0.dist-info/METADATA"
_TAG = "py3-none-any"
_TAIL = 22 + 0xFFFF
_BUDGET = 64 * 1024 * 1024

Handler = Callable[[httpx.Request], httpx.Response]


def _wheel(
    metadata: bytes = _METADATA,
    *,
    padding: int = 0,
    second_metadata: bool = False,
    compression: int = zipfile.ZIP_DEFLATED,
    comment: bytes = b"",
    extra_members: dict[str, bytes] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        archive.writestr("nvidia_vfx-0.2.0.0.dist-info/METADATA", metadata)
        if padding:
            archive.writestr(
                "nvidia_vfx/model.bin",
                bytes(index % 251 for index in range(padding)),
                compress_type=zipfile.ZIP_STORED,
            )
        if second_metadata:
            archive.writestr("other-1.0.dist-info/METADATA", metadata)
        for name, content in (extra_members or {}).items():
            archive.writestr(name, content)
        archive.writestr("nvidia_vfx/__init__.py", b"")
        archive.writestr("nvidia_vfx-0.2.0.0.dist-info/WHEEL", b"Wheel-Version: 1.0\n")
        archive.writestr("nvidia_vfx-0.2.0.0.dist-info/RECORD", b"")
        archive.comment = comment
    return buffer.getvalue()


def _patched(wheel: bytes, offset: int, layout: str, value: int) -> bytes:
    data = bytearray(wheel)
    struct.pack_into(layout, data, offset, value)
    return bytes(data)


def _central(wheel: bytes, name: bytes = _METADATA_NAME) -> int:
    # A name's last occurrence is in the central directory, just after its entry header.
    return wheel.rfind(b"PK\x01\x02", 0, wheel.rfind(name))


def _local(wheel: bytes, name: bytes = _METADATA_NAME) -> int:
    return wheel.find(name) - 30


def _end(wheel: bytes) -> int:
    return wheel.rfind(b"PK\x05\x06")


def _undecodable_name_wheel() -> bytes:
    wheel = _wheel(extra_members={"nvidia_vfx/café.py": b""})
    assert wheel.count("café".encode()) == 2
    return wheel.replace("café".encode(), b"caf\xc3\x28")


def _wheel_with_member_extra(extra: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nvidia_vfx-0.2.0.0.dist-info/METADATA", _METADATA)
        member = zipfile.ZipInfo("nvidia_vfx/payload.bin")
        member.extra = extra
        archive.writestr(member, b"payload")
    return buffer.getvalue()


def _unicode_path_extra() -> bytes:
    renamed = b"nvidia_vfx-0.2.0.0.dist-info/METADATA"
    payload = struct.pack("<BI", 1, zlib.crc32(b"nvidia_vfx/payload.bin")) + renamed
    return struct.pack("<HH", 0x7075, len(payload)) + payload


def _shifted_stored_metadata() -> bytes:
    # A longer local extra field shifts the data so a stored METADATA, last before the
    # directory, comes up short of its listed size; the central size and CRC are
    # rewritten to match what is left, so only the length check can tell.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("nvidia_vfx/__init__.py", b"")
        archive.writestr("nvidia_vfx-0.2.0.0.dist-info/METADATA", _METADATA)
    wheel = _patched(buffer.getvalue(), _local(buffer.getvalue()) + 28, "<H", 8)
    wheel = _patched(wheel, _central(wheel) + 24, "<I", len(_METADATA) - 8)
    return _patched(wheel, _central(wheel) + 16, "<I", zlib.crc32(_METADATA[8:]))


def _junk_after_directory(wheel: bytes) -> bytes:
    end = _end(wheel)
    return _end_field(12, lambda value: value + 8)(wheel[:end] + b"JUNKJUNK" + wheel[end:])


def _ranged(payload: bytes, request: httpx.Request) -> httpx.Response:
    unit, _, span = request.headers["range"].partition("=")
    assert unit == "bytes"
    first, _, last = span.partition("-")
    if first:
        start, end = int(first), min(int(last), len(payload) - 1)
    else:
        start, end = max(0, len(payload) - int(last)), len(payload) - 1
    return httpx.Response(
        206,
        content=payload[start : end + 1],
        headers={"content-range": f"bytes {start}-{end}/{len(payload)}"},
    )


def _link(filename: str, digest: str, *, href: str | None = None, extra: str = "") -> str:
    return f'<a href="{href or filename}#sha256={digest}"{extra}>{filename}</a><br>\n'


def _page(*links: str) -> bytes:
    return ("<!DOCTYPE html>\n<html><body>\n" + "".join(links) + "</body></html>\n").encode()


def _html(page: bytes) -> httpx.Response:
    return httpx.Response(200, stream=httpx.ByteStream(page), headers={"content-type": "text/html"})


async def _read(
    handler: Handler,
    url: str = _WHEEL_URL,
    budget: RemoteReadBudget | None = None,
) -> tuple[int, bytes, int]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        remote = await read_remote_wheel_metadata(
            client, url, _FILENAME, budget or RemoteReadBudget(_BUDGET)
        )
    return remote.size_bytes, remote.metadata, remote.downloaded_bytes


async def _read_error(wheel: bytes) -> tuple[str, int]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ranged(wheel, request)

    with pytest.raises(ComfyRegistryPackageIndexError) as raised:
        await _read(handler)
    return raised.value.code, len(requests)


async def _fetch(handler: Handler, names: list[str] | None = None) -> dict[str, object]:
    client = ComfyRegistryWheelProjectClient(transport=httpx.MockTransport(handler))
    try:
        return await client.fetch(names or [_NAME])
    finally:
        await client.close()


def _environment() -> dict[str, str]:
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    return environment


def _resolve(
    requirement: str, document: object, tags: tuple[str, ...] = (_TAG,)
) -> ComfyRegistryWheelArtifactManifest:
    return resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies([requirement]),
        {_NAME: document},
        marker_environment=_environment(),
        supported_tags=tags,
    )


def _manifest(wheel: bytes, metadata: bytes = _METADATA) -> ComfyRegistryWheelArtifactManifest:
    document = {
        "meta": {"api-version": "1.0"},
        "name": _NAME,
        "files": [
            {
                "filename": _FILENAME,
                "url": _WHEEL_URL,
                "hashes": {"sha256": hashlib.sha256(wheel).hexdigest()},
                "requires-python": ">=3.10",
                "yanked": False,
                "size": len(wheel),
                "core-metadata": {"sha256": hashlib.sha256(metadata).hexdigest()},
            }
        ],
    }
    return _resolve(f"{_NAME}==0.2.0.0", document)


async def _stage(
    wheel: bytes, destination: Path, metadata: bytes = _METADATA
) -> ComfyRegistryWheelStageReport:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=wheel)

    downloader = ComfyRegistryWheelDownloader(transport=httpx.MockTransport(handler))
    try:
        report = await downloader.download_and_stage(_manifest(wheel, metadata), destination)
    finally:
        await downloader.close()
    assert [str(request.url) for request in requests] == [_WHEEL_URL]
    return report


class _EndlessStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.sent = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(160):
            self.sent += 65536
            yield b"\0" * 65536


def test_nvidia_projects_come_only_from_nvidias_index() -> None:
    assert served_by_nvidia_index("nvidia-vfx")
    assert not served_by_nvidia_index("nvidia-cudnn-cu12")
    assert project_index_url("nvidia-vfx") == "https://pypi.nvidia.com/nvidia-vfx/"
    assert project_index_url("example-package") == "https://pypi.org/simple/example-package/"


_PYPI_FILE = "example_package-1.0-py3-none-any.whl"
_PYPI_URL = f"https://files.pythonhosted.org/packages/aa/bb/{_PYPI_FILE}"
_ENCODED_FILE = "nvidia_vfx-0.2.0.0+cu12-py3-none-any.whl"


@pytest.mark.parametrize(
    ("name", "url", "filename", "allowed"),
    [
        (_NAME, _WHEEL_URL, _FILENAME, True),
        (_NAME, f"{_PAGE}nvidia_vfx-0.2.0.0%2Bcu12-py3-none-any.whl", _ENCODED_FILE, True),
        ("example-package", _PYPI_URL, _PYPI_FILE, True),
        (
            "example-package",
            f"https://pypi.nvidia.com/example-package/{_PYPI_FILE}",
            _PYPI_FILE,
            False,
        ),
        (_NAME, f"https://files.pythonhosted.org/packages/aa/bb/{_FILENAME}", _FILENAME, False),
        (_NAME, f"{_PAGE}extra/{_FILENAME}", _FILENAME, False),
        (_NAME, f"https://pypi.nvidia.com/other/{_FILENAME}", _FILENAME, False),
        (_NAME, f"{_PAGE}/{_FILENAME}", _FILENAME, False),
        (_NAME, f"{_PAGE}nvidia_vfx-0.1.0-py3-none-any.whl", _FILENAME, False),
        (
            _NAME,
            f"{_PAGE}nvidia%2Fvfx-0.2.0.0-py3-none-any.whl",
            "nvidia/vfx-0.2.0.0-py3-none-any.whl",
            False,
        ),
        (_NAME, f"http://pypi.nvidia.com/nvidia-vfx/{_FILENAME}", _FILENAME, False),
        (_NAME, f"https://pypi.nvidia.com:8443/nvidia-vfx/{_FILENAME}", _FILENAME, False),
        (_NAME, f"https://pypi.nvidia.com:port/nvidia-vfx/{_FILENAME}", _FILENAME, False),
        (_NAME, f"{_WHEEL_URL}?download=1", _FILENAME, False),
        (_NAME, f"{_WHEEL_URL}#sha256={'a' * 64}", _FILENAME, False),
        (_NAME, f"https://user@pypi.nvidia.com/nvidia-vfx/{_FILENAME}", _FILENAME, False),
        (_NAME, f"https://:secret@pypi.nvidia.com/nvidia-vfx/{_FILENAME}", _FILENAME, False),
        (
            _NAME,
            f"{_PAGE}nvidia_vfx-0.2.0.0\x7f-py3-none-any.whl",
            "nvidia_vfx-0.2.0.0\x7f-py3-none-any.whl",
            False,
        ),
        (
            _NAME,
            f"{_PAGE}nvidia_vfx 0.2.0.0-py3-none-any.whl",
            "nvidia_vfx 0.2.0.0-py3-none-any.whl",
            False,
        ),
        (
            "example-package",
            f"https://files.pythonhosted.org/packages/{'a' * 2_000}/{_PYPI_FILE}",
            _PYPI_FILE,
            False,
        ),
        (
            "example-package",
            f"https://files.pythonhosted.org/packages/aa/bb/{_FILENAME}",
            _PYPI_FILE,
            False,
        ),
        (
            "example-package",
            f"https://files.pythonhosted.org/packages/../x/{_PYPI_FILE}",
            _PYPI_FILE,
            False,
        ),
        (
            "example-package",
            f"https://files.pythonhosted.org/other/{_PYPI_FILE}",
            _PYPI_FILE,
            False,
        ),
        ("example-package", f"https://example.com/packages/aa/bb/{_PYPI_FILE}", _PYPI_FILE, False),
    ],
)
def test_wheel_urls_must_point_into_the_index_that_serves_the_project(
    name: str, url: str, filename: str, allowed: bool
) -> None:
    assert wheel_url_allowed(name, url, filename) is allowed


def test_resolution_takes_nvidia_wheel_urls_for_nvidia_projects_only() -> None:
    manifest = _manifest(b"wheel")
    assert [artifact.url for artifact in manifest.artifacts] == [_WHEEL_URL]

    document = {
        "meta": {"api-version": "1.0"},
        "name": "example-package",
        "files": [
            {
                "filename": _PYPI_FILE,
                "url": f"https://pypi.nvidia.com/example-package/{_PYPI_FILE}",
                "hashes": {"sha256": "a" * 64},
                "size": 5,
                "core-metadata": {"sha256": "b" * 64},
            }
        ],
    }
    with pytest.raises(ComfyRegistryWheelArtifactError) as raised:
        resolve_comfy_registry_wheel_artifacts(
            plan_comfy_registry_dependencies(["example-package==1.0"]),
            {"example-package": document},
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )
    assert raised.value.code == "invalid_wheel_url"


def test_simple_index_page_links_keep_hashes_python_bounds_and_yanks() -> None:
    files = simple_index_files(
        _PAGE,
        _page(
            _link(_FILENAME, "C" * 64, extra=' data-requires-python="&gt;=3.10"'),
            _link("nvidia_vfx-0.1.0-py3-none-any.whl", "d" * 64, extra=" data-yanked"),
            _link("nvidia_vfx-0.1.1-py3-none-any.whl", "e" * 64, extra=' data-yanked="broken"'),
            '<a href="nvidia_vfx-0.1.2-py3-none-any.whl#md5=' + "f" * 32 + '">'
            "nvidia_vfx-0.1.2-py3-none-any.whl</a>",
            '<a href="nvidia_vfx-0.0.1.tar.gz">nvidia_vfx-0.0.1.tar.gz</a>',
        ),
    )

    assert [(item.filename, item.url, item.sha256) for item in files] == [
        (_FILENAME, _WHEEL_URL, "c" * 64),
        (
            "nvidia_vfx-0.1.0-py3-none-any.whl",
            f"{_PAGE}nvidia_vfx-0.1.0-py3-none-any.whl",
            "d" * 64,
        ),
        (
            "nvidia_vfx-0.1.1-py3-none-any.whl",
            f"{_PAGE}nvidia_vfx-0.1.1-py3-none-any.whl",
            "e" * 64,
        ),
        ("nvidia_vfx-0.1.2-py3-none-any.whl", f"{_PAGE}nvidia_vfx-0.1.2-py3-none-any.whl", None),
        ("nvidia_vfx-0.0.1.tar.gz", f"{_PAGE}nvidia_vfx-0.0.1.tar.gz", None),
    ]
    assert [item.requires_python for item in files] == [">=3.10", None, None, None, None]
    assert [item.yanked for item in files] == [False, True, "broken", False, False]


def test_a_page_may_list_exactly_the_most_files_allowed() -> None:
    links = "".join(
        f"<a href='nvidia_vfx-0.0.{index}-py3-none-any.whl'>"
        f"nvidia_vfx-0.0.{index}-py3-none-any.whl</a>"
        for index in range(256)
    )
    assert len(simple_index_files(_PAGE, links.encode())) == 256


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"<a href='x.whl'>x.whl", "invalid_project_html"),
        (b"<a href='x.whl'><a href='y.whl'>y.whl</a></a>", "invalid_project_html"),
        (b"<a href='x.whl' href='y.whl'>x.whl</a>", "invalid_project_html"),
        (b"<a>x.whl</a>", "invalid_project_html"),
        (b"<a href=''>x.whl</a>", "invalid_project_html"),
        (b"\xff<a href='x.whl'>x.whl</a>", "invalid_project_html"),
        (b"<a href='x.whl'>x.whl</a>" * 257, "too_many_project_files"),
    ],
    ids=["unclosed", "nested", "duplicate-href", "no-href", "empty-href", "not-utf8", "too-many"],
)
def test_malformed_simple_index_pages_fail_closed(body: bytes, code: str) -> None:
    with pytest.raises(ComfyRegistryPackageIndexError) as raised:
        simple_index_files(_PAGE, body)
    assert raised.value.code == code


def test_ordinary_html_around_the_links_is_read() -> None:
    body = (
        b"<!DOCTYPE html><HTML><head><!-- generated --><title>Links &amp; files</title>"
        b"</head><body>\n<h1>Links for nvidia-vfx</h1>\n"
        b"<A HREF='x-1.0-py3-none-any.whl' Data-Requires-Python=\"&gt;=3.10\">"
        b"<b>&#120;-1.0-py3-none-any.whl</b></A><br/>\n"
        b"<a href=y-1.0-py3-none-any.whl data-yanked>y-1.0-py3-none-any.whl</a> > end\n"
        b"</body></HTML>"
    )

    files = simple_index_files(_PAGE, body)

    assert [(item.filename, item.url, item.requires_python, item.yanked) for item in files] == [
        ("x-1.0-py3-none-any.whl", f"{_PAGE}x-1.0-py3-none-any.whl", ">=3.10", False),
        ("y-1.0-py3-none-any.whl", f"{_PAGE}y-1.0-py3-none-any.whl", None, True),
    ]


@pytest.mark.parametrize(
    "body",
    [
        b"<a href='x.whl'>x<y.whl</a>",
        b'<a href="x.whl>x.whl</a>',
        b"<a href='x.whl'/>",
        b"< a href='x.whl'>x.whl</a>",
        b"<a href='x.whl' =>x.whl</a>",
        b"<a x" * 32_768,
        b"<a href='x.whl'>x.whl</a><",
    ],
    ids=[
        "stray-bracket",
        "open-quote",
        "self-closing",
        "spaced-tag",
        "bare-equals",
        "unclosed-tags",
        "trailing-bracket",
    ],
)
def test_markup_the_reader_cannot_account_for_fails_closed(body: bytes) -> None:
    with pytest.raises(ComfyRegistryPackageIndexError) as raised:
        simple_index_files(_PAGE, body)
    assert raised.value.code == "invalid_project_html"


def test_unterminated_comments_do_not_hide_links() -> None:
    body = b"<!--x>" * 20_000 + b"<a href='x-1.0-py3-none-any.whl'>x-1.0-py3-none-any.whl</a>"

    assert [item.filename for item in simple_index_files(_PAGE, body)] == ["x-1.0-py3-none-any.whl"]


@pytest.mark.parametrize(
    ("wheel", "requests_made"),
    [
        (_wheel(), 1),
        (_wheel(compression=zipfile.ZIP_STORED), 1),
        (_wheel(comment=b"built for a test"), 1),
        (_wheel(comment=b"c" * 0xFFFF), 3),
        (_wheel(extra_members={"nvidia_vfx/templates/METADATA": b"template"}), 1),
        (_wheel(padding=1_000_000), 2),
        (_wheel_with_member_extra(struct.pack("<HHB", 0x5455, 1, 0)), 1),
    ],
    ids=[
        "deflated",
        "stored",
        "commented",
        "longest-comment",
        "data-named-metadata",
        "far-metadata",
        "timestamp-extra",
    ],
)
async def test_remote_wheel_metadata_is_read_from_bounded_ranges(
    wheel: bytes, requests_made: int
) -> None:
    requests: list[httpx.Request] = []
    served: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        response = _ranged(wheel, request)
        served.append(len(response.content))
        return response

    budget = RemoteReadBudget(_BUDGET)
    size, metadata, downloaded = await _read(handler, budget=budget)

    assert (size, metadata) == (len(wheel), _METADATA)
    assert len(requests) == requests_made
    assert requests[0].headers["range"] == f"bytes=-{_TAIL}"
    assert downloaded == sum(served)
    assert downloaded <= _TAIL + 30 + 2 * 0xFFFF + len(_METADATA) + 100
    assert downloaded <= budget.spent


async def test_a_wheel_whose_directory_lies_outside_the_tail_is_read_in_parts() -> None:
    wheel = _far_directory_wheel()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ranged(wheel, request)

    size, metadata, _ = await _read(handler)

    assert (size, metadata) == (len(wheel), _METADATA)
    assert len(requests) == 3


async def test_reads_stop_before_a_request_the_budget_cannot_cover() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("an unaffordable read must not be sent")

    with pytest.raises(ComfyRegistryPackageIndexError) as raised:
        await _read(handler, budget=RemoteReadBudget(_TAIL - 1))
    assert raised.value.code == "project_metadata_total_too_large"


def _whole(wheel: bytes, _request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=wheel)


def _missing(_wheel: bytes, _request: httpx.Request) -> httpx.Response:
    return httpx.Response(404)


def _unknown_total(wheel: bytes, _request: httpx.Request) -> httpx.Response:
    return httpx.Response(206, content=wheel[-10:], headers={"content-range": "bytes 0-9/*"})


def _wrong_end(wheel: bytes, _request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        206, content=wheel[:10], headers={"content-range": f"bytes 0-9/{len(wheel)}"}
    )


def _head_not_tail(wheel: bytes, _request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        206,
        content=wheel[:_TAIL],
        headers={"content-range": f"bytes 0-{_TAIL - 1}/{len(wheel)}"},
    )


def _short_tail(wheel: bytes, _request: httpx.Request) -> httpx.Response:
    total = len(wheel)
    return httpx.Response(
        206,
        content=wheel[-(_TAIL - 1) :],
        headers={"content-range": f"bytes {total - _TAIL}-{total - 1}/{total}"},
    )


def _encoded(wheel: bytes, _request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        206,
        stream=httpx.ByteStream(wheel),
        headers={
            "content-range": f"bytes 0-{len(wheel) - 1}/{len(wheel)}",
            "content-encoding": "gzip",
        },
    )


@pytest.mark.parametrize(
    ("respond", "wheel", "code"),
    [
        (_whole, _wheel(), "wheel_range_unavailable"),
        (_missing, _wheel(), "wheel_range_unavailable"),
        (_unknown_total, _wheel(), "invalid_wheel_range"),
        (_wrong_end, _wheel(), "invalid_wheel_range"),
        (_head_not_tail, _wheel(padding=200_000), "invalid_wheel_range"),
        (_short_tail, _wheel(padding=200_000), "invalid_wheel_range"),
        (_encoded, _wheel(), "encoded_wheel_range"),
    ],
    ids=["whole-file", "missing", "unknown-total", "wrong-end", "head", "short-tail", "encoded"],
)
async def test_servers_that_do_not_honour_the_range_fail_closed(
    respond: Callable[[bytes, httpx.Request], httpx.Response], wheel: bytes, code: str
) -> None:
    with pytest.raises(ComfyRegistryPackageIndexError) as raised:
        await _read(lambda request: respond(wheel, request))
    assert raised.value.code == code


async def test_a_body_longer_than_its_range_is_not_read_to_the_end() -> None:
    wheel = _wheel()
    stream = _EndlessStream()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            206,
            stream=stream,
            headers={"content-range": f"bytes 0-{len(wheel) - 1}/{len(wheel)}"},
        )

    with pytest.raises(ComfyRegistryPackageIndexError) as raised:
        await _read(handler)
    assert raised.value.code == "invalid_wheel_range"
    assert stream.sent <= 2 * 65536


async def test_a_later_range_answered_at_another_offset_fails_closed() -> None:
    wheel = _wheel(padding=200_000)

    def handler(request: httpx.Request) -> httpx.Response:
        first, _, last = request.headers["range"].removeprefix("bytes=").partition("-")
        if not first:
            return _ranged(wheel, request)
        start, end = int(first) + 7, int(last) + 7
        return httpx.Response(
            206,
            content=wheel[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(wheel)}"},
        )

    with pytest.raises(ComfyRegistryPackageIndexError) as raised:
        await _read(handler)
    assert raised.value.code == "invalid_wheel_range"


async def test_a_wheel_that_changes_between_ranges_fails_closed() -> None:
    served = iter((_wheel(padding=200_000), _wheel(padding=200_001)))

    with pytest.raises(ComfyRegistryPackageIndexError) as raised:
        await _read(lambda request: _ranged(next(served), request))
    assert raised.value.code == "invalid_wheel_range"


def _bad_crc(wheel: bytes) -> bytes:
    return _patched(wheel, _central(wheel) + 16, "<I", 0)


def _encrypted(wheel: bytes) -> bytes:
    return _patched(wheel, _central(wheel) + 8, "<H", 1)


def _unsupported_method(wheel: bytes) -> bytes:
    wheel = _patched(wheel, _central(wheel) + 10, "<H", 12)
    return _patched(wheel, _local(wheel) + 8, "<H", 12)


def _other_local_method(wheel: bytes) -> bytes:
    return _patched(wheel, _local(wheel) + 8, "<H", 0)


def _other_local_name(wheel: bytes) -> bytes:
    data = bytearray(wheel)
    data[_local(wheel) + 30] ^= 0x20
    return bytes(data)


def _trailing_compressed_byte(wheel: bytes) -> bytes:
    central = _central(wheel)
    (compressed,) = struct.unpack_from("<I", wheel, central + 20)
    return _patched(wheel, central + 20, "<I", compressed + 1)


def _offset_at_another_member(wheel: bytes) -> bytes:
    return _patched(wheel, _central(wheel) + 42, "<I", wheel.find(b"PK\x03\x04", 1))


def _offset_past_directory(wheel: bytes) -> bytes:
    (directory_offset,) = struct.unpack_from("<I", wheel, _end(wheel) + 16)
    return _patched(wheel, _central(wheel) + 42, "<I", directory_offset)


def _member_size_off_by_one(wheel: bytes) -> bytes:
    central = _central(wheel)
    (size,) = struct.unpack_from("<I", wheel, central + 24)
    return _patched(wheel, central + 24, "<I", size + 1)


def _end_field(offset: int, change: Callable[[int], int]) -> Callable[[bytes], bytes]:
    layout = "<I" if offset in {12, 16} else "<H"

    def patch(wheel: bytes) -> bytes:
        (value,) = struct.unpack_from(layout, wheel, _end(wheel) + offset)
        return _patched(wheel, _end(wheel) + offset, layout, change(value))

    return patch


def _both_entry_counts(change: Callable[[int], int]) -> Callable[[bytes], bytes]:
    def patch(wheel: bytes) -> bytes:
        return _end_field(10, change)(_end_field(8, change)(wheel))

    return patch


def _directory_short_of_its_end(wheel: bytes) -> bytes:
    # Leave the last entry outside the declared directory, as if data were prepended.
    last = wheel.rfind(b"PK\x01\x02", 0, _end(wheel))
    length = _end(wheel) - last
    wheel = _both_entry_counts(lambda value: value - 1)(wheel)
    return _end_field(12, lambda value: value - length)(wheel)


def _far_directory_wheel() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nvidia_vfx-0.2.0.0.dist-info/METADATA", _METADATA)
        for index in range(1_600):
            archive.writestr(f"nvidia_vfx/data/{index:05d}-{'x' * 40}.py", b"")
    return buffer.getvalue()


def _local_entry_copy(change: Callable[[bytearray], None]) -> bytes:
    # Point the METADATA entry at a stored copy of its own local entry, changed by one field.
    plain = _wheel()
    (compressed,) = struct.unpack_from("<I", plain, 18)
    copy = bytearray(plain[: 30 + len(_METADATA_NAME) + compressed])
    change(copy)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nvidia_vfx-0.2.0.0.dist-info/METADATA", _METADATA)
        archive.writestr("nvidia_vfx/copy.bin", bytes(copy), compress_type=zipfile.ZIP_STORED)
        archive.writestr("nvidia_vfx-0.2.0.0.dist-info/RECORD", b"")
    wheel = buffer.getvalue()
    return _patched(wheel, _central(wheel) + 42, "<I", wheel.find(bytes(copy), 1))


def _bad_signature(entry: bytearray) -> None:
    entry[0:4] = b"XXXX"


def _encrypted_local(entry: bytearray) -> None:
    entry[6] |= 1


@pytest.mark.parametrize(
    ("wheel", "code"),
    [
        (_wheel(second_metadata=True), "invalid_wheel_metadata"),
        (_wheel(metadata=b"x" * (1024 * 1024 + 1)), "invalid_wheel_metadata"),
        (_bad_crc(_wheel()), "invalid_wheel_metadata"),
        (_encrypted(_wheel()), "invalid_wheel_metadata"),
        (_unsupported_method(_wheel()), "invalid_wheel_metadata"),
        (_other_local_method(_wheel()), "invalid_wheel_metadata"),
        (_other_local_name(_wheel()), "invalid_wheel_metadata"),
        (_trailing_compressed_byte(_wheel()), "invalid_wheel_metadata"),
        (_offset_at_another_member(_wheel()), "invalid_wheel_metadata"),
        (_offset_past_directory(_wheel()), "invalid_wheel_metadata"),
        (_undecodable_name_wheel(), "invalid_wheel_archive"),
        (b"not a wheel at all, only some bytes that are long enough", "invalid_wheel_archive"),
        (_wheel()[:-1], "invalid_wheel_archive"),
        (_end_field(4, lambda value: 1)(_wheel()), "invalid_wheel_archive"),
        (_end_field(8, lambda value: value + 1)(_wheel()), "invalid_wheel_archive"),
        (_end_field(20, lambda value: 5)(_wheel()), "invalid_wheel_archive"),
        (_end_field(16, lambda value: value - 1)(_wheel()), "invalid_wheel_archive"),
        (_both_entry_counts(lambda value: value + 1)(_wheel()), "invalid_wheel_archive"),
        (_both_entry_counts(lambda value: 0)(_wheel()), "invalid_wheel_archive"),
        (_end_field(6, lambda value: 1)(_wheel()), "invalid_wheel_archive"),
        (_directory_short_of_its_end(_wheel()), "invalid_wheel_archive"),
        (_member_size_off_by_one(_wheel()), "invalid_wheel_metadata"),
        (_local_entry_copy(_bad_signature), "invalid_wheel_metadata"),
        (_local_entry_copy(_encrypted_local), "invalid_wheel_metadata"),
        (_shifted_stored_metadata(), "invalid_wheel_metadata"),
        (_junk_after_directory(_wheel()), "invalid_wheel_archive"),
        (_wheel_with_member_extra(_unicode_path_extra()), "invalid_wheel_archive"),
        (
            _wheel_with_member_extra(struct.pack("<HH", 0x5455, 50) + b"abcd"),
            "invalid_wheel_archive",
        ),
    ],
    ids=[
        "two-metadata",
        "oversized-metadata",
        "bad-crc",
        "encrypted",
        "unsupported-method",
        "local-method",
        "local-name",
        "trailing-compressed-byte",
        "offset-at-other-member",
        "offset-past-directory",
        "undecodable-name",
        "not-an-archive",
        "truncated",
        "disk-number",
        "disk-entries",
        "comment-length",
        "directory-offset",
        "entry-count",
        "no-entries",
        "directory-disk",
        "directory-short-of-end",
        "member-size",
        "local-signature",
        "local-encrypted",
        "data-short-of-its-size",
        "junk-after-directory",
        "unicode-path-extra",
        "overlong-extra",
    ],
)
async def test_unreadable_or_ambiguous_wheel_metadata_fails_closed(wheel: bytes, code: str) -> None:
    assert (await _read_error(wheel))[0] == code


async def test_an_unchanged_local_entry_copy_is_read() -> None:
    wheel = _local_entry_copy(lambda entry: None)

    size, metadata, _ = await _read(lambda request: _ranged(wheel, request))
    assert (size, metadata) == (len(wheel), _METADATA)


async def test_an_impossible_entry_count_is_refused_before_reading_the_directory() -> None:
    wheel = _both_entry_counts(lambda value: 0)(_far_directory_wheel())

    assert await _read_error(wheel) == ("invalid_wheel_archive", 1)


async def test_a_metadata_entry_reaching_into_the_directory_is_refused_before_reading_it() -> None:
    wheel = _wheel(padding=1_000_000)
    (directory_offset,) = struct.unpack_from("<I", wheel, _end(wheel) + 16)
    wheel = _patched(wheel, _central(wheel) + 20, "<I", directory_offset)

    assert await _read_error(wheel) == ("invalid_wheel_metadata", 1)


async def test_an_address_the_client_cannot_send_fails_closed() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("an invalid address must not be sent")

    with pytest.raises(ComfyRegistryPackageIndexError) as raised:
        await _read(handler, url=f"{_PAGE}nvidia_vfx\x00.whl")
    assert raised.value.code == "invalid_wheel_url"


async def test_a_wheel_without_metadata_fails_closed() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nvidia_vfx/__init__.py", b"")

    assert (await _read_error(buffer.getvalue()))[0] == "invalid_wheel_metadata"


async def test_an_oversized_compressed_metadata_entry_is_refused_before_reading_it() -> None:
    wheel = _wheel(padding=3_000_000)
    wheel = _patched(wheel, _central(wheel) + 20, "<I", 1024 * 1024 + 0xFFFF + 1)

    assert await _read_error(wheel) == ("invalid_wheel_metadata", 1)


async def test_an_oversized_central_directory_is_refused_before_reading_it() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("nvidia_vfx-0.2.0.0.dist-info/METADATA", _METADATA)
        for index in range(25_500):
            archive.writestr(f"nvidia_vfx/data/{index:05d}-{'x' * 100}.py", b"")
    wheel = buffer.getvalue()

    assert await _read_error(wheel) == ("invalid_wheel_archive", 1)


async def test_nvidia_project_page_becomes_a_hash_bound_project_document() -> None:
    current = _wheel()
    yanked = _wheel(padding=10)
    unreadable = "nvidia_vfx-0.1.0-cp312-cp312-win_amd64.whl"
    page = _page(
        _link(
            _FILENAME,
            hashlib.sha256(current).hexdigest(),
            extra=' data-requires-python="&gt;=3.10"',
        ),
        _link(
            "nvidia_vfx-0.1.0-py3-none-any.whl",
            hashlib.sha256(yanked).hexdigest(),
            extra=' data-yanked="superseded"',
        ),
        _link(unreadable, "b" * 64),
        _link(
            "nvidia_vfx-0.0.9-py3-none-any.whl",
            "f" * 64,
            href="https://example.com/nvidia-vfx/nvidia_vfx-0.0.9-py3-none-any.whl",
        ),
        _link(
            "nvidia_vfx-0.0.8-py3-none-any.whl", "e" * 64, href="nvidia_vfx-0.0.7-py3-none-any.whl"
        ),
        '<a href="nvidia_vfx-0.0.1.tar.gz">nvidia_vfx-0.0.1.tar.gz</a>',
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "pypi.nvidia.com"
        if request.url.path == "/nvidia-vfx/":
            return _html(page)
        if request.url.path.endswith(unreadable):
            return httpx.Response(404)
        assert request.url.path.endswith(_FILENAME)
        return _ranged(current, request)

    documents = await _fetch(handler)

    assert documents == {
        _NAME: {
            "meta": {"api-version": "1.0"},
            "name": _NAME,
            "files": [
                {
                    "filename": _FILENAME,
                    "url": _WHEEL_URL,
                    "hashes": {"sha256": hashlib.sha256(current).hexdigest()},
                    "yanked": False,
                    "requires-python": ">=3.10",
                    "size": len(current),
                    "core-metadata": {"sha256": hashlib.sha256(_METADATA).hexdigest()},
                },
                {
                    "filename": "nvidia_vfx-0.1.0-py3-none-any.whl",
                    "url": f"{_PAGE}nvidia_vfx-0.1.0-py3-none-any.whl",
                    "hashes": {"sha256": hashlib.sha256(yanked).hexdigest()},
                    "yanked": "superseded",
                },
                {
                    "filename": unreadable,
                    "url": f"{_PAGE}{unreadable}",
                    "hashes": {"sha256": "b" * 64},
                    "yanked": False,
                    "_unread": {
                        "code": "wheel_range_unavailable",
                        "message": f"Package server did not return part of {unreadable} (HTTP 404)",
                    },
                },
                {
                    "filename": "nvidia_vfx-0.0.9-py3-none-any.whl",
                    "url": "https://example.com/nvidia-vfx/nvidia_vfx-0.0.9-py3-none-any.whl",
                    "hashes": {"sha256": "f" * 64},
                    "yanked": False,
                    "_unread": {
                        "code": "invalid_wheel_url",
                        "message": "Package index link for "
                        "nvidia_vfx-0.0.9-py3-none-any.whl is not allowlisted",
                    },
                },
                {
                    "filename": "nvidia_vfx-0.0.8-py3-none-any.whl",
                    "url": f"{_PAGE}nvidia_vfx-0.0.7-py3-none-any.whl",
                    "hashes": {"sha256": "e" * 64},
                    "yanked": False,
                    "_unread": {
                        "code": "invalid_wheel_url",
                        "message": "Package index link for "
                        "nvidia_vfx-0.0.8-py3-none-any.whl is not allowlisted",
                    },
                },
                {
                    "filename": "nvidia_vfx-0.0.1.tar.gz",
                    "url": f"{_PAGE}nvidia_vfx-0.0.1.tar.gz",
                    "hashes": {},
                    "yanked": False,
                },
            ],
        }
    }
    assert [str(request.url) for request in requests] == [
        _PAGE,
        _WHEEL_URL,
        f"{_PAGE}{unreadable}",
    ]
    assert requests[0].headers["accept"] == "text/html"
    assert "range" not in requests[0].headers
    assert requests[1].headers["range"] == f"bytes=-{_TAIL}"

    both_tags = ("cp312-cp312-win_amd64", _TAG)
    for requirement in (f"{_NAME}==0.2.0.0", _NAME, f"{_NAME}>=0.1"):
        selected = _resolve(requirement, documents[_NAME], both_tags)
        assert [artifact.filename for artifact in selected.artifacts] == [_FILENAME]
    with pytest.raises(ComfyRegistryWheelArtifactError) as raised:
        _resolve(f"{_NAME}==0.1.0", documents[_NAME], both_tags)
    assert raised.value.code == "wheel_range_unavailable"
    assert str(raised.value) == f"Package server did not return part of {unreadable} (HTTP 404)"
    with pytest.raises(ComfyRegistryWheelArtifactError) as raised:
        _resolve(f"{_NAME}==0.0.9", documents[_NAME], both_tags)
    assert raised.value.code == "invalid_wheel_url"


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        (
            {"code": "wheel_range_unavailable", "message": "Server said no"},
            "wheel_range_unavailable",
        ),
        ({"code": "Not A Code", "message": "Server said no"}, "unreadable_wheel"),
        ({"code": "wheel_range_unavailable", "message": "line\nbreak"}, "unreadable_wheel"),
        ({"code": "wheel_range_unavailable", "message": "x" * 501}, "unreadable_wheel"),
        ("not an object", "unreadable_wheel"),
    ],
    ids=["kept", "bad-code", "control-character", "too-long", "not-an-object"],
)
def test_an_unread_wheel_refuses_with_its_reason_only_when_well_formed(
    reason: object, code: str
) -> None:
    document = {
        "meta": {"api-version": "1.0"},
        "name": _NAME,
        "files": [
            {"filename": _FILENAME, "url": _WHEEL_URL, "hashes": {}, "_unread": reason},
        ],
    }
    with pytest.raises(ComfyRegistryWheelArtifactError) as raised:
        _resolve(_NAME, document)
    assert raised.value.code == code


_BUILT_FILE = "example_package-1.0-1-py3-none-any.whl"


@pytest.mark.parametrize(
    ("first", "second", "unread"),
    [
        (
            (_PYPI_FILE, _PYPI_URL),
            (_PYPI_FILE, _PYPI_URL.replace("/aa/bb/", "/cc/dd/")),
            False,
        ),
        ((_PYPI_FILE, _PYPI_URL), (_BUILT_FILE, _PYPI_URL), True),
    ],
    ids=["same-file-other-address", "unread-same-address-other-file"],
)
def test_a_wheel_listed_twice_refuses_resolution(
    first: tuple[str, str], second: tuple[str, str], unread: bool
) -> None:
    # An unread wheel's address is checked only if it is selected, so a repeated
    # address must be caught before selection.
    def record(filename: str, url: str) -> dict[str, object]:
        if unread:
            reason = {"code": "wheel_range_unavailable", "message": "Server said no"}
            return {"filename": filename, "url": url, "hashes": {}, "_unread": reason}
        return {
            "filename": filename,
            "url": url,
            "hashes": {"sha256": "a" * 64},
            "size": 5,
            "core-metadata": {"sha256": "b" * 64},
        }

    document = {
        "meta": {"api-version": "1.0"},
        "name": "example-package",
        "files": [record(*first), record(*second)],
    }
    with pytest.raises(ComfyRegistryWheelArtifactError) as raised:
        resolve_comfy_registry_wheel_artifacts(
            plan_comfy_registry_dependencies(["example-package"]),
            {"example-package": document},
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )
    assert raised.value.code == "duplicate_wheel_artifact"


async def test_nvidia_project_page_must_be_html() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=httpx.ByteStream(b"{}"),
            headers={"content-type": "application/vnd.pypi.simple.v1+json"},
        )

    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(handler)
    assert raised.value.code == "invalid_project_content_type"


@pytest.mark.parametrize("declared", [False, True], ids=["streamed", "declared"])
async def test_nvidia_project_page_size_is_bounded(declared: bool) -> None:
    body = b"<p>" + b"x" * (128 * 1024)
    headers = {"content-type": "text/html"}
    if declared:
        headers["content-length"] = str(len(body))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(body), headers=headers)

    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(handler)
    assert raised.value.code == "project_metadata_too_large"


def _many_wheels_page(count: int) -> tuple[bytes, dict[str, bytes]]:
    wheels = {f"nvidia_vfx-0.0.{index}-py3-none-any.whl": _wheel() for index in range(count)}
    page = _page(
        *(_link(name, hashlib.sha256(wheel).hexdigest()) for name, wheel in wheels.items())
    )
    return page, wheels


async def test_wheel_reads_share_the_budget_left_by_earlier_projects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project_module, "MAX_REGISTRY_WHEEL_PROJECT_TOTAL_BYTES", 300_000)
    page, wheels = _many_wheels_page(5)
    earlier = json.dumps(
        {"meta": {"api-version": "1.0"}, "name": "alpha", "files": [], "note": "x" * 160_000}
    ).encode()
    wheel_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "pypi.org":
            return httpx.Response(
                200,
                stream=httpx.ByteStream(earlier),
                headers={"content-type": "application/vnd.pypi.simple.v1+json"},
            )
        if request.url.path == "/nvidia-vfx/":
            return _html(page)
        wheel_requests.append(request)
        return _ranged(wheels[request.url.path.rsplit("/", 1)[1]], request)

    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(handler, ["alpha", _NAME])
    assert raised.value.code == "project_metadata_total_too_large"
    assert len(wheel_requests) == 2


async def test_wheel_reads_stop_at_the_aggregate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(project_module, "MAX_REGISTRY_WHEEL_PROJECT_TOTAL_BYTES", 4 * _TAIL)
    page, wheels = _many_wheels_page(10)
    wheel_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/nvidia-vfx/":
            return _html(page)
        wheel_requests.append(request)
        return _ranged(wheels[request.url.path.rsplit("/", 1)[1]], request)

    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(handler)
    assert raised.value.code == "project_metadata_total_too_large"
    assert len(wheel_requests) == 3


async def test_wheel_reads_count_against_projects_fetched_after_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project_module, "MAX_REGISTRY_WHEEL_PROJECT_TOTAL_BYTES", 3 * _TAIL)
    page, wheels = _many_wheels_page(2)
    later = json.dumps(
        {"meta": {"api-version": "1.0"}, "name": "zeta", "files": [], "note": "x" * 70_000}
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "pypi.org":
            return httpx.Response(
                200,
                stream=httpx.ByteStream(later),
                headers={"content-type": "application/vnd.pypi.simple.v1+json"},
            )
        if request.url.path == "/nvidia-vfx/":
            return _html(page)
        return _ranged(wheels[request.url.path.rsplit("/", 1)[1]], request)

    with pytest.raises(ComfyRegistryWheelProjectError) as raised:
        await _fetch(handler, [_NAME, "zeta"])
    assert raised.value.code == "project_metadata_total_too_large"


async def test_nvidia_core_metadata_is_read_from_the_listed_wheel() -> None:
    wheel = _wheel(padding=200_000)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _ranged(wheel, request)

    client = ComfyRegistryWheelMetadataClient(transport=httpx.MockTransport(handler))
    try:
        documents = await client.fetch(_manifest(wheel))
    finally:
        await client.close()

    assert documents == {_FILENAME: _METADATA}
    assert len(requests) == 2
    assert {str(request.url) for request in requests} == {_WHEEL_URL}


@pytest.mark.parametrize(
    ("listed", "listed_metadata", "served", "code"),
    [
        (_wheel(), b"Name: other\n", _wheel(), "core_metadata_hash_mismatch"),
        (_wheel(), _METADATA, _wheel(padding=7), "core_metadata_size_mismatch"),
        (_wheel(), _METADATA, b"", "wheel_range_unavailable"),
    ],
    ids=["other-metadata", "other-size", "no-ranges"],
)
async def test_nvidia_core_metadata_must_match_the_listed_wheel(
    listed: bytes, listed_metadata: bytes, served: bytes, code: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if not served:
            return httpx.Response(200, content=listed)
        return _ranged(served, request)

    client = ComfyRegistryWheelMetadataClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ComfyRegistryWheelClosureDriverError) as raised:
            await client.fetch(_manifest(listed, listed_metadata))
    finally:
        await client.close()
    assert raised.value.code == code


@pytest.mark.parametrize(
    "wheel",
    [_wheel(), _wheel(extra_members={"nvidia_vfx/templates/METADATA": b"template"})],
    ids=["plain", "data-named-metadata"],
)
async def test_nvidia_wheels_stage_their_metadata_from_the_verified_wheel(
    tmp_path: Path, wheel: bytes
) -> None:
    destination = tmp_path / "staged"
    report = await _stage(wheel, destination)

    assert (destination / _FILENAME).read_bytes() == wheel
    assert (destination / f"{_FILENAME}.metadata").read_bytes() == _METADATA
    assert report.artifacts[0].metadata_size_bytes == len(_METADATA)
    assert report.total_bytes == len(wheel) + len(_METADATA)


@pytest.mark.parametrize(
    ("wheel", "code"),
    [
        (_wheel(metadata=_METADATA + b"Summary: changed\n"), "download_hash_mismatch"),
        (_wheel(second_metadata=True), "invalid_wheel_metadata"),
        (_undecodable_name_wheel(), "invalid_wheel_metadata"),
        (_wheel(compression=zipfile.ZIP_LZMA), "invalid_wheel_metadata"),
    ],
    ids=["other-metadata", "two-metadata", "undecodable-name", "lzma-metadata"],
)
async def test_an_nvidia_wheel_without_the_resolved_metadata_is_not_staged(
    tmp_path: Path, wheel: bytes, code: str
) -> None:
    destination = tmp_path / "staged"
    with pytest.raises(ComfyRegistryWheelDownloadError) as raised:
        await _stage(wheel, destination)

    assert raised.value.code == code
    assert not destination.exists()
    assert not list(tmp_path.glob(".registry-wheels-*"))
