from __future__ import annotations

import hashlib
import logging
import os
import re
import stat
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlsplit

import httpx

_CHUNK_BYTES = 1024 * 1024
_MAX_ALLOWED_HOSTS = 16
_MAX_EXPECTED_BYTES = 1024**4
_MAX_REDIRECTS = 5
_MAX_URL_LENGTH = 8_192
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HOST = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z"
)
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)\Z")
_SENSITIVE_QUERY_KEYS = {"access_token", "api_key", "apikey", "authorization", "token"}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class HttpsTransferError(RuntimeError):
    def __init__(self, code: str, detail: str | None = None) -> None:
        # The code stays a stable identifier that callers branch on; detail is
        # the part a person reads. They are kept apart so naming what happened
        # can never change what the code means.
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class HttpsArtifactRequest:
    url: str
    local_dir: Path
    filename: PurePosixPath
    expected_sha256: str
    expected_size: int
    allowed_hosts: frozenset[str]
    bearer_token: str | None


def download_https_artifact(
    payload: Mapping[str, Any],
    *,
    transport: httpx.BaseTransport | None = None,
) -> str:
    request = _parse_request(payload)
    destination, partial = _prepare_destination(request)
    if _verified_file(destination, request):
        return str(destination)
    if destination.exists() or destination.is_symlink():
        raise HttpsTransferError("destination_conflict")
    if _verified_partial(partial, destination, request):
        return str(destination)

    starting_size = partial.stat().st_size if partial.is_file() else 0
    if starting_size > request.expected_size:
        partial.unlink()
        starting_size = 0

    timeout = httpx.Timeout(connect=20, read=60, write=20, pool=20)
    try:
        with (
            _quiet_http_loggers(),
            httpx.Client(
                follow_redirects=False,
                timeout=timeout,
                transport=transport,
                trust_env=False,
            ) as client,
        ):
            downloaded = _stream_response(client, request, partial, starting_size)
    except HttpsTransferError:
        raise
    except (httpx.HTTPError, OSError) as exc:
        code = "network_error" if isinstance(exc, httpx.HTTPError) else "filesystem_error"
        raise HttpsTransferError(code) from None

    if downloaded != request.expected_size:
        raise HttpsTransferError("truncated_body")
    if _sha256_file(partial) != request.expected_sha256:
        partial.unlink(missing_ok=True)
        raise HttpsTransferError("digest_mismatch")
    os.replace(partial, destination)
    return str(destination)


def _parse_request(payload: Mapping[str, Any]) -> HttpsArtifactRequest:
    url = payload.get("url")
    local_dir = payload.get("local_dir")
    filename = payload.get("filename")
    digest = payload.get("expected_sha256")
    expected_size = payload.get("expected_size")
    raw_hosts = payload.get("allowed_hosts")
    token = payload.get("bearer_token")
    if not isinstance(url, str) or not url or len(url) > _MAX_URL_LENGTH:
        raise HttpsTransferError("invalid_url")
    if not isinstance(local_dir, str) or not local_dir:
        raise HttpsTransferError("invalid_local_dir")
    if not isinstance(filename, str) or not _safe_relative_filename(filename):
        raise HttpsTransferError("invalid_filename")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise HttpsTransferError("invalid_sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or not 0 < expected_size <= _MAX_EXPECTED_BYTES
    ):
        raise HttpsTransferError("invalid_expected_size")
    if (
        not isinstance(raw_hosts, list)
        or not raw_hosts
        or len(raw_hosts) > _MAX_ALLOWED_HOSTS
        or any(not isinstance(host, str) for host in raw_hosts)
    ):
        raise HttpsTransferError("invalid_allowed_hosts")
    allowed_hosts = frozenset(_normalize_host(host) for host in raw_hosts)
    if len(allowed_hosts) != len(raw_hosts):
        raise HttpsTransferError("invalid_allowed_hosts")
    if token is not None and (
        not isinstance(token, str)
        or not token
        or len(token) > 10_000
        or token != token.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in token)
    ):
        raise HttpsTransferError("invalid_bearer_token")
    _validate_url(url, allowed_hosts, initial=True)
    root = Path(local_dir)
    if not root.is_absolute():
        raise HttpsTransferError("invalid_local_dir")
    return HttpsArtifactRequest(
        url=url,
        local_dir=root,
        filename=PurePosixPath(filename),
        expected_sha256=digest,
        expected_size=expected_size,
        allowed_hosts=allowed_hosts,
        bearer_token=token,
    )


def _prepare_destination(request: HttpsArtifactRequest) -> tuple[Path, Path]:
    root = request.local_dir
    if not root.is_dir() or _is_link_or_reparse(root):
        raise HttpsTransferError("unsafe_local_dir")
    resolved_root = root.resolve(strict=True)
    if _is_link_or_reparse(resolved_root):
        raise HttpsTransferError("unsafe_local_dir")
    parent = resolved_root
    for part in request.filename.parts[:-1]:
        parent /= part
        if parent.exists() or parent.is_symlink():
            if not parent.is_dir() or _is_link_or_reparse(parent):
                raise HttpsTransferError("unsafe_destination")
        else:
            parent.mkdir()
    destination = parent / request.filename.name
    partial = parent / f".{request.filename.name}.{request.expected_sha256[:12]}.https-partial"
    for path in (destination, partial):
        if _is_link_or_reparse(path) or (path.exists() and not path.is_file()):
            raise HttpsTransferError("unsafe_destination")
    return destination, partial


def _stream_response(
    client: httpx.Client,
    request: HttpsArtifactRequest,
    partial: Path,
    starting_size: int,
) -> int:
    current_url = request.url
    redirected = False
    for redirect_count in range(_MAX_REDIRECTS + 1):
        headers = {"Accept": "application/octet-stream", "Accept-Encoding": "identity"}
        if starting_size:
            headers["Range"] = f"bytes={starting_size}-"
        if request.bearer_token and not redirected:
            headers["Authorization"] = f"Bearer {request.bearer_token}"
        with client.stream("GET", current_url, headers=headers) as response:
            if response.status_code in _REDIRECT_STATUSES:
                if redirect_count == _MAX_REDIRECTS:
                    raise HttpsTransferError("too_many_redirects")
                location = response.headers.get("location")
                if not location:
                    raise HttpsTransferError("invalid_redirect")
                current_url = urljoin(current_url, location)
                _validate_url(current_url, request.allowed_hosts, initial=False)
                redirected = True
                continue
            return _consume_response(response, partial, request, starting_size)
    raise HttpsTransferError("too_many_redirects")


def _consume_response(
    response: httpx.Response,
    partial: Path,
    request: HttpsArtifactRequest,
    starting_size: int,
) -> int:
    if response.status_code == 416 and starting_size == request.expected_size:
        content_range = response.headers.get("content-range")
        if content_range and content_range != f"bytes */{request.expected_size}":
            raise HttpsTransferError("invalid_content_range")
        return starting_size
    if response.status_code in {401, 403}:
        raise HttpsTransferError("unauthorized")
    if response.status_code == 429:
        raise HttpsTransferError("rate_limited")
    if response.status_code >= 500:
        raise HttpsTransferError("remote_unavailable")
    if response.status_code not in {200, 206}:
        raise HttpsTransferError("unexpected_status")
    if response.headers.get("content-encoding", "identity").casefold() != "identity":
        raise HttpsTransferError("encoded_body")

    if starting_size and response.status_code == 206:
        downloaded = starting_size
        mode = "ab"
        expected_body = _validate_content_range(response, request, starting_size)
    elif response.status_code == 200:
        downloaded = 0
        mode = "wb"
        expected_body = request.expected_size
    else:
        raise HttpsTransferError("unexpected_partial_response")
    _validate_content_length(response, expected_body)

    with partial.open(mode) as output:
        for chunk in response.iter_raw(_CHUNK_BYTES):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > request.expected_size:
                output.close()
                partial.unlink(missing_ok=True)
                raise HttpsTransferError("oversized_body")
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    return downloaded


def _validate_content_range(
    response: httpx.Response,
    request: HttpsArtifactRequest,
    starting_size: int,
) -> int:
    match = _CONTENT_RANGE.fullmatch(response.headers.get("content-range", ""))
    if not match:
        raise HttpsTransferError("invalid_content_range")
    start, end, total = (int(value) for value in match.groups())
    if (
        start != starting_size
        or end < start
        or end >= request.expected_size
        or total != request.expected_size
    ):
        raise HttpsTransferError("invalid_content_range")
    return end - start + 1


def _validate_content_length(response: httpx.Response, expected: int) -> None:
    raw_length = response.headers.get("content-length")
    if raw_length is None:
        return
    if not raw_length.isdecimal() or int(raw_length) != expected:
        raise HttpsTransferError("invalid_content_length")


def _validate_url(url: str, allowed_hosts: frozenset[str], *, initial: bool) -> None:
    if not url or len(url) > _MAX_URL_LENGTH:
        raise HttpsTransferError("invalid_url")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise HttpsTransferError("invalid_url") from None
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise HttpsTransferError("invalid_url")
    host = _normalize_host(parsed.hostname)
    if not _host_allowed(host, allowed_hosts):
        # The host is named, never the URL: a redirect chain is invisible from
        # outside, so a bare "untrusted host" meant working out by hand where a
        # provider had sent the transfer. The rest of the address can carry
        # credentials and stays out of it.
        raise HttpsTransferError("untrusted_host", host)
    if initial and any(
        key.casefold() in _SENSITIVE_QUERY_KEYS for key, _value in parse_qsl(parsed.query)
    ):
        raise HttpsTransferError("credential_in_url")


def _host_allowed(host: str, allowed_hosts: Collection[str]) -> bool:
    """Whether a redirect target is one this source is allowed to reach.

    Exact names, with one deliberate exception: an entry written as a leading
    dot is a domain suffix. Providers serve their larger objects from storage
    domains whose bucket name is an account detail rather than an identity -
    CivitAI hands files over about a gigabyte to Cloudflare R2 - and pinning
    the exact bucket means a routine rotation on their side reads here as an
    untrusted host, with the download simply failing.

    A suffix entry still names one provider's storage domain, and only the
    source that declares it is affected: nothing widens for anyone else.
    """
    if host in allowed_hosts:
        return True
    return any(
        entry.startswith(".") and (host.endswith(entry) or host == entry[1:])
        for entry in allowed_hosts
    )


def _normalize_host(value: str) -> str:
    host = value.casefold()
    # A leading dot marks a domain suffix in an allowlist. The rest still has
    # to be a well-formed host, so the entry names one domain rather than
    # opening a wildcard.
    candidate = host[1:] if host.startswith(".") else host
    if (
        host != value.strip().casefold()
        or host.endswith(".")
        or not candidate
        or not _HOST.fullmatch(candidate)
    ):
        raise HttpsTransferError("invalid_allowed_hosts")
    return host


def _safe_relative_filename(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        value
        and len(value) <= 1_000
        and "\\" not in value
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} and ":" not in part for part in path.parts)
    )


def _verified_file(path: Path, request: HttpsArtifactRequest) -> bool:
    return bool(
        path.is_file()
        and not _is_link_or_reparse(path)
        and path.stat().st_size == request.expected_size
        and _sha256_file(path) == request.expected_sha256
    )


def _verified_partial(
    partial: Path,
    destination: Path,
    request: HttpsArtifactRequest,
) -> bool:
    if not partial.is_file() or partial.stat().st_size != request.expected_size:
        return False
    if _sha256_file(partial) != request.expected_sha256:
        partial.unlink(missing_ok=True)
        raise HttpsTransferError("digest_mismatch")
    os.replace(partial, destination)
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _quiet_http_loggers() -> Iterator[None]:
    loggers = tuple(logging.getLogger(name) for name in ("httpx", "httpcore"))
    disabled = tuple(logger.disabled for logger in loggers)
    try:
        for logger in loggers:
            logger.disabled = True
        yield
    finally:
        for logger, previous in zip(loggers, disabled, strict=True):
            logger.disabled = previous


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)
