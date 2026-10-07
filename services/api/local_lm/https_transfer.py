from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlsplit

import httpx

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    create_publishable_entry,
    open_child_directory,
    open_entry,
    open_publishable_entry,
    publish_opened_file,
    remove_entry,
    rename_entry,
)

_CHUNK_BYTES = 1024 * 1024
_MAX_ALLOWED_HOSTS = 16
_MAX_EXPECTED_BYTES = 1024**4
_MAX_REDIRECTS = 5
_MAX_URL_LENGTH = 8_192
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
    with _hold_destination(request) as (parent, destination, partial):
        try:
            return _transfer_held(request, parent, destination.name, partial.name, transport)
        except AnchoredEntryExists as exc:
            raise HttpsTransferError("destination_conflict") from exc
        except AnchoredDirectoryError as exc:
            raise HttpsTransferError("unsafe_destination") from exc


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


@contextmanager
def _hold_destination(
    request: HttpsArtifactRequest,
) -> Iterator[tuple[AnchoredDirectory, Path, Path]]:
    try:
        root = AnchoredDirectory(request.local_dir)
    except AnchoredDirectoryError as exc:
        raise HttpsTransferError("unsafe_local_dir") from exc
    held = [root]
    try:
        parent = root
        for part in request.filename.parts[:-1]:
            try:
                child = open_child_directory(parent, part, create=True)
            except AnchoredDirectoryError as exc:
                raise HttpsTransferError("unsafe_destination") from exc
            held.append(child)
            parent = child
        destination_name = request.filename.name
        partial_name = f".{destination_name}.{request.expected_sha256[:12]}.https-partial"
        for name in (destination_name, partial_name):
            try:
                descriptor = open_entry(parent, name)
            except AnchoredDirectoryError as exc:
                raise HttpsTransferError("unsafe_destination") from exc
            if descriptor is not None:
                os.close(descriptor)
        yield parent, parent.path / destination_name, parent.path / partial_name
    finally:
        for directory in reversed(held):
            directory.close()


def _transfer_held(
    request: HttpsArtifactRequest,
    parent: AnchoredDirectory,
    destination_name: str,
    partial_name: str,
    transport: httpx.BaseTransport | None,
) -> str:
    """Download into a descriptor opened in the held directory, then publish it.

    The partial name is predictable, so a response can replace that name with
    another file before a path open. The descriptor is opened first and is the
    only object written. Publishing moves that same object.
    """

    destination = str(parent.path / destination_name)
    existing = open_entry(parent, destination_name)
    if existing is not None:
        try:
            if _descriptor_matches(existing, request):
                return destination
        finally:
            os.close(existing)
        raise HttpsTransferError("destination_conflict")

    write_name, descriptor, starting_size, digest = _open_partial_descriptor(
        parent, partial_name, destination_name, request
    )
    if descriptor is None:
        return destination
    # Name-based removal opens a second handle. Windows refuses that while the
    # write handle is still open, and the refusal would hide the transfer's
    # own error. Close first, then drop or keep the name.
    failure: HttpsTransferError | None = None
    try:
        try:
            downloaded, digest_hex = _stream_response(
                request, descriptor, starting_size, digest, transport
            )
        except HttpsTransferError as exc:
            failure = exc
        except httpx.HTTPError:
            failure = HttpsTransferError("network_error")
        except OSError:
            failure = HttpsTransferError("filesystem_error")
        else:
            # fstat and the final sync sit outside the stream's OSError handler.
            # A failure there is still a filesystem error: fold the staging file
            # back, or the next attempt finds the resume name already taken.
            try:
                if (
                    os.fstat(descriptor).st_size != downloaded
                    or downloaded != request.expected_size
                ):
                    failure = HttpsTransferError("truncated_body")
                elif digest_hex != request.expected_sha256:
                    failure = HttpsTransferError("digest_mismatch")
                else:
                    os.fsync(descriptor)
            except OSError:
                failure = HttpsTransferError("filesystem_error")
            if failure is None:
                publish_opened_file(
                    parent,
                    write_name,
                    descriptor,
                    into=parent,
                    destination=destination_name,
                )
                if write_name != partial_name:
                    remove_entry(parent, partial_name)
                return destination
    finally:
        os.close(descriptor)
    assert failure is not None
    if failure.code in {"truncated_body", "network_error", "filesystem_error"}:
        _preserve_truncated_partial(parent, write_name, partial_name)
    else:
        _discard_failed_partial(parent, write_name, failure.code)
    raise failure


def _open_partial_descriptor(
    parent: AnchoredDirectory,
    partial_name: str,
    destination_name: str,
    request: HttpsArtifactRequest,
) -> tuple[str, int | None, int, Any]:
    """Open the partial for writing, or publish it when it is already complete.

    None for the descriptor means the verified partial was published and the
    caller is done. An incomplete partial is copied into a new exclusive file
    so the later write does not reopen the predictable name.
    """

    opened = open_publishable_entry(parent, partial_name)
    if opened is None:
        return partial_name, _create_partial(parent, partial_name), 0, hashlib.sha256()
    try:
        size = os.fstat(opened).st_size
        if size == request.expected_size:
            if _sha256_descriptor(opened) == request.expected_sha256:
                publish_opened_file(
                    parent,
                    partial_name,
                    opened,
                    into=parent,
                    destination=destination_name,
                )
                return partial_name, None, size, hashlib.sha256()
            os.close(opened)
            opened = None
            remove_entry(parent, partial_name)
            raise HttpsTransferError("digest_mismatch")
        if size > request.expected_size:
            os.close(opened)
            opened = None
            remove_entry(parent, partial_name)
            return partial_name, _create_partial(parent, partial_name), 0, hashlib.sha256()
        digest = hashlib.sha256()
        os.lseek(opened, 0, os.SEEK_SET)
        staging_name = f"{partial_name}.resume"
        # A killed transfer can leave this name beside the partial. That open
        # file cannot be appended to, and creating the name again would refuse,
        # so the stale file goes before the new copy is created.
        remove_entry(parent, staging_name)
        descriptor = _create_partial(parent, staging_name)
        try:
            copied = _copy_descriptor(opened, descriptor, digest)
        except OSError:
            os.close(descriptor)
            remove_entry(parent, staging_name)
            raise
        return staging_name, descriptor, copied, digest
    finally:
        if opened is not None:
            os.close(opened)


def _create_partial(parent: AnchoredDirectory, name: str) -> int:
    try:
        return create_publishable_entry(parent, name)
    except AnchoredEntryExists as exc:
        raise HttpsTransferError("unsafe_destination") from exc


def _copy_descriptor(source: int, destination: int, digest: Any) -> int:
    copied = 0
    while True:
        chunk = os.read(source, _CHUNK_BYTES)
        if not chunk:
            return copied
        _write_all(destination, chunk)
        digest.update(chunk)
        copied += len(chunk)


def _write_all(descriptor: int, chunk: bytes) -> None:
    view = memoryview(chunk)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("incomplete write")
        view = view[written:]


def _sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, _CHUNK_BYTES)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _discard_failed_partial(parent: AnchoredDirectory, write_name: str, code: str) -> None:
    """Drop a failed new file. Keep bytes that a later resume can continue."""

    if code in {"truncated_body", "network_error", "filesystem_error"}:
        return
    remove_entry(parent, write_name)


def _preserve_truncated_partial(
    parent: AnchoredDirectory,
    write_name: str,
    partial_name: str,
) -> None:
    if write_name != partial_name:
        rename_entry(parent, write_name, partial_name, replace=True)


def _descriptor_matches(descriptor: int, request: HttpsArtifactRequest) -> bool:
    return (
        os.fstat(descriptor).st_size == request.expected_size
        and _sha256_descriptor(descriptor) == request.expected_sha256
    )


def _stream_response(
    request: HttpsArtifactRequest,
    descriptor: int,
    starting_size: int,
    digest: Any,
    transport: httpx.BaseTransport | None,
) -> tuple[int, str]:
    timeout = httpx.Timeout(connect=20, read=60, write=20, pool=20)
    with (
        _quiet_http_loggers(),
        httpx.Client(
            follow_redirects=False,
            timeout=timeout,
            transport=transport,
            trust_env=False,
        ) as client,
    ):
        return _stream_with_client(client, request, descriptor, starting_size, digest)


def _stream_with_client(
    client: httpx.Client,
    request: HttpsArtifactRequest,
    descriptor: int,
    starting_size: int,
    digest: Any,
) -> tuple[int, str]:
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
            return _consume_response(response, descriptor, request, starting_size, digest)
    raise HttpsTransferError("too_many_redirects")


def _consume_response(
    response: httpx.Response,
    descriptor: int,
    request: HttpsArtifactRequest,
    starting_size: int,
    digest: Any,
) -> tuple[int, str]:
    if response.status_code == 416 and starting_size == request.expected_size:
        content_range = response.headers.get("content-range")
        if content_range and content_range != f"bytes */{request.expected_size}":
            raise HttpsTransferError("invalid_content_range")
        return starting_size, digest.hexdigest()
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
        expected_body = _validate_content_range(response, request, starting_size)
    elif response.status_code == 200:
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        downloaded = 0
        digest = hashlib.sha256()
        expected_body = request.expected_size
    else:
        raise HttpsTransferError("unexpected_partial_response")
    _validate_content_length(response, expected_body)

    for chunk in response.iter_raw(_CHUNK_BYTES):
        if not chunk:
            continue
        downloaded += len(chunk)
        if downloaded > request.expected_size:
            raise HttpsTransferError("oversized_body")
        _write_all(descriptor, chunk)
        digest.update(chunk)
    os.fsync(descriptor)
    return downloaded, digest.hexdigest()


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
