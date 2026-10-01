from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

import local_lm.https_transfer as transfer_module
from local_lm import download_worker
from local_lm.filesystem_links import AnchoredDirectory, open_child_directory, open_entry
from local_lm.https_transfer import (
    HttpsArtifactRequest,
    HttpsTransferError,
    download_https_artifact,
)


def _stdin(payload: dict[str, Any]) -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(json.dumps(payload).encode()))


def test_download_worker_keeps_legacy_huggingface_payload_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def download(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "C:/models/model.gguf"

    output = io.StringIO()
    monkeypatch.setattr(download_worker, "hf_hub_download", download)
    monkeypatch.setattr(
        sys,
        "stdin",
        _stdin(
            {
                "repo_id": "owner/model",
                "filename": "model.gguf",
                "revision": "a" * 40,
                "local_dir": "C:/staging",
                "token": "secret",
            }
        ),
    )
    monkeypatch.setattr(sys, "stdout", output)

    assert download_worker.main() == 0
    assert json.loads(output.getvalue()) == {"path": "C:/models/model.gguf"}
    assert captured["repo_id"] == "owner/model"
    assert captured["token"] == "secret"


def test_download_worker_rejects_unknown_transfer_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", _stdin({"kind": "ftp"}))

    with pytest.raises(ValueError, match="unsupported download worker kind: ftp"):
        download_worker.main()


def _https_payload(tmp_path: Path, content: bytes = b"verified artifact") -> dict[str, Any]:
    return {
        "kind": "https",
        "url": "https://civitai.example/api/download/models/123",
        "filename": "models/example.safetensors",
        "local_dir": str(tmp_path),
        "expected_sha256": hashlib.sha256(content).hexdigest(),
        "expected_size": len(content),
        "allowed_hosts": ["civitai.example", "objects.civitai.example"],
        "bearer_token": "private-token",
    }


def _response(
    status: int,
    content: bytes = b"",
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(status, headers=headers, stream=httpx.ByteStream(content))


def test_https_transfer_verifies_and_atomically_publishes(tmp_path: Path) -> None:
    content = b"verified artifact"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(200, content, headers={"content-length": str(len(content))})

    path = download_https_artifact(
        _https_payload(tmp_path, content), transport=httpx.MockTransport(handler)
    )

    assert Path(path).read_bytes() == content
    assert requests[0].headers["authorization"] == "Bearer private-token"
    assert not list(tmp_path.rglob("*.https-partial"))


def test_https_transfer_strips_auth_on_an_allowed_redirect(tmp_path: Path) -> None:
    content = b"verified artifact"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "civitai.example":
            return httpx.Response(
                307,
                headers={"location": "https://objects.civitai.example/signed?secret=value"},
            )
        return _response(200, content)

    download_https_artifact(
        _https_payload(tmp_path, content), transport=httpx.MockTransport(handler)
    )

    assert requests[0].headers["authorization"] == "Bearer private-token"
    assert "authorization" not in requests[1].headers


def test_https_transfer_suppresses_signed_redirect_urls_from_http_logs(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    content = b"verified artifact"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "civitai.example":
            return httpx.Response(
                307,
                headers={"location": "https://objects.civitai.example/file?secret=value"},
            )
        return _response(200, content)

    download_https_artifact(
        _https_payload(tmp_path, content), transport=httpx.MockTransport(handler)
    )

    assert "secret=value" not in caplog.text


def test_https_transfer_refuses_redirect_outside_allowlist_without_leaking_url(
    tmp_path: Path,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.example/file?secret=value"})

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(_https_payload(tmp_path), transport=httpx.MockTransport(handler))

    assert raised.value.code == "untrusted_host"
    # The host is named so a redirect chain can be followed; the rest of the
    # address can carry credentials and stays out of it.
    assert "evil.example" in str(raised.value)
    assert "secret" not in str(raised.value)


def test_a_provider_storage_domain_is_reachable_by_suffix() -> None:
    """CivitAI hands anything large to Cloudflare R2, under a bucket of its own.

    Pinning the exact bucket meant every model above their small-file threshold
    refused at the second hop with "untrusted host", and a rotation on their
    side would do it again. A suffix entry names one provider's storage domain
    and widens nothing for any other source.
    """
    allowed = frozenset({"civitai.com", "b2.civitai.com", ".r2.cloudflarestorage.com"})

    assert transfer_module._host_allowed("civitai.com", allowed)
    assert transfer_module._host_allowed(
        "civitai-delivery-worker-prod.5ac0637cfd.r2.cloudflarestorage.com", allowed
    )
    # The bare domain itself, and nothing that merely ends in the same letters.
    assert transfer_module._host_allowed("r2.cloudflarestorage.com", allowed)
    assert not transfer_module._host_allowed("evil-r2.cloudflarestorage.com.attacker.test", allowed)
    assert not transfer_module._host_allowed("cloudflarestorage.com", allowed)
    # Only sources that declare the suffix get it.
    assert not transfer_module._host_allowed(
        "anything.r2.cloudflarestorage.com", frozenset({"civitai.com"})
    )


def test_https_transfer_still_refuses_a_storage_domain_no_source_declared(
    tmp_path: Path,
) -> None:
    """The suffix only helps the source that asked for it."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "https://someone-else.r2.cloudflarestorage.com/model.safetensors"},
        )

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(_https_payload(tmp_path), transport=httpx.MockTransport(handler))

    assert raised.value.code == "untrusted_host"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("url", "http://civitai.example/file", "invalid_url"),
        ("url", "https://civitai.example/file?token=secret", "credential_in_url"),
        ("filename", "../outside.bin", "invalid_filename"),
        ("expected_sha256", "not-a-digest", "invalid_sha256"),
        ("expected_size", True, "invalid_expected_size"),
        ("allowed_hosts", ["localhost"], "invalid_allowed_hosts"),
        ("bearer_token", "secret\r\ninjected: yes", "invalid_bearer_token"),
    ],
)
def test_https_transfer_rejects_invalid_envelope(
    tmp_path: Path,
    field: str,
    value: object,
    code: str,
) -> None:
    payload = _https_payload(tmp_path)
    payload[field] = value

    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("Invalid transfer envelopes must not send requests")

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(payload, transport=httpx.MockTransport(unexpected_request))

    assert raised.value.code == code


def test_https_transfer_rejects_a_linked_destination_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    linked_root = tmp_path / "linked"
    real_root.mkdir()
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError:
        pytest.skip("directory links are unavailable")
    payload = _https_payload(linked_root)

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(payload)

    assert raised.value.code == "unsafe_local_dir"


def test_https_transfer_rejects_a_linked_nested_destination_parent(tmp_path: Path) -> None:
    """Nested destination directories are opened through the held root.

    A linked local_dir is already refused. A junction planted as the first
    filename component is a different entry: mkdir-by-path would follow it.
    open_child_directory refuses the reparse instead.
    """
    real_models = tmp_path / "real-models"
    linked_models = tmp_path / "models"
    real_models.mkdir()
    try:
        linked_models.symlink_to(real_models, target_is_directory=True)
    except OSError:
        pytest.skip("directory links are unavailable")
    payload = _https_payload(tmp_path)

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(payload)

    assert raised.value.code == "unsafe_destination"


def test_prepare_destination_holds_the_nested_parent_after_a_name_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nested parent stays the opened directory after its path is replaced.

    The path-based walk looks the name up again for destination and partial.
    After open_child_directory, a swap would make those lookups follow the
    link. open_entry uses the held parent, so the original directory is still
    the one inspected. On the parent this test never reaches open_entry.
    """
    models = tmp_path / "models"
    models.mkdir()
    (models / "sentinel").write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("outside", encoding="utf-8")
    request = transfer_module._parse_request(_https_payload(tmp_path))
    original_open_entry = open_entry
    seen: list[str] = []

    def swap_then_open(parent: AnchoredDirectory, name: str) -> int | None:
        moved = tmp_path / "models-moved"
        if os.name == "nt":
            with pytest.raises(OSError):
                models.rename(moved)
        elif models.is_dir() and not models.is_symlink():
            models.rename(moved)
            try:
                (tmp_path / "models").symlink_to(outside, target_is_directory=True)
            except OSError:
                pytest.skip("directory links are unavailable")
        from local_lm.filesystem_links import list_entries

        seen.extend(entry.name for entry in list_entries(parent))
        return original_open_entry(parent, name)

    monkeypatch.setattr(transfer_module, "open_entry", swap_then_open)
    with transfer_module._hold_destination(request):
        assert "sentinel" in seen


def test_download_holds_the_nested_parent_through_the_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nested parent stays held while the body is written.

    Closing every handle before streaming hands back unheld paths. Listing
    the captured nested parent during _stream_response only works if that
    handle is still open. On the parent the listing refuses because the
    handle is already closed.
    """
    models = tmp_path / "models"
    models.mkdir()
    (models / "sentinel").write_text("inside", encoding="utf-8")
    content = b"verified artifact"
    captured: dict[str, AnchoredDirectory] = {}
    original_open_child = open_child_directory

    def wrap_open_child(
        parent: AnchoredDirectory, name: str, *, create: bool = False
    ) -> AnchoredDirectory:
        child = original_open_child(parent, name, create=create)
        captured["parent"] = child
        return child

    original_stream = transfer_module._stream_with_client
    seen: list[str] = []

    def wrap_stream(
        client: httpx.Client,
        request: HttpsArtifactRequest,
        descriptor: int,
        starting_size: int,
        digest: Any,
    ) -> tuple[int, str]:
        from local_lm.filesystem_links import list_entries

        seen.extend(entry.name for entry in list_entries(captured["parent"]))
        return original_stream(client, request, descriptor, starting_size, digest)

    monkeypatch.setattr(transfer_module, "open_child_directory", wrap_open_child)
    monkeypatch.setattr(transfer_module, "_stream_with_client", wrap_stream)

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(200, content, headers={"content-length": str(len(content))})

    path = download_https_artifact(
        _https_payload(tmp_path, content), transport=httpx.MockTransport(handler)
    )
    assert Path(path).read_bytes() == content
    assert "sentinel" in seen


def test_https_transfer_does_not_follow_a_partial_replaced_before_the_write(
    tmp_path: Path,
) -> None:
    """Replacing the partial name during the response must not receive the body.

    The destination directory is already held when the response arrives.
    Pointing the partial name at another file has to leave that file alone.
    """
    content = b"verified artifact"
    payload = _https_payload(tmp_path, content)
    partial = (
        tmp_path
        / "models"
        / f".example.safetensors.{payload['expected_sha256'][:12]}.https-partial"
    )
    outside = tmp_path / "outside-body"
    outside.write_bytes(b"")

    refused = False
    swapped = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal refused, swapped
        partial.parent.mkdir(parents=True, exist_ok=True)
        try:
            if partial.is_symlink() or partial.exists():
                partial.unlink()
            os.link(outside, partial)
        except PermissionError:
            # The partial is open, so this host will not let the name go.
            refused = True
        else:
            swapped = True
        return _response(200, content, headers={"content-length": str(len(content))})

    path = download_https_artifact(payload, transport=httpx.MockTransport(handler))

    assert outside.read_bytes() == b""
    destination = Path(path)
    assert not destination.is_symlink()
    assert destination.read_bytes() == content
    if os.name == "nt":
        assert refused
        assert not swapped
    else:
        assert swapped
        assert not refused


def _partial(tmp_path: Path, payload: dict[str, Any]) -> Path:
    return (
        tmp_path
        / "models"
        / f".example.safetensors.{payload['expected_sha256'][:12]}.https-partial"
    )


class _BytesThenReadError(httpx.SyncByteStream):
    """Yield one chunk, then fail the way a dropped connection does."""

    def __init__(self, chunk: bytes) -> None:
        self._chunk = chunk

    def __iter__(self) -> Any:
        yield self._chunk
        raise httpx.ReadError("connection reset")


def test_a_complete_partial_publishes_the_verified_bytes_when_its_name_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified partial is published from the descriptor that was hashed.

    Closing it and renaming the name lets a replacement at that name become
    the destination. The other file's bytes must not land there.
    """

    content = b"verified artifact"
    payload = _https_payload(tmp_path, content)
    partial = _partial(tmp_path, payload)
    partial.parent.mkdir()
    partial.write_bytes(content)
    outside = tmp_path / "outside-body"
    outside.write_bytes(b"x" * len(content))

    def swap() -> None:
        try:
            if partial.is_symlink() or partial.exists():
                partial.unlink()
            os.link(outside, partial)
        except OSError:
            pass

    real_rename = transfer_module.rename_entry
    real_publish = transfer_module.publish_opened_file

    def rename_after_swap(*args: Any, **kwargs: Any) -> None:
        swap()
        real_rename(*args, **kwargs)

    def publish_after_swap(*args: Any, **kwargs: Any) -> None:
        swap()
        real_publish(*args, **kwargs)

    monkeypatch.setattr(transfer_module, "rename_entry", rename_after_swap)
    monkeypatch.setattr(transfer_module, "publish_opened_file", publish_after_swap)

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("a verified partial does not contact the server")

    path = download_https_artifact(payload, transport=httpx.MockTransport(handler))

    assert Path(path).read_bytes() == content
    assert outside.read_bytes() == b"x" * len(content)


def test_a_resumed_transfer_survives_a_dropped_connection(tmp_path: Path) -> None:
    """Bytes received before a network error stay on the canonical partial.

    A hidden resume copy left beside that partial makes the next attempt
    refuse, so the interrupted download can never finish.
    """

    content = b"0123456789" * 4 + b"abcdef"
    assert len(content) == 46
    payload = _https_payload(tmp_path, content)
    partial = _partial(tmp_path, payload)
    partial.parent.mkdir()
    split = 5
    partial.write_bytes(content[:split])
    attempts = {"count": 0}

    def fail_then_finish(request: httpx.Request) -> httpx.Response:
        start = int(request.headers["range"].removeprefix("bytes=").rstrip("-"))
        attempts["count"] += 1
        if attempts["count"] == 1:
            assert start == split
            return httpx.Response(
                206,
                stream=_BytesThenReadError(content[split : split + 4]),
                headers={"content-range": f"bytes {split}-{len(content) - 1}/{len(content)}"},
            )
        assert start == len(partial.read_bytes())
        return _response(
            206,
            content[start:],
            headers={"content-range": f"bytes {start}-{len(content) - 1}/{len(content)}"},
        )

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(payload, transport=httpx.MockTransport(fail_then_finish))

    assert raised.value.code == "network_error"
    assert content.startswith(partial.read_bytes())
    assert len(partial.read_bytes()) >= split
    assert not list(tmp_path.rglob("*.resume"))

    path = download_https_artifact(payload, transport=httpx.MockTransport(fail_then_finish))

    assert Path(path).read_bytes() == content
    assert not list(tmp_path.rglob("*.https-partial"))
    assert not list(tmp_path.rglob("*.resume"))


def test_a_resumed_transfer_survives_a_filesystem_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write error keeps the bytes already stored and the next attempt finishes."""

    content = b"0123456789" * 4 + b"abcdef"
    payload = _https_payload(tmp_path, content)
    partial = _partial(tmp_path, payload)
    partial.parent.mkdir()
    split = 5
    partial.write_bytes(content[:split])
    kept = 4
    remainder = content[split:]
    real_write = os.write

    def fail_after_the_new_bytes(descriptor: int, data: bytes) -> int:
        if bytes(data) == remainder:
            real_write(descriptor, data[:kept])
            raise OSError("no space")
        return real_write(descriptor, data)

    monkeypatch.setattr(os, "write", fail_after_the_new_bytes)

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.headers["range"].removeprefix("bytes=").rstrip("-"))
        assert start == split
        return _response(
            206,
            remainder,
            headers={"content-range": f"bytes {split}-{len(content) - 1}/{len(content)}"},
        )

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(payload, transport=httpx.MockTransport(handler))

    assert raised.value.code == "filesystem_error"
    assert partial.read_bytes() == content[: split + kept]
    assert not list(tmp_path.rglob("*.resume"))

    monkeypatch.undo()

    def finish(request: httpx.Request) -> httpx.Response:
        start = int(request.headers["range"].removeprefix("bytes=").rstrip("-"))
        assert start == split + 4
        return _response(
            206,
            content[start:],
            headers={"content-range": f"bytes {start}-{len(content) - 1}/{len(content)}"},
        )

    path = download_https_artifact(payload, transport=httpx.MockTransport(finish))

    assert Path(path).read_bytes() == content


def test_a_completed_resume_survives_a_failed_final_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sync failure after the resumed body is stored folds that file back.

    The final sync is outside the stream's error handler. Leaving the resume
    file in place makes the next attempt refuse, so the finished bytes never
    land at the destination.
    """

    content = b"0123456789" * 4 + b"abcdef"
    payload = _https_payload(tmp_path, content)
    partial = _partial(tmp_path, payload)
    partial.parent.mkdir()
    split = 5
    partial.write_bytes(content[:split])
    calls = {"count": 0}
    real_fsync = os.fsync

    def fail_the_final_sync(descriptor: int) -> None:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise OSError("sync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_the_final_sync)

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.headers["range"].removeprefix("bytes=").rstrip("-"))
        assert start == split
        return _response(
            206,
            content[split:],
            headers={"content-range": f"bytes {split}-{len(content) - 1}/{len(content)}"},
        )

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(payload, transport=httpx.MockTransport(handler))

    assert raised.value.code == "filesystem_error"
    assert partial.read_bytes() == content
    assert not list(tmp_path.rglob("*.resume"))

    monkeypatch.undo()

    def refuse_network(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("the stored body is already complete")

    path = download_https_artifact(payload, transport=httpx.MockTransport(refuse_network))

    assert Path(path).read_bytes() == content


def test_a_leftover_resume_file_does_not_block_the_next_attempt(tmp_path: Path) -> None:
    """A resume file left beside the partial must not stop the next attempt.

    The partial holds the last folded prefix. The leftover name is stale, so
    the transfer continues from that prefix and stores the whole body.
    """

    content = b"0123456789" * 4 + b"abcdef"
    assert len(content) == 46
    payload = _https_payload(tmp_path, content)
    partial = _partial(tmp_path, payload)
    partial.parent.mkdir()
    split = 5
    partial.write_bytes(content[:split])
    leftover = partial.with_name(partial.name + ".resume")
    leftover.write_bytes(content[:9])

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.headers["range"].removeprefix("bytes=").rstrip("-"))
        assert start == split
        return _response(
            206,
            content[start:],
            headers={"content-range": f"bytes {start}-{len(content) - 1}/{len(content)}"},
        )

    path = download_https_artifact(payload, transport=httpx.MockTransport(handler))

    assert Path(path).read_bytes() == content
    assert not list(tmp_path.rglob("*.https-partial"))
    assert not list(tmp_path.rglob("*.resume"))


def test_a_full_partial_with_the_wrong_digest_is_replaced_by_the_body(tmp_path: Path) -> None:
    """A complete partial that does not match is removed, and the next fetch stores the body."""

    content = b"verified artifact"
    payload = _https_payload(tmp_path, content)
    partial = _partial(tmp_path, payload)
    partial.parent.mkdir()
    partial.write_bytes(b"x" * len(content))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("range"):
            return httpx.Response(416, headers={"content-range": f"bytes */{len(content)}"})
        return _response(200, content, headers={"content-length": str(len(content))})

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(payload, transport=httpx.MockTransport(handler))

    assert raised.value.code == "digest_mismatch"
    assert not list(tmp_path.rglob("*.https-partial"))
    assert not list(tmp_path.rglob("*.resume"))

    path = download_https_artifact(payload, transport=httpx.MockTransport(handler))

    assert Path(path).read_bytes() == content


def test_https_transfer_resumes_only_from_the_exact_content_range(tmp_path: Path) -> None:
    content = b"verified artifact"
    payload = _https_payload(tmp_path, content)
    partial = (
        tmp_path
        / "models"
        / (f".example.safetensors.{payload['expected_sha256'][:12]}.https-partial")
    )
    partial.parent.mkdir()
    split = 5
    partial.write_bytes(content[:split])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["range"] == f"bytes={split}-"
        return _response(
            206,
            content[split:],
            headers={"content-range": f"bytes {split}-{len(content) - 1}/{len(content)}"},
        )

    path = download_https_artifact(payload, transport=httpx.MockTransport(handler))

    assert Path(path).read_bytes() == content


def test_https_transfer_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    content = b"verified artifact"
    payload = _https_payload(tmp_path, content)
    partial = (
        tmp_path
        / "models"
        / (f".example.safetensors.{payload['expected_sha256'][:12]}.https-partial")
    )
    partial.parent.mkdir()
    partial.write_bytes(content[:4])

    path = download_https_artifact(
        payload,
        transport=httpx.MockTransport(lambda _request: _response(200, content)),
    )

    assert Path(path).read_bytes() == content


def test_https_transfer_rejects_incorrect_resume_range(tmp_path: Path) -> None:
    content = b"verified artifact"
    payload = _https_payload(tmp_path, content)
    partial = (
        tmp_path
        / "models"
        / (f".example.safetensors.{payload['expected_sha256'][:12]}.https-partial")
    )
    partial.parent.mkdir()
    partial.write_bytes(content[:4])

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(
            payload,
            transport=httpx.MockTransport(
                lambda _request: _response(
                    206,
                    content[4:],
                    headers={"content-range": f"bytes 3-{len(content) - 1}/{len(content)}"},
                )
            ),
        )

    assert raised.value.code == "invalid_content_range"


def test_https_transfer_rejects_lying_content_length(tmp_path: Path) -> None:
    content = b"verified artifact"

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(
            _https_payload(tmp_path, content),
            transport=httpx.MockTransport(
                lambda _request: _response(
                    200,
                    content,
                    headers={"content-length": str(len(content) + 1)},
                )
            ),
        )

    assert raised.value.code == "invalid_content_length"


def test_https_transfer_removes_oversized_and_digest_mismatch_partials(tmp_path: Path) -> None:
    expected = b"verified artifact"
    cases = (
        (expected + b"!", "oversized_body"),
        (b"x" * len(expected), "digest_mismatch"),
    )
    for body, code in cases:
        attempt = tmp_path / code
        attempt.mkdir()
        with pytest.raises(HttpsTransferError) as raised:
            download_https_artifact(
                _https_payload(attempt, expected),
                transport=httpx.MockTransport(lambda _request, body=body: _response(200, body)),
            )
        assert raised.value.code == code
        assert not list(attempt.rglob("*.https-partial"))


def test_https_transfer_keeps_truncated_partial_for_a_valid_resume(tmp_path: Path) -> None:
    content = b"verified artifact"
    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(
            _https_payload(tmp_path, content),
            transport=httpx.MockTransport(lambda _request: _response(200, content[:5])),
        )

    assert raised.value.code == "truncated_body"
    assert [path.read_bytes() for path in tmp_path.rglob("*.https-partial")] == [content[:5]]


def test_https_download_worker_returns_only_the_verified_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"verified artifact"
    payload = _https_payload(tmp_path, content)
    destination = tmp_path / "models" / "example.safetensors"
    destination.parent.mkdir()
    destination.write_bytes(content)
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", _stdin(payload))
    monkeypatch.setattr(sys, "stdout", output)

    assert download_worker.main() == 0
    assert json.loads(output.getvalue()) == {"path": str(destination)}


def test_https_download_worker_reports_only_a_stable_error_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _https_payload(tmp_path)
    monkeypatch.setattr(sys, "stdin", _stdin(payload))
    monkeypatch.setattr(
        download_worker,
        "download_https_artifact",
        lambda _payload: (_ for _ in ()).throw(HttpsTransferError("unauthorized")),
    )

    with pytest.raises(ValueError) as raised:
        download_worker.main()

    surfaced = str(raised.value)
    assert surfaced == "https transfer failed: unauthorized"
    assert payload["bearer_token"] not in surfaced
    assert payload["url"] not in surfaced
