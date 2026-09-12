from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

import local_lm.https_transfer as transfer_module
from local_lm import download_worker
from local_lm.https_transfer import HttpsTransferError, download_https_artifact


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
        download_worker.sys,
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
    monkeypatch.setattr(download_worker.sys, "stdout", output)

    assert download_worker.main() == 0
    assert json.loads(output.getvalue()) == {"path": "C:/models/model.gguf"}
    assert captured["repo_id"] == "owner/model"
    assert captured["token"] == "secret"


def test_download_worker_rejects_unknown_transfer_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(download_worker.sys, "stdin", _stdin({"kind": "ftp"}))

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

    with pytest.raises(HttpsTransferError) as raised:
        download_https_artifact(payload, transport=httpx.MockTransport(lambda _request: None))

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
    original_open_entry = transfer_module.open_entry
    seen: list[str] = []

    def swap_then_open(parent: object, name: str) -> int | None:
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
    captured: dict[str, object] = {}
    original_open_child = transfer_module.open_child_directory

    def wrap_open_child(parent: object, name: str, *, create: bool = False) -> object:
        child = original_open_child(parent, name, create=create)
        captured["parent"] = child
        return child

    original_stream = transfer_module._stream_response
    seen: list[str] = []

    def wrap_stream(*args: object, **kwargs: object) -> int:
        from local_lm.filesystem_links import list_entries

        seen.extend(entry.name for entry in list_entries(captured["parent"]))
        return original_stream(*args, **kwargs)

    monkeypatch.setattr(transfer_module, "open_child_directory", wrap_open_child)
    monkeypatch.setattr(transfer_module, "_stream_response", wrap_stream)

    def handler(request: httpx.Request) -> httpx.Response:
        return _response(200, content, headers={"content-length": str(len(content))})

    path = download_https_artifact(
        _https_payload(tmp_path, content), transport=httpx.MockTransport(handler)
    )
    assert Path(path).read_bytes() == content
    assert "sentinel" in seen


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
    monkeypatch.setattr(download_worker.sys, "stdin", _stdin(payload))
    monkeypatch.setattr(download_worker.sys, "stdout", output)

    assert download_worker.main() == 0
    assert json.loads(output.getvalue()) == {"path": str(destination)}


def test_https_download_worker_reports_only_a_stable_error_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _https_payload(tmp_path)
    monkeypatch.setattr(download_worker.sys, "stdin", _stdin(payload))
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
