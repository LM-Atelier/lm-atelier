from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

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
    assert "secret" not in str(raised.value)


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
