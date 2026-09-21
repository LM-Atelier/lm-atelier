from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session
from test_comfy_registry_reviewed_wheel_metadata import _HEADERS, _review_metadata
from test_comfy_registry_source_artifacts import DECLARATION
from test_comfy_registry_wheel_artifacts import _document, _environment, _file, _resolve

from local_lm import comfy_registry_wheel_downloads as downloads
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_source_artifacts import (
    ComfyRegistrySourceArtifactError,
    verified_reviewed_source_wheel,
)
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.comfy_registry_wheel_inputs_v1 import (
    ComfyRegistryReviewedWheelInput,
    ComfyRegistryWheelInputError,
    build_comfy_registry_wheel_input_manifest,
    parse_wheel_input_manifest,
    reviewed_wheel_input,
)
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.models import ComfyRegistrySourceArtifactReview


@pytest.fixture
def source_review_context(tmp_path: Path) -> Iterator[tuple[Session, ArtifactStore]]:
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'reviews.sqlite3').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    settings = Settings(data_dir=tmp_path / "data", dev=True)
    settings.prepare()
    try:
        with Session(engine) as session:
            yield session, ArtifactStore(settings)
    finally:
        engine.dispose()


def _local(session: Session, store: ArtifactStore) -> ComfyRegistryReviewedWheelInput:
    _review_metadata(session, store, _HEADERS + b"Requires-Python: >=3.12\n\n")
    return reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )


@pytest.mark.parametrize("include_remote", [False, True])
async def test_mixed_stage_copies_verified_local_bytes_without_a_download_url(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    include_remote: bool,
) -> None:
    session, store = source_review_context
    local = _local(session, store)
    verified = verified_reviewed_source_wheel(session, store, declaration=DECLARATION)
    remote_bytes = b"remote wheel bytes"
    remote_metadata = b"remote metadata bytes"
    remote = _resolve(
        ["example-package==1.2.3"] if include_remote else [],
        {
            "example-package": _document(
                _file(
                    hashes={"sha256": hashlib.sha256(remote_bytes).hexdigest()},
                    size=len(remote_bytes),
                    **{"core-metadata": {"sha256": hashlib.sha256(remote_metadata).hexdigest()}},
                )
            )
        }
        if include_remote
        else {},
    )
    manifest = build_comfy_registry_wheel_input_manifest("f" * 64, remote, [local])
    requests: list[str] = []
    sessions: list[Session] = []

    def fresh_session() -> Session:
        created = Session(session.get_bind())
        sessions.append(created)
        return created

    def fetch(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200, content=remote_metadata if request.url.path.endswith(".metadata") else remote_bytes
        )

    downloader = ComfyRegistryWheelDownloader(transport=httpx.MockTransport(fetch))
    destination = tmp_path / "staged"
    try:
        report = await downloader.download_and_stage_inputs(
            manifest,
            destination,
            session_factory=fresh_session,
            store=store,
            marker_environment=_environment(),
            supported_tags=("py3-none-any",),
        )
    finally:
        await downloader.close()
    assert (destination / local.filename).read_bytes() == verified.payload
    assert (destination / f"{local.filename}.metadata").read_bytes() == verified.core_metadata
    assert len(requests) == (2 if include_remote else 0)
    assert all("files.pythonhosted.org" in url and "example_package" in url for url in requests)
    if include_remote:
        assert (destination / remote.artifacts[0].filename).read_bytes() == remote_bytes
    assert len(sessions) == 2 and all(item is not session for item in sessions)
    assert sessions[0] is not sessions[1]
    encoded = (destination / "stage-manifest.json").read_bytes()
    payload = json.loads(encoded)
    assert payload["version"] == 2
    assert parse_wheel_input_manifest(payload["input_manifest"]) == manifest
    assert (
        payload["input_manifest_sha256"]
        == report.artifact_manifest_sha256
        == manifest.manifest_sha256
    )
    assert report.stage_manifest_sha256 == hashlib.sha256(encoded).hexdigest()
    assert report.total_bytes == len(verified.payload) + len(verified.core_metadata) + (
        len(remote_bytes) + len(remote_metadata) if include_remote else 0
    )
    assert not list(tmp_path.glob(".registry-wheels-staged-*"))


@pytest.mark.parametrize("reason", ["wheel-total", "metadata-total", "target"])
async def test_local_stage_refuses_incompatible_targets_and_combined_size_overflow(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    session, store = source_review_context
    local = _local(session, store)
    manifest = build_comfy_registry_wheel_input_manifest("f" * 64, _resolve([], {}), [local])
    tags = ("cp312-cp312-win_amd64",) if reason == "target" else ("py3-none-any",)
    if reason != "target":
        monkeypatch.setattr(
            downloads,
            "MAX_REGISTRY_WHEEL_STAGE_BYTES",
            local.size_bytes - 1 if reason == "wheel-total" else local.size_bytes,
        )
    downloader = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("A local input must not download bytes")
        )
    )
    destination = tmp_path / "staged"
    try:
        with pytest.raises(
            (ComfyRegistryWheelInputError, downloads.ComfyRegistryWheelDownloadError)
        ):
            await downloader.download_and_stage_inputs(
                manifest,
                destination,
                session_factory=lambda: Session(session.get_bind()),
                store=store,
                marker_environment=_environment(),
                supported_tags=tags,
            )
    finally:
        await downloader.close()
    assert not destination.exists()
    assert not list(tmp_path.glob(".registry-wheels-staged-*"))


async def test_mixed_remote_metadata_cannot_exceed_the_remaining_aggregate_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel, metadata = b"wheel", b"metadata"
    remote = _resolve(
        ["example-package==1.2.3"],
        {
            "example-package": _document(
                _file(
                    hashes={"sha256": hashlib.sha256(wheel).hexdigest()},
                    size=len(wheel),
                    **{"core-metadata": {"sha256": hashlib.sha256(metadata).hexdigest()}},
                )
            )
        },
    )
    manifest = build_comfy_registry_wheel_input_manifest("f" * 64, remote, [])
    monkeypatch.setattr(downloads, "MAX_REGISTRY_WHEEL_STAGE_BYTES", len(wheel))
    downloader = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=metadata if request.url.path.endswith(".metadata") else wheel
            )
        )
    )
    destination = tmp_path / "staged"
    try:
        with pytest.raises(downloads.ComfyRegistryWheelDownloadError) as caught:
            await downloader.download_and_stage_inputs(
                manifest,
                destination,
                session_factory=lambda: pytest.fail("No local review needed"),
                store=ArtifactStore(Settings(data_dir=tmp_path / "data", dev=True)),
                marker_environment=_environment(),
                supported_tags=("py3-none-any",),
            )
        assert caught.value.code == "download_too_large"
    finally:
        await downloader.close()
    assert not destination.exists()
    assert not list(tmp_path.glob(".registry-wheels-staged-*"))


@pytest.mark.parametrize("when", ["before", "after-copy", "cancel"])
async def test_review_changes_and_cancellation_refuse_publication_and_clean_staging(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    when: str,
) -> None:
    session, store = source_review_context
    local = _local(session, store)
    manifest = build_comfy_registry_wheel_input_manifest("f" * 64, _resolve([], {}), [local])
    cached = session.scalars(select(ComfyRegistrySourceArtifactReview)).one()
    cached_digest = cached.review_sha256

    def change_review() -> None:
        with Session(session.get_bind()) as other:
            other.execute(update(ComfyRegistrySourceArtifactReview).values(review_sha256="e" * 64))
            other.commit()
        assert cached.review_sha256 == cached_digest

    async def progress(_filename: str, downloaded: int, _total: int | None) -> None:
        if downloaded:
            if when == "cancel":
                raise asyncio.CancelledError
            change_review()

    if when == "before":
        change_review()
    downloader = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("A local input must not download bytes")
        )
    )
    destination = tmp_path / "staged"
    error = asyncio.CancelledError if when == "cancel" else ComfyRegistrySourceArtifactError
    try:
        with pytest.raises(error):
            await downloader.download_and_stage_inputs(
                manifest,
                destination,
                session_factory=lambda: Session(session.get_bind()),
                store=store,
                marker_environment=_environment(),
                supported_tags=("py3-none-any",),
                progress=progress,
            )
    finally:
        await downloader.close()
    assert not destination.exists()
    assert not list(tmp_path.glob(".registry-wheels-staged-*"))


async def test_changed_expected_review_identity_cannot_stage_the_same_wheel_bytes(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
) -> None:
    session, store = source_review_context
    local = _local(session, store)
    manifest = build_comfy_registry_wheel_input_manifest(
        "f" * 64, _resolve([], {}), [replace(local, review_sha256="e" * 64)]
    )
    downloader = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("A local input must not download bytes")
        )
    )
    destination = tmp_path / "staged"
    try:
        with pytest.raises(ComfyRegistryWheelInputError) as caught:
            await downloader.download_and_stage_inputs(
                manifest,
                destination,
                session_factory=lambda: Session(session.get_bind()),
                store=store,
                marker_environment=_environment(),
                supported_tags=("py3-none-any",),
            )
        assert caught.value.code == "reviewed_wheel_input_changed"
    finally:
        await downloader.close()
    assert not destination.exists()
    assert not list(tmp_path.glob(".registry-wheels-staged-*"))


async def test_local_verification_and_writes_leave_the_event_loop_thread(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store = source_review_context
    local = _local(session, store)
    manifest = build_comfy_registry_wheel_input_manifest("f" * 64, _resolve([], {}), [local])
    event_loop_thread = threading.get_ident()
    writes: list[int] = []
    reads: list[int] = []
    original_write = downloads._write_new_file

    def write(path: Path, content: bytes) -> None:
        writes.append(threading.get_ident())
        original_write(path, content)

    def fresh_session() -> Session:
        reads.append(threading.get_ident())
        return Session(session.get_bind())

    monkeypatch.setattr(downloads, "_write_new_file", write)
    downloader = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("A local input must not download bytes")
        )
    )
    try:
        await downloader.download_and_stage_inputs(
            manifest,
            tmp_path / "staged",
            session_factory=fresh_session,
            store=store,
            marker_environment=_environment(),
            supported_tags=("py3-none-any",),
        )
    finally:
        await downloader.close()
    assert len(reads) == 2 and len(writes) == 3
    assert all(identity != event_loop_thread for identity in reads + writes)


@pytest.mark.parametrize("during_manifest", [False, True])
async def test_repeated_cancellation_drains_local_writes_before_cleanup(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    during_manifest: bool,
) -> None:
    session, store = source_review_context
    local = _local(session, store)
    manifest = build_comfy_registry_wheel_input_manifest("f" * 64, _resolve([], {}), [local])
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    completed = threading.Event()
    original_write = downloads._write_new_file

    def write(path: Path, content: bytes) -> None:
        selected = path.name == ("stage-manifest.json" if during_manifest else local.filename)
        if selected:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10), "The blocked write must be released by the event loop"
        original_write(path, content)
        if selected:
            completed.set()

    monkeypatch.setattr(downloads, "_write_new_file", write)
    downloader = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("A local input must not download bytes")
        )
    )
    destination = tmp_path / "staged"
    task = asyncio.create_task(
        downloader.download_and_stage_inputs(
            manifest,
            destination,
            session_factory=lambda: Session(session.get_bind()),
            store=store,
            marker_environment=_environment(),
            supported_tags=("py3-none-any",),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "Cleanup must wait for the uncancellable file operation"
        assert not destination.exists()
        assert list(tmp_path.glob(".registry-wheels-staged-*"))
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed.is_set()
        assert not destination.exists()
        assert not list(tmp_path.glob(".registry-wheels-staged-*"))
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await downloader.close()
