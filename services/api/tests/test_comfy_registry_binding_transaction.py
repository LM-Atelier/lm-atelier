from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session
from test_comfy_registry_mixed_wheel_environments import _input_context
from test_comfy_registry_reviewed_lifecycle import _downloader, _inputs, _prepare
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)

from local_lm import comfy_registry_lifecycle as lifecycle
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_installs import (
    ComfyRegistryInstallError,
    ComfyRegistryVerifiedWheelBinding,
    verify_comfy_registry_wheel_binding,
)
from local_lm.comfy_registry_wheel_inputs_v1 import ComfyRegistryWheelInputError
from local_lm.models import Artifact, ComfyRegistryInstall, ComfyRegistrySourceArtifactReview


@pytest.mark.parametrize("renew", [False, True])
@pytest.mark.parametrize("revoke", [False, True])
async def test_final_binding_allows_another_writer_and_refuses_a_late_revocation(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renew: bool,
    revoke: bool,
) -> None:
    session, _store = source_review_context
    inputs = _inputs(source_review_context, tmp_path)
    downloader = _downloader(inputs, [])
    prior = await _prepare(source_review_context, inputs, downloader) if renew else None
    install = session.get(ComfyRegistryInstall, prior.install_id) if prior else None
    if install is not None:
        install.trusted = True
        install.review_json = {**install.review_json, "preserved": "value"}
        session.commit()
    cached_review = session.scalar(select(ComfyRegistrySourceArtifactReview))
    assert cached_review is not None
    engine = session.get_bind()
    event_loop_thread = threading.get_ident()
    started, release = threading.Event(), threading.Event()

    def verify(*args: Any, **kwargs: Any) -> ComfyRegistryVerifiedWheelBinding:
        assert threading.get_ident() != event_loop_thread
        binding = verify_comfy_registry_wheel_binding(*args, **kwargs)
        started.set()
        assert release.wait(15)
        return binding

    def write() -> None:
        with Session(engine) as writer:
            if revoke:
                writer.execute(delete(ComfyRegistrySourceArtifactReview))
            else:
                writer.execute(update(Artifact).values(favorite=True))
            writer.commit()

    monkeypatch.setattr(lifecycle, "verify_comfy_registry_wheel_binding", verify)
    task = asyncio.create_task(
        lifecycle.renew_comfy_registry_install_environment(
            session,
            install_id=prior.install_id,
            resolution=inputs.resolution,
            closure=inputs.closure,
            wheel_downloader=downloader,
            python_executable=Path(sys.executable),
            custom_node_root=inputs.nodes,
            state_root=inputs.state,
            media_worker_stopped=True,
            reviewed_inputs=_input_context(source_review_context),
        )
        if prior
        else _prepare(source_review_context, inputs, downloader)
    )
    try:
        try:
            assert await asyncio.to_thread(started.wait, 15)
            await asyncio.wait_for(asyncio.to_thread(write), timeout=3)
            assert not task.done()
            # This object still exists in the caller's identity map even when
            # the other connection deleted its row. The final query must see
            # the database, not mistake this cached object for current consent.
            assert cached_review.review_sha256
        finally:
            release.set()
        if revoke:
            with pytest.raises(ComfyRegistryInstallError) as error:
                await task
            assert error.value.code == "source_review_verification_failed"
            assert session.scalar(select(func.count()).select_from(ComfyRegistryInstall)) == int(
                renew
            )
            assert len(list(inputs.nodes.iterdir())) == int(renew)
            assert len(
                list((inputs.state / "registry-wheel-environments").glob("registry-wheels-*"))
            ) == int(renew)
            if prior and install:
                session.refresh(install)
                assert install.wheel_environment_sha256 == prior.wheel_environment_sha256
                assert install.trusted and install.review_json["preserved"] == "value"
        else:
            prepared = await task
            assert prepared.wheel_closure_sha256 == inputs.closure.closure_sha256
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await downloader.close()


async def test_cancelling_final_binding_drains_its_worker_before_removing_files(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(source_review_context, tmp_path)
    downloader = _downloader(inputs, [])
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def verify(*args: Any, **kwargs: Any) -> ComfyRegistryVerifiedWheelBinding:
        result = verify_comfy_registry_wheel_binding(*args, **kwargs)
        started.set()
        assert release.wait(15)
        try:
            assert list(inputs.nodes.iterdir())
            assert list((inputs.state / "registry-wheel-environments").glob("registry-wheels-*"))
            return result
        finally:
            finished.set()

    monkeypatch.setattr(lifecycle, "verify_comfy_registry_wheel_binding", verify)
    task = asyncio.create_task(_prepare(source_review_context, inputs, downloader))
    try:
        assert await asyncio.to_thread(started.wait, 15)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await downloader.close()
    assert finished.is_set()
    assert not list(inputs.nodes.iterdir())
    assert not list((inputs.state / "registry-wheel-environments").glob("registry-wheels-*"))
    assert (
        source_review_context[0].scalar(select(func.count()).select_from(ComfyRegistryInstall)) == 0
    )


async def test_a_review_can_be_revoked_while_its_bytes_are_read_from_a_consistent_snapshot(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(source_review_context, tmp_path)
    context = _input_context(source_review_context)
    engine = source_review_context[0].get_bind()
    with context.session_factory() as configure:
        assert (
            configure.connection().exec_driver_sql("PRAGMA journal_mode=WAL").scalar_one() == "wal"
        )
    started, release = threading.Event(), threading.Event()
    original = ArtifactStore.verified_bytes

    def read(self: ArtifactStore, artifact: Artifact, **kwargs: Any) -> bytes:
        result = original(self, artifact, **kwargs)
        started.set()
        assert release.wait(15)
        return result

    def revoke() -> None:
        with Session(engine) as writer:
            writer.execute(delete(ComfyRegistrySourceArtifactReview))
            writer.commit()

    monkeypatch.setattr(ArtifactStore, "verified_bytes", read)
    operation = asyncio.create_task(
        asyncio.to_thread(context.verified_authority, inputs.closure.manifest)
    )
    try:
        assert await asyncio.to_thread(started.wait, 15)
        await asyncio.wait_for(asyncio.to_thread(revoke), timeout=3)
    finally:
        release.set()
    authority = await operation
    assert authority.rows
    with Session(engine) as current:
        with pytest.raises(ComfyRegistryWheelInputError) as error:
            authority.require_current(current)
        assert error.value.code == "reviewed_wheel_input_changed"
