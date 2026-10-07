from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from test_workflow_activations import _model
from test_workflow_package_import_endpoint import _ui_graph
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import (
    _approve,
    _state,
)
from test_workflow_source_completion import source_runtime as source_runtime
from test_workflow_source_completion import (
    test_source_completion_requires_its_exact_downloaded_declared_resource as install_source,
)

from local_lm import models, workflow_activations, workflow_source_completion
from local_lm.db import SessionLocal
from local_lm.workflow_activation_files import verify_workflow_files
from local_lm.workflow_completion_jobs import complete_workflow_job

if TYPE_CHECKING:
    from local_lm.workflow_activation_files import VerifiedWorkflowFiles


@pytest.mark.parametrize("model", [False, True])
async def test_source_file_verification_leaves_the_database_writer_available(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    model: bool,
    tmp_path: Path,
) -> None:
    finalizing = threading.Event()
    attempted: list[bool] = []
    blocked: list[int] = []
    original_finish = workflow_source_completion._finish
    original_verify = workflow_activations._verify_file_digest

    def finish(*args: Any, **kwargs: Any) -> str:
        finalizing.set()
        try:
            return original_finish(*args, **kwargs)
        finally:
            finalizing.clear()

    def verify(path: Path, expected: str, message: str) -> None:
        if finalizing.is_set():
            attempted.append(True)
            with SessionLocal() as session:
                session.execute(text("PRAGMA busy_timeout=50"))
                try:
                    session.add(
                        models.AppSetting(
                            key=f"file-verification-write-{len(attempted)}",
                            value_json=True,
                        )
                    )
                    session.commit()
                except OperationalError as exc:
                    assert isinstance(exc.orig, sqlite3.Error)
                    assert exc.orig.sqlite_errorcode == sqlite3.SQLITE_BUSY
                    blocked.append(len(attempted))
        original_verify(path, expected, message)

    monkeypatch.setattr(workflow_source_completion, "_finish", finish)
    monkeypatch.setattr(workflow_activations, "_verify_file_digest", verify)
    if model:
        with SessionLocal() as session:
            installed = _model(session, tmp_path / "model", suffix="source")
            session.commit()
            model_id = installed.id
        response = await client.post(
            "/api/workflows/packages/install-plans",
            json={
                "name": "Source with a required model",
                "operation": "text_to_image",
                "ui_graph": _ui_graph(),
                "dependencies": {
                    "version": 1,
                    "slots": [
                        {
                            "name": "base",
                            "resource_kind": "model_install",
                            "required": True,
                            "satisfaction": "all_of",
                            "requirements": [
                                {
                                    "key": "base",
                                    "constraints": {"role": "image", "engine": "comfyui"},
                                }
                            ],
                        }
                    ],
                },
                "selections": [],
            },
        )
        assert response.status_code == 201 and response.json()["can_accept"], response.text
        offer_id = await _approve(client, app, str(response.json()["id"]))
        assert _state(offer_id)[0] == "completed", _state(offer_id)
        with SessionLocal() as session:
            offer = session.get(models.WorkflowInstallOffer, offer_id)
            assert offer is not None
            bindings = session.scalars(
                select(models.WorkflowDependencyBinding).where(
                    models.WorkflowDependencyBinding.workflow_revision_id
                    == offer.workflow_revision_id,
                    models.WorkflowDependencyBinding.model_install_id == model_id,
                )
            ).all()
            assert len(bindings) == 1
    else:
        await install_source(client, app, monkeypatch, source_runtime, "none")
    assert attempted, "Finalization must verify the accepted resource's bytes"
    assert not blocked, "File verification held the database writer during source completion"


@pytest.mark.parametrize("change", ["bytes", "record"])
async def test_source_completion_rejects_drift_after_the_files_were_verified(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    change: str,
) -> None:
    original = verify_workflow_files
    changed: list[str] = []

    def verify(
        session_factory: Callable[[], Session], model_ids: Sequence[str], asset_ids: Sequence[str]
    ) -> VerifiedWorkflowFiles:
        proof = original(session_factory, model_ids, asset_ids)
        assert len(proof.assets) == 1
        expected, _files = proof.assets[0]
        if change == "bytes":
            binding = expected.binding
            path = binding.base_path / binding.runtime_reference
            path.write_bytes(path.read_bytes() + b"changed")
        else:
            with SessionLocal() as session:
                asset = session.get(
                    models.ModelAssetInstall, expected.binding.model_asset_install_id
                )
                assert asset is not None
                asset.manifest_json = {**asset.manifest_json, "sha256": "e" * 64}
                session.commit()
        changed.append(change)
        return proof

    monkeypatch.setattr(workflow_source_completion, "verify_workflow_files", verify)
    await install_source(client, app, monkeypatch, source_runtime, "after-verification")
    assert changed == [change]


@pytest.mark.parametrize("changed_bytes", [False, True])
async def test_cancelling_asset_verification_retains_the_worker_until_the_file_check_finishes(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
    changed_bytes: bool,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    verifying = threading.Event()
    acquired = asyncio.Event()
    completion: asyncio.Task[Any] | None = None
    original_complete = workflow_source_completion.complete_workflow_source
    original_finish = workflow_source_completion._finish
    original_digest = workflow_activations._verify_file_digest
    main_thread = threading.get_ident()

    async def complete(*args: Any, **kwargs: Any) -> str | None:
        nonlocal completion
        completion = asyncio.current_task()
        return await original_complete(*args, **kwargs)

    def finish(*args: Any, **kwargs: Any) -> str:
        verifying.set()
        try:
            return original_finish(*args, **kwargs)
        finally:
            verifying.clear()
            finished.set()

    def digest(path: Path, expected: str, message: str) -> None:
        if verifying.is_set():
            assert threading.get_ident() != main_thread
            entered.set()
            if not release.wait(30):
                raise AssertionError("Asset verification was not released")
            if changed_bytes:
                path.write_bytes(path.read_bytes() + b"changed")
        original_digest(path, expected, message)

    async def take_lease() -> None:
        async with app.state.services.scheduler.lease("primary"):
            acquired.set()

    async def waiting() -> bool:
        return entered.is_set()

    async def settled() -> bool:
        return finished.is_set()

    monkeypatch.setattr(workflow_source_completion, "complete_workflow_source", complete)
    monkeypatch.setattr(workflow_source_completion, "_finish", finish)
    monkeypatch.setattr(workflow_activations, "_verify_file_digest", digest)
    installation = asyncio.create_task(
        install_source(
            client,
            app,
            monkeypatch,
            source_runtime,
            "after-verification" if changed_bytes else "none",
        )
    )
    waiter: asyncio.Task[None] | None = None
    try:
        await wait_until(waiting, bool, what="source asset verification")
        assert completion is not None
        waiter = asyncio.create_task(take_lease())
        completion.cancel()
        await asyncio.sleep(0)
        completion.cancel()
        await asyncio.sleep(0)
        assert not completion.done(), "Cancellation abandoned asset verification"
        assert not acquired.is_set(), "Cancellation released the worker before verification"
        release.set()
        await installation
        await waiter
        assert finished.is_set() and acquired.is_set()
    finally:
        release.set()
        await asyncio.gather(
            installation, *([waiter] if waiter is not None else []), return_exceptions=True
        )
        if entered.is_set():
            await wait_until(settled, bool, what="asset verification cleanup")


async def test_source_completion_refuses_file_drift_after_final_activation_revalidation(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    source_runtime: dict[str, Any],
) -> None:
    original_verify = verify_workflow_files
    original_revalidate = workflow_activations.revalidate_workflow_activation
    original_complete = complete_workflow_job
    files: list[Path] = []
    revalidated: list[str] = []
    changed: list[Path] = []

    def verify(
        session_factory: Callable[[], Session], model_ids: Sequence[str], asset_ids: Sequence[str]
    ) -> VerifiedWorkflowFiles:
        proof = original_verify(session_factory, model_ids, asset_ids)
        assert len(proof.assets) == 1
        expected, _files = proof.assets[0]
        files.append(expected.binding.base_path / expected.binding.runtime_reference)
        return proof

    def revalidate(*args: Any, **kwargs: Any) -> Any:
        scope = original_revalidate(*args, **kwargs)
        revalidated.append(scope.activation_id)
        return scope

    def complete(*args: Any, **kwargs: Any) -> None:
        original_complete(*args, **kwargs)
        assert len(revalidated) == 1 and len(files) == 1
        path = files[0]
        path.write_bytes(path.read_bytes() + b"changed before commit")
        changed.append(path)

    monkeypatch.setattr(workflow_source_completion, "verify_workflow_files", verify)
    monkeypatch.setattr(workflow_source_completion, "revalidate_workflow_activation", revalidate)
    monkeypatch.setattr(workflow_source_completion, "complete_workflow_job", complete)
    await install_source(client, app, monkeypatch, source_runtime, "after-verification")
    assert len(changed) == 1
