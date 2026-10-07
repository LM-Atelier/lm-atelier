from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from sqlalchemy import select, text
from test_workflow_activations import _asset, _model

from local_lm import workflow_activation_files, workflow_activations
from local_lm.db import SessionLocal
from local_lm.filesystem_links import _windows_api
from local_lm.models import ModelAssetInstall, ModelComponentManifest, ModelInstall
from local_lm.workflow_activation_files import verify_workflow_files
from local_lm.workflow_activations import WorkflowActivationError


def _installs(app: FastAPI, tmp_path: Path) -> tuple[str, str]:
    with SessionLocal() as session:
        model = _model(session, tmp_path / "model", suffix="verified")
        asset = _asset(session, tmp_path / "asset", suffix="verified")
        session.commit()
        return model.id, asset.id


def test_verification_hashes_files_with_no_database_writer(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id, asset_id = _installs(app, tmp_path)
    original = workflow_activations._verify_file_digest
    verified: list[Path] = []

    def verify(path: Path, digest: str, message: str) -> None:
        with SessionLocal() as session:
            session.execute(text("UPDATE model_installs SET active=active WHERE 0"))
            session.commit()
        verified.append(path)
        original(path, digest, message)

    monkeypatch.setattr(workflow_activations, "_verify_file_digest", verify)
    proof = verify_workflow_files(SessionLocal, [model_id], [asset_id])
    assert len(verified) == 2

    def unexpected_hash(*_args: object) -> None:
        pytest.fail("The activation transaction must not hash files again")

    monkeypatch.setattr(workflow_activations, "_verify_file_digest", unexpected_hash)
    with SessionLocal() as session:
        session.execute(text("UPDATE model_installs SET active=active WHERE 0"))
        proof.require_current(session)
        assert proof.model_binding(session, model_id).model_install_id == model_id
        assert proof.asset_binding(session, asset_id).model_asset_install_id == asset_id


@pytest.mark.parametrize("kind", ["model", "asset"])
@pytest.mark.parametrize("change", ["content", "restored-time", "replacement", "row", "missing"])
def test_verification_refuses_files_or_records_changed_after_hashing(
    app: FastAPI, tmp_path: Path, kind: str, change: str
) -> None:
    model_id, asset_id = _installs(app, tmp_path)
    proof = verify_workflow_files(SessionLocal, [model_id], [asset_id])
    path = (
        tmp_path / "model/weights/model.safetensors"
        if kind == "model"
        else tmp_path / "asset/styles/style.safetensors"
    )
    if change == "content":
        path.write_bytes(b"changed file content")
    elif change == "restored-time":
        before = path.stat()
        path.write_bytes(b"x" * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "replacement":
        before = path.stat()
        replacement = path.with_suffix(".new")
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        replacement.replace(path)
    elif change == "missing":
        path.rename(path.with_suffix(".moved"))
    else:
        with SessionLocal() as session:
            if kind == "model":
                component = session.scalar(
                    select(ModelComponentManifest).where(
                        ModelComponentManifest.model_install_id == model_id
                    )
                )
                assert component is not None
                component.relative_path = "weights/another.safetensors"
            else:
                asset = session.get(ModelAssetInstall, asset_id)
                assert asset is not None
                asset.manifest_json = {**asset.manifest_json, "sha256": "a" * 64}
            session.commit()
    with SessionLocal() as session, pytest.raises(WorkflowActivationError):
        proof.require_current(session)


def test_verification_refreshes_an_already_loaded_component_record(
    app: FastAPI, tmp_path: Path
) -> None:
    model_id, asset_id = _installs(app, tmp_path)
    proof = verify_workflow_files(SessionLocal, [model_id], [asset_id])
    with SessionLocal() as session:
        install = session.get(ModelInstall, model_id)
        component = session.scalar(
            select(ModelComponentManifest).where(
                ModelComponentManifest.model_install_id == model_id
            )
        )
        assert install is not None and component is not None
        with SessionLocal() as writer:
            writer.execute(
                text("UPDATE model_component_manifests SET sha256=:digest WHERE id=:id"),
                {"digest": "a" * 64, "id": component.id},
            )
            writer.commit()
        with pytest.raises(WorkflowActivationError):
            proof.model_binding(session, model_id)


def test_verification_refuses_unverified_assets_and_a_replaced_seal(
    app: FastAPI, tmp_path: Path
) -> None:
    model_id, asset_id = _installs(app, tmp_path)
    proof = verify_workflow_files(SessionLocal, [model_id], [])
    with SessionLocal() as session:
        with pytest.raises(WorkflowActivationError):
            proof.asset_binding(session, asset_id)
        with pytest.raises(WorkflowActivationError):
            replace(proof, seal=object()).require_current(session)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file metadata query")
def test_windows_verification_refuses_when_change_time_cannot_be_read(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id, asset_id = _installs(app, tmp_path)
    proof = verify_workflow_files(SessionLocal, [model_id], [asset_id])
    native = _windows_api()
    unavailable = SimpleNamespace(
        ctypes=native.ctypes,
        FileBasicInformation=native.FileBasicInformation,
        IoStatusBlock=native.IoStatusBlock,
        ntdll=SimpleNamespace(NtQueryInformationFile=lambda *_args: 0xC0000001),
    )
    monkeypatch.setattr(workflow_activation_files, "_windows_api", lambda: unavailable)
    with SessionLocal() as session, pytest.raises(WorkflowActivationError) as raised:
        proof.require_current(session)
    assert raised.value.code == "dependency_content_drift"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file metadata query")
def test_windows_verification_refuses_an_unavailable_initial_change_time(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id, asset_id = _installs(app, tmp_path)
    native = _windows_api()
    queried: list[bool] = []

    def unavailable_query(*_args: object) -> int:
        queried.append(True)
        return 0xC0000001

    unavailable = SimpleNamespace(
        ctypes=native.ctypes,
        FileBasicInformation=native.FileBasicInformation,
        IoStatusBlock=native.IoStatusBlock,
        ntdll=SimpleNamespace(NtQueryInformationFile=unavailable_query),
    )
    monkeypatch.setattr(workflow_activation_files, "_windows_api", lambda: unavailable)
    with pytest.raises(WorkflowActivationError) as raised:
        verify_workflow_files(SessionLocal, [model_id], [asset_id])
    assert queried == [True]
    assert raised.value.code == "dependency_content_drift"
