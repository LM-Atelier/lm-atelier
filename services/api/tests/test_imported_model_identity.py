"""What an activation knows about the files it proved, when nobody declared them."""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm import downloads as downloads_module
from local_lm.capability_evidence import current_capability_evidence, record_capability_evidence
from local_lm.config import Settings
from local_lm.db import Base, SessionLocal
from local_lm.domain import new_id
from local_lm.filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    WalkedEntry,
    open_entry,
)
from local_lm.models import Job, ModelInstall
from local_lm.scheduler import JobClaim


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def _imported(tmp_path: Path, body: bytes = b"a model, imported by hand") -> tuple[Path, str]:
    model = tmp_path / "imported.gguf"
    model.write_bytes(body)
    return model, hashlib.sha256(body).hexdigest()


def _link_dir(link: Path, target: Path) -> bool:
    """Point `link` at `target`, or report that this host will not allow it.

    A junction on Windows and a symbolic link elsewhere, and the two are not
    equally interesting. Measured on CPython 3.12.10: a recursive glob descends
    into a junction and reports the far side's files as the tree's own, and
    does NOT descend into a symbolic link. So a link made here reproduces that
    descent only on Windows; everywhere else the measurement refuses the link
    all the same, which is worth having but is not the same claim.
    """

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def _imported_directory(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """An install folder, a folder outside it, and the digests the install owns."""

    install = tmp_path / "imported-model"
    (install / "nested").mkdir(parents=True)
    own = {
        "weights.bin": b"the bytes this install is made of",
        "nested/part.bin": b"a second file, one level down",
    }
    for name, body in own.items():
        (install / name).write_bytes(body)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "foreign.bin").write_bytes(b"bytes that belong to something else")
    return install, outside, {name: hashlib.sha256(body).hexdigest() for name, body in own.items()}


async def _activate_imported(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch, local_path: Path
) -> tuple[str, str, dict[str, dict[str, str]]]:
    """Activate a hand-placed install, recording what the probe was asked to prove."""

    install_id = new_id("model")
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id=install_id,
                name="Imported by hand",
                role="chat",
                engine="llama.cpp",
                local_path=str(local_path),
                manifest_json={"imported": True},
                active=True,
            )
        )
        session.commit()

    downloads = app.state.services.downloads
    proved: dict[str, dict[str, str]] = {}

    async def capture(
        *,
        job_id: str,
        install_id: str,
        default_settings: Any,
        component_hashes: dict[str, str],
        primary_lease_held: bool = False,
        claim: JobClaim | None = None,
    ) -> str:
        proved["hashes"] = component_hashes
        return "ok"

    monkeypatch.setattr(downloads, "start_activation", lambda job_id: None)
    monkeypatch.setattr(downloads, "_activate_chat_install", capture)
    with SessionLocal() as session:
        record = session.get(ModelInstall, install_id)
        assert record is not None
        job = downloads.reactivate(session, record)

    await downloads._reactivate(job.id)
    return install_id, job.id, proved


def _assert_refused_unmeasured(install_id: str, job_id: str, items: str) -> None:
    """The activation failed, said why, and wrote down no identity at all."""

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.status == "failed"
        assert job.error is not None
        assert job.error.startswith(f"{items} in the model folder could not be measured")
        record = session.get(ModelInstall, install_id)
        assert record is not None
        assert "expected_sha256" not in record.manifest_json
        assert "file_signatures" not in record.manifest_json


async def test_a_folder_that_leads_outside_itself_is_refused_not_measured_in_part(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity covers the whole install or nothing, and never reads past it.

    Sharing one large file between installs by pointing at it is an ordinary
    thing to do on a full disk, but the runtime loads through the pointer as
    readily as through a file. Measuring only the install's own files would let
    the shared bytes change under evidence that still holds; measuring through
    the pointer would call another folder's bytes this install's own. So the
    activation stops and says why, having read nothing beyond the link.
    """

    install, outside, own = _imported_directory(tmp_path)
    if not _link_dir(install / "shared", outside):
        pytest.skip("this host does not allow a directory link to be created")
    read: list[str] = []
    measure = downloads_module._measured_entry

    def recording(walked: WalkedEntry) -> tuple[str, list[int]] | None:
        read.append("/".join(walked.parts))
        return measure(walked)

    monkeypatch.setattr(downloads_module, "_measured_entry", recording)

    install_id, job_id, proved = await _activate_imported(app, monkeypatch, install)

    assert proved == {}
    assert sorted(read) == sorted(own)
    _assert_refused_unmeasured(install_id, job_id, "1 item")


@pytest.mark.skipif(os.name != "nt", reason="only on Windows does the search enter a junction")
async def test_a_model_the_runtime_finds_through_a_junction_is_never_proved(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file the runtime would load is measured, or the activation stops.

    A model the manifest does not name is found by searching the folder, and
    on Windows that search enters a junction and picks up whatever is on the
    far side. Proving such an install with only its own files measured would
    record evidence that names none of the bytes the runtime then runs.
    """

    from local_lm.processes import ProcessSupervisor

    root = tmp_path / "imported"
    root.mkdir()
    (root / "notes.txt").write_bytes(b"ordinary local metadata")
    outside = tmp_path / "shared-weights"
    outside.mkdir()
    weights = outside / "model.gguf"
    weights.write_bytes(b"neutral model fixture")
    if not _link_dir(root / "shared", outside):
        pytest.skip("this host does not allow a directory junction")
    selected = ProcessSupervisor._gguf_paths(root, {"imported": True})
    assert [path.resolve() for path in selected] == [weights.resolve()]

    install_id, job_id, proved = await _activate_imported(app, monkeypatch, root)

    assert proved == {}
    _assert_refused_unmeasured(install_id, job_id, "1 item")


async def test_a_file_that_will_not_open_stops_the_measurement(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file left out because it refused to open is a file left out.

    It was listed as a file, so it is not a link the walk declined to follow:
    it is still there, and still part of what the install holds.
    """

    install, _outside, _own = _imported_directory(tmp_path)
    # The same function the downloads module opens entries with.
    opening = open_entry

    def refusing(anchor: AnchoredDirectory, name: str) -> int | None:
        if name == "part.bin":
            raise AnchoredDirectoryError("refused")
        return opening(anchor, name)

    monkeypatch.setattr(downloads_module, "open_entry", refusing)

    install_id, job_id, proved = await _activate_imported(app, monkeypatch, install)

    assert proved == {}
    _assert_refused_unmeasured(install_id, job_id, "1 item")


async def test_activating_an_imported_model_measures_the_files_it_proves(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download declares its digests; an import declares nothing, so they are taken."""

    model, digest = _imported(tmp_path)
    install_id = new_id("model")
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id=install_id,
                name="Imported by hand",
                role="chat",
                engine="llama.cpp",
                local_path=str(model),
                manifest_json={"imported": True},
                active=True,
            )
        )
        session.commit()

    downloads = app.state.services.downloads
    proved: dict[str, dict[str, str]] = {}

    async def capture(
        *,
        job_id: str,
        install_id: str,
        default_settings: Any,
        component_hashes: dict[str, str],
        primary_lease_held: bool = False,
        claim: JobClaim | None = None,
    ) -> str:
        proved["hashes"] = component_hashes
        return "ok"

    monkeypatch.setattr(downloads, "start_activation", lambda job_id: None)
    monkeypatch.setattr(downloads, "_activate_chat_install", capture)
    with SessionLocal() as session:
        install = session.get(ModelInstall, install_id)
        assert install is not None
        job = downloads.reactivate(session, install)

    await downloads._reactivate(job.id)

    assert proved["hashes"] == {"imported.gguf": digest}
    with SessionLocal() as session:
        install = session.get(ModelInstall, install_id)
        assert install is not None
        assert install.manifest_json["expected_sha256"] == {"imported.gguf": digest}
        assert install.manifest_json["file_signatures"]["imported.gguf"][0] == model.stat().st_size


def test_evidence_for_a_measured_install_does_not_survive_its_files_changing(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    """Evidence names exact bytes, so bytes that moved on leave nothing standing."""

    model, digest = _imported(tmp_path)
    install = ModelInstall(
        id=new_id("model"),
        name="Imported by hand",
        role="chat",
        engine="llama.cpp",
        local_path=str(model),
        manifest_json={
            "imported": True,
            "expected_sha256": {"imported.gguf": digest},
            "file_signatures": {"imported.gguf": [model.stat().st_size, model.stat().st_mtime_ns]},
        },
        active=True,
    )
    session.add(install)
    session.commit()
    record_capability_evidence(
        session,
        install,
        settings,
        None,
        component_hashes={"imported.gguf": digest},
        runtime_build="test",
        workflow_contract_version=None,
        details={"modalities": ["text"]},
    )
    session.commit()

    assert current_capability_evidence(session, install, settings, None) is not None

    model.write_bytes(b"different bytes entirely, same name")

    assert current_capability_evidence(session, install, settings, None) is None


def test_evidence_for_a_declared_install_is_unaffected(
    session: Session, settings: Settings, tmp_path: Path
) -> None:
    """A download's identity comes from its plan, and nothing here second-guesses it."""

    model, digest = _imported(tmp_path)
    install = ModelInstall(
        id=new_id("model"),
        name="Downloaded",
        role="chat",
        engine="llama.cpp",
        local_path=str(model),
        manifest_json={"remote_id": "synthetic/chat", "expected_sha256": {"imported.gguf": digest}},
        active=True,
    )
    session.add(install)
    session.commit()
    record_capability_evidence(
        session,
        install,
        settings,
        None,
        component_hashes={"imported.gguf": digest},
        runtime_build="test",
        workflow_contract_version=None,
        details={"modalities": ["text"]},
    )
    session.commit()

    model.write_bytes(b"different bytes entirely, same name")

    assert current_capability_evidence(session, install, settings, None) is not None


async def test_activating_a_replaced_imported_model_measures_the_new_bytes(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first measurement describes the first files, and files can be replaced."""

    model, first = _imported(tmp_path)
    install_id = new_id("model")
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id=install_id,
                name="Imported by hand",
                role="chat",
                engine="llama.cpp",
                local_path=str(model),
                manifest_json={"imported": True},
                active=True,
            )
        )
        session.commit()

    downloads = app.state.services.downloads
    proved: list[dict[str, str]] = []

    async def capture(
        *,
        job_id: str,
        install_id: str,
        default_settings: Any,
        component_hashes: dict[str, str],
        primary_lease_held: bool = False,
        claim: JobClaim | None = None,
    ) -> str:
        proved.append(component_hashes)
        return "ok"

    monkeypatch.setattr(downloads, "start_activation", lambda job_id: None)
    monkeypatch.setattr(downloads, "_activate_chat_install", capture)

    async def activate() -> None:
        with SessionLocal() as session:
            install = session.get(ModelInstall, install_id)
            assert install is not None
            job = downloads.reactivate(session, install)
        await downloads._reactivate(job.id)

    await activate()
    replacement = b"the same name over entirely different bytes"
    model.write_bytes(replacement)
    second = hashlib.sha256(replacement).hexdigest()
    await activate()

    assert [entry["imported.gguf"] for entry in proved] == [first, second]
    with SessionLocal() as session:
        install = session.get(ModelInstall, install_id)
        assert install is not None
        assert install.manifest_json["expected_sha256"] == {"imported.gguf": second}
