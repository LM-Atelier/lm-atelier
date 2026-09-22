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

from local_lm.capability_evidence import current_capability_evidence, record_capability_evidence
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import new_id
from local_lm.models import Base, ModelInstall


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
    does NOT descend into a symbolic link. So the case below only reproduces
    the descent on Windows; everywhere else it checks that the walk still finds
    the install's own files, including the nested one, which is worth having
    but is not the same claim.
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


async def test_measuring_an_imported_folder_stays_inside_it(
    app: FastAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identity covers what the install holds, not what it points at.

    Sharing one large file between installs by pointing at it is an ordinary
    thing to do on a full disk, and a folder that does it is not broken. What
    would be broken is calling the other folder's bytes this install's own: the
    digests taken here decide later whether an activation still holds, so a
    file outside the install could invalidate evidence it never backed, and a
    name like a link's would read as though the install contained it.
    """

    install, outside, own = _imported_directory(tmp_path)
    if not _link_dir(install / "shared", outside):
        pytest.skip("this host does not allow a directory link to be created")

    install_id = new_id("model")
    with SessionLocal() as session:
        session.add(
            ModelInstall(
                id=install_id,
                name="Imported by hand",
                role="chat",
                engine="llama.cpp",
                local_path=str(install),
                manifest_json={"imported": True},
                active=True,
            )
        )
        session.commit()

    downloads = app.state.services.downloads
    proved: dict[str, dict[str, str]] = {}

    async def capture(
        *, job_id: str, install_id: str, default_settings: Any, component_hashes: dict[str, str]
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

    # Both halves stated: the install's own files down to the nested one are
    # measured, and nothing beyond the link is, under any name.
    assert proved["hashes"] == own
    with SessionLocal() as session:
        record = session.get(ModelInstall, install_id)
        assert record is not None
        assert record.manifest_json["expected_sha256"] == own
        assert set(record.manifest_json["file_signatures"]) == set(own)


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
        *, job_id: str, install_id: str, default_settings: Any, component_hashes: dict[str, str]
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
        *, job_id: str, install_id: str, default_settings: Any, component_hashes: dict[str, str]
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
