"""A project archive encrypted with a passphrase, from export to import, and what it refuses."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import pytest
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from directory_security import create_owned_directory
from httpx2 import AsyncClient
from run_waits import PATIENCE_SECONDS

from local_lm import exports as exports_module
from local_lm import portable_archive_v1, project_archive_encryption
from local_lm.config import Settings
from local_lm.filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    create_private_directory,
    directory_private_to_current_user,
)
from local_lm.main import create_app
from local_lm.portable_archive_v1 import MAGIC, ArchiveKind, open_archive, write_archive
from local_lm.project_archive_encryption import STAGING_FOLDER, STAGING_PREFIX, encrypt_export

PASSPHRASE = "correct horse battery staple"


async def _project(client: AsyncClient) -> dict[str, object]:
    project: dict[str, object] = (
        await client.post("/api/projects", json={"name": "Garden notes"})
    ).json()
    chat = await client.post("/api/chats", json={"title": "Seed plan", "project_id": project["id"]})
    assert chat.status_code == 201, chat.text
    return project


async def _export(client: AsyncClient, project: dict[str, object], **body: object) -> bytes:
    exported = await client.post(f"/api/projects/{project['id']}/export", json=body or None)
    assert exported.status_code == 201, exported.text
    return (await client.get(exported.json()["url"])).content


async def _import(client: AsyncClient, archive: bytes, passphrase: str | None = None) -> Any:
    return await client.post(
        "/api/projects/import",
        files={"archive": ("garden.lm-atelier.encrypted", archive, "application/octet-stream")},
        data={} if passphrase is None else {"passphrase": passphrase},
    )


def _staged(settings: Settings) -> list[str]:
    folders = (settings.export_dir, settings.export_dir / STAGING_FOLDER)
    return sorted(
        path.name
        for folder in folders
        if folder.is_dir()
        for path in folder.iterdir()
        if path.name.startswith(STAGING_PREFIX)
    )


def _private(folder: Path) -> bool:
    with AnchoredDirectory(folder, read_security=True) as anchor:
        return directory_private_to_current_user(anchor)


async def _project_count(client: AsyncClient) -> int:
    return len((await client.get("/api/projects", params={"include_archived": True})).json())


async def test_an_encrypted_export_holds_only_ciphertext_and_opens_to_the_project(
    client: AsyncClient, settings: Settings
) -> None:
    archive = await _export(client, await _project(client), passphrase=PASSPHRASE)

    assert archive.startswith(MAGIC)
    assert b"Garden notes" not in archive and b"Seed plan" not in archive
    opened = io.BytesIO()
    open_archive(
        io.BytesIO(archive), opened, kind=ArchiveKind.PROJECT, passphrase=PASSPHRASE.encode()
    )
    with zipfile.ZipFile(opened) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    assert manifest["project"]["name"] == "Garden notes"
    assert _staged(settings) == []


async def test_an_encrypted_archive_imports_with_its_passphrase(
    client: AsyncClient, settings: Settings
) -> None:
    archive = await _export(client, await _project(client), passphrase=PASSPHRASE)

    imported = await _import(client, archive, PASSPHRASE)

    assert imported.status_code == 201, imported.text
    chats = await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    assert [chat["title"] for chat in chats.json()] == ["Seed plan"]
    assert _staged(settings) == []


@pytest.mark.parametrize(
    ("change", "passphrase", "status", "code"),
    [
        ("none", "a wrong passphrase", 422, "archive-passphrase-or-archive-invalid"),
        ("last byte", PASSPHRASE, 422, "archive-passphrase-or-archive-invalid"),
        ("cut short", PASSPHRASE, 422, "archive-passphrase-or-archive-invalid"),
        ("none", None, 422, "archive-passphrase-required"),
    ],
    ids=["wrong-passphrase", "damaged", "cut-short", "no-passphrase"],
)
async def test_an_encrypted_archive_that_cannot_be_opened_imports_nothing(
    client: AsyncClient,
    settings: Settings,
    change: str,
    passphrase: str | None,
    status: int,
    code: str,
) -> None:
    archive = bytearray(await _export(client, await _project(client), passphrase=PASSPHRASE))
    if change == "last byte":
        archive[-1] ^= 0x01
    elif change == "cut short":
        del archive[-20:]
    before = await _project_count(client)

    refused = await _import(client, bytes(archive), passphrase)

    assert (refused.status_code, refused.json()["code"]) == (status, code)
    assert PASSPHRASE not in refused.text
    assert await _project_count(client) == before
    assert _staged(settings) == []


async def test_a_plain_archive_still_imports_with_or_without_a_passphrase(
    client: AsyncClient,
) -> None:
    archive = await _export(client, await _project(client))
    assert archive.startswith(b"PK")

    for passphrase in (None, PASSPHRASE):
        imported = await _import(client, archive, passphrase)
        assert imported.status_code == 201, imported.text


async def test_a_passphrase_too_long_to_seal_exports_nothing(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = await _project(client)
    # 600 characters, so within the field's length, but 1200 bytes.
    too_long = "é" * 600

    def never(*args: object, **kwargs: object) -> object:
        raise AssertionError("the archive was built before the passphrase was refused")

    monkeypatch.setattr(exports_module, "tempfile", SimpleNamespace(NamedTemporaryFile=never))

    refused = await client.post(
        f"/api/projects/{project['id']}/export", json={"passphrase": too_long}
    )

    assert (refused.status_code, refused.json()["code"]) == (422, "archive-passphrase-invalid")
    assert too_long not in refused.text
    assert _staged(settings) == []


async def test_a_derivation_without_memory_is_not_called_a_wrong_passphrase(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    archive = await _export(client, await _project(client), passphrase=PASSPHRASE)

    class _NoMemory:
        def __init__(self, **kwargs: object) -> None:
            pass

        def derive(self, key_material: bytes) -> bytes:
            raise MemoryError("not enough memory")

    monkeypatch.setattr(portable_archive_v1, "Argon2id", _NoMemory)
    refused = await _import(client, archive, PASSPHRASE)

    assert (refused.status_code, refused.json()["code"]) == (503, "archive-key-derivation-failed")
    assert _staged(settings) == []


async def test_a_check_without_memory_is_not_called_an_unverified_export(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    project = await _project(client)
    real = Argon2id
    derivations: list[int] = []

    class _CheckWithoutMemory:
        def __init__(self, **kwargs: Any) -> None:
            self._real = real(**kwargs)

        def derive(self, key_material: bytes) -> bytes:
            derivations.append(len(derivations) + 1)
            if len(derivations) == 2:
                raise MemoryError("not enough memory")
            return bytes(self._real.derive(key_material))

    monkeypatch.setattr(portable_archive_v1, "Argon2id", _CheckWithoutMemory)
    refused = await client.post(
        f"/api/projects/{project['id']}/export", json={"passphrase": PASSPHRASE}
    )

    # The first derivation seals the archive; the second is the check.
    assert derivations == [1, 2]
    assert (refused.status_code, refused.json()["code"]) == (503, "archive-key-derivation-failed")
    assert _staged(settings) == []


async def test_an_encrypted_export_leaves_the_application_answering(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = await _project(client)
    started, release = threading.Event(), threading.Event()
    real = encrypt_export

    def held(plaintext: Path, directory: Path, passphrase: bytes) -> Path:
        started.set()
        release.wait(PATIENCE_SECONDS)
        return real(plaintext, directory, passphrase)

    monkeypatch.setattr(exports_module, "encrypt_export", held)
    export = asyncio.create_task(
        client.post(f"/api/projects/{project['id']}/export", json={"passphrase": PASSPHRASE})
    )
    try:
        assert await asyncio.to_thread(started.wait, PATIENCE_SECONDS)
        health = await asyncio.wait_for(client.get("/api/health"), timeout=PATIENCE_SECONDS)

        # Answered while the export was still held, not after it finished.
        assert health.status_code == 200 and not export.done()
    finally:
        release.set()
    assert (await export).status_code == 201


async def test_a_cancelled_import_leaves_no_decrypted_file_and_imports_nothing(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = await _export(client, await _project(client), passphrase=PASSPHRASE)
    staged, release, finished = threading.Event(), threading.Event(), threading.Event()
    real = project_archive_encryption.decrypt_import

    def held(source: Any, directory: Path, passphrase: bytes) -> Path:
        try:
            path = real(source, directory, passphrase)
            staged.set()
            release.wait(10)
            return path
        finally:
            finished.set()

    monkeypatch.setattr(project_archive_encryption, "decrypt_import", held)
    before = await _project_count(client)
    request = asyncio.create_task(_import(client, archive, PASSPHRASE))
    assert await asyncio.to_thread(staged.wait, 20)

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    release.set()
    assert await asyncio.to_thread(finished.wait, 10)
    for _ in range(100):
        if not _staged(settings):
            break
        await asyncio.sleep(0.02)

    assert _staged(settings) == []
    assert await _project_count(client) == before


async def test_an_encrypted_archive_sent_without_a_passphrase_says_where_to_enter_one(
    client: AsyncClient,
) -> None:
    archive = await _export(client, await _project(client), passphrase=PASSPHRASE)

    refused = await _import(client, archive)

    assert refused.json()["code"] == "archive-passphrase-required"
    assert "Data & backups" in refused.json()["detail"]


async def test_an_export_that_does_not_open_again_is_not_kept(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    project = await _project(client)
    real = write_archive

    def damaging(source: BinaryIO, destination: BinaryIO, **kwargs: Any) -> int:
        # A byte past the end, which no honest archive has.
        return real(source, destination, **kwargs) + destination.write(b"\x00")

    monkeypatch.setattr(project_archive_encryption, "write_archive", damaging)
    before = len(list(settings.artifact_dir.rglob("*")))

    refused = await client.post(
        f"/api/projects/{project['id']}/export", json={"passphrase": PASSPHRASE}
    )

    assert (refused.status_code, refused.json()["code"]) == (500, "project-export-unverified")
    assert len(list(settings.artifact_dir.rglob("*"))) == before
    assert _staged(settings) == []


async def test_the_plaintext_of_an_encrypted_export_is_staged_under_the_name_startup_removes(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = await _project(client)
    staged: list[str] = []

    folders: list[Path] = []

    def refusing(plaintext: Path, directory: Path, passphrase: bytes) -> Path:
        staged.append(plaintext.name)
        folders.append(plaintext.parent)
        raise project_archive_encryption.ExportUnverified

    monkeypatch.setattr(exports_module, "encrypt_export", refusing)
    await client.post(f"/api/projects/{project['id']}/export", json={"passphrase": PASSPHRASE})

    assert len(staged) == 1 and staged[0].startswith(STAGING_PREFIX)
    assert folders[0].name == STAGING_FOLDER and _private(folders[0])


async def test_staged_files_a_crash_left_behind_are_removed_at_startup(settings: Settings) -> None:
    staging = settings.export_dir / STAGING_FOLDER
    staging.mkdir(parents=True, exist_ok=True)
    left = [f"{STAGING_PREFIX}one.lm-atelier.zip", f"{STAGING_PREFIX}two.lm-atelier.encrypted"]
    kept = ["tmpkept.lm-atelier.zip", "notes.txt"]
    for name in left + kept:
        (staging / name).write_bytes(b"neutral staged bytes")
    (staging / f"{STAGING_PREFIX}folder").mkdir()

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        remaining = sorted(path.name for path in staging.iterdir())

    assert remaining == sorted([*kept, f"{STAGING_PREFIX}folder"])


async def test_an_export_folder_that_cannot_be_listed_does_not_stop_startup(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    (settings.export_dir / STAGING_FOLDER).mkdir(parents=True, exist_ok=True)
    left = settings.export_dir / STAGING_FOLDER / f"{STAGING_PREFIX}left.lm-atelier.zip"
    left.write_bytes(b"neutral staged bytes")

    def refusing(*args: object, **kwargs: object) -> tuple[object, ...]:
        raise AnchoredDirectoryError("refused")

    monkeypatch.setattr(project_archive_encryption, "list_entries", refusing)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        assert left.exists()


def test_a_staging_file_is_created_new_and_never_reused(tmp_path: Path) -> None:
    from local_lm.project_archive_encryption import staging_path

    first, second = staging_path(tmp_path, ".zip"), staging_path(tmp_path, ".zip")

    assert first != second
    assert first.name.startswith(STAGING_PREFIX) and first.read_bytes() == b""


async def test_a_staging_folder_others_can_open_refuses_export_and_import(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = await _project(client)
    archive = await _export(client, project, passphrase=PASSPHRASE)
    stored = len(list(settings.artifact_dir.rglob("*")))
    before = await _project_count(client)
    monkeypatch.setattr(
        project_archive_encryption, "directory_private_to_current_user", lambda anchor: False
    )

    exported = await client.post(
        f"/api/projects/{project['id']}/export", json={"passphrase": PASSPHRASE}
    )
    imported = await _import(client, archive, PASSPHRASE)

    for refused in (exported, imported):
        assert (refused.status_code, refused.json()["code"]) == (409, "archive-staging-not-private")
        assert PASSPHRASE not in refused.text
    assert len(list(settings.artifact_dir.rglob("*"))) == stored
    assert await _project_count(client) == before
    assert _staged(settings) == []


def test_the_staging_folder_is_made_private_even_where_new_files_would_not_be(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    if sys.platform == "win32":
        # The parent hands read access for everyone down to all it holds.
        create_owned_directory(exports, extra_aces="(A;OICI;FR;;;WD)")
    else:
        exports.mkdir(mode=0o755)
        os.chmod(exports, 0o755)

    folder = project_archive_encryption.private_staging(exports)

    assert folder == exports / STAGING_FOLDER and _private(folder)
    if sys.platform != "win32":
        assert folder.stat().st_mode & 0o777 == 0o700


def test_an_existing_staging_folder_others_can_open_is_refused_and_left_as_it_is(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "exports"
    exports.mkdir()
    if sys.platform == "win32":
        create_owned_directory(exports / STAGING_FOLDER, extra_aces="(A;OICI;FR;;;WD)")
    else:
        (exports / STAGING_FOLDER).mkdir()
        os.chmod(exports / STAGING_FOLDER, 0o755)

    with pytest.raises(project_archive_encryption.StagingNotPrivate):
        project_archive_encryption.private_staging(exports)

    assert not _private(exports / STAGING_FOLDER)
    if sys.platform != "win32":
        assert (exports / STAGING_FOLDER).stat().st_mode & 0o777 == 0o755


def test_a_private_directory_is_never_made_over_an_existing_entry(tmp_path: Path) -> None:
    (tmp_path / "taken").mkdir()

    with AnchoredDirectory(tmp_path) as parent, pytest.raises(AnchoredEntryExists):
        create_private_directory(parent, "taken")
