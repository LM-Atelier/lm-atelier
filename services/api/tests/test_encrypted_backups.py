"""An encrypted backup: made with a passphrase, checked before it is offered, and checked later."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import io
import json
import sqlite3
import struct
import threading
import zipfile
from collections.abc import AsyncIterator, Awaitable
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import pytest
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from httpx2 import AsyncClient
from PIL import Image

from local_lm import backup_archives, portable_archive_v1, project_archive_encryption
from local_lm.artifacts import ArtifactStore
from local_lm.backups import BackupManager
from local_lm.config import Settings
from local_lm.filesystem_links import AnchoredDirectory, directory_private_to_current_user
from local_lm.portable_archive_v1 import (
    MAGIC,
    MIN_MEMORY_KIB,
    ArchiveKind,
    KeyDerivation,
    open_archive,
    read_header,
    write_archive,
)
from local_lm.project_archive_encryption import STAGING_FOLDER, STAGING_PREFIX, staging_path

PASSPHRASE = "correct horse battery staple"
# Archives sealed by hand here open the same way with a cheaper key; the
# derivation an archive used is recorded in it.
CHEAP = KeyDerivation(iterations=1, lanes=1, memory_kib=MIN_MEMORY_KIB)


def _staged(settings: Settings) -> list[str]:
    folder = settings.export_dir / STAGING_FOLDER
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.iterdir())


def _private(folder: Path) -> bool:
    with AnchoredDirectory(folder, read_security=True) as anchor:
        return directory_private_to_current_user(anchor)


def _stored_files(settings: Settings) -> list[Path]:
    return sorted(path for path in settings.artifact_dir.rglob("*") if path.is_file())


async def _chat(client: AsyncClient, title: str = "Seed plan") -> None:
    created = await client.post("/api/chats", json={"title": title})
    assert created.status_code == 201, created.text


async def _picture(client: AsyncClient) -> str:
    content = io.BytesIO()
    Image.new("RGB", (4, 4), (40, 120, 200)).save(content, format="PNG")
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("square.png", content.getvalue(), "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    return str(uploaded.json()["id"])


async def _create(client: AsyncClient, **body: object) -> Any:
    return await client.post("/api/backups/encrypted", json={"passphrase": PASSPHRASE, **body})


async def _backup(client: AsyncClient, **body: object) -> bytes:
    created = await _create(client, **body)
    assert created.status_code == 201, created.text
    return (await client.get(created.json()["url"])).content


def _passphrase_header(passphrase: str) -> dict[str, str]:
    return {"x-archive-passphrase": base64.b64encode(passphrase.encode("utf-8")).decode("ascii")}


async def _check(client: AsyncClient, content: Any, passphrase: str | None = PASSPHRASE) -> Any:
    headers = {"content-type": "application/octet-stream"}
    if passphrase is not None:
        headers |= _passphrase_header(passphrase)
    return await client.post("/api/backups/encrypted/check", content=content, headers=headers)


def _opened(archive: bytes) -> bytes:
    opened = io.BytesIO()
    open_archive(
        io.BytesIO(archive), opened, kind=ArchiveKind.BACKUP, passphrase=PASSPHRASE.encode()
    )
    return opened.getvalue()


def _framed(stream: bytes) -> tuple[dict[str, Any], bytes, bytes]:
    """Take an opened stream apart by the documented layout, not by the code that wrote it."""

    assert stream[:8] == b"LMABKUP\x00"
    (length,) = struct.unpack(">I", stream[8:12])
    header = json.loads(stream[12 : 12 + length].decode("utf-8"))
    rest = stream[12 + length :]
    database_size = header["database"]["size_bytes"]
    database, media = rest[:database_size], rest[database_size:]
    assert hashlib.sha256(database).hexdigest() == header["database"]["sha256"]
    if header["media"] is None:
        assert media == b""
    else:
        assert len(media) == header["media"]["size_bytes"]
        assert hashlib.sha256(media).hexdigest() == header["media"]["sha256"]
    return header, database, media


def _titles(database: bytes, tmp_path: Path) -> set[str]:
    copy = tmp_path / "opened.sqlite3"
    copy.write_bytes(database)
    with closing(sqlite3.connect(copy)) as connection:
        return {row[0] for row in connection.execute("SELECT title FROM chats")}


def _live_copy(settings: Settings, tmp_path: Path) -> tuple[bytes, str]:
    copy = tmp_path / "live-copy.sqlite3"
    with (
        closing(sqlite3.connect(settings.state_dir / "local-lm.sqlite3")) as live,
        closing(sqlite3.connect(copy)) as target,
    ):
        live.backup(target)
    with closing(sqlite3.connect(copy)) as connection:
        (revision,) = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    return copy.read_bytes(), revision


def _payload(
    copy: bytes, revision: str, *, media: bytes | None = None, **overrides: object
) -> bytes:
    """A backup stream built from the documented layout; ``overrides`` replace header keys."""

    header: dict[str, object] = {
        "format": "lm-atelier-backup",
        "version": 1,
        "created_at": "2026-10-04T12:00:00Z",
        "app_version": "0.0.0",
        "schema_revision": revision,
        "database": {"size_bytes": len(copy), "sha256": hashlib.sha256(copy).hexdigest()},
        "media": None
        if media is None
        else {"size_bytes": len(media), "sha256": hashlib.sha256(media).hexdigest()},
    }
    header.update(overrides)
    encoded = json.dumps(header).encode("utf-8")
    return b"LMABKUP\x00" + struct.pack(">I", len(encoded)) + encoded + copy + (media or b"")


def _sealed(payload: bytes, kind: ArchiveKind = ArchiveKind.BACKUP) -> bytes:
    sealed = io.BytesIO()
    write_archive(
        io.BytesIO(payload), sealed, kind=kind, passphrase=PASSPHRASE.encode(), derivation=CHEAP
    )
    return sealed.getvalue()


async def test_an_encrypted_backup_holds_only_ciphertext_and_opens_to_the_data(
    client: AsyncClient, settings: Settings, tmp_path: Path
) -> None:
    await _chat(client)

    created = await _create(client)
    assert created.status_code == 201, created.text
    assert created.json()["original_name"].endswith(".lm-atelier.encrypted")
    assert PASSPHRASE not in created.json()["url"]
    archive = (await client.get(created.json()["url"])).content

    assert archive.startswith(MAGIC)
    assert read_header(io.BytesIO(archive)).kind is ArchiveKind.BACKUP
    assert b"Seed plan" not in archive
    header, database, media = _framed(_opened(archive))
    assert (header["format"], header["version"], header["media"]) == ("lm-atelier-backup", 1, None)
    assert media == b""
    assert "Seed plan" in _titles(database, tmp_path)
    assert _staged(settings) == []


async def test_an_encrypted_backup_with_media_carries_the_pictures_its_data_refers_to(
    client: AsyncClient, settings: Settings
) -> None:
    await _picture(client)

    archive = await _backup(client, include_media=True)

    header, _database, media = _framed(_opened(archive))
    assert header["media"] is not None
    with zipfile.ZipFile(io.BytesIO(media)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    with closing(sqlite3.connect(settings.state_dir / "local-lm.sqlite3")) as live:
        backed_up = set(
            live.execute(
                "SELECT sha256, size_bytes, relative_path FROM artifacts "
                "WHERE kind IN ('image', 'video', 'thumbnail', 'input')"
            ).fetchall()
        )
    assert backed_up and {entry["sha256"] for entry in manifest["artifacts"]} == {
        row[0] for row in backed_up
    }

    checked = await _check(client, archive)

    assert checked.status_code == 200, checked.text
    assert checked.headers["cache-control"] == "no-store"
    report = checked.json()
    assert report["media_included"] is True
    assert report["artifact_count"] == len(backed_up)
    assert report["media_size_bytes"] == header["media"]["size_bytes"]
    assert report["schema_revision"] == header["schema_revision"]
    assert _staged(settings) == []


async def test_a_backup_built_from_the_documented_layout_checks_as_one(
    client: AsyncClient, settings: Settings, tmp_path: Path
) -> None:
    database, revision = _live_copy(settings, tmp_path)

    checked = await _check(client, _sealed(_payload(database, revision)))

    assert checked.status_code == 200, checked.text
    assert checked.json() | {"database_size_bytes": 0} == {
        "created_at": "2026-10-04T12:00:00Z",
        "app_version": "0.0.0",
        "schema_revision": revision,
        "database_size_bytes": 0,
        "media_included": False,
        "media_size_bytes": None,
        "artifact_count": 0,
    }
    assert checked.json()["database_size_bytes"] == len(database)
    assert _staged(settings) == []


@pytest.mark.parametrize(
    "damage",
    [
        "trailing bytes",
        "a short database",
        "a digest that does not match",
        "an unknown version",
        "a header longer than the format allows",
        "the wrong magic",
        "a revision the database does not have",
        "media the database does not refer to",
        "a creation time that is not one",
        "a creation time not written the documented way",
    ],
)
async def test_an_opened_stream_that_is_not_a_whole_backup_is_refused_and_leaves_nothing(
    client: AsyncClient, settings: Settings, tmp_path: Path, damage: str
) -> None:
    database, revision = _live_copy(settings, tmp_path)
    payload = {
        "trailing bytes": lambda: _payload(database, revision) + b"\x00",
        "a short database": lambda: _payload(database, revision)[:-1],
        "a digest that does not match": lambda: _payload(
            database,
            revision,
            database={"size_bytes": len(database), "sha256": "0" * 64},
        ),
        "an unknown version": lambda: _payload(database, revision, version=2),
        "a header longer than the format allows": lambda: (
            b"LMABKUP\x00" + struct.pack(">I", 64 * 1024 + 1) + b" " * (64 * 1024 + 1) + database
        ),
        "the wrong magic": lambda: b"LMABKUP\x01" + _payload(database, revision)[8:],
        "a revision the database does not have": lambda: _payload(database, "000000000000"),
        "media the database does not refer to": lambda: _payload(
            database, revision, media=b"not a zip"
        ),
        "a creation time that is not one": lambda: _payload(
            database, revision, created_at="yesterday"
        ),
        # A real moment, but not in the one shape a backup writes and reports.
        "a creation time not written the documented way": lambda: _payload(
            database, revision, created_at="2026-1-1T1:1:1Z"
        ),
    }[damage]()

    refused = await _check(client, _sealed(payload))

    assert (refused.status_code, refused.json()["code"]) == (422, "backup-invalid")
    assert _staged(settings) == []


async def test_a_wrong_passphrase_and_a_damaged_file_are_one_refusal(
    client: AsyncClient, settings: Settings
) -> None:
    archive = await _backup(client)
    flipped = bytearray(archive)
    flipped[-40] ^= 1

    wrong = await _check(client, archive, passphrase="not the passphrase")
    damaged = await _check(client, bytes(flipped))

    for refused in (wrong, damaged):
        assert (refused.status_code, refused.json()["code"]) == (
            422,
            "archive-passphrase-or-archive-invalid",
        )
    assert _staged(settings) == []


async def test_a_check_without_a_passphrase_asks_for_one_before_reading_the_file(
    client: AsyncClient, settings: Settings
) -> None:
    refused = await _check(client, b"anything", passphrase=None)

    assert (refused.status_code, refused.json()["code"]) == (422, "archive-passphrase-required")
    assert _staged(settings) == []


async def test_a_project_archive_is_not_taken_for_a_backup(
    client: AsyncClient, settings: Settings
) -> None:
    project_archive = _sealed(b"a project archive", kind=ArchiveKind.PROJECT)

    refused = await _check(client, project_archive)

    assert (refused.status_code, refused.json()["code"]) == (422, "archive-kind-mismatch")
    assert refused.json()["detail"] == "This file is not an encrypted LM Atelier backup."
    assert _staged(settings) == []


async def test_a_file_larger_than_a_backup_can_be_is_refused_by_its_length_or_as_it_arrives(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backup_archives, "max_encrypted_bytes", lambda _settings: 1024)

    declared = await _check(client, b"\x00" * 1025)

    async def undeclared() -> AsyncIterator[bytes]:
        for _ in range(4):
            yield b"\x00" * 512

    streamed = await _check(client, undeclared())

    for refused in (declared, streamed):
        assert (refused.status_code, refused.json()["code"]) == (413, "backup-file-too-large")
    assert _staged(settings) == []


async def test_a_file_of_unknown_length_is_not_read_onto_a_disk_without_room(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[int] = []

    async def undeclared() -> AsyncIterator[bytes]:
        for _ in range(4):
            sent.append(1)
            yield b"\x00" * 512

    monkeypatch.setattr(
        backup_archives,
        "shutil",
        SimpleNamespace(disk_usage=lambda folder: SimpleNamespace(free=0)),
    )

    refused = await _check(client, undeclared())

    assert (refused.status_code, refused.json()["code"]) == (507, "backup-storage-insufficient")
    assert sent == []
    assert _staged(settings) == []


async def test_a_file_of_unknown_length_stops_while_what_arrived_could_still_be_opened(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk = 1024 * 1024
    sent: list[int] = []

    async def undeclared() -> AsyncIterator[bytes]:
        for _ in range(8):
            sent.append(chunk)
            yield b"\x00" * chunk

    # A disk with room for three of the eight parts, which fills as they arrive.
    monkeypatch.setattr(backup_archives, "_SPARE_DISK_BYTES", 0)
    monkeypatch.setattr(
        backup_archives,
        "shutil",
        SimpleNamespace(disk_usage=lambda folder: SimpleNamespace(free=3 * chunk - sum(sent))),
    )

    refused = await _check(client, undeclared())

    assert (refused.status_code, refused.json()["code"]) == (507, "backup-storage-insufficient")
    # Stopped once a part could no longer be written with as much room again
    # left for opening it, and long before the whole stream was read.
    assert sum(sent) <= 3 * chunk
    assert _staged(settings) == []


async def test_a_check_cancelled_mid_write_keeps_the_upload_until_the_write_ends(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = await _backup(client)
    writing, release = threading.Event(), threading.Event()
    real = backup_archives._Receiving._write_with_room

    def held(self: Any, data: bytes, received: int) -> None:
        writing.set()
        release.wait(20)
        real(self, data, received)

    monkeypatch.setattr(backup_archives._Receiving, "_write_with_room", held)

    async def slowly() -> AsyncIterator[bytes]:
        yield b"\x00" * (1024 * 1024)
        yield b"\x00"

    request = asyncio.create_task(_check(client, slowly()))
    assert await asyncio.to_thread(writing.wait, 20)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    # The write is still running, so the upload and the lock are still held.
    assert backup_archives._busy.locked()
    assert _staged(settings)
    release.set()
    for _ in range(250):
        if not backup_archives._busy.locked() and not _staged(settings):
            break
        await asyncio.sleep(0.02)

    assert not backup_archives._busy.locked()
    assert _staged(settings) == []
    monkeypatch.setattr(backup_archives._Receiving, "_write_with_room", real)
    assert (await _check(client, archive)).status_code == 200


async def test_a_backup_that_does_not_open_again_is_not_kept(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = write_archive

    def damaging(source: BinaryIO, destination: BinaryIO, **kwargs: Any) -> int:
        # A byte past the end, which no honest archive has.
        return real(source, destination, **kwargs) + destination.write(b"\x00")

    monkeypatch.setattr(backup_archives, "write_archive", damaging)
    before = _stored_files(settings)

    refused = await _create(client)

    assert (refused.status_code, refused.json()["code"]) == (500, "backup-export-unverified")
    assert _stored_files(settings) == before
    assert _staged(settings) == []


async def test_the_plaintext_copy_is_staged_privately_under_the_name_startup_removes(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = staging_path
    made: list[Path] = []

    def recording(directory: Path, suffix: str) -> Path:
        made.append(real(directory, suffix))
        return made[-1]

    monkeypatch.setattr(backup_archives, "staging_path", recording)
    await _picture(client)

    archive = await _backup(client, include_media=True)
    assert (await _check(client, archive)).status_code == 200

    assert {path.name.removeprefix(STAGING_PREFIX).split(".", 1)[1] for path in made} == {
        "sqlite3",
        "media.zip",
        "lm-atelier.encrypted",
    }
    assert all(path.name.startswith(STAGING_PREFIX) for path in made)
    assert {path.parent for path in made} == {settings.export_dir / STAGING_FOLDER}
    assert _private(settings.export_dir / STAGING_FOLDER)
    assert not any(path.exists() for path in made)
    assert _staged(settings) == []


async def test_a_staging_folder_others_can_open_refuses_both_and_writes_nothing(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = await _backup(client)
    before = _stored_files(settings)
    monkeypatch.setattr(
        project_archive_encryption, "directory_private_to_current_user", lambda anchor: False
    )

    created = await _create(client)
    checked = await _check(client, archive)

    for refused in (created, checked):
        assert (refused.status_code, refused.json()["code"]) == (409, "archive-staging-not-private")
    assert _stored_files(settings) == before
    assert _staged(settings) == []


async def test_making_an_encrypted_backup_leaves_the_managed_backups_as_they_were(
    client: AsyncClient, settings: Settings
) -> None:
    managed = await client.post("/api/backups", params={"include_media": True})
    assert managed.status_code == 201, managed.text
    listed = (await client.get("/api/backups")).json()
    files = sorted(path.name for path in settings.backup_dir.iterdir())

    await _backup(client, include_media=True)

    assert (await client.get("/api/backups")).json() == listed
    assert sorted(path.name for path in settings.backup_dir.iterdir()) == files


async def _health_while_held(
    client: AsyncClient, started: threading.Event, release: threading.Event, work: Awaitable[Any]
) -> tuple[list[int], Any]:
    """Run ``work`` and, while it is held, ask the application for its health.

    The question comes from another thread. Asked from the test's own task, it
    could only be sent once a held event loop was free again, and would then be
    answered as if nothing had been held. From outside, a held loop cannot
    answer it at all. The work is released either way.
    """

    loop = asyncio.get_running_loop()
    answered: list[int] = []

    def probe() -> None:
        try:
            if started.wait(20):
                asked = asyncio.run_coroutine_threadsafe(client.get("/api/health"), loop)
                try:
                    answered.append(asked.result(timeout=5).status_code)
                except concurrent.futures.TimeoutError:
                    asked.cancel()
        finally:
            release.set()

    prober = threading.Thread(target=probe, daemon=True)
    prober.start()
    finished = await work
    await asyncio.to_thread(prober.join, 30)
    return answered, finished


async def test_making_an_encrypted_backup_leaves_the_application_answering(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, release = threading.Event(), threading.Event()
    real = BackupManager.snapshot_to

    def held(self: BackupManager, database: Path, media: Path | None) -> str:
        started.set()
        release.wait(20)
        return real(self, database, media)

    monkeypatch.setattr(BackupManager, "snapshot_to", held)

    answered, created = await _health_while_held(client, started, release, _create(client))

    assert answered == [200]
    assert created.status_code == 201


async def test_checking_an_encrypted_backup_leaves_the_application_answering(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = await _backup(client)
    started, release = threading.Event(), threading.Event()
    real = backup_archives._check_staged

    def held(*args: Any) -> Any:
        started.set()
        release.wait(20)
        return real(*args)

    monkeypatch.setattr(backup_archives, "_check_staged", held)

    answered, checked = await _health_while_held(client, started, release, _check(client, archive))

    assert answered == [200]
    assert checked.status_code == 200


async def test_the_first_download_of_a_stored_file_leaves_the_application_answering(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    picture = await _picture(client)
    started, release = threading.Event(), threading.Event()
    real = ArtifactStore.delivery_metadata

    def held(self: ArtifactStore, artifact: Any) -> Any:
        started.set()
        release.wait(20)
        return real(self, artifact)

    monkeypatch.setattr(ArtifactStore, "delivery_metadata", held)

    # Finding and hashing the file runs beside the event loop, not on it.
    answered, downloaded = await _health_while_held(
        client, started, release, client.get(f"/api/artifacts/{picture}/content")
    )

    assert answered == [200]
    assert downloaded.status_code == 200


async def test_a_cancelled_check_leaves_nothing_staged_and_frees_the_next_one(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = await _backup(client)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    real = backup_archives._check_staged

    def held(*args: Any) -> Any:
        try:
            started.set()
            release.wait(10)
            return real(*args)
        finally:
            finished.set()

    monkeypatch.setattr(backup_archives, "_check_staged", held)
    request = asyncio.create_task(_check(client, archive))
    assert await asyncio.to_thread(started.wait, 20)
    assert _staged(settings), "the upload should be staged while the check is held"

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
    monkeypatch.setattr(backup_archives, "_check_staged", real)
    assert (await _check(client, archive)).status_code == 200


async def test_a_passphrase_longer_than_the_format_allows_is_refused_before_any_copy(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied: list[Path] = []
    monkeypatch.setattr(
        BackupManager, "snapshot_to", lambda self, database, media: copied.append(database)
    )
    # Fewer than 1024 characters, more than 1024 bytes.
    too_long = "é" * 600

    refused = await client.post("/api/backups/encrypted", json={"passphrase": too_long})

    assert (refused.status_code, refused.json()["code"]) == (422, "archive-passphrase-invalid")
    assert too_long not in refused.text
    assert copied == [] and _staged(settings) == []


async def test_a_check_after_sealing_without_memory_is_not_called_an_unverified_backup(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    before = _stored_files(settings)

    refused = await _create(client)

    # The first derivation seals the backup; the second is the check.
    assert derivations == [1, 2]
    assert (refused.status_code, refused.json()["code"]) == (503, "archive-key-derivation-failed")
    assert _stored_files(settings) == before
    assert _staged(settings) == []


async def test_one_encrypted_backup_is_made_or_checked_at_a_time(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = await _backup(client)
    started, release = threading.Event(), threading.Event()
    real = BackupManager.snapshot_to

    def held(self: BackupManager, database: Path, media: Path | None) -> str:
        started.set()
        release.wait(10)
        return real(self, database, media)

    monkeypatch.setattr(BackupManager, "snapshot_to", held)
    first = asyncio.create_task(_create(client))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        second = await _create(client)
        checked = await _check(client, archive)
    finally:
        release.set()

    for refused in (second, checked):
        assert (refused.status_code, refused.json()["code"]) == (409, "encrypted-backup-busy")
    assert (await first).status_code == 201
    assert _staged(settings) == []


async def test_a_disk_without_room_for_the_copy_refuses_before_writing(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = await _backup(client)
    before = _stored_files(settings)
    monkeypatch.setattr(
        backup_archives,
        "shutil",
        SimpleNamespace(disk_usage=lambda folder: SimpleNamespace(free=0)),
    )

    created = await _create(client)
    checked = await _check(client, archive)

    for refused in (created, checked):
        assert (refused.status_code, refused.json()["code"]) == (507, "backup-storage-insufficient")
    assert _stored_files(settings) == before
    assert _staged(settings) == []
