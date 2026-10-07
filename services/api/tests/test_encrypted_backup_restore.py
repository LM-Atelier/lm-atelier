"""Restoring an encrypted backup on the next start, with its key held in the system vault."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import sqlite3
import struct
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from keyring.errors import KeyringError

from local_lm import archive_key_store, db
from local_lm.backups import BackupManager
from local_lm.config import Settings
from local_lm.main import create_app
from local_lm.portable_archive_v1 import MIN_MEMORY_KIB, ArchiveKind, KeyDerivation, write_archive
from local_lm.project_archive_encryption import STAGING_FOLDER

PASSPHRASE = "correct horse battery staple"
CHEAP = KeyDerivation(iterations=1, lanes=1, memory_kib=MIN_MEMORY_KIB)


class _Vault:
    """An operating-system vault, in memory."""

    def __init__(self, priority: int = 1) -> None:
        self.priority = priority
        self.entries: dict[tuple[str, str], str] = {}

    def get_keyring(self) -> _Vault:
        return self

    def get_password(self, service: str, account: str) -> str | None:
        return self.entries.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.entries[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        if (service, account) not in self.entries:
            raise KeyringError("no such entry")
        del self.entries[(service, account)]


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> _Vault:
    held = _Vault()
    monkeypatch.setattr(archive_key_store, "keyring", held)
    return held


@asynccontextmanager
async def _running(settings: Settings) -> AsyncIterator[AsyncClient]:
    app: FastAPI = create_app(settings)
    try:
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(app.state.retention_sweep, timeout=30)
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://testserver"
            ) as client:
                session = await client.post("/api/session")
                client.headers["x-local-lm-csrf"] = session.json()["csrf_token"]
                yield client
    finally:
        # Nothing may keep the database open into the next start, which
        # replaces the file when it applies a restore.
        db.engine.dispose()


def _headers(passphrase: str = PASSPHRASE) -> dict[str, str]:
    encoded = base64.b64encode(passphrase.encode("utf-8")).decode("ascii")
    return {"content-type": "application/octet-stream", "x-archive-passphrase": encoded}


async def _chat(client: AsyncClient, title: str) -> None:
    created = await client.post("/api/chats", json={"title": title})
    assert created.status_code == 201, created.text


async def _titles(client: AsyncClient) -> set[str]:
    listed = await client.get("/api/chats")
    assert listed.status_code == 200, listed.text
    return {chat["title"] for chat in listed.json()}


async def _encrypted_backup(client: AsyncClient) -> bytes:
    created = await client.post("/api/backups/encrypted", json={"passphrase": PASSPHRASE})
    assert created.status_code == 201, created.text
    return (await client.get(created.json()["url"])).content


async def _restore(client: AsyncClient, content: bytes, passphrase: str = PASSPHRASE) -> Any:
    return await client.post(
        "/api/backups/encrypted/restore", content=content, headers=_headers(passphrase)
    )


def _staged(settings: Settings) -> list[Path]:
    folder = settings.export_dir / STAGING_FOLDER
    return sorted(folder.iterdir()) if folder.is_dir() else []


def _marker(settings: Settings) -> Path:
    return settings.state_dir / "restore-on-next-start.json"


def _entry(settings: Settings) -> tuple[str, str]:
    """Where the vault holds the key of this data directory's restore."""

    return archive_key_store.SERVICE, archive_key_store.vault_account(settings.data_dir)


async def _scheduled(settings: Settings, vault: _Vault) -> bytes:
    """Back up one chat, add another, and ask for the backup back; returns the backup."""

    async with _running(settings) as client:
        await _chat(client, "Kept plan")
        backup = await _encrypted_backup(client)
        await _chat(client, "Later plan")
        scheduled = await _restore(client, backup)
        assert scheduled.status_code == 200, scheduled.text
    return backup


def _live_titles(settings: Settings) -> set[str]:
    with closing(sqlite3.connect(settings.state_dir / "local-lm.sqlite3")) as connection:
        return {row[0] for row in connection.execute("SELECT title FROM chats")}


async def test_an_encrypted_backup_asked_for_replaces_the_data_on_the_next_start(
    settings: Settings, vault: _Vault
) -> None:
    async with _running(settings) as client:
        await _chat(client, "Kept plan")
        backup = await _encrypted_backup(client)
        await _chat(client, "Later plan")

        scheduled = await _restore(client, backup)

        assert scheduled.status_code == 200, scheduled.text
        assert scheduled.headers["cache-control"] == "no-store"
        assert scheduled.json()["media_included"] is False
        state = (await client.get("/api/backups/restore-state")).json()
        assert (state["state"], state["backup"], state["encrypted"]) == ("pending", None, True)
        # Until then only the encrypted file waits, beside its key in the vault.
        (waiting,) = _staged(settings)
        assert waiting.read_bytes() == backup
        assert set(vault.entries) == {_entry(settings)}
        assert PASSPHRASE not in vault.entries[_entry(settings)]
        assert PASSPHRASE not in _marker(settings).read_text(encoding="utf-8")
        assert await _titles(client) == {"Kept plan", "Later plan"}

    async with _running(settings) as client:
        assert await _titles(client) == {"Kept plan"}
        state = (await client.get("/api/backups/restore-state")).json()
        assert state["state"] == "none"

    assert vault.entries == {}
    assert _staged(settings) == []


async def test_the_waiting_file_is_on_disk_before_it_is_put_in_place(
    settings: Settings, vault: _Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, int]] = []
    fsync, replace = os.fsync, os.replace

    def synced(fd: int) -> None:
        events.append(("synced", os.fstat(fd).st_ino))
        fsync(fd)

    def moved(source: Any, destination: Any) -> None:
        events.append(("moved", os.stat(source).st_ino))
        replace(source, destination)

    async with _running(settings) as client:
        await _chat(client, "Kept plan")
        backup = await _encrypted_backup(client)
        with monkeypatch.context() as watching:
            watching.setattr(os, "fsync", synced)
            watching.setattr(os, "replace", moved)
            scheduled = await _restore(client, backup)

        assert scheduled.status_code == 200, scheduled.text
        (waiting,) = _staged(settings)
        identity = waiting.stat().st_ino
        assert ("moved", identity) in events
        assert ("synced", identity) in events[: events.index(("moved", identity))]


async def test_a_wrong_passphrase_schedules_nothing_and_keeps_nothing(
    settings: Settings, vault: _Vault
) -> None:
    async with _running(settings) as client:
        await _chat(client, "Kept plan")
        backup = await _encrypted_backup(client)

        refused = await _restore(client, backup, passphrase="another passphrase")

        assert refused.status_code == 422, refused.text
        assert refused.json()["code"] == "archive-passphrase-or-archive-invalid"
        assert (await client.get("/api/backups/restore-state")).json()["state"] == "none"
    assert not _marker(settings).exists()
    assert vault.entries == {}
    assert _staged(settings) == []


async def test_without_a_vault_nothing_is_received_or_scheduled(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archive_key_store, "keyring", _Vault(priority=0))
    async with _running(settings) as client:
        await _chat(client, "Kept plan")
        backup = await _encrypted_backup(client)

        refused = await _restore(client, backup)

        assert refused.status_code == 503, refused.text
        assert refused.json()["code"] == "restore-needs-key-vault"
    assert not _marker(settings).exists()
    assert _staged(settings) == []


async def test_a_restore_whose_key_left_the_vault_keeps_the_data_and_says_why(
    settings: Settings, vault: _Vault
) -> None:
    await _scheduled(settings, vault)
    vault.entries.clear()

    async with _running(settings) as client:
        assert await _titles(client) == {"Kept plan", "Later plan"}
        state = (await client.get("/api/backups/restore-state")).json()
        assert (state["state"], state["reason"], state["encrypted"]) == (
            "failed",
            "backup-key-missing",
            True,
        )
    assert _staged(settings) == []


async def test_a_key_held_for_another_restore_is_not_used(
    settings: Settings, vault: _Vault
) -> None:
    await _scheduled(settings, vault)
    # The vault now holds the right key, but under another restore's name.
    operation = json.loads(_marker(settings).read_text(encoding="utf-8"))["encrypted"]["operation"]
    vault.entries[_entry(settings)] = vault.entries[_entry(settings)].replace(operation, "0" * 32)
    held = dict(vault.entries)

    assert BackupManager(settings).apply_pending_restore() is False

    assert _live_titles(settings) == {"Kept plan", "Later plan"}
    assert BackupManager(settings).restore_state().reason == "backup-key-missing"
    # Nor is it removed: it is not this restore's.
    assert vault.entries == held
    assert _staged(settings) == []


async def test_a_changed_encrypted_file_is_not_restored_and_its_key_is_removed(
    settings: Settings, vault: _Vault
) -> None:
    await _scheduled(settings, vault)
    (waiting,) = _staged(settings)
    changed = bytearray(waiting.read_bytes())
    changed[-40] ^= 1
    waiting.write_bytes(bytes(changed))

    assert BackupManager(settings).apply_pending_restore() is False

    assert _live_titles(settings) == {"Kept plan", "Later plan"}
    assert BackupManager(settings).restore_state().reason == "backup-invalid"
    assert vault.entries == {}
    assert not waiting.exists()


async def test_a_file_other_than_the_one_asked_for_is_not_restored_even_with_its_key(
    settings: Settings, vault: _Vault
) -> None:
    await _scheduled(settings, vault)
    # The marker and the vault both now name a file with another digest; the
    # waiting file and its key are untouched, so only the digest check refuses.
    other = hashlib.sha256(b"another file").hexdigest()
    payload = json.loads(_marker(settings).read_text(encoding="utf-8"))
    recorded = payload["encrypted"]["sha256"]
    payload["encrypted"]["sha256"] = other
    _marker(settings).write_text(json.dumps(payload), encoding="utf-8")
    vault.entries[_entry(settings)] = vault.entries[_entry(settings)].replace(recorded, other)

    assert BackupManager(settings).apply_pending_restore() is False

    assert _live_titles(settings) == {"Kept plan", "Later plan"}
    assert BackupManager(settings).restore_state().reason == "backup-invalid"
    assert vault.entries == {}
    assert _staged(settings) == []


async def test_withdrawing_an_encrypted_restore_removes_its_file_and_its_key(
    settings: Settings, vault: _Vault
) -> None:
    async with _running(settings) as client:
        await _chat(client, "Kept plan")
        backup = await _encrypted_backup(client)
        await _chat(client, "Later plan")
        assert (await _restore(client, backup)).status_code == 200

        withdrawn = await client.post("/api/backups/restore-state/cancel")

        assert withdrawn.status_code == 204
        assert (await client.get("/api/backups/restore-state")).json()["state"] == "none"
        assert vault.entries == {}
        assert _staged(settings) == []
        # Withdrawing again finds nothing, and says nothing is wrong.
        assert (await client.post("/api/backups/restore-state/cancel")).status_code == 204

    async with _running(settings) as client:
        assert await _titles(client) == {"Kept plan", "Later plan"}


async def test_a_backup_restore_asked_for_afterwards_replaces_the_encrypted_one(
    settings: Settings, vault: _Vault
) -> None:
    async with _running(settings) as client:
        await _chat(client, "Kept plan")
        backup = await _encrypted_backup(client)
        assert (await _restore(client, backup)).status_code == 200
        created = await client.post("/api/backups")
        assert created.status_code == 201, created.text

        scheduled = await client.post(f"/api/backups/{created.json()['name']}/restore")

        assert scheduled.status_code == 200, scheduled.text
        state = (await client.get("/api/backups/restore-state")).json()
        assert (state["state"], state["backup"], state["encrypted"]) == (
            "pending",
            created.json()["name"],
            False,
        )
        assert vault.entries == {}
        assert _staged(settings) == []


async def test_a_backup_made_by_a_newer_version_is_refused_when_it_is_asked_for(
    settings: Settings, vault: _Vault, tmp_path: Path
) -> None:
    copy = tmp_path / "newer.sqlite3"
    with (
        closing(sqlite3.connect(settings.state_dir / "local-lm.sqlite3")) as live,
        closing(sqlite3.connect(copy)) as target,
    ):
        live.backup(target)
    with closing(sqlite3.connect(copy)) as connection:
        connection.execute("UPDATE alembic_version SET version_num = 'ffffffffffff'")
        connection.commit()
    database = copy.read_bytes()
    header = json.dumps(
        {
            "format": "lm-atelier-backup",
            "version": 1,
            "created_at": "2026-10-04T12:00:00Z",
            "app_version": "0.0.0",
            "schema_revision": "ffffffffffff",
            "database": {
                "size_bytes": len(database),
                "sha256": hashlib.sha256(database).hexdigest(),
            },
            "media": None,
        }
    ).encode("utf-8")
    sealed = io.BytesIO()
    write_archive(
        io.BytesIO(b"LMABKUP\x00" + struct.pack(">I", len(header)) + header + database),
        sealed,
        kind=ArchiveKind.BACKUP,
        passphrase=PASSPHRASE.encode("utf-8"),
        derivation=CHEAP,
    )

    async with _running(settings) as client:
        refused = await _restore(client, sealed.getvalue())

        assert refused.status_code == 422, refused.text
        assert refused.json()["code"] == "backup-newer"
    assert not _marker(settings).exists()
    assert vault.entries == {}
    assert _staged(settings) == []


async def test_a_restore_whose_live_log_cannot_be_put_back_keeps_its_file_and_key(
    settings: Settings, vault: _Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _scheduled(settings, vault)
    live = settings.state_dir / "local-lm.sqlite3"
    log = live.with_name(live.name + "-wal")
    log.write_bytes(b"neutral uncheckpointed frames")
    real_replace = os.replace

    def refusing(source: str | Path, target: str | Path) -> None:
        # The database cannot be replaced, and its log cannot go back.
        if Path(target) in (live, log):
            raise PermissionError("the file is in use")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", refusing)

    with pytest.raises(OSError):
        BackupManager(settings).apply_pending_restore()

    # The start stops, and the next one must still be able to finish the restore.
    assert _marker(settings).exists()
    assert set(vault.entries) == {_entry(settings)}
    assert len(_staged(settings)) == 1


async def test_each_data_directory_keeps_its_own_waiting_restore(
    settings: Settings, vault: _Vault, tmp_path: Path
) -> None:
    # One vault serves every data directory on the computer.
    other = settings.model_copy(update={"data_dir": tmp_path / "other-data"})
    await _scheduled(settings, vault)
    await _scheduled(other, vault)
    assert _entry(settings) != _entry(other)
    assert set(vault.entries) == {_entry(settings), _entry(other)}

    assert BackupManager(settings).apply_pending_restore() is True
    assert _live_titles(settings) == {"Kept plan"}
    assert set(vault.entries) == {_entry(other)}

    assert BackupManager(other).apply_pending_restore() is True
    assert _live_titles(other) == {"Kept plan"}
    assert vault.entries == {}


async def test_withdrawing_a_restore_leaves_another_data_directorys_restore_waiting(
    settings: Settings, vault: _Vault, tmp_path: Path
) -> None:
    other = settings.model_copy(update={"data_dir": tmp_path / "other-data"})
    await _scheduled(settings, vault)
    await _scheduled(other, vault)
    waiting = _staged(other)

    assert BackupManager(settings).withdraw_requested_restore() is True

    assert not _marker(settings).exists() and _staged(settings) == []
    assert set(vault.entries) == {_entry(other)}
    assert _marker(other).exists() and _staged(other) == waiting
    assert BackupManager(other).apply_pending_restore() is True
    assert _live_titles(other) == {"Kept plan"}
