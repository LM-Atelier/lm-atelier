from __future__ import annotations

import os
import threading
from collections.abc import Callable, Generator
from datetime import UTC, datetime, timedelta

import pytest

from local_lm import filesystem_links
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.filesystem_links import AnchoredDirectory, AnchoredEntry


@pytest.mark.parametrize("level", [0, 1, 2])
def test_orphan_listing_observes_stop_during_native_enumeration(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, level: int
) -> None:
    store = ArtifactStore(settings)
    target = store.root.joinpath(*("aa", "00")[:level])
    target.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (target / f"keep-{index}").write_bytes(b"unrecognized file stays untouched")
    earlier = store.root / "ingest-abandoned.tmp"
    if level:
        earlier.write_bytes(b"aged temporary")
        old = (datetime.now(UTC) - timedelta(hours=25)).timestamp()
        os.utime(earlier, (old, old))
    stop = threading.Event()
    records_seen: list[str] = []
    real_records = filesystem_links._iter_anchored_entries

    def observed_records(
        anchor: AnchoredDirectory, limit: int, should_stop: Callable[[], bool] | None
    ) -> Generator[AnchoredEntry, None, None]:
        records = real_records(anchor, limit, should_stop)
        try:
            for entry in records:
                if anchor.path == target:
                    records_seen.append(entry.name)
                yield entry
                if anchor.path == target and len(records_seen) == 1:
                    stop.set()
        finally:
            records.close()

    monkeypatch.setattr(filesystem_links, "_iter_anchored_entries", observed_records)
    with SessionLocal() as session:
        summary = store.cleanup_retention(
            session,
            retention_days=30,
            temporary_hours=24,
            dry_run=False,
            should_stop=stop.is_set,
        )
        session.commit()
    assert stop.is_set(), "the native record boundary must have been reached"
    assert len(records_seen) == 1, "a stop must prevent fetching another native record"
    assert summary.truncated is True
    assert summary.removed_count == int(bool(level)), "completed removals must remain counted"
    assert summary.reclaimed_bytes == (len(b"aged temporary") if level else 0)
    assert not earlier.exists()
    assert target.is_dir(), "a stopped listing must not prune its directory"
    assert all((target / f"keep-{index}").is_file() for index in range(3))
