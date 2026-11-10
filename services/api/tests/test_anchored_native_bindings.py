"""Anchored operations on Windows share one set of native bindings."""

from __future__ import annotations

import gc
import os
import tracemalloc
from pathlib import Path

import pytest

from local_lm.filesystem_links import AnchoredDirectory, list_entries, remove_entry


@pytest.mark.skipif(os.name != "nt", reason="native Windows bindings")
def test_removing_many_entries_holds_no_memory_for_each_one(tmp_path: Path) -> None:
    folder = tmp_path / "many"
    folder.mkdir()
    for index in range(300):
        (folder / f"{index:03}").write_bytes(b"neutral")
    with AnchoredDirectory(folder) as anchor:
        names = [entry.name for entry in list_entries(anchor)]
        for name in names[:20]:
            remove_entry(anchor, name)
        tracemalloc.start()
        try:
            # Each removal also leaves short-lived cycles that only the
            # collector frees; collecting first counts what stays held.
            gc.collect()
            first = tracemalloc.take_snapshot()
            for name in names[20:]:
                remove_entry(anchor, name)
            gc.collect()
            second = tracemalloc.take_snapshot()
        finally:
            tracemalloc.stop()
    held = sum(change.size_diff for change in second.compare_to(first, "filename"))
    assert held < 256 * 1024
    assert not any(folder.iterdir())
