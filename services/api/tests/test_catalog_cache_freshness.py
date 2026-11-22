"""A cached page is admitted only while it is younger than the window it is read under.

Every catalog read names a window: the freshness window decides whether the
source is asked at all, and the stale window decides whether a page may stand
in once the source has failed. A page exactly as old as its window has outlived
it, so a window of zero admits nothing and a zero freshness window always asks
the source.

The clock is held here rather than read. Whether a write and the next read
share a clock tick is timing, and a file time can run slightly ahead of
`time.time()`, which the store counts as no age at all. Admitting an age equal
to the window let either case serve a zero window's page, so whether the source
was asked depended on when the read happened to run.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from local_lm.catalog_cache import CatalogCachePolicy, CatalogCacheStore
from local_lm.civitai_catalog import CivitaiCatalog
from local_lm.config import Settings

KEY = "e" * 64
PAGE = '{"items":[]}'
#: A whole second, so the filesystem stores it exactly and the file time the
#: store reads back is this instant rather than a rounding of it. It is long
#: past, so a read that consulted the real clock would find the page years old
#: and refuse it under every window these tests use.
WRITTEN_AT = 1_700_000_000
WRITTEN_AT_NS = WRITTEN_AT * 1_000_000_000


def _page_read_at(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    now: float,
) -> tuple[CatalogCacheStore, Path]:
    """A cached page whose file time is `WRITTEN_AT`, read by a clock held at `now`."""

    store = CatalogCacheStore(tmp_path / "cache")
    path = store.path(KEY)
    store.write_text(path, PAGE)
    os.utime(path, ns=(WRITTEN_AT_NS, WRITTEN_AT_NS))
    # The store reads back exactly this time, so a clock held at `WRITTEN_AT`
    # is at the file's own modification time rather than near it.
    assert path.stat().st_mtime == WRITTEN_AT
    monkeypatch.setattr(store, "_now", lambda: now)
    return store, path


def test_a_zero_window_is_not_fresh_at_the_instant_the_file_was_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No time has passed since the write, and a zero window still admits nothing.

    This is a read in the same clock tick as the write. Age zero is not younger
    than a window of zero, so the caller goes to the source.
    """

    store, path = _page_read_at(tmp_path, monkeypatch, now=WRITTEN_AT)

    assert store.read_text(path, max_age_seconds=0) is None
    assert store.read_bytes(path, max_age_seconds=0) is None
    # Read without a window, the same file is there, so the refusals above are
    # the window's and not a read that failed.
    assert store.read_text(path) == PAGE


def test_a_file_time_ahead_of_the_clock_is_not_fresh_under_a_zero_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file time ahead of the clock is no age, not a negative one.

    Counted as negative, the age would be younger than a zero window and the
    page would be served without asking the source.
    """

    store, path = _page_read_at(tmp_path, monkeypatch, now=WRITTEN_AT - 0.5)

    assert store.read_text(path, max_age_seconds=0) is None
    # Counted as no age, the page is still current under a positive window.
    assert store.read_text(path, max_age_seconds=300) == PAGE


def test_a_file_time_far_ahead_of_the_clock_is_still_no_age(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """However far a file time runs ahead of the clock, the page is no age, not that old.

    Read as the distance between the two times, a clock set back by more than
    the window would turn a page just written away as expired.
    """

    store, path = _page_read_at(tmp_path, monkeypatch, now=WRITTEN_AT - 400)

    assert store.read_text(path, max_age_seconds=300) == PAGE


@pytest.mark.parametrize("age", [0, 299.5])
def test_a_positive_window_still_serves_a_page_younger_than_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    age: float,
) -> None:
    """Making zero mean never must not turn away a page that is still current."""

    store, path = _page_read_at(tmp_path, monkeypatch, now=WRITTEN_AT + age)

    assert store.read_text(path, max_age_seconds=300) == PAGE
    assert store.read_bytes(path, max_age_seconds=300) == PAGE.encode()


def test_a_page_exactly_as_old_as_its_window_has_outlived_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound is the same at every window, not a special case for zero.

    The store applies one rule whichever window a caller reads under, the
    freshness window or the stale one, so a stale page is served as a fallback
    only while it is younger than the stale window too.
    """

    store, path = _page_read_at(tmp_path, monkeypatch, now=WRITTEN_AT + 300)

    assert store.read_text(path, max_age_seconds=300) is None


async def test_a_zero_freshness_window_asks_the_source_at_the_instant_of_the_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog search asks the source even with the clock at the cached page's own time.

    The source is down for the second search, so the page can only come back
    marked stale if the freshness read refused it and the source was asked.
    """

    asked = 0
    failing = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal asked
        asked += 1
        if failing:
            raise httpx.ConnectError("source unreachable", request=request)
        return httpx.Response(200, json={"items": [], "metadata": {}})

    catalog = CivitaiCatalog(
        Settings(data_dir=tmp_path),
        transport=httpx.MockTransport(handler),
        cache_policy=CatalogCachePolicy(fresh_seconds=0, stale_seconds=3600),
    )
    try:
        await catalog.search_workflows(query="portrait")
        pages = sorted(catalog._cache.root.glob("*.json"))
        assert len(pages) == 1
        os.utime(pages[0], ns=(WRITTEN_AT_NS, WRITTEN_AT_NS))
        assert pages[0].stat().st_mtime == WRITTEN_AT
        monkeypatch.setattr(catalog._cache, "_now", lambda: float(WRITTEN_AT))
        failing = True
        page = await catalog.search_workflows(query="portrait")
    finally:
        await catalog.close()

    assert asked == 2, "the zero freshness window served the cached page without asking"
    assert page.stale is True
