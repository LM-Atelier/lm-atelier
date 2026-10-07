"""About & support reads the third-party notices a release carries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from httpx2 import AsyncClient

from local_lm import release_notices
from local_lm.release_notices import (
    LICENSES_FOLDER,
    MAX_NOTICES_BYTES,
    NOTICES_FILE,
    read_third_party_notices,
)

INVENTORY = "# Third-party notices\n\n| Ecosystem | Package | Version | Declared license |\n"


def _bundle(root: Path, *, notices: str | None = INVENTORY, licenses: bool = True) -> Path:
    root.mkdir()
    if notices is not None:
        (root / NOTICES_FILE).write_text(notices, encoding="utf-8")
    if licenses:
        (root / LICENSES_FOLDER).mkdir()
    return root


def test_a_run_from_source_has_no_notices() -> None:
    notices = read_third_party_notices(None)

    assert (notices.text, notices.license_folder) == (None, None)


def test_a_release_bundle_gives_its_inventory_and_license_folder(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle")

    notices = read_third_party_notices(root)

    assert notices.text == INVENTORY
    assert notices.license_folder == str(root / LICENSES_FOLDER)


def test_each_part_is_reported_only_where_it_exists(tmp_path: Path) -> None:
    without_licenses = read_third_party_notices(_bundle(tmp_path / "a", licenses=False))
    without_inventory = read_third_party_notices(_bundle(tmp_path / "b", notices=None))

    assert (without_licenses.text, without_licenses.license_folder) == (INVENTORY, None)
    assert without_inventory.text is None
    assert without_inventory.license_folder == str(tmp_path / "b" / LICENSES_FOLDER)


def test_an_inventory_past_the_size_limit_is_not_shown(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "bundle", notices="x" * (MAX_NOTICES_BYTES + 1))

    assert read_third_party_notices(root).text is None


def _make_link_dir(link: Path, target: Path) -> bool:
    """Create a directory-shaped redirection, or False without privileges."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def test_a_linked_notices_file_is_not_read(tmp_path: Path) -> None:
    outside = tmp_path / "outside-notices.md"
    outside.write_text("A neutral fixture.", encoding="utf-8")
    root = tmp_path / "bundle"
    root.mkdir()
    link = root / NOTICES_FILE
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    notices = read_third_party_notices(root)

    assert notices.text is None
    assert outside.read_text(encoding="utf-8") == "A neutral fixture."
    assert link.is_symlink()


def test_a_linked_license_folder_is_not_reported(tmp_path: Path) -> None:
    outside = tmp_path / "outside-licenses"
    outside.mkdir()
    (outside / "license.txt").write_text("A neutral fixture.", encoding="utf-8")
    root = tmp_path / "bundle"
    root.mkdir()
    link = root / LICENSES_FOLDER
    if not _make_link_dir(link, outside):
        pytest.skip("directory links are unavailable")

    notices = read_third_party_notices(root)

    assert notices.license_folder is None
    assert (outside / "license.txt").read_text(encoding="utf-8") == "A neutral fixture."


def test_a_linked_release_root_is_not_read(tmp_path: Path) -> None:
    real = _bundle(tmp_path / "real")
    link = tmp_path / "linked-root"
    if not _make_link_dir(link, real):
        pytest.skip("directory links are unavailable")

    notices = read_third_party_notices(link)

    assert (notices.text, notices.license_folder) == (None, None)
    assert (real / NOTICES_FILE).read_text(encoding="utf-8") == INVENTORY


def test_the_bundle_is_the_frozen_release_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert release_notices.release_bundle_root() is None

    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert release_notices.release_bundle_root() == tmp_path


async def test_about_serves_the_notices_of_the_running_release(
    client: AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    from_source = await client.get("/api/about/third-party-notices")
    assert from_source.status_code == 200
    assert from_source.json() == {"text": None, "license_folder": None}

    root = _bundle(tmp_path / "bundle")
    monkeypatch.setattr(sys, "_MEIPASS", str(root), raising=False)
    released = await client.get("/api/about/third-party-notices")
    assert released.json() == {
        "text": INVENTORY,
        "license_folder": str(root / LICENSES_FOLDER),
    }
