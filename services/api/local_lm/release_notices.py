"""The third-party notices an installed release carries.

A release build writes an inventory of the third-party software it contains,
and the license texts beside it, into the application bundle. A run from
source has neither, which is reported as not available rather than as an
error, so About & support can say where the notices are found instead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

NOTICES_FILE = "THIRD_PARTY_NOTICES.md"
LICENSES_FOLDER = "third-party-licenses"
#: Far above any real inventory; a larger file is not shown whole or in part.
MAX_NOTICES_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class ThirdPartyNotices:
    text: str | None
    license_folder: str | None


def release_bundle_root() -> Path | None:
    """The folder a frozen release unpacks its bundled files into, if this is one."""

    root = getattr(sys, "_MEIPASS", None)
    return Path(str(root)) if root else None


def read_third_party_notices(root: Path | None) -> ThirdPartyNotices:
    """The notices inventory and the license folder under `root`, where they exist."""

    if root is None:
        return ThirdPartyNotices(text=None, license_folder=None)
    notices = root / NOTICES_FILE
    text: str | None = None
    if notices.is_file() and notices.stat().st_size <= MAX_NOTICES_BYTES:
        text = notices.read_text(encoding="utf-8", errors="replace")
    licenses = root / LICENSES_FOLDER
    return ThirdPartyNotices(
        text=text,
        license_folder=str(licenses) if licenses.is_dir() else None,
    )
