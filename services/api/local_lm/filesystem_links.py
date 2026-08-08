from __future__ import annotations

import stat
from pathlib import Path
from typing import Literal

LinkInspectionFailure = Literal["assume_link", "assume_regular", "raise"]


def is_link_or_reparse(
    path: Path,
    *,
    missing: LinkInspectionFailure,
    unreadable: LinkInspectionFailure,
) -> bool:
    """Inspect a path without following filesystem links.

    Callers must choose how absence and other inspection failures affect their
    own safety boundary. Destructive and trust-sensitive paths generally fail
    closed; optional discovery paths can treat a missing entry as ordinary.
    """

    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        return _failure_result(missing, exc)
    except OSError as exc:
        return _failure_result(unreadable, exc)
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _failure_result(policy: LinkInspectionFailure, error: OSError) -> bool:
    if policy == "raise":
        raise error
    return policy == "assume_link"
