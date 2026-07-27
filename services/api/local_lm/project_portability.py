from __future__ import annotations

import re
from typing import Any

LOCAL_PATH_REDACTION = "[local path removed]"

_FILE_URI = re.compile(r"(?i)\bfile:(?://)?(?:[^\r\n;,\"'<>]|%[0-9a-f]{2})+")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?<![\w])(?:\\\\\?\\)?[a-z]:[\\/](?:[^\r\n;,\"'<>])+")
_WINDOWS_UNC_PATH = re.compile(
    r"(?<![:\w])[\\/]{2}[^\\/\s\"'<>]+[\\/][^\\/\s\"'<>]+"
    r"(?:[\\/][^\r\n;,\"'<>]*)?"
)
_HOME_RELATIVE_PATH = re.compile(r"(?<![\w])~[\\/](?:[^\r\n;,\"'<>])*")
_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![:/\w])/(?!/)[^\r\n;,\"'<>]*")
_RELATIVE_TRAVERSAL_PATH = re.compile(r"(?<![\w])(?:\.\.[\\/])+(?:[^\r\n;,\"'<>])*")


def redact_local_paths(value: Any) -> Any:
    """Return JSON-compatible data without local or traversing filesystem paths."""
    if isinstance(value, dict):
        return {
            _redact_path_text(key) if isinstance(key, str) else key: redact_local_paths(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_local_paths(child) for child in value]
    if isinstance(value, str):
        return _redact_path_text(value)
    return value


def has_local_path(value: Any) -> bool:
    """Report whether JSON-compatible data contains a local or traversing path."""
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and _contains_path_text(key)) or has_local_path(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(has_local_path(child) for child in value)
    return isinstance(value, str) and _contains_path_text(value)


def _redact_path_text(value: str) -> str:
    redacted = value
    for pattern in (
        _FILE_URI,
        _WINDOWS_UNC_PATH,
        _WINDOWS_ABSOLUTE_PATH,
        _HOME_RELATIVE_PATH,
        _RELATIVE_TRAVERSAL_PATH,
        _POSIX_ABSOLUTE_PATH,
    ):
        redacted = pattern.sub(LOCAL_PATH_REDACTION, redacted)
    return redacted


def _contains_path_text(value: str) -> bool:
    return any(
        pattern.search(value)
        for pattern in (
            _FILE_URI,
            _WINDOWS_UNC_PATH,
            _WINDOWS_ABSOLUTE_PATH,
            _HOME_RELATIVE_PATH,
            _RELATIVE_TRAVERSAL_PATH,
            _POSIX_ABSOLUTE_PATH,
        )
    )
