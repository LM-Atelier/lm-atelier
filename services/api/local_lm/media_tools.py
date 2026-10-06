"""Which FFmpeg and FFprobe the system path offers, described as they were found.

Elsewhere the app finds these tools by name on the system path and uses
whatever answers. A result that records how it was made needs more than a
name: the version the program reports and a digest of the program file. Both
are measured afresh for every call, from the file the path names at that
moment. The digest covers that one file, not shared libraries it may load; when
that file is a launcher, such as a package manager's shim, the digest is the
launcher's and the version is what the program it starts reports.

This describes the program that was found; it does not prove which bytes the
operating system then ran. Nothing here holds the file between measuring it
and starting it, so a program replaced in that moment would run unmeasured. A
tool found on the system path is also a build nobody here reviewed, so its
origin is recorded as ``system``, and its record never claims a reviewed build
or a verified execution.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from .media_process import ToolOutputTooLarge, run_tool

ToolName = Literal["ffmpeg", "ffprobe"]
ToolUnavailableCode = Literal["media-tool-missing", "media-tool-unreadable"]

VERSION_SECONDS: Final = 10.0
MAX_VERSION_BYTES: Final = 64 * 1024

#: The first line of ``-version``. Only plain build-name characters are kept, so
#: the version a record carries is never arbitrary text from the program.
_VERSION: Final = re.compile(rb"(ffmpeg|ffprobe) version ([0-9A-Za-z._+~-]{1,80})[ \r\n]")
_DIGEST_READ: Final = 1024 * 1024


class MediaToolUnavailable(RuntimeError):
    """The tool is not on the system path, or does not answer as that tool."""

    def __init__(self, code: ToolUnavailableCode) -> None:
        super().__init__(code)
        self.code: ToolUnavailableCode = code


@dataclass(frozen=True)
class MediaTool:
    name: ToolName
    executable: Path
    version: str
    sha256: str
    #: Where the build came from. Only the system path exists until a reviewed
    #: build can be installed.
    origin: Literal["system"] = "system"

    def record(self) -> dict[str, str]:
        """What a result says about the tool that was found.

        Never its path, which can name the person's folders.
        """

        return {
            "name": self.name,
            "version": self.version,
            "sha256": self.sha256,
            "origin": self.origin,
        }


async def find_media_tool(name: ToolName) -> MediaTool:
    """The tool on the system path with the version it reports and its file's digest.

    Nothing is remembered between calls: a program replaced with one of the same
    size and time would otherwise keep the identity of the program it replaced.
    """

    found = shutil.which(name)
    if found is None:
        raise MediaToolUnavailable("media-tool-missing")
    executable = Path(found)
    try:
        digest = await asyncio.to_thread(_digest, executable)
        answer = await run_tool(
            executable,
            ["-version"],
            seconds=VERSION_SECONDS,
            stdout_limit=MAX_VERSION_BYTES,
            stderr_limit=MAX_VERSION_BYTES,
        )
    except (OSError, TimeoutError, ToolOutputTooLarge) as exc:
        raise MediaToolUnavailable("media-tool-unreadable") from exc
    match = _VERSION.match(answer.stdout)
    if answer.returncode != 0 or match is None or match.group(1).decode() != name:
        raise MediaToolUnavailable("media-tool-unreadable")
    return MediaTool(name, executable, match.group(2).decode("ascii"), digest)


def _digest(executable: Path) -> str:
    digest = hashlib.sha256()
    with executable.open("rb") as program:
        while chunk := program.read(_DIGEST_READ):
            digest.update(chunk)
    return digest.hexdigest()
