"""Read what a commit-pinned package declares, after staging and before trust.

A package resolved from the Registry states its dependencies in its Registry
record. A package pinned to a commit states them nowhere but inside its own
tree, so they can only be read once the archive is staged. Reading is all that
happens here: the file is parsed as text, nothing in it is executed, and the
result is a list of declarations for the ordinary dependency planner to judge.

This module makes no claim about the archive's provenance. A commit-addressed
archive is identified by its repository and revision; the hash observed at
staging is what makes later local operations exact and is not evidence about
the Git object it came from.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path, PurePosixPath

MAX_REQUIREMENTS_BYTES = 64 * 1024
MAX_REQUIREMENTS_LINES = 512
REQUIREMENTS_NAME = "requirements.txt"


class StagedRequirementsError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def select_requirements_manifest(manifests: Sequence[str]) -> str | None:
    """Pick the package's own requirements file, or none if it declares nothing.

    An archive carries more than one candidate. A commit archive from GitHub
    wraps the tree in a single directory, so the package's own file sits one
    level down, while anything deeper belongs to something the package
    vendored rather than to the package itself. The shallowest wins for that
    reason, and a genuine tie is refused rather than guessed: two files at the
    same depth mean the archive does not say which one describes it.
    """
    candidates = [
        path
        for path in manifests
        if PurePosixPath(path).name.casefold() == REQUIREMENTS_NAME
        and not PurePosixPath(path).is_absolute()
    ]
    if not candidates:
        return None
    shallowest = min(len(PurePosixPath(path).parts) for path in candidates)
    at_depth = sorted(path for path in candidates if len(PurePosixPath(path).parts) == shallowest)
    if len(at_depth) > 1:
        raise StagedRequirementsError(
            "ambiguous_requirements",
            "The package stages more than one requirements file at its root",
        )
    return at_depth[0]


def read_staged_requirements(root: Path, manifest: str) -> tuple[str, ...]:
    """Read one staged requirements file into inert declaration lines.

    Bounded on the way in and never interpreted. Installer options, comments,
    and malformed requirements are the dependency planner's business; this
    returns the file's lines and lets one place decide what they mean.
    """
    target = _inside(root, manifest)
    try:
        with target.open("rb") as handle:
            # One byte past the bound, so the limit is enforced by what is read
            # rather than by what was already allocated. A staged package is
            # not trusted yet, and a file large enough to matter must not be
            # held in memory on the way to being refused.
            payload = handle.read(MAX_REQUIREMENTS_BYTES + 1)
    except OSError as exc:
        raise StagedRequirementsError(
            "unreadable_requirements", "The staged requirements file could not be read"
        ) from exc
    if len(payload) > MAX_REQUIREMENTS_BYTES:
        raise StagedRequirementsError(
            "requirements_too_large", "The staged requirements file is too large to read"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StagedRequirementsError(
            "unreadable_requirements", "The staged requirements file is not valid UTF-8"
        ) from exc
    lines = text.splitlines()
    if len(lines) > MAX_REQUIREMENTS_LINES:
        raise StagedRequirementsError(
            "too_many_requirements", "The staged requirements file declares too many lines"
        )
    return tuple(lines)


def _inside(root: Path, manifest: str) -> Path:
    """Resolve a staged path, refusing anything that leaves the staged tree.

    Staging validated these entries already. This checks again because the
    path is being used to read from disk, and a check that costs nothing is
    worth repeating at the boundary that acts on it.
    """
    relative = PurePosixPath(manifest)
    if relative.is_absolute() or any(part in {"..", ""} for part in relative.parts):
        raise StagedRequirementsError(
            "invalid_requirements_path", "The staged requirements path is not inside the package"
        )
    try:
        staged_root = root.resolve(strict=True)
        target = root.joinpath(*relative.parts).resolve(strict=True)
    except OSError as exc:
        raise StagedRequirementsError(
            "unreadable_requirements", "The staged requirements file could not be read"
        ) from exc
    if not target.is_file() or staged_root not in target.parents:
        raise StagedRequirementsError(
            "invalid_requirements_path", "The staged requirements path is not inside the package"
        )
    return target
