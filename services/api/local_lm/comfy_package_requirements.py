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
from contextlib import closing
from pathlib import Path, PurePosixPath

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryKind,
    walk_entries,
)

MAX_REQUIREMENTS_BYTES = 64 * 1024
MAX_REQUIREMENTS_LINES = 512
REQUIREMENTS_NAME = "requirements.txt"


class StagedRequirementsError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def select_requirements_manifest(manifests: Sequence[str]) -> str | None:
    """Pick the package's own requirements file, or none if it declares nothing.

    An archive can carry more than one candidate. Staging may remove a
    provider's synthetic repository wrapper, but the package's own file is
    still the shallowest candidate while anything deeper belongs to something
    it vendored. The shallowest wins for that reason, and a genuine tie is
    refused rather than guessed: two files at the same depth mean the archive
    does not say which one describes it.
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


MAX_STAGED_MANIFEST_SCAN = 20_000


def staged_requirements_manifests(root: Path) -> tuple[str, ...]:
    """Every requirements file already staged under this package, by relative path.

    Staging reports these while it unpacks. A renewal has no archive to report
    them, because it deliberately reuses the node code it already reviewed, so
    the same question is asked of the copy on disk and answered the same way -
    the selector below still decides which of them describes the package.

    Only the package's own files count. The tree is walked through held
    directories and a link is never entered, so a folder linked into the
    package cannot lend it a requirements file, and a link or an entry of
    unknown kind is never reported as one. A tree that cannot be walked that
    way - the package folder is itself a link, or the tree changes while it is
    read - refuses, because a package that declares nothing and a package that
    could not be read are different answers.

    Bounded: a staged tree that is somehow enormous stops the scan rather than
    walking it forever, and the caller sees only what was found before the
    stop instead of hanging. The walk is breadth first, so what falls inside
    the bound is the shallowest part of the tree, where the package's own file
    is; a large folder cannot use up the bound before the root is read. A
    single folder holding more entries than the scan reads refuses instead,
    since it is listed whole.
    """
    if not root.is_dir():
        return ()
    found: list[str] = []
    try:
        with (
            AnchoredDirectory(root) as anchor,
            closing(
                walk_entries(
                    anchor,
                    # The scan stops itself on the entry past its bound, so the
                    # walk's total never refuses first. A folder is listed
                    # whole, so one folder may hold as much as the scan reads.
                    limit=MAX_STAGED_MANIFEST_SCAN + 1,
                    level_limit=MAX_STAGED_MANIFEST_SCAN + 1,
                    breadth_first=True,
                    include_metadata=False,
                )
            ) as walked,
        ):
            for seen, item in enumerate(walked, start=1):
                if seen > MAX_STAGED_MANIFEST_SCAN:
                    break
                if (
                    item.entry.kind is AnchoredEntryKind.FILE
                    and item.entry.name.casefold() == REQUIREMENTS_NAME
                ):
                    found.append("/".join(item.parts))
    except AnchoredDirectoryError as exc:
        raise StagedRequirementsError(
            "unreadable_requirements", "The staged package could not be read safely"
        ) from exc
    return tuple(sorted(found))


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
