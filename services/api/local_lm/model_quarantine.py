"""Model deletion quarantine: stage, restore, finalize and recover removed model files.

Deleting a model or an auxiliary asset first moves its files into a quarantine
directory under the model root, marked with the id of the install they belong
to, and only then commits the deletion. After the commit the quarantine is
finalized; if the deletion fails, the files are moved back. Recovery settles
what an interruption left behind: a quarantine whose install still exists is
restored, and one whose install is gone is finalized. A link met along the way
is refused rather than followed.
"""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import suppress
from pathlib import Path, PurePosixPath

from sqlalchemy import select
from sqlalchemy.orm import Session

from .domain import new_id
from .filesystem_links import is_link_or_reparse
from .models import ModelAssetInstall, ModelInstall

logger = logging.getLogger(__name__)

_MODEL_DELETE_QUARANTINE = ".delete-pending"
_MODEL_DELETE_MARKER = ".model-id"


def _managed_model_path(model_root: Path, value: str) -> Path | None:
    """Return a confined, link-free managed path or None for external imports."""

    raw = Path(os.path.abspath(os.fspath(Path(value).expanduser())))
    try:
        relative = raw.relative_to(model_root)
    except ValueError:
        return None
    if not relative.parts or relative.parts[0] == _MODEL_DELETE_QUARANTINE:
        return None
    cursor = model_root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and _model_path_is_link(cursor):
            raise ValueError("managed model paths cannot use filesystem links")
    resolved = raw.resolve(strict=False)
    if model_root not in resolved.parents or resolved == model_root:
        raise ValueError("managed model path escapes model storage")
    return resolved


def _model_path_is_link(path: Path) -> bool:
    return is_link_or_reparse(
        path,
        missing="assume_regular",
        unreadable="assume_link",
    )


def _ensure_model_tree_link_free(path: Path) -> None:
    if _model_path_is_link(path):
        raise ValueError("managed model paths cannot use filesystem links")
    if not path.is_dir():
        return
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                candidate = Path(entry.path)
                if _model_path_is_link(candidate):
                    raise ValueError("managed model directories cannot contain filesystem links")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(candidate)


def _quarantine_model_files(
    session: Session,
    install: ModelInstall,
    model_root: Path,
    path: Path,
) -> tuple[list[tuple[Path, Path]], Path | None]:
    if not path.exists():
        return [], None
    siblings = _related_model_installs(session, install, model_root, path)
    quarantine: Path | None = None
    moves: list[tuple[Path, Path]] = []
    try:
        if not siblings:
            _ensure_model_tree_link_free(path)
            quarantine = _new_model_quarantine(model_root, install.id)
            staged = quarantine / "payload"
            os.replace(path, staged)
            return [(staged, path)], quarantine

        if not path.is_dir():
            # Another install owns the same managed file.
            return [], None

        retained_paths, retained_roots = _retained_model_paths(
            siblings,
            model_root,
        )
        for relative in _manifest_model_files(install):
            candidate = _confined_manifest_file(path, relative)
            if not candidate.exists():
                continue
            if _model_path_is_link(candidate):
                raise ValueError("managed model files cannot use filesystem links")
            if not candidate.is_file():
                raise ValueError("managed model manifests may only reference files")
            if candidate in retained_paths or any(
                candidate == root or root in candidate.parents for root in retained_roots
            ):
                continue
            if quarantine is None:
                quarantine = _new_model_quarantine(model_root, install.id)
            staged = _safe_quarantine_file_path(quarantine, relative)
            os.replace(candidate, staged)
            moves.append((staged, candidate))
        return moves, quarantine
    except Exception:
        try:
            _restore_model_moves(moves)
        except Exception:
            logger.exception(
                "Model staging failure left recoverable files in quarantine %s",
                quarantine,
            )
        else:
            try:
                _finalize_model_quarantine(quarantine)
            except Exception:
                logger.warning(
                    "Could not prune a failed model deletion quarantine %s",
                    quarantine,
                    exc_info=True,
                )
        raise


def _related_model_installs(
    session: Session,
    install: ModelInstall,
    model_root: Path,
    path: Path,
) -> list[tuple[ModelInstall, Path]]:
    related: list[tuple[ModelInstall, Path]] = []
    for sibling in session.scalars(select(ModelInstall).where(ModelInstall.id != install.id)).all():
        try:
            sibling_path = _managed_model_path(model_root, sibling.local_path)
        except ValueError:
            # A linked sibling is never evidence that it is safe to remove
            # content through the current install.
            continue
        if sibling_path is None:
            continue
        if sibling_path == path or sibling_path in path.parents or path in sibling_path.parents:
            related.append((sibling, sibling_path))
    return related


def _retained_model_paths(
    siblings: list[tuple[ModelInstall, Path]],
    model_root: Path,
) -> tuple[set[Path], set[Path]]:
    retained_paths: set[Path] = set()
    retained_roots: set[Path] = set()
    for sibling, sibling_path in siblings:
        if sibling_path.is_file():
            retained_paths.add(sibling_path)
            continue
        try:
            files = _manifest_model_files(sibling)
        except ValueError:
            retained_roots.add(sibling_path)
            continue
        if not files:
            retained_roots.add(sibling_path)
            continue
        for relative in files:
            try:
                candidate = _confined_manifest_file(sibling_path, relative)
            except ValueError:
                retained_roots.add(sibling_path)
                break
            if candidate == model_root or model_root not in candidate.parents:
                retained_roots.add(sibling_path)
                break
            retained_paths.add(candidate)
    return retained_paths, retained_roots


def _manifest_model_files(install: ModelInstall) -> list[PurePosixPath]:
    raw_files = install.manifest_json.get("files", [])
    if not isinstance(raw_files, list):
        raise ValueError("managed model manifest files must be a list")
    files: list[PurePosixPath] = []
    seen: set[str] = set()
    for value in raw_files:
        if not isinstance(value, str) or not value:
            raise ValueError("managed model manifest contains an invalid file path")
        relative = PurePosixPath(value.replace("\\", "/"))
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} or ":" in part for part in relative.parts)
        ):
            raise ValueError("managed model manifest contains an unsafe file path")
        identity = relative.as_posix().casefold()
        if identity in seen:
            continue
        seen.add(identity)
        files.append(relative)
    return files


def _confined_manifest_file(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and _model_path_is_link(cursor):
            raise ValueError("managed model files cannot use filesystem links")
    resolved = candidate.resolve(strict=False)
    if root not in resolved.parents:
        raise ValueError("managed model manifest path escapes its install directory")
    return resolved


def _new_model_quarantine(model_root: Path, model_id: str) -> Path:
    parent = model_root / _MODEL_DELETE_QUARANTINE
    if parent.exists() and _model_path_is_link(parent):
        raise ValueError("model deletion quarantine cannot use a filesystem link")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.resolve().parent != model_root:
        raise ValueError("model deletion quarantine escapes model storage")
    quarantine = parent / new_id("delete")
    quarantine.mkdir(mode=0o700)
    try:
        marker = quarantine / _MODEL_DELETE_MARKER
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(model_id)
            handle.flush()
            os.fsync(handle.fileno())
        marker.chmod(0o600)
    except Exception:
        with suppress(OSError):
            quarantine.rmdir()
        raise
    return quarantine


def _safe_quarantine_file_path(
    quarantine: Path,
    relative: PurePosixPath,
) -> Path:
    if _model_path_is_link(quarantine) or not quarantine.is_dir():
        raise ValueError("model deletion quarantine contains a filesystem link")
    files = quarantine / "files"
    if _model_path_is_link(files):
        raise ValueError("model deletion quarantine contains a filesystem link")
    if not files.exists():
        files.mkdir()
    if _model_path_is_link(files) or not files.is_dir():
        raise ValueError("model deletion quarantine contains an unsafe files directory")
    files_root = files.resolve()
    if files_root.parent != quarantine.resolve():
        raise ValueError("model deletion quarantine escapes model storage")
    cursor = files
    for part in relative.parts[:-1]:
        cursor /= part
        if _model_path_is_link(cursor):
            raise ValueError("model deletion quarantine contains a filesystem link")
        if cursor.exists():
            if not cursor.is_dir():
                raise ValueError("model deletion quarantine contains a filesystem link")
        else:
            cursor.mkdir()
            if _model_path_is_link(cursor) or not cursor.is_dir():
                raise ValueError("model deletion quarantine contains a filesystem link")
    staged = cursor / relative.name
    if staged.exists() or _model_path_is_link(staged):
        raise ValueError("model deletion quarantine contains an unexpected file")
    resolved_parent = staged.parent.resolve()
    if resolved_parent != files_root and files_root not in resolved_parent.parents:
        raise ValueError("model deletion quarantine escapes model storage")
    return staged


def _restore_model_moves(moves: list[tuple[Path, Path]]) -> None:
    for staged, original in reversed(moves):
        if not staged.exists():
            continue
        if original.exists():
            raise RuntimeError("cannot restore a quarantined model over an existing path")
        original.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, original)


def _finalize_model_quarantine(quarantine: Path | None) -> None:
    if quarantine is None or not quarantine.exists():
        return
    if _model_path_is_link(quarantine) or not quarantine.is_dir():
        raise OSError("refusing to follow an unsafe model deletion quarantine")
    marker = quarantine / _MODEL_DELETE_MARKER
    if _model_path_is_link(marker) or not marker.is_file():
        raise OSError("model deletion quarantine has no safe ownership marker")
    for child in quarantine.iterdir():
        if child == marker:
            continue
        if _model_path_is_link(child):
            raise OSError("refusing to follow a link in model deletion quarantine")
        if child.is_dir():
            try:
                _ensure_model_tree_link_free(child)
            except ValueError as exc:
                raise OSError("refusing to follow a link in model deletion quarantine") from exc
            shutil.rmtree(child)
        elif child.is_file():
            child.unlink()
        else:
            raise OSError("model deletion quarantine contains an unsupported entry")
    # Remove the ownership marker last. If payload cleanup fails partway
    # through, the next recovery pass can still determine whether to restore
    # the remaining files or finish deleting them.
    marker.unlink(missing_ok=True)
    quarantine.rmdir()
    with suppress(OSError):
        quarantine.parent.rmdir()


def recover_model_delete_quarantines(
    session: Session,
    model_root: Path,
    *,
    strict: bool = False,
) -> None:
    parent = model_root / _MODEL_DELETE_QUARANTINE
    if not parent.exists():
        return
    if _model_path_is_link(parent) or not parent.is_dir():
        raise ValueError("model deletion quarantine is not a safe directory")
    for quarantine in list(parent.iterdir()):
        if not quarantine.name.startswith("delete_"):
            continue
        try:
            if not quarantine.is_dir() or _model_path_is_link(quarantine):
                raise ValueError("model deletion quarantine contains a filesystem link")
            marker = quarantine / _MODEL_DELETE_MARKER
            if not marker.exists():
                if not any(quarantine.iterdir()):
                    quarantine.rmdir()
                    continue
                raise ValueError("model deletion quarantine has no ownership marker")
            if _model_path_is_link(marker) or not marker.is_file():
                raise ValueError("model deletion quarantine contains an unsafe marker")
            marker_model_id = marker.read_text(encoding="utf-8")
            if (
                not marker_model_id
                or marker_model_id != marker_model_id.strip()
                or len(marker_model_id) > 200
            ):
                raise ValueError("model deletion quarantine has an invalid owner")
            install: ModelInstall | ModelAssetInstall | None = session.get(
                ModelInstall, marker_model_id
            )
            if install is None:
                install = session.get(ModelAssetInstall, marker_model_id)
            if install is None:
                _finalize_model_quarantine(quarantine)
                continue
            path = _managed_model_path(model_root, install.local_path)
            if path is None:
                raise ValueError("model deletion quarantine belongs to an external model")
            _restore_model_quarantine(quarantine, path)
        except (OSError, UnicodeError, ValueError):
            if strict:
                raise
            logger.warning(
                "Could not reconcile model deletion quarantine %s",
                quarantine,
                exc_info=True,
            )
    with suppress(OSError):
        parent.rmdir()


def _restore_model_quarantine(quarantine: Path, path: Path) -> None:
    moves: list[tuple[Path, Path]] = []
    payload = quarantine / "payload"
    if payload.exists():
        _ensure_model_tree_link_free(payload)
        moves.append((payload, path))
    files = quarantine / "files"
    if files.exists():
        if _model_path_is_link(files) or not files.is_dir():
            raise ValueError("model deletion quarantine contains an unsafe files directory")
        _ensure_model_tree_link_free(files)
        for staged in files.rglob("*"):
            if _model_path_is_link(staged):
                raise ValueError("model deletion quarantine contains a filesystem link")
            if not staged.is_file():
                continue
            relative = staged.relative_to(files)
            original = _confined_manifest_file(
                path,
                PurePosixPath(*relative.parts),
            )
            moves.append((staged, original))
    try:
        _restore_model_moves(moves)
    except RuntimeError as exc:
        raise ValueError("model deletion recovery needs manual conflict resolution") from exc
    _finalize_model_quarantine(quarantine)
