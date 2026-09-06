from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import re
import secrets
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import IO, Any, Final

from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from .artifact_deletion_authority import (
    activate_artifact_deletion_proof,
    artifact_deletion_proof_references,
    mint_artifact_deletion_proof,
    record_artifact_deletion,
    restrict_artifact_deletion_proof,
)
from .artifact_library import (
    artifacts_naming,
    begin_artifact_write_fence,
    fenced_reference_snapshot,
    metadata_referrers,
    referenced_artifact_ids,
)
from .artifact_library_schema import ARTIFACT_METADATA_REFERENCE_KEYS
from .config import Settings
from .domain import ArtifactKind, MessageRole, PartType
from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntry,
    AnchoredEntryKind,
    create_entry,
    discard_entry,
    is_link_or_reparse,
    list_entries,
    open_child_directory,
    open_entry,
    remove_directory_entry,
    remove_entry,
    rename_entry,
    sync_directory,
)
from .models import (
    Artifact,
    ArtifactLibraryEntry,
    Message,
    MessagePart,
    ResponseRevision,
    ResponseRevisionPart,
)
from .subprocess_env import subprocess_environment

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD = re.compile(r"[0-9a-f]{2}")
_RESTORE_PARTIAL = re.compile(r"(?:[0-9a-f]{64}|\.[0-9a-f]{64}\.[^.]+)\.restore-partial")

#: How many entries the orphan sweep may enumerate in the store root.
#:
#: The root holds at most 256 shard directories, and beyond those only
#: temporaries left behind by an interrupted ingest or proxy encode. Those are
#: precisely what this pass deletes, so they accumulate only while it is not
#: running - and a ceiling near the primitive's own default would then refuse
#: the enumeration exactly when the backlog is largest. This one is far above
#: any plausible backlog while still bounding a single listing.
_ROOT_LISTING_LIMIT: Final = 65536
_STAGED_DELETION = re.compile(r"^(?P<digest>[0-9a-f]{64})\.[0-9a-f]{32}$")
_MAX_VIDEO_POSTER_BYTES = 16 * 1024 * 1024


def _is_temporary_name(name: str) -> bool:
    """True only for a name this store's own staging could have produced.

    `ingest_bytes` stages as `ingest-<hex>.tmp`, and the proxy encoder uses
    `mkstemp(prefix="video-proxy-", suffix=".mp4")`. Reading the shapes the
    store WRITES on the way back out means a pass can only ever delete
    something this store could have left behind.
    """

    return name.startswith("ingest-") or (name.startswith("video-proxy-") and name.endswith(".mp4"))


def _aged_file_size(entry: AnchoredEntry, cutoff: datetime) -> int | None:
    """The size of a plain file old enough to remove, or None to leave it.

    One answer for three different reasons - not an ordinary file, no
    measurement, or not old enough - because all three mean the same thing to
    a caller: leave it alone. Size and time are checked individually rather
    than through `has_metadata` so that the type narrows for `mypy`; the
    contract guarantees they are both present or both absent.
    """

    if entry.kind is not AnchoredEntryKind.FILE:
        return None
    size = entry.size_bytes
    modified = entry.modified_at
    if size is None or modified is None or modified > cutoff:
        return None
    return size


def _removed(anchor: AnchoredDirectory, name: str, *, counted: int, cutoff: datetime) -> bool:
    """Remove one entry, but only if the name still holds what was measured.

    `list_entries` takes kind, size and time from one record; `unlink` resolves
    the NAME again. Holding the parent stabilises the DIRECTORY and not the
    leaf, so a same-name replacement between the two would be deleted using the
    replaced entry's age and size.

    That is reachable rather than theoretical, and it is this store that
    reaches it. `_publish_under_its_digest` renames a freshly written file over
    its digest with `replace=True`, and the Artifact row is flushed AFTER. A
    pass that measured the old file could delete the newly published one and
    leave a row pointing at nothing.

    So the entry is reopened through the held parent and measured again from
    that descriptor rather than from the record. A publication replacement
    fails both tests: its size need not match what was counted, and its
    modification time is NOW rather than older than the cutoff.

    The descriptor is closed before the unlink because it has to be: Windows
    opens these without FILE_SHARE_DELETE, so holding it would refuse our own
    deletion. That leaves a window between the check and the unlink which no
    POSIX call can close - there is no unlink-by-inode - but nothing this store
    does can put an AGED entry into it, which is what the reachable path needs.

    A refusal is not an error and not a deletion. Counting it would report
    bytes that are still on disk, and raising would abandon everything the pass
    had already reclaimed.
    """

    try:
        descriptor = open_entry(anchor, name)
    except (AnchoredDirectoryError, OSError):
        return False
    if descriptor is None:
        return False
    try:
        measured = os.fstat(descriptor)
    except OSError:
        return False
    finally:
        with suppress(OSError):
            os.close(descriptor)
    if measured.st_size != counted:
        return False
    if datetime.fromtimestamp(measured.st_mtime, UTC) > cutoff:
        return False
    try:
        remove_entry(anchor, name)
    except (AnchoredDirectoryError, OSError):
        return False
    return True


# Automatic and manual cleanup share a count ceiling and deletion-phase clock.
# One required reference snapshot precedes that clock; the writer reservation
# includes its fixed cost too. The ceiling lets fast authorized deletions
# amortize that snapshot while the time budget still bounds slower deletion work.
RETENTION_BATCH_DELETIONS = 1000
RETENTION_BATCH_SECONDS = 2.0


@dataclass(frozen=True)
class RetentionCleanupSummary:
    marked_count: int
    pending_count: int
    removed_count: int
    reclaimed_bytes: int
    #: True when the pass stopped before examining every artifact - a deletion
    #: bound was reached or the caller asked it to stop - so another pass is
    #: needed before the sweep can be called complete.
    truncated: bool = False
    #: Rows inspected in this pass, including retained rows and the stopping boundary.
    examined_count: int = 0


class _DeletionBudget:
    """What a pass may still remove, and whether it may go on at all.

    The orphan-file walk is the part of a pass that has no rows to count, so a
    caller's bound and stop request had no purchase on it: a store whose aged
    backlog is files rather than rows got one unbounded batch. The budget is
    consulted before every removal and records, in ``truncated``, that the walk
    stopped short, so the caller knows another pass is owed.
    """

    def __init__(
        self,
        remaining: int | None,
        should_stop: Callable[[], bool] | None,
        *,
        report_removed: Callable[[], None] | None = None,
    ) -> None:
        self.report_removed = report_removed
        self.remaining = remaining
        self.should_stop = should_stop
        self.truncated = False

    def allow(self) -> bool:
        if self.should_stop is not None and self.should_stop():
            self.truncated = True
            return False
        if self.remaining is not None and self.remaining <= 0:
            self.truncated = True
            return False
        return True

    def spend(self) -> None:
        if self.remaining is not None:
            self.remaining -= 1
        if self.report_removed is not None:
            self.report_removed()


@dataclass(frozen=True)
class StagedArtifactFile:
    path: Path
    media_type: str
    original_name: str

    def discard(self) -> None:
        self.path.unlink(missing_ok=True)


def _path_follows_a_link(path: Path) -> bool:
    """True when resolving this path would traverse a link or reparse point.

    `.resolve()` also absolutizes and normalizes case, so a string comparison
    of requested versus resolved is not this question. Walk the named path and
    its parents with the existing inspection primitive instead.
    """

    cursor = path if path.is_absolute() else Path.cwd() / path
    while True:
        if is_link_or_reparse(
            cursor,
            missing="assume_regular",
            unreadable="assume_link",
        ):
            return True
        parent = cursor.parent
        if parent == cursor:
            return False
        cursor = parent


class ArtifactStore:
    def __init__(self, settings: Settings, *, root: Path | None = None) -> None:
        requested = root or settings.artifact_dir
        self.requested_root = requested
        self.root_followed_a_link = _path_follows_a_link(requested)
        self.root = requested.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._verified_files: dict[Path, tuple[int, int]] = {}

    def _destination(self, digest: str) -> Path:
        if not _SHA256.fullmatch(digest):
            raise ValueError("invalid artifact digest")
        return self.root / digest[:2] / digest[2:4] / digest

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def resolve(self, artifact: Artifact) -> Path:
        if artifact.id != f"sha256:{artifact.sha256}" or not _SHA256.fullmatch(artifact.sha256):
            raise ValueError("artifact identity is invalid")
        expected_relative = PurePosixPath(
            artifact.sha256[:2],
            artifact.sha256[2:4],
            artifact.sha256,
        )
        if artifact.relative_path != expected_relative.as_posix():
            raise ValueError("artifact path is not canonical")
        candidate = self.root.joinpath(*expected_relative.parts)
        cursor = self.root
        for part in expected_relative.parts:
            cursor /= part
            if self._is_link(cursor):
                raise ValueError("artifact path uses a filesystem link")
        path = candidate.resolve()
        if self.root not in path.parents:
            raise ValueError("artifact path escapes store")
        return path

    def verified_path(self, artifact: Artifact) -> Path:
        path = self.resolve(artifact)
        try:
            stat_result = path.stat()
        except OSError as exc:
            raise FileNotFoundError(path) from exc
        if not path.is_file() or stat_result.st_size != artifact.size_bytes:
            raise ValueError("artifact file size does not match its record")
        cached = self._verified_files.get(path)
        fingerprint = (stat_result.st_size, stat_result.st_mtime_ns)
        if cached != fingerprint:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != artifact.sha256:
                raise ValueError("artifact file checksum does not match its record")
            self._verified_files[path] = fingerprint
        return path

    def verified_bytes(self, artifact: Artifact, *, maximum_bytes: int) -> bytes:
        """Read bounded artifact bytes from the descriptor that is verified.

        A pathname check followed by a later pathname read does not bind both
        operations to one file. This walk holds the store and both digest
        shards, opens the digest entry through the held leaf, and derives the
        size, digest, and returned bytes from that one descriptor. The ordinary
        verified-path cache is intentionally irrelevant to an authorization
        decision about exact bytes.
        """

        if maximum_bytes < 0:
            raise ValueError("maximum artifact read size is invalid")
        digest_value = artifact.sha256
        if artifact.id != f"sha256:{digest_value}" or not _SHA256.fullmatch(digest_value):
            raise ValueError("artifact identity is invalid")
        expected_relative = PurePosixPath(
            digest_value[:2],
            digest_value[2:4],
            digest_value,
        ).as_posix()
        if artifact.relative_path != expected_relative:
            raise ValueError("artifact path is not canonical")

        descriptor: int | None = None
        try:
            with (
                AnchoredDirectory(self.root) as root,
                open_child_directory(root, digest_value[:2]) as first,
                open_child_directory(first, digest_value[2:4]) as second,
            ):
                descriptor = open_entry(second, digest_value)
                if descriptor is None:
                    raise FileNotFoundError("artifact file is missing")
                measured = os.fstat(descriptor)
                if not stat.S_ISREG(measured.st_mode):
                    raise ValueError("artifact entry is not a regular file")
                if measured.st_size != artifact.size_bytes:
                    raise ValueError("artifact file size does not match its record")
                if measured.st_size > maximum_bytes:
                    raise ValueError("artifact is larger than this read allows")

                content = bytearray()
                content_digest = hashlib.sha256()
                with os.fdopen(descriptor, "rb") as source:
                    descriptor = None
                    while chunk := source.read(1024 * 1024):
                        content.extend(chunk)
                        if len(content) > measured.st_size:
                            raise ValueError("artifact file size does not match its record")
                        content_digest.update(chunk)
        except AnchoredDirectoryError as exc:
            raise ValueError("artifact path could not be held for reading") from exc
        finally:
            if descriptor is not None:
                with suppress(OSError):
                    os.close(descriptor)

        if len(content) != artifact.size_bytes:
            raise ValueError("artifact file size does not match its record")
        if content_digest.hexdigest() != digest_value:
            raise ValueError("artifact file checksum does not match its record")
        return bytes(content)

    def delivery_metadata(self, artifact: Artifact) -> tuple[Path, str, str]:
        path = self.verified_path(artifact)
        detected = self._detect_media_type(path)
        if detected is None:
            return path, "application/octet-stream", "attachment"
        return path, detected, "inline"

    def ingest_path(
        self,
        session: Session,
        source: Path,
        *,
        kind: ArtifactKind,
        media_type: str | None = None,
        original_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Artifact:
        source = source.resolve(strict=True)
        with source.open("rb") as handle:
            return self.ingest_stream(
                session,
                handle,
                kind=kind,
                media_type=media_type or mimetypes.guess_type(source.name)[0],
                original_name=original_name or source.name,
                metadata=metadata,
            )

    def ingest_bytes(
        self,
        session: Session,
        content: bytes,
        *,
        kind: ArtifactKind,
        media_type: str,
        original_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Artifact:
        with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as handle:
            handle.write(content)
            handle.seek(0)
            return self.ingest_stream(
                session,
                handle,
                kind=kind,
                media_type=media_type,
                original_name=original_name,
                metadata=metadata,
            )

    def ingest_stream(
        self,
        session: Session,
        source: IO[bytes],
        *,
        kind: ArtifactKind,
        media_type: str | None,
        original_name: str | None,
        metadata: dict[str, object] | None,
    ) -> Artifact:
        sha256, size = self._publish_under_its_digest(source, session)
        destination_path = self._destination(sha256)
        # The verified-file cache is NOT primed here, and that is deliberate.
        # _remember_verified stats by pathname, so priming it immediately after
        # an anchored publish would record a fingerprint taken through a name
        # that was just resolved by a handle - and verified_path skips hashing
        # whenever the fingerprint matches. An entry swapped in that narrow
        # window would then be trusted without ever being hashed. Leaving the
        # cache cold costs one hash on the first read and closes that.

        existing = session.scalar(select(Artifact).where(Artifact.sha256 == sha256))
        if existing:
            changed = False
            if existing.size_bytes != size:
                existing.size_bytes = size
                changed = True
            sanitized_name = self._safe_original_name(existing.original_name)
            if existing.original_name != sanitized_name:
                existing.original_name = sanitized_name
                changed = True
            sanitized_media_type = self._safe_media_type(existing.media_type)
            if existing.media_type != sanitized_media_type:
                existing.media_type = sanitized_media_type
                changed = True
            if existing.metadata_json.get("temporary_preview") and not (metadata or {}).get(
                "temporary_preview"
            ):
                existing.kind = kind.value
                existing.media_type = self._safe_media_type(media_type or existing.media_type)
                existing.original_name = (
                    self._safe_original_name(original_name) or existing.original_name
                )
                existing.metadata_json = metadata or {}
                changed = True
            if changed:
                session.flush()
            return existing

        artifact = Artifact(
            id=f"sha256:{sha256}",
            sha256=sha256,
            kind=kind.value,
            media_type=self._safe_media_type(media_type),
            size_bytes=size,
            relative_path=self._relative(destination_path),
            original_name=self._safe_original_name(original_name),
            metadata_json=metadata or {},
        )
        session.add(artifact)
        session.flush()
        return artifact

    def _publish_under_its_digest(self, source: IO[bytes], session: Session) -> tuple[str, int]:
        """Consume the stream into the store and publish it under its digest.

        Everything from the store root down is HELD. The two digest directories
        are created through their held parent, so a link anywhere in that
        ancestry refuses before a byte is written into it, and publication is a
        rename between two held directories rather than an operation on a path
        that was inspected a moment ago.

        Measured before this existed: with a junction planted at the second
        digest component, the previous code wrote the artifact straight through
        it and out of the store, because `_is_link` was applied to the final
        name only and the two intermediate directories were created by this
        method itself with `mkdir(parents=True, exist_ok=True)`.

        The digest is not known until the bytes have been consumed, so staging
        necessarily precedes the destination and publication crosses
        directories. That is what `rename_entry(..., into=...)` is for.

        Staging uses an unpredictable name created EXCLUSIVELY: a fixed sibling
        can be planted in advance, and the plant would then be what gets
        published under a digest it does not have.
        """

        try:
            root_anchor = AnchoredDirectory(self.root, create=True)
        except AnchoredDirectoryError as exc:
            raise ValueError("artifact store root uses a filesystem link") from exc

        with root_anchor:
            staging = f"ingest-{secrets.token_hex(8)}.tmp"
            descriptor = create_entry(root_anchor, staging)
            digest = hashlib.sha256()
            size = 0
            try:
                with os.fdopen(descriptor, "wb") as sink:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        sink.write(chunk)
                        size += len(chunk)
                    sink.flush()
                    os.fsync(sink.fileno())
                sha256 = digest.hexdigest()
                # The writer reservation is taken HERE: after the bytes are
                # staged and before they are published, and it is held through
                # the caller's row work.
                #
                # The sweep takes the same reservation, so it cannot run between
                # the moment these bytes appear under their digest and the moment
                # a row exists to protect them. Without it the deduplication path
                # can return having written nothing at all - no insert, no update,
                # no fence - so a live request holds nothing the sweep can see,
                # and the sweep deletes the row and the bytes underneath it.
                #
                # Taken here rather than at the top of the ingest, because the
                # streaming above is the expensive part and holding SQLite's
                # single writer slot for the length of an upload would block
                # every other write for as long as the upload takes.
                begin_artifact_write_fence(session)
                first = open_child_directory(root_anchor, sha256[:2], create=True)
                try:
                    second = open_child_directory(first, sha256[2:4], create=True)
                    try:
                        # Content-addressed, so replacing is idempotent: the
                        # destination name can only ever hold these bytes.
                        rename_entry(root_anchor, staging, sha256, into=second, replace=True)
                        sync_directory(second)
                    finally:
                        second.close()
                finally:
                    first.close()
            except AnchoredDirectoryError as exc:
                with suppress(AnchoredDirectoryError):
                    discard_entry(root_anchor, staging)
                raise ValueError("artifact destination uses a filesystem link") from exc
            except BaseException:
                with suppress(AnchoredDirectoryError):
                    discard_entry(root_anchor, staging)
                raise
            return sha256, size

    def export_copy(self, artifact: Artifact, destination: Path) -> Path:
        source = self.resolve(artifact)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    async def video_poster(self, artifact: Artifact) -> bytes | None:
        executable = shutil.which("ffmpeg")
        if not executable or not artifact.media_type.startswith("video/"):
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self.resolve(artifact)),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                env=subprocess_environment(),
            )
            stdout = await self._bounded_stdout(
                process,
                maximum_bytes=_MAX_VIDEO_POSTER_BYTES,
                timeout_seconds=30,
            )
        except asyncio.CancelledError:
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except (OSError, TimeoutError, ValueError):
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            return None
        return stdout if process.returncode == 0 and stdout else None

    async def browser_video_proxy(self, artifact: Artifact) -> StagedArtifactFile | None:
        if artifact.media_type in {"video/mp4", "video/webm"}:
            return None
        executable = shutil.which("ffmpeg")
        if not executable:
            return None
        fd, temporary_name = tempfile.mkstemp(prefix="video-proxy-", suffix=".mp4", dir=self.root)
        os.close(fd)
        temporary = Path(temporary_name)
        retained = False
        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(self.resolve(artifact)),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(temporary),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=subprocess_environment(),
            )
            await asyncio.wait_for(process.wait(), timeout=600)
            if process.returncode or not temporary.is_file() or temporary.stat().st_size == 0:
                return None
            retained = True
            return StagedArtifactFile(
                path=temporary,
                media_type="video/mp4",
                original_name=f"{artifact.original_name or 'video'}.proxy.mp4",
            )
        except asyncio.CancelledError:
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except (OSError, TimeoutError):
            if "process" in locals() and process.returncode is None:
                process.kill()
                await process.wait()
            return None
        finally:
            if not retained:
                temporary.unlink(missing_ok=True)

    @staticmethod
    async def _bounded_stdout(
        process: asyncio.subprocess.Process,
        *,
        maximum_bytes: int,
        timeout_seconds: float,
    ) -> bytes:
        stdout = process.stdout
        if stdout is None:
            raise ValueError("media process did not expose output")

        async def collect() -> bytes:
            content = bytearray()
            while chunk := await stdout.read(64 * 1024):
                content.extend(chunk)
                if len(content) > maximum_bytes:
                    raise ValueError("media process output exceeded its configured limit")
            await process.wait()
            return bytes(content)

        return await asyncio.wait_for(collect(), timeout=timeout_seconds)

    def delete_temporary_preview(self, session: Session, artifact_id: str) -> bool:
        artifact = session.get(Artifact, artifact_id)
        if not artifact or not artifact.metadata_json.get("temporary_preview"):
            return False
        references = (
            session.scalar(
                select(func.count(MessagePart.id)).where(MessagePart.artifact_id == artifact_id)
            )
            or 0
        )
        references += (
            session.scalar(
                select(func.count(ResponseRevisionPart.id)).where(
                    ResponseRevisionPart.artifact_id == artifact_id
                )
            )
            or 0
        )
        if references:
            return False
        # This returns bool, and `orchestrator.recover_interrupted` calls it from
        # the `orchestrator-recovery` startup stage, which `_startup_stage` wraps
        # in try/finally with no except. A ValueError here would therefore
        # propagate out of lifespan and stop the application starting, so a
        # preview something still retains is declined rather than raised.
        #
        # The parts counted above are a fast path, not the whole answer: the walk
        # follows metadata links too, and `_delete_artifact` raises on exactly
        # this set. Taking the fence first is what makes the snapshot safe to
        # hand back, so the check costs one walk rather than two.
        begin_artifact_write_fence(session)
        retained = self.referenced_artifact_ids(session, for_deletion=True)
        if artifact.id in retained:
            return False
        try:
            self._delete_artifact(session, artifact, references=retained)
        except OSError as exc:
            if getattr(exc, "winerror", None) in {32, 33}:
                return False
            raise
        return True

    @staticmethod
    def referenced_artifact_ids(
        session: Session, *, for_deletion: bool = False
    ) -> AbstractSet[str]:
        return referenced_artifact_ids(session, for_deletion=for_deletion)

    def cleanup_retention(
        self,
        session: Session,
        *,
        retention_days: int,
        temporary_hours: int,
        dry_run: bool,
        now: datetime | None = None,
        max_deletions: int | None = None,
        should_stop: Callable[[], bool] | None = None,
        report_phase: Callable[[str], None] | None = None,
    ) -> RetentionCleanupSummary:
        """Remove expired unretained artifacts within the caller's stop/budget.

        Hold the writer reservation while deriving one reference snapshot and
        deleting eligible rows. Each deletion receives exact authority from
        that snapshot and advances it only after the flush succeeds. A stop is
        observed immediately before an actual deletion. Metadata updates survive
        truncated passes; orphan cleanup runs only after a complete row pass.
        Callers may commit each bounded pass to preserve completed work.
        """

        phase: Callable[[str], None] = report_phase or (lambda _name: None)
        current = now or datetime.now(UTC)
        if not dry_run:
            phase("acquire-writer")
            begin_artifact_write_fence(session)
            phase("writer-acquired")
            phase("recover-staged-deletions")
            self._recover_staged_deletions(session)
        phase("reference-snapshot")
        referenced = self.referenced_artifact_ids(session, for_deletion=not dry_run)
        examined_count = 0
        marked_count = 0
        pending_count = 0
        removed_count = 0
        reclaimed_bytes = 0
        truncated = False
        phase("load-artifact-rows")
        ordered = session.scalars(select(Artifact).order_by(Artifact.created_at)).all()
        # Who names whom, from rows already loaded above: no extra query, and
        # bounded by this sweep's own working set rather than by the store.
        phase("metadata-references")
        referrers = metadata_referrers(ordered)
        removed_ids: set[str] = set()
        metadata_updates: list[tuple[Artifact, dict[str, Any]]] = []
        phase("examine-artifacts")
        for artifact in ordered:
            examined_count += 1
            metadata = dict(artifact.metadata_json)
            # A favorite pins against the automatic sweep exactly like a live
            # reference: never marked, never removed here. Explicit deletion
            # is untouched - a user deleting a favorite means it.
            if artifact.id in referenced or artifact.favorite:
                if "unreferenced_at" in metadata and not dry_run:
                    metadata.pop("unreferenced_at", None)
                    metadata_updates.append((artifact, metadata))
                continue
            temporary = bool(metadata.get("temporary_preview") or metadata.get("intermediate"))
            age = current - self._aware(artifact.created_at)
            eligible = temporary and age >= timedelta(hours=temporary_hours)
            unreferenced_at = self._metadata_datetime(metadata.get("unreferenced_at"))
            if not temporary and unreferenced_at:
                eligible = current - unreferenced_at >= timedelta(days=retention_days)
            if eligible and any(
                naming not in removed_ids for naming in referrers.get(artifact.id, ())
            ):
                # The delete trigger refuses while a SURVIVING artifact names
                # this one, so attempting it aborts the pass and everything
                # after it. A referrer already removed in this pass no longer
                # counts, which is what lets a video and its poster go together
                # rather than one per pass.
                eligible = False
            if eligible:
                if should_stop is not None and should_stop():
                    truncated = True
                    break
                if max_deletions is not None and removed_count >= max_deletions:
                    truncated = True
                    break
                if not dry_run:
                    phase("delete-artifact")
                    proof = mint_artifact_deletion_proof(session, {artifact.id}, referenced)
                    self._delete_artifact(session, artifact, proof=proof)
                    phase("examine-artifacts")
                removed_count += 1
                reclaimed_bytes += artifact.size_bytes
                removed_ids.add(artifact.id)
                continue
            if not temporary:
                pending_count += 1
                if not unreferenced_at:
                    marked_count += 1
                    if not dry_run:
                        metadata["unreferenced_at"] = current.isoformat()
                        metadata_updates.append((artifact, metadata))
        if not dry_run:
            phase("flush-metadata")
            for artifact, metadata in metadata_updates:
                artifact.metadata_json = metadata
            # Marking `unreferenced_at` makes an artifact dirty, and if its
            # metadata names a poster then _pending_json_reference_ids is
            # non-empty here, so the listener walks the graph again on this final
            # flush. That is O(1) rather than the per-deletion walk, but "once per
            # sweep" is only true if this flush lends the snapshot too.
            with fenced_reference_snapshot(session, referenced):
                session.flush()
        orphan_count = 0
        orphan_bytes = 0
        if not truncated:
            # The orphan walk spends what the row loop left of the bound and
            # answers to the same stop request, so the final batch of a pass
            # is bounded exactly like the ones before it.
            remaining = None if max_deletions is None else max(max_deletions - removed_count, 0)
            budget = _DeletionBudget(
                remaining, should_stop, report_removed=lambda: phase("orphan-file-removed")
            )
            if budget.allow():
                phase("cleanup-orphan-files")
                orphan_count, orphan_bytes = self._cleanup_orphan_files(
                    session,
                    current=current,
                    temporary_hours=temporary_hours,
                    dry_run=dry_run,
                    budget=budget,
                )
            truncated = budget.truncated
        return RetentionCleanupSummary(
            examined_count=examined_count,
            marked_count=marked_count,
            pending_count=pending_count,
            removed_count=removed_count + orphan_count,
            reclaimed_bytes=reclaimed_bytes + orphan_bytes,
            truncated=truncated,
        )

    def delete_library_artifact(
        self,
        session: Session,
        artifact: Artifact,
    ) -> tuple[int, int, int]:
        begin_artifact_write_fence(session)
        if artifact.kind not in {ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value}:
            raise ValueError("only image and video library artifacts can be deleted directly")

        entry_id = session.scalar(
            select(ArtifactLibraryEntry.id).where(ArtifactLibraryEntry.artifact_id == artifact.id)
        )
        if entry_id:
            raise ValueError("This media item is retained by its Media Library membership.")

        linked_ids = {
            linked_id
            for key in ARTIFACT_METADATA_REFERENCE_KEYS
            if isinstance((linked_id := artifact.metadata_json.get(key)), str)
        }
        parts = session.scalars(
            select(MessagePart).where(MessagePart.artifact_id == artifact.id)
        ).all()
        revision_parts = session.scalars(
            select(ResponseRevisionPart).where(ResponseRevisionPart.artifact_id == artifact.id)
        ).all()
        for part in parts:
            part.artifact_id = None
        for revision_part in revision_parts:
            revision_part.artifact_id = None
        session.flush()

        removed_count = 1
        reclaimed_bytes = artifact.size_bytes
        self._delete_artifact(session, artifact)

        referenced = self.referenced_artifact_ids(session)
        for linked_id in linked_ids:
            linked = session.get(Artifact, linked_id)
            if not linked or linked.id in referenced:
                continue
            # The parent is already deleted above, so anything still naming this
            # poster or proxy is a DIFFERENT artifact - ingest deduplicates on
            # sha256, so two videos with identical extracted frames share one
            # poster row. referenced_artifact_ids does not see that referrer when
            # it is itself unreferenced, but the delete trigger does, and refuses.
            if artifacts_naming(session, linked.id):
                continue
            removed_count += 1
            reclaimed_bytes += linked.size_bytes
            self._delete_artifact(session, linked)
        # Revision parts are internal snapshots of the same user-visible message
        # reference. Clear them as well, but do not inflate the public reference
        # count with implementation details.
        return len(parts), removed_count, reclaimed_bytes

    def generated_media_artifact_ids_for_chat(
        self,
        session: Session,
        chat_id: str,
    ) -> tuple[str, ...]:
        artifacts_by_id = {
            artifact.id: artifact
            for artifact in session.scalars(
                select(Artifact)
                .join(MessagePart, MessagePart.artifact_id == Artifact.id)
                .join(Message, Message.id == MessagePart.message_id)
                .where(
                    Message.chat_id == chat_id,
                    Message.role == MessageRole.ASSISTANT.value,
                    MessagePart.type.in_((PartType.IMAGE.value, PartType.VIDEO.value)),
                    Artifact.kind.in_((ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value)),
                )
                .order_by(Artifact.created_at)
            )
            .unique()
            .all()
        }
        revision_artifacts = session.scalars(
            select(Artifact)
            .join(
                ResponseRevisionPart,
                ResponseRevisionPart.artifact_id == Artifact.id,
            )
            .join(
                ResponseRevision,
                ResponseRevision.id == ResponseRevisionPart.response_revision_id,
            )
            .join(Message, Message.id == ResponseRevision.message_id)
            .where(
                Message.chat_id == chat_id,
                ResponseRevisionPart.type.in_((PartType.IMAGE.value, PartType.VIDEO.value)),
                Artifact.kind.in_((ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value)),
            )
            .order_by(Artifact.created_at)
        ).unique()
        for artifact in revision_artifacts:
            artifacts_by_id[artifact.id] = artifact
        return tuple(
            artifact.id
            for artifact in sorted(artifacts_by_id.values(), key=lambda item: item.created_at)
        )

    def delete_generated_media_artifacts(
        self,
        session: Session,
        artifact_ids: tuple[str, ...],
    ) -> int:
        """Delete a removed chat's now-unreferenced generated media.

        Callers snapshot the ids before deleting the chat, then flush the chat
        deletion before entering here. The canonical reference graph therefore
        protects every surviving consumer without treating the deleted chat's
        own Run and Job rows as external retention.
        """

        removed = 0
        for artifact_id in artifact_ids:
            artifact = session.get(Artifact, artifact_id)
            if not artifact or artifact.kind not in {
                ArtifactKind.IMAGE.value,
                ArtifactKind.VIDEO.value,
            }:
                continue
            if session.scalar(
                select(ArtifactLibraryEntry.id).where(
                    ArtifactLibraryEntry.artifact_id == artifact.id
                )
            ):
                continue
            if artifact.id in self.referenced_artifact_ids(session):
                continue
            _references, removed_count, _reclaimed_bytes = self.delete_library_artifact(
                session, artifact
            )
            removed += removed_count
        return removed

    def _delete_artifact(
        self,
        session: Session,
        artifact: Artifact,
        *,
        references: AbstractSet[str] | None = None,
        proof: object | None = None,
    ) -> None:
        """Delete one unretained artifact under the current writer reservation.

        A supplied proof must match this artifact and the unchanged reference
        snapshot in the same session, connection and transaction. Without one,
        derive that evidence here. Stage existing bytes before flushing and
        restore them if the flush fails; commit finalizes their removal.
        """

        begin_artifact_write_fence(session)
        if references is not None and proof is not None:
            raise ValueError("artifact deletion accepts one authority source")
        if proof is None:
            known = (
                self.referenced_artifact_ids(session, for_deletion=True)
                if references is None
                else references
            )
        else:
            selected_proof = restrict_artifact_deletion_proof(session, proof, {artifact.id})
            known = artifact_deletion_proof_references(selected_proof)
        if artifact.id in known:
            raise ValueError("This artifact is still retained.")
        if proof is None:
            selected_proof = mint_artifact_deletion_proof(session, {artifact.id}, known)
        try:
            path = self.resolve(artifact)
        except ValueError:
            # Invalid metadata must never redirect deletion to another file.
            session.delete(artifact)
            with activate_artifact_deletion_proof(session, selected_proof):
                session.flush()
                record_artifact_deletion(session, selected_proof)
            return
        staged: Path | None = None
        if path.exists():
            trash = self.root / ".delete-pending"
            if self._is_link(trash):
                raise ValueError("artifact deletion staging uses a filesystem link")
            trash.mkdir(parents=True, exist_ok=True)
            if not trash.is_dir() or trash.resolve().parent != self.root:
                raise ValueError("artifact deletion staging escapes the store")
            staged = trash / f"{artifact.sha256}.{uuid.uuid4().hex}"
            os.replace(path, staged)
            self._verified_files.pop(path, None)
        try:
            session.delete(artifact)
            with activate_artifact_deletion_proof(session, selected_proof):
                session.flush()
                record_artifact_deletion(session, selected_proof)
        except Exception:
            if staged is not None:
                self._restore_staged_file(staged, path)
            raise
        if staged is not None:
            self._register_staged_deletion(session, staged, path)

    def _register_staged_deletion(
        self,
        session: Session,
        staged: Path,
        original: Path,
    ) -> None:
        def finalize(_session: Session) -> None:
            with suppress(OSError):
                staged.unlink(missing_ok=True)
            self._prune_empty_parents(original)
            with suppress(OSError):
                staged.parent.rmdir()

        def restore(_session: Session) -> None:
            self._restore_staged_file(staged, original)

        event.listen(session, "after_commit", finalize, once=True)
        event.listen(session, "after_rollback", restore, once=True)

    def _restore_staged_file(self, staged: Path, original: Path) -> None:
        if not staged.exists():
            return
        original.parent.mkdir(parents=True, exist_ok=True)
        if original.exists():
            staged.unlink(missing_ok=True)
        else:
            os.replace(staged, original)
            self._remember_verified(original)

    def _recover_staged_deletions(self, session: Session) -> None:
        trash = self.root / ".delete-pending"
        if not trash.is_dir() or self._is_link(trash):
            return
        artifacts_by_sha = {
            artifact.sha256: artifact for artifact in session.scalars(select(Artifact)).all()
        }
        for staged in trash.iterdir():
            match = _STAGED_DELETION.fullmatch(staged.name)
            if not match or (not staged.is_file() and not staged.is_symlink()):
                continue
            if self._is_link(staged):
                staged.unlink(missing_ok=True)
                continue
            artifact = artifacts_by_sha.get(match.group("digest"))
            if artifact is None:
                staged.unlink(missing_ok=True)
                continue
            try:
                original = self.resolve(artifact)
            except ValueError:
                continue
            self._restore_staged_file(staged, original)
        with suppress(OSError):
            trash.rmdir()

    def _cleanup_orphan_files(
        self,
        session: Session,
        *,
        current: datetime,
        temporary_hours: int,
        dry_run: bool,
        budget: _DeletionBudget | None = None,
    ) -> tuple[int, int]:
        """Remove aged temporaries and unindexed files through held directories.

        Every candidate used to be resolved by name four more times after
        `iterdir` had already named it - `is_file`, a link check, a `stat` and
        an `unlink` - and each of those reopened the window between deciding
        and deleting. A name that was an ordinary file when it was checked
        could be a link by the time it was unlinked, and the unlink would have
        followed it out of the store.

        Now the root is held for the whole pass and each shard is opened
        through the level above it, so no directory on the way to a candidate
        can be swapped while the walk is inside it, and the kind, the size and
        the age all come from the record that named the entry.

        An entry whose age could not be established is skipped. This pass
        deletes by age, and a measurement it does not have is not a
        measurement it may assume.

        A root that refuses - a link where the store should be, or more entries
        than one enumeration may report - prunes nothing rather than pruning
        something else. What that leaves unsaid is the gap the store root
        already has: the refusal is not reported to anyone.
        """

        indexed = {artifact.relative_path for artifact in session.scalars(select(Artifact)).all()}
        cutoff = current - timedelta(hours=temporary_hours)
        allowance = budget or _DeletionBudget(None, None)
        try:
            with AnchoredDirectory(self.root) as anchor:
                return self._sweep_orphans(
                    anchor, indexed=indexed, cutoff=cutoff, dry_run=dry_run, budget=allowance
                )
        except (AnchoredDirectoryError, OSError):
            return 0, 0

    def _sweep_orphans(
        self,
        anchor: AnchoredDirectory,
        *,
        indexed: set[str],
        cutoff: datetime,
        dry_run: bool,
        budget: _DeletionBudget,
    ) -> tuple[int, int]:
        """One enumeration of the held root, read twice for its two jobs."""

        removed_count = 0
        reclaimed_bytes = 0
        # Poll while the native listing is collected, not only after it has
        # materialized. A stop also marks the shared budget truncated, so
        # callers retain completed counts and leave unfinished shards intact.
        entries = list_entries(
            anchor, limit=_ROOT_LISTING_LIMIT, should_stop=lambda: not budget.allow()
        )
        for entry in entries:
            if not _is_temporary_name(entry.name):
                continue
            size = _aged_file_size(entry, cutoff)
            if size is None:
                continue
            if not budget.allow():
                return removed_count, reclaimed_bytes
            if not dry_run and not _removed(anchor, entry.name, counted=size, cutoff=cutoff):
                continue
            budget.spend()
            removed_count += 1
            reclaimed_bytes += size
        for entry in entries:
            if entry.kind is not AnchoredEntryKind.DIRECTORY or not _SHARD.fullmatch(entry.name):
                continue
            if budget.truncated:
                return removed_count, reclaimed_bytes
            count, reclaimed = self._sweep_first_shard(
                anchor, entry.name, indexed=indexed, cutoff=cutoff, dry_run=dry_run, budget=budget
            )
            removed_count += count
            reclaimed_bytes += reclaimed
        return removed_count, reclaimed_bytes

    def _sweep_first_shard(
        self,
        anchor: AnchoredDirectory,
        first: str,
        *,
        indexed: set[str],
        cutoff: datetime,
        dry_run: bool,
        budget: _DeletionBudget,
    ) -> tuple[int, int]:
        """Sweep one first-level shard, then drop it if this pass emptied it.

        A shard is dropped only when the walk through it ran to the end; a
        walk cut short by the budget may have left entries it never reached.
        """

        removed_count = 0
        reclaimed_bytes = 0
        try:
            with open_child_directory(anchor, first) as held:
                for entry in list_entries(held, should_stop=lambda: not budget.allow()):
                    if entry.kind is not AnchoredEntryKind.DIRECTORY or not _SHARD.fullmatch(
                        entry.name
                    ):
                        continue
                    if budget.truncated:
                        return removed_count, reclaimed_bytes
                    count, reclaimed = self._sweep_second_shard(
                        held,
                        first,
                        entry.name,
                        indexed=indexed,
                        cutoff=cutoff,
                        dry_run=dry_run,
                        budget=budget,
                    )
                    removed_count += count
                    reclaimed_bytes += reclaimed
        except (AnchoredDirectoryError, OSError):
            return removed_count, reclaimed_bytes
        if removed_count and not dry_run and not budget.truncated:
            with suppress(AnchoredDirectoryError, OSError):
                remove_directory_entry(anchor, first)
        return removed_count, reclaimed_bytes

    def _sweep_second_shard(
        self,
        parent: AnchoredDirectory,
        first: str,
        second: str,
        *,
        indexed: set[str],
        cutoff: datetime,
        dry_run: bool,
        budget: _DeletionBudget,
    ) -> tuple[int, int]:
        """Sweep one leaf shard, then drop it if this pass emptied it.

        The shard is released before it is removed. A held directory can be
        neither renamed nor deleted, which is the property the rest of this
        walk relies on and would otherwise trip over here. A shard the budget
        cut short is kept, since entries after the cut were never examined.
        """

        removed_count = 0
        reclaimed_bytes = 0
        try:
            with open_child_directory(parent, second) as held:
                for entry in list_entries(held, should_stop=lambda: not budget.allow()):
                    size = self._removable_size(
                        entry, first, second, indexed=indexed, cutoff=cutoff
                    )
                    if size is None:
                        continue
                    if not budget.allow():
                        return removed_count, reclaimed_bytes
                    if not dry_run and not _removed(held, entry.name, counted=size, cutoff=cutoff):
                        continue
                    budget.spend()
                    removed_count += 1
                    reclaimed_bytes += size
        except (AnchoredDirectoryError, OSError):
            return removed_count, reclaimed_bytes
        if removed_count and not dry_run and not budget.truncated:
            with suppress(AnchoredDirectoryError, OSError):
                remove_directory_entry(parent, second)
        return removed_count, reclaimed_bytes

    @staticmethod
    def _removable_size(
        entry: AnchoredEntry,
        first: str,
        second: str,
        *,
        indexed: set[str],
        cutoff: datetime,
    ) -> int | None:
        """How many bytes removing this entry reclaims, or None to leave it.

        A restore partial is judged on age alone: it is this store's own
        scratch file and no row ever refers to it. Anything else has to be a
        canonical file SHARDED WHERE IT WAS FOUND - a digest whose first four
        characters are the two directories containing it - that no artifact row
        still points at. A name that does not match the layout it was found in
        was not written by `_destination`, and this pass does not delete files
        it cannot account for.
        """

        if _RESTORE_PARTIAL.fullmatch(entry.name):
            return _aged_file_size(entry, cutoff)
        if (
            not _SHA256.fullmatch(entry.name)
            or entry.name[:2] != first
            or entry.name[2:4] != second
            or f"{first}/{second}/{entry.name}" in indexed
        ):
            return None
        return _aged_file_size(entry, cutoff)

    @staticmethod
    def _safe_original_name(value: str | None) -> str | None:
        if not value:
            return None
        basename = value.replace("\\", "/").rsplit("/", 1)[-1]
        basename = "".join(character for character in basename if character.isprintable()).strip()
        if basename in {"", ".", ".."}:
            return None
        return basename[:500]

    @staticmethod
    def _safe_media_type(value: str | None) -> str:
        normalized = (value or "").split(";", 1)[0].strip().lower()
        if len(normalized) <= 120 and re.fullmatch(
            r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+",
            normalized,
        ):
            return normalized
        return "application/octet-stream"

    @staticmethod
    def _is_link(path: Path) -> bool:
        return is_link_or_reparse(
            path,
            missing="assume_regular",
            unreadable="assume_link",
        )

    def _remember_verified(self, path: Path) -> None:
        stat_result = path.stat()
        self._verified_files[path] = (stat_result.st_size, stat_result.st_mtime_ns)

    def _prune_empty_parents(self, path: Path) -> None:
        for parent in (path.parent, path.parent.parent):
            with suppress(OSError):
                parent.rmdir()

    @staticmethod
    def _detect_media_type(path: Path) -> str | None:
        with path.open("rb") as source:
            header = source.read(4096)
        stripped = header.lstrip(b"\xef\xbb\xbf \t\r\n")
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if header.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if header.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
            return "image/webp"
        if header.startswith(b"BM"):
            return "image/bmp"
        if header.startswith((b"II*\x00", b"MM\x00*")):
            return "image/tiff"
        if stripped.startswith(b"<svg") or (
            stripped.startswith(b"<?xml") and b"<svg" in stripped[:2048]
        ):
            return "image/svg+xml"
        if len(header) >= 12 and header[4:8] == b"ftyp":
            brand = header[8:12]
            if brand in {b"avif", b"avis"}:
                return "image/avif"
            if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
                return "image/heic"
            if brand == b"qt  ":
                return "video/quicktime"
            return "video/mp4"
        if header.startswith(b"\x1aE\xdf\xa3"):
            return "video/webm"
        if header.startswith(b"RIFF") and header[8:12] == b"AVI ":
            return "video/x-msvideo"
        if header.startswith((b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")):
            return "video/mpeg"
        return None

    @staticmethod
    def _provenance_input_ids(provenance: object) -> set[str]:
        if not isinstance(provenance, dict):
            return set()
        artifact_ids = provenance.get("input_artifact_ids")
        if not isinstance(artifact_ids, list):
            return set()
        return {artifact_id for artifact_id in artifact_ids if isinstance(artifact_id, str)}

    @staticmethod
    def _aware(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @classmethod
    def _metadata_datetime(cls, value: object) -> datetime | None:
        if not isinstance(value, str):
            return None
        with suppress(ValueError):
            return cls._aware(datetime.fromisoformat(value))
        return None
