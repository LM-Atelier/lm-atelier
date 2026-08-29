from __future__ import annotations

import hashlib
import hmac
import json
import stat
from dataclasses import dataclass
from email.parser import BytesParser
from io import BytesIO
from pathlib import PurePosixPath
from typing import NoReturn
from zipfile import BadZipFile, ZipFile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version
from sqlalchemy import select
from sqlalchemy.orm import Session

from .artifacts import ArtifactStore
from .domain import utcnow
from .models import Artifact, ComfyRegistrySourceArtifactReview
from .package_sources import SourceDependency, classify_source_url

MAX_SOURCE_DECLARATION_CHARACTERS = 1_000
MAX_REVIEWED_SOURCE_WHEEL_BYTES = 512 * 1024 * 1024
MAX_REVIEWED_SOURCE_WHEEL_ENTRIES = 20_000
MAX_REVIEWED_SOURCE_WHEEL_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_REVIEWED_SOURCE_METADATA_BYTES = 2 * 1024 * 1024
MAX_REVIEWED_SOURCE_RECORD_BYTES = 16 * 1024 * 1024
REVIEWER_KIND = "local-human"
INVALID_SOURCE_ARTIFACT = "Reviewed source artifact evidence is invalid."


class ComfyRegistrySourceArtifactError(ValueError):
    def __init__(self, code: str, message: str = INVALID_SOURCE_ARTIFACT) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VerifiedSourceWheel:
    declaration: str
    repository: str
    commit: str
    artifact_id: str
    artifact_sha256: str
    filename: str
    distribution: str
    version: str
    review_sha256: str
    payload: bytes


@dataclass(frozen=True)
class _SourceIdentity:
    declaration: str
    declaration_sha256: str
    distribution: str
    source: SourceDependency


@dataclass(frozen=True)
class _WheelInspection:
    filename: str
    distribution: str
    version: str
    evidence: dict[str, object]


def _fail(code: str) -> NoReturn:
    raise ComfyRegistrySourceArtifactError(code)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _source_identity(declaration: object) -> _SourceIdentity:
    if type(declaration) is not str:
        _fail("source_declaration_invalid")
    value = declaration.strip()
    if (
        not value
        or len(value) > MAX_SOURCE_DECLARATION_CHARACTERS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _fail("source_declaration_invalid")
    try:
        requirement = Requirement(value)
    except InvalidRequirement:
        _fail("source_declaration_invalid")
    if requirement.url is None:
        _fail("source_declaration_not_url")
    source = classify_source_url(requirement.url)
    if source.repository is None or source.commit is None or source.reference is not None:
        _fail("source_commit_not_exact")
    canonical = str(requirement)
    if len(canonical) > MAX_SOURCE_DECLARATION_CHARACTERS:
        _fail("source_declaration_invalid")
    return _SourceIdentity(
        canonical,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        canonicalize_name(requirement.name),
        source,
    )


def _safe_member(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        _fail("source_wheel_invalid")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail("source_wheel_invalid")
    if path.as_posix() != name.rstrip("/"):
        _fail("source_wheel_invalid")
    return path


def _member_bytes(archive: ZipFile, name: str, maximum: int) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError:
        _fail("source_wheel_invalid")
    if info.file_size > maximum:
        _fail("source_wheel_invalid")
    with archive.open(info, "r") as source:
        payload = source.read(maximum + 1)
    if len(payload) > maximum or len(payload) != info.file_size:
        _fail("source_wheel_invalid")
    return payload


def _inspect_wheel(
    payload: bytes, filename: object, expected_distribution: str
) -> _WheelInspection:
    if (
        type(filename) is not str
        or not filename
        or len(filename) > 500
        or "/" in filename
        or "\\" in filename
    ):
        _fail("source_wheel_invalid")
    try:
        parsed_name, parsed_version, _build, tags = parse_wheel_filename(filename)
    except (InvalidVersion, InvalidWheelFilename):
        _fail("source_wheel_invalid")
    distribution = canonicalize_name(str(parsed_name))
    if distribution != expected_distribution:
        _fail("source_wheel_distribution_mismatch")
    try:
        with ZipFile(BytesIO(payload), "r") as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_REVIEWED_SOURCE_WHEEL_ENTRIES:
                _fail("source_wheel_invalid")
            names: set[str] = set()
            expanded = 0
            metadata_names: list[str] = []
            for entry in entries:
                path = _safe_member(entry.filename)
                if entry.filename in names or entry.flag_bits & 1:
                    _fail("source_wheel_invalid")
                names.add(entry.filename)
                mode = entry.external_attr >> 16
                if mode and stat.S_ISLNK(mode):
                    _fail("source_wheel_invalid")
                expanded += entry.file_size
                if expanded > MAX_REVIEWED_SOURCE_WHEEL_EXPANDED_BYTES:
                    _fail("source_wheel_invalid")
                if (
                    len(path.parts) == 2
                    and path.parts[0].endswith(".dist-info")
                    and path.parts[1] == "METADATA"
                ):
                    metadata_names.append(entry.filename)
            if len(metadata_names) != 1:
                _fail("source_wheel_invalid")
            metadata_name = metadata_names[0]
            dist_info = metadata_name.rsplit("/", 1)[0]
            wheel_name = f"{dist_info}/WHEEL"
            record_name = f"{dist_info}/RECORD"
            if wheel_name not in names or record_name not in names:
                _fail("source_wheel_invalid")
            metadata_bytes = _member_bytes(
                archive, metadata_name, MAX_REVIEWED_SOURCE_METADATA_BYTES
            )
            wheel_bytes = _member_bytes(archive, wheel_name, MAX_REVIEWED_SOURCE_METADATA_BYTES)
            record_bytes = _member_bytes(archive, record_name, MAX_REVIEWED_SOURCE_RECORD_BYTES)
    except (BadZipFile, OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, ComfyRegistrySourceArtifactError):
            raise
        raise ComfyRegistrySourceArtifactError("source_wheel_invalid") from exc
    metadata = BytesParser().parsebytes(metadata_bytes)
    metadata_name_value = metadata.get("Name")
    metadata_version_value = metadata.get("Version")
    try:
        metadata_version = Version(str(metadata_version_value))
    except InvalidVersion:
        _fail("source_wheel_invalid")
    if (
        not isinstance(metadata_name_value, str)
        or canonicalize_name(metadata_name_value) != distribution
        or metadata_version != parsed_version
    ):
        _fail("source_wheel_distribution_mismatch")
    evidence: dict[str, object] = {
        "entry_count": len(entries),
        "expanded_bytes": expanded,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "wheel_sha256": hashlib.sha256(wheel_bytes).hexdigest(),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "tags": sorted(str(tag) for tag in tags),
    }
    return _WheelInspection(filename, distribution, str(parsed_version), evidence)


def _artifact_bytes(
    session: Session,
    store: ArtifactStore,
    artifact_id: object,
) -> tuple[Artifact, bytes]:
    if type(artifact_id) is not str or not artifact_id:
        _fail("source_artifact_missing")
    artifact = session.get(Artifact, artifact_id)
    if artifact is None:
        _fail("source_artifact_missing")
    if (
        artifact.id != f"sha256:{artifact.sha256}"
        or type(artifact.size_bytes) is not int
        or artifact.size_bytes < 1
        or artifact.size_bytes > MAX_REVIEWED_SOURCE_WHEEL_BYTES
    ):
        _fail("source_artifact_invalid")
    try:
        payload = store.verified_bytes(artifact, maximum_bytes=MAX_REVIEWED_SOURCE_WHEEL_BYTES)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ComfyRegistrySourceArtifactError("source_artifact_unreadable") from exc
    if len(payload) != artifact.size_bytes:
        _fail("source_artifact_invalid")
    return artifact, payload


def _review_payload(
    source: _SourceIdentity,
    artifact: Artifact,
    inspection: _WheelInspection,
) -> dict[str, object]:
    return {
        "version": 1,
        "source_declaration": source.declaration,
        "source_declaration_sha256": source.declaration_sha256,
        "repository": source.source.repository,
        "commit": source.source.commit,
        "artifact_id": artifact.id,
        "artifact_sha256": artifact.sha256,
        "artifact_size_bytes": artifact.size_bytes,
        "wheel_filename": inspection.filename,
        "wheel_distribution": inspection.distribution,
        "wheel_version": inspection.version,
        "evidence": inspection.evidence,
        "reviewer_kind": REVIEWER_KIND,
    }


def record_local_source_artifact_review(
    session: Session,
    store: ArtifactStore,
    *,
    declaration: object,
    artifact_id: object,
) -> ComfyRegistrySourceArtifactReview:
    """Record one explicit local-human review of exact retained wheel bytes.

    This function never runs a build backend and does not make the dependency
    planner accept a URL. It only creates the persistent, content-bound review
    that a later verified-byte consumer may require.
    """

    source = _source_identity(declaration)
    artifact, payload = _artifact_bytes(session, store, artifact_id)
    inspection = _inspect_wheel(payload, artifact.original_name, source.distribution)
    review_payload = _review_payload(source, artifact, inspection)
    review_sha256 = _digest(review_payload)
    existing = session.scalar(
        select(ComfyRegistrySourceArtifactReview).where(
            ComfyRegistrySourceArtifactReview.source_declaration_sha256 == source.declaration_sha256
        )
    )
    if existing is not None:
        if (
            existing.source_declaration == source.declaration
            and existing.repository == source.source.repository
            and existing.source_commit == source.source.commit
            and existing.artifact_id == artifact.id
            and existing.artifact_sha256 == artifact.sha256
            and existing.artifact_size_bytes == artifact.size_bytes
            and existing.wheel_filename == inspection.filename
            and existing.wheel_distribution == inspection.distribution
            and existing.wheel_version == inspection.version
            and existing.evidence_json == inspection.evidence
            and existing.reviewer_kind == REVIEWER_KIND
            and existing.review_sha256 == review_sha256
        ):
            return existing
        _fail("source_artifact_review_conflict")
    review = ComfyRegistrySourceArtifactReview(
        source_declaration=source.declaration,
        source_declaration_sha256=source.declaration_sha256,
        repository=str(source.source.repository),
        source_commit=str(source.source.commit),
        artifact_id=artifact.id,
        artifact_sha256=artifact.sha256,
        artifact_size_bytes=artifact.size_bytes,
        wheel_filename=inspection.filename,
        wheel_distribution=inspection.distribution,
        wheel_version=inspection.version,
        evidence_json=inspection.evidence,
        reviewer_kind=REVIEWER_KIND,
        review_sha256=review_sha256,
        reviewed_at=utcnow(),
    )
    session.add(review)
    session.flush()
    return review


def verified_reviewed_source_wheel(
    session: Session,
    store: ArtifactStore,
    *,
    declaration: object,
) -> VerifiedSourceWheel:
    """Return exact reviewed wheel bytes after re-deriving every authority fact."""

    source = _source_identity(declaration)
    review = session.scalar(
        select(ComfyRegistrySourceArtifactReview).where(
            ComfyRegistrySourceArtifactReview.source_declaration_sha256 == source.declaration_sha256
        )
    )
    if review is None or review.reviewed_at is None or review.reviewer_kind != REVIEWER_KIND:
        _fail("source_artifact_review_missing")
    artifact, payload = _artifact_bytes(session, store, review.artifact_id)
    inspection = _inspect_wheel(payload, artifact.original_name, source.distribution)
    review_payload = _review_payload(source, artifact, inspection)
    expected_sha256 = _digest(review_payload)
    exact = (
        review.source_declaration == source.declaration
        and review.repository == source.source.repository
        and review.source_commit == source.source.commit
        and review.artifact_id == artifact.id
        and review.artifact_sha256 == artifact.sha256
        and review.artifact_size_bytes == artifact.size_bytes
        and review.wheel_filename == inspection.filename
        and review.wheel_distribution == inspection.distribution
        and review.wheel_version == inspection.version
        and review.evidence_json == inspection.evidence
        and hmac.compare_digest(review.review_sha256, expected_sha256)
    )
    if not exact:
        _fail("source_artifact_review_stale")
    return VerifiedSourceWheel(
        source.declaration,
        str(source.source.repository),
        str(source.source.commit),
        artifact.id,
        artifact.sha256,
        inspection.filename,
        inspection.distribution,
        inspection.version,
        expected_sha256,
        payload,
    )
