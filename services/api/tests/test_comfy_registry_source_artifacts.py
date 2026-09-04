from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from sqlalchemy import Select, create_engine
from sqlalchemy.orm import Session

from local_lm import comfy_registry_source_artifacts as source_artifacts
from local_lm.artifact_library import (
    deletion_restricted_artifact_ids,
    referenced_artifact_ids,
)
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_source_artifacts import (
    ComfyRegistrySourceArtifactError,
    record_local_source_artifact_review,
    verified_reviewed_source_wheel,
)
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact, ComfyRegistrySourceArtifactReview

COMMIT = "0123456789abcdef0123456789abcdef01234567"
ALT_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
DECLARATION = f"example-pkg @ git+https://github.com/example/project@{COMMIT}"
ALT_DECLARATION = f"example-pkg @ git+https://github.com/example/project@{ALT_COMMIT}"


@pytest.fixture
def source_review_context(
    tmp_path: Path,
) -> Iterator[tuple[Session, ArtifactStore]]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    settings = Settings(data_dir=tmp_path / "data", dev=True)
    settings.prepare()
    with Session(engine) as session:
        yield session, ArtifactStore(settings)


def _wheel(
    *,
    name: str = "example-pkg",
    version: str = "1.2.3",
    filename_distribution: str = "example_pkg",
) -> tuple[str, bytes]:
    filename = f"{filename_distribution}-{version}-py3-none-any.whl"
    dist_info = f"{filename_distribution}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n".encode()
    wheel = (
        b"Wheel-Version: 1.0\nGenerator: lm-atelier-test\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n\n"
    )
    record = (f"{dist_info}/METADATA,,\n{dist_info}/WHEEL,,\n{dist_info}/RECORD,,\n").encode()
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel)
        archive.writestr(f"{dist_info}/RECORD", record)
        archive.writestr("example_pkg/__init__.py", b"__version__ = '1.2.3'\n")
    return filename, buffer.getvalue()


def _wheel_with_extra_member(
    name: str,
    *,
    external_attr: int = 0,
) -> tuple[str, bytes]:
    filename, payload = _wheel()
    rewritten = io.BytesIO()
    with (
        ZipFile(io.BytesIO(payload), "r") as source,
        ZipFile(rewritten, "w", ZIP_DEFLATED) as target,
    ):
        for entry in source.infolist():
            target.writestr(entry, source.read(entry))
        extra = ZipInfo(name)
        extra.external_attr = external_attr
        target.writestr(extra, b"payload")
    return filename, rewritten.getvalue()


def _mark_last_member_encrypted(payload: bytes) -> bytes:
    rewritten = bytearray(payload)
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        header = rewritten.rfind(signature)
        assert header >= 0
        start = header + flag_offset
        flags = int.from_bytes(rewritten[start : start + 2], "little") | 1
        rewritten[start : start + 2] = flags.to_bytes(2, "little")
    return bytes(rewritten)


def _artifact(
    session: Session,
    store: ArtifactStore,
    *,
    payload: bytes | None = None,
    filename: str | None = None,
) -> Artifact:
    default_filename, default_payload = _wheel()
    return store.ingest_bytes(
        session,
        payload if payload is not None else default_payload,
        kind=ArtifactKind.OTHER,
        media_type="application/vnd.python.wheel",
        original_name=filename or default_filename,
    )


def test_reviewed_source_archive_policy_bounds_cannot_silently_relax() -> None:
    assert source_artifacts.MAX_SOURCE_DECLARATION_CHARACTERS <= 1_000
    assert source_artifacts.MAX_REVIEWED_SOURCE_WHEEL_BYTES <= 512 * 1024 * 1024
    assert source_artifacts.MAX_REVIEWED_SOURCE_WHEEL_ENTRIES <= 20_000
    assert source_artifacts.MAX_REVIEWED_SOURCE_WHEEL_EXPANDED_BYTES <= 2 * 1024 * 1024 * 1024
    assert source_artifacts.MAX_REVIEWED_SOURCE_METADATA_BYTES <= 2 * 1024 * 1024
    assert source_artifacts.MAX_REVIEWED_SOURCE_RECORD_BYTES <= 16 * 1024 * 1024


@pytest.mark.parametrize(
    ("constant", "lowered_limit", "code"),
    [
        (
            "MAX_SOURCE_DECLARATION_CHARACTERS",
            len(DECLARATION) - 1,
            "source_declaration_invalid",
        ),
        ("MAX_REVIEWED_SOURCE_WHEEL_BYTES", 1, "source_artifact_invalid"),
        ("MAX_REVIEWED_SOURCE_WHEEL_ENTRIES", 3, "source_wheel_invalid"),
        ("MAX_REVIEWED_SOURCE_WHEEL_EXPANDED_BYTES", 1, "source_wheel_invalid"),
        ("MAX_REVIEWED_SOURCE_METADATA_BYTES", 1, "source_wheel_invalid"),
        ("MAX_REVIEWED_SOURCE_RECORD_BYTES", 1, "source_wheel_invalid"),
    ],
)
def test_each_reviewed_source_archive_bound_is_enforced(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    lowered_limit: int,
    code: str,
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    monkeypatch.setattr(source_artifacts, constant, lowered_limit)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session,
            store,
            declaration=DECLARATION,
            artifact_id=artifact.id,
        )

    assert caught.value.code == code


def test_review_persists_exact_source_and_verified_bytes(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)

    review = record_local_source_artifact_review(
        session,
        store,
        declaration=DECLARATION,
        artifact_id=artifact.id,
    )
    session.commit()

    verified = verified_reviewed_source_wheel(
        session,
        store,
        declaration=DECLARATION,
    )
    assert review.reviewer_kind == "local-human"
    assert review.repository == "example/project"
    assert review.source_commit == COMMIT
    assert review.artifact_sha256 == artifact.sha256
    assert verified.payload == store.verified_bytes(
        artifact, maximum_bytes=source_artifacts.MAX_REVIEWED_SOURCE_WHEEL_BYTES
    )
    assert verified.review_sha256 == review.review_sha256
    assert verified.distribution == "example-pkg"
    assert verified.version == "1.2.3"


def test_exact_retry_is_idempotent_and_does_not_rewrite_review_time(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    first = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()
    reviewed_at = first.reviewed_at

    second = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )

    assert second.id == first.id
    assert second.reviewed_at == reviewed_at
    assert session.query(ComfyRegistrySourceArtifactReview).count() == 1


def test_same_artifact_under_a_different_declaration_is_a_coded_conflict(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    first = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session,
            store,
            declaration=ALT_DECLARATION,
            artifact_id=artifact.id,
        )

    assert caught.value.code == "source_artifact_review_conflict"
    assert str(caught.value) == "Reviewed source artifact evidence is invalid."
    assert session.query(ComfyRegistrySourceArtifactReview).count() == 1
    assert session.get(ComfyRegistrySourceArtifactReview, first.id) is first
    exact_retry = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    assert exact_retry is first


def test_late_artifact_uniqueness_race_is_coded_without_poisoning_session(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    first = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()
    real_scalar = session.scalar
    scalar_calls = 0

    def hide_artifact_preflight(
        statement: Select[tuple[ComfyRegistrySourceArtifactReview]],
    ) -> ComfyRegistrySourceArtifactReview | None:
        nonlocal scalar_calls
        scalar_calls += 1
        result = real_scalar(statement)
        if scalar_calls == 2:
            return None
        return result

    monkeypatch.setattr(session, "scalar", hide_artifact_preflight)
    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session,
            store,
            declaration=ALT_DECLARATION,
            artifact_id=artifact.id,
        )

    assert scalar_calls == 3
    assert caught.value.code == "source_artifact_review_conflict"
    assert str(caught.value) == "Reviewed source artifact evidence is invalid."
    assert session.query(ComfyRegistrySourceArtifactReview).count() == 1
    assert session.get(ComfyRegistrySourceArtifactReview, first.id) is first
    exact_retry = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    assert exact_retry is first
    assert scalar_calls == 4


def test_exact_retry_race_returns_the_existing_review(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    first = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()
    real_scalar = session.scalar
    scalar_calls = 0

    def hide_declaration_preflight(
        statement: Select[tuple[ComfyRegistrySourceArtifactReview]],
    ) -> ComfyRegistrySourceArtifactReview | None:
        nonlocal scalar_calls
        scalar_calls += 1
        result = real_scalar(statement)
        if scalar_calls == 1:
            return None
        return result

    monkeypatch.setattr(session, "scalar", hide_declaration_preflight)
    second = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )

    assert scalar_calls == 2
    assert second is first
    assert session.query(ComfyRegistrySourceArtifactReview).count() == 1


def test_exact_retry_after_uniqueness_conflict_returns_the_existing_review(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    first = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()
    real_scalar = session.scalar
    scalar_calls = 0

    def hide_both_preflights(
        statement: Select[tuple[ComfyRegistrySourceArtifactReview]],
    ) -> ComfyRegistrySourceArtifactReview | None:
        nonlocal scalar_calls
        scalar_calls += 1
        result = real_scalar(statement)
        if scalar_calls <= 2:
            return None
        return result

    monkeypatch.setattr(session, "scalar", hide_both_preflights)
    second = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )

    assert scalar_calls == 3
    assert second is first
    assert session.query(ComfyRegistrySourceArtifactReview).count() == 1
    exact_retry = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    assert exact_retry is first


@pytest.mark.parametrize(
    "declaration,code",
    [
        ("example-pkg==1.2.3", "source_declaration_not_url"),
        (
            "example-pkg @ git+https://github.com/example/project@main",
            "source_commit_not_exact",
        ),
        (
            "example-pkg @ git+https://example.invalid/example/project@" + COMMIT,
            "source_commit_not_exact",
        ),
        (
            "git+https://github.com/example/project@" + COMMIT,
            "source_declaration_invalid",
        ),
    ],
)
def test_review_refuses_any_source_without_named_allowed_exact_commit(
    source_review_context: tuple[Session, ArtifactStore],
    declaration: str,
    code: str,
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session, store, declaration=declaration, artifact_id=artifact.id
        )

    assert caught.value.code == code
    assert str(caught.value) == "Reviewed source artifact evidence is invalid."


def test_consumer_requires_a_persistent_review(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    _artifact(session, store)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        verified_reviewed_source_wheel(session, store, declaration=DECLARATION)

    assert caught.value.code == "source_artifact_review_missing"


def test_review_refuses_distribution_identity_mismatch(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    filename, payload = _wheel(name="different-package")
    artifact = _artifact(session, store, payload=payload, filename=filename)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session, store, declaration=DECLARATION, artifact_id=artifact.id
        )

    assert caught.value.code == "source_wheel_distribution_mismatch"


def test_verified_consumer_refuses_changed_review_evidence(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    review = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()
    review.evidence_json = {**review.evidence_json, "entry_count": 999}
    session.commit()

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        verified_reviewed_source_wheel(session, store, declaration=DECLARATION)

    assert caught.value.code == "source_artifact_review_stale"


def test_verified_consumer_requires_the_review_digest_itself(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    review = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()
    review.review_sha256 = "0" * 64
    session.commit()

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        verified_reviewed_source_wheel(session, store, declaration=DECLARATION)

    assert caught.value.code == "source_artifact_review_stale"


def test_review_keeps_verified_bytes_when_verified_path_is_swapped(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    _, five_payload = _wheel_with_extra_member("example_pkg/extra.py")
    original_verified_path = store.verified_path

    def swapped_path(target: Artifact) -> Path:
        path = original_verified_path(target)
        path.write_bytes(five_payload)
        return path

    monkeypatch.setattr(store, "verified_path", swapped_path)

    review = record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )

    assert review.evidence_json["entry_count"] == 4


def test_verified_consumer_refuses_changed_artifact_bytes(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()
    path = store.resolve(artifact)
    path.write_bytes(b"x" * artifact.size_bytes)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        verified_reviewed_source_wheel(session, store, declaration=DECLARATION)

    assert caught.value.code == "source_artifact_unreadable"


def test_reviewed_artifact_is_a_strong_retention_edge(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    artifact = _artifact(session, store)
    record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()

    assert artifact.id in referenced_artifact_ids(session)
    assert artifact.id in deletion_restricted_artifact_ids(session)


def test_wheel_archive_refuses_traversal_and_symlink_entries(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    filename, payload = _wheel()
    buffer = io.BytesIO(payload)
    rewritten = io.BytesIO()
    with ZipFile(buffer, "r") as source, ZipFile(rewritten, "w", ZIP_DEFLATED) as target:
        for entry in source.infolist():
            target.writestr(entry, source.read(entry))
        target.writestr("../escape", b"no")
        symlink = ZipInfo("example_pkg/link")
        symlink.external_attr = 0o120777 << 16
        target.writestr(symlink, b"target")
    artifact = _artifact(
        session,
        store,
        payload=rewritten.getvalue(),
        filename=filename,
    )

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session, store, declaration=DECLARATION, artifact_id=artifact.id
        )

    assert caught.value.code == "source_wheel_invalid"


def test_wheel_archive_refuses_traversal_without_another_invalid_member(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    filename, payload = _wheel_with_extra_member("../escape")
    artifact = _artifact(session, store, payload=payload, filename=filename)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session, store, declaration=DECLARATION, artifact_id=artifact.id
        )

    assert caught.value.code == "source_wheel_invalid"


def test_wheel_archive_refuses_noncanonical_member_without_traversal(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    filename, payload = _wheel_with_extra_member("example_pkg//ambiguous.py")
    artifact = _artifact(session, store, payload=payload, filename=filename)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session, store, declaration=DECLARATION, artifact_id=artifact.id
        )

    assert caught.value.code == "source_wheel_invalid"


def test_wheel_archive_refuses_symlink_without_traversal(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    filename, payload = _wheel_with_extra_member(
        "example_pkg/link",
        external_attr=0o120777 << 16,
    )
    artifact = _artifact(session, store, payload=payload, filename=filename)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session, store, declaration=DECLARATION, artifact_id=artifact.id
        )

    assert caught.value.code == "source_wheel_invalid"


def test_wheel_archive_refuses_encrypted_member_independently(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    filename, payload = _wheel()
    encrypted_payload = _mark_last_member_encrypted(payload)
    with ZipFile(io.BytesIO(encrypted_payload), "r") as archive:
        entries = archive.infolist()
    assert entries[-1].filename == "example_pkg/__init__.py"
    assert entries[-1].flag_bits & 1
    assert all(not (entry.flag_bits & 1) for entry in entries[:-1])
    artifact = _artifact(
        session,
        store,
        payload=encrypted_payload,
        filename=filename,
    )

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        record_local_source_artifact_review(
            session, store, declaration=DECLARATION, artifact_id=artifact.id
        )

    assert caught.value.code == "source_wheel_invalid"
