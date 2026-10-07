"""A deleted library membership cannot be edited through its retained artifact."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session
from test_artifact_library_entries import library_session as library_session

from local_lm.artifact_library import (
    ArtifactLibraryConflict,
    ensure_library_entry,
    set_library_favorite,
)
from local_lm.artifacts import ArtifactStore
from local_lm.domain import ArtifactKind


@pytest.mark.parametrize("favorite", [False, True])
def test_deleted_library_membership_refuses_favorite_writes(
    library_session: tuple[ArtifactStore, Session], favorite: bool
) -> None:
    store, session = library_session
    artifact = store.ingest_bytes(
        session, b"constructed garden media", kind=ArtifactKind.IMAGE, media_type="image/png"
    )
    entry = ensure_library_entry(session, artifact)
    assert entry is not None
    entry.state = "trashed"
    entry.deleted_at = datetime.now(UTC)
    entry.recovery_id = "deleted-garden-library-item"
    entry.version += 1
    session.commit()
    session.refresh(entry)
    version, deleted_at = entry.version, entry.deleted_at

    with pytest.raises(ArtifactLibraryConflict, match="^Restore this Media Library item"):
        set_library_favorite(session, artifact, favorite)

    session.rollback()
    session.refresh(entry)
    session.refresh(artifact)
    assert entry.state == "trashed"
    assert entry.version == version
    assert entry.deleted_at == deleted_at
    assert entry.recovery_id == "deleted-garden-library-item"
    assert entry.favorite is artifact.favorite is False
    assert store.resolve(artifact).read_bytes() == b"constructed garden media"
