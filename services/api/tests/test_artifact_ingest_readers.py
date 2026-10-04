"""Repeated media ingest remains available while its verified bytes are being read."""

from __future__ import annotations

import os
from contextlib import ExitStack

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.filesystem_links import AnchoredDirectory, open_child_directory, open_entry
from local_lm.models import Artifact


@pytest.mark.parametrize("held_reader", [False, True])
async def test_repeated_ingest_preserves_bytes_while_the_artifact_is_open(
    client: AsyncClient, settings: Settings, held_reader: bool
) -> None:
    store = ArtifactStore(settings)
    content = b"Neutral repeated media fixture"
    with SessionLocal() as session:
        artifact = store.ingest_bytes(
            session, content, kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        session.commit()
        artifact_id, digest = artifact.id, artifact.sha256

    with ExitStack() as readers:
        descriptor = None
        if held_reader:
            root = readers.enter_context(AnchoredDirectory(store.root))
            first = readers.enter_context(open_child_directory(root, digest[:2]))
            second = readers.enter_context(open_child_directory(first, digest[2:4]))
            descriptor = open_entry(second, digest)
            assert descriptor is not None
            readers.callback(os.close, descriptor)
        with SessionLocal() as session:
            repeated = store.ingest_bytes(
                session, content, kind=ArtifactKind.IMAGE, media_type="image/png"
            )
            session.commit()
            assert repeated.id == artifact_id
            assert store.verified_bytes(repeated, maximum_bytes=len(content)) == content
            assert session.scalar(select(func.count()).select_from(Artifact)) == 1
        if descriptor is not None:
            assert os.read(descriptor, len(content) + 1) == content


@pytest.mark.parametrize("held_reader", [False, True])
async def test_repeated_ingest_does_not_reuse_corrupt_bytes_of_the_same_size(
    client: AsyncClient, settings: Settings, held_reader: bool
) -> None:
    store = ArtifactStore(settings)
    content = b"Neutral repeated media fixture"
    with SessionLocal() as session:
        artifact = store.ingest_bytes(
            session, content, kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        session.commit()
        digest = artifact.sha256
        path = store.resolve(artifact)
        path.write_bytes(b"x" * len(content))

    with (
        AnchoredDirectory(store.root) as root,
        open_child_directory(root, digest[:2]) as first,
        open_child_directory(first, digest[2:4]) as second,
        ExitStack() as readers,
        SessionLocal() as session,
    ):
        if held_reader and os.name == "nt":
            descriptor = open_entry(second, digest)
            assert descriptor is not None
            readers.callback(os.close, descriptor)
            with pytest.raises(ValueError):
                store.ingest_bytes(
                    session, content, kind=ArtifactKind.IMAGE, media_type="image/png"
                )
            assert os.read(descriptor, len(content) + 1) == b"x" * len(content)
        else:
            repeated = store.ingest_bytes(
                session, content, kind=ArtifactKind.IMAGE, media_type="image/png"
            )
            session.commit()
            assert store.verified_bytes(repeated, maximum_bytes=len(content)) == content
