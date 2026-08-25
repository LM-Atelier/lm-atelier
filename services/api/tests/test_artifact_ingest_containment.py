from __future__ import annotations

import hashlib
import io
import os
import subprocess
from pathlib import Path

import pytest

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.domain import ArtifactKind


class _Session:
    """Enough Session for ingest_stream, with no database behind it.

    The defect and its fix are entirely about where BYTES land, so a real
    session would add setup without adding evidence - and would hide the
    failure being pinned, because the artifact row is written after the file.
    """

    def __init__(self) -> None:
        self.added: list[object] = []

    def scalar(self, *_args: object, **_kwargs: object) -> None:
        return None

    def add(self, value: object) -> None:
        self.added.append(value)

    def flush(self) -> None:
        return None


def _make_link_dir(link: Path, target: Path) -> bool:
    """Point `link` at `target`, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def _make_link_file(link: Path, target: Path) -> bool:
    """Point `link` at a FILE, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link)
    except OSError:
        return False
    return True


def _store(tmp_path: Path) -> tuple[ArtifactStore, Path]:
    root = tmp_path / "artifacts"
    root.mkdir()
    settings = Settings(
        data_dir=tmp_path / "data", dev=True, chat_engine="mock", media_engine="mock"
    )
    settings.prepare()
    return ArtifactStore(settings, root=root), root


def _payload() -> tuple[bytes, str]:
    payload = b"artifact bytes for a containment regression"
    return payload, hashlib.sha256(payload).hexdigest()


def test_an_ordinary_ingest_still_lands_where_the_record_says(tmp_path: Path) -> None:
    """The control. Without it every refusal below could be a broken store."""

    store, root = _store(tmp_path)
    payload, digest = _payload()

    artifact = store.ingest_stream(
        _Session(),
        io.BytesIO(payload),
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="ordinary.png",
        metadata=None,
    )

    landed = root / digest[:2] / digest[2:4] / digest
    assert landed.read_bytes() == payload
    assert artifact.sha256 == digest
    assert artifact.relative_path == f"{digest[:2]}/{digest[2:4]}/{digest}"
    assert artifact.size_bytes == len(payload)
    assert not list(root.glob("ingest-*")), "a staging entry was left behind"


@pytest.mark.parametrize("depth", [1, 2])
def test_a_link_at_any_digest_directory_is_refused(tmp_path: Path, depth: int) -> None:
    """The defect this file exists for, at both intermediate components.

    Measured before the fix: with a junction at the SECOND component,
    ingest_stream returned without refusing and the artifact landed outside the
    store under its content-addressed name. The old guard inspected the final
    name only, and these two directories are created by the ingest path itself,
    so the window was opened by the writer.
    """

    store, root = _store(tmp_path)
    payload, digest = _payload()
    outside = tmp_path / "outside"
    outside.mkdir()

    parts = [digest[:2], digest[2:4]]
    parent = root
    for part in parts[: depth - 1]:
        parent = parent / part
        parent.mkdir()
    if not _make_link_dir(parent / parts[depth - 1], outside):
        pytest.skip("this host does not permit directory links")

    with pytest.raises(ValueError):
        store.ingest_stream(
            _Session(),
            io.BytesIO(payload),
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
            original_name="hostile.png",
            metadata=None,
        )

    assert list(outside.iterdir()) == [], "bytes were written through the link"
    assert not list(root.glob("ingest-*")), "a staging entry was left behind"


def test_a_planted_staging_name_cannot_be_published(tmp_path: Path) -> None:
    """Staging names are unpredictable, so a plant cannot be the thing renamed.

    A fixed sibling - "ingest.tmp" or a predictable counter - can be created in
    advance, and the publish would then rename the PLANT under a digest computed
    from bytes it does not contain. Every artifact read back afterwards would
    fail its own checksum, which is the failure this test exists to keep away.
    """

    store, root = _store(tmp_path)
    payload, digest = _payload()
    (root / "ingest-0000000000000000.tmp").write_bytes(b"planted, not ours")

    store.ingest_stream(
        _Session(),
        io.BytesIO(payload),
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="ordinary.png",
        metadata=None,
    )

    landed = root / digest[:2] / digest[2:4] / digest
    assert landed.read_bytes() == payload
    assert (root / "ingest-0000000000000000.tmp").read_bytes() == b"planted, not ours"


def test_a_link_at_the_artifact_name_is_replaced_not_followed(tmp_path: Path) -> None:
    """Acquisition succeeding is not the whole guard.

    With legitimate digest directories, the only difference left between a
    contained publish and a pathname one is the final operation. `os.replace`
    on a path FOLLOWS a link at the destination name and overwrites whatever it
    points at; renaming through the held directory replaces the directory ENTRY
    and leaves the link's target alone.

    Every other test in this file passes with a pathname publish, because
    acquisition refuses their junction first. This is the one that does not.
    """

    store, root = _store(tmp_path)
    payload, digest = _payload()
    destination = root / digest[:2] / digest[2:4]
    destination.mkdir(parents=True)
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"not ours")
    if not _make_link_file(destination / digest, victim):
        pytest.skip("this host cannot create a file symlink unprivileged")

    store.ingest_stream(
        _Session(),
        io.BytesIO(payload),
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="ordinary.png",
        metadata=None,
    )

    assert victim.read_bytes() == b"not ours", "the link target was overwritten"
    assert (destination / digest).read_bytes() == payload


# Deliberately NOT tested here: a store root that is ITSELF a link.
#
# `ArtifactStore.__init__` does `(root or settings.artifact_dir).resolve()`,
# and resolve() follows the junction - so by the time any anchor is acquired
# the root IS the outside directory and there is nothing left to refuse. The
# redirect is erased at construction, before this module sees it.
#
# That is a real finding and a separate one: it is about which directory the
# store decides to be, not about how ingest publishes into it. Widening this
# slice to cover it would change the meaning of every caller's root. Recorded
# on tasks/artifact-ingest-writes-through-a-junction.md.
