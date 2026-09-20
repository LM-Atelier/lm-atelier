from __future__ import annotations

import io
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from sqlalchemy.orm import Session
from test_comfy_registry_source_artifacts import (
    DECLARATION,
    _artifact,
    _wheel,
)
from test_comfy_registry_source_artifacts import (
    source_review_context as source_review_context,
)

from local_lm import comfy_registry_wheel_metadata as metadata_module
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_source_artifacts import (
    ComfyRegistrySourceArtifactError,
    record_local_source_artifact_review,
    verified_reviewed_source_wheel,
)

_HEADERS = b"Metadata-Version: 2.1\nName: example-pkg\nVersion: 1.2.3\n"


def _review_metadata(session: Session, store: ArtifactStore, metadata: bytes) -> None:
    filename, payload = _wheel()
    rewritten = io.BytesIO()
    with ZipFile(io.BytesIO(payload)) as source, ZipFile(rewritten, "w", ZIP_DEFLATED) as target:
        for entry in source.infolist():
            content = metadata if entry.filename.endswith("/METADATA") else source.read(entry)
            target.writestr(entry, content)
    artifact = _artifact(session, store, payload=rewritten.getvalue(), filename=filename)
    record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()


def test_reviewed_wheel_returns_exact_metadata_and_unevaluated_dependencies(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, store = source_review_context
    metadata = (
        _HEADERS
        + b'Requires-Dist: helper_pkg[fast]>=2; python_version < "3.13"\n'
        + b'Requires-Dist: optional_pkg; extra == "optional"\n'
        + b'Requires-Dist: helper_pkg[fast]>=2; python_version < "3.13"\n'
        + b"\nAn inert description, preserved byte for byte.\n"
    )
    _review_metadata(session, store, metadata)

    verified = verified_reviewed_source_wheel(session, store, declaration=DECLARATION)

    assert verified.core_metadata == metadata
    assert [item.name for item in verified.metadata_requirements] == [
        "helper-pkg",
        "optional-pkg",
    ]
    first, second = verified.metadata_requirements
    assert (first.source_name, first.source_version) == ("example-pkg", "1.2.3")
    assert first.specifier == ">=2"
    assert first.extras == ("fast",)
    assert first.marker == 'python_version < "3.13"'
    assert second.marker == 'extra == "optional"'
    assert second.extras == ()
    with ZipFile(io.BytesIO(verified.payload)) as archive:
        assert archive.read("example_pkg-1.2.3.dist-info/METADATA") == verified.core_metadata


@pytest.mark.parametrize(
    ("metadata", "code"),
    [
        (_HEADERS + b"Name: another-package\n\n", "invalid_core_metadata"),
        (_HEADERS.replace(b"Metadata-Version: 2.1\n", b"") + b"\n", "invalid_core_metadata"),
        (_HEADERS.replace(b"2.1", b"9.0") + b"\n", "unsupported_core_metadata"),
        (_HEADERS + b"Requires-Dist: [invalid\n\n", "invalid_transitive_requirement"),
        (
            _HEADERS + b"Requires-Dist: helper @ https://example.invalid/helper.whl\n\n",
            "direct_transitive_url",
        ),
        (_HEADERS + b"\nDescription with a null \x00 byte.\n", "invalid_core_metadata"),
    ],
)
def test_verified_wheel_refuses_dependency_metadata_the_planner_cannot_use(
    source_review_context: tuple[Session, ArtifactStore], metadata: bytes, code: str
) -> None:
    session, store = source_review_context
    _review_metadata(session, store, metadata)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        verified_reviewed_source_wheel(session, store, declaration=DECLARATION)

    assert caught.value.code == code
    assert str(caught.value) == "Reviewed source artifact evidence is invalid."


@pytest.mark.parametrize(
    ("constant", "limit", "code"),
    [
        ("MAX_WHEEL_CORE_METADATA_BYTES", 32, "core_metadata_too_large"),
        ("MAX_WHEEL_CORE_METADATA_LINES", 3, "invalid_core_metadata"),
        ("MAX_WHEEL_CORE_METADATA_LINE_BYTES", 12, "invalid_core_metadata"),
        ("MAX_WHEEL_REQUIRES_DIST", 1, "too_many_transitive_requirements"),
    ],
)
def test_reviewed_metadata_uses_the_same_bounds_as_remote_metadata(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    limit: int,
    code: str,
) -> None:
    session, store = source_review_context
    metadata = _HEADERS + b"Requires-Dist: first>=1\nRequires-Dist: second>=2\n\n"
    _review_metadata(session, store, metadata)
    monkeypatch.setattr(metadata_module, constant, limit)

    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        verified_reviewed_source_wheel(session, store, declaration=DECLARATION)

    assert caught.value.code == code
