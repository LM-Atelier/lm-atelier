from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_comfy_registry_reviewed_wheel_metadata import _HEADERS, _review_metadata
from test_comfy_registry_source_artifacts import DECLARATION
from test_comfy_registry_source_artifacts import source_review_context as source_review_context
from test_comfy_registry_wheel_artifacts import _document, _environment, _file, _resolve

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_source_artifacts import ComfyRegistrySourceArtifactError
from local_lm.models import ComfyRegistrySourceArtifactReview


def test_mixed_manifest_preserves_remote_records_and_binds_local_review_without_a_url(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    metadata = _HEADERS + b"Requires-Python: >=3.12\nRequires-Dist: helper>=2\n\n"
    _review_metadata(session, store, metadata)
    local = inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    remote = _resolve(["example-package==1.2.3"], {"example-package": _document()})
    manifest = inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, [local])

    assert inputs.validate_comfy_registry_wheel_input_manifest(manifest) == manifest
    assert manifest.remote == remote
    assert manifest.reviewed_local == (local,)
    assert local.declaration_sha256 == hashlib.sha256(local.declaration.encode()).hexdigest()
    assert local.metadata_sha256 == hashlib.sha256(metadata).hexdigest()
    assert local.compatibility_tag == "py3-none-any"
    assert local.target_sha256 == remote.target_sha256
    assert (
        local.review_sha256
        == session.scalars(select(ComfyRegistrySourceArtifactReview)).one().review_sha256
    )
    assert not hasattr(local, "url") and not hasattr(local, "payload")
    assert manifest.manifest_sha256 != remote.manifest_sha256


@pytest.mark.parametrize("field", ["sha256", "review_sha256", "metadata_sha256", "size_bytes"])
def test_manifest_identity_changes_when_retained_provenance_changes(
    source_review_context: tuple[Session, ArtifactStore],
    field: str,
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    local = inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    remote = _resolve([], {})
    before = inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, [local])
    value: object = "e" * 64
    if field == "size_bytes":
        value = local.size_bytes + 1
    changed = replace(local)
    object.__setattr__(changed, field, value)
    if field == "sha256":
        changed = replace(changed, artifact_id="sha256:" + str(value))
    after = inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, [changed])
    assert after.manifest_sha256 != before.manifest_sha256
    with pytest.raises(inputs.ComfyRegistryWheelInputError, match="Wheel input identity"):
        inputs.validate_comfy_registry_wheel_input_manifest(
            replace(before, reviewed_local=(changed,))
        )


def test_equal_wheel_bytes_from_another_declaration_do_not_alias(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    local = inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    declaration = local.declaration.replace(local.commit, "d" * 40)
    other = replace(
        local,
        declaration=declaration,
        declaration_sha256=hashlib.sha256(declaration.encode()).hexdigest(),
        commit="d" * 40,
    )
    remote = _resolve([], {})
    first = inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, [local])
    second = inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, [other])
    assert local.filename == other.filename and local.sha256 == other.sha256
    assert first.manifest_sha256 != second.manifest_sha256


@pytest.mark.parametrize(
    "metadata",
    [
        _HEADERS + b"Requires-Python: <3\n\n",
        _HEADERS + b"Requires-Python: invalid\n\n",
        _HEADERS + b"Requires-Python: >=3\nRequires-Python: <4\n\n",
    ],
)
def test_local_input_refuses_incompatible_or_ambiguous_python_requirements(
    source_review_context: tuple[Session, ArtifactStore],
    metadata: bytes,
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, metadata)
    with pytest.raises(inputs.ComfyRegistryWheelInputError) as caught:
        inputs.reviewed_wheel_input(
            session,
            store,
            declaration=DECLARATION,
            marker_environment=_environment(),
            supported_tags=("py3-none-any",),
        )
    assert caught.value.code == "source_wheel_python_incompatible"


def test_local_input_refuses_an_incompatible_wheel_tag(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    with pytest.raises(inputs.ComfyRegistryWheelInputError) as caught:
        inputs.reviewed_wheel_input(
            session,
            store,
            declaration=DECLARATION,
            marker_environment=_environment(),
            supported_tags=("cp312-cp312-win_amd64",),
        )
    assert caught.value.code == "source_wheel_target_incompatible"


def test_input_construction_rechecks_the_current_review(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    review = session.scalars(select(ComfyRegistrySourceArtifactReview)).one()
    review.review_sha256 = "e" * 64
    session.commit()
    with pytest.raises(ComfyRegistrySourceArtifactError) as caught:
        inputs.reviewed_wheel_input(
            session,
            store,
            declaration=DECLARATION,
            marker_environment=_environment(),
            supported_tags=("py3-none-any",),
        )
    assert caught.value.code == "source_artifact_review_stale"


@pytest.mark.parametrize("reason", ["duplicate-local", "duplicate-remote", "target"])
def test_mixed_manifest_refuses_ambiguous_or_differently_targeted_inputs(
    source_review_context: tuple[Session, ArtifactStore],
    reason: str,
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    local = inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    remote = _resolve([], {})
    locals_ = [local]
    expected = "duplicate_wheel_input"
    if reason == "duplicate-local":
        locals_.append(local)
    elif reason == "duplicate-remote":
        remote = _resolve(
            ["example-pkg==1.2.3"],
            {
                "example-pkg": _document(
                    _file("example_pkg-1.2.3-py3-none-any.whl"), name="example-pkg"
                ),
            },
        )
    else:
        remote = _resolve(
            [], {}, environment=_environment(python_version="3.11", python_full_version="3.11.9")
        )
        expected = "wheel_input_target_mismatch"
    with pytest.raises(inputs.ComfyRegistryWheelInputError) as caught:
        inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, locals_)
    assert caught.value.code == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_id", "unrelated"),
        ("declaration_sha256", "0" * 64),
        ("repository", "different/repository"),
        ("commit", "d" * 40),
        ("name", "different"),
        ("version", "9.0"),
        ("filename", "../example.whl"),
        ("wheel_tags", ("cp312-cp312-win_amd64",)),
        ("compatibility_tag", "cp312-cp312-win_amd64"),
        ("size_bytes", True),
        ("size_bytes", 0),
        ("sha256", "invalid"),
    ],
)
def test_local_manifest_refuses_inconsistent_identity_fields(
    source_review_context: tuple[Session, ArtifactStore],
    field: str,
    value: object,
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    local = inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    changed = replace(local)
    object.__setattr__(changed, field, value)
    with pytest.raises(inputs.ComfyRegistryWheelInputError):
        inputs.build_comfy_registry_wheel_input_manifest("f" * 64, _resolve([], {}), [changed])


def test_selected_tag_uses_the_same_canonical_target_as_remote_wheels(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    local = inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("PY3-NONE-ANY",),
    )
    remote = _resolve([], {}, tags=("PY3-NONE-ANY",))
    assert local.compatibility_tag == "py3-none-any"
    inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, [local])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("version", True),
        ("manifest_sha256", "e" * 64),
        ("declaration_sha256", "e" * 64),
        ("reviewed_local", []),
    ],
)
def test_manifest_validation_refuses_changed_contract_fields(field: str, value: object) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    manifest = inputs.build_comfy_registry_wheel_input_manifest("f" * 64, _resolve([], {}), [])
    changed = replace(manifest)
    object.__setattr__(changed, field, value)
    with pytest.raises(inputs.ComfyRegistryWheelInputError):
        inputs.validate_comfy_registry_wheel_input_manifest(changed)


def test_mixed_limit_counts_both_remote_and_local_records(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    local = inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    remote = _resolve(["example-package==1.2.3"], {"example-package": _document()})
    monkeypatch.setattr(inputs, "MAX_WHEEL_MANIFEST_ARTIFACTS", 1)
    with pytest.raises(inputs.ComfyRegistryWheelInputError) as caught:
        inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, [local])
    assert caught.value.code == "too_many_wheel_inputs"


@pytest.mark.parametrize(
    "change",
    ["none", "kind", "local-url", "root-field", "remote-declaration", "hash", "order", "tags"],
)
def test_persisted_manifest_round_trip_preserves_exact_sources_and_refuses_changes(
    source_review_context: tuple[Session, ArtifactStore],
    change: str,
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    local = inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    remote = _resolve(["example-package==1.2.3"], {"example-package": _document()})
    manifest = inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, [local])
    payload = json.loads(json.dumps(inputs.wheel_input_manifest_payload(manifest)))
    original = deepcopy(payload)
    local_record = next(
        record for record in payload["inputs"] if record["kind"] == "reviewed-local"
    )
    assert "payload" not in local_record and "url" not in local_record
    if change == "none":
        assert inputs.parse_wheel_input_manifest(payload) == manifest
        assert payload == original
        return
    if change == "kind":
        local_record["kind"] = "remote"
    elif change == "local-url":
        local_record["url"] = "file:///wheel.whl"
    elif change == "root-field":
        payload["unrecognized"] = True
    elif change == "remote-declaration":
        payload["remote_declaration_sha256"] = "a" * 64
    elif change == "hash":
        payload["manifest_sha256"] = "a" * 64
    elif change == "order":
        payload["inputs"].reverse()
    else:
        local_record["wheel_tags"] = "py3-none-any"
    with pytest.raises(inputs.ComfyRegistryWheelInputError):
        inputs.parse_wheel_input_manifest(payload)


def test_untrusted_filename_refuses_before_expanding_an_excessive_tag_product(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    session, store = source_review_context
    _review_metadata(session, store, _HEADERS + b"\n")
    local = inputs.reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
    )
    tags = [".".join(f"{prefix}{number}" for number in range(17)) for prefix in ("py", "a", "p")]
    filename = "example_pkg-1.2.3-" + "-".join(tags) + ".whl"
    assert len(filename) < 500

    def unexpected(_filename: str) -> None:
        raise AssertionError("Oversized tag products must refuse before parsing")

    monkeypatch.setattr(inputs, "parse_wheel_filename", unexpected)
    with pytest.raises(inputs.ComfyRegistryWheelInputError):
        inputs.build_comfy_registry_wheel_input_manifest(
            "f" * 64, _resolve([], {}), [replace(local, filename=filename)]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", None),
        ("requirement", []),
        ("filename", 12),
        ("compatibility_tag", {}),
        ("metadata_sha256", False),
        ("size_bytes", True),
        ("url", "https://files.pythonhosted.org:invalid/packages/example.whl"),
    ],
)
def test_persisted_remote_fields_refuse_malformed_types_and_addresses(
    field: str, value: object
) -> None:
    from local_lm import comfy_registry_wheel_inputs_v1 as inputs

    remote = _resolve(["example-package==1.2.3"], {"example-package": _document()})
    manifest = inputs.build_comfy_registry_wheel_input_manifest("f" * 64, remote, [])
    payload = json.loads(json.dumps(inputs.wheel_input_manifest_payload(manifest)))
    payload["inputs"][0][field] = value
    with pytest.raises(inputs.ComfyRegistryWheelInputError):
        inputs.parse_wheel_input_manifest(payload)
