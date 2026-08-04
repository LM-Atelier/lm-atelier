from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from packaging.markers import default_environment

import local_lm.comfy_registry_wheel_artifacts as artifact_module
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactError,
    build_comfy_registry_wheel_artifact_manifest,
    comfy_registry_wheel_target_sha256,
    current_comfy_registry_wheel_target,
    resolve_comfy_registry_wheel_artifacts,
    select_comfy_registry_wheel_artifact,
    validate_comfy_registry_wheel_artifact_manifest,
)

_SHA256 = "a" * 64
_METADATA_SHA256 = "b" * 64
_UNIVERSAL_TAG = "py3-none-any"


def _environment(**updates: str) -> dict[str, str]:
    environment = default_environment()
    environment["extra"] = ""
    environment.update(updates)
    return environment


def _file(
    filename: str = "example_package-1.2.3-py3-none-any.whl",
    **updates: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "filename": filename,
        "url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
        "hashes": {"sha256": _SHA256},
        "requires-python": ">=3.12",
        "yanked": False,
        "size": 1234,
        "core-metadata": {"sha256": _METADATA_SHA256},
    }
    record.update(updates)
    return record


def _document(
    *files: object,
    name: str = "example-package",
    api_version: str = "1.4",
) -> dict[str, object]:
    return {
        "meta": {"api-version": api_version},
        "name": name,
        "files": list(files or (_file(),)),
        "versions": ["1.2.3"],
    }


def _resolve(
    declarations: list[str],
    documents: dict[str, object],
    *,
    environment: dict[str, str] | None = None,
    tags: tuple[str, ...] = (_UNIVERSAL_TAG,),
) -> artifact_module.ComfyRegistryWheelArtifactManifest:
    return resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies(declarations),
        documents,
        marker_environment=environment or _environment(),
        supported_tags=tags,
    )


def _assert_error(
    code: str,
    declarations: list[str],
    documents: dict[str, object],
    *,
    environment: dict[str, str] | None = None,
    tags: tuple[str, ...] = (_UNIVERSAL_TAG,),
) -> None:
    with pytest.raises(ComfyRegistryWheelArtifactError) as raised:
        _resolve(declarations, documents, environment=environment, tags=tags)
    assert raised.value.code == code


def test_empty_plan_produces_stable_empty_manifest() -> None:
    first = _resolve([], {})
    second = _resolve([], {})

    assert first == second
    assert first.artifacts == ()
    assert len(first.target_sha256) == 64
    assert len(first.manifest_sha256) == 64


def test_binds_exact_requirement_to_hash_bound_compatible_wheel() -> None:
    manifest = _resolve(["Example_Package==1.2.3"], {"EXAMPLE.package": _document()})

    assert manifest.artifacts == (
        artifact_module.ComfyRegistryWheelArtifact(
            name="example-package",
            version="1.2.3",
            requirement="example-package==1.2.3",
            filename="example_package-1.2.3-py3-none-any.whl",
            url=(
                "https://files.pythonhosted.org/packages/aa/bb/"
                "example_package-1.2.3-py3-none-any.whl"
            ),
            sha256=_SHA256,
            size_bytes=1234,
            metadata_sha256=_METADATA_SHA256,
            compatibility_tag=_UNIVERSAL_TAG,
            wheel_tags=(_UNIVERSAL_TAG,),
        ),
    )


def test_prefers_the_highest_ranked_compatible_platform_wheel() -> None:
    platform = "example_package-1.2.3-cp312-cp312-win_amd64.whl"
    document = _document(_file(), _file(platform))

    manifest = _resolve(
        ["example-package==1.2.3"],
        {"example-package": document},
        tags=("cp312-cp312-win_amd64", _UNIVERSAL_TAG),
    )

    assert manifest.artifacts[0].filename == platform
    assert manifest.artifacts[0].compatibility_tag == "cp312-cp312-win_amd64"


def test_prefers_the_higher_wheel_build_at_the_same_tag_rank() -> None:
    build_one = "example_package-1.2.3-1-py3-none-any.whl"
    build_two = "example_package-1.2.3-2-py3-none-any.whl"

    manifest = _resolve(
        ["example-package==1.2.3"],
        {"example-package": _document(_file(build_one), _file(build_two))},
    )

    assert manifest.artifacts[0].filename == build_two


def test_metadata_file_order_does_not_change_manifest_identity() -> None:
    platform = _file("example_package-1.2.3-cp312-cp312-win_amd64.whl")
    universal = _file()
    tags = ("cp312-cp312-win_amd64", _UNIVERSAL_TAG)

    first = _resolve(
        ["example-package==1.2.3"],
        {"example-package": _document(platform, universal)},
        tags=tags,
    )
    second = _resolve(
        ["example-package==1.2.3"],
        {"example-package": _document(universal, platform)},
        tags=tags,
    )

    assert first == second


def test_inactive_marker_needs_no_project_metadata() -> None:
    manifest = _resolve(
        ["example-package==1.2.3; sys_platform == 'linux'"],
        {},
        environment=_environment(sys_platform="win32"),
    )

    assert manifest.artifacts == ()


def test_active_marker_is_resolved() -> None:
    manifest = _resolve(
        ["example-package==1.2.3; sys_platform == 'win32'"],
        {"example-package": _document()},
        environment=_environment(sys_platform="win32"),
    )

    assert len(manifest.artifacts) == 1


def test_overlapping_active_markers_fail_closed() -> None:
    _assert_error(
        "overlapping_dependency_markers",
        [
            "example-package==1.2.3; sys_platform == 'win32'",
            "example-package==1.2.3; os_name == 'nt'",
        ],
        {"example-package": _document()},
        environment=_environment(sys_platform="win32", os_name="nt"),
    )


def test_a_version_range_resolves_to_the_newest_version_it_admits() -> None:
    """Packages declare ranges, not pins, so a range must resolve rather than refuse."""
    manifest = _resolve(
        ["example-package>=1,<3"],
        {
            "example-package": _document(
                _file("example_package-1.2.3-py3-none-any.whl"),
                _file("example_package-2.0.0-py3-none-any.whl"),
            )
        },
    )

    assert [artifact.version for artifact in manifest.artifacts] == ["2.0.0"]


def test_the_version_is_chosen_before_the_wheel_within_it() -> None:
    """A tag preference must never outvote the version a range admits.

    Ranking every candidate wheel across several versions at once would let a
    better-matching tag on an older release win, so the newest eligible version
    is selected first and wheels are ranked only inside it.
    """
    manifest = _resolve(
        ["example-package>=1"],
        {
            "example-package": _document(
                _file("example_package-1.0.0-py3-none-any.whl"),
                _file("example_package-2.0.0-py3-none-any.whl"),
            )
        },
        tags=("py3-none-any",),
    )

    assert [artifact.version for artifact in manifest.artifacts] == ["2.0.0"]


def test_a_range_that_admits_no_compatible_wheel_still_refuses() -> None:
    _assert_error(
        "no_compatible_wheel",
        ["example-package>=9"],
        {"example-package": _document(_file("example_package-1.2.3-py3-none-any.whl"))},
    )


def test_single_requirement_selector_prefers_newest_stable_compatible_version() -> None:
    selected = select_comfy_registry_wheel_artifact(
        "example-package>=1,<3",
        _document(
            _file("example_package-1.2.3-py3-none-any.whl"),
            _file("example_package-2.0.0-py3-none-any.whl"),
            _file("example_package-2.1.0rc1-py3-none-any.whl"),
        ),
        marker_environment=_environment(),
        supported_tags=(_UNIVERSAL_TAG,),
    )

    assert selected.version == "2.0.0"
    assert selected.requirement == "example-package<3,>=1"


def test_single_requirement_selector_uses_prerelease_when_it_is_only_match() -> None:
    selected = select_comfy_registry_wheel_artifact(
        "example-package>=2",
        _document(_file("example_package-3.0.0rc1-py3-none-any.whl")),
        marker_environment=_environment(),
        supported_tags=(_UNIVERSAL_TAG,),
    )

    assert selected.version == "3.0.0rc1"


def test_single_requirement_selector_rejects_unevaluated_marker() -> None:
    with pytest.raises(ComfyRegistryWheelArtifactError) as raised:
        select_comfy_registry_wheel_artifact(
            "example-package>=1; sys_platform == 'win32'",
            _document(),
            marker_environment=_environment(),
            supported_tags=(_UNIVERSAL_TAG,),
        )

    assert raised.value.code == "invalid_wheel_requirement"


def test_explicit_wheel_target_identity_is_stable_and_order_sensitive() -> None:
    environment = _environment()
    first = comfy_registry_wheel_target_sha256(
        environment,
        ("cp312-cp312-win_amd64", _UNIVERSAL_TAG),
    )
    second = comfy_registry_wheel_target_sha256(
        dict(reversed(list(environment.items()))),
        ("cp312-cp312-win_amd64", _UNIVERSAL_TAG),
    )
    reversed_tags = comfy_registry_wheel_target_sha256(
        environment,
        (_UNIVERSAL_TAG, "cp312-cp312-win_amd64"),
    )

    assert first == second
    assert first != reversed_tags


@pytest.mark.parametrize(
    ("documents", "code"),
    [
        ({}, "missing_project_metadata"),
        (
            {
                "example-package": _document(),
                "unrequested": _document(name="unrequested"),
            },
            "unexpected_project_metadata",
        ),
    ],
)
def test_project_metadata_set_must_exactly_match_active_dependencies(
    documents: dict[str, object],
    code: str,
) -> None:
    _assert_error(code, ["example-package==1.2.3"], documents)


def test_canonical_duplicate_project_documents_fail_closed() -> None:
    _assert_error(
        "invalid_project_metadata",
        ["example-package==1.2.3"],
        {
            "example-package": _document(),
            "Example_Package": _document(),
        },
    )


@pytest.mark.parametrize(
    "document",
    [
        _document(_file("example_package-1.2.3.tar.gz")),
        _document(_file(yanked=True)),
        _document(_file(**{"requires-python": ">=4"})),
        _document(_file("example_package-1.2.3-cp311-cp311-win_amd64.whl")),
    ],
)
def test_source_yanked_python_incompatible_and_tag_incompatible_files_are_rejected(
    document: dict[str, object],
) -> None:
    _assert_error(
        "no_compatible_wheel",
        ["example-package==1.2.3"],
        {"example-package": document},
    )


@pytest.mark.parametrize(
    ("updates", "code"),
    [
        ({"hashes": {}}, "missing_wheel_hash"),
        ({"hashes": {"sha256": "not-a-digest"}}, "missing_wheel_hash"),
        ({"size": -1}, "invalid_wheel_size"),
        ({"size": True}, "invalid_wheel_size"),
        ({"core-metadata": {"sha256": "bad"}}, "missing_wheel_hash"),
        ({"yanked": 1}, "invalid_project_metadata"),
        ({"requires-python": "not a specifier"}, "invalid_project_metadata"),
    ],
)
def test_malformed_selected_file_metadata_fails_closed(
    updates: dict[str, object],
    code: str,
) -> None:
    _assert_error(
        code,
        ["example-package==1.2.3"],
        {"example-package": _document(_file(**updates))},
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://files.pythonhosted.org/packages/aa/bb/example_package-1.2.3-py3-none-any.whl",
        "https://example.com/packages/aa/bb/example_package-1.2.3-py3-none-any.whl",
        (
            "https://user@files.pythonhosted.org/packages/aa/bb/"
            "example_package-1.2.3-py3-none-any.whl"
        ),
        (
            "https://files.pythonhosted.org/packages/aa/bb/"
            "example_package-1.2.3-py3-none-any.whl?download=1"
        ),
        ("https://files.pythonhosted.org/packages/aa/bb/different-1.2.3-py3-none-any.whl"),
    ],
)
def test_wheel_urls_are_strictly_allowlisted(url: str) -> None:
    _assert_error(
        "invalid_wheel_url",
        ["example-package==1.2.3"],
        {"example-package": _document(_file(url=url))},
    )


def test_wrong_project_or_wheel_identity_fails_closed() -> None:
    _assert_error(
        "invalid_project_metadata",
        ["example-package==1.2.3"],
        {"example-package": _document(name="other")},
    )
    _assert_error(
        "invalid_project_metadata",
        ["example-package==1.2.3"],
        {"example-package": _document(_file("other-1.2.3-py3-none-any.whl"))},
    )


def test_duplicate_selected_wheel_record_fails_closed() -> None:
    record = _file()
    _assert_error(
        "duplicate_wheel_artifact",
        ["example-package==1.2.3"],
        {"example-package": _document(record, deepcopy(record))},
    )


def test_manifest_changes_when_artifact_hash_or_target_changes() -> None:
    first = _resolve(["example-package==1.2.3"], {"example-package": _document()})
    changed_hash = _resolve(
        ["example-package==1.2.3"],
        {"example-package": _document(_file(hashes={"sha256": "c" * 64}))},
    )
    changed_target = _resolve(
        ["example-package==1.2.3"],
        {"example-package": _document()},
        environment=_environment(platform_release="different"),
    )

    assert first.manifest_sha256 != changed_hash.manifest_sha256
    assert first.target_sha256 != changed_target.target_sha256
    assert first.manifest_sha256 != changed_target.manifest_sha256


def test_public_manifest_builder_and_validator_preserve_exact_manifest() -> None:
    manifest = _resolve(["example-package==1.2.3"], {"example-package": _document()})

    rebuilt = build_comfy_registry_wheel_artifact_manifest(
        manifest.declaration_sha256,
        manifest.target_sha256,
        manifest.artifacts,
    )

    assert rebuilt == manifest
    assert validate_comfy_registry_wheel_artifact_manifest(rebuilt) == manifest.artifacts


def test_manifest_validator_rejects_stale_hash_and_duplicate_packages() -> None:
    manifest = _resolve(["example-package==1.2.3"], {"example-package": _document()})
    changed = replace(manifest, manifest_sha256="0" * 64)

    with pytest.raises(ComfyRegistryWheelArtifactError) as stale:
        validate_comfy_registry_wheel_artifact_manifest(changed)
    assert stale.value.code == "artifact_manifest_hash_mismatch"

    with pytest.raises(ComfyRegistryWheelArtifactError) as duplicate:
        build_comfy_registry_wheel_artifact_manifest(
            manifest.declaration_sha256,
            manifest.target_sha256,
            (*manifest.artifacts, *manifest.artifacts),
        )
    assert duplicate.value.code == "invalid_artifact_manifest"


def test_future_minor_simple_api_is_feature_detected_but_new_major_fails() -> None:
    manifest = _resolve(
        ["example-package==1.2.3"],
        {"example-package": _document(api_version="1.99")},
    )
    assert len(manifest.artifacts) == 1

    _assert_error(
        "unsupported_project_api",
        ["example-package==1.2.3"],
        {"example-package": _document(api_version="2.0")},
    )


def test_bounds_project_documents_file_lists_wheels_and_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_module, "MAX_PYPI_PROJECT_DOCUMENT_BYTES", 10)
    _assert_error(
        "invalid_project_metadata",
        ["example-package==1.2.3"],
        {"example-package": _document()},
    )
    monkeypatch.setattr(artifact_module, "MAX_PYPI_PROJECT_DOCUMENT_BYTES", 2 * 1024 * 1024)
    monkeypatch.setattr(artifact_module, "MAX_WHEEL_ARTIFACT_BYTES", 10)
    _assert_error(
        "wheel_too_large",
        ["example-package==1.2.3"],
        {"example-package": _document()},
    )
    monkeypatch.setattr(artifact_module, "MAX_SUPPORTED_WHEEL_TAGS", 1)
    _assert_error(
        "invalid_wheel_target",
        ["example-package==1.2.3"],
        {"example-package": _document()},
        tags=("cp312-cp312-win_amd64", _UNIVERSAL_TAG),
    )


def test_resolver_accepts_project_document_above_the_former_two_mib_limit() -> None:
    document = _document()
    document["padding"] = "x" * (2 * 1024 * 1024)

    manifest = _resolve(["example-package==1.2.3"], {"example-package": document})

    assert len(manifest.artifacts) == 1


def test_target_requires_complete_environment_and_expanded_unique_tags() -> None:
    _assert_error(
        "invalid_wheel_target",
        ["example-package==1.2.3"],
        {"example-package": _document()},
        environment={"python_full_version": "3.12.0"},
    )
    _assert_error(
        "invalid_wheel_target",
        ["example-package==1.2.3"],
        {"example-package": _document()},
        tags=("py2.py3-none-any",),
    )
    _assert_error(
        "invalid_wheel_target",
        ["example-package==1.2.3"],
        {"example-package": _document()},
        tags=(_UNIVERSAL_TAG, _UNIVERSAL_TAG),
    )
    _assert_error(
        "invalid_wheel_target",
        ["example-package==1.2.3"],
        {"example-package": _document()},
        tags=("not-a-wheel-tag",),
    )


def test_non_json_project_metadata_fails_closed() -> None:
    document = _document()
    document["unexpected"] = object()

    _assert_error(
        "invalid_project_metadata",
        ["example-package==1.2.3"],
        {"example-package": document},
    )


def test_current_target_snapshot_is_explicit_and_usable() -> None:
    environment, tags = current_comfy_registry_wheel_target()

    assert _MARKER_KEYS.issubset(environment)
    assert environment["extra"] == ""
    assert tags
    assert len(tags) == len(set(tags))


_MARKER_KEYS = {
    "implementation_name",
    "implementation_version",
    "os_name",
    "platform_machine",
    "platform_python_implementation",
    "platform_release",
    "platform_system",
    "platform_version",
    "python_full_version",
    "python_version",
    "sys_platform",
}
