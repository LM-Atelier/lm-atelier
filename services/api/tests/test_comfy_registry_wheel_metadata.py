from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest
from packaging.markers import default_environment

import local_lm.comfy_registry_wheel_metadata as metadata_module
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactManifest,
    resolve_comfy_registry_wheel_artifacts,
)
from local_lm.comfy_registry_wheel_metadata import (
    ComfyRegistryWheelMetadataError,
    plan_comfy_registry_wheel_metadata,
)

_TAG = "py3-none-any"


def _environment(**updates: str) -> dict[str, str]:
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    environment.update(updates)
    return environment


def _core_metadata(
    name: str,
    version: str,
    requirements: list[str] | None = None,
    *,
    metadata_version: str = "2.4",
) -> bytes:
    headers = [
        f"Metadata-Version: {metadata_version}",
        f"Name: {name}",
        f"Version: {version}",
    ]
    headers.extend(f"Requires-Dist: {requirement}" for requirement in requirements or [])
    return ("\n".join(headers) + "\n\nDescription body is ignored.\n").encode()


def _inputs(
    entries: list[tuple[str, str, str, list[str], bool]] | None = None,
) -> tuple[ComfyRegistryWheelArtifactManifest, dict[str, bytes]]:
    selected = (
        entries
        if entries is not None
        else [("example-package", "1.2.3", "example-package==1.2.3", [], True)]
    )
    declarations: list[str] = []
    projects: dict[str, object] = {}
    metadata: dict[str, bytes] = {}
    for name, version, declaration, requirements, hash_bound in selected:
        filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        content = _core_metadata(name, version, requirements)
        declarations.append(declaration)
        projects[name] = {
            "meta": {"api-version": "1.4"},
            "name": name,
            "files": [
                {
                    "filename": filename,
                    "url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
                    "hashes": {"sha256": "a" * 64},
                    "requires-python": ">=3.12",
                    "yanked": False,
                    "size": 100,
                    "core-metadata": (
                        {"sha256": hashlib.sha256(content).hexdigest()} if hash_bound else False
                    ),
                }
            ],
        }
        if hash_bound:
            metadata[filename] = content
    manifest = resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies(declarations),
        projects,
        marker_environment=_environment(),
        supported_tags=(_TAG,),
    )
    return manifest, metadata


def _plan(
    entries: list[tuple[str, str, str, list[str], bool]] | None = None,
    *,
    environment: dict[str, str] | None = None,
) -> metadata_module.ComfyRegistryWheelMetadataPlan:
    manifest, documents = _inputs(entries)
    return plan_comfy_registry_wheel_metadata(
        manifest,
        documents,
        marker_environment=environment or _environment(),
    )


def _assert_error(
    code: str,
    manifest: ComfyRegistryWheelArtifactManifest,
    documents: Any,
    *,
    environment: dict[str, str] | None = None,
) -> None:
    with pytest.raises(ComfyRegistryWheelMetadataError) as raised:
        plan_comfy_registry_wheel_metadata(
            manifest,
            documents,
            marker_environment=environment or _environment(),
        )
    assert raised.value.code == code


def _rehash_manifest(
    manifest: ComfyRegistryWheelArtifactManifest,
    artifacts: tuple[metadata_module.ComfyRegistryWheelArtifact, ...],
) -> ComfyRegistryWheelArtifactManifest:
    payload = {
        "version": 1,
        "declaration_sha256": manifest.declaration_sha256,
        "target_sha256": manifest.target_sha256,
        "artifacts": [metadata_module._artifact_payload(artifact) for artifact in artifacts],
    }
    return replace(
        manifest,
        artifacts=artifacts,
        manifest_sha256=metadata_module._payload_sha256(payload),
    )


def test_dependency_free_metadata_produces_stable_empty_frontier() -> None:
    first = _plan()
    second = _plan()

    assert first == second
    assert first.requirements == ()
    assert first.frontier == ()
    assert first.unavailable_metadata == ()
    assert first.resolution_required is False
    assert first.conflicts == ()
    assert len(first.plan_sha256) == 64


def test_canonicalizes_active_transitive_requirement_and_frontier() -> None:
    plan = _plan(
        [
            (
                "example-package",
                "1.2.3",
                "example-package==1.2.3",
                ["Other_Package[Second,First]>=2 ; python_version >= '3.12'"],
                True,
            )
        ]
    )

    assert plan.requirements == (
        metadata_module.ComfyRegistryWheelMetadataRequirement(
            source_name="example-package",
            source_version="1.2.3",
            name="other-package",
            requirement=('other-package[first,second]>=2; python_version >= "3.12"'),
            specifier=">=2",
            marker='python_version >= "3.12"',
            extras=("first", "second"),
        ),
    )
    assert plan.frontier == (
        metadata_module.ComfyRegistryWheelMetadataFrontier(
            name="other-package",
            requirements=('other-package[first,second]>=2; python_version >= "3.12"',),
            sources=("example-package",),
            requested_extras=("first", "second"),
            locked_version=None,
            status="resolve",
        ),
    )
    assert plan.resolution_required is True


def test_environment_markers_are_evaluated_against_explicit_target() -> None:
    entries = [
        (
            "example-package",
            "1.2.3",
            "example-package==1.2.3",
            [
                "windows-only==1; sys_platform == 'win32'",
                "linux-only==1; sys_platform == 'linux'",
            ],
            True,
        )
    ]

    windows = _plan(entries, environment=_environment(sys_platform="win32"))
    linux = _plan(entries, environment=_environment(sys_platform="linux"))

    assert [item.name for item in windows.requirements] == ["windows-only"]
    assert [item.name for item in linux.requirements] == ["linux-only"]
    assert windows.plan_sha256 != linux.plan_sha256


def test_only_requested_extra_dependencies_are_active() -> None:
    entries = [
        (
            "example-package",
            "1.2.3",
            "example-package[image]==1.2.3",
            [
                "image-dependency==1; extra == 'image'",
                "video-dependency==1; extra == 'video'",
            ],
            True,
        )
    ]

    plan = _plan(entries)

    assert [item.name for item in plan.requirements] == ["image-dependency"]


def test_locked_dependency_extras_propagate_to_fixed_point() -> None:
    entries = [
        ("alpha", "1.0", "alpha==1.0", ["beta[feature]==2.0"], True),
        (
            "beta",
            "2.0",
            "beta==2.0",
            ["optional==3.0; extra == 'feature'"],
            True,
        ),
    ]

    plan = _plan(entries)

    assert [item.name for item in plan.requirements] == ["beta", "optional"]
    assert plan.frontier[0].name == "beta"
    assert plan.frontier[0].requested_extras == ("feature",)
    assert plan.frontier[0].status == "satisfied"
    assert plan.frontier[1].name == "optional"
    assert plan.frontier[1].status == "resolve"


def test_locked_version_is_satisfied_or_reported_as_conflict() -> None:
    satisfied = _plan(
        [
            ("alpha", "1.0", "alpha==1.0", ["beta>=2"], True),
            ("beta", "2.0", "beta==2.0", [], True),
        ]
    )
    conflict = _plan(
        [
            ("alpha", "1.0", "alpha==1.0", ["beta>=3"], True),
            ("beta", "2.0", "beta==2.0", [], True),
        ]
    )

    assert satisfied.frontier[0].status == "satisfied"
    assert satisfied.conflicts == ()
    assert conflict.frontier[0].status == "conflict"
    assert conflict.conflicts == ("beta",)
    assert conflict.resolution_required is True


def test_conflicting_exact_transitive_pins_are_reported() -> None:
    plan = _plan(
        [
            ("alpha", "1.0", "alpha==1.0", ["shared==1"], True),
            ("beta", "2.0", "beta==2.0", ["shared==2"], True),
        ]
    )

    assert len(plan.frontier) == 1
    assert plan.frontier[0].requirements == ("shared==1", "shared==2")
    assert plan.frontier[0].sources == ("alpha", "beta")
    assert plan.frontier[0].status == "conflict"
    assert plan.conflicts == ("shared",)
    assert plan.resolution_required is True


def test_duplicate_metadata_requirements_are_deduplicated() -> None:
    plan = _plan(
        [
            (
                "example-package",
                "1.2.3",
                "example-package==1.2.3",
                ["shared>=1", "shared>=1"],
                True,
            )
        ]
    )

    assert len(plan.requirements) == 1
    assert plan.frontier[0].requirements == ("shared>=1",)


def test_missing_hash_bound_metadata_and_unexpected_metadata_fail_closed() -> None:
    manifest, documents = _inputs()
    filename = next(iter(documents))
    _assert_error("missing_core_metadata", manifest, {})
    _assert_error(
        "unexpected_core_metadata",
        manifest,
        {**documents, "unexpected.whl": b"metadata"},
    )
    assert filename


def test_artifact_without_hash_bound_metadata_is_reported_unavailable() -> None:
    plan = _plan(
        [
            (
                "example-package",
                "1.2.3",
                "example-package==1.2.3",
                [],
                False,
            )
        ]
    )

    assert plan.requirements == ()
    assert plan.unavailable_metadata == ("example_package-1.2.3-py3-none-any.whl",)
    assert plan.resolution_required is True


def test_metadata_hash_mismatch_fails_closed() -> None:
    manifest, documents = _inputs()
    filename = next(iter(documents))

    _assert_error(
        "core_metadata_hash_mismatch",
        manifest,
        {filename: documents[filename] + b"changed"},
    )


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda content: content.replace(b"Name: example-package", b"Name: another-package"),
            "core_metadata_hash_mismatch",
        ),
        (
            lambda content: content.replace(b"Metadata-Version: 2.4", b"Metadata-Version: 9.0"),
            "core_metadata_hash_mismatch",
        ),
    ],
)
def test_content_changes_cannot_bypass_hash_binding(mutate: Any, code: str) -> None:
    manifest, documents = _inputs()
    filename, content = next(iter(documents.items()))

    _assert_error(code, manifest, {filename: mutate(content)})


def test_identity_and_metadata_version_are_validated_after_hash_verification() -> None:
    manifest, documents = _inputs()
    filename = next(iter(documents))
    artifact = manifest.artifacts[0]

    wrong_identity = _core_metadata("another-package", artifact.version)
    identity_manifest = _rehash_manifest(
        manifest,
        (
            replace(
                artifact,
                metadata_sha256=hashlib.sha256(wrong_identity).hexdigest(),
            ),
        ),
    )
    _assert_error(
        "core_metadata_identity_mismatch",
        identity_manifest,
        {filename: wrong_identity},
    )

    unsupported = _core_metadata(artifact.name, artifact.version, metadata_version="9.0")
    unsupported_manifest = _rehash_manifest(
        manifest,
        (
            replace(
                artifact,
                metadata_sha256=hashlib.sha256(unsupported).hexdigest(),
            ),
        ),
    )
    _assert_error(
        "unsupported_core_metadata",
        unsupported_manifest,
        {filename: unsupported},
    )


@pytest.mark.parametrize(
    "content",
    [
        b"Metadata-Version: 2.4\nVersion: 1.2.3\n\n",
        (b"Metadata-Version: 2.4\nName: example-package\nName: duplicate\nVersion: 1.2.3\n\n"),
        b"Metadata-Version: 2.4\nName: example-package\n\n",
    ],
)
def test_required_singleton_headers_are_enforced(content: bytes) -> None:
    manifest, documents = _inputs()
    filename = next(iter(documents))
    artifact = manifest.artifacts[0]
    changed = _rehash_manifest(
        manifest,
        (replace(artifact, metadata_sha256=hashlib.sha256(content).hexdigest()),),
    )

    _assert_error("invalid_core_metadata", changed, {filename: content})


@pytest.mark.parametrize(
    ("requirement", "code"),
    [
        ("not a valid requirement !!!", "invalid_transitive_requirement"),
        (
            "dependency @ https://example.com/dependency.whl",
            "direct_transitive_url",
        ),
        ("dependency==1\x00", "invalid_core_metadata"),
    ],
)
def test_invalid_and_direct_transitive_requirements_fail_closed(
    requirement: str,
    code: str,
) -> None:
    manifest, documents = _inputs()
    filename = next(iter(documents))
    artifact = manifest.artifacts[0]
    content = _core_metadata(artifact.name, artifact.version, [requirement])
    changed = _rehash_manifest(
        manifest,
        (replace(artifact, metadata_sha256=hashlib.sha256(content).hexdigest()),),
    )

    _assert_error(code, changed, {filename: content})


def test_metadata_byte_line_requirement_and_extra_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, documents = _inputs()
    filename, content = next(iter(documents.items()))

    monkeypatch.setattr(metadata_module, "MAX_WHEEL_CORE_METADATA_BYTES", 10)
    _assert_error("core_metadata_too_large", manifest, documents)

    monkeypatch.setattr(metadata_module, "MAX_WHEEL_CORE_METADATA_BYTES", 1024 * 1024)
    monkeypatch.setattr(metadata_module, "MAX_WHEEL_CORE_METADATA_LINES", 1)
    _assert_error("invalid_core_metadata", manifest, documents)

    monkeypatch.setattr(metadata_module, "MAX_WHEEL_CORE_METADATA_LINES", 10_000)
    monkeypatch.setattr(metadata_module, "MAX_WHEEL_CORE_METADATA_LINE_BYTES", 5)
    _assert_error("invalid_core_metadata", manifest, documents)

    monkeypatch.setattr(metadata_module, "MAX_WHEEL_CORE_METADATA_LINE_BYTES", 10_000)
    monkeypatch.setattr(metadata_module, "MAX_WHEEL_REQUIRES_DIST", 0)
    dependency_content = _core_metadata("example-package", "1.2.3", ["dependency"])
    dependency_manifest = _rehash_manifest(
        manifest,
        (
            replace(
                manifest.artifacts[0],
                metadata_sha256=hashlib.sha256(dependency_content).hexdigest(),
            ),
        ),
    )
    _assert_error(
        "too_many_transitive_requirements",
        dependency_manifest,
        {filename: dependency_content},
    )
    assert content


def test_global_requirement_and_extra_frontier_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        (
            "example-package",
            "1.2.3",
            "example-package==1.2.3",
            ["dependency[feature]==1"],
            True,
        )
    ]
    manifest, documents = _inputs(entries)

    monkeypatch.setattr(metadata_module, "MAX_WHEEL_TRANSITIVE_REQUIREMENTS", 0)
    _assert_error("too_many_transitive_requirements", manifest, documents)

    monkeypatch.setattr(metadata_module, "MAX_WHEEL_TRANSITIVE_REQUIREMENTS", 4_096)
    monkeypatch.setattr(metadata_module, "MAX_WHEEL_TRANSITIVE_EXTRAS", 0)
    locked_entries = [
        ("alpha", "1.0", "alpha==1.0", ["beta[feature]==2"], True),
        ("beta", "2.0", "beta==2.0", [], True),
    ]
    locked_manifest, locked_documents = _inputs(locked_entries)
    _assert_error(
        "too_many_transitive_extras",
        locked_manifest,
        locked_documents,
    )


def test_invalid_document_mapping_environment_and_duplicate_artifacts_fail_closed() -> None:
    manifest, documents = _inputs()
    _assert_error("invalid_core_metadata", manifest, [])
    _assert_error(
        "invalid_marker_environment",
        manifest,
        documents,
        environment={"python_full_version": "3.12.0"},
    )
    duplicate = replace(
        manifest,
        artifacts=(manifest.artifacts[0], manifest.artifacts[0]),
    )
    _assert_error("invalid_artifact_manifest", duplicate, documents)

    invalid_requirement = replace(
        manifest,
        artifacts=(replace(manifest.artifacts[0], requirement="not valid !!!"),),
    )
    _assert_error("invalid_artifact_manifest", invalid_requirement, documents)

    stale_manifest = replace(
        manifest,
        artifacts=(replace(manifest.artifacts[0], requirement="example-package>=1"),),
    )
    _assert_error("artifact_manifest_hash_mismatch", stale_manifest, documents)


def test_input_order_does_not_change_plan_identity() -> None:
    first = _plan(
        [
            ("zeta", "1.0", "zeta==1.0", ["shared>=1"], True),
            ("alpha", "2.0", "alpha==2.0", ["other==3"], True),
        ]
    )
    second = _plan(
        [
            ("alpha", "2.0", "alpha==2.0", ["other==3"], True),
            ("zeta", "1.0", "zeta==1.0", ["shared>=1"], True),
        ]
    )

    assert first == second
