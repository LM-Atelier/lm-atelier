from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from packaging.markers import default_environment

import local_lm.comfy_registry_wheel_closure as closure_module
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_wheel_artifacts import (
    ComfyRegistryWheelArtifactManifest,
    resolve_comfy_registry_wheel_artifacts,
)
from local_lm.comfy_registry_wheel_closure import (
    ComfyRegistryWheelClosureError,
    advance_comfy_registry_wheel_closure,
    plan_comfy_registry_wheel_closure,
)
from local_lm.comfy_registry_wheel_selection import (
    select_comfy_registry_wheel_versions,
)

_TAG = "py3-none-any"
_SHA256 = "a" * 64


def _environment(**updates: str) -> dict[str, str]:
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    environment.update(updates)
    return environment


def _core_metadata(name: str, version: str, requirements: list[str]) -> bytes:
    headers = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
    ]
    headers.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    return ("\n".join(headers) + "\n\n").encode()


def _wheel_file(
    name: str,
    version: str,
    metadata: bytes,
) -> dict[str, object]:
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    return {
        "filename": filename,
        "url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
        "hashes": {"sha256": _SHA256},
        "requires-python": ">=3.12",
        "yanked": False,
        "size": 100,
        "core-metadata": {"sha256": hashlib.sha256(metadata).hexdigest()},
    }


def _initial(
    entries: list[tuple[str, str, list[str]]],
) -> tuple[ComfyRegistryWheelArtifactManifest, dict[str, bytes]]:
    declarations: list[str] = []
    projects: dict[str, object] = {}
    metadata_documents: dict[str, bytes] = {}
    for name, version, requirements in entries:
        metadata = _core_metadata(name, version, requirements)
        record = _wheel_file(name, version, metadata)
        declarations.append(f"{name}=={version}")
        projects[name] = {
            "meta": {"api-version": "1.4"},
            "name": name,
            "files": [record],
        }
        metadata_documents[str(record["filename"])] = metadata
    manifest = resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies(declarations),
        projects,
        marker_environment=_environment(),
        supported_tags=(_TAG,),
    )
    return manifest, metadata_documents


def _project(
    name: str,
    entries: list[tuple[str, list[str]]],
) -> tuple[dict[str, object], dict[str, bytes]]:
    files: list[object] = []
    metadata_documents: dict[str, bytes] = {}
    for version, requirements in entries:
        metadata = _core_metadata(name, version, requirements)
        record = _wheel_file(name, version, metadata)
        files.append(record)
        metadata_documents[str(record["filename"])] = metadata
    return (
        {
            "meta": {"api-version": "1.4"},
            "name": name,
            "files": files,
        },
        metadata_documents,
    )


def _selection(
    closure: closure_module.ComfyRegistryWheelClosure,
    documents: dict[str, object],
    *,
    environment: dict[str, str] | None = None,
) -> closure_module.ComfyRegistryWheelSelection:
    return select_comfy_registry_wheel_versions(
        closure.metadata_plan,
        documents,
        marker_environment=environment or _environment(),
        supported_tags=(_TAG,),
    )


def test_two_transitive_rounds_close_to_one_complete_locked_manifest() -> None:
    manifest, metadata = _initial([("alpha", "1.0", ["beta>=1"])])
    root_declaration_sha256 = manifest.declaration_sha256
    closure = plan_comfy_registry_wheel_closure(
        manifest,
        metadata,
        marker_environment=_environment(),
    )

    assert closure.round_number == 0
    assert closure.pending_projects == ("beta",)
    assert closure.complete is False

    beta_project, beta_metadata = _project(
        "beta",
        [("1.0", []), ("2.0", ["gamma>=1"])],
    )
    beta_selection = _selection(closure, {"beta": beta_project})
    beta_filename = beta_selection.artifacts[0].filename
    second = advance_comfy_registry_wheel_closure(
        closure,
        beta_selection,
        {**metadata, beta_filename: beta_metadata[beta_filename]},
        marker_environment=_environment(),
    )

    assert second.round_number == 1
    assert second.pending_projects == ("gamma",)
    assert [artifact.name for artifact in second.manifest.artifacts] == [
        "alpha",
        "beta",
    ]

    gamma_project, gamma_metadata = _project("gamma", [("3.0", [])])
    gamma_selection = _selection(second, {"gamma": gamma_project})
    gamma_filename = gamma_selection.artifacts[0].filename
    complete = advance_comfy_registry_wheel_closure(
        second,
        gamma_selection,
        {
            **metadata,
            beta_filename: beta_metadata[beta_filename],
            gamma_filename: gamma_metadata[gamma_filename],
        },
        marker_environment=_environment(),
    )

    assert complete.round_number == 2
    assert complete.pending_projects == ()
    assert complete.complete is True
    assert [artifact.name for artifact in complete.manifest.artifacts] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert len(complete.manifest_history) == 3
    assert len(complete.closure_sha256) == 64
    assert complete.manifest.declaration_sha256 == root_declaration_sha256


def test_dependency_free_manifest_is_complete_without_selection() -> None:
    manifest, metadata = _initial([("alpha", "1.0", [])])

    closure = plan_comfy_registry_wheel_closure(
        manifest,
        metadata,
        marker_environment=_environment(),
    )

    assert closure.complete is True
    assert closure.pending_projects == ()

    with pytest.raises(ComfyRegistryWheelClosureError) as raised:
        advance_comfy_registry_wheel_closure(
            closure,
            replace(
                _selection_for_unrelated_dependency(),
                artifact_manifest_sha256=manifest.manifest_sha256,
            ),
            metadata,
            marker_environment=_environment(),
        )
    assert raised.value.code == "closure_already_complete"


def test_new_metadata_conflict_fails_before_another_round() -> None:
    manifest, metadata = _initial([("alpha", "1.0", ["beta>=1"])])
    closure = plan_comfy_registry_wheel_closure(
        manifest,
        metadata,
        marker_environment=_environment(),
    )
    beta_project, beta_metadata = _project("beta", [("1.0", ["alpha>=2"])])
    selection = _selection(closure, {"beta": beta_project})
    filename = selection.artifacts[0].filename

    with pytest.raises(ComfyRegistryWheelClosureError) as raised:
        advance_comfy_registry_wheel_closure(
            closure,
            selection,
            {**metadata, filename: beta_metadata[filename]},
            marker_environment=_environment(),
        )
    assert raised.value.code == "dependency_conflict"


def test_extended_manifest_requires_complete_hash_bound_metadata_set() -> None:
    manifest, metadata = _initial([("alpha", "1.0", ["beta>=1"])])
    closure = plan_comfy_registry_wheel_closure(
        manifest,
        metadata,
        marker_environment=_environment(),
    )
    beta_project, _ = _project("beta", [("1.0", [])])
    selection = _selection(closure, {"beta": beta_project})

    with pytest.raises(ComfyRegistryWheelClosureError) as raised:
        advance_comfy_registry_wheel_closure(
            closure,
            selection,
            metadata,
            marker_environment=_environment(),
        )
    assert raised.value.code == "missing_core_metadata"


def test_selection_must_match_source_plan_manifest_target_and_frontier() -> None:
    first_manifest, first_metadata = _initial([("alpha", "1.0", ["beta>=1"])])
    second_manifest, second_metadata = _initial([("other", "1.0", ["beta>=1"])])
    first = plan_comfy_registry_wheel_closure(
        first_manifest,
        first_metadata,
        marker_environment=_environment(),
    )
    second = plan_comfy_registry_wheel_closure(
        second_manifest,
        second_metadata,
        marker_environment=_environment(),
    )
    beta_project, beta_metadata = _project("beta", [("1.0", [])])
    selection = _selection(first, {"beta": beta_project})
    filename = selection.artifacts[0].filename

    with pytest.raises(ComfyRegistryWheelClosureError) as source:
        advance_comfy_registry_wheel_closure(
            second,
            selection,
            {**second_metadata, filename: beta_metadata[filename]},
            marker_environment=_environment(),
        )
    assert source.value.code == "selection_source_mismatch"

    target_selection = _selection(
        first,
        {"beta": beta_project},
        environment=_environment(platform_release="different"),
    )
    with pytest.raises(ComfyRegistryWheelClosureError) as target:
        advance_comfy_registry_wheel_closure(
            first,
            target_selection,
            {**first_metadata, filename: beta_metadata[filename]},
            marker_environment=_environment(),
        )
    assert target.value.code == "selection_source_mismatch"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda closure: replace(closure, closure_sha256="0" * 64),
        lambda closure: replace(closure, pending_projects=()),
        lambda closure: replace(closure, manifest_history=()),
        lambda closure: replace(
            closure,
            metadata_plan=replace(
                closure.metadata_plan,
                artifact_manifest_sha256="0" * 64,
            ),
        ),
    ],
)
def test_closure_state_is_revalidated_before_advancement(mutate: object) -> None:
    manifest, metadata = _initial([("alpha", "1.0", ["beta>=1"])])
    closure = plan_comfy_registry_wheel_closure(
        manifest,
        metadata,
        marker_environment=_environment(),
    )
    beta_project, beta_metadata = _project("beta", [("1.0", [])])
    selection = _selection(closure, {"beta": beta_project})
    filename = selection.artifacts[0].filename

    with pytest.raises(ComfyRegistryWheelClosureError) as raised:
        advance_comfy_registry_wheel_closure(
            mutate(closure),  # type: ignore[operator]
            selection,
            {**metadata, filename: beta_metadata[filename]},
            marker_environment=_environment(),
        )
    assert raised.value.code in {"invalid_closure", "closure_hash_mismatch"}


def test_round_count_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, metadata = _initial([("alpha", "1.0", ["beta>=1"])])
    closure = plan_comfy_registry_wheel_closure(
        manifest,
        metadata,
        marker_environment=_environment(),
    )
    beta_project, beta_metadata = _project("beta", [("1.0", [])])
    selection = _selection(closure, {"beta": beta_project})
    filename = selection.artifacts[0].filename
    monkeypatch.setattr(closure_module, "MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS", 0)

    with pytest.raises(ComfyRegistryWheelClosureError) as raised:
        advance_comfy_registry_wheel_closure(
            closure,
            selection,
            {**metadata, filename: beta_metadata[filename]},
            marker_environment=_environment(),
        )
    assert raised.value.code == "closure_round_limit"


def _selection_for_unrelated_dependency() -> closure_module.ComfyRegistryWheelSelection:
    manifest, metadata = _initial([("source", "1.0", ["dependency>=1"])])
    closure = plan_comfy_registry_wheel_closure(
        manifest,
        metadata,
        marker_environment=_environment(),
    )
    project, _ = _project("dependency", [("1.0", [])])
    return _selection(closure, {"dependency": project})
