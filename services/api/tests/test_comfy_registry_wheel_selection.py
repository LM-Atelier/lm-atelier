from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
from packaging.markers import default_environment

import local_lm.comfy_registry_wheel_selection as selection_module
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_wheel_artifacts import (
    resolve_comfy_registry_wheel_artifacts,
)
from local_lm.comfy_registry_wheel_metadata import (
    ComfyRegistryWheelMetadataPlan,
    plan_comfy_registry_wheel_metadata,
)
from local_lm.comfy_registry_wheel_selection import (
    ComfyRegistryWheelSelectionError,
    select_comfy_registry_wheel_versions,
    validate_comfy_registry_wheel_selection,
)

_TAG = "py3-none-any"
_SHA256 = "a" * 64
_METADATA_SHA256 = "b" * 64


def _environment(**updates: str) -> dict[str, str]:
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    environment.update(updates)
    return environment


def _core_metadata(
    name: str,
    version: str,
    requirements: list[str],
) -> bytes:
    headers = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
    ]
    headers.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    return ("\n".join(headers) + "\n\n").encode()


def _metadata_plan(
    entries: list[tuple[str, str, list[str], bool]],
) -> ComfyRegistryWheelMetadataPlan:
    declarations: list[str] = []
    project_documents: dict[str, object] = {}
    metadata_documents: dict[str, bytes] = {}
    for name, version, requirements, hash_bound in entries:
        filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
        content = _core_metadata(name, version, requirements)
        declarations.append(f"{name}=={version}")
        project_documents[name] = {
            "meta": {"api-version": "1.4"},
            "name": name,
            "files": [
                {
                    "filename": filename,
                    "url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
                    "hashes": {"sha256": _SHA256},
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
            metadata_documents[filename] = content
    manifest = resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies(declarations),
        project_documents,
        marker_environment=_environment(),
        supported_tags=(_TAG,),
    )
    return plan_comfy_registry_wheel_metadata(
        manifest,
        metadata_documents,
        marker_environment=_environment(),
    )


def _file(
    name: str,
    version: str,
    *,
    tag: str = _TAG,
    yanked: bool = False,
) -> dict[str, object]:
    filename = f"{name.replace('-', '_')}-{version}-{tag}.whl"
    return {
        "filename": filename,
        "url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
        "hashes": {"sha256": _SHA256},
        "requires-python": ">=3.12",
        "yanked": yanked,
        "size": 100,
        "core-metadata": {"sha256": _METADATA_SHA256},
    }


def _document(name: str, *versions: str) -> dict[str, object]:
    return {
        "meta": {"api-version": "1.4"},
        "name": name,
        "files": [_file(name, version) for version in versions],
    }


def _select(
    plan: ComfyRegistryWheelMetadataPlan,
    documents: dict[str, object],
    *,
    environment: dict[str, str] | None = None,
    tags: tuple[str, ...] = (_TAG,),
) -> selection_module.ComfyRegistryWheelSelection:
    return select_comfy_registry_wheel_versions(
        plan,
        documents,
        marker_environment=environment or _environment(),
        supported_tags=tags,
    )


def _assert_error(
    code: str,
    plan: ComfyRegistryWheelMetadataPlan,
    documents: object,
) -> None:
    with pytest.raises(ComfyRegistryWheelSelectionError) as raised:
        select_comfy_registry_wheel_versions(
            plan,
            documents,  # type: ignore[arg-type]
            marker_environment=_environment(),
            supported_tags=(_TAG,),
        )
    assert raised.value.code == code


def test_selects_newest_wheel_satisfying_every_active_constraint() -> None:
    plan = _metadata_plan(
        [
            ("alpha", "1.0", ["shared>=1"], True),
            ("beta", "1.0", ["shared<3"], True),
        ]
    )

    selection = _select(
        plan,
        {"shared": _document("shared", "1.0", "2.0", "3.0")},
    )

    assert selection.metadata_plan_sha256 == plan.plan_sha256
    assert selection.artifact_manifest_sha256 == plan.artifact_manifest_sha256
    assert selection.artifacts[0].name == "shared"
    assert selection.artifacts[0].version == "2.0"
    assert selection.artifacts[0].requirement == "shared<3,>=1"
    assert len(selection.target_sha256) == 64
    assert len(selection.selection_sha256) == 64
    assert validate_comfy_registry_wheel_selection(selection) == selection.artifacts


def test_selection_validator_rejects_mutated_source_and_payload() -> None:
    plan = _metadata_plan([("alpha", "1.0", ["dependency>=1"], True)])
    selection = _select(
        plan,
        {"dependency": _document("dependency", "1.0")},
    )

    with pytest.raises(ComfyRegistryWheelSelectionError) as source:
        validate_comfy_registry_wheel_selection(
            replace(selection, artifact_manifest_sha256="0" * 64)
        )
    assert source.value.code == "selection_hash_mismatch"

    with pytest.raises(ComfyRegistryWheelSelectionError) as payload:
        validate_comfy_registry_wheel_selection(replace(selection, selection_sha256="0" * 64))
    assert payload.value.code == "selection_hash_mismatch"


def test_requested_extras_survive_constraint_consolidation() -> None:
    plan = _metadata_plan([("alpha", "1.0", ["dependency[image]>=1"], True)])

    selection = _select(
        plan,
        {"dependency": _document("dependency", "1.0")},
    )

    assert selection.artifacts[0].requirement == "dependency[image]>=1"


def test_satisfied_locked_frontier_needs_no_project_metadata() -> None:
    plan = _metadata_plan(
        [
            ("alpha", "1.0", ["beta>=1"], True),
            ("beta", "2.0", [], True),
        ]
    )

    selection = _select(plan, {})

    assert selection.artifacts == ()


def test_conflicts_and_unavailable_metadata_stop_selection() -> None:
    conflict = _metadata_plan(
        [
            ("alpha", "1.0", ["shared==1"], True),
            ("beta", "1.0", ["shared==2"], True),
        ]
    )
    unavailable = _metadata_plan([("alpha", "1.0", [], False)])

    _assert_error("dependency_conflict", conflict, {"shared": _document("shared", "1.0")})
    _assert_error("metadata_unavailable", unavailable, {})


def test_project_document_set_must_exactly_match_unresolved_frontier() -> None:
    plan = _metadata_plan([("alpha", "1.0", ["dependency>=1"], True)])

    _assert_error("missing_project_metadata", plan, {})
    _assert_error(
        "unexpected_project_metadata",
        plan,
        {
            "dependency": _document("dependency", "1.0"),
            "extra": _document("extra", "1.0"),
        },
    )


def test_incompatible_frontier_has_no_silent_fallback() -> None:
    plan = _metadata_plan([("alpha", "1.0", ["dependency>=2"], True)])

    _assert_error(
        "no_compatible_wheel",
        plan,
        {"dependency": _document("dependency", "1.0")},
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda plan: replace(plan, plan_sha256="0" * 64),
        lambda plan: replace(plan, artifact_manifest_sha256="0" * 64),
        lambda plan: replace(plan, resolution_required=False),
    ],
)
def test_metadata_plan_is_revalidated_before_selection(mutate: object) -> None:
    plan = _metadata_plan([("alpha", "1.0", ["dependency>=1"], True)])
    changed = mutate(plan)  # type: ignore[operator]

    with pytest.raises(ComfyRegistryWheelSelectionError) as raised:
        _select(changed, {"dependency": _document("dependency", "1.0")})

    assert raised.value.code in {
        "metadata_plan_hash_mismatch",
        "invalid_metadata_plan",
    }


def test_file_order_does_not_change_selection_identity() -> None:
    plan = _metadata_plan([("alpha", "1.0", ["dependency>=1"], True)])
    document = _document("dependency", "1.0", "2.0")
    reversed_document = {
        **document,
        "files": list(reversed(document["files"])),  # type: ignore[arg-type]
    }

    first = _select(plan, {"dependency": document})
    second = _select(plan, {"dependency": reversed_document})

    assert first == second


def test_target_changes_selection_identity() -> None:
    plan = _metadata_plan([("alpha", "1.0", ["dependency>=1"], True)])
    documents = {"dependency": _document("dependency", "1.0")}

    first = _select(plan, documents)
    second = _select(
        plan,
        documents,
        environment=_environment(platform_release="different"),
    )

    assert first.target_sha256 != second.target_sha256
    assert first.selection_sha256 != second.selection_sha256


def test_selection_artifact_count_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _metadata_plan([("alpha", "1.0", ["dependency>=1"], True)])
    monkeypatch.setattr(
        selection_module,
        "MAX_REGISTRY_WHEEL_SELECTION_ARTIFACTS",
        0,
    )

    _assert_error(
        "too_many_selection_artifacts",
        plan,
        {"dependency": _document("dependency", "1.0")},
    )
