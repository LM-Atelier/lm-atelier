from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import pytest
from sqlalchemy.orm import Session
from test_comfy_registry_closure_driver import _project, _Sources
from test_comfy_registry_mixed_wheel_metadata import _mixed
from test_comfy_registry_source_artifacts import source_review_context as source_review_context
from test_comfy_registry_wheel_metadata import _environment

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_closure_driver import (
    ClosurePhase,
    ComfyRegistryWheelClosureDriverError,
    RegistryClosureProgress,
)
from local_lm.comfy_registry_wheel_artifacts import ComfyRegistryWheelArtifactManifest
from local_lm.comfy_registry_wheel_inputs_v1 import ComfyRegistryWheelInputManifest

if TYPE_CHECKING:
    from local_lm.comfy_registry_mixed_wheel_closure import ComfyRegistryMixedWheelClosure


async def _drive(
    manifest: ComfyRegistryWheelInputManifest,
    local: Mapping[str, bytes],
    sources: _Sources,
    *,
    environment: dict[str, str] | None = None,
    tags: Sequence[str] = ("py3-none-any",),
    runtime: dict[str, str] | None = None,
    progress: RegistryClosureProgress | None = None,
) -> ComfyRegistryMixedWheelClosure:
    from local_lm.comfy_registry_mixed_closure_driver import (
        drive_comfy_registry_mixed_wheel_closure,
    )

    return await drive_comfy_registry_mixed_wheel_closure(
        manifest,
        local,
        project_fetcher=sources.fetch_projects,
        metadata_fetcher=sources.fetch_metadata,
        marker_environment=environment if environment is not None else _environment(),
        supported_tags=tags,
        runtime_distributions=runtime or {},
        progress=progress,
    )


@pytest.mark.asyncio
async def test_fetches_only_remote_metadata_through_cross_source_extras(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    manifest, documents = _mixed(
        source_review_context,
        ["beta>=1", 'alpha>=3; extra == "deep"'],
        [("root", "1.0", "root==1.0", [], True)],
    )
    beta, beta_documents = _project("beta", [("2.0", ["example-pkg[deep]>=1"])])
    alpha, alpha_documents = _project("alpha", [("3.0", [])])
    local = {item.filename: documents[item.filename] for item in manifest.reviewed_local}
    remote = {item.filename: documents[item.filename] for item in manifest.remote.artifacts}
    sources = _Sources(
        {"beta": beta, "alpha": alpha}, {**remote, **beta_documents, **alpha_documents}
    )
    progress: list[tuple[str, int, tuple[str, ...]]] = []

    async def record(phase: ClosurePhase, number: int, names: tuple[str, ...]) -> None:
        progress.append((phase, number, names))

    result = await _drive(manifest, local, sources, progress=record)
    assert result.complete and result.round_number == 2
    assert result.manifest.reviewed_local == manifest.reviewed_local
    assert result.manifest.declaration_sha256 == manifest.declaration_sha256
    assert result.manifest.remote.declaration_sha256 == manifest.remote.declaration_sha256
    assert sources.project_calls == [("beta",), ("alpha",)]
    assert sources.metadata_calls == [tuple(remote), tuple(beta_documents), tuple(alpha_documents)]
    assert [item.name for item in result.manifest.remote.artifacts] == ["alpha", "beta", "root"]
    assert [(phase, number) for phase, number, _ in progress] == [
        ("fetching_metadata", 0),
        ("validating_closure", 0),
        ("fetching_projects", 1),
        ("selecting_wheels", 1),
        ("fetching_metadata", 1),
        ("validating_closure", 1),
        ("fetching_projects", 2),
        ("selecting_wheels", 2),
        ("fetching_metadata", 2),
        ("validating_closure", 2),
    ]


@pytest.mark.asyncio
async def test_completed_local_graph_and_runtime_need_no_fetches(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    manifest, documents = _mixed(source_review_context, ["helper>=2"], [])
    sources = _Sources({}, {})
    result = await _drive(manifest, documents, sources, runtime={"helper": "2.0"})
    assert result.complete and result.round_number == 0
    assert not sources.project_calls and not sources.metadata_calls
    assert result.manifest == manifest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem", ["environment", "tags", "missing", "extra", "hash", "size", "type"]
)
async def test_invalid_target_or_local_metadata_refuses_before_fetching(
    source_review_context: tuple[Session, ArtifactStore],
    problem: str,
) -> None:
    manifest, documents = _mixed(
        source_review_context, [], [("root", "1.0", "root==1.0", [], True)]
    )
    filename = manifest.reviewed_local[0].filename
    local = {filename: documents[filename]}
    environment = _environment()
    tags = ("py3-none-any",)
    if problem == "environment":
        environment["python_version"] = "3.13"
    elif problem == "tags":
        tags = ("py2-none-any",)
    elif problem == "missing":
        local.clear()
    elif problem == "extra":
        local["extra.whl"] = b"extra"
    elif problem == "hash":
        local[filename] += b"changed"
    elif problem == "size":
        local[filename] = b"x" * (1024 * 1024 + 1)
    else:
        local[filename] = cast(bytes, bytearray(local[filename]))
    sources = _Sources({}, documents)
    with pytest.raises(ComfyRegistryWheelClosureDriverError) as caught:
        await _drive(manifest, local, sources, environment=environment, tags=tags)
    expected = {
        "environment": "wheel_input_target_mismatch",
        "tags": "wheel_input_target_mismatch",
        "missing": "core_metadata_set_mismatch",
        "extra": "core_metadata_set_mismatch",
        "hash": "core_metadata_hash_mismatch",
        "size": "core_metadata_too_large",
        "type": "invalid_core_metadata",
    }
    assert caught.value.code == expected[problem]
    assert not sources.project_calls and not sources.metadata_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["missing", "extra", "local", "hash", "unavailable"])
async def test_remote_metadata_cannot_replace_or_add_unrequested_inputs(
    source_review_context: tuple[Session, ArtifactStore],
    problem: str,
) -> None:
    manifest, local = _mixed(source_review_context, ["helper>=2"], [])
    project, metadata = _project(
        "helper", [("2.0", [])], metadata_available=problem != "unavailable"
    )

    class BrokenSources(_Sources):
        async def fetch_metadata(
            self, requested: ComfyRegistryWheelArtifactManifest
        ) -> Mapping[str, bytes]:
            result = dict(await super().fetch_metadata(requested))
            if problem == "missing":
                result.clear()
            elif problem == "extra":
                result["extra.whl"] = b"extra"
            elif problem == "local":
                result.update(local)
            elif problem == "hash":
                result[requested.artifacts[0].filename] += b"changed"
            return result

    sources = BrokenSources({"helper": project}, metadata)
    with pytest.raises(ComfyRegistryWheelClosureDriverError) as caught:
        await _drive(manifest, local, sources)
    assert caught.value.code == (
        "metadata_unavailable"
        if problem == "unavailable"
        else "core_metadata_hash_mismatch"
        if problem == "hash"
        else "core_metadata_set_mismatch"
    )
    assert sources.project_calls == [("helper",)]


@pytest.mark.asyncio
async def test_total_size_limit_includes_local_roots_and_prior_remote_rounds(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import comfy_registry_mixed_closure_driver as module

    manifest, local = _mixed(source_review_context, ["beta>=1"], [])
    beta, beta_metadata = _project("beta", [("1.0", ["alpha>=1"])])
    alpha, alpha_metadata = _project("alpha", [("1.0", [])])
    total = sum(map(len, (*local.values(), *beta_metadata.values(), *alpha_metadata.values())))
    monkeypatch.setattr(module, "MAX_REGISTRY_WHEEL_METADATA_TOTAL_BYTES", total - 1)
    sources = _Sources({"beta": beta, "alpha": alpha}, {**beta_metadata, **alpha_metadata})
    with pytest.raises(ComfyRegistryWheelClosureDriverError) as caught:
        await _drive(manifest, local, sources)
    assert caught.value.code == "metadata_total_too_large"
    assert sources.project_calls == [("beta",), ("alpha",)]
    monkeypatch.setattr(module, "MAX_REGISTRY_WHEEL_METADATA_TOTAL_BYTES", total)
    result = await _drive(manifest, local, sources)
    assert result.complete


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["project", "metadata", "progress"])
async def test_cancellation_propagates_without_more_fetches(
    source_review_context: tuple[Session, ArtifactStore],
    stage: str,
) -> None:
    manifest, local = _mixed(source_review_context, ["helper>=2"], [])
    project, metadata = _project("helper", [("2.0", ["leaf>=1"])])

    class CancelledSources(_Sources):
        async def fetch_projects(self, names: Sequence[str]) -> Mapping[str, object]:
            result = await super().fetch_projects(names)
            if stage == "project":
                raise asyncio.CancelledError
            return result

        async def fetch_metadata(
            self, requested: ComfyRegistryWheelArtifactManifest
        ) -> Mapping[str, bytes]:
            result = await super().fetch_metadata(requested)
            if stage == "metadata":
                raise asyncio.CancelledError
            return result

    async def cancel(phase: ClosurePhase, number: int, names: tuple[str, ...]) -> None:
        if stage == "progress":
            raise asyncio.CancelledError

    sources = CancelledSources({"helper": project}, metadata)
    with pytest.raises(asyncio.CancelledError):
        await _drive(manifest, local, sources, progress=cancel)
    assert sources.project_calls == ([] if stage == "progress" else [("helper",)])
    assert len(sources.metadata_calls) == (1 if stage == "metadata" else 0)


@pytest.mark.asyncio
async def test_progress_errors_do_not_abort_and_caller_mutation_does_not_change_target(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    manifest, local = _mixed(source_review_context, ["helper>=2"], [])
    project, metadata = _project("helper", [("2.0", [])])
    environment = _environment()
    tags = ["py3-none-any"]
    runtime = {"other": "1.0"}

    async def mutate(phase: ClosurePhase, number: int, names: tuple[str, ...]) -> None:
        environment["python_version"] = "2.7"
        tags[:] = ["py2-none-any"]
        runtime["helper"] = "0.1"
        local.clear()
        raise ValueError("Progress display unavailable")

    sources = _Sources({"helper": project}, metadata)
    result = await _drive(
        manifest,
        local,
        sources,
        environment=environment,
        tags=tags,
        runtime=runtime,
        progress=mutate,
    )
    assert result.complete and sources.project_calls == [("helper",)]
    assert result.manifest.reviewed_local == manifest.reviewed_local
    assert [(item.name, item.version) for item in result.runtime_distributions] == [
        ("other", "1.0")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["project", "metadata"])
async def test_fetch_errors_preserve_retry_information(
    source_review_context: tuple[Session, ArtifactStore],
    stage: str,
) -> None:
    manifest, local = _mixed(source_review_context, ["helper>=2"], [])
    project, metadata = _project("helper", [("2.0", [])])

    class Limited(ValueError):
        code = "registry_rate_limited"
        retry_after_seconds = 17

    class LimitedSources(_Sources):
        async def fetch_projects(self, names: Sequence[str]) -> Mapping[str, object]:
            if stage == "project":
                raise Limited("Retry later")
            return await super().fetch_projects(names)

        async def fetch_metadata(
            self, requested: ComfyRegistryWheelArtifactManifest
        ) -> Mapping[str, bytes]:
            raise Limited("Retry later")

    with pytest.raises(ComfyRegistryWheelClosureDriverError) as caught:
        await _drive(manifest, local, LimitedSources({"helper": project}, metadata))
    assert caught.value.code == "registry_rate_limited"
    assert caught.value.retry_after_seconds == 17


@pytest.mark.asyncio
async def test_driver_round_limit_prevents_another_fetch(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import comfy_registry_mixed_closure_driver as module

    manifest, local = _mixed(source_review_context, ["helper>=2"], [])
    sources = _Sources({}, {})
    monkeypatch.setattr(module, "MAX_REGISTRY_WHEEL_CLOSURE_ROUNDS", 0)
    with pytest.raises(ComfyRegistryWheelClosureDriverError) as caught:
        await _drive(manifest, local, sources)
    assert caught.value.code == "closure_round_limit"
    assert not sources.project_calls and not sources.metadata_calls
