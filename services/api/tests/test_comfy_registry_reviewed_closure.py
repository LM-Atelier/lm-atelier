from __future__ import annotations

import asyncio
import hashlib
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from test_comfy_registry_closure_driver import _project, _resolution, _Sources
from test_comfy_registry_reviewed_wheel_metadata import _HEADERS, _review_metadata
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)
from test_comfy_registry_source_artifacts import DECLARATION
from test_comfy_registry_wheel_artifacts import _environment

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_closure_driver import (
    ClosurePhase,
    ComfyRegistryWheelClosureDriverError,
    RegistryClosureProgress,
    drive_comfy_registry_wheel_closure,
)
from local_lm.comfy_registry_source_artifacts import (
    VerifiedSourceWheel,
    record_local_source_artifact_review,
    verified_reviewed_source_wheel,
)
from local_lm.models import Artifact, ComfyRegistrySourceArtifactReview

if TYPE_CHECKING:
    from local_lm.comfy_registry_mixed_wheel_closure import ComfyRegistryMixedWheelClosure


def _review(
    context: tuple[Session, ArtifactStore],
    requirements: Sequence[str] = (),
    declaration: str = DECLARATION,
) -> None:
    session, store = context
    metadata = (
        _HEADERS + b"".join(f"Requires-Dist: {item}\n".encode() for item in requirements) + b"\n"
    )
    _review_metadata(session, store, metadata)
    if declaration != DECLARATION:
        wheel = verified_reviewed_source_wheel(session, store, declaration=DECLARATION)
        session.execute(delete(ComfyRegistrySourceArtifactReview))
        session.commit()
        record_local_source_artifact_review(
            session, store, declaration=declaration, artifact_id=wheel.artifact_id
        )
        session.commit()


async def _drive(
    declarations: Sequence[str],
    context: tuple[Session, ArtifactStore],
    sources: _Sources,
    *,
    environment: dict[str, str] | None = None,
    tags: Sequence[str] = ("py3-none-any",),
    runtime: dict[str, str] | None = None,
    progress: RegistryClosureProgress | None = None,
    sessions: list[Session] | None = None,
) -> ComfyRegistryMixedWheelClosure:
    from local_lm.comfy_registry_reviewed_closure import resolve_comfy_registry_reviewed_closure

    original, store = context
    engine = original.get_bind()

    def fresh() -> Session:
        session = Session(engine)
        if sessions is not None:
            sessions.append(session)
        return session

    return await resolve_comfy_registry_reviewed_closure(
        declarations,
        session_factory=fresh,
        store=store,
        project_fetcher=sources.fetch_projects,
        metadata_fetcher=sources.fetch_metadata,
        marker_environment=environment if environment is not None else _environment(),
        supported_tags=tags,
        runtime_distributions=runtime or {},
        progress=progress,
    )


async def test_resolves_declared_roots_and_cross_source_extras_in_one_closure(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    _review(source_review_context, ['helper>=2; extra == "fast"'])
    root, root_metadata = _project("remote", [("1.0", ["example-pkg[fast]>=1"])])
    helper, helper_metadata = _project("helper", [("2.0", [])])
    sources = _Sources({"remote": root, "helper": helper}, {**root_metadata, **helper_metadata})
    declarations = [DECLARATION, "remote>=1"]
    sessions: list[Session] = []
    result = await _drive(declarations, source_review_context, sources, sessions=sessions)
    assert result.complete and result.round_number == 1
    assert [item.name for item in result.manifest.remote.artifacts] == ["helper", "remote"]
    assert result.manifest.reviewed_local[0].declaration == DECLARATION
    assert (
        result.manifest.declaration_sha256
        == plan_comfy_registry_mixed_dependencies(declarations).declaration_sha256
    )
    assert sources.project_calls == [("remote",), ("helper",)]
    assert sources.metadata_calls == [tuple(root_metadata), tuple(helper_metadata)]
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert all(not session.in_transaction() for session in sessions)


async def test_root_extras_select_local_metadata_dependencies(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    declaration = DECLARATION.replace("example-pkg @", "example-pkg[fast] @")
    _review(source_review_context, ['helper>=2; extra == "fast"'], declaration)
    helper, metadata = _project("helper", [("2.0", [])])
    sources = _Sources({"helper": helper}, metadata)
    result = await _drive([declaration], source_review_context, sources)
    assert result.complete and result.manifest.reviewed_local[0].declaration == declaration
    assert sources.project_calls == [("helper",)]


async def test_remote_only_roots_match_existing_resolution_and_never_open_reviews(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    project, metadata = _project("remote", [("2.0", ["runtime>=1"])])
    sources = _Sources({"remote": project}, metadata)
    sessions: list[Session] = []
    result = await _drive(
        ["remote>=1"], source_review_context, sources, runtime={"runtime": "1.0"}, sessions=sessions
    )
    baseline = await drive_comfy_registry_wheel_closure(
        _resolution("remote>=1"),
        project_fetcher=sources.fetch_projects,
        metadata_fetcher=sources.fetch_metadata,
        marker_environment=_environment(),
        supported_tags=("py3-none-any",),
        runtime_distributions={"runtime": "1.0"},
    )
    assert result.manifest.remote == baseline.closure.manifest
    assert result.manifest.declaration_sha256 == baseline.closure.manifest.declaration_sha256
    assert result.runtime_distributions == baseline.closure.runtime_distributions
    assert not result.manifest.reviewed_local and sessions == []


async def test_inactive_unreviewed_source_stays_bound_without_a_review_lookup(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    declaration = DECLARATION + ' ; sys_platform == "missing-platform"'
    sessions: list[Session] = []
    sources = _Sources({}, {})
    result = await _drive([declaration], source_review_context, sources, sessions=sessions)
    assert result.complete and not result.manifest.reviewed_local and sessions == []
    assert not sources.project_calls and not sources.metadata_calls
    assert (
        result.manifest.declaration_sha256
        == plan_comfy_registry_mixed_dependencies([declaration]).declaration_sha256
    )
    assert (
        result.manifest.declaration_sha256
        != plan_comfy_registry_mixed_dependencies([]).declaration_sha256
    )


@pytest.mark.parametrize(
    "other", [DECLARATION + ' ; python_version >= "3"', 'example-pkg>=1; python_version >= "3"']
)
async def test_overlapping_active_markers_refuse_before_review_or_network(
    source_review_context: tuple[Session, ArtifactStore],
    other: str,
) -> None:
    sessions: list[Session] = []
    sources = _Sources({}, {})
    with pytest.raises(ComfyRegistryWheelClosureDriverError) as error:
        await _drive([DECLARATION, other], source_review_context, sources, sessions=sessions)
    assert error.value.code == "overlapping_dependency_markers"
    assert sessions == [] and not sources.project_calls


@pytest.mark.parametrize("version", ["1.2.3", "9.0"])
async def test_exact_source_cannot_alias_an_installed_runtime_package(
    source_review_context: tuple[Session, ArtifactStore],
    version: str,
) -> None:
    sources = _Sources({}, {})
    sessions: list[Session] = []
    with pytest.raises(ComfyRegistryWheelClosureDriverError) as error:
        await _drive(
            [DECLARATION],
            source_review_context,
            sources,
            runtime={"example-pkg": version},
            sessions=sessions,
        )
    assert error.value.code == "managed_runtime_source_conflict"
    assert error.value.requirement == DECLARATION
    assert sessions == [] and not sources.project_calls


@pytest.mark.parametrize("problem", ["missing", "review", "bytes"])
async def test_source_review_and_bytes_are_verified_before_remote_fetches(
    source_review_context: tuple[Session, ArtifactStore],
    problem: str,
) -> None:
    session, store = source_review_context
    if problem != "missing":
        _review(source_review_context)
        if problem == "review":
            review = session.scalars(select(ComfyRegistrySourceArtifactReview)).one()
            review.review_sha256 = "0" * 64
            session.commit()
        else:
            wheel = verified_reviewed_source_wheel(session, store, declaration=DECLARATION)
            artifact = session.get(Artifact, wheel.artifact_id)
            assert artifact is not None
            store.verified_path(artifact).write_bytes(b"changed")
    sources = _Sources({}, {})
    with pytest.raises(ComfyRegistryWheelClosureDriverError):
        await _drive([DECLARATION, "remote==1"], source_review_context, sources)
    assert not sources.project_calls and not sources.metadata_calls


async def test_review_removed_during_resolution_refuses_the_completed_closure(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    session, _store = source_review_context
    _review(source_review_context)
    project, metadata = _project("remote", [("1.0", [])])
    sources = _Sources({"remote": project}, metadata)

    async def revoke(phase: ClosurePhase, number: int, names: tuple[str, ...]) -> None:
        if phase == "fetching_projects":
            with Session(session.get_bind()) as fresh:
                fresh.execute(delete(ComfyRegistrySourceArtifactReview))
                fresh.commit()

    with pytest.raises(ComfyRegistryWheelClosureDriverError) as error:
        await _drive([DECLARATION, "remote==1.0"], source_review_context, sources, progress=revoke)
    assert error.value.code == "source_artifact_review_missing"
    assert sources.project_calls == [("remote",)] and len(sources.metadata_calls) == 1


async def test_resolution_snapshots_inputs_before_progress_callbacks(
    source_review_context: tuple[Session, ArtifactStore],
) -> None:
    from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies

    _review(source_review_context, ["runtime>=1"])
    declarations = [DECLARATION, "remote==1.0"]
    expected = plan_comfy_registry_mixed_dependencies(declarations).declaration_sha256
    environment = _environment()
    tags = ["py3-none-any"]
    runtime = {"runtime": "1.0"}
    project, metadata = _project("remote", [("1.0", [])])
    sources = _Sources({"remote": project}, metadata)

    async def mutate(phase: ClosurePhase, number: int, names: tuple[str, ...]) -> None:
        declarations.clear()
        environment.clear()
        tags.clear()
        runtime.clear()

    result = await _drive(
        declarations,
        source_review_context,
        sources,
        environment=environment,
        tags=tags,
        runtime=runtime,
        progress=mutate,
    )
    assert result.complete and result.manifest.declaration_sha256 == expected
    assert result.runtime_distributions[0].name == "runtime"
    assert result.runtime_distributions[0].version == "1.0"


@pytest.mark.parametrize("problem", ["target", "wheel", "metadata", "aggregate"])
async def test_invalid_target_or_local_metadata_budget_refuses_before_network(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    from local_lm import comfy_registry_reviewed_closure as resolver

    _review(source_review_context)
    environment = _environment()
    tags = ("py3-none-any",)
    if problem == "target":
        environment.pop("python_full_version")
    elif problem == "wheel":
        tags = ("py2-none-any",)
    else:
        constant = (
            "MAX_WHEEL_CORE_METADATA_BYTES"
            if problem == "metadata"
            else "MAX_REGISTRY_WHEEL_METADATA_TOTAL_BYTES"
        )
        monkeypatch.setattr(resolver, constant, 1)
    sources = _Sources({}, {})
    with pytest.raises(ComfyRegistryWheelClosureDriverError):
        await _drive(
            [DECLARATION, "remote==1"],
            source_review_context,
            sources,
            environment=environment,
            tags=tags,
        )
    assert not sources.project_calls and not sources.metadata_calls


async def test_repeated_cancellation_drains_the_review_reader_and_closes_its_session(
    source_review_context: tuple[Session, ArtifactStore],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import comfy_registry_reviewed_closure as resolver

    _review(source_review_context)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = verified_reviewed_source_wheel

    def read(session: Session, store: ArtifactStore, *, declaration: str) -> VerifiedSourceWheel:
        started.set()
        assert release.wait(10)
        try:
            return original(session, store, declaration=declaration)
        finally:
            finished.set()

    monkeypatch.setattr(resolver, "verified_reviewed_source_wheel", read)
    sources = _Sources({}, {})
    sessions: list[Session] = []
    task = asyncio.create_task(
        _drive([DECLARATION], source_review_context, sources, sessions=sessions)
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set() and len(sessions) == 1 and not sessions[0].in_transaction()
    assert not sources.project_calls and not sources.metadata_calls


@pytest.mark.parametrize("cancel", [False, True])
async def test_project_failure_preserves_retry_or_cancellation(
    source_review_context: tuple[Session, ArtifactStore],
    cancel: bool,
) -> None:
    _review(source_review_context)

    class Sources(_Sources):
        async def fetch_projects(self, names: Sequence[str]) -> Mapping[str, object]:
            if cancel:
                raise asyncio.CancelledError
            raise ComfyRegistryWheelClosureDriverError(
                "project_rate_limited", "Try again later", retry_after_seconds=17
            )

    sources = Sources({}, {})
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await _drive([DECLARATION, "remote==1"], source_review_context, sources)
    else:
        with pytest.raises(ComfyRegistryWheelClosureDriverError) as error:
            await _drive([DECLARATION, "remote==1"], source_review_context, sources)
        assert error.value.code == "project_rate_limited" and error.value.retry_after_seconds == 17


async def test_declared_roots_stage_and_install_with_the_same_complete_identity(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
) -> None:
    from test_comfy_registry_closure_driver import _wheel_file
    from test_comfy_registry_mixed_wheel_environments import (
        _assemble,
        _destination,
        _input_context,
        _mixed_files,
    )
    from test_comfy_registry_wheel_environments import _marker_environment, _wheel_files

    from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader

    prepared, paths = _mixed_files(source_review_context, tmp_path)
    remote = prepared.manifest.remote.artifacts[0]
    content = paths[remote.filename].read_bytes()
    metadata = _wheel_files(content)["alpha-1.0.dist-info/METADATA"]
    artifact = _wheel_file("alpha", "1.0", metadata)
    artifact.update(hashes={"sha256": hashlib.sha256(content).hexdigest()}, size=len(content))
    project = {"meta": {"api-version": "1.4"}, "name": "alpha", "files": [artifact]}
    sources = _Sources({"alpha": project}, {remote.filename: metadata})
    closure = await _drive(
        [DECLARATION, "alpha==1.0"],
        source_review_context,
        sources,
        environment=_marker_environment(),
        runtime={"helper": "2.0"},
    )
    assert closure.complete and closure.manifest.declaration_sha256 != "f" * 64
    requests: list[str] = []

    def fetch(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200, content=metadata if request.url.path.endswith(".metadata") else content
        )

    context = _input_context(source_review_context)
    downloader = ComfyRegistryWheelDownloader(transport=httpx.MockTransport(fetch))
    stage = tmp_path / "stage"
    try:
        report = await downloader.download_and_stage_inputs(
            closure.manifest,
            stage,
            session_factory=context.session_factory,
            store=context.store,
            marker_environment=context.marker_environment,
            supported_tags=context.supported_tags,
        )
    finally:
        await downloader.close()
    assert report.artifact_manifest_sha256 == closure.manifest.manifest_sha256
    installed = await _assemble(
        closure,
        {name: stage / name for name in paths},
        _destination(tmp_path, closure),
        source_review_context,
    )
    assert installed.closure_sha256 == closure.closure_sha256 and installed.artifact_count == 2
    assert len(requests) == 2 and all(remote.filename in url for url in requests)
