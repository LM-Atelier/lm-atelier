"""The built-in matting workflow: what it runs, where it comes from, where it is never used.

A background-removal model on its own does nothing: something has to run it.
This workflow is the something, authored by the application and installed for
each background-removal model, and it is the one workflow that may say it
returns a cut-out subject, because its author is the one saying so.

The cases hold the graph to the result it promises, hold the install to one
workflow per model that survives a restart, and hold selection to using the
workflow only when it is asked for by name. That last part matters as much as
the rest: a workflow that declares no model is exactly what an ordinary edit
falls back to, and an ordinary edit answered with a cutout is a broken edit.
Neutral node definitions stand in for a runtime's object info.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from local_lm.comfy_templates import compile_authored_workflow
from local_lm.config import Settings
from local_lm.db import Base, SessionLocal, configure_database, init_db
from local_lm.domain import JobKind, JobStatus, Operation
from local_lm.downloads import DownloadManager
from local_lm.events import EventBroker
from local_lm.matting_workflows import (
    built_in_matting_graph,
    declare_matting,
    runtime_lists_model_file,
    workflow_declares_matting,
)
from local_lm.model_planner import INSTALL_RESOLVER_VERSION, revision_declares_a_model
from local_lm.models import (
    Chat,
    ChatWorkflowSelection,
    InstallPlan,
    Job,
    ModelAssetInstall,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.scheduler import ResourceScheduler
from local_lm.schemas import DownloadRequest
from local_lm.settings_registry import IMAGE_SETTINGS, workflow_settings
from local_lm.studio_capabilities import tool_capabilities

MODEL_FILE = "background-model.safetensors"
EMPTY: dict[str, Any] = {"type": "object", "properties": {}}


def _object_info(*model_files: str) -> dict[str, Any]:
    """The runtime's node definitions, seeing exactly the given model files."""

    return {
        "LoadImage": {
            "input": {"required": {"image": [["source.png"], {"image_upload": True}]}},
            "output": ["IMAGE", "MASK"],
        },
        "LoadBackgroundRemovalModel": {
            "input": {"required": {"bg_removal_name": ["COMBO", {"options": list(model_files)}]}},
            "output": ["BACKGROUND_REMOVAL"],
        },
        "RemoveBackground": {
            "input": {
                "required": {
                    "bg_removal_model": ["BACKGROUND_REMOVAL", {}],
                    "image": ["IMAGE", {}],
                }
            },
            "output": ["MASK"],
        },
        "InvertMask": {"input": {"required": {"mask": ["MASK", {}]}}, "output": ["MASK"]},
        "JoinImageWithAlpha": {
            "input": {"required": {"image": ["IMAGE", {}], "alpha": ["MASK", {}]}},
            "output": ["IMAGE"],
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE", {}],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "output": [],
            "output_node": True,
        },
    }


def _asset(session: Session, *, sha256: str = "a" * 64) -> ModelAssetInstall:
    asset = ModelAssetInstall(
        id="asset_background_model",
        name="background-model",
        kind="background_removal",
        local_path="background-model",
        manifest_json={"comfy_name": MODEL_FILE, "sha256": sha256},
        active=True,
    )
    session.add(asset)
    session.flush()
    return asset


def test_the_graph_cuts_the_subject_out_with_the_mask_the_right_way_round() -> None:
    api_graph, schema = compile_authored_workflow(
        built_in_matting_graph(MODEL_FILE),
        _object_info(MODEL_FILE),
        operation="image_to_image",
    )
    by_type = {node["class_type"]: (node_id, node) for node_id, node in api_graph.items()}

    load_id, load = by_type["LoadImage"]
    assert load["inputs"]["image"] == "${input_image}"
    assert schema["properties"]["input_image"] == {"type": "string"}
    model_id, model = by_type["LoadBackgroundRemovalModel"]
    assert model["inputs"]["bg_removal_name"] == MODEL_FILE
    remove_id, remove = by_type["RemoveBackground"]
    assert remove["inputs"]["image"] == [load_id, 0]
    assert remove["inputs"]["bg_removal_model"] == [model_id, 0]
    invert_id, invert = by_type["InvertMask"]
    assert invert["inputs"]["mask"] == [remove_id, 0]
    join_id, join = by_type["JoinImageWithAlpha"]
    # JoinImageWithAlpha writes alpha as 1 - mask and the removal mask is 1 over
    # the subject, so only the inverted mask keeps the subject and not the rest.
    assert join["inputs"]["alpha"] == [invert_id, 0]
    assert join["inputs"]["image"] == [load_id, 0]
    _, save = by_type["SaveImage"]
    assert save["inputs"]["images"] == [join_id, 0]


def test_the_promise_comes_from_its_author_and_offers_nothing_to_set() -> None:
    _, compiled = compile_authored_workflow(
        built_in_matting_graph(MODEL_FILE),
        _object_info(MODEL_FILE),
        operation="image_to_image",
    )

    # Compiling the graph declares nothing, however plainly it removes a
    # background: the statement is made by declare_matting, on purpose.
    assert workflow_declares_matting(compiled) is False
    declared = declare_matting(compiled)
    assert workflow_declares_matting(declared) is True
    assert "matte" not in {field.key for field in workflow_settings(IMAGE_SETTINGS, declared)}


def test_a_model_file_the_runtime_cannot_see_is_refused_when_it_compiles() -> None:
    with pytest.raises(ValueError):
        compile_authored_workflow(
            built_in_matting_graph(MODEL_FILE),
            _object_info("some-other-model.safetensors"),
            operation="image_to_image",
        )


def test_the_built_in_needs_the_runtime_to_list_its_exact_model_file() -> None:
    """Stricter than the compiler, which lets an empty list of choices through."""

    assert runtime_lists_model_file(_object_info(MODEL_FILE), MODEL_FILE) is True
    assert runtime_lists_model_file(_object_info(), MODEL_FILE) is False
    assert runtime_lists_model_file(_object_info("other.safetensors"), MODEL_FILE) is False
    older_shape = _object_info()
    older_shape["LoadBackgroundRemovalModel"]["input"]["required"]["bg_removal_name"] = [
        [MODEL_FILE],
        {},
    ]
    assert runtime_lists_model_file(older_shape, MODEL_FILE) is True
    assert runtime_lists_model_file({}, MODEL_FILE) is False


def test_a_background_removal_model_gets_one_workflow_that_declares_no_model(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    with SessionLocal() as session:
        asset = _asset(session)

        revision = DownloadManager._ensure_matting_workflow(
            session, asset, _object_info(MODEL_FILE)
        )

        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        assert definition.operation == "image_to_image"
        assert definition.current_revision_id == revision.id
        assert revision.trusted is True
        assert workflow_declares_matting(revision.input_schema_json) is True
        # Declaring the model would bind the workflow to a model profile, and a
        # background-removal model has none, so it could never run.
        assert revision_declares_a_model(revision.dependencies_json) is False
        assert revision.dependencies_json["model_asset_ids"] == [asset.id]

        again = DownloadManager._ensure_matting_workflow(session, asset, _object_info(MODEL_FILE))
        assert again.id == revision.id

        asset.manifest_json = {**asset.manifest_json, "sha256": "b" * 64}
        replaced = DownloadManager._ensure_matting_workflow(
            session, asset, _object_info(MODEL_FILE)
        )
        assert replaced.id != revision.id
        assert replaced.version == revision.version + 1
        assert definition.current_revision_id == replaced.id


async def test_a_model_installed_before_the_built_in_gets_it_at_the_next_start(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()

    class MediaAdapter:
        async def object_info(self) -> dict[str, Any]:
            return _object_info(MODEL_FILE)

    manager = DownloadManager(settings, EventBroker(), media_adapter=cast(Any, MediaAdapter()))
    with SessionLocal() as session:
        _asset(session)
        session.commit()

    assert await manager.refresh_installed_media_workflows() == 1
    assert await manager.refresh_installed_media_workflows() == 0
    with SessionLocal() as session:
        names = session.scalars(
            select(WorkflowDefinition.name).where(WorkflowDefinition.operation == "image_to_image")
        ).all()
    assert names == [f"Remove background · {MODEL_FILE}"]


def test_a_workflow_that_cannot_compile_leaves_the_model_installed_and_nothing_behind(
    settings: Settings,
) -> None:
    settings.prepare()
    configure_database(settings)
    init_db()
    manager = DownloadManager(settings, EventBroker())
    with SessionLocal() as session:
        asset_id = _asset(session).id
        session.commit()
        before = set(session.scalars(select(WorkflowDefinition.id)).all())

    # The runtime cannot see the file, so the workflow cannot compile.
    manager._install_matting_workflow(asset_id, _object_info())

    with SessionLocal() as session:
        asset = session.get(ModelAssetInstall, asset_id)
        assert asset is not None and asset.active is True
        assert set(session.scalars(select(WorkflowDefinition.id)).all()) == before

    manager._install_matting_workflow(asset_id, _object_info(MODEL_FILE))

    with SessionLocal() as session:
        added = set(session.scalars(select(WorkflowDefinition.id)).all()) - before
        assert [session.get(WorkflowDefinition, item).name for item in added] == [
            f"Remove background · {MODEL_FILE}"
        ]


def test_isolate_names_its_workflow_and_a_cutout_alone_is_no_edit_workflow() -> None:
    only = {
        item.kind: item
        for item in tool_capabilities(
            edit_input_schemas=[declare_matting(EMPTY)],
            matting_workflow_ids=["revision_cutout"],
        )
    }
    assert only["isolate"].available is True
    assert only["isolate"].workflow_revision_id == "revision_cutout"
    assert only["instruct"].available is False

    both = {
        item.kind: item
        for item in tool_capabilities(
            edit_input_schemas=[declare_matting(EMPTY), EMPTY],
            matting_workflow_ids=["revision_one", "revision_two"],
        )
    }
    assert both["instruct"].available is True
    # With two, one is still named: any of them cuts a subject out, and the
    # studio's own choice never would.
    assert both["isolate"].workflow_revision_id == "revision_one"


@pytest.fixture
def memory_session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def _orchestrator() -> ConversationOrchestrator:
    return ConversationOrchestrator(
        engines=SimpleNamespace(
            settings=SimpleNamespace(media_engine="comfyui", chat_engine="mock"),
            chat=SimpleNamespace(cancel=AsyncMock()),
            media=SimpleNamespace(cancel=AsyncMock()),
        ),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(publish_job=AsyncMock()),
        processes=SimpleNamespace(statuses=Mock(return_value=[])),
        session_factory=Mock(),
    )


def _edit_workflow(
    session: Session,
    name: str,
    schema: dict[str, Any],
    *,
    created_at: datetime,
    use_case: str = "",
    is_default: bool = False,
    definition_id: str | None = None,
) -> WorkflowRevision:
    family = WorkflowFamily(name=name, use_case=use_case)
    definition = WorkflowDefinition(
        family=family,
        variant_key="edit",
        name=name,
        operation=Operation.IMAGE_TO_IMAGE.value,
        created_at=created_at,
    )
    if definition_id is not None:
        definition.id = definition_id
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="comfyui",
        ui_graph_json={"nodes": []},
        api_graph_json={"save": {"class_type": "SaveImage", "inputs": {}}},
        input_schema_json=schema,
        dependencies_json={},
        trusted=True,
    )
    preference = WorkflowPreference(
        family=family, selector_capability="image", is_default=is_default
    )
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return revision


def test_an_ordinary_edit_never_falls_back_to_the_cutout(memory_session: Session) -> None:
    orchestrator = _orchestrator()
    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    ordinary = _edit_workflow(memory_session, "Picture edits", EMPTY, created_at=earlier)
    # Newer, and declaring no model, so it would be the first generic choice.
    cutout = _edit_workflow(
        memory_session,
        "Remove background",
        declare_matting(EMPTY),
        created_at=earlier + timedelta(days=1),
    )

    chosen = orchestrator._workflow_for_operation(memory_session, Operation.IMAGE_TO_IMAGE)
    assert chosen is not None and chosen.id == ordinary.id
    named = orchestrator._workflow_for_operation(
        memory_session, Operation.IMAGE_TO_IMAGE, preferred_revision_id=cutout.id
    )
    assert named is not None and named.id == cutout.id
    assert orchestrator.installed_matting_workflow_ids(memory_session) == [cutout.id]


def test_with_only_the_cutout_installed_an_ordinary_edit_finds_nothing(
    memory_session: Session,
) -> None:
    _edit_workflow(
        memory_session,
        "Remove background",
        declare_matting(EMPTY),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert _orchestrator()._workflow_for_operation(memory_session, Operation.IMAGE_TO_IMAGE) is None


def test_auto_never_hands_an_edit_to_the_cutout(memory_session: Session) -> None:
    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    # Described, and the default, so every other rule would choose it first.
    _edit_workflow(
        memory_session,
        "Remove background",
        declare_matting(EMPTY),
        created_at=earlier,
        use_case="remove the background",
        is_default=True,
    )
    ordinary = _edit_workflow(
        memory_session,
        "Picture edits",
        EMPTY,
        created_at=earlier + timedelta(days=1),
        use_case="studio photos",
    )
    chat = Chat(title="Edits")
    memory_session.add(chat)
    memory_session.flush()
    memory_session.add(
        ChatWorkflowSelection(chat_id=chat.id, selector_capability="image", mode="automatic")
    )
    memory_session.flush()

    _profile, selection, revision = _orchestrator()._profile_and_workflow_for_operation(
        memory_session, chat, Operation.IMAGE_TO_IMAGE, "Remove the background."
    )

    assert revision is not None and revision.id == ordinary.id
    assert selection["workflow_revision_id"] == ordinary.id


def test_the_installed_cutouts_come_in_the_order_a_person_reads_them(
    memory_session: Session,
) -> None:
    earlier = datetime(2026, 1, 1, tzinfo=UTC)
    # Ids that sort the other way from the names, so name order is what is held.
    last = _edit_workflow(
        memory_session,
        "Remove background · zeta.safetensors",
        declare_matting(EMPTY),
        created_at=earlier,
        definition_id="workflow_a",
    )
    first = _edit_workflow(
        memory_session,
        "Remove background · alpha.safetensors",
        declare_matting(EMPTY),
        created_at=earlier,
        definition_id="workflow_b",
    )

    assert _orchestrator().installed_matting_workflow_ids(memory_session) == [first.id, last.id]


def _safetensors(tensor_name: str) -> bytes:
    header = json.dumps(
        {
            tensor_name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
            "__metadata__": {},
        },
        separators=(",", ":"),
    ).encode()
    return len(header).to_bytes(8, "little") + header


async def test_installing_a_background_removal_model_installs_its_workflow(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Through the download that installs the model, since that is where it has to happen."""

    settings.prepare()
    configure_database(settings)
    init_db()
    content = _safetensors("background.encoder.weight")
    digest = hashlib.sha256(content).hexdigest()
    with SessionLocal() as session:
        # A catalog plan whose files are fetched by the stand-in below, so
        # nothing here asks a model host about a repository that is not real.
        plan = InstallPlan(
            id="plan_background_model",
            provider="civitai",
            remote_id="101",
            revision="202",
            role="image",
            engine="comfyui",
            plan_hash="8" * 64,
            resolver_version=INSTALL_RESOLVER_VERSION,
            compatibility="supported",
            artifacts_json=[
                {
                    "path": MODEL_FILE,
                    "kind": "background_removal",
                    "target_folder": "background_removal",
                    "size_bytes": len(content),
                    "sha256": digest,
                    "required": True,
                    "reuse": "download",
                    "source_version_id": "202",
                    "source_file_id": "301",
                }
            ],
            runtime_contract_json={
                "auxiliary_kind": None,
                "workflow_asset_kind": "background_removal",
                "comfy_paths": {"background_removal": "."},
                "workflow_component_folders": {MODEL_FILE: "background_removal"},
            },
            activation_probe_json={"kind": "workflow_asset", "required": False},
            status="planned",
        )
        session.add(plan)
        request = DownloadRequest(
            install_plan_id=plan.id,
            remote_id=plan.remote_id,
            revision=plan.revision,
            role="image",
            engine="comfyui",
            allow_patterns=[MODEL_FILE],
            expected_sha256={MODEL_FILE: digest},
            comfy_paths={"background_removal": "."},
            workflow_asset_kind="background_removal",
        )
        session.add(
            Job(
                id="job_background_model",
                kind=JobKind.DOWNLOAD.value,
                status=JobStatus.QUEUED.value,
                payload_json=request.model_dump(mode="json"),
            )
        )
        session.commit()

    class Processes:
        def statuses(self) -> list[object]:
            return [SimpleNamespace(name="media", running=False, profile_id=None)]

        async def start_media(self, model_root: tuple[Path, dict[str, str]]) -> None:
            return None

        async def stop(self, name: str) -> None:
            return None

    class MediaAdapter:
        def invalidate_object_info_cache(self) -> None:
            return None

        async def object_info(self) -> dict[str, Any]:
            return _object_info(MODEL_FILE)

    manager = DownloadManager(
        settings,
        EventBroker(),
        scheduler=ResourceScheduler(),
        media_adapter=cast(Any, MediaAdapter()),
        processes=cast(Any, Processes()),
    )
    # A lookup against a model host fails this test here rather than leaving it.
    manager._api = cast(Any, SimpleNamespace())

    async def download_file(**kwargs: Any) -> str:
        target = kwargs["staging"] / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)

    monkeypatch.setattr(manager, "_download_file", download_file)
    await manager._download("job_background_model")

    with SessionLocal() as session:
        job = session.get(Job, "job_background_model")
        assert job is not None and job.status == JobStatus.COMPLETE.value, job and job.error
        asset = session.scalars(select(ModelAssetInstall)).one()
        assert asset.kind == "background_removal" and asset.active is True
        definition = session.scalars(
            select(WorkflowDefinition).where(
                WorkflowDefinition.name == f"Remove background · {MODEL_FILE}"
            )
        ).one()
        revision = session.get(WorkflowRevision, definition.current_revision_id)
        assert revision is not None
        assert workflow_declares_matting(revision.input_schema_json) is True
        assert revision.dependencies_json["model_asset_ids"] == [asset.id]
