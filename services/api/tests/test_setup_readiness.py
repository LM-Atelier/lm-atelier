from __future__ import annotations

from collections.abc import Sequence

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.adapters.contracts import ADAPTER_CONTRACT_VERSION
from local_lm.comfy_templates import COMFY_TEMPLATE_COMPILER_VERSION
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.hardware import hardware_capability_class
from local_lm.model_planner import (
    ACTIVATION_PROBE_VERSION,
    LAUNCH_CONTRACT_VERSION,
    media_workflow_contract_version,
)
from local_lm.models import (
    Job,
    ModelCapabilityEvidence,
    ModelInstall,
    ModelProfile,
    SetupVerification,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.schemas import RuntimeStatus, WorkerStatus
from local_lm.setup_readiness import _workflow_check
from local_lm.setup_verification import verification_evidence_key
from local_lm.workflow_package_drafts import workflow_package_draft_dependencies

pytestmark = pytest.mark.asyncio


def _runtime(engine: str, state: str = "ready") -> RuntimeStatus:
    return RuntimeStatus(
        engine=engine,  # type: ignore[arg-type]
        release=f"{engine}-test",
        state=state,  # type: ignore[arg-type]
        supported=state != "unsupported",
        distribution="test",
        license="test",
    )


def _workers(*, chat_state: str = "ready") -> list[WorkerStatus]:
    return [
        WorkerStatus(
            name="chat",
            state=chat_state,  # type: ignore[arg-type]
            managed=True,
            running=chat_state == "ready",
            profile_id="profile_chat",
        ),
        WorkerStatus(
            name="media",
            state="ready",
            managed=True,
            running=True,
        ),
    ]


def _set_runtime_and_worker_state(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workers: Sequence[WorkerStatus],
    runtime_overrides: dict[str, RuntimeStatus] | None = None,
) -> None:
    statuses = {
        "llama.cpp": _runtime("llama.cpp"),
        "vllm": _runtime("vllm"),
        "comfyui": _runtime("comfyui"),
    }
    statuses.update(runtime_overrides or {})
    services = app.state.services
    monkeypatch.setattr(services.runtimes, "statuses", lambda: list(statuses.values()))
    monkeypatch.setattr(services.runtimes, "status", lambda engine: statuses[engine])
    monkeypatch.setattr(services.processes, "statuses", lambda: list(workers))


def _add_install(
    *,
    role: str,
    engine: str,
    template_sha256: str | None = None,
) -> ModelInstall:
    manifest: dict[str, object] = {
        "files": [f"{role}.bin"],
        "expected_sha256": {f"{role}.bin": role[0] * 64},
    }
    if template_sha256:
        manifest["workflow_template_sha256"] = template_sha256
        manifest["workflow_template_id"] = f"template_{role}"
    install = ModelInstall(
        id=f"model_{role}",
        name=f"Synthetic {role}",
        role=role,
        engine=engine,
        local_path=f"C:/synthetic/{role}",
        compatibility="likely",
        manifest_json=manifest,
        active=True,
    )
    return install


def _add_evidence(
    settings: Settings,
    install: ModelInstall,
    *,
    result: str = "ready",
    current: bool = True,
) -> ModelCapabilityEvidence:
    template_sha256 = install.manifest_json.get("workflow_template_sha256")
    workflow_contract = (
        media_workflow_contract_version(template_sha256)
        if isinstance(template_sha256, str)
        else None
    )
    return ModelCapabilityEvidence(
        model_install_id=install.id,
        evidence_key=(install.role[0] * 63) + ("1" if current else "0"),
        result=result,
        component_hashes_json={
            str(path): str(digest)
            for path, digest in install.manifest_json["expected_sha256"].items()
        },
        runtime_build="synthetic-test",
        adapter_contract_version=ADAPTER_CONTRACT_VERSION if current else 0,
        launch_contract_version=LAUNCH_CONTRACT_VERSION,
        workflow_contract_version=workflow_contract,
        hardware_class=hardware_capability_class(settings),
        probe_version=ACTIVATION_PROBE_VERSION,
        details_json={"runtime_release": f"{install.engine}-test"},
    )


def _add_profile(install: ModelInstall) -> ModelProfile:
    return ModelProfile(
        id=f"profile_{install.role}",
        model_install_id=install.id,
        name=f"Synthetic {install.role}",
        role=install.role,
        engine=install.engine,
    )


def _add_workflow(
    install: ModelInstall, operation: str
) -> tuple[WorkflowDefinition, WorkflowRevision]:
    definition = WorkflowDefinition(
        id=f"workflow_{install.role}",
        name=f"Synthetic {install.role}",
        operation=operation,
    )
    # Mirrors what the compiler records, so the legacy evidence path is
    # exercised the way it behaves in a real install rather than trivially.
    revision = WorkflowRevision(
        id=f"revision_{install.role}",
        workflow_id=definition.id,
        version=1,
        engine=install.engine,
        api_graph_json={"node": {"class_type": "Synthetic"}},
        dependencies_json={
            "model_install_ids": [install.id],
            "compiler_version": COMFY_TEMPLATE_COMPILER_VERSION,
            "template_id": f"template_{install.role}",
            "template_sha256": install.manifest_json.get("workflow_template_sha256"),
        },
        trusted=True,
    )
    definition.current_revision_id = revision.id
    return definition, revision


def _add_activation(
    revision: WorkflowRevision,
    *,
    active: bool = True,
    contract_sha256: str | None = None,
    launch_sha256: str = "c" * 64,
) -> WorkflowActivation:
    return WorkflowActivation(
        id=f"activation_{revision.id}",
        workflow_revision_id=revision.id,
        resolver_version="test",
        dependency_contract_sha256=(
            contract_sha256 or revision.dependency_contract_sha256 or "a" * 64
        ),
        binding_sha256="b" * 64,
        state="ready" if active else "stale",
        is_active=active,
        details_json={"launch_sha256": launch_sha256},
    )


async def test_internal_package_draft_is_not_a_setup_candidate(app: FastAPI) -> None:
    del app
    install = _add_install(role="image", engine="comfyui")
    definition, revision = _add_workflow(install, "text_to_image")
    revision.api_graph_json = {}
    revision.dependencies_json = workflow_package_draft_dependencies("a" * 64)
    revision.trusted = False

    with SessionLocal() as session:
        session.add_all([install, definition])
        session.flush()
        session.add(revision)
        session.commit()
        selected, check = _workflow_check(session, "image", install)

    assert selected is None
    assert check.code == "workflow_missing"
    assert check.action == "repair_workflow"


async def test_contract_workflow_requires_a_ready_activation(app: FastAPI) -> None:
    del app
    install = _add_install(role="image", engine="comfyui")
    definition, revision = _add_workflow(install, "text_to_image")
    revision.dependency_contract_sha256 = "a" * 64

    with SessionLocal() as session:
        session.add_all([install, definition])
        session.flush()
        session.add(revision)
        session.commit()
        selected, check = _workflow_check(session, "image", install)

    assert selected is None
    assert check.code == "workflow_activation_not_ready"
    assert check.status == "fail"
    assert check.action == "repair_workflow"


async def test_contract_workflow_with_a_ready_activation_is_ready(app: FastAPI) -> None:
    del app
    install = _add_install(role="image", engine="comfyui")
    definition, revision = _add_workflow(install, "text_to_image")
    revision.dependency_contract_sha256 = "a" * 64

    with SessionLocal() as session:
        session.add_all([install, definition])
        session.flush()
        session.add(revision)
        session.flush()
        session.add(_add_activation(revision))
        session.commit()
        selected, check = _workflow_check(session, "image", install)

    assert selected is not None
    assert selected.id == revision.id
    assert check.code == "workflow_ready"
    assert check.status == "pass"


@pytest.mark.parametrize(
    ("active", "contract_sha256", "launch_sha256"),
    [
        (False, None, "c" * 64),
        (True, "d" * 64, "c" * 64),
        (True, None, "not-a-launch-digest"),
    ],
)
async def test_contract_workflow_refuses_invalid_activation_authority(
    app: FastAPI,
    *,
    active: bool,
    contract_sha256: str | None,
    launch_sha256: str,
) -> None:
    del app
    install = _add_install(role="image", engine="comfyui")
    definition, revision = _add_workflow(install, "text_to_image")
    revision.dependency_contract_sha256 = "a" * 64

    with SessionLocal() as session:
        session.add_all([install, definition])
        session.flush()
        session.add(revision)
        session.flush()
        session.add(
            _add_activation(
                revision,
                active=active,
                contract_sha256=contract_sha256,
                launch_sha256=launch_sha256,
            )
        )
        session.commit()
        selected, check = _workflow_check(session, "image", install)

    assert selected is None
    assert check.code == "workflow_activation_not_ready"
    assert check.action == "repair_workflow"


def _add_verification(
    install: ModelInstall,
    profile: ModelProfile,
    capability_evidence: ModelCapabilityEvidence,
    workflow: WorkflowRevision | None = None,
) -> SetupVerification:
    return SetupVerification(
        role=install.role,
        evidence_key=verification_evidence_key(
            install.role,  # type: ignore[arg-type]
            install,
            profile,
            workflow,
            capability_evidence,
        ),
        state="ready",
        model_install_id=install.id,
        profile_id=profile.id,
        workflow_revision_id=workflow.id if workflow else None,
    )


async def test_fresh_setup_reports_one_stable_model_action_per_role(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/setup/readiness")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == 2
    assert payload["state"] == "action_required"
    assert [role["role"] for role in payload["roles"]] == ["chat", "image", "video"]
    for role in payload["roles"]:
        assert role["state"] == "action_required"
        assert role["next_action"] == "select_model"
        assert [check["code"] for check in role["checks"]] == ["model_missing"]
        assert role["verification_level"] == "generation_probe"


async def test_unsupported_runtime_is_reported_before_any_model_download(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine that cannot run the engine must be told so, not sent shopping."""
    unsupported = _runtime("comfyui", state="unsupported")
    unsupported.message = "This machine has no supported accelerator."
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    _set_runtime_and_worker_state(
        app,
        monkeypatch,
        workers=_workers(),
        runtime_overrides={"comfyui": unsupported},
    )

    payload = (await client.get("/api/setup/readiness")).json()
    by_role = {role["role"]: role for role in payload["roles"]}

    for role in ("image", "video"):
        assert [check["code"] for check in by_role[role]["checks"]] == ["runtime_unsupported"]
        assert "no supported accelerator" in by_role[role]["checks"][0]["message"]
        # Terminal: offering an action here is what produced the endless
        # "choose a model" loop on machines that can never run the engine.
        assert by_role[role]["next_action"] is None
    assert by_role["chat"]["checks"][0]["code"] == "model_missing"


async def test_missing_runtime_is_reported_before_any_model_download(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    _set_runtime_and_worker_state(
        app,
        monkeypatch,
        workers=_workers(),
        runtime_overrides={"comfyui": _runtime("comfyui", state="missing")},
    )

    payload = (await client.get("/api/setup/readiness")).json()
    by_role = {role["role"]: role for role in payload["roles"]}

    assert [check["code"] for check in by_role["image"]["checks"]] == ["runtime_missing"]
    assert by_role["image"]["next_action"] == "install_runtime"
    assert by_role["image"]["engine"] == "comfyui"


async def test_ready_runtime_without_a_model_still_asks_for_a_model(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    _set_runtime_and_worker_state(app, monkeypatch, workers=_workers())

    payload = (await client.get("/api/setup/readiness")).json()
    by_role = {role["role"]: role for role in payload["roles"]}

    assert [check["code"] for check in by_role["image"]["checks"]] == ["model_missing"]
    assert by_role["image"]["next_action"] == "select_model"


async def test_partial_setup_reports_role_specific_install_progress(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        session.add(
            Job(
                id="job_image_download",
                kind="download",
                status="running",
                payload_json={"role": "image", "remote_id": "synthetic/image"},
            )
        )
        session.commit()

    payload = (await client.get("/api/setup/readiness")).json()
    by_role = {role["role"]: role for role in payload["roles"]}

    assert by_role["image"]["state"] == "in_progress"
    assert by_role["image"]["next_action"] == "wait_for_install"
    assert by_role["image"]["checks"][0]["code"] == "install_in_progress"
    assert by_role["chat"]["checks"][0]["code"] == "model_missing"
    assert by_role["video"]["checks"][0]["code"] == "model_missing"


async def test_failed_and_stale_activation_are_distinct_and_bounded(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_runtime_and_worker_state(app, monkeypatch, workers=_workers())
    chat = _add_install(role="chat", engine="llama.cpp")
    image = _add_install(role="image", engine="comfyui", template_sha256="a" * 64)
    with SessionLocal() as session:
        session.add_all([chat, image])
        session.flush()
        session.add_all(
            [
                _add_profile(chat),
                _add_profile(image),
                _add_evidence(settings, chat, result="failed"),
                _add_evidence(settings, image, current=False),
            ]
        )
        session.commit()

    payload = (await client.get("/api/setup/readiness")).json()
    by_role = {role["role"]: role for role in payload["roles"]}

    assert by_role["chat"]["checks"][-1] == {
        "code": "activation_failed",
        "status": "fail",
        "message": "The model did not pass its activation probe.",
        "action": "activate_model",
    }
    assert by_role["image"]["checks"][-1]["code"] == "activation_stale"
    assert "failure_reason" not in response_text(payload)


async def test_runtime_and_workflow_failures_have_distinct_repair_actions(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _add_install(role="chat", engine="llama.cpp")
    image = _add_install(role="image", engine="comfyui", template_sha256="d" * 64)
    definition, revision = _add_workflow(image, "text_to_image")
    revision.trusted = False
    with SessionLocal() as session:
        session.add_all([chat, image])
        session.flush()
        session.add_all(
            [
                _add_profile(image),
                _add_evidence(settings, image),
                definition,
            ]
        )
        session.flush()
        session.add(revision)
        session.commit()

    _set_runtime_and_worker_state(
        app,
        monkeypatch,
        workers=_workers(),
        runtime_overrides={"llama.cpp": _runtime("llama.cpp", "failed")},
    )
    payload = (await client.get("/api/setup/readiness")).json()
    by_role = {role["role"]: role for role in payload["roles"]}

    assert by_role["chat"]["checks"][-1]["code"] == "runtime_failed"
    assert by_role["chat"]["next_action"] == "retry_runtime"
    assert by_role["image"]["checks"][-1]["code"] == "workflow_untrusted"
    assert by_role["image"]["next_action"] == "review_workflow"


async def test_ready_roles_surface_worker_failure_without_requiring_residency(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = _add_install(role="chat", engine="llama.cpp")
    image = _add_install(role="image", engine="comfyui", template_sha256="b" * 64)
    video = _add_install(role="video", engine="comfyui", template_sha256="c" * 64)
    image_workflow = _add_workflow(image, "text_to_image")
    video_workflow = _add_workflow(video, "image_to_video")
    chat_profile = _add_profile(chat)
    image_profile = _add_profile(image)
    video_profile = _add_profile(video)
    chat_evidence = _add_evidence(settings, chat)
    image_evidence = _add_evidence(settings, image)
    video_evidence = _add_evidence(settings, video)
    with SessionLocal() as session:
        session.add_all([chat, image, video])
        session.flush()
        session.add_all(
            [
                chat_profile,
                image_profile,
                video_profile,
                chat_evidence,
                image_evidence,
                video_evidence,
            ]
        )
        session.add_all([image_workflow[0], video_workflow[0]])
        session.flush()
        session.add_all([image_workflow[1], video_workflow[1]])
        session.flush()
        session.add_all(
            [
                _add_verification(chat, chat_profile, chat_evidence),
                _add_verification(image, image_profile, image_evidence, image_workflow[1]),
                _add_verification(video, video_profile, video_evidence, video_workflow[1]),
            ]
        )
        session.commit()

    _set_runtime_and_worker_state(app, monkeypatch, workers=_workers(chat_state="exited"))
    failed = (await client.get("/api/setup/readiness")).json()
    failed_by_role = {role["role"]: role for role in failed["roles"]}
    assert failed_by_role["chat"]["state"] == "action_required"
    assert failed_by_role["chat"]["checks"][-1]["code"] == "worker_failed"
    assert failed_by_role["image"]["state"] == "ready"
    assert failed_by_role["video"]["state"] == "ready"

    stopped_workers = _workers(chat_state="stopped")
    _set_runtime_and_worker_state(app, monkeypatch, workers=stopped_workers)
    ready = (await client.get("/api/setup/readiness")).json()
    assert ready["state"] == "ready"
    assert {role["state"] for role in ready["roles"]} == {"ready"}
    chat_ready = next(role for role in ready["roles"] if role["role"] == "chat")
    not_loaded = next(
        check for check in chat_ready["checks"] if check["code"] == "worker_not_loaded"
    )
    # Ready, because it will work - but the wait is named rather than hidden,
    # and it is offered as something the user may do now instead of later.
    assert not_loaded["status"] == "pass"
    assert not_loaded["action"] == "prepare_worker"
    assert chat_ready["next_action"] is None
    assert chat_ready["checks"][-1]["code"] == "generation_verified"

    mismatched_workers = _workers()
    mismatched_workers[0] = mismatched_workers[0].model_copy(update={"profile_id": "profile_other"})
    _set_runtime_and_worker_state(app, monkeypatch, workers=mismatched_workers)
    mismatched = (await client.get("/api/setup/readiness")).json()
    mismatched_chat = next(role for role in mismatched["roles"] if role["role"] == "chat")
    assert mismatched_chat["state"] == "ready"
    assert any(check["code"] == "worker_not_loaded" for check in mismatched_chat["checks"])

    loaded = _workers()
    _set_runtime_and_worker_state(app, monkeypatch, workers=loaded)
    resident = (await client.get("/api/setup/readiness")).json()
    resident_chat = next(role for role in resident["roles"] if role["role"] == "chat")
    # Nothing is left to prepare once the model is resident, so nothing is offered.
    assert any(check["code"] == "worker_ready" for check in resident_chat["checks"])
    assert all(check["action"] != "prepare_worker" for check in resident_chat["checks"])


def response_text(payload: object) -> str:
    return repr(payload)


async def test_polled_endpoints_do_not_block_the_event_loop() -> None:
    """Both read the filesystem, and readiness is polled every few seconds.

    Declaring them synchronously is what makes the framework run them in a worker
    thread; declaring them `async` would put that work on the event loop and stall
    every other request for its duration.
    """
    import inspect

    from local_lm import api as api_module

    assert not inspect.iscoroutinefunction(api_module.get_setup_readiness)
    assert not inspect.iscoroutinefunction(api_module.runtime_status)
