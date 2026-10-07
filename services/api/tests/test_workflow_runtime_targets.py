"""Extension planning uses the selected runtime, including before its first install."""

from __future__ import annotations

import hashlib
import importlib
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.config import Settings
from local_lm.runtime_config import runtime_config_path
from local_lm.runtime_provisioning import RuntimeProvisioner


def _module() -> ModuleType:
    return importlib.import_module("local_lm.workflow_runtime_targets")


@pytest.fixture
async def runtime(settings: Settings, tmp_path: Path) -> AsyncIterator[RuntimeProvisioner]:
    root = Path(__file__).resolve().parents[3]
    manifest = tmp_path / "engines.json"
    manifest.write_bytes((root / "packaging/engines.json").read_bytes())
    review = tmp_path / "runtime-reviews/comfyui-v0.28.0.json"
    review.parent.mkdir()
    review.write_bytes((root / "packaging/runtime-reviews/comfyui-v0.28.0.json").read_bytes())

    def refuse(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("Runtime planning must not download or install anything")

    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="windows-x86_64-nvidia-cu13",
        )
        try:
            yield provisioner
        finally:
            await provisioner.close()


def _host(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    environment = {
        "implementation_name": "cpython",
        "implementation_version": "3.12.10",
        "os_name": "nt",
        "platform_machine": "AMD64",
        "platform_release": "11",
        "platform_system": "Windows",
        "platform_version": "10.0.26000",
        "python_full_version": "3.12.10",
        "platform_python_implementation": "CPython",
        "python_version": "3.12",
        "sys_platform": "win32",
    }
    monkeypatch.setattr(_module(), "default_environment", lambda: dict(environment))
    return environment


async def test_fresh_runtime_target_has_exact_python_tags_and_reviewed_packages_without_installing(
    runtime: RuntimeProvisioner, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    host = _host(monkeypatch)
    plan = runtime.preflight("comfyui")
    assert plan.operation == "install_managed"

    target = await module.preflight_workflow_runtime_target(runtime, plan)

    environment, tags, distributions = await target.probe(Path("unused-python"))
    assert environment["python_full_version"] == "3.13.14"
    assert environment["implementation_version"] == "3.13.14"
    assert environment["python_version"] == "3.13"
    assert environment["platform_version"] == host["platform_version"]
    assert tags[:3] == ("cp313-cp313-win_amd64", "cp313-abi3-win_amd64", "cp313-none-win_amd64")
    assert tags[3:14] == tuple(f"cp3{minor}-abi3-win_amd64" for minor in range(12, 1, -1))
    assert len(tags) == 45 and tags[-1] == "py30-none-any"
    assert "cp312-cp312-win_amd64" not in tags
    assert len(distributions) == 89
    assert ComfyRegistryRuntimeDistribution("torch", "2.13.0+cu130") in distributions
    assert ComfyRegistryRuntimeDistribution("packaging", "26.2") in distributions
    environment["python_version"] = "changed"
    assert (await target.probe(Path("unused-python")))[0]["python_version"] == "3.13"
    assert runtime.settings.comfy_executable is None
    assert not runtime_config_path(runtime.settings.data_dir).exists()
    assert list(runtime.runtime_root.iterdir()) == []


@pytest.mark.parametrize("change", ["overlay", "rewrite", "destination", "version", "host"])
async def test_an_unmeasured_interpreter_cannot_inherit_a_target(
    runtime: RuntimeProvisioner, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    module = _module()
    host = _host(monkeypatch)
    asset = runtime._definition("comfyui")["runtime_assets"]["windows-x86_64-nvidia-cu13"]
    if change == "overlay":
        asset["security_overlays"][0]["sha256"] = "a" * 64
    elif change == "rewrite":
        asset["security_overlays"][0]["rewrite_files"]["python313._pth"] = "another path\n"
    elif change == "destination":
        asset["executable"] = "another/python.exe"
    elif change == "version":
        asset["runtime_probe"]["python"] = "3.14.0"
    else:
        host["platform_machine"] = "ARM64"
    with pytest.raises(module.WorkflowRuntimeTargetError) as refused:
        await module.preflight_workflow_runtime_target(runtime, runtime.preflight("comfyui"))
    assert refused.value.code == "workflow-runtime-target-unsupported"


@pytest.mark.parametrize("change", ["review", "packaging"])
async def test_distribution_metadata_must_match_the_exact_component_review(
    runtime: RuntimeProvisioner, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    module = _module()
    _host(monkeypatch)
    asset = runtime._definition("comfyui")["runtime_assets"]["windows-x86_64-nvidia-cu13"]
    path = runtime.manifest_path.parent / asset["dependency_review"]["file"]
    review = json.loads(path.read_text())
    reviewed = review["assets"]["windows-x86_64-nvidia-cu13"]
    package = next(item for item in reviewed["distributions"] if item["name"] == "packaging")
    package.update(version="26.3", dist_info="packaging-26.3.dist-info")
    identities = sorted(item["dist_info"] for item in reviewed["distributions"])
    reviewed["inventory_sha256"] = hashlib.sha256(
        ("\n".join(identities) + "\n").encode()
    ).hexdigest()
    content = json.dumps(review).encode()
    path.write_bytes(content)
    if change == "packaging":
        asset["dependency_review"]["sha256"] = hashlib.sha256(content).hexdigest()
        asset["dependency_inventory_sha256"] = reviewed["inventory_sha256"]
    with pytest.raises(module.WorkflowRuntimeTargetError) as refused:
        await module.preflight_workflow_runtime_target(runtime, runtime.preflight("comfyui"))
    assert refused.value.code == (
        "workflow-runtime-target-unsupported"
        if change == "packaging"
        else "workflow-runtime-target-unavailable"
    )


async def test_planning_rechecks_approval_after_inspection_and_runs_file_reads_off_the_loop(
    runtime: RuntimeProvisioner, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _host(monkeypatch)
    original = module._planned_target
    loop_thread = threading.get_ident()

    def inspect(*args: Any) -> Any:
        assert threading.get_ident() != loop_thread
        result = original(*args)
        runtime._definition("comfyui")["license"] = "Changed terms"
        return result

    monkeypatch.setattr(module, "_planned_target", inspect)
    with pytest.raises(module.WorkflowRuntimeTargetError) as refused:
        await module.preflight_workflow_runtime_target(runtime, runtime.preflight("comfyui"))
    assert refused.value.code == "workflow-runtime-target-changed"


async def test_an_old_runtime_approval_refuses_before_target_inspection(
    runtime: RuntimeProvisioner, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    plan = runtime.preflight("comfyui")
    runtime._definition("comfyui")["license"] = "Changed terms"

    def forbidden(*_args: Any) -> Any:
        pytest.fail("A changed plan must refuse before inspecting the target")

    monkeypatch.setattr(module, "_planned_target", forbidden)
    with pytest.raises(module.WorkflowRuntimeTargetError) as refused:
        await module.preflight_workflow_runtime_target(runtime, plan)
    assert refused.value.code == "workflow-runtime-target-changed"


@pytest.mark.parametrize("change", [False, True])
async def test_configured_runtimes_use_their_own_probe_and_refuse_changed_files(
    runtime: RuntimeProvisioner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: bool
) -> None:
    module = _module()
    host = _host(monkeypatch)
    directory = tmp_path / "configured"
    directory.mkdir()
    executable = directory / "python.exe"
    executable.write_bytes(b"neutral executable")
    (directory / "main.py").write_bytes(b"neutral source")
    runtime.settings.comfy_executable = executable
    runtime.settings.comfy_directory = directory
    host.update(python_version="3.11", python_full_version="3.11.9", extra="")

    async def probe(path: Path) -> tuple[dict[str, str], tuple[str, ...], tuple[Any, ...]]:
        assert path == executable
        if change:
            executable.write_bytes(b"changed executable")
        return host, ("cp311-cp311-win_amd64",), ()

    monkeypatch.setattr(module, "probe_comfy_registry_runtime_target", probe)
    plan = runtime.preflight("comfyui")
    assert plan.operation == "reuse_configured"
    if change:
        with pytest.raises(module.WorkflowRuntimeTargetError) as refused:
            await module.preflight_workflow_runtime_target(runtime, plan)
        assert refused.value.code == "workflow-runtime-target-changed"
    else:
        target = await module.preflight_workflow_runtime_target(runtime, plan)
        assert dict(target.environment)["python_full_version"] == "3.11.9"
        assert target.supported_tags == ("cp311-cp311-win_amd64",)
        assert target.distributions == ()


async def test_fresh_target_drives_real_extension_planning_and_binds_later_inspection(
    runtime: RuntimeProvisioner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_workflow_package_execution_plan import _inputs, _plan

    from local_lm.workflow_package_execution_plan import WorkflowPackageExecutionPlanError

    module = _module()
    _host(monkeypatch)
    target = await module.preflight_workflow_runtime_target(runtime, runtime.preflight("comfyui"))
    inputs = _inputs(dependencies=True)
    inputs["interpreter_probe"] = target.probe
    inputs["python_executable"] = runtime.runtime_root / "not-installed/python.exe"
    try:
        plan = await _plan(inputs)
        assert plan.wheel_bytes == 200
        environment = dict(target.environment)
        plan.verify_target(environment, target.supported_tags)
        environment["python_full_version"] = "3.13.15"
        with pytest.raises(WorkflowPackageExecutionPlanError) as refused:
            plan.verify_target(environment, target.supported_tags)
        assert refused.value.code == "workflow-package-runtime-changed"
    finally:
        await inputs["archive_downloader"].close()
