"""Runtime setup spends only the downloads and configuration that were previewed."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_runtime_provisioning import _write_manifest, _zip_bytes

from local_lm.config import Settings
from local_lm.runtime_config import runtime_config_path
from local_lm.runtime_provisioning import RuntimeProvisioner, RuntimeProvisioningError


@pytest.fixture
async def runtime(
    settings: Settings, tmp_path: Path
) -> AsyncIterator[tuple[RuntimeProvisioner, list[httpx.Request], bytes]]:
    content = _zip_bytes({"llama-server.exe": b"neutral executable"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        try:
            yield provisioner, requests, content
        finally:
            await provisioner.close()


async def test_preflight_counts_the_selected_archive_and_security_overlays_without_installing(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
) -> None:
    provisioner, requests, content = runtime
    asset = provisioner._definition("llama.cpp")["runtime_assets"]["test-platform"]
    asset["security_overlays"] = [{"url": "https://runtime.test/security.zip", "size_bytes": 73}]

    plan = provisioner.preflight("llama.cpp")

    assert plan.operation == "install_managed"
    assert plan.release == "b-test"
    assert plan.license == "MIT"
    assert plan.download_bytes == len(content) + 73
    assert plan.required_free_bytes == 1
    assert plan == provisioner.preflight("llama.cpp")
    assert requests == []
    assert provisioner.settings.llama_executable is None
    assert not runtime_config_path(provisioner.settings.data_dir).exists()


async def test_the_unchanged_approved_runtime_installs_and_then_can_be_reused(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
) -> None:
    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")

    installed = await provisioner.provision("llama.cpp", expected_plan=plan)

    assert installed.state == "ready" and installed.managed
    assert len(requests) == 1
    executable = provisioner.settings.llama_executable
    assert executable is not None and executable.read_bytes() == b"neutral executable"
    reuse = provisioner.preflight("llama.cpp")
    assert reuse.operation == "reuse_managed" and reuse.download_bytes == 0
    reused = await provisioner.provision("llama.cpp", expected_plan=reuse)
    assert reused.state == "ready" and reused.managed and reused.release == installed.release
    assert len(requests) == 1


@pytest.mark.parametrize(
    "change", ["archive", "overlay", "platform", "release", "destination", "hosts"]
)
async def test_a_changed_runtime_plan_refuses_before_any_download(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes], change: str, tmp_path: Path
) -> None:
    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    definition = provisioner._definition("llama.cpp")
    asset = definition["runtime_assets"]["test-platform"]
    if change == "archive":
        asset["sha256"] = "f" * 64
    elif change == "overlay":
        asset["security_overlays"] = [{"url": "https://runtime.test/extra.zip", "size_bytes": 17}]
    elif change == "platform":
        definition["runtime_assets"]["other-platform"] = dict(asset)
        provisioner._platform_keys["llama.cpp"] = "other-platform"
    elif change == "release":
        definition["pinned_release"] = "another-release"
    elif change == "destination":
        provisioner.settings.data_dir = tmp_path / "another-data-directory"
    else:
        provisioner.allowed_download_hosts.add("another.test")

    with pytest.raises(RuntimeProvisioningError, match="approved runtime setup changed"):
        await provisioner.provision("llama.cpp", expected_plan=plan)

    assert requests == []
    assert provisioner.settings.llama_executable is None
    assert not runtime_config_path(provisioner.settings.data_dir).exists()


async def test_the_plan_is_checked_after_waiting_for_another_runtime_setup(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
) -> None:
    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    lock = provisioner._locks["llama.cpp"]
    await lock.acquire()
    task = asyncio.create_task(provisioner.provision("llama.cpp", expected_plan=plan))
    try:
        await asyncio.sleep(0)
        provisioner._definition("llama.cpp")["pinned_release"] = "changed-while-waiting"
    finally:
        lock.release()
    with pytest.raises(RuntimeProvisioningError, match="approved runtime setup changed"):
        await task
    assert requests == []


async def test_a_configured_executable_is_reused_only_while_its_bytes_match(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes], tmp_path: Path
) -> None:
    provisioner, requests, _content = runtime
    executable = tmp_path / "configured-runtime.exe"
    executable.write_bytes(b"neutral configured runtime")
    provisioner.settings.llama_executable = executable
    plan = provisioner.preflight("llama.cpp")
    assert plan.operation == "reuse_configured" and plan.release is None
    assert plan.download_bytes == 0
    assert (await provisioner.provision("llama.cpp", expected_plan=plan)).state == "ready"
    executable.write_bytes(b"different configured runtime")
    with pytest.raises(RuntimeProvisioningError, match="approved runtime setup changed"):
        await provisioner.provision("llama.cpp", expected_plan=plan)
    assert requests == []
    assert not runtime_config_path(provisioner.settings.data_dir).exists()


async def test_manifest_drift_during_download_cannot_install_or_save_configuration(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    download = provisioner._download

    async def changed(*args: Any, **kwargs: Any) -> Path:
        result = await download(*args, **kwargs)
        provisioner._definition("llama.cpp")["license"] = "Changed terms"
        return result

    monkeypatch.setattr(provisioner, "_download", changed)
    with pytest.raises(RuntimeProvisioningError, match="approved runtime setup changed"):
        await provisioner.provision("llama.cpp", expected_plan=plan)
    assert len(requests) == 1
    assert provisioner.settings.llama_executable is None
    assert not (provisioner.runtime_root / "llama.cpp" / "b-test").exists()
    assert not runtime_config_path(provisioner.settings.data_dir).exists()


async def test_blocked_and_unsupported_runtime_plans_cannot_be_approved(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
) -> None:
    provisioner, requests, _content = runtime
    with pytest.raises(RuntimeProvisioningError, match="pending dependency review"):
        provisioner.preflight("vllm")
    provisioner._platform_keys["llama.cpp"] = "unsupported-platform"
    with pytest.raises(RuntimeProvisioningError, match="not available for this machine"):
        provisioner.preflight("llama.cpp")
    assert requests == []


async def test_approved_runtime_file_inspection_does_not_run_on_the_event_loop(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provisioner, requests, _content = runtime
    executable = tmp_path / "configured-runtime.exe"
    executable.write_bytes(b"neutral runtime")
    provisioner.settings.llama_executable = executable
    plan = provisioner.preflight("llama.cpp")
    calls: list[int] = []
    original = provisioner._sha256_file

    def inspected(path: Path, **kwargs: Any) -> str:
        calls.append(threading.get_ident())
        return original(path, **kwargs)

    monkeypatch.setattr(provisioner, "_sha256_file", inspected)
    event_loop = threading.get_ident()
    assert (await provisioner.provision("llama.cpp", expected_plan=plan)).state == "ready"
    assert calls and event_loop not in calls
    assert requests == []


async def test_runtime_drift_before_the_installation_worker_runs_cannot_install(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    original = asyncio.to_thread

    async def delayed(function: Any, *args: Any, **kwargs: Any) -> Any:
        if function.__name__ == "install":
            provisioner._definition("llama.cpp")["license"] = "Changed terms"
        return await original(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", delayed)
    with pytest.raises(RuntimeProvisioningError, match="approved runtime setup changed"):
        await provisioner.provision("llama.cpp", expected_plan=plan)
    assert len(requests) == 1
    assert provisioner.settings.llama_executable is None
    assert not (provisioner.runtime_root / "llama.cpp" / "b-test").exists()
    assert not runtime_config_path(provisioner.settings.data_dir).exists()


async def test_changed_inputs_after_installation_cannot_save_runtime_configuration(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    original = provisioner._install_archive

    def installed(*args: Any, **kwargs: Any) -> dict[str, Path]:
        result = original(*args, **kwargs)
        provisioner._definition("llama.cpp")["license"] = "Changed terms"
        return result

    monkeypatch.setattr(provisioner, "_install_archive", installed)
    with pytest.raises(RuntimeProvisioningError, match="approved runtime setup changed"):
        await provisioner.provision("llama.cpp", expected_plan=plan)
    assert len(requests) == 1
    assert provisioner.settings.llama_executable is None
    assert not runtime_config_path(provisioner.settings.data_dir).exists()
