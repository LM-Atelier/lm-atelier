"""Retry approved runtime setup without losing a verified published installation."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_runtime_provisioning import _write_manifest, _zip_bytes
from test_runtime_provisioning_plans import runtime as runtime

from local_lm.config import Settings
from local_lm.runtime_config import runtime_config_path
from local_lm.runtime_provisioning import RuntimeProvisioner, RuntimeProvisioningError


async def test_the_original_approval_can_resume_a_completed_runtime_installation(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
) -> None:
    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    first = await provisioner.provision("llama.cpp", expected_plan=plan)
    second = await provisioner.provision("llama.cpp", expected_plan=plan)
    assert first.state == second.state == "ready"
    assert first.release == second.release and second.managed
    assert len(requests) == 1


@pytest.mark.parametrize("moment", ["before-configuration", "during-persistence"])
async def test_a_published_runtime_survives_configuration_interruption_and_restart(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    moment: str,
) -> None:
    from local_lm import runtime_provisioning

    provisioner, requests, content = runtime
    plan = provisioner.preflight("llama.cpp")
    with monkeypatch.context() as patch:

        def interrupted(*args: Any, **kwargs: Any) -> None:
            raise OSError("Neutral persistence interruption")

        if moment == "before-configuration":
            patch.setattr(provisioner, "_apply_configuration", interrupted)
        else:
            patch.setattr(runtime_provisioning, "persist_runtime_values", interrupted)
        with pytest.raises((OSError, RuntimeProvisioningError)):
            await provisioner.provision("llama.cpp", expected_plan=plan)

    assert len(requests) == 1
    assert not runtime_config_path(provisioner.settings.data_dir).exists()
    settings = provisioner.settings.model_copy(deep=True)
    settings.llama_executable = None

    def serve(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        restarted = RuntimeProvisioner(
            settings,
            manifest_path=tmp_path / "engines.json",
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        try:

            def reinstall(*args: Any, **kwargs: Any) -> Any:
                pytest.fail("A verified published runtime must not be installed again.")

            monkeypatch.setattr(restarted, "_install_archive", reinstall)
            result = await restarted.provision("llama.cpp", expected_plan=plan)
            assert result.state == "ready" and result.managed
            assert settings.llama_executable is not None
            assert settings.llama_executable.read_bytes() == b"neutral executable"
            assert runtime_config_path(settings.data_dir).exists()
            assert len(requests) == 1
        finally:
            await restarted.close()


@pytest.mark.parametrize("change", ["license", "executable", "configuration", "approval", "marker"])
async def test_recovery_refuses_changed_inputs_or_damaged_installed_files(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
    tmp_path: Path,
    change: str,
) -> None:
    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    await provisioner.provision("llama.cpp", expected_plan=plan)
    executable = provisioner.settings.llama_executable
    assert executable is not None
    marker = executable.parent / ".lm-atelier-runtime.json"
    if change == "license":
        provisioner._definition("llama.cpp")["license"] = "Changed terms"
    elif change == "executable":
        executable.write_bytes(b"damaged neutral executable")
    elif change == "configuration":
        other = tmp_path / "another.exe"
        other.write_bytes(b"another neutral runtime")
        provisioner.settings.llama_executable = other
    elif change == "approval":
        value = json.loads(marker.read_text())
        value["approved_setup"]["plan"]["plan_sha256"] = "f" * 64
        marker.write_text(json.dumps(value))
    else:
        marker.write_text("{}")
    before = runtime_config_path(provisioner.settings.data_dir).read_bytes()
    with pytest.raises(RuntimeProvisioningError):
        await provisioner.provision("llama.cpp", expected_plan=plan)
    assert len(requests) == 1
    assert runtime_config_path(provisioner.settings.data_dir).read_bytes() == before


async def test_an_installation_without_this_approval_does_not_gain_it_on_retry(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
) -> None:
    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    await provisioner.provision("llama.cpp")
    with pytest.raises(RuntimeProvisioningError):
        await provisioner.provision("llama.cpp", expected_plan=plan)
    assert len(requests) == 1


async def test_recovery_checks_files_off_the_event_loop_and_rechecks_before_configuration(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import runtime_provisioning_recovery

    provisioner, requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    await provisioner.provision("llama.cpp", expected_plan=plan)
    original = runtime_provisioning_recovery.recover_approved_runtime
    loop_thread = threading.get_ident()
    calls: list[int] = []

    def changed(*args: Any, **kwargs: Any) -> Any:
        calls.append(threading.get_ident())
        result = original(*args, **kwargs)
        assert result is not None
        provisioner._definition("llama.cpp")["license"] = "Changed after inspection"
        return result

    monkeypatch.setattr(runtime_provisioning_recovery, "recover_approved_runtime", changed)
    before = runtime_config_path(provisioner.settings.data_dir).read_bytes()
    with pytest.raises(RuntimeProvisioningError):
        await provisioner.provision("llama.cpp", expected_plan=plan)
    assert calls and loop_thread not in calls
    assert runtime_config_path(provisioner.settings.data_dir).read_bytes() == before
    assert len(requests) == 1


async def test_recovery_does_not_replace_a_runtime_that_appeared_at_the_original_path(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provisioner, requests, _content = runtime
    original_path = tmp_path / "configured.exe"
    provisioner.settings.llama_executable = original_path
    plan = provisioner.preflight("llama.cpp")
    with monkeypatch.context() as patch:

        def interrupted(*args: Any, **kwargs: Any) -> None:
            raise OSError("Neutral configuration interruption")

        patch.setattr(provisioner, "_apply_configuration", interrupted)
        with pytest.raises((OSError, RuntimeProvisioningError)):
            await provisioner.provision("llama.cpp", expected_plan=plan)
    original_path.write_bytes(b"a newly configured neutral runtime")
    with pytest.raises(RuntimeProvisioningError):
        await provisioner.provision("llama.cpp", expected_plan=plan)
    assert provisioner.settings.llama_executable == original_path
    assert not runtime_config_path(provisioner.settings.data_dir).exists()
    assert len(requests) == 1


async def test_changed_terms_during_extraction_cannot_publish_an_approved_runtime(
    runtime: tuple[RuntimeProvisioner, list[httpx.Request], bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioner, _requests, _content = runtime
    plan = provisioner.preflight("llama.cpp")
    original = provisioner._verify_runtime_contract

    def changed(*args: Any, **kwargs: Any) -> None:
        original(*args, **kwargs)
        provisioner._definition("llama.cpp")["license"] = "Changed during extraction"

    monkeypatch.setattr(provisioner, "_verify_runtime_contract", changed)
    with pytest.raises(RuntimeProvisioningError):
        await provisioner.provision("llama.cpp", expected_plan=plan)
    assert not (provisioner.runtime_root / "llama.cpp/b-test").exists()
    assert not runtime_config_path(provisioner.settings.data_dir).exists()


@pytest.mark.parametrize("change", [None, "executable", "directory"])
async def test_comfy_recovery_binds_both_runtime_paths(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str | None,
) -> None:
    from local_lm import runtime_provisioning

    content = _zip_bytes(
        {
            "python/python.exe": b"neutral python executable",
            "python/Lib/site-packages/example-1.0.dist-info/METADATA": (
                b"Name: example\nVersion: 1.0\n"
            ),
            "ComfyUI/main.py": b"neutral source",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=b"unused", comfy_content=content)
    probe = {"python": "3.13.14", "comfyui": "0.28.0", "packages": {"example": "1.0"}}

    def inspect(*args: Any, **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=runtime_provisioning._RUNTIME_PROBE_SENTINEL + json.dumps(probe) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", inspect)
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
            plan = provisioner.preflight("comfyui")
            await provisioner.provision("comfyui", expected_plan=plan)
            before = runtime_config_path(settings.data_dir).read_bytes()
            if change == "executable":
                other = tmp_path / "another-python.exe"
                other.write_bytes(b"another neutral executable")
                settings.comfy_executable = other
            elif change == "directory":
                other = tmp_path / "another-comfy"
                other.mkdir()
                (other / "main.py").write_bytes(b"another neutral source")
                settings.comfy_directory = other
            if change is None:
                assert (await provisioner.provision("comfyui", expected_plan=plan)).state == "ready"
            else:
                with pytest.raises(RuntimeProvisioningError):
                    await provisioner.provision("comfyui", expected_plan=plan)
            assert len(requests) == 1
            assert runtime_config_path(settings.data_dir).read_bytes() == before
        finally:
            await provisioner.close()
