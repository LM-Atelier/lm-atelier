"""Provision the pinned runtime only when a live installation is explicitly requested."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from local_lm.config import Settings
from local_lm.runtime_config import runtime_config_path
from local_lm.runtime_provisioning import RuntimeProvisioner, RuntimeProvisioningError


@pytest.mark.asyncio
async def test_pinned_comfy_runtime_installs_from_its_approved_archive(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    if os.name != "nt" or os.environ.get("LM_ATELIER_TEST_PROVISION_COMFY") != "1":
        pytest.skip("A live Windows ComfyUI installation must be explicitly requested")
    # Keep native DLL paths within Windows limits despite the long archive layout.
    tmp_path = tmp_path_factory.mktemp("runtime")
    settings = Settings(
        data_dir=tmp_path / "data",
        chat_engine="mock",
        media_engine="mock",
        comfy_executable=None,
        comfy_directory=None,
    )
    settings.prepare()
    provisioner = RuntimeProvisioner(
        settings, environment={}, platform_key="windows-x86_64-nvidia-cu13"
    )
    transport_logger = logging.getLogger("httpx")
    previous_level = transport_logger.level
    transport_logger.setLevel(logging.WARNING)
    try:
        approved = provisioner.preflight("comfyui")
        assert approved.operation == "install_managed"
        definition = provisioner._definition("comfyui")
        asset = definition["runtime_assets"]["windows-x86_64-nvidia-cu13"]
        assert approved.download_bytes == asset["size_bytes"] + sum(
            item["size_bytes"] for item in asset["security_overlays"]
        )
        cached_inputs = {
            "LM_ATELIER_TEST_RUNTIME_ARCHIVE_FILE": asset,
            "LM_ATELIER_TEST_RUNTIME_OVERLAY_FILE": asset["security_overlays"][0],
        }
        for variable, selected in cached_inputs.items():
            cached = os.environ.get(variable)
            if not cached:
                continue
            source = Path(cached)
            assert source.is_file() and not source.is_symlink()
            assert source.stat().st_size == selected["size_bytes"]
            with source.open("rb") as archive:
                assert hashlib.file_digest(archive, "sha256").hexdigest() == selected["sha256"]
            destination = provisioner._archive_path("comfyui", definition, selected)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

        print("Runtime setup approved: " + json.dumps(asdict(approved)), flush=True)
        pending = asyncio.create_task(provisioner.provision("comfyui", expected_plan=approved))
        while not pending.done():
            done, _ = await asyncio.wait({pending}, timeout=30)
            status = provisioner.status("comfyui")
            print(
                "Runtime setup progress: "
                + json.dumps(
                    {"state": status.state, "progress": status.progress, "message": status.message}
                ),
                flush=True,
            )
            if done:
                break
        installed = await pending
        assert installed.state == "ready" and installed.managed
        verified = await asyncio.to_thread(provisioner.verify_status, "comfyui")
        assert verified.state == "ready" and verified.release == approved.release
        executable, directory = settings.comfy_executable, settings.comfy_directory
        assert executable is not None and directory is not None
        assert executable.is_file() and (directory / "main.py").is_file()
        assert runtime_config_path(settings.data_dir).is_file()
        result = {
            "approved": asdict(approved),
            "verified": verified.model_dump(mode="json"),
            "data_dir": str(settings.data_dir),
            "executable": str(executable),
            "directory": str(directory),
        }
        (tmp_path / "provisioned-runtime.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print("Runtime setup verified: " + json.dumps(result), flush=True)
    except RuntimeProvisioningError as exc:
        if isinstance(exc.__cause__, subprocess.CalledProcessError):
            cause = exc.__cause__
            print(
                "Runtime setup probe failure: "
                + json.dumps({"exit": cause.returncode, "stderr": (cause.stderr or "")[-8000:]}),
                flush=True,
            )
        raise
    finally:
        transport_logger.setLevel(previous_level)
        await provisioner.close()
