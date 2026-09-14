"""A configured runtime is judged by who installed it, not by where its folder is."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_runtime_provisioning import _write_manifest, _zip_bytes

from local_lm.config import Settings
from local_lm.runtime_provisioning import RuntimeProvisioner

pytestmark = pytest.mark.asyncio


def _portable_runtime(root: Path) -> tuple[Path, Path]:
    executable = root / "python" / "python.exe"
    directory = root / "ComfyUI"
    executable.parent.mkdir(parents=True)
    directory.mkdir(parents=True)
    executable.write_bytes(b"python")
    (directory / "main.py").write_bytes(b"# entry point\n")
    return executable, directory


def _marker(root: Path, release: str) -> None:
    (root / ".lm-atelier-runtime.json").write_text(
        json.dumps({"schema_version": 3, "engine": "comfyui", "release": release}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("outside", "ready"),
        ("inside", "ready"),
        ("earlier-managed-release", "runtime_other_version"),
        ("marker-for-another-folder", "runtime_missing"),
        ("staging", "runtime_missing"),
        ("pinned-release-without-marker", "runtime_missing"),
        ("damaged-pinned-release", "runtime_missing"),
        ("engine-folder", "runtime_missing"),
    ],
)
async def test_a_configured_runtime_is_judged_by_its_installer_not_its_folder(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    expected: str,
) -> None:
    content = _zip_bytes({"python/python.exe": b"python", "ComfyUI/main.py": b"# pinned\n"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content, comfy_content=content)
    settings.prepare()
    releases = settings.data_dir / "runtimes" / "comfyui"
    root = {
        "outside": tmp_path / "portable-comfyui",
        "inside": releases / "v-portable",
        "earlier-managed-release": releases / "v-earlier",
        "marker-for-another-folder": releases / "v-renamed",
        "staging": releases / ".v-test.partial-0123abcd",
        # Deleting the marker from the pinned install must not turn it into an
        # unverified runtime that is trusted as configured.
        "pinned-release-without-marker": releases / "v-test",
        "damaged-pinned-release": releases / "v-test",
        "engine-folder": releases,
    }[location]
    executable, directory = _portable_runtime(root)
    if location == "engine-folder":
        directory = root
        (directory / "main.py").write_bytes(b"# entry point\n")
    if location == "earlier-managed-release":
        # What another build leaves behind: a verified install of a different pin.
        _marker(root, "v-earlier")
    if location == "damaged-pinned-release":
        # This build's own install failing verification is not another version's.
        _marker(root, "v-test")
    if location == "marker-for-another-folder":
        # A marker that does not name its own folder is not trusted for a name.
        _marker(root, "v-earlier")
    monkeypatch.setattr(settings, "comfy_executable", executable)
    monkeypatch.setattr(settings, "comfy_directory", directory)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(404))
    ) as transport:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=transport,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        restore = provisioner.start_restore()
        if restore is not None:
            await restore
        status = provisioner.status("comfyui")

        services = app.state.services
        monkeypatch.setattr(services.runtimes, "statuses", provisioner.statuses)
        monkeypatch.setattr(services.runtimes, "status", provisioner.status)
        monkeypatch.setattr(settings, "media_engine", "comfyui")
        payload = (await client.get("/api/setup/readiness")).json()
        runtimes = (await client.get("/api/runtimes")).json()
        await provisioner.close()

    by_role = {role["role"]: role for role in payload["roles"]}
    listed = next(item for item in runtimes if item["engine"] == "comfyui")
    if expected == "ready":
        assert (status.state, status.managed) == ("ready", False), status.message
        for role in ("image", "video"):
            codes = [check["code"] for check in by_role[role]["checks"]]
            assert not {"runtime_missing", "runtime_other_version"} & set(codes)
            assert by_role[role]["next_action"] != "install_runtime"
    else:
        # A folder this application manages is still held to the pinned release,
        # which is how a pinned update reaches anyone at all.
        assert status.state == "missing"
        for role in ("image", "video"):
            assert [check["code"] for check in by_role[role]["checks"]] == [expected]
            assert by_role[role]["next_action"] == "install_runtime"
    if expected == "runtime_other_version":
        # Still asked to install, but told the truth about what is running.
        assert status.installed_release == listed["installed_release"] == "v-earlier"
        assert listed["message"] == (
            "ComfyUI v-earlier, installed by another version of LM Atelier, is still in use."
            " This version uses v-test. Install it to switch."
        )
        assert "not installed" not in by_role["image"]["checks"][0]["message"]
    else:
        assert status.installed_release is None
        if location == "damaged-pinned-release":
            assert (
                listed["message"] == "Managed runtime verification failed; reinstall it to repair."
            )
        assert listed["installed_release"] is None
    # Reporting never reconfigures or provisions anything.
    assert settings.comfy_executable == executable
    assert settings.comfy_directory == directory
