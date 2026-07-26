from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

import local_lm.runtime_provisioning as runtime_provisioning
from local_lm.runtime_config import runtime_config_path
from local_lm.runtime_provisioning import RuntimeProvisioner, RuntimeProvisioningError


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return output.getvalue()


def _inventory_sha256(identities: list[str]) -> str:
    canonical = ("\n".join(sorted(identities)) + "\n").encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_manifest(
    path: Path,
    *,
    llama_content: bytes,
    llama_sha256: str | None = None,
    comfy_content: bytes | None = None,
) -> None:
    llama_hash = llama_sha256 or hashlib.sha256(llama_content).hexdigest()
    comfy_hash = hashlib.sha256(comfy_content).hexdigest() if comfy_content else "0" * 64
    comfy_size = len(comfy_content) if comfy_content else 1
    comfy_identity = "example-1.0.dist-info"
    review = {
        "schema_version": 1,
        "release": "v-test",
        "assets": {
            "test-platform": {
                "source_asset_url": "https://runtime.test/comfy-runtime.zip",
                "source_asset_sha256": comfy_hash,
                "inventory_count": 1,
                "inventory_sha256": _inventory_sha256([comfy_identity]),
                "distributions": [
                    {
                        "dist_info": comfy_identity,
                        "name": "example",
                        "version": "1.0",
                        "license": "MIT",
                        "license_source": "https://runtime.test/example-license",
                    }
                ],
            }
        },
    }
    review_path = path.parent / "runtime-reviews" / "comfyui-v-test.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_bytes = (json.dumps(review, indent=2) + "\n").encode()
    review_path.write_bytes(review_bytes)
    payload = {
        "schema_version": 2,
        "updated_at": "2026-07-25",
        "engines": {
            "llama.cpp": {
                "pinned_release": "b-test",
                "distribution": "external",
                "license": "MIT",
                "runtime_assets": {
                    "test-platform": {
                        "url": "https://runtime.test/llama-runtime.zip",
                        "sha256": llama_hash,
                        "size_bytes": len(llama_content),
                        "required_free_bytes": 1,
                        "archive_type": "zip",
                        "executable": "llama-server.exe",
                    }
                },
            },
            "comfyui": {
                "pinned_release": "v-test",
                "distribution": "external-gpl-3.0",
                "license": "GPL-3.0-only",
                "security_review": {
                    "reviewed_at": "2026-07-25",
                    "release_is_immutable": True,
                    "package_audit": "Test fixture dependency review.",
                    "upstream_advisories": ["https://runtime.test/security-advisories"],
                },
                "runtime_assets": {
                    "test-platform": {
                        "url": "https://runtime.test/comfy-runtime.zip",
                        "sha256": comfy_hash,
                        "size_bytes": comfy_size,
                        "required_free_bytes": 1,
                        "archive_type": "zip",
                        "executable": "python/python.exe",
                        "directory": "ComfyUI",
                        "dependency_inventory_count": 1,
                        "dependency_inventory_sha256": _inventory_sha256([comfy_identity]),
                        "dependency_review": {
                            "file": "runtime-reviews/comfyui-v-test.json",
                            "sha256": hashlib.sha256(review_bytes).hexdigest(),
                            "asset_key": "test-platform",
                        },
                        "security_overlays": [],
                        "runtime_probe": {
                            "python": "3.13.14",
                            "comfyui": "0.28.0",
                            "imports": ["example"],
                            "packages": {"example": "1.0"},
                        },
                    }
                },
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


async def test_runtime_download_resumes_verifies_and_persists(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"verified executable"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()
    partial = settings.download_dir / "runtimes" / "llama.cpp-b-test-llama-runtime.zip.part"
    partial.parent.mkdir(parents=True)
    split = len(content) // 3
    partial.write_bytes(content[:split])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["range"] == f"bytes={split}-"
        return httpx.Response(
            206,
            headers={"content-range": f"bytes {split}-{len(content) - 1}/{len(content)}"},
            content=content[split:],
        )

    environment: dict[str, str] = {}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment=environment,
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = await provisioner.ensure("llama.cpp")
        await provisioner.close()

    assert len(requests) == 1
    assert status.state == "ready"
    assert status.managed is True
    assert settings.llama_executable
    assert settings.llama_executable.read_bytes() == b"verified executable"
    assert environment["LOCAL_LM_CHAT_ENGINE"] == "llama.cpp"
    assert environment["LOCAL_LM_LLAMA_EXECUTABLE"] == str(settings.llama_executable)
    saved = json.loads(runtime_config_path(settings.data_dir).read_text(encoding="utf-8"))
    assert saved["LOCAL_LM_LLAMA_EXECUTABLE"] == str(settings.llama_executable)
    assert not partial.exists()
    assert not list((settings.download_dir / "runtimes").glob("*.zip"))


async def test_runtime_checksum_failure_removes_untrusted_partial(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    expected = _zip_bytes({"llama-server.exe": b"expected"})
    corrupted = b"x" * len(expected)
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=expected)
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=corrupted)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        with pytest.raises(RuntimeProvisioningError, match="SHA-256"):
            await provisioner.ensure("llama.cpp")
        assert provisioner.status("llama.cpp").state == "failed"
        await provisioner.close()

    assert not list((settings.download_dir / "runtimes").glob("*.part"))
    assert settings.llama_executable is None


async def test_runtime_archive_cannot_escape_install_root(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes(
        {
            "../outside.exe": b"unsafe",
            "llama-server.exe": b"otherwise valid",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        with pytest.raises(RuntimeProvisioningError, match="unsafe path"):
            await provisioner.ensure("llama.cpp")
        await provisioner.close()

    assert not (settings.data_dir / "outside.exe").exists()
    assert not (tmp_path / "outside.exe").exists()


def test_7z_listing_rejects_links_and_uncompressed_size_overflow() -> None:
    with pytest.raises(RuntimeProvisioningError, match="regular files and directories"):
        RuntimeProvisioner._validate_7z_listing(
            "runtime/link.dll\n",
            "lrwxrwxrwx  0 0  0  0 Jan 01  2026 runtime/link.dll -> outside.dll\n",
            max_entries=10,
            max_uncompressed_bytes=100,
        )

    with pytest.raises(RuntimeProvisioningError, match="allowed size"):
        RuntimeProvisioner._validate_7z_listing(
            "runtime/first.dll\nruntime/second.dll\n",
            (
                "-rw-r--r--  0 0  0  60 Jan 01  2026 runtime/first.dll\n"
                "-rw-r--r--  0 0  0  60 Jan 01  2026 runtime/second.dll\n"
            ),
            max_entries=10,
            max_uncompressed_bytes=100,
        )


async def test_explicit_external_runtime_is_never_replaced(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"managed"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    external = tmp_path / "external-llama-server.exe"
    external.write_bytes(b"external")
    settings.llama_executable = external
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("configured external runtimes must not be downloaded")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = await provisioner.ensure("llama.cpp")
        await provisioner.close()

    assert status.state == "ready"
    assert status.managed is False
    assert external.read_bytes() == b"external"


async def test_external_comfy_archive_is_provisioned_without_bundling_it(
    settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    llama_content = _zip_bytes({"llama-server.exe": b"llama"})
    comfy_content = _zip_bytes(
        {
            "python/python.exe": b"python",
            "python/Lib/site-packages/example-1.0.dist-info/METADATA": (
                b"Name: example\nVersion: 1.0\n"
            ),
            "ComfyUI/main.py": b"print('comfy')",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(
        manifest,
        llama_content=llama_content,
        comfy_content=comfy_content,
    )
    settings.prepare()
    expected_probe = {
        "python": "3.13.14",
        "comfyui": "0.28.0",
        "packages": {"example": "1.0"},
    }
    monkeypatch.setattr(
        runtime_provisioning.subprocess,
        "run",
        lambda *args, **kwargs: runtime_provisioning.subprocess.CompletedProcess(  # noqa: ARG005
            args[0],
            0,
            stdout=(
                f"{runtime_provisioning._RUNTIME_PROBE_SENTINEL}{json.dumps(expected_probe)}\n"
            ),
            stderr="",
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/comfy-runtime.zip")
        return httpx.Response(200, content=comfy_content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = await provisioner.ensure("comfyui")
        assert provisioner.status("comfyui").state == "ready"
        managed_asset = provisioner._manifest["engines"]["comfyui"]["runtime_assets"][
            "test-platform"
        ]
        managed_asset["runtime_probe"]["python"] = "3.13.15"
        assert provisioner.status("comfyui").state == "missing"
        managed_asset["runtime_probe"]["python"] = "3.13.14"
        assert provisioner.status("comfyui").state == "ready"
        await provisioner.close()

    assert status.state == "ready"
    assert status.distribution == "external-gpl-3.0"
    assert settings.comfy_executable
    assert settings.comfy_executable.read_bytes() == b"python"
    assert settings.comfy_directory
    assert (settings.comfy_directory / "main.py").is_file()


async def test_managed_runtime_integrity_change_is_not_reported_ready(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes(
        {
            "llama-server.exe": b"verified executable",
            "backend.dll": b"verified dependency",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        assert (await provisioner.ensure("llama.cpp")).state == "ready"
        assert settings.llama_executable
        marker = json.loads(
            (settings.llama_executable.parent / ".lm-atelier-runtime.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(marker["files"]) == {"backend.dll", "llama-server.exe"}
        (settings.llama_executable.parent / "backend.dll").write_bytes(b"changed dependency")

        assert provisioner.status("llama.cpp").state == "missing"
        await provisioner.close()


async def test_same_release_asset_correction_replaces_only_the_owned_runtime(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    original = _zip_bytes({"llama-server.exe": b"original executable"})
    corrected = _zip_bytes({"llama-server.exe": b"corrected executable"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=original)
    settings.prepare()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=original))
    ) as client:
        first = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        await first.ensure("llama.cpp")
        await first.close()

    _write_manifest(manifest, llama_content=corrected)
    requests: list[httpx.Request] = []

    def corrected_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=corrected)

    async with httpx.AsyncClient(transport=httpx.MockTransport(corrected_handler)) as client:
        second = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        assert second.status("llama.cpp").state == "missing"
        status = await second.ensure("llama.cpp")
        await second.close()

    assert len(requests) == 1
    assert status.state == "ready"
    assert settings.llama_executable
    assert settings.llama_executable.read_bytes() == b"corrected executable"


def test_platform_selection_gates_nvidia_runtime_generations(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime_provisioning.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime_provisioning.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(
        RuntimeProvisioner,
        "_nvidia_runtime_info",
        staticmethod(lambda: (610, 12)),
    )
    assert RuntimeProvisioner._platform_key("llama.cpp") == "windows-x86_64-nvidia"
    assert RuntimeProvisioner._platform_key("comfyui") == "windows-x86_64-nvidia-cu13"

    monkeypatch.setattr(
        RuntimeProvisioner,
        "_nvidia_runtime_info",
        staticmethod(lambda: (570, 6)),
    )
    assert RuntimeProvisioner._platform_key("comfyui") == "windows-x86_64-nvidia-cu126"

    monkeypatch.setattr(
        RuntimeProvisioner,
        "_nvidia_runtime_info",
        staticmethod(lambda: (550, 8)),
    )
    assert RuntimeProvisioner._platform_key("comfyui") == "windows-x86_64"


def test_platform_selection_advertises_the_pinned_ubuntu_nvidia_chat_asset(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime_provisioning.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_provisioning.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        runtime_provisioning.platform,
        "freedesktop_os_release",
        lambda: {"ID": "ubuntu"},
    )
    monkeypatch.setattr(
        RuntimeProvisioner,
        "_nvidia_runtime_info",
        staticmethod(lambda: (610, 12)),
    )

    assert RuntimeProvisioner._platform_key("llama.cpp") == "ubuntu-x86_64-nvidia"
    assert RuntimeProvisioner._platform_key("comfyui") == "ubuntu-x86_64-nvidia"


def test_nvidia_probe_parses_driver_and_compute_capability(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(runtime_provisioning.shutil, "which", lambda _name: "nvidia-smi")
    monkeypatch.setattr(
        runtime_provisioning.subprocess,
        "run",
        lambda *args, **kwargs: runtime_provisioning.subprocess.CompletedProcess(  # noqa: ARG005
            args[0],
            0,
            stdout="610.74, 12.0\n",
            stderr="",
        ),
    )

    assert RuntimeProvisioner._nvidia_runtime_info() == (610, 12)


async def test_blocked_runtime_security_status_prevents_automatic_download(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["engines"]["comfyui"]["security_status"] = "blocked"
    payload["engines"]["comfyui"]["security_message"] = (
        "Automatic setup is paused because dependency security review failed."
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("blocked runtimes must not be downloaded")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = provisioner.status("comfyui")
        assert status.state == "unsupported"
        assert status.security_status == "blocked"
        with pytest.raises(RuntimeProvisioningError, match="security review failed"):
            await provisioner.ensure("comfyui")
        await provisioner.close()


async def test_asset_level_security_block_prevents_automatic_download(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    asset = payload["engines"]["comfyui"]["runtime_assets"]["test-platform"]
    asset["security_status"] = "blocked"
    asset["security_message"] = "The selected compatibility tier is not audited."
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    settings.prepare()

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("blocked runtime assets must not be downloaded")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        status = provisioner.status("comfyui")
        assert status.state == "unsupported"
        assert status.security_status == "blocked"
        assert status.security_message == asset["security_message"]
        with pytest.raises(RuntimeProvisioningError, match="not audited"):
            await provisioner.ensure("comfyui")
        await provisioner.close()


async def test_security_overlay_requires_exact_files_and_rewrites_deterministically(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()
    overlay_files = {
        "LICENSE.txt": b"Python license",
        "python.exe": b"patched python",
        "python313._pth": b"python313.zip\n.\n",
    }
    overlay = {
        "name": "Test Python security overlay",
        "url": "https://runtime.test/python-overlay.zip",
        "sha256": "a" * 64,
        "size_bytes": 1,
        "max_uncompressed_bytes": 1024,
        "max_entries": 8,
        "archive_type": "zip",
        "target_directory": "runtime/python",
        "compatibility_tag": "cp313-win_amd64-embeddable",
        "license": "Python-2.0",
        "license_file": "LICENSE.txt",
        "expected_files": {
            name: hashlib.sha256(value).hexdigest() for name, value in overlay_files.items()
        },
        "rewrite_files": {"python313._pth": "../ComfyUI\npython313.zip\n.\nimport site\n"},
    }
    archive = tmp_path / "overlay.zip"
    archive.write_bytes(_zip_bytes(overlay_files))
    install_root = tmp_path / "install"
    target = install_root / "runtime" / "python"
    target.mkdir(parents=True)

    async with httpx.AsyncClient() as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        provisioner._apply_security_overlay(install_root, overlay, archive)
        assert (target / "python.exe").read_bytes() == b"patched python"
        assert (target / "python313._pth").read_bytes() == (
            b"../ComfyUI\npython313.zip\n.\nimport site\n"
        )

        unexpected = tmp_path / "unexpected-overlay.zip"
        unexpected.write_bytes(_zip_bytes({**overlay_files, "unreviewed.dll": b"unexpected"}))
        with pytest.raises(RuntimeProvisioningError, match="inventory"):
            provisioner._apply_security_overlay(
                install_root,
                overlay,
                unexpected,
            )
        await provisioner.close()


async def test_runtime_contract_rejects_inventory_and_probe_drift(
    settings,
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    settings.prepare()
    install_root = tmp_path / "staged"
    executable = install_root / "python" / "python.exe"
    site_packages = executable.parent / "Lib" / "site-packages"
    identity = "example-1.0.dist-info"
    (site_packages / identity).mkdir(parents=True)
    executable.write_bytes(b"python")
    comfy_directory = install_root / "ComfyUI"
    comfy_directory.mkdir()
    (comfy_directory / "main.py").write_text("", encoding="utf-8")
    probe = {
        "python": "3.13.14",
        "comfyui": "0.28.0",
        "imports": ["example"],
        "packages": {"example": "1.0"},
    }
    asset = {
        "dependency_inventory_count": 1,
        "dependency_inventory_sha256": _inventory_sha256([identity]),
        "runtime_probe": probe,
    }
    installed = {
        "executable": executable,
        "directory": comfy_directory,
    }
    expected_result = {
        "python": probe["python"],
        "comfyui": probe["comfyui"],
        "packages": probe["packages"],
    }
    probe_result = runtime_provisioning.subprocess.CompletedProcess(
        [str(executable)],
        0,
        stdout=(f"{runtime_provisioning._RUNTIME_PROBE_SENTINEL}{json.dumps(expected_result)}\n"),
        stderr="",
    )
    monkeypatch.setattr(
        runtime_provisioning.subprocess,
        "run",
        lambda *args, **kwargs: probe_result,  # noqa: ARG005
    )

    async with httpx.AsyncClient() as client:
        provisioner = RuntimeProvisioner(
            settings,
            manifest_path=manifest,
            client=client,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        provisioner._verify_runtime_contract(install_root, asset, installed)

        (site_packages / "unreviewed-2.0.dist-info").mkdir()
        with pytest.raises(RuntimeProvisioningError, match="inventory"):
            provisioner._verify_runtime_contract(install_root, asset, installed)
        (site_packages / "unreviewed-2.0.dist-info").rmdir()

        drifted_result = {
            **expected_result,
            "python": "3.13.15",
        }
        monkeypatch.setattr(
            runtime_provisioning.subprocess,
            "run",
            lambda *args, **kwargs: runtime_provisioning.subprocess.CompletedProcess(  # noqa: ARG005
                args[0],
                0,
                stdout=(
                    f"{runtime_provisioning._RUNTIME_PROBE_SENTINEL}{json.dumps(drifted_result)}\n"
                ),
                stderr="",
            ),
        )
        with pytest.raises(RuntimeProvisioningError, match="versions"):
            provisioner._verify_runtime_contract(install_root, asset, installed)
        await provisioner.close()


def test_comfy_manifest_fails_closed_when_audit_contract_is_omitted(
    tmp_path: Path,
) -> None:
    content = _zip_bytes({"llama-server.exe": b"llama"})
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    asset = payload["engines"]["comfyui"]["runtime_assets"]["test-platform"]

    del asset["dependency_review"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeProvisioningError, match="review reference"):
        RuntimeProvisioner._read_manifest(manifest)

    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["engines"]["comfyui"]["runtime_assets"]["test-platform"]["runtime_probe"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeProvisioningError, match="compatibility probe"):
        RuntimeProvisioner._read_manifest(manifest)

    _write_manifest(manifest, llama_content=content)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["engines"]["comfyui"]["runtime_assets"]["test-platform"]["security_overlays"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeProvisioningError, match="overlay list"):
        RuntimeProvisioner._read_manifest(manifest)


def test_pinned_comfy_review_accounts_for_every_distribution_and_license() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    manifest_path = repository_root / "packaging" / "engines.json"
    manifest = RuntimeProvisioner._read_manifest(manifest_path)
    definition = manifest["engines"]["comfyui"]
    assets = definition["runtime_assets"]
    review_path = repository_root / "packaging" / "runtime-reviews" / "comfyui-v0.28.0.json"
    review_bytes = review_path.read_bytes()
    review = json.loads(review_bytes)

    assert hashlib.sha256(review_bytes).hexdigest() == (
        "138a1432a49fafe465ea74c1b38c8211dc3b0e0a9fc1ae1d237067e3c92861a5"
    )
    assert review["vulnerability_audit"] == {
        "tool": "pip-audit 2.10.1",
        "service": "OSV",
        "dependency_count": 89,
        "known_vulnerabilities": 0,
        "requirements_source": (
            "Exact dist-info identities from both portable archive indexes; "
            "CUDA local version suffixes were normalized only for advisory lookup."
        ),
        "advisory_sources": [
            "https://github.com/Comfy-Org/ComfyUI/security/advisories",
            "https://github.com/pytorch/pytorch/security/advisories",
            "https://docs.python.org/3.13/whatsnew/changelog.html",
            "https://docs.python.org/3.12/whatsnew/changelog.html",
        ],
    }
    expected_review_hash = hashlib.sha256(review_bytes).hexdigest()
    expected_inventory = {
        "windows-x86_64-nvidia-cu13": (
            "f12087e3dfc278fc1f6521b6c61f8df03bdc9b0322df9c6464fb78a6dd08b69c"
        ),
        "windows-x86_64-nvidia-cu126": (
            "2917012ad55e024468a0e0ca2cf128db1cdac17d1d055008759f9842839e71c2"
        ),
    }
    for asset_key, inventory_hash in expected_inventory.items():
        asset = assets[asset_key]
        reviewed_asset = review["assets"][asset_key]
        distributions = reviewed_asset["distributions"]
        assert asset["dependency_inventory_count"] == len(distributions) == 89
        assert asset["dependency_inventory_sha256"] == inventory_hash
        assert asset["dependency_review"]["sha256"] == expected_review_hash
        assert all(
            distribution["dist_info"]
            and distribution["name"]
            and distribution["version"]
            and distribution["license"]
            and distribution["license_source"].startswith("https://")
            for distribution in distributions
        )
        canonical = (
            "\n".join(sorted(distribution["dist_info"] for distribution in distributions)) + "\n"
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == inventory_hash

    cu13 = assets["windows-x86_64-nvidia-cu13"]
    cu13_identities = {
        item["dist_info"]: item
        for item in review["assets"]["windows-x86_64-nvidia-cu13"]["distributions"]
    }
    assert cu13_identities["torch-2.13.0+cu130.dist-info"]["license"].startswith("Apache-2.0")
    assert cu13_identities["comfy_aimdo-0.4.10.dist-info"]["license"] == ("GPL-3.0-only")
    assert cu13["runtime_probe"]["python"] == "3.13.14"
    assert cu13["runtime_probe"]["packages"]["torch"] == "2.13.0+cu130"
    overlay = cu13["security_overlays"][0]
    assert overlay["url"] == (
        "https://www.python.org/ftp/python/3.13.14/python-3.13.14-embed-amd64.zip"
    )
    assert overlay["sha256"] == ("90b4e5b9898b72d744650524bff92377c367f44bd5fbd09e3148656c080ad907")
    assert overlay["compatibility_tag"] == "cp313-win_amd64-embeddable"
    assert overlay["license"] == "Python-2.0"
    assert len(overlay["expected_files"]) == 34
    assert assets["windows-x86_64-nvidia-cu126"]["security_status"] == "blocked"
