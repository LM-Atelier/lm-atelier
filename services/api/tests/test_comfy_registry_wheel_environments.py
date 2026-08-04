from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from packaging.markers import default_environment

import local_lm.comfy_registry_wheel_environments as environment_module
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_wheel_artifacts import resolve_comfy_registry_wheel_artifacts
from local_lm.comfy_registry_wheel_closure import (
    ComfyRegistryWheelClosure,
    plan_comfy_registry_wheel_closure,
)
from local_lm.comfy_registry_wheel_environments import (
    ComfyRegistryWheelEnvironmentError,
    assemble_comfy_registry_wheel_environment,
    verify_comfy_registry_wheel_environment,
)

_TAG = "py3-none-any"


def _marker_environment() -> dict[str, str]:
    environment = {key: str(value) for key, value in default_environment().items()}
    environment["extra"] = ""
    return environment


def _wheel_content(*, extra_path: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("alpha/__init__.py", "VALUE = 1")
        archive.writestr(
            "alpha-1.0.dist-info/METADATA",
            chr(10).join(["Metadata-Version: 2.4", "Name: alpha", "Version: 1.0", ""]),
        )
        archive.writestr(
            "alpha-1.0.dist-info/WHEEL",
            chr(10).join(["Wheel-Version: 1.0", "Tag: py3-none-any", ""]),
        )
        archive.writestr("alpha-1.0.dist-info/RECORD", "")
        if extra_path is not None:
            archive.writestr(extra_path, "unsafe")
    return buffer.getvalue()


def _closure(
    wheel_content: bytes,
    *,
    requirements: tuple[str, ...] = (),
) -> ComfyRegistryWheelClosure:
    metadata_lines = ["Metadata-Version: 2.4", "Name: alpha", "Version: 1.0"]
    metadata_lines.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    metadata = (chr(10).join(metadata_lines) + chr(10) * 2).encode()
    filename = "alpha-1.0-py3-none-any.whl"
    manifest = resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies(["alpha==1.0"]),
        {
            "alpha": {
                "meta": {"api-version": "1.4"},
                "name": "alpha",
                "files": [
                    {
                        "filename": filename,
                        "url": f"https://files.pythonhosted.org/packages/aa/{filename}",
                        "hashes": {"sha256": hashlib.sha256(wheel_content).hexdigest()},
                        "requires-python": ">=3.12",
                        "yanked": False,
                        "size": len(wheel_content),
                        "core-metadata": {"sha256": hashlib.sha256(metadata).hexdigest()},
                    }
                ],
            }
        },
        marker_environment=_marker_environment(),
        supported_tags=(_TAG,),
    )
    return plan_comfy_registry_wheel_closure(
        manifest,
        {filename: metadata},
        marker_environment=_marker_environment(),
    )


def _empty_closure() -> ComfyRegistryWheelClosure:
    manifest = resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies([]),
        {},
        marker_environment=_marker_environment(),
        supported_tags=(_TAG,),
    )
    return plan_comfy_registry_wheel_closure(
        manifest,
        {},
        marker_environment=_marker_environment(),
    )


def _wheel_file(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "alpha-1.0-py3-none-any.whl"
    path.write_bytes(content)
    return path


def _destination(tmp_path: Path, closure: ComfyRegistryWheelClosure) -> Path:
    return tmp_path / f"registry-wheels-{closure.closure_sha256}"


def _installed_distribution(target: Path, *, name: str = "alpha", version: str = "1.0") -> None:
    package = target / name.replace("-", "_")
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1" + chr(10), encoding="utf-8")
    dist_info = target / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        chr(10).join(["Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}", ""])
        + chr(10),
        encoding="utf-8",
    )


async def test_assembly_stages_verified_wheels_and_publishes_audited_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _wheel_content()
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)
    destination = _destination(tmp_path, closure)
    observed: list[Path] = []

    async def fake_run_pip(
        _python: Path,
        wheels: tuple[Path, ...],
        target: Path,
    ) -> None:
        observed.extend(wheels)
        assert wheels[0] != source
        assert wheels[0].parent.name == "wheels"
        assert wheels[0].read_bytes() == content
        _installed_distribution(target)

    monkeypatch.setattr(environment_module, "_run_pip", fake_run_pip)

    report = await assemble_comfy_registry_wheel_environment(
        closure,
        {source.name: source},
        python_executable=Path(sys.executable),
        destination=destination,
        media_worker_stopped=True,
    )

    assert len(observed) == 1
    assert destination.is_dir()
    assert not (destination / "wheels").exists()
    assert not (tmp_path / f".{destination.name}.lock").exists()
    assert report.closure_sha256 == closure.closure_sha256
    assert report.artifact_count == 1
    assert report.file_count == 2
    assert report.total_bytes > 0
    assert [(item.name, item.version) for item in report.distributions] == [("alpha", "1.0")]
    manifest = json.loads((destination / "environment-manifest.json").read_bytes())
    assert manifest["closure_sha256"] == closure.closure_sha256
    assert manifest["file_count"] == 2
    assert [item["path"] for item in manifest["inventory"]] == [
        "alpha",
        "alpha/__init__.py",
        "alpha-1.0.dist-info",
        "alpha-1.0.dist-info/METADATA",
    ]
    assert (
        report.environment_sha256
        == hashlib.sha256((destination / "environment-manifest.json").read_bytes()).hexdigest()
    )
    assert (
        verify_comfy_registry_wheel_environment(
            destination,
            expected_closure_sha256=closure.closure_sha256,
            expected_environment_sha256=report.environment_sha256,
        )
        == report
    )

    (destination / "site-packages" / "alpha" / "__init__.py").write_text(
        "VALUE = 2" + chr(10), encoding="utf-8"
    )
    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        verify_comfy_registry_wheel_environment(
            destination,
            expected_closure_sha256=closure.closure_sha256,
            expected_environment_sha256=report.environment_sha256,
        )
    assert raised.value.code == "environment_inventory_mismatch"


async def test_worker_must_be_stopped_before_any_other_validation(tmp_path: Path) -> None:
    closure = _closure(_wheel_content())

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {},
            python_executable=tmp_path / "missing-python",
            destination=_destination(tmp_path, closure),
            media_worker_stopped=False,
        )
    assert raised.value.code == "media_worker_running"


def test_managed_python_alias_resolves_to_an_exact_executable(tmp_path: Path) -> None:
    alias = tmp_path / "python-alias"
    try:
        alias.symlink_to(Path(sys.executable))
    except OSError:
        pytest.skip("file symlinks are unavailable")

    assert environment_module._python_executable(alias) == Path(sys.executable).resolve(strict=True)


async def test_empty_complete_closure_publishes_without_invoking_pip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = _empty_closure()
    destination = _destination(tmp_path, closure)

    async def unexpected_pip(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pip must not run for an empty closure")

    monkeypatch.setattr(environment_module, "_run_pip", unexpected_pip)

    report = await assemble_comfy_registry_wheel_environment(
        closure,
        {},
        python_executable=Path(sys.executable),
        destination=destination,
        media_worker_stopped=True,
    )

    assert report.artifact_count == 0
    assert report.file_count == 0
    assert report.total_bytes == 0
    assert report.distributions == ()
    assert (destination / "site-packages").is_dir()


async def test_incomplete_closure_cannot_be_assembled(tmp_path: Path) -> None:
    closure = _closure(_wheel_content(), requirements=("beta>=1",))
    assert closure.complete is False

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {},
            python_executable=Path(sys.executable),
            destination=_destination(tmp_path, closure),
            media_worker_stopped=True,
        )
    assert raised.value.code == "closure_incomplete"


@pytest.mark.parametrize(
    ("files", "expected_code"),
    [
        ({}, "missing_wheel_file"),
        ({"unexpected.whl": Path("unexpected.whl")}, "missing_wheel_file"),
        ({1: Path("unexpected.whl")}, "invalid_wheel_files"),
    ],
)
async def test_wheel_mapping_must_match_the_complete_closure(
    tmp_path: Path,
    files: dict[object, Path],
    expected_code: str,
) -> None:
    closure = _closure(_wheel_content())

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            files,  # type: ignore[arg-type]
            python_executable=Path(sys.executable),
            destination=_destination(tmp_path, closure),
            media_worker_stopped=True,
        )
    assert raised.value.code == expected_code


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [("truncate", "wheel_size_mismatch"), ("replace", "wheel_hash_mismatch")],
)
async def test_wheel_bytes_are_revalidated_before_staging(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    original = _wheel_content()
    closure = _closure(original)
    content = original[:-1] if mutation == "truncate" else bytes([original[0] ^ 1]) + original[1:]
    source = _wheel_file(tmp_path, content)

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=_destination(tmp_path, closure),
            media_worker_stopped=True,
        )
    assert raised.value.code == expected_code


async def test_unsafe_archive_path_is_rejected_before_pip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _wheel_content(extra_path="../escape.py")
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)

    async def unexpected_pip(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unsafe archives must not reach pip")

    monkeypatch.setattr(environment_module, "_run_pip", unexpected_pip)

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=_destination(tmp_path, closure),
            media_worker_stopped=True,
        )
    assert raised.value.code == "unsafe_wheel_archive"


async def test_install_failure_cleans_staging_lock_and_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _wheel_content()
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)
    destination = _destination(tmp_path, closure)

    async def fail_install(_python: Path, _wheels: tuple[Path, ...], _target: Path) -> None:
        raise ComfyRegistryWheelEnvironmentError("wheel_install_failed", "failed")

    monkeypatch.setattr(environment_module, "_run_pip", fail_install)

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=destination,
            media_worker_stopped=True,
        )
    assert raised.value.code == "wheel_install_failed"
    assert not destination.exists()
    assert not (tmp_path / f".{destination.name}.lock").exists()
    assert not list(tmp_path.glob(f".{destination.name}-*"))


async def test_cancellation_cleans_staging_and_preserves_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _wheel_content()
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)
    destination = _destination(tmp_path, closure)

    async def cancel_install(_python: Path, _wheels: tuple[Path, ...], _target: Path) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(environment_module, "_run_pip", cancel_install)

    with pytest.raises(asyncio.CancelledError):
        await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=destination,
            media_worker_stopped=True,
        )
    assert not destination.exists()
    assert not (tmp_path / f".{destination.name}.lock").exists()
    assert not list(tmp_path.glob(f".{destination.name}-*"))


async def test_post_install_audit_rejects_unexpected_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _wheel_content()
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)

    async def fake_run_pip(_python: Path, _wheels: tuple[Path, ...], target: Path) -> None:
        _installed_distribution(target, version="2.0")

    monkeypatch.setattr(environment_module, "_run_pip", fake_run_pip)

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=_destination(tmp_path, closure),
            media_worker_stopped=True,
        )
    assert raised.value.code == "distribution_set_mismatch"


async def test_pth_content_is_inert_audited_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _wheel_content()
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)
    destination = _destination(tmp_path, closure)

    async def fake_run_pip(_python: Path, _wheels: tuple[Path, ...], target: Path) -> None:
        _installed_distribution(target)
        (target / "runtime-hook.pth").write_text("import unreviewed_code\n", encoding="utf-8")

    monkeypatch.setattr(environment_module, "_run_pip", fake_run_pip)

    report = await assemble_comfy_registry_wheel_environment(
        closure,
        {source.name: source},
        python_executable=Path(sys.executable),
        destination=destination,
        media_worker_stopped=True,
    )

    manifest = json.loads((destination / "environment-manifest.json").read_bytes())
    pth = next(item for item in manifest["inventory"] if item["path"] == "runtime-hook.pth")
    assert pth["kind"] == "file"
    pth_path = destination / "site-packages" / "runtime-hook.pth"
    assert pth["sha256"] == hashlib.sha256(pth_path.read_bytes()).hexdigest()
    assert (
        verify_comfy_registry_wheel_environment(
            destination,
            expected_closure_sha256=closure.closure_sha256,
            expected_environment_sha256=report.environment_sha256,
        )
        == report
    )

    pth_path.write_text("import changed_code\n", encoding="utf-8")
    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        verify_comfy_registry_wheel_environment(
            destination,
            expected_closure_sha256=closure.closure_sha256,
            expected_environment_sha256=report.environment_sha256,
        )
    assert raised.value.code == "environment_inventory_mismatch"


async def test_existing_lock_and_destination_are_never_replaced(
    tmp_path: Path,
) -> None:
    content = _wheel_content()
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)
    destination = _destination(tmp_path, closure)
    lock = tmp_path / f".{destination.name}.lock"
    lock.write_text("busy", encoding="utf-8")

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=destination,
            media_worker_stopped=True,
        )
    assert raised.value.code == "environment_locked"
    assert lock.read_text(encoding="utf-8") == "busy"

    lock.unlink()
    destination.mkdir()
    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=destination,
            media_worker_stopped=True,
        )
    assert raised.value.code == "invalid_environment_destination"


async def test_mutated_closure_is_rejected_before_filesystem_changes(tmp_path: Path) -> None:
    closure = _closure(_wheel_content())
    mutated = replace(closure, complete=False)

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            mutated,
            {},
            python_executable=Path(sys.executable),
            destination=_destination(tmp_path, closure),
            media_worker_stopped=True,
        )
    assert raised.value.code == "invalid_closure"
    assert list(tmp_path.iterdir()) == []


async def test_pip_is_invoked_offline_in_isolated_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class CompletedProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def create_process(*args: object, **kwargs: object) -> CompletedProcess:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return CompletedProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    target = tmp_path / "target"
    target.mkdir()
    wheel = tmp_path / "alpha.whl"
    wheel.write_bytes(b"wheel")

    await environment_module._run_pip(Path(sys.executable), (wheel,), target)

    arguments = observed["args"]
    assert isinstance(arguments, tuple)
    assert arguments[:4] == (sys.executable, "-I", "-m", "pip")
    assert "--isolated" in arguments
    assert "--no-index" in arguments
    assert "--no-deps" in arguments
    assert "--target" in arguments
    assert arguments[-1] == str(wheel)
    keywords = observed["kwargs"]
    assert isinstance(keywords, dict)
    environment = keywords["env"]
    assert isinstance(environment, dict)
    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"


async def test_pip_timeout_kills_the_process_and_returns_a_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess:
        returncode: int | None = None
        killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            assert self.returncode is not None
            return self.returncode

    process = HangingProcess()

    async def create_process(*_args: object, **_kwargs: object) -> HangingProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(environment_module, "WHEEL_INSTALL_TIMEOUT_SECONDS", 0.001)
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await environment_module._run_pip(Path(sys.executable), (), target)
    assert raised.value.code == "wheel_install_timeout"
    assert process.killed is True
