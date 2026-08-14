from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from packaging.markers import default_environment

import local_lm.comfy_registry_wheel_environments as environment_module
from local_lm.comfy_registry_dependencies import plan_comfy_registry_dependencies
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
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


def _wheel_content(
    *,
    extra_path: str | None = None,
    extra_content: bytes = b"unsafe",
    compressible_bytes: int | None = None,
) -> bytes:
    files: dict[str, bytes] = {}
    filler = chr(10) + "# " + "a" * 97
    padding = filler * (compressible_bytes // 100) if compressible_bytes else ""
    files["alpha/__init__.py"] = ("VALUE = 1" + padding).encode()
    files["alpha-1.0.dist-info/METADATA"] = (
        chr(10).join(["Metadata-Version: 2.4", "Name: alpha", "Version: 1.0", ""]).encode()
    )
    files["alpha-1.0.dist-info/WHEEL"] = (
        chr(10).join(["Wheel-Version: 1.0", "Tag: py3-none-any", ""]).encode()
    )
    if extra_path is not None:
        files[extra_path] = extra_content
    record_path = "alpha-1.0.dist-info/RECORD"
    files[record_path] = _record_bytes(files, record_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, value in files.items():
            archive.writestr(relative, value)

    return buffer.getvalue()


def _record_bytes(files: dict[str, bytes], record_path: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for relative, value in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode()
        writer.writerow((relative, f"sha256={digest}", len(value)))
    writer.writerow((record_path, "", ""))
    return output.getvalue().encode()


def _wheel_files(content: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return {entry.filename: archive.read(entry) for entry in archive.infolist()}


def _wheel_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, value in files.items():
            archive.writestr(relative, value)
    return output.getvalue()


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


def _empty_closure(
    runtime_distributions: tuple[ComfyRegistryRuntimeDistribution, ...] = (),
) -> ComfyRegistryWheelClosure:
    manifest = resolve_comfy_registry_wheel_artifacts(
        plan_comfy_registry_dependencies([]),
        {},
        marker_environment=_marker_environment(),
        runtime_distributions=runtime_distributions,
        supported_tags=(_TAG,),
    )
    return plan_comfy_registry_wheel_closure(
        manifest,
        {},
        marker_environment=_marker_environment(),
        runtime_distributions=runtime_distributions,
    )


def _wheel_file(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "alpha-1.0-py3-none-any.whl"
    path.write_bytes(content)
    return path


def _destination(tmp_path: Path, closure: ComfyRegistryWheelClosure) -> Path:
    return tmp_path / f"registry-wheels-v3-{closure.closure_sha256}"


def _installed_distribution(
    target: Path,
    wheel_content: bytes,
    *,
    version: str = "1.0",
) -> None:
    record_path = "alpha-1.0.dist-info/RECORD"
    with zipfile.ZipFile(io.BytesIO(wheel_content)) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or entry.filename == record_path:
                continue
            destination = target.joinpath(*Path(entry.filename).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(entry))
    metadata = target / "alpha-1.0.dist-info" / "METADATA"
    if version != "1.0":
        metadata.write_text(
            chr(10).join(["Metadata-Version: 2.4", "Name: alpha", f"Version: {version}", ""]),
            encoding="utf-8",
        )
    dist_info = target / "alpha-1.0.dist-info"
    (dist_info / "INSTALLER").write_text("pip\n", encoding="utf-8")
    (dist_info / "REQUESTED").write_bytes(b"")
    (dist_info / "direct_url.json").write_text("{}\n", encoding="utf-8")
    _rewrite_installed_record(target)


def _rewrite_installed_record(target: Path) -> None:
    record_path = "alpha-1.0.dist-info/RECORD"
    final_files = {
        path.relative_to(target).as_posix(): path.read_bytes()
        for path in target.rglob("*")
        if path.is_file() and path.relative_to(target).as_posix() != record_path
    }
    (target / record_path).write_bytes(_record_bytes(final_files, record_path))


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
        _installed_distribution(target, content)

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
    assert report.file_count == 7
    assert report.total_bytes > 0
    assert [(item.name, item.version) for item in report.distributions] == [("alpha", "1.0")]
    manifest = json.loads((destination / "environment-manifest.json").read_bytes())
    assert manifest["closure_sha256"] == closure.closure_sha256
    assert manifest["version"] == 3
    assert manifest["ownership_attestation"] == "wheel-source-record-v1"
    assert manifest["runtime_distributions"] == []
    assert manifest["file_count"] == 7
    assert [item["path"] for item in manifest["inventory"]] == [
        "alpha",
        "alpha/__init__.py",
        "alpha-1.0.dist-info",
        "alpha-1.0.dist-info/direct_url.json",
        "alpha-1.0.dist-info/INSTALLER",
        "alpha-1.0.dist-info/METADATA",
        "alpha-1.0.dist-info/RECORD",
        "alpha-1.0.dist-info/REQUESTED",
        "alpha-1.0.dist-info/WHEEL",
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


@pytest.mark.parametrize("mutation", ["missing", "extra", "hash", "noncanonical-hash", "duplicate"])
def test_source_record_must_be_an_exact_honest_archive_bijection(
    tmp_path: Path,
    mutation: str,
) -> None:
    files = _wheel_files(_wheel_content())
    record_path = "alpha-1.0.dist-info/RECORD"
    rows = files[record_path].decode().splitlines()
    if mutation == "missing":
        rows.pop(0)
    elif mutation == "extra":
        rows.insert(0, "not-in-wheel.py,sha256=" + "A" * 43 + ",1")
    elif mutation == "hash":
        columns = rows[0].split(",")
        columns[1] = "sha256=" + "A" * 43
        rows[0] = ",".join(columns)
    elif mutation == "noncanonical-hash":
        columns = rows[0].split(",")
        columns[1] = (
            columns[1][:-1]
            + {
                "A": "B",
                "Q": "R",
                "g": "h",
                "w": "x",
            }[columns[1][-1]]
        )
        rows[0] = ",".join(columns)
    else:
        rows.insert(0, rows[0])
    files[record_path] = ("\n".join(rows) + "\n").encode()
    source = _wheel_file(tmp_path, _wheel_bytes(files))

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        environment_module._inspect_wheel_archive(source, set())

    assert raised.value.code == "invalid_wheel_record"


def test_source_paths_are_compared_after_data_scheme_rewriting(tmp_path: Path) -> None:
    files = _wheel_files(_wheel_content())
    record_path = "alpha-1.0.dist-info/RECORD"
    files.pop(record_path)
    files["alpha-1.0.data/purelib/alpha/__init__.py"] = b"replacement"
    files[record_path] = _record_bytes(files, record_path)
    source = _wheel_file(tmp_path, _wheel_bytes(files))

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        environment_module._inspect_wheel_archive(source, set())

    assert raised.value.code == "overlapping_wheel_archives"


@pytest.mark.parametrize(
    "aliases",
    [
        ("alpha/Thing.py", "alpha/thing.py"),
        ("alpha/caf\u00e9.py", "alpha/cafe\u0301.py"),
    ],
)
def test_source_paths_refuse_windows_and_unicode_aliases(
    tmp_path: Path,
    aliases: tuple[str, str],
) -> None:
    files = _wheel_files(_wheel_content())
    record_path = "alpha-1.0.dist-info/RECORD"
    files.pop(record_path)
    files[aliases[0]] = b"first"
    files[aliases[1]] = b"second"
    files[record_path] = _record_bytes(files, record_path)
    source = _wheel_file(tmp_path, _wheel_bytes(files))

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        environment_module._inspect_wheel_archive(source, set())

    assert raised.value.code == "invalid_wheel_record"


def test_source_paths_refuse_file_directory_collisions(tmp_path: Path) -> None:
    files = _wheel_files(_wheel_content())
    record_path = "alpha-1.0.dist-info/RECORD"
    files.pop(record_path)
    files["alpha/collision"] = b"file"
    files["alpha/collision/child.py"] = b"child"
    files[record_path] = _record_bytes(files, record_path)
    source = _wheel_file(tmp_path, _wheel_bytes(files))

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        environment_module._inspect_wheel_archive(source, set())

    assert raised.value.code == "invalid_wheel_record"


def test_data_projection_refuses_installed_file_directory_collisions(
    tmp_path: Path,
) -> None:
    files = _wheel_files(_wheel_content())
    record_path = "alpha-1.0.dist-info/RECORD"
    files.pop(record_path)
    files["collision"] = b"file"
    files["alpha-1.0.data/purelib/collision/child.py"] = b"child"
    files[record_path] = _record_bytes(files, record_path)
    source = _wheel_file(tmp_path, _wheel_bytes(files))

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        environment_module._inspect_wheel_archive(source, set())

    assert raised.value.code == "overlapping_wheel_archives"


@pytest.mark.parametrize(
    "relative",
    [
        "alpha/CON.py",
        "alpha/lpt9.txt",
        "alpha/name.py.",
        "alpha/name.py ",
        "alpha/control\x1f.py",
    ],
)
def test_source_paths_refuse_unstable_windows_components(
    tmp_path: Path,
    relative: str,
) -> None:
    files = _wheel_files(_wheel_content())
    record_path = "alpha-1.0.dist-info/RECORD"
    files.pop(record_path)
    files[relative] = b"unsafe"
    files[record_path] = _record_bytes(files, record_path)
    source = _wheel_file(tmp_path, _wheel_bytes(files))

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        environment_module._inspect_wheel_archive(source, set())

    assert raised.value.code == "unsafe_wheel_archive"


@pytest.mark.parametrize(
    ("relative", "code"),
    [
        ("alpha-1.0.data/scripts/alpha-tool", "unsupported_wheel_scheme"),
        ("alpha-1.0.data/data/share/alpha.txt", "unsupported_wheel_scheme"),
        ("alpha-1.0.dist-info/INSTALLER", "invalid_wheel_record"),
    ],
)
def test_unstable_schemes_and_generated_metadata_are_not_source_authority(
    tmp_path: Path,
    relative: str,
    code: str,
) -> None:
    files = _wheel_files(_wheel_content())
    record_path = "alpha-1.0.dist-info/RECORD"
    files.pop(record_path)
    files[relative] = b"spoofed"
    files[record_path] = _record_bytes(files, record_path)
    source = _wheel_file(tmp_path, _wheel_bytes(files))

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        environment_module._inspect_wheel_archive(source, set())

    assert raised.value.code == code


@pytest.mark.parametrize("mutation", ["changed-source", "extra-final"])
async def test_post_pip_files_must_equal_the_source_ownership_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    content = _wheel_content()
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)

    async def fake_run_pip(_python: Path, _wheels: tuple[Path, ...], target: Path) -> None:
        _installed_distribution(target, content)
        if mutation == "changed-source":
            (target / "alpha" / "__init__.py").write_text("VALUE = 2", encoding="utf-8")
        else:
            (target / "unowned.py").write_text("UNOWNED = True", encoding="utf-8")
        # An honest rewritten final RECORD is insufficient: it cannot grant
        # ownership that the immutable source wheel did not carry.
        _rewrite_installed_record(target)

    monkeypatch.setattr(environment_module, "_run_pip", fake_run_pip)

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=_destination(tmp_path, closure),
            media_worker_stopped=True,
        )

    assert raised.value.code == "wheel_ownership_mismatch"


async def test_real_pip_output_satisfies_the_exact_ownership_plan() -> None:
    # Keep CreateProcess.cwd below the legacy Windows MAX_PATH boundary. The
    # assertion is about pip's output contract, not pytest's nested temp name.
    with tempfile.TemporaryDirectory(prefix="lm-wheel-v3-") as temporary:
        root = Path(temporary)
        files = _wheel_files(_wheel_content())
        record_path = "alpha-1.0.dist-info/RECORD"
        files.pop(record_path)
        files["alpha-1.0.data/purelib/alpha/pure.py"] = b"PURE = 1\n"
        files["alpha-1.0.data/platlib/alpha/plat.py"] = b"PLAT = 1\n"
        files[record_path] = _record_bytes(files, record_path)
        content = _wheel_bytes(files)
        closure = _closure(content)
        source = _wheel_file(root, content)
        destination = _destination(root, closure)

        report = await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=destination,
            media_worker_stopped=True,
        )

        assert report.artifact_count == 1
        assert (destination / "site-packages" / "alpha" / "pure.py").read_bytes() == (b"PURE = 1\n")
        assert (destination / "site-packages" / "alpha" / "plat.py").read_bytes() == (b"PLAT = 1\n")
        assert (
            verify_comfy_registry_wheel_environment(
                destination,
                expected_closure_sha256=closure.closure_sha256,
                expected_environment_sha256=report.environment_sha256,
            )
            == report
        )


async def test_legacy_path_and_manifest_are_fixed_refusals(tmp_path: Path) -> None:
    closure = _empty_closure()
    destination = _destination(tmp_path, closure)
    report = await assemble_comfy_registry_wheel_environment(
        closure,
        {},
        python_executable=Path(sys.executable),
        destination=destination,
        media_worker_stopped=True,
    )
    legacy = tmp_path / f"registry-wheels-{closure.closure_sha256}"
    destination.rename(legacy)

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        verify_comfy_registry_wheel_environment(
            legacy,
            expected_closure_sha256=closure.closure_sha256,
            expected_environment_sha256=report.environment_sha256,
        )
    assert raised.value.code == "legacy_environment_manifest"

    legacy.rename(destination)
    manifest_path = destination / "environment-manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    payload["version"] = 2
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    manifest_path.write_bytes(encoded)
    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        verify_comfy_registry_wheel_environment(
            destination,
            expected_closure_sha256=closure.closure_sha256,
            expected_environment_sha256=hashlib.sha256(encoded).hexdigest(),
        )
    assert raised.value.code == "legacy_environment_manifest"


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


async def test_environment_manifest_preserves_runtime_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = (ComfyRegistryRuntimeDistribution("torch", "2.13.0+cu130"),)
    closure = _empty_closure(runtime)
    destination = _destination(tmp_path, closure)

    async def unexpected_pip(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pip must not run for an empty closure")

    monkeypatch.setattr(environment_module, "_run_pip", unexpected_pip)
    assembled = await assemble_comfy_registry_wheel_environment(
        closure,
        {},
        python_executable=Path(sys.executable),
        destination=destination,
        media_worker_stopped=True,
    )
    verified = verify_comfy_registry_wheel_environment(
        destination,
        expected_closure_sha256=closure.closure_sha256,
        expected_environment_sha256=assembled.environment_sha256,
    )

    assert assembled.runtime_distributions == runtime
    assert verified.runtime_distributions == runtime
    manifest = json.loads((destination / "environment-manifest.json").read_text())
    assert manifest["version"] == 3
    assert manifest["ownership_attestation"] == "wheel-source-record-v1"


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


def test_small_but_very_compressible_entries_are_not_treated_as_bombs(tmp_path: Path) -> None:
    """Ordinary packages ship compressible test data, and it is not a bomb.

    pooch - a transitive dependency of scikit-image, and so of much of the
    scientific ecosystem - ships `tests/data/large-data.txt`: 0.1 MB expanded,
    compressing at 321 to 1. A ratio that high on an entry that small cannot
    fill anything, and refusing it made a whole workflow uninstallable.

    What bounds the real risk is total expansion, which the environment caps
    enforce across every archive together.
    """
    source = _wheel_file(tmp_path, _wheel_content(compressible_bytes=200_000))

    file_count, expanded, ownership = environment_module._inspect_wheel_archive(source, set())

    assert file_count > 0
    assert expanded > 200_000
    assert ownership.record_path == "alpha-1.0.dist-info/RECORD"


def test_aggregate_cap_refuses_before_any_archive_payload_is_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _wheel_file(tmp_path, _wheel_content())

    def unexpected_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("an over-limit archive must be refused from central metadata")

    monkeypatch.setattr(environment_module, "_read_wheel_entry", unexpected_read)
    monkeypatch.setattr(environment_module, "_wheel_entry_identity", unexpected_read)

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        environment_module._inspect_wheel_archive(
            source,
            set(),
            remaining_files=0,
            remaining_bytes=environment_module.MAX_REGISTRY_WHEEL_ENVIRONMENT_BYTES,
        )

    assert raised.value.code == "wheel_archive_too_large"


def test_directory_entries_cannot_hide_a_payload_from_ownership(tmp_path: Path) -> None:
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("hidden/", b"payload")
    source = _wheel_file(tmp_path, content.getvalue())

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        environment_module._inspect_wheel_archive(source, set())

    assert raised.value.code == "unsafe_wheel_archive"


async def test_a_large_entry_at_an_absurd_ratio_is_still_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard still does its job where the job exists.

    Big enough to matter and compressed far beyond anything real, which is what
    a decompression bomb looks like.
    """

    async def unexpected_pip(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a bomb must not reach pip")

    monkeypatch.setattr(environment_module, "_run_pip", unexpected_pip)

    content = _wheel_content(compressible_bytes=96 * 1024 * 1024)
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)

    with pytest.raises(ComfyRegistryWheelEnvironmentError) as raised:
        await assemble_comfy_registry_wheel_environment(
            closure,
            {source.name: source},
            python_executable=Path(sys.executable),
            destination=_destination(tmp_path, closure),
            media_worker_stopped=True,
        )

    assert raised.value.code == "unsafe_wheel_archive"
    # Named, so the reader knows which entry in which wheel was refused.
    assert "alpha/__init__.py" in str(raised.value)


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
        _installed_distribution(target, content, version="2.0")

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
    content = _wheel_content(
        extra_path="runtime-hook.pth",
        extra_content=b"import unreviewed_code\n",
    )
    closure = _closure(content)
    source = _wheel_file(tmp_path, content)
    destination = _destination(tmp_path, closure)

    async def fake_run_pip(_python: Path, _wheels: tuple[Path, ...], target: Path) -> None:
        _installed_distribution(target, content)

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
