from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

import local_lm.comfy_editor_bridge as editor_bridge
from local_lm.comfy_editor_bridge import (
    BRIDGE_COORDINATOR_CONFIG,
    ComfyEditorBridgeError,
    bridge_directory_name,
    inspect_comfy_editor_bridge_support,
    prepare_comfy_editor_bridge,
    stage_comfy_editor_bridge,
)
from local_lm.comfy_version_support import (
    CertifiedComfyCompatibility,
    ComfyCompatibilityContract,
    VersionInterval,
)

COORDINATOR_ORIGINS = (
    "http://127.0.0.1:12340",
    "http://localhost:12340",
    "http://[::1]:12340",
)


def _stage(custom_node_root: Path) -> Path:
    return stage_comfy_editor_bridge(
        custom_node_root,
        coordinator_origins=COORDINATOR_ORIGINS,
    )


def _runtime(
    root: Path,
    *,
    comfyui_version: str = "0.28.0",
    frontend_version: str = "1.45.21",
) -> tuple[Path, Path, Path]:
    portable = root / "ComfyUI_windows_portable"
    executable = portable / "python_embeded" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"python")
    directory = portable / "ComfyUI"
    directory.mkdir()
    (directory / "main.py").write_text("", encoding="utf-8")
    site_packages = executable.parent / "Lib" / "site-packages"
    dist_info = site_packages / f"comfyui_frontend_package-{frontend_version}.dist-info"
    dist_info.mkdir(parents=True)
    (directory / "comfyui_version.py").write_text(
        f'__version__ = "{comfyui_version}"' + chr(10),
        encoding="utf-8",
    )
    (dist_info / "METADATA").write_text(
        chr(10).join(
            [
                "Metadata-Version: 2.4",
                "Name: comfyui-frontend-package",
                f"Version: {frontend_version}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return executable, directory, site_packages


def test_exact_audited_runtime_supports_the_editor_bridge(tmp_path: Path) -> None:
    executable, directory, _site_packages = _runtime(tmp_path)

    support = inspect_comfy_editor_bridge_support(
        comfy_executable=executable,
        comfy_directory=directory,
    )

    assert support.supported
    assert support.code == "ready"
    assert support.comfyui_version == "0.28.0"
    assert support.frontend_version == "1.45.21"


@pytest.mark.parametrize(
    ("comfyui_version", "frontend_version", "code"),
    [
        ("0.27.0", "1.45.21", "workflow-editor-comfyui-unsupported"),
        ("0.28.0", "1.46.0", "workflow-editor-frontend-unsupported"),
    ],
)
def test_unpinned_runtime_versions_fail_closed_without_blocking_media(
    tmp_path: Path,
    comfyui_version: str,
    frontend_version: str,
    code: str,
) -> None:
    executable, directory, _site_packages = _runtime(
        tmp_path,
        comfyui_version=comfyui_version,
        frontend_version=frontend_version,
    )

    prepared = prepare_comfy_editor_bridge(
        comfy_executable=executable,
        comfy_directory=directory,
        custom_node_root=directory / "custom_nodes",
        coordinator_origins=COORDINATOR_ORIGINS,
    )

    assert not prepared.support.supported
    assert prepared.support.code == code
    expected = (
        f"Native workflow editing requires ComfyUI 0.28.0; the configured runtime "
        f"uses {comfyui_version}."
        if code == "workflow-editor-comfyui-unsupported"
        else (
            "Native workflow editing requires the ComfyUI frontend 1.45.21; "
            f"the configured runtime uses {frontend_version}."
        )
    )
    assert prepared.support.message == expected
    assert prepared.support.comfyui_version == comfyui_version
    assert prepared.support.frontend_version == frontend_version
    assert prepared.folder is None
    assert not (directory / "custom_nodes").exists()


def test_individually_supported_but_uncertified_runtime_pair_is_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ComfyCompatibilityContract(
        certified=(
            CertifiedComfyCompatibility(
                comfyui=VersionInterval("0.28.0", "0.28.0"),
                frontend=VersionInterval("1.45.21", "1.45.21"),
            ),
            CertifiedComfyCompatibility(
                comfyui=VersionInterval("0.29.0", "0.29.0"),
                frontend=VersionInterval("1.46.0", "1.46.0"),
            ),
        )
    )
    monkeypatch.setattr(editor_bridge, "COMFY_VERSION_SUPPORT", contract)
    executable, directory, _site_packages = _runtime(
        tmp_path,
        comfyui_version="0.28.0",
        frontend_version="1.46.0",
    )

    support = inspect_comfy_editor_bridge_support(
        comfy_executable=executable,
        comfy_directory=directory,
    )

    assert not support.supported
    assert support.code == "workflow-editor-runtime-pair-unsupported"
    assert support.message == (
        "Native workflow editing has not certified ComfyUI 0.28.0 with the ComfyUI frontend 1.46.0."
    )
    assert support.comfyui_version == "0.28.0"
    assert support.frontend_version == "1.46.0"


def test_conflicting_frontend_distributions_fail_closed(tmp_path: Path) -> None:
    executable, directory, site_packages = _runtime(tmp_path)
    duplicate = site_packages / "comfyui_frontend_package-1.44.0.dist-info"
    duplicate.mkdir()
    (duplicate / "METADATA").write_text(
        chr(10).join(["Name: comfyui-frontend-package", "Version: 1.44.0", ""]),
        encoding="utf-8",
    )

    support = inspect_comfy_editor_bridge_support(
        comfy_executable=executable,
        comfy_directory=directory,
    )

    assert not support.supported
    assert support.code == "workflow-editor-frontend-ambiguous"


def test_bridge_staging_is_hash_pinned_and_idempotent(tmp_path: Path) -> None:
    custom_nodes = tmp_path / "custom_nodes"

    first = _stage(custom_nodes)
    second = _stage(custom_nodes)

    assert first == second == custom_nodes.resolve() / bridge_directory_name(COORDINATOR_ORIGINS)
    assert sorted(
        path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file()
    ) == [
        "__init__.py",
        "js/lm_atelier_workflow_editor.js",
        BRIDGE_COORDINATOR_CONFIG,
    ]
    assert "NODE_CLASS_MAPPINGS: dict[str, object] = {}" in (first / "__init__.py").read_text(
        encoding="utf-8"
    )
    config = (first / BRIDGE_COORDINATOR_CONFIG).read_text(encoding="utf-8")
    for origin in COORDINATOR_ORIGINS:
        assert origin in config
    assert "secret" not in config.casefold()


def test_new_bridge_assets_leave_an_older_directory_untouched(tmp_path: Path) -> None:
    custom_nodes = tmp_path / "custom_nodes"
    config_digest = hashlib.sha256(
        editor_bridge._coordinator_config(COORDINATOR_ORIGINS)
    ).hexdigest()
    older = (
        custom_nodes / f"_lm_atelier_workflow_editor_bridge_v2_9d1c852c52e1_{config_digest[:12]}"
    )
    cache = older / "__pycache__" / "__init__.cpython-313.pyc"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"previously generated bytecode")

    current = _stage(custom_nodes)

    assert current != older
    assert (current / "__init__.py").is_file()
    assert not (current / "__pycache__").exists()
    assert cache.read_bytes() == b"previously generated bytecode"


def test_bridge_origin_configuration_changes_the_staged_identity(tmp_path: Path) -> None:
    custom_nodes = tmp_path / "custom_nodes"
    first = _stage(custom_nodes)
    second_origins = ("http://127.0.0.1:22340",)
    second = stage_comfy_editor_bridge(
        custom_nodes,
        coordinator_origins=second_origins,
    )

    assert first != second
    assert first.name == bridge_directory_name(COORDINATOR_ORIGINS)
    assert second.name == bridge_directory_name(second_origins)
    assert "http://127.0.0.1:22340" in (second / BRIDGE_COORDINATOR_CONFIG).read_text(
        encoding="utf-8"
    )


def test_bridge_staging_accepts_a_single_pass_origin_iterable(tmp_path: Path) -> None:
    origins = (origin for origin in COORDINATOR_ORIGINS)

    destination = stage_comfy_editor_bridge(
        tmp_path / "custom_nodes",
        coordinator_origins=origins,
    )

    assert destination.name == bridge_directory_name(COORDINATOR_ORIGINS)


@pytest.mark.parametrize(
    "origin",
    (
        "http://127.0.0.1",
        "https://127.0.0.1:12340",
        "http://127.0.0.1:12340/path",
        "http://user@127.0.0.1:12340",
        "http://example.com:12340",
    ),
)
def test_bridge_refuses_invalid_coordinator_origins(tmp_path: Path, origin: str) -> None:
    with pytest.raises(ComfyEditorBridgeError) as refused:
        stage_comfy_editor_bridge(
            tmp_path / "custom_nodes",
            coordinator_origins=(origin,),
        )

    assert refused.value.code == "workflow-editor-coordinator-origin-invalid"


def test_modified_staged_origin_configuration_is_refused(tmp_path: Path) -> None:
    destination = _stage(tmp_path / "custom_nodes")
    (destination / BRIDGE_COORDINATOR_CONFIG).write_text(
        'export const COORDINATOR_ORIGINS = ["http://127.0.0.1:9"];\n',
        encoding="utf-8",
    )

    with pytest.raises(ComfyEditorBridgeError) as refused:
        _stage(tmp_path / "custom_nodes")

    assert refused.value.code == "workflow-editor-bridge-integrity-failed"


def test_modified_bundled_bridge_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "assets"
    (source / "js").mkdir(parents=True)
    (source / "__init__.py").write_text("changed", encoding="utf-8")
    (source / "js" / "lm_atelier_workflow_editor.js").write_text("changed", encoding="utf-8")
    monkeypatch.setattr(editor_bridge, "_BRIDGE_ASSET_DIRECTORY", source)

    with pytest.raises(ComfyEditorBridgeError) as refused:
        _stage(tmp_path / "custom_nodes")

    assert refused.value.code == "workflow-editor-bridge-integrity-failed"


def test_modified_staged_bridge_is_refused(tmp_path: Path) -> None:
    destination = _stage(tmp_path / "custom_nodes")
    (destination / "js" / "lm_atelier_workflow_editor.js").write_text("changed", encoding="utf-8")

    with pytest.raises(ComfyEditorBridgeError) as refused:
        _stage(tmp_path / "custom_nodes")

    assert refused.value.code == "workflow-editor-bridge-integrity-failed"


def test_failed_atomic_stage_leaves_no_partial_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_nodes = tmp_path / "custom_nodes"

    def fail_rename(_source: Path, _destination: Path) -> None:
        raise OSError("locked")

    monkeypatch.setattr(os, "rename", fail_rename)

    with pytest.raises(ComfyEditorBridgeError) as refused:
        _stage(custom_nodes)

    assert refused.value.code == "workflow-editor-bridge-staging-failed"
    assert list(custom_nodes.iterdir()) == []


def _rename_refused(times: int, calls: list[str]) -> Callable[[Path, Path], None]:
    """A rename that is refused access `times` times and then goes ahead."""

    real_rename = os.rename

    def rename(source: Path, destination: Path) -> None:
        calls.append("rename")
        if len(calls) <= times:
            raise PermissionError(13, "held by another process")
        real_rename(source, destination)

    return rename


def test_a_briefly_held_staging_directory_is_renamed_once_it_is_released(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_nodes = tmp_path / "custom_nodes"
    calls: list[str] = []
    waits: list[float] = []
    monkeypatch.setattr(os, "rename", _rename_refused(2, calls))
    monkeypatch.setattr(editor_bridge.time, "sleep", waits.append)

    staged = _stage(custom_nodes)

    assert calls == ["rename"] * 3
    assert waits == [0.05, 0.1]
    assert [entry.name for entry in custom_nodes.iterdir()] == [staged.name]
    assert (staged / BRIDGE_COORDINATOR_CONFIG).is_file()


def test_a_staging_directory_held_too_long_fails_and_leaves_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_nodes = tmp_path / "custom_nodes"
    calls: list[str] = []
    waits: list[float] = []
    monkeypatch.setattr(os, "rename", _rename_refused(99, calls))
    monkeypatch.setattr(editor_bridge.time, "sleep", waits.append)

    with pytest.raises(ComfyEditorBridgeError) as refused:
        _stage(custom_nodes)

    assert refused.value.code == "workflow-editor-bridge-staging-failed"
    assert len(calls) == 5
    assert waits == [0.05, 0.1, 0.2, 0.4]
    assert list(custom_nodes.iterdir()) == []


def test_only_a_refusal_of_access_is_tried_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_nodes = tmp_path / "custom_nodes"
    calls: list[str] = []

    def fail_rename(_source: Path, _destination: Path) -> None:
        calls.append("rename")
        raise OSError(5, "input/output error")

    monkeypatch.setattr(os, "rename", fail_rename)
    monkeypatch.setattr(editor_bridge.time, "sleep", lambda _seconds: None)

    with pytest.raises(ComfyEditorBridgeError) as refused:
        _stage(custom_nodes)

    assert refused.value.code == "workflow-editor-bridge-staging-failed"
    assert calls == ["rename"]
    assert list(custom_nodes.iterdir()) == []


def test_a_destination_that_appears_while_held_is_checked_not_waited_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another start staged the same bridge meanwhile; its copy is verified and used."""

    custom_nodes = tmp_path / "custom_nodes"
    finished = _stage(tmp_path / "elsewhere")
    calls: list[str] = []
    waits: list[float] = []

    def rename(_source: Path, destination: Path) -> None:
        calls.append("rename")
        shutil.copytree(finished, destination)
        raise PermissionError(13, "held by another process")

    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(editor_bridge.time, "sleep", waits.append)

    staged = _stage(custom_nodes)

    assert calls == ["rename"]
    assert waits == []
    assert [entry.name for entry in custom_nodes.iterdir()] == [staged.name]


def test_bridge_script_uses_only_the_explicit_message_channel_save_path() -> None:
    script = (
        Path(editor_bridge.__file__).with_name("comfy_editor_bridge_assets")
        / "js"
        / "lm_atelier_workflow_editor.js"
    ).read_text(encoding="utf-8")

    for required in (
        'import { app } from "../../scripts/app.js"',
        'import { COORDINATOR_ORIGINS } from "./lm_atelier_workflow_editor_config.js"',
        "app.registerExtension",
        "actionBarButtons",
        "Save to LM Atelier",
        "app.loadGraphData",
        "app.graphToPrompt",
        "result?.workflow",
        "result?.output",
        "{ nonce: editorNonce, graph, prompt }",
        "window.parent !== window ? window.parent : window.opener",
        "event.source !== editorCoordinator",
        "coordinatorOrigins.has(event.origin)",
        "event.ports.length !== 1",
        "message.nonce !== editorNonce",
        "MAX_GRAPH_BYTES = 1024 * 1024",
        "editorCoordinator?.postMessage(",
        '    "*",',
    ):
        assert required in script
    for forbidden in (
        "URLSearchParams",
        "location.search",
        "document.referrer",
        "fetch(",
        "localStorage",
        "isLoopbackOrigin",
    ):
        assert forbidden not in script


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symbolic links are unavailable")
def test_linked_destination_is_never_trusted(tmp_path: Path) -> None:
    custom_nodes = tmp_path / "custom_nodes"
    custom_nodes.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = custom_nodes / bridge_directory_name(COORDINATOR_ORIGINS)
    try:
        destination.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symbolic links is not permitted")

    with pytest.raises(ComfyEditorBridgeError) as refused:
        _stage(custom_nodes)

    assert refused.value.code == "workflow-editor-bridge-integrity-failed"


def test_unreadable_runtime_metadata_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable metadata file is not treated as ordinary bytes.

    Path.exists still succeeds. lstat raising PermissionError is the
    unreadable half of the shared inspection. Mutating unreadable back to
    assume_regular opened the file and returned its contents.
    """
    path = tmp_path / "distribution.json"
    path.write_bytes(b'{"ok": true}')
    real_lstat = Path.lstat

    def fake_lstat(self: Path) -> os.stat_result:
        if self.resolve() == path.resolve():
            raise PermissionError("locked")
        return real_lstat(self)

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    with pytest.raises(ComfyEditorBridgeError) as refused:
        editor_bridge._read_bounded(path, 1024)
    assert refused.value.code == "workflow-editor-runtime-metadata-invalid"
