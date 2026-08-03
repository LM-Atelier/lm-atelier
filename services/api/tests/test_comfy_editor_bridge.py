from __future__ import annotations

import os
from pathlib import Path

import pytest

import local_lm.comfy_editor_bridge as editor_bridge
from local_lm.comfy_editor_bridge import (
    BRIDGE_DIRECTORY_NAME,
    ComfyEditorBridgeError,
    inspect_comfy_editor_bridge_support,
    prepare_comfy_editor_bridge,
    stage_comfy_editor_bridge,
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
    )

    assert not prepared.support.supported
    assert prepared.support.code == code
    assert prepared.folder is None
    assert not (directory / "custom_nodes").exists()


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

    first = stage_comfy_editor_bridge(custom_nodes)
    second = stage_comfy_editor_bridge(custom_nodes)

    assert first == second == custom_nodes.resolve() / BRIDGE_DIRECTORY_NAME
    assert sorted(
        path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file()
    ) == ["__init__.py", "js/lm_atelier_workflow_editor.js"]
    assert "NODE_CLASS_MAPPINGS: dict[str, object] = {}" in (first / "__init__.py").read_text(
        encoding="utf-8"
    )


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
        stage_comfy_editor_bridge(tmp_path / "custom_nodes")

    assert refused.value.code == "workflow-editor-bridge-integrity-failed"


def test_modified_staged_bridge_is_refused(tmp_path: Path) -> None:
    destination = stage_comfy_editor_bridge(tmp_path / "custom_nodes")
    (destination / "js" / "lm_atelier_workflow_editor.js").write_text("changed", encoding="utf-8")

    with pytest.raises(ComfyEditorBridgeError) as refused:
        stage_comfy_editor_bridge(tmp_path / "custom_nodes")

    assert refused.value.code == "workflow-editor-bridge-integrity-failed"


def test_failed_atomic_stage_leaves_no_partial_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_nodes = tmp_path / "custom_nodes"

    def fail_rename(_source: Path, _destination: Path) -> None:
        raise OSError("locked")

    monkeypatch.setattr(editor_bridge.os, "rename", fail_rename)

    with pytest.raises(ComfyEditorBridgeError) as refused:
        stage_comfy_editor_bridge(custom_nodes)

    assert refused.value.code == "workflow-editor-bridge-staging-failed"
    assert list(custom_nodes.iterdir()) == []


def test_bridge_script_uses_only_the_explicit_message_channel_save_path() -> None:
    script = (
        Path(editor_bridge.__file__).with_name("comfy_editor_bridge_assets")
        / "js"
        / "lm_atelier_workflow_editor.js"
    ).read_text(encoding="utf-8")

    for required in (
        'import { app } from "../../scripts/app.js"',
        "app.registerExtension",
        "actionBarButtons",
        "Save to LM Atelier",
        "app.loadGraphData",
        "app.graphToPrompt",
        "result?.workflow",
        "event.source !== window.opener",
        "!isLoopbackOrigin(event.origin)",
        "event.ports.length !== 1",
        "message.nonce !== editorNonce",
        "MAX_GRAPH_BYTES = 1024 * 1024",
        "window.opener.postMessage(",
        '    "*",',
    ):
        assert required in script
    for forbidden in (
        "URLSearchParams",
        "location.search",
        "document.referrer",
        "fetch(",
        "localStorage",
    ):
        assert forbidden not in script


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symbolic links are unavailable")
def test_linked_destination_is_never_trusted(tmp_path: Path) -> None:
    custom_nodes = tmp_path / "custom_nodes"
    custom_nodes.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = custom_nodes / BRIDGE_DIRECTORY_NAME
    try:
        destination.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symbolic links is not permitted")

    with pytest.raises(ComfyEditorBridgeError) as refused:
        stage_comfy_editor_bridge(custom_nodes)

    assert refused.value.code == "workflow-editor-bridge-integrity-failed"
