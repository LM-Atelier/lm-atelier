from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from local_lm import filesystem_links
from local_lm.adapters import comfyui
from local_lm.adapters.comfyui import ComfyUIAdapter


def _layout(fixture: Path, kind: str) -> tuple[Path, Path]:
    backend = fixture / "backend"
    selected = backend / ("lm-atelier" if kind == "upload" else "output")
    selected.mkdir(parents=True)
    return backend, selected


def _adapter(backend: Path, selected: Path, kind: str) -> ComfyUIAdapter:
    return ComfyUIAdapter(
        "http://comfy.test",
        managed_temp_root=backend if kind == "upload" else None,
        managed_output_root=selected if kind == "output" else None,
        stale_output_seconds=3600,
    )


def _age(path: Path) -> None:
    aged = time.time() - 7200
    os.utime(path, (aged, aged))


def _link(link: Path, target: Path, fixture: Path) -> None:
    assert link.absolute().is_relative_to(fixture.absolute())
    assert target.resolve().is_relative_to(fixture.resolve())
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
    else:
        link.symlink_to(target, target_is_directory=True)


def _unlink(link: Path, fixture: Path) -> None:
    assert link.absolute().is_relative_to(fixture.absolute())
    try:
        linked = filesystem_links.is_link_or_reparse(link, missing="raise", unreadable="raise")
    except FileNotFoundError:
        return
    assert linked
    if os.name == "nt":
        link.rmdir()
    else:
        link.unlink()


@pytest.mark.parametrize("kind", ["upload", "output"])
async def test_stale_cleanup_reclaims_old_nested_files_and_keeps_fresh(
    tmp_path: Path, kind: str
) -> None:
    backend, selected = _layout(tmp_path, kind)
    old_directory = selected / "old" / "nested"
    old_directory.mkdir(parents=True)
    old = old_directory / "old.bin"
    old.write_bytes(b"old neutral fixture")
    _age(old)
    fresh = selected / "fresh.bin"
    fresh.write_bytes(b"fresh neutral fixture")
    backend_owned = backend / "backend-owned.bin"
    backend_owned.write_bytes(b"other backend work")
    _age(backend_owned)
    adapter = _adapter(backend, selected, kind)
    try:
        await adapter._sweep_stale_outputs()
        assert not old.exists()
        assert not (selected / "old").exists()
        assert fresh.read_bytes() == b"fresh neutral fixture"
        assert backend_owned.read_bytes() == b"other backend work"
        assert selected.is_dir()
    finally:
        await adapter.close()


@pytest.mark.parametrize("kind", ["upload", "output"])
@pytest.mark.parametrize("when", ["before_construction", "after_construction"])
@pytest.mark.parametrize("where", ["root", "ancestor", "child"])
async def test_stale_cleanup_refuses_linked_namespaces(
    tmp_path: Path, kind: str, when: str, where: str
) -> None:
    backend, selected = _layout(tmp_path, kind)
    outside = tmp_path / "outside"
    outside.mkdir()
    retained_directory = outside / selected.name if where == "ancestor" else outside
    retained_directory.mkdir(exist_ok=True)
    retained = retained_directory / "keep.bin"
    retained.write_bytes(b"outside neutral fixture")
    _age(retained)
    adapter = _adapter(backend, selected, kind) if when == "after_construction" else None
    if where == "ancestor":
        selected.rmdir()
        backend.rmdir()
        link = backend
    elif where == "root":
        selected.rmdir()
        link = selected
    else:
        link = selected / "redirect"
    _link(link, outside, tmp_path)
    try:
        if adapter is None:
            adapter = _adapter(backend, selected, kind)
        await adapter._sweep_stale_outputs()
        assert retained.is_file(), "cleanup followed a linked namespace"
        assert retained.read_bytes() == b"outside neutral fixture"
    finally:
        if adapter is not None:
            await adapter.close()
        _unlink(link, tmp_path)


@pytest.mark.parametrize("kind", ["upload", "output"])
@pytest.mark.parametrize("where", ["root", "child"])
async def test_stale_cleanup_retains_selected_directory_through_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, where: str
) -> None:
    backend, selected = _layout(tmp_path, kind)
    target = selected if where == "root" else selected / "nested"
    target.mkdir(exist_ok=True)
    old = target / "keep.bin"
    old.write_bytes(b"old selected fixture")
    _age(old)
    parked = tmp_path / "selected-original"
    outside = tmp_path / "outside"
    outside.mkdir()
    retained = outside / "keep.bin"
    retained.write_bytes(b"outside neutral fixture")
    _age(retained)
    adapter = _adapter(backend, selected, kind)
    attempted = False
    shifted = False

    def replace_selected(*, anchored: bool) -> None:
        nonlocal attempted, shifted
        if attempted:
            return
        attempted = True
        assert target.resolve().is_relative_to(tmp_path.resolve())
        assert parked.absolute().is_relative_to(tmp_path.absolute())
        try:
            target.rename(parked)
        except PermissionError:
            # Windows holds the directory against rename; POSIX allows a rename
            # but subsequent relative operations must stay on the original inode.
            assert anchored and os.name == "nt"
            return
        shifted = True
        _link(target, outside, tmp_path)

    original_is_dir = Path.is_dir

    def inspect(path: Path) -> bool:
        result = original_is_dir(path)
        if path == target and result:
            replace_selected(anchored=False)
        return result

    original_enter = filesystem_links.AnchoredDirectory.__enter__

    def enter(anchor: filesystem_links.AnchoredDirectory) -> filesystem_links.AnchoredDirectory:
        result = original_enter(anchor)
        if anchor.path == target:
            replace_selected(anchored=True)
        return result

    original_open = filesystem_links.open_child_directory

    def open_child(
        anchor: filesystem_links.AnchoredDirectory, name: str, *, create: bool = False
    ) -> filesystem_links.AnchoredDirectory:
        child = original_open(anchor, name, create=create)
        if child.path == target:
            replace_selected(anchored=True)
        return child

    # The pathname seam keeps the original rejected implementation measurable;
    # the held-directory seams exercise the replacement at the equivalent point.
    monkeypatch.setattr(Path, "is_dir", inspect)
    monkeypatch.setattr(filesystem_links.AnchoredDirectory, "__enter__", enter)
    monkeypatch.setattr(comfyui, "open_child_directory", open_child, raising=False)
    try:
        await adapter._sweep_stale_outputs()
        assert attempted, "the selected-directory transition was not exercised"
        removed = parked / old.name if shifted else old
        assert not removed.exists(), "cleanup did not reach the old file in the held directory"
        assert retained.is_file(), "cleanup deleted outside the selected directory"
        assert retained.read_bytes() == b"outside neutral fixture"
    finally:
        await adapter.close()
        if shifted:
            _unlink(target, tmp_path)
            assert parked.resolve().is_relative_to(tmp_path.resolve())
            assert target.absolute().is_relative_to(tmp_path.absolute())
            parked.rename(target)


@pytest.mark.parametrize("kind", ["upload", "output"])
async def test_stale_cleanup_reclaims_a_backlog_larger_than_default_listing_limit(
    tmp_path: Path, kind: str
) -> None:
    backend, selected = _layout(tmp_path, kind)
    # Existing installations can already exceed the listing helper's ordinary
    # 8192-entry limit. Refusing the whole list would leave that backlog forever.
    for index in range(8193):
        old = selected / f"old-{index}.bin"
        old.write_bytes(b"old")
        _age(old)
    fresh = selected / "fresh.bin"
    fresh.write_bytes(b"fresh")
    adapter = _adapter(backend, selected, kind)
    try:
        await adapter._sweep_stale_outputs()
        assert list(selected.iterdir()) == [fresh], "the accumulated backlog was not reclaimed"
    finally:
        await adapter.close()
