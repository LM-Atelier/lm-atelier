"""A staged package's requirements come from its own tree, never through a link."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_lm import comfy_package_requirements
from local_lm import workflow_package_preparation as composition
from local_lm.comfy_package_requirements import (
    StagedRequirementsError,
    staged_requirements_manifests,
)
from local_lm.comfy_registry import ComfyNodeResolution, ComfyRegistryResolution
from local_lm.workflow_package_preparation import PreparationContext, prepare_workflow_package


def _make_link_dir(link: Path, target: Path) -> bool:
    """Point `link` at `target`, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def _outside(tmp_path: Path) -> Path:
    """Make a folder outside every package, holding a requirements file of its own."""

    outside = tmp_path / "shared"
    outside.mkdir()
    (outside / "requirements.txt").write_text("unrelated-dependency\n", encoding="utf-8")
    return outside


def test_a_linked_folder_does_not_lend_the_package_its_requirements(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "requirements.txt").write_text("own-dependency\n", encoding="utf-8")
    if not _make_link_dir(package / "models", _outside(tmp_path)):
        pytest.skip("this host refuses to create a directory link")

    assert staged_requirements_manifests(package) == ("requirements.txt",)


def test_a_package_folder_that_is_itself_a_link_is_refused_rather_than_read_through(
    tmp_path: Path,
) -> None:
    """Refused, not answered with nothing: a package that declares nothing and
    a package whose tree could not be read are different answers."""

    package = tmp_path / "package"
    if not _make_link_dir(package, _outside(tmp_path)):
        pytest.skip("this host refuses to create a directory link")

    with pytest.raises(StagedRequirementsError) as refused:
        staged_requirements_manifests(package)

    assert refused.value.code == "unreadable_requirements"


def test_a_folder_wider_than_one_listing_still_yields_its_manifest(tmp_path: Path) -> None:
    """The scan's own entry bound is the limit, not the width of any one folder."""

    package = tmp_path / "package"
    wide = package / "data"
    wide.mkdir(parents=True)
    for index in range(8200):
        (wide / f"{index}.json").touch()
    (package / "requirements.txt").write_text("own-dependency\n", encoding="utf-8")

    assert staged_requirements_manifests(package) == ("requirements.txt",)


def test_the_scan_still_stops_at_its_entry_bound_instead_of_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    (package / "a" / "b").mkdir(parents=True)
    (package / "a" / "b" / "requirements.txt").write_text("deep\n", encoding="utf-8")
    monkeypatch.setattr(comfy_package_requirements, "MAX_STAGED_MANIFEST_SCAN", 2)

    assert staged_requirements_manifests(package) == ()


def test_a_large_folder_does_not_use_up_the_bound_before_the_root_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The folder is listed first and fills the bound with its own files."""

    package = tmp_path / "package"
    data = package / "a-data"
    data.mkdir(parents=True)
    for index in range(3):
        (data / f"{index}.txt").write_text("neutral data", encoding="utf-8")
    (package / "requirements.txt").write_text("own-dependency\n", encoding="utf-8")
    monkeypatch.setattr(comfy_package_requirements, "MAX_STAGED_MANIFEST_SCAN", 3)

    assert staged_requirements_manifests(package) == ("requirements.txt",)


def test_a_deep_sibling_does_not_use_up_the_bound_before_a_shallower_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    deep = package / "a-data" / "nested"
    deep.mkdir(parents=True)
    for index in range(3):
        (deep / f"{index}.txt").write_text("neutral data", encoding="utf-8")
    (package / "node").mkdir()
    (package / "node" / "requirements.txt").write_text("own-dependency\n", encoding="utf-8")
    monkeypatch.setattr(comfy_package_requirements, "MAX_STAGED_MANIFEST_SCAN", 4)

    assert staged_requirements_manifests(package) == ("node/requirements.txt",)


class _Registry:
    def __init__(self, resolution: ComfyNodeResolution) -> None:
        self._resolution = resolution

    async def resolve(self, _requirements: Any) -> ComfyRegistryResolution:
        return ComfyRegistryResolution(packages=(self._resolution,))


class _Sessions:
    """Answers the one lookup a renewal makes: the install it is refreshing."""

    def __init__(self, install: object) -> None:
        self._install = install

    def __call__(self) -> _Sessions:
        return self

    def __enter__(self) -> _Sessions:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, _model: object, _identity: object) -> object:
        return self._install


async def _probe(_python: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    return {"sys_platform": "win32"}, ("py3-none-any",)


async def test_a_commit_pinned_renewal_is_not_refused_over_a_file_behind_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A renewal reads the package's declarations from the tree already on disk.

    A package that declares nothing of its own renews with no dependencies. A
    folder linked into its tree does not change that: whatever requirements
    file the link leads to belongs to something else, and taking it as the
    package's would refuse the renewal over a file the package never staged.
    """

    folder = tmp_path / "custom_nodes" / "example-pack"
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text("", encoding="utf-8")
    if not _make_link_dir(folder / "models", _outside(tmp_path)):
        pytest.skip("this host refuses to create a directory link")
    install = SimpleNamespace(
        installed_path=folder.name,
        review_json={},
        registry_record_id="github-commit:" + "b" * 60,
        download_url="https://codeload.github.com/example/example-pack/zip/" + "a" * 40,
    )
    planned: list[tuple[str, ...]] = []

    async def drive(resolution: Any, **_kwargs: Any) -> Any:
        planned.append(resolution.pip_dependencies)
        return SimpleNamespace(closure="closure")

    async def renew(_session: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(install_id="install_1")

    async def unexpected_stage(**_kwargs: Any) -> Any:
        raise AssertionError("renewal must not fetch the node code again")

    monkeypatch.setattr(composition, "drive_comfy_registry_wheel_closure", drive)
    monkeypatch.setattr(composition, "renew_comfy_registry_install_environment", renew)
    monkeypatch.setattr(composition, "stage_comfy_registry_install_archive", unexpected_stage)
    revision = "a" * 40

    # None of these doubles is the type it stands in for, so they travel as one
    # mapping rather than as arguments checked against those types.
    collaborators: dict[str, Any] = {
        "session_factory": _Sessions(install),
        "registry_client": _Registry(
            ComfyNodeResolution(
                package_id="example-pack",
                declared_version=revision,
                node_types=("ExampleNode",),
                install_kind="git_commit",
                repository_url="https://github.com/example/example-pack.git",
            )
        ),
        "project_client": SimpleNamespace(fetch=None),
        "metadata_client": SimpleNamespace(fetch=None),
        "archive_downloader": SimpleNamespace(),
        "wheel_downloader": SimpleNamespace(),
    }

    await prepare_workflow_package(
        package_id="example-pack",
        node_types=("ExampleNode",),
        version=revision,
        context=PreparationContext(
            python_executable=tmp_path / "python.exe",
            custom_node_root=folder.parent,
            state_root=tmp_path / "registry",
        ),
        media_worker_stopped=True,
        interpreter_probe=_probe,
        renew_install_id="install_1",
        **collaborators,
    )

    assert planned == [()]
