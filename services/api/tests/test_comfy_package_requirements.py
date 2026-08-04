"""Reading a commit-pinned package's own declarations from its staged tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_lm.comfy_package_requirements import (
    MAX_REQUIREMENTS_LINES,
    StagedRequirementsError,
    read_staged_requirements,
    select_requirements_manifest,
)


def _stage(root: Path, relative: str, text: str) -> None:
    target = root.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_a_package_that_declares_nothing_has_no_manifest() -> None:
    assert select_requirements_manifest(()) is None
    assert select_requirements_manifest(("node-1.0/pyproject.toml",)) is None


def test_the_packages_own_file_wins_over_anything_it_vendored() -> None:
    """A commit archive wraps the tree, so the package's own file is one level down."""
    selected = select_requirements_manifest(
        (
            "node-abc123/requirements.txt",
            "node-abc123/vendor/other/requirements.txt",
            "node-abc123/pyproject.toml",
        )
    )

    assert selected == "node-abc123/requirements.txt"


def test_two_files_at_the_same_depth_are_refused_rather_than_guessed() -> None:
    with pytest.raises(StagedRequirementsError) as refused:
        select_requirements_manifest(("one/requirements.txt", "two/requirements.txt"))

    assert refused.value.code == "ambiguous_requirements"


def test_reads_the_lines_without_interpreting_them(tmp_path: Path) -> None:
    """Comments and options survive to the planner, which is what judges them."""
    _stage(
        tmp_path,
        "node/requirements.txt",
        "# pinned for CUDA\nnumpy>=1.26  # sampler\n\n-r extra.txt\n",
    )

    assert read_staged_requirements(tmp_path, "node/requirements.txt") == (
        "# pinned for CUDA",
        "numpy>=1.26  # sampler",
        "",
        "-r extra.txt",
    )


def test_refuses_a_file_that_is_not_valid_utf8(tmp_path: Path) -> None:
    target = tmp_path / "node" / "requirements.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"numpy>=1\xff\xfe\n")

    with pytest.raises(StagedRequirementsError) as refused:
        read_staged_requirements(tmp_path, "node/requirements.txt")

    assert refused.value.code == "unreadable_requirements"


def test_refuses_a_file_with_too_many_lines(tmp_path: Path) -> None:
    _stage(
        tmp_path,
        "node/requirements.txt",
        "\n".join(f"package-{index}==1" for index in range(MAX_REQUIREMENTS_LINES + 1)),
    )

    with pytest.raises(StagedRequirementsError) as refused:
        read_staged_requirements(tmp_path, "node/requirements.txt")

    assert refused.value.code == "too_many_requirements"


@pytest.mark.parametrize("manifest", ["../outside.txt", "node/../../outside.txt"])
def test_refuses_a_path_that_leaves_the_staged_tree(tmp_path: Path, manifest: str) -> None:
    _stage(tmp_path, "node/requirements.txt", "numpy==1.26.0")
    (tmp_path.parent / "outside.txt").write_text("evil==1", encoding="utf-8")

    with pytest.raises(StagedRequirementsError) as refused:
        read_staged_requirements(tmp_path, manifest)

    assert refused.value.code == "invalid_requirements_path"


def test_refuses_a_manifest_that_is_not_a_file(tmp_path: Path) -> None:
    (tmp_path / "node" / "requirements.txt").mkdir(parents=True)

    with pytest.raises(StagedRequirementsError) as refused:
        read_staged_requirements(tmp_path, "node/requirements.txt")

    assert refused.value.code == "invalid_requirements_path"


def test_a_missing_file_is_reported_as_unreadable(tmp_path: Path) -> None:
    with pytest.raises(StagedRequirementsError) as refused:
        read_staged_requirements(tmp_path, "node/requirements.txt")

    assert refused.value.code == "unreadable_requirements"
