"""Which tree this run is measuring.

conftest refuses to collect anything unless `local_lm` resolves to THIS
checkout's package directory exactly, because a run that imports another tree
passes while testing code that is not in the commit.

The rule is exact identity rather than containment, and that distinction is the
whole point: this repository keeps its worktrees BELOW the main checkout, so a
nested worktree is "inside" the main root and a containment test accepts it.
"""

from __future__ import annotations

from pathlib import Path

import local_lm


def _package(root: Path) -> Path:
    return root / "services" / "api" / "local_lm"


def test_this_run_is_measuring_its_own_tree() -> None:
    """The live assertion, stated as a test rather than only as a refusal.

    conftest raises before collection when this is false, so in normal use it
    never fails. It earns its place if that guard is ever removed: this is what
    still notices.
    """

    repository_root = Path(__file__).resolve().parents[3]
    imported = Path(local_lm.__file__).resolve().parent

    assert imported == _package(repository_root).resolve(), (
        "local_lm imports from a different tree than these tests came from.\n"
        f"  imported from : {imported}\n"
        f"  expected      : {_package(repository_root).resolve()}"
    )


def test_a_nested_worktree_is_not_this_tree(tmp_path: Path) -> None:
    """Containment was the wrong rule, and a nested worktree is why.

    Linked worktrees can live under the main checkout, so their package IS
    relative to the main root. Run the
    main checkout's tests with PYTHONPATH pointed at one of them and a
    containment check happily agrees while a different tree is measured.
    """

    root = tmp_path / "lm-atelier"
    main_package = _package(root)
    main_package.mkdir(parents=True)
    nested = _package(root / "temp" / "worktrees" / "candidate")
    nested.mkdir(parents=True)

    assert nested.is_relative_to(root), "the trap requires the worktree to be nested"
    assert nested != main_package


def test_a_sibling_checkout_is_not_this_tree(tmp_path: Path) -> None:
    """The trap a string prefix falls into.

    `lm-atelier-other` starts with `lm-atelier`, so a startswith test would
    call a different checkout "inside this one".
    """

    root = tmp_path / "lm-atelier"
    main_package = _package(root)
    main_package.mkdir(parents=True)
    sibling = _package(tmp_path / "lm-atelier-other")
    sibling.mkdir(parents=True)

    assert str(sibling).startswith(str(root)), "the trap needs a shared prefix"
    assert sibling != main_package


def test_the_expected_package_is_the_one_this_checkout_holds(tmp_path: Path) -> None:
    """Exact identity accepts the right tree, not merely rejects wrong ones.

    A rule that refused everything would pass both tests above and be useless.
    """

    root = tmp_path / "lm-atelier"
    main_package = _package(root)
    main_package.mkdir(parents=True)

    assert main_package.resolve() == _package(root).resolve()
