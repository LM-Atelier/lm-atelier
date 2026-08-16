"""The registry archive ceilings are policy, and policy that no check pins is not policy.

`comfy_registry_archives` bounds an archive that arrived from the ComfyUI
registry - somebody else's bytes. Nine constants carry those bounds, and eight
of them could be relaxed a thousandfold with every test that touches the module
still green. Only `MAX_ARCHIVE_ENTRIES` failed anything. So the numbers were
correct and undefended: nothing would notice a later edit that made a zip bomb
or a path-length attack fit comfortably inside the limit.

This is deliberately a bound and not a pin. Asserting equality would be a
change-detector that fires on every legitimate tightening and teaches people to
update the number without thinking. Asserting `<=` lets the ceiling be lowered
freely, which is always safe, and refuses the direction that costs something.
The one exception is the compression-ratio guard, where safety runs the other
way: a larger ratio admits more expansion, so it is bounded above too, and a
floor keeps a well-meaning tightening from refusing ordinary archives.

Sibling precedent: this is the same defect class the size ceiling in the
repository hygiene guard had, and the six resource ceilings in
`comfy_registry_source_artifacts`. Both were closed by bounding rather than
pinning.
"""

from __future__ import annotations

import pytest

from local_lm import comfy_registry_archives as archives

MIB = 1024 * 1024


@pytest.mark.parametrize(
    "name,ceiling",
    [
        ("MAX_ARCHIVE_BYTES", 64 * MIB),
        ("MAX_ARCHIVE_ENTRIES", 4_096),
        ("MAX_ARCHIVE_EXPANDED_BYTES", 256 * MIB),
        ("MAX_ARCHIVE_FILE_BYTES", 64 * MIB),
        ("MAX_ARCHIVE_PATH_CHARACTERS", 1_024),
        ("MAX_ARCHIVE_COMPONENT_CHARACTERS", 255),
        ("MAX_RUNTIME_FILE_COUNT", 256),
        ("MAX_RUNTIME_FILE_BYTES", 1 * MIB),
        ("MAX_RUNTIME_FILES_BYTES", 8 * MIB),
    ],
)
def test_no_archive_ceiling_may_be_raised_without_this_failing(name: str, ceiling: int) -> None:
    """Lowering any of these is free; raising one has to be argued for here first."""

    actual = getattr(archives, name)
    assert isinstance(actual, int), f"{name} is {type(actual).__name__}, not an int"
    assert actual > 0, f"{name} is {actual}, which disables the bound it exists to impose"
    assert actual <= ceiling, (
        f"{name} is {actual}, above the reviewed ceiling of {ceiling}. Raising a bound on "
        "archive bytes that arrived from the registry widens what a hostile package can "
        "spend. If the new value is genuinely wanted, change it here in the same commit so "
        "the decision is visible in review rather than inferred from a diff."
    )


def test_the_component_bound_stays_inside_the_path_bound() -> None:
    """A single component may not be allowed to exceed a whole path.

    Not merely tidiness: `_safe_member` checks both, and a component ceiling above
    the path ceiling would make the component check unreachable, which is the
    quiet way a guard stops guarding.
    """

    assert archives.MAX_ARCHIVE_COMPONENT_CHARACTERS <= archives.MAX_ARCHIVE_PATH_CHARACTERS


def test_a_single_file_may_not_exceed_the_whole_archive_expansion() -> None:
    """One member cannot be permitted more expanded bytes than the archive total."""

    assert archives.MAX_ARCHIVE_FILE_BYTES <= archives.MAX_ARCHIVE_EXPANDED_BYTES
    assert archives.MAX_RUNTIME_FILE_BYTES <= archives.MAX_RUNTIME_FILES_BYTES


def test_the_declared_archive_size_cannot_exceed_what_expansion_allows() -> None:
    """A compressed archive larger than the expansion ceiling could never be admitted.

    If `MAX_ARCHIVE_BYTES` ever rose above `MAX_ARCHIVE_EXPANDED_BYTES` the two
    would disagree about what is acceptable, and the looser one would be dead
    configuration that reads as protection.
    """

    assert archives.MAX_ARCHIVE_BYTES <= archives.MAX_ARCHIVE_EXPANDED_BYTES
