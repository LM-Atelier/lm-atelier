"""The wheel-environment ceilings bound somebody else's wheels, and six are undefended.

Same defect class as the repository hygiene size ceiling, the six resource
ceilings in `comfy_registry_source_artifacts`, and the archive ceilings beside
this one: the values are correct and no check would notice them changing. Six of
the eight bounds here survive being multiplied by a thousand with every test
that imports the module still green. `MAX_REGISTRY_WHEEL_EXPANSION_RATIO` and
`MAX_REGISTRY_WHEEL_UNCHECKED_ENTRY_BYTES` already fail something, so they are
covered here for symmetry rather than because they were missing.

Bounds rather than pins, so a ceiling can always be tightened without editing
this file, and can only be widened by arguing for it here in the same commit.

The ratio is the one that runs the other way. Every other constant is safer
smaller; the ratio admits *more* expansion as it grows, so it is bounded above,
and it also carries a floor, because a well-meant tightening to a value near 1
would refuse ordinary well-compressed wheels and look like corruption rather
than policy.
"""

from __future__ import annotations

import pytest

from local_lm import comfy_registry_wheel_environments as wheels

MIB = 1024 * 1024
GIB = 1024 * MIB


@pytest.mark.parametrize(
    "name,ceiling",
    [
        ("MAX_REGISTRY_WHEEL_ENVIRONMENT_FILES", 100_000),
        ("MAX_REGISTRY_WHEEL_ENVIRONMENT_BYTES", 32 * GIB),
        ("MAX_REGISTRY_WHEEL_METADATA_FILE_BYTES", 1 * MIB),
        ("MAX_REGISTRY_WHEEL_ENVIRONMENT_MANIFEST_BYTES", 16 * MIB),
        ("MAX_REGISTRY_WHEEL_ARCHIVE_PATH_CHARACTERS", 1_000),
        ("MAX_REGISTRY_WHEEL_ARCHIVE_COMPONENT_CHARACTERS", 255),
        ("MAX_REGISTRY_WHEEL_UNCHECKED_ENTRY_BYTES", 64 * MIB),
    ],
)
def test_no_wheel_ceiling_may_be_raised_without_this_failing(name: str, ceiling: int) -> None:
    """Tightening any of these is free; widening one has to be argued for here."""

    actual = getattr(wheels, name)
    assert isinstance(actual, int), f"{name} is {type(actual).__name__}, not an int"
    assert actual > 0, f"{name} is {actual}, which disables the bound it exists to impose"
    assert actual <= ceiling, (
        f"{name} is {actual}, above the reviewed ceiling of {ceiling}. These bound wheels "
        "fetched from an external index and expanded onto this machine. If the larger value "
        "is genuinely wanted, raise it here in the same commit so the decision is read in "
        "review rather than inferred from a diff."
    )


def test_the_expansion_ratio_is_bounded_in_both_directions() -> None:
    """The one constant that is more dangerous larger, and useless too small.

    It gates `entry.file_size > compressed * RATIO`, so raising it admits a
    fatter decompression bomb. Lowering it far enough starts refusing ordinary
    wheels, which reads to a user as a corrupt download rather than as policy,
    so the floor is part of the contract too.
    """

    assert wheels.MAX_REGISTRY_WHEEL_EXPANSION_RATIO <= 200
    assert wheels.MAX_REGISTRY_WHEEL_EXPANSION_RATIO >= 20


def test_the_component_bound_stays_inside_the_path_bound() -> None:
    """A component ceiling above the path ceiling makes the component check unreachable."""

    assert (
        wheels.MAX_REGISTRY_WHEEL_ARCHIVE_COMPONENT_CHARACTERS
        <= wheels.MAX_REGISTRY_WHEEL_ARCHIVE_PATH_CHARACTERS
    )


def test_no_single_member_may_outgrow_the_environment_it_lands_in() -> None:
    """Per-file budgets have to stay inside the whole-environment budget.

    A metadata file, a manifest, or one unchecked entry allowed more bytes than
    the environment total would be a limit that can never bind - dead
    configuration that still reads as protection.
    """

    total = wheels.MAX_REGISTRY_WHEEL_ENVIRONMENT_BYTES
    assert total >= wheels.MAX_REGISTRY_WHEEL_METADATA_FILE_BYTES
    assert total >= wheels.MAX_REGISTRY_WHEEL_ENVIRONMENT_MANIFEST_BYTES
    assert total >= wheels.MAX_REGISTRY_WHEEL_UNCHECKED_ENTRY_BYTES
