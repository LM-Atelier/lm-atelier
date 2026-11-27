"""List one part of the API test suite, so CI can run the suite in two jobs at once.

The API tests are most of what Ubuntu verification spends its time on, so CI
splits them: Ubuntu compatibility runs one part and a job beside it runs the
other. A module that fell into neither part would never run, and nothing would
fail to say so. So this lists the modules the way pytest collects the whole
suite, assigns each to a part by its path alone, and reports what the merge
gate needs to show that the parts ran all of them, each once.

    python scripts/api-test-part.py 1/2

prints the modules of part 1 of 2, one path per line, relative to the
repository root. With GITHUB_OUTPUT set it also writes three step outputs:
api_tests_total, the number of modules in the whole suite; api_tests_selected,
the number in this part; and api_tests_digest, the SHA-256 of the whole suite's
module list, each path relative to the tests directory and followed by a line
feed, in sorted order. Every job that lists a part of one tree reports the same
total and digest, and the parts' counts add up to the total.

The list mirrors pytest's collection rather than asking pytest for it, because
collecting means importing every module. A mirror is only right while pytest
collects as it is mirrored here, so the script refuses anything that would
change that: the collection settings, a configuration file that pytest would
read ahead of the API project's own, or a test module in a subdirectory. Only
the standard library is used, so the listing does not depend on what the
environment has installed.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
API_PROJECT = Path("services/api")
TESTS = API_PROJECT / "tests"
# pytest's default test module patterns. Matching is case-sensitive, as it is on
# Linux, where the parts run.
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")
# The settings that decide which files, classes and functions pytest collects.
# Each is left at pytest's default, which is what the patterns above mirror.
COLLECTION_SETTINGS = ("python_files", "python_classes", "python_functions")
TESTPATHS = ["tests"]
# The files pytest takes its configuration from, in the order it tries them in
# each directory it searches. It searches the tests directory before the API
# project's, so any of these beside the tests, or one tried before
# pyproject.toml beside the project's own, would be read instead of it.
CONFIGURATION_FILES = (
    "pytest.toml",
    ".pytest.toml",
    "pytest.ini",
    ".pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
)
PART = re.compile(r"([1-9][0-9]{0,2})/([1-9][0-9]{0,2})")


class Refusal(Exception):
    """A reason this script cannot list a part the way pytest would collect it."""


def parse_part(value: str) -> tuple[int, int]:
    """Return the part and the number of parts that `value` names, as in 1/2."""

    match = PART.fullmatch(value)
    if match is None or int(match.group(1)) > int(match.group(2)):
        raise Refusal(
            f"{value!r} does not name a part; give it as PART/PARTS, for example 1/2, "
            "with PART from 1 to PARTS"
        )
    return int(match.group(1)), int(match.group(2))


def part_of(module: str, parts: int) -> int:
    """Return the part, counting from 1, that a module belongs to.

    `module` is the module's path relative to the tests directory, as POSIX
    text. The part depends on that path alone, so a module keeps its part
    whatever else is added or removed, and every job that lists a part agrees
    on where each module belongs.
    """

    return int(hashlib.sha256(module.encode("utf-8")).hexdigest(), 16) % parts + 1


def module_list_digest(modules: Sequence[str]) -> str:
    """Return the SHA-256 of a module list, each path followed by a line feed, sorted."""

    listed = "".join(f"{module}\n" for module in sorted(modules))
    return hashlib.sha256(listed.encode("utf-8")).hexdigest()


def is_test_module(name: str) -> bool:
    """Return whether pytest collects a file of this name as a test module."""

    return any(fnmatch.fnmatchcase(name, pattern) for pattern in TEST_FILE_PATTERNS)


def collection_problems(root: Path, environment: Mapping[str, str]) -> list[str]:
    """Return every reason pytest might collect the suite other than as it is mirrored."""

    problems: list[str] = []
    ahead = CONFIGURATION_FILES[: CONFIGURATION_FILES.index("pyproject.toml")]
    for directory, names in ((root / TESTS, CONFIGURATION_FILES), (root / API_PROJECT, ahead)):
        problems.extend(
            f"{(directory / name).relative_to(root).as_posix()} can hold configuration "
            f"pytest would read instead of {API_PROJECT.as_posix()}/pyproject.toml"
            for name in names
            if (directory / name).exists()
        )
    pyproject = f"{API_PROJECT.as_posix()}/pyproject.toml"
    try:
        document = tomllib.loads(
            (root / API_PROJECT / "pyproject.toml").read_text(encoding="utf-8")
        )
    except (OSError, tomllib.TOMLDecodeError) as error:
        return problems + [f"{pyproject} cannot be read: {error}"]
    tool = document.get("tool")
    pytest_table = tool.get("pytest") if isinstance(tool, dict) else None
    if not isinstance(pytest_table, dict) or not pytest_table:
        # Without one, pytest keeps looking in the directories above for its
        # configuration, and whatever it finds there is not checked here.
        return problems + [f"{pyproject} has no pytest configuration"]
    # pytest reads native settings from [tool.pytest] and string ones from
    # [tool.pytest.ini_options], so a setting is checked in either.
    ini_options = pytest_table.get("ini_options")
    tables: list[tuple[str, Mapping[str, Any]]] = [
        (
            "[tool.pytest]",
            {key: value for key, value in pytest_table.items() if key != "ini_options"},
        ),
        ("[tool.pytest.ini_options]", ini_options if isinstance(ini_options, dict) else {}),
    ]
    options = [environment.get("PYTEST_ADDOPTS", "")]
    for table, settings in tables:
        problems.extend(
            f"{pyproject} {table} sets {setting}, which changes what pytest collects"
            for setting in COLLECTION_SETTINGS
            if setting in settings
        )
        if "testpaths" in settings and settings["testpaths"] != TESTPATHS:
            problems.append(
                f"{pyproject} {table} sets testpaths to {settings['testpaths']!r} "
                f"rather than {TESTPATHS!r}"
            )
        addopts = settings.get("addopts", "")
        options.append(" ".join(map(str, addopts)) if isinstance(addopts, list) else str(addopts))
    problems.extend(
        f"pytest's options override {setting}, which changes what pytest collects"
        for setting in (*COLLECTION_SETTINGS, "testpaths")
        if any(setting in option for option in options)
    )
    return problems


def suite_modules(root: Path) -> list[str]:
    """Return every API test module, relative to the tests directory, in sorted order.

    These are the files directly in the tests directory whose names pytest
    collects. A test module in a subdirectory is refused rather than listed:
    pytest would collect it from the whole suite, and no part would run it.
    """

    tests = root / TESTS
    modules: list[str] = []
    nested: list[str] = []
    for entry in tests.iterdir():
        if entry.is_dir():
            for directory, _subdirectories, files in os.walk(entry):
                nested.extend(
                    (Path(directory) / name).relative_to(root).as_posix()
                    for name in files
                    if is_test_module(name)
                )
        elif entry.is_file() and is_test_module(entry.name):
            modules.append(entry.name)
    if nested:
        raise Refusal(
            "these test modules are in a subdirectory of the API tests, so no part would "
            "run them: " + ", ".join(sorted(nested))
        )
    if not modules:
        raise Refusal(f"{TESTS.as_posix()} holds no test modules")
    return sorted(modules)


def main(arguments: Sequence[str], environment: Mapping[str, str]) -> int:
    if len(arguments) != 1:
        print("usage: api-test-part.py PART/PARTS, for example 1/2", file=sys.stderr)
        return 2
    try:
        part, parts = parse_part(arguments[0])
        problems = collection_problems(ROOT, environment)
        if problems:
            raise Refusal("; ".join(problems))
        modules = suite_modules(ROOT)
    except Refusal as refusal:
        print(f"Cannot list API test part {arguments[0]!r}: {refusal}.", file=sys.stderr)
        return 1
    selected = [module for module in modules if part_of(module, parts) == part]
    if not selected:
        print(
            f"API test part {part}/{parts} holds none of the suite's {len(modules)} modules, "
            "so it would pass without running a test.",
            file=sys.stderr,
        )
        return 1
    digest = module_list_digest(modules)
    output = environment.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8", newline="\n") as outputs:
            outputs.write(f"api_tests_total={len(modules)}\n")
            outputs.write(f"api_tests_selected={len(selected)}\n")
            outputs.write(f"api_tests_digest={digest}\n")
    print(
        f"API test part {part}/{parts}: {len(selected)} of the suite's {len(modules)} modules, "
        f"whose list has the digest {digest}.",
        file=sys.stderr,
    )
    for module in selected:
        print(f"{TESTS.as_posix()}/{module}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], os.environ))
