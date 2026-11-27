"""The API tests split into parts that between them run every module exactly once.

CI runs the API suite in two jobs at once. scripts/api-test-part.py lists a part
the way pytest collects the whole suite, verify.sh runs the part it names, and
the merge gate compares what each part reports. These tests hold the listing to
pytest's own collection, to a part that depends on a module's path alone, and
to refusing whatever it cannot list the way pytest would collect it.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import runpy
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import yaml
from _pytest.python import path_matches_patterns

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "api-test-part.py"
PYPROJECT = Path("services/api/pyproject.toml")
TESTS = Path("services/api/tests")
# Four modules that split one and three: test_a.py falls in the first part of
# two, and the others in the second.
SUITE = ("test_a.py", "test_b.py", "test_c.py", "one_test.py")


class Listing(NamedTuple):
    """What one run of the listing did: its exit status, output, outputs and message."""

    status: int
    modules: list[str]
    outputs: dict[str, str]
    message: str


def _suite(tree: Path, modules: tuple[str, ...] = SUITE) -> Path:
    """Lay out what the listing reads in `tree`, with `modules` as the suite.

    Beside the modules sit the kinds of file the real tests directory holds and
    pytest does not collect: a conftest, a helper module and a fixtures folder.
    """

    (tree / "scripts").mkdir(parents=True)
    shutil.copyfile(SCRIPT, tree / "scripts" / SCRIPT.name)
    tests = tree / TESTS
    (tests / "fixtures").mkdir(parents=True)
    shutil.copyfile(ROOT / PYPROJECT, tree / PYPROJECT)
    (tests / "conftest.py").write_text("", encoding="utf-8")
    (tests / "run_waits.py").write_text("", encoding="utf-8")
    (tests / "fixtures" / "corpus.json").write_text("{}", encoding="utf-8")
    (tests / "fixtures" / "sample_fixture.py").write_text("", encoding="utf-8")
    for module in modules:
        (tests / module).parent.mkdir(parents=True, exist_ok=True)
        (tests / module).write_text("def test_nothing() -> None:\n    pass\n", encoding="utf-8")
    return tree


def _list_part(
    tree: Path, *arguments: str, environment: dict[str, str] | None = None, outputs: bool = True
) -> Listing:
    """Run the listing copied into `tree`, as CI runs it, and read what it wrote."""

    written = tree / "github-output"
    written.write_text("", encoding="utf-8")
    variables = {**os.environ, "PYTEST_ADDOPTS": "", **(environment or {})}
    variables.pop("GITHUB_OUTPUT", None)
    if outputs:
        variables["GITHUB_OUTPUT"] = str(written)
    finished = subprocess.run(
        [sys.executable, str(tree / "scripts" / SCRIPT.name), *arguments],
        cwd=tree,
        env=variables,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    lines = written.read_text(encoding="utf-8").splitlines()
    return Listing(
        finished.returncode,
        finished.stdout.splitlines(),
        {name: value for name, _, value in (line.partition("=") for line in lines)},
        finished.stderr,
    )


def test_every_api_test_module_runs_in_exactly_one_part(pytestconfig: pytest.Config) -> None:
    """The real suite's modules split into two parts that share none and miss none.

    The list is checked against pytest's own settings and file matching, read
    from the run this test is part of, so a listing that drifted from how pytest
    collects the suite fails here rather than leaving a module in neither part.
    """

    namespace = runpy.run_path(str(SCRIPT))
    modules = namespace["suite_modules"](ROOT)
    part_of = namespace["part_of"]
    patterns = pytestconfig.getini("python_files")
    collected = sorted(
        path.name
        for path in (ROOT / TESTS).iterdir()
        if path.is_file() and path.suffix == ".py" and path_matches_patterns(path, patterns)
    )
    parts = {part: [module for module in modules if part_of(module, 2) == part] for part in (1, 2)}

    assert patterns == ["test_*.py", "*_test.py"]
    assert pytestconfig.getini("python_classes") == ["Test"]
    assert pytestconfig.getini("python_functions") == ["test"]
    assert namespace["collection_problems"](ROOT, {}) == []
    assert modules == collected
    assert Path(__file__).name in modules
    assert parts[1] and parts[2]
    assert not set(parts[1]) & set(parts[2])
    assert sorted(parts[1] + parts[2]) == modules


def test_a_modules_part_depends_on_its_path_alone(tmp_path: Path) -> None:
    """A module keeps its part whatever else is in the suite, and in any tree.

    The parts are pinned for a few names, so a change to the hash, to the text
    it hashes or to how parts are counted moves them and fails here, rather than
    reshuffling the suite between the two jobs unseen. The text is the path
    relative to the tests directory: from the repository root the same two
    modules would fall the other way.
    """

    part_of = runpy.run_path(str(SCRIPT))["part_of"]
    names = ("test_packaging.py", "test_verification_test_typing.py", "one_test.py", "two_test.py")
    smaller = _list_part(
        _suite(tmp_path / "smaller", ("test_a.py", "test_b.py", "test_c.py")), "2/2"
    )
    larger = _list_part(
        _suite(tmp_path / "larger", ("test_a.py", "test_b.py", "test_c.py", "test_d.py")), "2/2"
    )

    assert [part_of(name, 2) for name in names] == [1, 2, 2, 1]
    assert [part_of(f"{TESTS.as_posix()}/{name}", 2) for name in names[:2]] == [2, 1]
    assert part_of("test_merge_queue_configuration.py", 3) == 3
    assert smaller.modules == [f"{TESTS.as_posix()}/test_b.py", f"{TESTS.as_posix()}/test_c.py"]
    assert larger.modules == [*smaller.modules, f"{TESTS.as_posix()}/test_d.py"]


def test_each_part_lists_its_modules_and_reports_the_whole_suite(tmp_path: Path) -> None:
    """Both parts report one suite, by length and digest, and counts that add up to it.

    The conftest, the helper module and the fixtures folder beside the modules
    are not test modules, so neither part lists them.
    """

    tree = _suite(tmp_path)
    digest = hashlib.sha256(b"one_test.py\ntest_a.py\ntest_b.py\ntest_c.py\n").hexdigest()
    summary = f"API test part 1/2: 1 of the suite's 4 modules, whose list has the digest {digest}."

    first = _list_part(tree, "1/2")
    second = _list_part(tree, "2/2")
    unrecorded = _list_part(tree, "2/2", outputs=False)

    assert (first.status, second.status, unrecorded.status) == (0, 0, 0), first.message
    assert first.modules == [f"{TESTS.as_posix()}/test_a.py"]
    assert second.modules == [
        f"{TESTS.as_posix()}/one_test.py",
        f"{TESTS.as_posix()}/test_b.py",
        f"{TESTS.as_posix()}/test_c.py",
    ]
    assert first.outputs == {
        "api_tests_total": "4",
        "api_tests_selected": "1",
        "api_tests_digest": digest,
    }
    assert second.outputs == {
        "api_tests_total": "4",
        "api_tests_selected": "3",
        "api_tests_digest": digest,
    }
    assert unrecorded.modules == second.modules
    assert unrecorded.outputs == {}
    assert summary in first.message


def test_a_part_with_no_modules_fails_and_reports_nothing(tmp_path: Path) -> None:
    """A part that would run nothing fails rather than passing without a test."""

    empty = _list_part(_suite(tmp_path / "one", ("test_a.py",)), "2/2")
    nothing = _list_part(_suite(tmp_path / "none", ()), "1/2")

    assert (empty.status, empty.modules, empty.outputs) == (1, [], {})
    assert "API test part 2/2 holds none of the suite's 1 modules" in empty.message
    assert (nothing.status, nothing.modules, nothing.outputs) == (1, [], {})
    assert f"{TESTS.as_posix()} holds no test modules" in nothing.message


def _project_line(old: str, new: str) -> Callable[[Path], None]:
    """Return a change to the copied API project configuration, replacing one line."""

    def apply(tree: Path) -> None:
        path = tree / PYPROJECT
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1, old
        path.write_text(text.replace(old, new), encoding="utf-8")

    return apply


def _project(content: str) -> Callable[[Path], None]:
    """Return a change that replaces the copied API project configuration whole."""

    def apply(tree: Path) -> None:
        (tree / PYPROJECT).write_text(content, encoding="utf-8")

    return apply


def _beside(directory: Path, name: str) -> Callable[[Path], None]:
    """Return a change that adds an empty configuration file pytest would read."""

    def apply(tree: Path) -> None:
        (tree / directory / name).write_text("", encoding="utf-8")

    return apply


def _unchanged(tree: Path) -> None:
    """Leave the copied tree as it is."""


PYTEST_TABLE = "[tool.pytest.ini_options]\n"
COLLECTION_DRIFTS: list[tuple[str, Callable[[Path], None], dict[str, str], str]] = [
    (
        "module patterns",
        _project_line(PYTEST_TABLE, PYTEST_TABLE + 'python_files = "check_*.py"\n'),
        {},
        "[tool.pytest.ini_options] sets python_files",
    ),
    (
        "class patterns",
        _project_line(PYTEST_TABLE, PYTEST_TABLE + 'python_classes = "Check"\n'),
        {},
        "[tool.pytest.ini_options] sets python_classes",
    ),
    (
        "function patterns",
        _project_line(PYTEST_TABLE, PYTEST_TABLE + 'python_functions = "check"\n'),
        {},
        "[tool.pytest.ini_options] sets python_functions",
    ),
    (
        "search paths",
        _project_line('testpaths = ["tests"]\n', 'testpaths = ["tests", "more"]\n'),
        {},
        "sets testpaths to ['tests', 'more'] rather than ['tests']",
    ),
    (
        "native module patterns",
        _project('[tool.pytest]\npython_files = ["check_*.py"]\n'),
        {},
        "[tool.pytest] sets python_files",
    ),
    (
        "options that override the patterns",
        _project_line('addopts = "-q', 'addopts = "-o python_files=check_*.py -q'),
        {},
        "pytest's options override python_files",
    ),
    (
        "options from the environment",
        _unchanged,
        {"PYTEST_ADDOPTS": "-o testpaths=more"},
        "pytest's options override testpaths",
    ),
    (
        "configuration beside the tests",
        _beside(TESTS, "pytest.ini"),
        {},
        f"{TESTS.as_posix()}/pytest.ini can hold configuration pytest would read instead",
    ),
    (
        "project file beside the tests",
        _beside(TESTS, "pyproject.toml"),
        {},
        f"{TESTS.as_posix()}/pyproject.toml can hold configuration pytest would read instead",
    ),
    (
        "settings file beside the tests",
        _beside(TESTS, "tox.ini"),
        {},
        f"{TESTS.as_posix()}/tox.ini can hold configuration pytest would read instead",
    ),
    (
        "configuration beside the project",
        _beside(Path("services/api"), "pytest.toml"),
        {},
        "services/api/pytest.toml can hold configuration pytest would read instead",
    ),
    (
        "no pytest configuration",
        _project('[project]\nname = "lm-atelier-api"\n'),
        {},
        f"{PYPROJECT.as_posix()} has no pytest configuration",
    ),
]


@pytest.mark.parametrize(
    ("change", "environment", "refusal"),
    [drift[1:] for drift in COLLECTION_DRIFTS],
    ids=[drift[0] for drift in COLLECTION_DRIFTS],
)
def test_collection_the_listing_does_not_mirror_is_refused(
    tmp_path: Path,
    change: Callable[[Path], None],
    environment: dict[str, str],
    refusal: str,
) -> None:
    """Anything that would change what pytest collects stops the listing.

    The listing mirrors pytest's defaults, so a setting that changes them, a
    configuration file pytest would read instead of the API project's, or no
    configuration at all, which sends pytest looking above the project, would
    leave the parts listing other modules than the whole suite runs.
    """

    tree = _suite(tmp_path)
    change(tree)

    listing = _list_part(tree, "1/2", environment=environment)

    assert (listing.status, listing.modules, listing.outputs) == (1, [], {}), listing.message
    assert "Cannot list API test part '1/2'" in listing.message
    assert refusal in listing.message


def test_settings_pytest_reads_after_the_project_file_are_left_alone(tmp_path: Path) -> None:
    """Beside the API project's pyproject.toml, pytest reads that file first.

    So a tox.ini or setup.cfg there cannot change what pytest collects, and the
    listing lists the part as usual.
    """

    tree = _suite(tmp_path)
    (tree / "services/api/tox.ini").write_text("[pytest]\n", encoding="utf-8")
    (tree / "services/api/setup.cfg").write_text("[tool:pytest]\n", encoding="utf-8")

    listing = _list_part(tree, "1/2")

    assert listing.status == 0, listing.message
    assert listing.modules == [f"{TESTS.as_posix()}/test_a.py"]


def test_a_test_module_in_a_subdirectory_is_refused(tmp_path: Path) -> None:
    """pytest collects a test module in a subdirectory, and no part would run it.

    So the listing refuses the suite and names each one, rather than leaving
    them out of both parts.
    """

    tree = _suite(tmp_path, (*SUITE, "nested/test_x.py", "nested/deeper/y_test.py"))

    listing = _list_part(tree, "1/2")

    assert (listing.status, listing.modules, listing.outputs) == (1, [], {})
    assert (
        f"no part would run them: {TESTS.as_posix()}/nested/deeper/y_test.py, "
        f"{TESTS.as_posix()}/nested/test_x.py."
    ) in listing.message


@pytest.mark.parametrize(
    "part", ["0/2", "3/2", "1/0", "01/2", "1/02", "-1/2", "2", "1/2/3", "a/b", " 1/2", "1/2 ", ""]
)
def test_a_value_that_names_no_part_is_refused(tmp_path: Path, part: str) -> None:
    listing = _list_part(_suite(tmp_path), part)

    assert (listing.status, listing.modules, listing.outputs) == (1, [], {})
    assert f"{part!r} does not name a part" in listing.message


@pytest.mark.parametrize("arguments", [(), ("1/2", "2/2")], ids=["none", "two"])
def test_the_listing_takes_exactly_one_part(tmp_path: Path, arguments: tuple[str, ...]) -> None:
    listing = _list_part(_suite(tmp_path), *arguments)

    assert (listing.status, listing.modules, listing.outputs) == (2, [], {})
    assert "usage: api-test-part.py PART/PARTS" in listing.message


def test_the_listing_needs_only_the_standard_library() -> None:
    """The listing must not depend on what the environment running it has installed."""

    imported: set[str] = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    assert imported, "the listing imports nothing at all"
    assert sorted(imported - sys.stdlib_module_names - {"__future__"}) == []


def _bash() -> str:
    """Return a bash that runs scripts the way the Linux gate does."""

    shell = shutil.which("bash")
    if os.name == "nt":
        # Git's own bash, not the one Windows may put first on PATH, which is
        # a different system's.
        installed = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
        if installed.is_file():
            shell = str(installed)
    assert shell is not None
    return shell


def _bash_path(path: Path) -> str:
    value = path.as_posix()
    if os.name == "nt":
        assert len(path.drive) == 2 and path.drive.endswith(":")
        return "/" + path.drive[0].lower() + value[2:]
    return value


def _gate_tree(tmp_path: Path, modules: tuple[str, ...]) -> tuple[Path, Path]:
    """Return a copy of the gate over `modules`, and the file its pytest records arguments in.

    The gate's own interpreter runs the listing for real, and pytest is replaced
    by a script that records what it was asked to run.
    """

    tree = _suite(tmp_path / "tree", modules)
    shutil.copyfile(ROOT / "scripts" / "verify.sh", tree / "scripts" / "verify.sh")
    tools = tree / ".venv" / "bin"
    tools.mkdir(parents=True)
    recorded = tmp_path / "pytest-arguments"
    interpreter = shlex.quote(_bash_path(Path(sys.executable)))
    destination = shlex.quote(_bash_path(recorded))
    scripts = {
        "python": f'#!/usr/bin/env bash\nexec {interpreter} "$@"\n',
        "pytest": f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {destination}\n",
    }
    for name, script in scripts.items():
        (tools / name).write_text(script, encoding="utf-8", newline="\n")
        (tools / name).chmod(0o700)
    return tree, recorded


def _run_gate(tree: Path, *arguments: str, part: str | None) -> subprocess.CompletedProcess[str]:
    variables = {
        name: value
        for name, value in os.environ.items()
        if name not in {"LOCAL_LM_API_TEST_PART", "GITHUB_OUTPUT"}
    }
    variables["PYTEST_ADDOPTS"] = ""
    if part is not None:
        variables["LOCAL_LM_API_TEST_PART"] = part
    return subprocess.run(
        [_bash(), "scripts/verify.sh", *arguments],
        cwd=tree,
        env=variables,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def _recorded_arguments(recorded: Path) -> list[str]:
    """Return what pytest was asked to run, with the process id the scratch is named by fixed."""

    return [
        re.sub(r"^--basetemp=.*/temp/verify-pytest-[0-9]+$", "--basetemp=temp/verify-pytest", line)
        for line in recorded.read_text(encoding="utf-8").splitlines()
    ]


def test_the_gate_runs_a_named_part_with_the_options_of_the_whole_suite(tmp_path: Path) -> None:
    """With a part named, the API stage runs that part's modules and nothing else changes.

    Without the variable it runs the whole suite with the options the gate has
    always used, which is how a local run of the gate stays as it was.
    """

    tree, recorded = _gate_tree(tmp_path, SUITE)
    options = [
        "-q",
        "--durations=20",
        "--basetemp=temp/verify-pytest",
        "-p",
        "no:cacheprovider",
        "-n",
        "auto",
        "--dist",
        "loadfile",
    ]

    runs: dict[str | None, list[str]] = {}
    for part in (None, "1/2", "2/2"):
        finished = _run_gate(tree, "--api-tests-only", part=part)
        assert finished.returncode == 0, finished.stdout + finished.stderr
        assert "The API test stage passed." in finished.stdout
        runs[part] = _recorded_arguments(recorded)

    assert runs[None] == [TESTS.as_posix(), *options]
    assert runs["1/2"] == [f"{TESTS.as_posix()}/test_a.py", *options]
    assert runs["2/2"] == [
        f"{TESTS.as_posix()}/one_test.py",
        f"{TESTS.as_posix()}/test_b.py",
        f"{TESTS.as_posix()}/test_c.py",
        *options,
    ]


@pytest.mark.parametrize(
    ("part", "refusal"),
    [
        ("2/2", "API test part 2/2 holds none of the suite's 1 modules"),
        ("3/2", "'3/2' does not name a part"),
        ("", "'' does not name a part"),
    ],
    ids=["an empty part", "no such part", "an empty value"],
)
def test_the_gate_runs_no_tests_for_a_part_it_cannot_list(
    tmp_path: Path, part: str, refusal: str
) -> None:
    """A part the listing refuses fails the gate before pytest is started."""

    tree, recorded = _gate_tree(tmp_path, ("test_a.py",))

    finished = _run_gate(tree, "--api-tests-only", part=part)

    assert finished.returncode != 0
    assert refusal in finished.stderr
    assert "The API test stage passed." not in finished.stdout
    assert not recorded.exists()


def test_the_gate_refuses_an_argument_it_does_not_know(tmp_path: Path) -> None:
    tree, recorded = _gate_tree(tmp_path, SUITE)

    finished = _run_gate(tree, "--api-tests", part=None)

    assert finished.returncode == 2
    assert "Usage: scripts/verify.sh [--api-tests-only]" in finished.stderr
    assert not recorded.exists()


def test_both_parts_run_with_the_same_tools_installed() -> None:
    """A test that skips without a tool must not skip in only one part of the suite.

    Some API tests skip rather than fail when a tool they drive is missing,
    such as the web client's Node tooling. A part run without a tool the other
    part has would pass while its share of those tests never ran, so both jobs
    set up the same toolchains and install the same dependencies, and ffmpeg,
    before their part starts.
    """

    workflow: dict[str, Any] = yaml.safe_load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )

    def installed(job: str, part: str) -> tuple[list[tuple[str, Any]], list[str]]:
        steps = workflow["jobs"][job]["steps"]
        before = steps[: [step.get("name") for step in steps].index(part)]
        actions = [(step["uses"], step.get("with")) for step in before if "uses" in step]
        commands = [
            " ".join(line.split())
            for step in before
            if "run" in step
            for line in str(step["run"]).replace("\\\n", " ").splitlines()
            if line.strip().startswith(("uv sync", "npm ci", "sudo apt-get install"))
        ]
        return actions, commands

    first = installed("compatibility", "Run repository verification")
    second = installed("api-tests", "Run the second part of the API tests")

    assert [action for action, _inputs in first[0]] == [
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
        "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
    ]
    assert first[1] == [
        "sudo apt-get install --yes --no-install-recommends ffmpeg",
        "uv sync --project services/api --locked --python 3.12 --extra dev --extra package",
        "npm ci",
    ]
    assert second == first


def test_every_api_test_report_the_gate_reads_is_one_the_listing_writes(tmp_path: Path) -> None:
    """Each report reaches the merge gate from the listing that made it, by one name throughout.

    The gate reads each part's report from that job's outputs, each job
    publishes them from the step that runs its part, and that step writes them
    by running the listing. A name that differs anywhere along the way reads as
    empty, which the gate refuses for every head a full plan verifies, so
    nothing would merge. So each value is followed from the gate to the names
    the listing really writes, and the jobs that report are shown to run every
    part of one split between them.
    """

    written = set(_list_part(_suite(tmp_path), "1/2").outputs)
    workflow: dict[str, Any] = yaml.safe_load(
        (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    gate = jobs["merge-gate"]
    decision = next(
        step for step in gate["steps"] if step.get("run") == "python scripts/ci-merge-gate.py"
    )
    decided = (ROOT / "scripts/ci-merge-gate.py").read_text(encoding="utf-8")
    reports: dict[str, set[str]] = {}
    parts: dict[str, str] = {}
    for name, value in decision["env"].items():
        bound = re.fullmatch(r"\$\{\{ needs\.([a-z-]+)\.outputs\.([a-z_]+) \}\}", str(value))
        if bound is None or bound.group(1) == "verification-plan":
            continue
        job, output = bound.groups()
        source = re.fullmatch(
            r"\$\{\{ steps\.([a-z-]+)\.outputs\.([a-z_]+) \}\}", str(jobs[job]["outputs"][output])
        )
        assert source is not None, (name, jobs[job]["outputs"][output])
        step = next(step for step in jobs[job]["steps"] if step.get("id") == source.group(1))
        assert job in gate["needs"], name
        assert f'"{name}"' in decided, name
        assert source.group(2) == output
        assert str(step["run"]).startswith("./scripts/verify.sh"), step
        reports.setdefault(job, set()).add(source.group(2))
        parts[job] = step["env"]["LOCAL_LM_API_TEST_PART"]

    assert written == {"api_tests_total", "api_tests_selected", "api_tests_digest"}
    assert reports == {"compatibility": written, "api-tests": written}
    assert sorted(parts.values()) == ["1/2", "2/2"]
