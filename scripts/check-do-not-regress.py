"""Fail when something recorded as finished loses the thing that proved it.

A commitment marked done is only as durable as the check that holds it up. Tests
get renamed during a refactor, ceilings get raised to make a lint error go away,
and the record still says the work is finished - so the next audit re-derives the
same answer from scratch, which is how a body of planning starts disagreeing with
itself.

The register lives outside this repository because it names planning items that
are not public. When it is absent - a fresh clone, continuous integration - this
check reports that and succeeds, because a missing private file is not a
regression.

Entries come in three kinds:

    test      a named function that must still exist in a named file
    ceiling   a numeric cap that may fall but never rise
    decision  a decision with a date, which nothing can prove and nothing should

An entry may name `superseded_by`, so a later phase can retire a proof by
pointing at what replaced it rather than by deleting it quietly.

A proof is registered when its work lands, not while it is in flight: the gate
runs on whatever branch you are standing on, so an entry naming a test that only
exists on an unmerged branch would fail every other branch's gate.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REGISTER = Path(".private/do-not-regress.json")
KINDS = {"test", "ceiling", "decision"}


def _fail(entry_id: str, detail: str) -> str:
    return f"  {entry_id}: {detail}"


def _check_test(root: Path, entry: dict[str, object]) -> str | None:
    path = root / str(entry["file"])
    name = str(entry["name"])
    if not path.is_file():
        return _fail(str(entry["id"]), f"{entry['file']} no longer exists")
    source = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix in {".ts", ".tsx", ".js", ".jsx"}:
        # A vitest case is named by its string, not by an identifier.
        found = re.search(rf"""\b(?:it|test)\(\s*["'`]{re.escape(name)}["'`]""", source)
    else:
        found = re.search(rf"(?:async\s+)?def\s+{re.escape(name)}\s*\(", source)
    if not found:
        return _fail(str(entry["id"]), f"{name} is gone from {entry['file']}")
    return None


def _check_ceiling(root: Path, entry: dict[str, object]) -> str | None:
    path = root / str(entry["file"])
    if not path.is_file():
        return _fail(str(entry["id"]), f"{entry['file']} no longer exists")
    source = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(str(entry["pattern"]), source)
    if not match:
        return _fail(str(entry["id"]), f"the {entry['pattern']!r} ceiling is gone")
    current = int(match.group(1))
    recorded = int(str(entry["max"]))
    if current > recorded:
        return _fail(
            str(entry["id"]),
            f"ceiling rose from {recorded} to {current}; it may fall but never rise",
        )
    return None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    register = root / REGISTER
    if not register.is_file():
        print(f"No {REGISTER} here; nothing to hold up.")
        return 0

    try:
        entries = json.loads(register.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"{REGISTER} could not be read: {error}")
        return 1
    if not isinstance(entries, list):
        print(f"{REGISTER} must contain a list of entries.")
        return 1

    problems: list[str] = []
    checked = 0
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry or entry.get("kind") not in KINDS:
            problems.append(f"  {entry!r:.60}: not a registered entry")
            continue
        if entry.get("superseded_by"):
            continue
        kind = entry["kind"]
        if kind == "decision":
            if not entry.get("decided"):
                problems.append(_fail(str(entry["id"]), "a decision needs the date it was taken"))
            continue
        checked += 1
        problem = _check_test(root, entry) if kind == "test" else _check_ceiling(root, entry)
        if problem:
            problems.append(problem)

    if problems:
        print(f"{len(problems)} registered proof(s) no longer hold:")
        print("\n".join(problems))
        print("\nEither restore the proof, or record what supersedes it.")
        return 1
    print(f"{checked} registered proof(s) still hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
