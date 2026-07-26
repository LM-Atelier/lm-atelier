from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "services" / "api" / "local_lm" / "__init__.py"
VERSION_PATTERN = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)

JSON_VERSION_PATHS: dict[Path, tuple[tuple[str, ...], ...]] = {
    ROOT / "package.json": (("version",),),
    ROOT / "apps" / "web" / "package.json": (("version",),),
    ROOT / "package-lock.json": (
        ("version",),
        ("packages", "", "version"),
        ("packages", "apps/web", "version"),
    ),
    ROOT / "packaging" / "release-manifest.json": (("version",),),
}


def canonical_version() -> str:
    match = VERSION_PATTERN.search(VERSION_FILE.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Could not read __version__ from {VERSION_FILE}")
    return match.group(1)


def value_at(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for part in path:
        if not isinstance(value, dict) or part not in value:
            raise KeyError(".".join(path))
        value = value[part]
    return value


def set_value_at(document: dict[str, Any], path: tuple[str, ...], value: str) -> None:
    target: Any = document
    for part in path[:-1]:
        if not isinstance(target, dict) or part not in target:
            raise KeyError(".".join(path))
        target = target[part]
    if not isinstance(target, dict) or path[-1] not in target:
        raise KeyError(".".join(path))
    target[path[-1]] = value


def sync_json_versions(version: str, *, write: bool) -> list[str]:
    mismatches: list[str] = []
    for path, locations in JSON_VERSION_PATHS.items():
        document = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for location in locations:
            current = value_at(document, location)
            if current == version:
                continue
            mismatches.append(
                f"{path.relative_to(ROOT)}:{'.'.join(location)} is {current!r}, expected {version!r}"
            )
            if write:
                set_value_at(document, location, version)
                changed = True
        if changed:
            path.write_text(
                json.dumps(document, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    return mismatches


def validate_dynamic_sources() -> list[str]:
    problems: list[str] = []
    pyproject = (ROOT / "services" / "api" / "pyproject.toml").read_text(encoding="utf-8")
    if 'dynamic = ["version"]' not in pyproject:
        problems.append("services/api/pyproject.toml must declare version as dynamic")
    if 'path = "local_lm/__init__.py"' not in pyproject:
        problems.append("services/api/pyproject.toml must read the canonical version module")

    installer = (ROOT / "packaging" / "windows" / "LMAtelier.iss").read_text(
        encoding="utf-8"
    )
    if re.search(r'#define MyAppVersion "[^"]+"', installer):
        problems.append("packaging/windows/LMAtelier.iss must receive MyAppVersion at build time")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check or update generated version fields from local_lm.__version__."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update generated JSON fields instead of only checking them.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_version",
        help="Print the canonical version.",
    )
    args = parser.parse_args()

    version = canonical_version()
    mismatches = sync_json_versions(version, write=args.write)
    problems = validate_dynamic_sources()
    if args.print_version:
        print(version)
    if mismatches and args.write:
        mismatches = sync_json_versions(version, write=False)
    errors = [*mismatches, *problems]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if not args.print_version:
        print(f"Version metadata matches {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
