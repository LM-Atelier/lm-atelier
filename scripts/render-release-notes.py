from __future__ import annotations

import argparse
import re
from pathlib import Path

VERSION = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

REPLACEMENTS = {
    "<!-- build-smoked / physically tested / certified -->": (
        "automated build smoke passed; physical certification pending"
    ),
    "<!-- State prerequisites, supported upgrade path, and data-preservation behavior. -->": (
        "Use the platform instructions in the README. Upgrades preserve the managed local "
        "data directory; uninstall preserves it unless purge is explicitly selected."
    ),
    "<!-- List material limitations, including unsigned status when applicable. -->": (
        "These preview binaries are not code-signed. Model, workflow, and hardware support "
        "is limited to the matrix explicitly recorded for this release."
    ),
}


def render(template: str, version: str, source_sha: str) -> str:
    if not VERSION.fullmatch(version):
        raise ValueError("version must be a SemVer-style identifier")
    if not COMMIT_SHA.fullmatch(source_sha):
        raise ValueError("source SHA must be a full lowercase Git commit identifier")
    result = template.replace("<!-- version -->", version)
    result = result.replace("<!-- full tagged commit SHA -->", source_sha)
    for placeholder, value in REPLACEMENTS.items():
        result = result.replace(placeholder, value)
    if "<!--" in result or "-->" in result:
        raise ValueError("release template contains an unresolved review placeholder")
    return result.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render reviewed LM Atelier release notes.")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    content = render(
        arguments.template.read_text(encoding="utf-8"),
        arguments.version,
        arguments.source_sha,
    )
    arguments.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
