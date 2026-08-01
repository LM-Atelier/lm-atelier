"""Export the API's OpenAPI schema deterministically for contract tooling.

The browser's hand-written `types.ts` mirrors the API's pydantic models with
no gate, and the two have drifted before (`Job` lacked the phase-duration
fields #220 added). This export is the first half of that gate: one stable
document - sorted keys, fixed formatting - so type generation and diffing
compare byte for byte. The generation and verify-script wiring land
separately once the schema surfaces are free.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the OpenAPI schema")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the schema to this file instead of standard output",
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "services" / "api"))
    from local_lm.config import Settings
    from local_lm.main import create_app

    with tempfile.TemporaryDirectory() as scratch:
        settings = Settings(
            data_dir=Path(scratch) / "data",
            dev=True,
            chat_engine="mock",
            media_engine="mock",
        )
        settings.prepare()
        document = create_app(settings).openapi()
    serialized = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
