"""The LoRA cap a workflow is given must be the cap it is later judged by.

When a workflow is compiled from a built-in template and its graph can take a
LoRA, the compiler writes a `loras` property into the workflow's settings schema
with a maximum number of entries. Every run of such a workflow is then checked
against `MAX_LORA_STACK_SIZE`, and a declared maximum above it is refused.

Those two numbers are written in different places and agree only because
somebody chose the same digit twice. Raise the constant and the declaration
keeps the old, smaller cap, so people are offered fewer than the application
would accept; lower it, and every workflow already carrying the larger
declaration is refused outright, with nothing pointing at the change that did
it. Neither failure appears where the edit was made.

Reading the declaration as text is the point rather than a shortcut: importing
it would reuse whatever the module computes and could only ever agree with
itself.
"""

from __future__ import annotations

import re
from pathlib import Path

from local_lm.auxiliary_assets import MAX_LORA_STACK_SIZE

REPOSITORY = Path(__file__).resolve().parents[3]
DOWNLOADS = REPOSITORY / "services" / "api" / "local_lm" / "downloads.py"
DECLARATION = re.compile(r"properties\[\"loras\"\]\s*=\s*\{(?P<body>.*?)\}", re.DOTALL)
MAXIMUM = re.compile(r"\"maxItems\"\s*:\s*(?P<value>\d+)")


def test_the_declared_lora_cap_matches_the_one_the_contract_enforces() -> None:
    source = DOWNLOADS.read_text(encoding="utf-8")
    declarations = DECLARATION.findall(source)
    # A declaration that moved or gained a sibling fails here rather than
    # leaving the comparison below with nothing to measure.
    assert len(declarations) == 1, "expected exactly one declared LoRA setting"
    caps = MAXIMUM.findall(declarations[0])
    assert len(caps) == 1, "expected exactly one maximum in the declared setting"

    assert int(caps[0]) == MAX_LORA_STACK_SIZE
