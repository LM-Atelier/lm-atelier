from __future__ import annotations

import json
from pathlib import Path

import pytest

import local_lm.capability_packs as capability_packs
from local_lm.capability_packs import CapabilityPackError, architecture_family_contracts


def test_bundled_architecture_contracts_are_hash_pinned_data() -> None:
    capability_packs.architecture_family_contracts.cache_clear()
    contracts = architecture_family_contracts()

    assert any(
        contract.id == "qwen"
        and contract.roles == ("chat",)
        and "qwen" in contract.architecture_markers
        for contract in contracts
    )
    assert any(contract.id == "stable-diffusion-xl" for contract in contracts)


def test_modified_capability_pack_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "architecture-families-v1.json").write_text(
        json.dumps(
            {
                "version": 1,
                "families": [
                    {
                        "id": "untrusted",
                        "roles": ["chat"],
                        "architecture_markers": ["untrusted"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "capability-packs.lock.json").write_text(
        json.dumps(
            {
                "version": 1,
                "packs": {"architecture-families-v1.json": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(capability_packs, "_PACK_DIRECTORY", tmp_path)
    capability_packs.architecture_family_contracts.cache_clear()
    try:
        with pytest.raises(CapabilityPackError, match="integrity"):
            architecture_family_contracts()
    finally:
        capability_packs.architecture_family_contracts.cache_clear()
