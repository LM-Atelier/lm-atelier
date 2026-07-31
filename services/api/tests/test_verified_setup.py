"""The record that makes a working setup portable.

Every field has to identify something by content or by a name the receiving
machine can resolve. If a local identifier ever appears, the record has quietly
stopped being portable while still looking correct - which is the failure these
tests exist to catch.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.hardware import hardware_envelope
from local_lm.models import ModelComponentManifest, ModelInstall
from local_lm.verified_setup import (
    VERIFIED_SETUP_VERSION,
    build_verified_setup,
    local_identifiers_in,
    resolve_verified_setup,
    verified_setup_digest,
)

_COUNTER = itertools.count()
_SHA = "a" * 64


def _install(session, suffix: str) -> ModelInstall:  # type: ignore[no-untyped-def]
    install = ModelInstall(
        id=f"install_{suffix}",
        name="acme/illustrator",
        role="image",
        engine="comfyui",
        local_path=f"C:/models/{suffix}",
        size_bytes=1,
        manifest_json={"remote_id": "acme/illustrator", "revision": "main"},
    )
    session.add(install)
    session.flush()
    session.add(
        ModelComponentManifest(
            model_install_id=install.id,
            kind="checkpoint",
            relative_path="checkpoints/weights.safetensors",
            target_folder="checkpoints",
            sha256=_SHA,
            size_bytes=1,
            required=True,
            metadata_json={},
        )
    )
    session.flush()
    return install


def _parts(install_id: str) -> dict[str, Any]:
    return {
        "verification": SimpleNamespace(
            role="image",
            state="verified",
            completed_at=datetime(2026, 7, 31, tzinfo=UTC),
        ),
        "profile": SimpleNamespace(
            load_settings_json={"context": 8192, "model_install_id": install_id},
            request_settings_json={"steps": 30, "_internal": "no"},
        ),
        "revision": SimpleNamespace(
            engine="comfyui",
            artifact_sha256="b" * 64,
            dependencies_json={"template_id": "image_basic", "template_sha256": "c" * 64},
        ),
        "evidence": SimpleNamespace(
            hardware_envelope_json={"version": 1, "platform": "windows"},
            probe_version="activation-probe-v2",
            adapter_contract_version=1,
            launch_contract_version="worker-launch-v1",
            runtime_build="comfy-0.28.0",
        ),
    }


def test_a_verified_setup_carries_no_local_identifiers(settings: Settings) -> None:
    """The whole point: nothing in here names a row on this machine."""
    with SessionLocal() as session:
        install = _install(session, f"case{next(_COUNTER)}")

        payload = build_verified_setup(session, install=install, **_parts(install.id))

        assert local_identifiers_in(payload) == []
        # Specifically, the profile setting that named an install is gone.
        assert "model_install_id" not in payload["settings"]["load"]
        assert install.id not in repr(payload)
        session.rollback()


def test_models_travel_as_hashes_rather_than_names(settings: Settings) -> None:
    with SessionLocal() as session:
        install = _install(session, f"case{next(_COUNTER)}")

        payload = build_verified_setup(session, install=install, **_parts(install.id))

        assert payload["model"]["components"] == [{"target_folder": "checkpoints", "sha256": _SHA}]
        assert payload["model"]["remote_id"] == "acme/illustrator"
        session.rollback()


def test_the_workflow_travels_as_what_it_executes(settings: Settings) -> None:
    with SessionLocal() as session:
        install = _install(session, f"case{next(_COUNTER)}")

        payload = build_verified_setup(session, install=install, **_parts(install.id))

        assert payload["workflow"]["artifact_sha256"] == "b" * 64
        assert payload["workflow"]["template_sha256"] == "c" * 64
        session.rollback()


def test_the_attestation_says_a_generation_actually_happened(settings: Settings) -> None:
    with SessionLocal() as session:
        install = _install(session, f"case{next(_COUNTER)}")

        payload = build_verified_setup(session, install=install, **_parts(install.id))

        assert payload["attestation"]["generated_output"] is True
        assert payload["attestation"]["verified_at"].startswith("2026-07-31")
        session.rollback()


def test_an_unverified_setup_does_not_claim_a_generation(settings: Settings) -> None:
    """A record that only says "this ought to work" must not say more than that."""
    with SessionLocal() as session:
        install = _install(session, f"case{next(_COUNTER)}")
        parts = _parts(install.id)
        parts["verification"] = SimpleNamespace(role="image", state="failed", completed_at=None)

        payload = build_verified_setup(session, install=install, **parts)

        assert payload["attestation"]["generated_output"] is False
        session.rollback()


def test_the_same_setup_produces_the_same_digest(settings: Settings) -> None:
    with SessionLocal() as session:
        install = _install(session, f"case{next(_COUNTER)}")

        first = build_verified_setup(session, install=install, **_parts(install.id))
        second = build_verified_setup(session, install=install, **_parts(install.id))

        assert first["digest"] == second["digest"]
        # And it identifies the content, not the object.
        without = {key: value for key, value in first.items() if key != "digest"}
        assert first["digest"] == verified_setup_digest(without)
        assert first["version"] == VERIFIED_SETUP_VERSION
        session.rollback()


def test_a_changed_setting_changes_the_digest(settings: Settings) -> None:
    with SessionLocal() as session:
        install = _install(session, f"case{next(_COUNTER)}")
        parts = _parts(install.id)
        baseline = build_verified_setup(session, install=install, **parts)

        parts["profile"] = SimpleNamespace(
            load_settings_json={"context": 4096},
            request_settings_json={"steps": 30},
        )
        changed = build_verified_setup(session, install=install, **parts)

        assert changed["digest"] != baseline["digest"]
        session.rollback()


def test_local_identifiers_in_finds_what_it_is_for() -> None:
    """The guard has to actually catch something, or it is decoration."""
    assert local_identifiers_in({"settings": {"load": {"model_install_id": "x"}}}) == [
        "settings.load.model_install_id"
    ]
    assert local_identifiers_in({"nested": [{"profile_id": "x"}]}) == ["nested[0].profile_id"]
    # Portable identifiers are not flagged.
    assert local_identifiers_in({"remote_id": "acme/x", "template_id": "image_basic"}) == []


def _artifact(settings: Settings, sha256: str = _SHA, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": VERIFIED_SETUP_VERSION,
        "role": "image",
        "engine": "comfyui",
        "model": {"components": [{"target_folder": "checkpoints", "sha256": sha256}]},
        "workflow": None,
        "settings": {"load": {}, "request": {}},
        "hardware": hardware_envelope(settings),
        "attestation": {"generated_output": True},
        "digest": "d" * 64,
    }
    payload.update(overrides)
    return payload


def test_an_imported_setup_resolves_against_bytes_this_machine_already_has(
    settings: Settings,
) -> None:
    """The point of content addressing: the local install has another name."""
    with SessionLocal() as session:
        _install(session, f"case{next(_COUNTER)}")

        resolved = resolve_verified_setup(session, _artifact(settings), settings)

        assert resolved["components"] == [
            {"target_folder": "checkpoints", "sha256": _SHA, "present": True}
        ]
        assert resolved["missing_components"] == []
        assert resolved["requires_approval"] is False
        assert resolved["ready_to_verify"] is True
        session.rollback()


def test_missing_components_are_reported_for_approval_not_fetched(
    settings: Settings,
) -> None:
    """An imported file must not be able to start a multi-gigabyte download."""
    with SessionLocal() as session:
        _install(session, f"case{next(_COUNTER)}")

        resolved = resolve_verified_setup(session, _artifact(settings, sha256="e" * 64), settings)

        assert resolved["missing_components"] == [
            {"target_folder": "checkpoints", "sha256": "e" * 64}
        ]
        assert resolved["requires_approval"] is True
        assert resolved["ready_to_verify"] is False
        session.rollback()


def test_an_imported_attestation_never_counts_as_local_verification(
    settings: Settings,
) -> None:
    """Configuration travels; trust does not.

    The artifact says a generation succeeded elsewhere. That is provenance, and
    it must stay visibly distinct from anything this machine has earned.
    """
    with SessionLocal() as session:
        _install(session, f"case{next(_COUNTER)}")

        resolved = resolve_verified_setup(session, _artifact(settings), settings)

        assert resolved["verified_elsewhere"] is True
        assert resolved["verified_here"] is False
        session.rollback()


def test_incompatible_hardware_withdraws_even_the_elsewhere_claim(
    settings: Settings,
) -> None:
    """A proof from hardware this machine cannot match recommends nothing here."""
    with SessionLocal() as session:
        _install(session, f"case{next(_COUNTER)}")
        elsewhere = _artifact(
            settings,
            hardware={
                "version": 1,
                "platform": "linux",
                "architecture": "amd64",
                "memory_bytes": 1,
                "accelerator_memory_bytes": 0,
                "backends": [],
            },
        )

        resolved = resolve_verified_setup(session, elsewhere, settings)

        assert resolved["hardware_compatible"] is False
        assert resolved["verified_elsewhere"] is False
        assert resolved["ready_to_verify"] is False
        session.rollback()
