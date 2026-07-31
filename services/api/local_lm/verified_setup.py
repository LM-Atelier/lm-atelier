"""A working setup, as data that means the same thing on another machine.

The owner tunes a setup until it works. Today that result is not portable: it is
a directory shape plus rows keyed by local UUIDs, and nothing that survives an
export describes *why* the setup works or how to recognise the same thing here.

This is the record that does travel. Every field identifies something by content
or by a name the receiving machine can resolve for itself - model files by hash,
the template by identity, the workflow by the artifact hash of what it executes,
the machine by an envelope rather than a fingerprint. No install id, profile id,
revision id, chat, run or job appears anywhere in it.

It also carries an attestation: not "this configuration is believed good" but
"this exact configuration produced a real generation on a machine within this
envelope, at this time". That is the part a new user cannot obtain by reading
documentation, and the part that makes the record worth shipping.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy.orm import Session

from .model_planner import model_components_for_install
from .models import (
    ModelCapabilityEvidence,
    ModelInstall,
    ModelProfile,
    SetupVerification,
    WorkflowRevision,
)

VERIFIED_SETUP_VERSION = 1

# Settings that name something local, and so must never be shipped. They would
# resolve to nothing on the receiving machine, or worse, to the wrong thing.
_LOCAL_SETTING_KEYS = frozenset(
    {
        "model_install_id",
        "profile_id",
        "workflow_revision_id",
        "install_plan_id",
        "local_path",
        "model_path",
    }
)

# Identifiers that end in `_id` but name something every machine can resolve:
# a catalog template, and a remote repository.
_PORTABLE_ID_KEYS = frozenset({"template_id", "remote_id"})


def _portable_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(settings.items())
        if key not in _LOCAL_SETTING_KEYS and not key.startswith("_")
    }


def verified_setup_digest(payload: Mapping[str, Any]) -> str:
    """Identify a setup record by its content, so two copies compare equal."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_verified_setup(
    session: Session,
    *,
    verification: SetupVerification,
    install: ModelInstall,
    profile: ModelProfile,
    revision: WorkflowRevision | None,
    evidence: ModelCapabilityEvidence,
) -> dict[str, Any]:
    """Describe a locally verified setup in terms another machine can resolve.

    The caller supplies the rows rather than having them looked up here, because
    which verification counts as current is a question the setup service already
    answers and should not be answered twice.
    """
    manifest = install.manifest_json or {}
    workflow: dict[str, Any] | None = None
    if revision:
        dependencies = revision.dependencies_json or {}
        workflow = {
            "engine": revision.engine,
            "template_id": dependencies.get("template_id"),
            "template_sha256": dependencies.get("template_sha256"),
            # What the revision executes, not which template produced it - the
            # distinction #287 introduced and #289 keys evidence on.
            "artifact_sha256": revision.artifact_sha256,
        }

    payload: dict[str, Any] = {
        "version": VERIFIED_SETUP_VERSION,
        "role": verification.role,
        "engine": install.engine,
        "model": {
            "name": install.name,
            "remote_id": manifest.get("remote_id"),
            "revision": manifest.get("revision"),
            # The load-bearing part: files by hash, so the receiving machine can
            # recognise what it already has instead of trusting a name.
            "components": sorted(
                model_components_for_install(session, install.id),
                key=lambda item: (item["target_folder"], item["sha256"]),
            ),
        },
        "workflow": workflow,
        "settings": {
            "load": _portable_settings(profile.load_settings_json or {}),
            "request": _portable_settings(profile.request_settings_json or {}),
        },
        # Requirements, not a fingerprint. See `hardware_envelope`.
        "hardware": evidence.hardware_envelope_json,
        "attestation": {
            "generated_output": verification.state == "verified",
            "verified_at": (
                verification.completed_at.isoformat() if verification.completed_at else None
            ),
            "probe_version": evidence.probe_version,
            "adapter_contract_version": evidence.adapter_contract_version,
            "launch_contract_version": evidence.launch_contract_version,
            "runtime_build": evidence.runtime_build,
        },
    }
    payload["digest"] = verified_setup_digest(payload)
    return payload


def local_identifiers_in(payload: Mapping[str, Any]) -> list[str]:
    """Any key in the record that names something local to one machine.

    Used as an assertion rather than a filter: if this ever returns anything, the
    record has stopped being portable and the fix belongs at the point that added
    the field, not here.
    """
    found: list[str] = []

    def is_local(key: str) -> bool:
        if key in _PORTABLE_ID_KEYS:
            return False
        return key in _LOCAL_SETTING_KEYS or key.endswith("_id")

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                where = f"{path}.{key}" if path else str(key)
                if is_local(str(key)):
                    found.append(where)
                walk(item, where)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(payload, "")
    return found
