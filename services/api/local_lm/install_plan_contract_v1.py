"""Bind model install evidence to the contract that created its identity."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Any

from .models import InstallPlan

INSTALL_RESOLVER_VERSION = "install-resolver-v10"
LEGACY_INSTALL_RESOLVER_VERSION = "install-resolver-v9"


def install_plan_contract_hash(payload: Mapping[str, Any]) -> str:
    """Hash the complete canonical contract without changing its stored version."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def stored_install_plan_identity_matches(plan: InstallPlan) -> bool:
    """Reject altered evidence or a version label that does not match its hash."""
    payload = {
        "provider": plan.provider,
        "remote_id": plan.remote_id,
        "revision": plan.revision,
        "role": plan.role,
        "engine": plan.engine,
        "architecture": plan.architecture,
        "family": plan.family,
        "compatibility": plan.compatibility,
        "artifacts": plan.artifacts_json,
        "runtime_contract": plan.runtime_contract_json,
        "activation_probe": plan.activation_probe_json,
        "resolver_version": plan.resolver_version,
    }
    try:
        digest = install_plan_contract_hash(payload)
        return isinstance(plan.plan_hash, str) and hmac.compare_digest(digest, plan.plan_hash)
    except (TypeError, ValueError, OverflowError):
        return False
