"""Create the canonical identity for constructed install-plan evidence."""

from __future__ import annotations

import hashlib
import json

from local_lm.models import InstallPlan


def bind_install_plan_identity(plan: InstallPlan) -> InstallPlan:
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
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    plan.plan_hash = hashlib.sha256(encoded.encode()).hexdigest()
    return plan
