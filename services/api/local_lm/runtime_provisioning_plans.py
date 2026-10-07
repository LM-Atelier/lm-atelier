"""Describe the exact runtime operation before an installation is approved."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RuntimeProvisioningPlan:
    engine: str
    operation: Literal["reuse_configured", "reuse_managed", "install_managed"]
    release: str | None
    license: str
    download_bytes: int
    required_free_bytes: int
    inputs_sha256: str
    plan_sha256: str
