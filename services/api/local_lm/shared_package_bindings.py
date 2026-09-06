"""Typed preparation records in the profile database; no shared-store mutation."""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import NoReturn

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import SharedPackageBinding
from .shared_asset_package_v1 import (
    PACKAGE_ROLES,
    PACKAGE_ROLES_V2,
    SCHEMA_ID,
    SCHEMA_ID_V2,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    _canonical_package_payload,
)

INVALID_BINDING = "shared package binding is invalid"
_STATES = frozenset({"preparing", "ready", "release_pending", "repair_required"})


class SharedPackageBindingError(ValueError):
    """A fixed refusal that does not echo paths or stored values."""


def _invalid() -> NoReturn:
    raise SharedPackageBindingError(INVALID_BINDING) from None


@dataclass(frozen=True)
class SharedPackageReference:
    library_id: str
    consumer_id: str
    package_digest: str
    members: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.library_id) is not str or len(self.library_id) != 36:
            _invalid()
        try:
            canonical_id = str(uuid.UUID(self.library_id))
        except ValueError:
            _invalid()
        if canonical_id != self.library_id:
            _invalid()
        if type(self.consumer_id) is not str or not re.fullmatch(
            r"[0-9a-f]{32,64}", self.consumer_id
        ):
            _invalid()
        if type(self.package_digest) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", self.package_digest
        ):
            _invalid()
        if type(self.members) is not dict or not self.members:
            _invalid()
        workflow = "workflow" in self.members
        roles = PACKAGE_ROLES_V2 if workflow else PACKAGE_ROLES
        if len(self.members) > len(roles):
            _invalid()
        for role, digest in self.members.items():
            if type(role) is not str or role not in roles:
                _invalid()
            if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
                _invalid()
        members = dict(sorted(self.members.items()))
        payload = _canonical_package_payload(
            members=members,
            schema=SCHEMA_ID_V2 if workflow else SCHEMA_ID,
            version=SCHEMA_VERSION_V2 if workflow else SCHEMA_VERSION,
        )
        if hashlib.sha256(payload).hexdigest() != self.package_digest:
            _invalid()
        object.__setattr__(self, "members", MappingProxyType(members))


def binding_reference(binding: SharedPackageBinding) -> SharedPackageReference:
    """Validate the local record before a later consumer resolves shared bytes."""

    if type(binding.state) is not str or binding.state not in _STATES:
        _invalid()
    if binding.claim_id is not None and (
        type(binding.claim_id) is not str or not re.fullmatch(r"[0-9a-f]{32}", binding.claim_id)
    ):
        _invalid()
    if binding.state == "ready" and binding.claim_id is None:
        _invalid()
    return SharedPackageReference(
        library_id=binding.library_id,
        consumer_id=binding.consumer_id,
        package_digest=binding.package_digest,
        members=binding.member_digests_json,
    )


def prepare_binding(session: Session, reference: SharedPackageReference) -> SharedPackageBinding:
    """Prepare idempotently in the caller's transaction, without activating an install.

    Reserving and finalizing an external claim belong to the later coordinator.
    Rollback leaves the existing install and any older binding authoritative.
    """

    # Take the profile writer before the identity lookup. This statement changes
    # no rows, but serializes concurrent preparations in the same SQLite database.
    session.connection().exec_driver_sql("UPDATE shared_package_bindings SET id = id WHERE 0")
    binding = session.scalar(
        select(SharedPackageBinding).where(
            SharedPackageBinding.library_id == reference.library_id,
            SharedPackageBinding.consumer_id == reference.consumer_id,
            SharedPackageBinding.package_digest == reference.package_digest,
        )
    )
    if binding is not None:
        recorded = binding_reference(binding)
        if recorded != reference or binding.state not in {"preparing", "ready"}:
            _invalid()
        return binding
    binding = SharedPackageBinding(
        library_id=reference.library_id,
        consumer_id=reference.consumer_id,
        package_digest=reference.package_digest,
        member_digests_json=dict(reference.members),
        state="preparing",
    )
    session.add(binding)
    session.flush()
    return binding
