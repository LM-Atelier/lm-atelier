"""Recover durable profile bindings without switching installs or deleting shared bytes."""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .filesystem_links import AnchoredDirectory, open_child_directory
from .models import ModelAssetInstall, ModelInstall, SharedPackageBinding
from .shared_asset_contract_v1 import (
    _anchored,
    _read_identity_anchored,
    _require_absolute_root,
    negotiate_store_access,
)
from .shared_asset_lock_v1 import hold
from .shared_asset_package_v1 import _load_package_from_store_versions
from .shared_asset_registry_v1 import (
    PROVISIONAL,
    claims_for_consumer,
    finalize_claim,
    release_claim,
    reserve_claim,
)
from .shared_package_bindings import (
    INVALID_BINDING,
    SharedPackageBindingError,
    SharedPackageReference,
    binding_reference,
)

SessionFactory = Callable[[], Session]


def _refuse() -> None:
    raise SharedPackageBindingError(INVALID_BINDING)


def _record(
    session: Session, binding_id: str, reference: SharedPackageReference
) -> SharedPackageBinding:
    session.connection().exec_driver_sql("UPDATE shared_package_bindings SET id=id WHERE 0")
    row = session.get(SharedPackageBinding, binding_id)
    if row is None or binding_reference(row) != reference:
        raise SharedPackageBindingError(INVALID_BINDING)
    return row


def _reference(sessions: SessionFactory, binding_id: str) -> SharedPackageReference | None:
    with sessions() as session:
        row = session.get(SharedPackageBinding, binding_id)
        return None if row is None else binding_reference(row)


@contextlib.contextmanager
def _operation(root: Path, reference: SharedPackageReference) -> Iterator[AnchoredDirectory]:
    # Attachment must already have certified the selected filesystem. Keeping
    # the root handle here does not strengthen the registry's POSIX pathname
    # opening contract; registry confinement remains that module's boundary.
    with _anchored(_require_absolute_root(root)) as anchor:
        identity = _read_identity_anchored(anchor)
        if (
            identity is None
            or identity.library_uuid != reference.library_id
            or negotiate_store_access(identity) != "read_write"
        ):
            _refuse()
        key = hashlib.sha256(
            (reference.consumer_id + ":" + reference.package_digest).encode("ascii")
        ).hexdigest()
        with (
            open_child_directory(anchor, "locks", create=True) as locks,
            hold(locks, "binding-" + key + ".lock"),
        ):
            yield anchor


def complete_binding_claim(*, sessions: SessionFactory, root: Path, binding_id: str) -> str:
    """Verify and finalize an existing local preparation, returning its claim.

    The caller supplies its own profile session factory and an already attached
    library. This is not browser authority or activation. A failed operation
    retains its provisional/final claim so retry cannot free protected bytes.
    Concurrent operations for this consumer/package refuse until retried.
    """
    reference = _reference(sessions, binding_id)
    if reference is None:
        raise SharedPackageBindingError(INVALID_BINDING)
    with _operation(root, reference) as anchor:
        with sessions() as session:
            row = _record(session, binding_id, reference)
            if row.state not in {"preparing", "ready"}:
                _refuse()
            previous_claim = row.claim_id
        claim = reserve_claim(
            database=root / "index.sqlite3",
            consumer_id=reference.consumer_id,
            package_digest=reference.package_digest,
        )
        if previous_claim is not None and previous_claim != claim.claim_id:
            _refuse()
        with sessions() as session:
            row = _record(session, binding_id, reference)
            if row.state not in {"preparing", "ready"} or row.claim_id != previous_claim:
                _refuse()
            row.claim_id = claim.claim_id
            session.commit()

        # Neither database holds a writer while package and member bytes hash.
        members = _load_package_from_store_versions(
            store=anchor, digest=reference.package_digest, allow_v2=True
        )
        if dict(members) != dict(reference.members):
            _refuse()
        finalized = finalize_claim(
            database=root / "index.sqlite3",
            consumer_id=reference.consumer_id,
            claim_id=claim.claim_id,
        )
        if finalized.package_digest != reference.package_digest:
            _refuse()
        with sessions() as session:
            row = _record(session, binding_id, reference)
            if row.state not in {"preparing", "ready"} or row.claim_id != claim.claim_id:
                _refuse()
            row.state = "ready"
            session.commit()
        return claim.claim_id


def _unreferenced(session: Session, binding_id: str) -> None:
    for model in (ModelInstall, ModelAssetInstall):
        if (
            session.scalar(
                select(model.id).where(model.shared_package_binding_id == binding_id).limit(1)
            )
            is not None
        ):
            _refuse()


def release_binding_claim(*, sessions: SessionFactory, root: Path, binding_id: str) -> None:
    """Release an unreferenced profile record; preserve all shared payload bytes.

    release_pending is committed before the external release. The final short
    profile writer spans the registry release and local deletion, preventing an
    install reference from appearing between the last reference check and delete.
    Recovery of an absent claim is allowed only after release_pending is durable.
    """
    reference = _reference(sessions, binding_id)
    if reference is None:
        return
    with _operation(root, reference):
        with sessions() as session:
            row = _record(session, binding_id, reference)
            _unreferenced(session, binding_id)
            claims = claims_for_consumer(
                database=root / "index.sqlite3", consumer_id=reference.consumer_id
            )
            matching = next(
                (item for item in claims if item.package_digest == reference.package_digest),
                None,
            )
            if row.claim_id is None:
                # A crash may leave a provisional reservation before its local
                # identity commit. Never adopt an unexplained finalized claim.
                if matching is not None:
                    if matching.state != PROVISIONAL:
                        _refuse()
                    row.claim_id = matching.claim_id
            elif matching is None:
                if row.state != "release_pending":
                    _refuse()
            elif matching.claim_id != row.claim_id:
                _refuse()
            row.state = "release_pending"
            session.commit()
        with sessions() as session:
            row = _record(session, binding_id, reference)
            _unreferenced(session, binding_id)
            if row.state != "release_pending":
                _refuse()
            claims = claims_for_consumer(
                database=root / "index.sqlite3", consumer_id=reference.consumer_id
            )
            matching = next(
                (item for item in claims if item.package_digest == reference.package_digest),
                None,
            )
            if matching is not None:
                if matching.claim_id != row.claim_id:
                    _refuse()
                release_claim(
                    database=root / "index.sqlite3",
                    consumer_id=reference.consumer_id,
                    claim_id=matching.claim_id,
                )
            session.delete(row)
            session.commit()
