"""Connection-local authority for checked artifact deletions.

Application SQLite engines register a closed-by-default function. Connections
without that function cannot execute the guarded deletion statement.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Connection, Engine, event
from sqlalchemy.orm import Session

from .artifact_library_schema import (
    ARTIFACT_DELETION_AUTHORIZED_SQL,
    STORED_JSON_INVALID_SQL,
)

_CONNECTION_STATE = "artifact_deletion_authority"
_PROOF_SEAL = object()
_REFERENCE_GRAPH_COVERAGE = object()


class ArtifactDeletionProofError(ValueError):
    pass


class _ArtifactReferenceSnapshot(frozenset[str]):
    _session: Session
    _transaction: object
    _nested_transaction: object
    _connection: Connection
    _connection_transaction: object
    _driver: sqlite3.Connection
    _changes: int
    _coverage: object
    _referrers: dict[str, frozenset[str]]
    _removed: set[str]

    def __new__(
        cls,
        values: Iterable[str],
        *,
        session: Session,
        connection: Connection,
        driver: sqlite3.Connection,
        referrers: dict[str, set[str]],
    ) -> _ArtifactReferenceSnapshot:
        snapshot = super().__new__(cls, values)
        snapshot._session = session
        snapshot._transaction = session.get_transaction()
        snapshot._nested_transaction = session.get_nested_transaction()
        snapshot._connection = connection
        snapshot._connection_transaction = connection.get_transaction()
        snapshot._driver = driver
        snapshot._changes = driver.total_changes
        snapshot._coverage = _REFERENCE_GRAPH_COVERAGE
        snapshot._referrers = {key: frozenset(ids) for key, ids in referrers.items()}
        snapshot._removed = set()
        return snapshot

    def current_identity(self, session: Session) -> bool:
        return (
            self._session is session
            and self._transaction is session.get_transaction()
            and self._nested_transaction is session.get_nested_transaction()
            and not self._connection.closed
            and self._connection_transaction is self._connection.get_transaction()
            and self._driver is self._connection.connection.driver_connection
            and self._driver.in_transaction
        )


@dataclass(frozen=True)
class _ArtifactDeletionProof:
    session: Session
    transaction: object
    candidate_ids: frozenset[str]
    references: _ArtifactReferenceSnapshot
    seal: object


@dataclass
class _ConnectionAuthority:
    active: _ArtifactDeletionProof | None = None

    def allows(self, artifact_id: object) -> int:
        proof = self.active
        return int(
            proof is not None
            and isinstance(artifact_id, str)
            and artifact_id in proof.candidate_ids
            and proof.references.current_identity(proof.session)
            and proof.references._changes == proof.references._driver.total_changes
        )


@event.listens_for(Engine, "connect")
def _register_artifact_deletion_authority(
    dbapi_connection: object,
    connection_record: Any,
) -> None:
    create_function = getattr(dbapi_connection, "create_function", None)
    if create_function is None:
        return
    state = _ConnectionAuthority()
    connection_record.info[_CONNECTION_STATE] = state
    create_function(ARTIFACT_DELETION_AUTHORIZED_SQL, 1, state.allows)


def complete_reference_snapshot(
    session: Session,
    values: Iterable[str],
    *,
    referrers: dict[str, set[str]],
) -> AbstractSet[str]:
    connection = session.connection()
    driver = connection.connection.driver_connection
    if not isinstance(driver, sqlite3.Connection) or not driver.in_transaction:
        return frozenset(values)
    if connection.exec_driver_sql(STORED_JSON_INVALID_SQL).scalar_one():
        raise ArtifactDeletionProofError("artifact JSON reference is invalid")
    return _ArtifactReferenceSnapshot(
        values, session=session, connection=connection, driver=driver, referrers=referrers
    )


def validate_complete_reference_snapshot(
    session: Session,
    references: object,
) -> _ArtifactReferenceSnapshot:
    if (
        not isinstance(references, _ArtifactReferenceSnapshot)
        or references._coverage is not _REFERENCE_GRAPH_COVERAGE
        or not references.current_identity(session)
    ):
        raise ArtifactDeletionProofError(
            "artifact operation requires this transaction's complete reference graph"
        )
    if references._changes != references._driver.total_changes:
        raise ArtifactDeletionProofError("artifact reference data changed after the snapshot")
    return references


def mint_artifact_deletion_proof(
    session: Session,
    candidate_ids: Iterable[str],
    references: AbstractSet[str],
) -> object:
    transaction = session.get_transaction()
    references = validate_complete_reference_snapshot(session, references)
    candidates = frozenset(candidate_ids)
    if not candidates:
        raise ArtifactDeletionProofError("artifact deletion proof cannot be empty")
    if candidates & references:
        raise ArtifactDeletionProofError("artifact deletion proof includes a retained artifact")
    if any(
        references._referrers.get(artifact_id, frozenset()) - references._removed
        for artifact_id in candidates
    ):
        raise ArtifactDeletionProofError("This artifact is still retained by metadata.")
    return _ArtifactDeletionProof(
        session=session,
        transaction=transaction,
        candidate_ids=candidates,
        references=references,
        seal=_PROOF_SEAL,
    )


def validate_artifact_deletion_proof(
    session: Session,
    proof: object,
    candidate_ids: AbstractSet[str],
) -> None:
    if not isinstance(proof, _ArtifactDeletionProof) or proof.seal is not _PROOF_SEAL:
        raise ArtifactDeletionProofError("artifact deletion proof is invalid")
    if proof.session is not session:
        raise ArtifactDeletionProofError("artifact deletion proof belongs to another session")
    if proof.transaction is not session.get_transaction():
        raise ArtifactDeletionProofError("artifact deletion proof belongs to another transaction")
    validate_complete_reference_snapshot(session, proof.references)
    if proof.candidate_ids != frozenset(candidate_ids):
        raise ArtifactDeletionProofError("artifact deletion proof does not match the deletion set")


def restrict_artifact_deletion_proof(
    session: Session,
    proof: object,
    candidate_ids: AbstractSet[str],
) -> object:
    if not isinstance(proof, _ArtifactDeletionProof) or proof.seal is not _PROOF_SEAL:
        raise ArtifactDeletionProofError("artifact deletion proof is invalid")
    validate_artifact_deletion_proof(session, proof, proof.candidate_ids)
    requested = frozenset(candidate_ids)
    if not requested or not requested <= proof.candidate_ids:
        raise ArtifactDeletionProofError("artifact deletion proof does not cover the deletion set")
    return _ArtifactDeletionProof(
        session=session,
        transaction=proof.transaction,
        candidate_ids=requested,
        references=proof.references,
        seal=_PROOF_SEAL,
    )


def active_artifact_deletion_proof(session: Session) -> object | None:
    return session.info.get(_CONNECTION_STATE)


@contextmanager
def activate_artifact_deletion_proof(
    session: Session,
    proof: object,
) -> Iterator[None]:
    if not isinstance(proof, _ArtifactDeletionProof):
        raise ArtifactDeletionProofError("artifact deletion proof is invalid")
    validate_artifact_deletion_proof(session, proof, proof.candidate_ids)
    if session.new or session.dirty:
        raise ArtifactDeletionProofError("artifact reference data changed in pending writes")
    connection = session.connection()
    state = connection.connection.info.get(_CONNECTION_STATE)
    if not isinstance(state, _ConnectionAuthority):
        raise ArtifactDeletionProofError("artifact deletion authority is unavailable")
    if state.active is not None or _CONNECTION_STATE in session.info:
        raise ArtifactDeletionProofError("artifact deletion proof is already active")
    state.active = proof
    session.info[_CONNECTION_STATE] = proof
    try:
        yield
    finally:
        if state.active is proof:
            state.active = None
        if session.info.get(_CONNECTION_STATE) is proof:
            session.info.pop(_CONNECTION_STATE, None)


def artifact_deletion_proof_references(proof: object) -> AbstractSet[str]:
    if not isinstance(proof, _ArtifactDeletionProof) or proof.seal is not _PROOF_SEAL:
        raise ArtifactDeletionProofError("artifact deletion proof is invalid")
    return proof.references


def record_artifact_deletion(session: Session, proof: object) -> None:
    """Advance a snapshot after its exact, otherwise clean deletion flush.

    Other writes expire the snapshot before the next deletion. The completed
    deletion can only remove reference edges; record its ids and SQLite change
    counter after its cascading removals and nullable foreign keys have settled.
    """
    if not isinstance(proof, _ArtifactDeletionProof) or proof.seal is not _PROOF_SEAL:
        raise ArtifactDeletionProofError("artifact deletion proof is invalid")
    snapshot = proof.references
    if not snapshot.current_identity(session):
        raise ArtifactDeletionProofError("artifact deletion transaction changed")
    snapshot._removed.update(proof.candidate_ids)
    snapshot._changes = snapshot._driver.total_changes
