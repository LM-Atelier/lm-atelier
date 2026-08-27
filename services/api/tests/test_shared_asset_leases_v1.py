from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from local_lm.shared_asset_leases_v1 import (
    INVALID_LEASE,
    SharedAssetLeaseError,
    _release_all_for_testing,
    acquire_read_lease,
    leases_for_consumer,
    package_has_read_lease,
    release_read_lease,
    renew_read_lease,
    steal_expired_read_lease,
)
from local_lm.shared_asset_registry_v1 import (
    claims_for_consumer,
    finalize_claim,
    reserve_claim,
)


def _consumer(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


@pytest.fixture(autouse=True)
def _release_process_locks() -> None:
    _release_all_for_testing()
    yield
    _release_all_for_testing()


def _ready(
    tmp_path: Path, *, consumer: str | None = None, digest: str | None = None
) -> tuple[Path, str, str]:
    database = tmp_path / "index.sqlite3"
    owner = consumer or _consumer("profile-a")
    package = digest or _digest("package-1")
    claim = reserve_claim(database=database, consumer_id=owner, package_digest=package)
    finalize_claim(database=database, consumer_id=owner, claim_id=claim.claim_id)
    return database, owner, package


_ACQUIRE_CHILD = """
import sys
from pathlib import Path

from local_lm.shared_asset_leases_v1 import SharedAssetLeaseError, acquire_read_lease

try:
    acquire_read_lease(
        database=Path(sys.argv[1]),
        consumer_id=sys.argv[2],
        package_digest=sys.argv[3],
        now=50,
        ttl=9,
    )
except SharedAssetLeaseError:
    print("refused")
else:
    print("acquired")
"""


def _acquire_in_child(
    database: Path, consumer: str, digest: str
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return subprocess.run(
        [sys.executable, "-c", _ACQUIRE_CHILD, str(database), consumer, digest],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )


def test_two_consumers_hold_scoped_leases_on_one_package(tmp_path: Path) -> None:
    database, alpha, digest = _ready(tmp_path)
    beta = _consumer("profile-b")
    beta_claim = reserve_claim(database=database, consumer_id=beta, package_digest=digest)
    finalize_claim(database=database, consumer_id=beta, claim_id=beta_claim.claim_id)

    first = acquire_read_lease(
        database=database, consumer_id=alpha, package_digest=digest, now=10, ttl=5
    )
    second = acquire_read_lease(
        database=database, consumer_id=beta, package_digest=digest, now=11, ttl=5
    )

    assert first.lease_id != second.lease_id
    assert leases_for_consumer(database=database, consumer_id=alpha) == (first,)
    assert leases_for_consumer(database=database, consumer_id=beta) == (second,)
    assert package_has_read_lease(database=database, package_digest=digest)

    release_read_lease(database=database, consumer_id=alpha, lease_id=first.lease_id)
    assert leases_for_consumer(database=database, consumer_id=alpha) == ()
    assert package_has_read_lease(database=database, package_digest=digest)
    release_read_lease(database=database, consumer_id=beta, lease_id=second.lease_id)
    assert not package_has_read_lease(database=database, package_digest=digest)


def test_acquire_is_idempotent_for_the_local_holder_and_renew_extends(
    tmp_path: Path,
) -> None:
    database, consumer, digest = _ready(tmp_path)
    first = acquire_read_lease(
        database=database, consumer_id=consumer, package_digest=digest, now=20, ttl=5
    )
    again = acquire_read_lease(
        database=database, consumer_id=consumer, package_digest=digest, now=21, ttl=7
    )
    renewed = renew_read_lease(
        database=database, consumer_id=consumer, lease_id=first.lease_id, now=30, ttl=9
    )

    assert again.lease_id == first.lease_id
    assert again.expires_at == 28
    assert renewed.lease_id == first.lease_id
    assert renewed.expires_at == 39
    assert leases_for_consumer(database=database, consumer_id=consumer) == (renewed,)


def test_another_process_cannot_reacquire_or_rewrite_a_live_lease(tmp_path: Path) -> None:
    database, consumer, digest = _ready(tmp_path)
    original = acquire_read_lease(
        database=database, consumer_id=consumer, package_digest=digest, now=1, ttl=5
    )

    child = _acquire_in_child(database, consumer, digest)

    assert child.stdout.strip() == "refused"
    assert child.stderr == ""
    assert leases_for_consumer(database=database, consumer_id=consumer) == (original,)


def test_a_provisional_or_missing_claim_cannot_take_a_lease(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    consumer = _consumer("profile-a")
    digest = _digest("package-1")
    reserve_claim(database=database, consumer_id=consumer, package_digest=digest)

    with pytest.raises(SharedAssetLeaseError, match=INVALID_LEASE):
        acquire_read_lease(database=database, consumer_id=consumer, package_digest=digest)
    with pytest.raises(SharedAssetLeaseError, match=INVALID_LEASE):
        acquire_read_lease(
            database=database,
            consumer_id=_consumer("profile-b"),
            package_digest=digest,
        )


def test_live_holder_refuses_expired_lease_reassignment(tmp_path: Path) -> None:
    database, alpha, digest = _ready(tmp_path)
    beta = _consumer("profile-b")
    beta_claim = reserve_claim(database=database, consumer_id=beta, package_digest=digest)
    finalize_claim(database=database, consumer_id=beta, claim_id=beta_claim.claim_id)
    original = acquire_read_lease(
        database=database, consumer_id=alpha, package_digest=digest, now=1, ttl=1
    )

    with pytest.raises(SharedAssetLeaseError, match=INVALID_LEASE):
        steal_expired_read_lease(
            database=database,
            consumer_id=beta,
            package_digest=digest,
            now=100,
            ttl=5,
        )

    assert leases_for_consumer(database=database, consumer_id=alpha) == (original,)
    assert leases_for_consumer(database=database, consumer_id=beta) == ()


def test_expired_lease_can_move_only_after_its_lock_is_released(tmp_path: Path) -> None:
    database, alpha, digest = _ready(tmp_path)
    beta = _consumer("profile-b")
    beta_claim = reserve_claim(database=database, consumer_id=beta, package_digest=digest)
    finalize_claim(database=database, consumer_id=beta, claim_id=beta_claim.claim_id)
    original = acquire_read_lease(
        database=database, consumer_id=alpha, package_digest=digest, now=1, ttl=1
    )
    _release_all_for_testing()

    moved = steal_expired_read_lease(
        database=database, consumer_id=beta, package_digest=digest, now=100, ttl=5
    )

    assert moved.lease_id == original.lease_id
    assert moved.expires_at == 105
    assert leases_for_consumer(database=database, consumer_id=alpha) == ()
    assert leases_for_consumer(database=database, consumer_id=beta) == (moved,)


def test_replacing_a_released_lock_entry_does_not_prove_the_holder_gone(
    tmp_path: Path,
) -> None:
    database, alpha, digest = _ready(tmp_path)
    beta = _consumer("profile-b")
    beta_claim = reserve_claim(database=database, consumer_id=beta, package_digest=digest)
    finalize_claim(database=database, consumer_id=beta, claim_id=beta_claim.claim_id)
    original = acquire_read_lease(
        database=database, consumer_id=alpha, package_digest=digest, now=1, ttl=1
    )
    _release_all_for_testing()
    entry = tmp_path / "locks" / f"lease-{original.lease_id}.lock"
    entry.unlink()
    entry.write_bytes(b"\0" + b"replacement-tokn")

    with pytest.raises(SharedAssetLeaseError, match=INVALID_LEASE):
        steal_expired_read_lease(
            database=database,
            consumer_id=beta,
            package_digest=digest,
            now=100,
            ttl=5,
        )

    assert leases_for_consumer(database=database, consumer_id=alpha) == (original,)


def test_exact_v1_registry_upgrades_without_moving_claims(tmp_path: Path) -> None:
    database, consumer, digest = _ready(tmp_path)
    claim_before = claims_for_consumer(database=database, consumer_id=consumer)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TABLE package_leases")
        connection.execute("PRAGMA user_version=1")
        connection.commit()
    finally:
        connection.close()

    lease = acquire_read_lease(
        database=database, consumer_id=consumer, package_digest=digest, now=10, ttl=5
    )

    assert claims_for_consumer(database=database, consumer_id=consumer) == claim_before
    assert leases_for_consumer(database=database, consumer_id=consumer) == (lease,)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert tables == {"registry_meta", "package_claims", "package_leases"}
    finally:
        connection.close()


def test_foreign_v1_shape_is_refused_without_an_upgrade_write(tmp_path: Path) -> None:
    database, consumer, digest = _ready(tmp_path)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TABLE package_leases")
        connection.execute("PRAGMA user_version=1")
        connection.execute("CREATE TABLE stowaway (value TEXT)")
        connection.commit()
    finally:
        connection.close()
    before = database.read_bytes()

    with pytest.raises(SharedAssetLeaseError, match=INVALID_LEASE):
        acquire_read_lease(
            database=database, consumer_id=consumer, package_digest=digest, now=10, ttl=5
        )

    assert database.read_bytes() == before
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'package_leases'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()
