from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import threading
import uuid
from pathlib import Path

import pytest

from local_lm.shared_asset_registry_v1 import (
    INVALID_REGISTRY,
    SharedAssetRegistryError,
    claims_for_consumer,
    finalize_claim,
    package_is_claimed,
    release_claim,
    reserve_claim,
)


def _consumer(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _digest(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def test_reserve_finalize_release_is_the_claim_lifecycle(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    consumer = _consumer("profile-a")
    digest = _digest("package-1")
    reserved = reserve_claim(database=database, consumer_id=consumer, package_digest=digest)
    assert reserved.state == "provisional"
    assert package_is_claimed(database=database, package_digest=digest)
    finalized = finalize_claim(database=database, consumer_id=consumer, claim_id=reserved.claim_id)
    assert finalized.state == "final"
    assert finalized.claim_id == reserved.claim_id
    release_claim(database=database, consumer_id=consumer, claim_id=reserved.claim_id)
    assert not package_is_claimed(database=database, package_digest=digest)
    assert claims_for_consumer(database=database, consumer_id=consumer) == []


def test_reserve_is_idempotent_and_never_demotes(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    consumer = _consumer("profile-a")
    digest = _digest("package-1")
    first = reserve_claim(database=database, consumer_id=consumer, package_digest=digest)
    again = reserve_claim(database=database, consumer_id=consumer, package_digest=digest)
    assert again == first
    finalize_claim(database=database, consumer_id=consumer, claim_id=first.claim_id)
    retried = reserve_claim(database=database, consumer_id=consumer, package_digest=digest)
    assert retried.state == "final"
    assert retried.claim_id == first.claim_id
    twice = finalize_claim(database=database, consumer_id=consumer, claim_id=first.claim_id)
    assert twice.state == "final"


def test_one_consumer_never_sees_or_moves_anothers_claims(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    alpha = _consumer("profile-a")
    beta = _consumer("profile-b")
    digest = _digest("package-1")
    held = reserve_claim(database=database, consumer_id=alpha, package_digest=digest)
    assert claims_for_consumer(database=database, consumer_id=beta) == []
    with pytest.raises(SharedAssetRegistryError):
        finalize_claim(database=database, consumer_id=beta, claim_id=held.claim_id)
    with pytest.raises(SharedAssetRegistryError):
        release_claim(database=database, consumer_id=beta, claim_id=held.claim_id)
    assert package_is_claimed(database=database, package_digest=digest)
    # A FINALIZED foreign claim must refuse too: returning it would leak the
    # other consumer's membership through the finalize entry point.
    finalize_claim(database=database, consumer_id=alpha, claim_id=held.claim_id)
    with pytest.raises(SharedAssetRegistryError):
        finalize_claim(database=database, consumer_id=beta, claim_id=held.claim_id)
    # The cross-consumer view is exactly one bit.
    mine = claims_for_consumer(database=database, consumer_id=alpha)
    assert [claim.package_digest for claim in mine] == [digest]


def test_deletion_gate_holds_until_the_last_claim_releases(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    alpha = _consumer("profile-a")
    beta = _consumer("profile-b")
    digest = _digest("package-1")
    first = reserve_claim(database=database, consumer_id=alpha, package_digest=digest)
    second = reserve_claim(database=database, consumer_id=beta, package_digest=digest)
    release_claim(database=database, consumer_id=alpha, claim_id=first.claim_id)
    assert package_is_claimed(database=database, package_digest=digest)
    release_claim(database=database, consumer_id=beta, claim_id=second.claim_id)
    assert not package_is_claimed(database=database, package_digest=digest)


def test_releasing_a_claim_you_do_not_hold_refuses(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    consumer = _consumer("profile-a")
    with pytest.raises(SharedAssetRegistryError) as caught:
        release_claim(database=database, consumer_id=consumer, claim_id=uuid.uuid4().hex)
    # The literal, not the imported constant: the refusal is a fixed public
    # contract and must not drift with the module.
    assert str(caught.value) == "shared asset registry is invalid"
    assert INVALID_REGISTRY == "shared asset registry is invalid"


@pytest.mark.parametrize(
    ("consumer", "digest"),
    [
        ("", _digest("x")),
        ("not-hex", _digest("x")),
        (_consumer("a").upper(), _digest("x")),
        (_consumer("a"), ""),
        (_consumer("a"), "shortdigest"),
        (_consumer("a"), _digest("x").upper()),
    ],
)
def test_hostile_identifiers_are_refused(tmp_path: Path, consumer: str, digest: str) -> None:
    database = tmp_path / "index.sqlite3"
    with pytest.raises(SharedAssetRegistryError):
        reserve_claim(database=database, consumer_id=consumer, package_digest=digest)


@pytest.mark.parametrize(
    "database",
    [Path("relative/index.sqlite3"), Path("//server/share/index.sqlite3")],
)
def test_hostile_database_paths_are_refused(database: Path) -> None:
    with pytest.raises(SharedAssetRegistryError):
        claims_for_consumer(database=database, consumer_id=_consumer("a"))


def _tables_of(database: Path) -> set[str]:
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        connection.close()
    return {row[0] for row in rows}


def test_a_foreign_database_is_refused_byte_preserved(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO registry_meta (key, value) VALUES ('schema', 'foreign-schema')"
        )
    connection.close()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    with pytest.raises(SharedAssetRegistryError):
        package_is_claimed(database=database, package_digest=_digest("x"))
    # Refusal is byte-preserving: no table appears, no byte changes.
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert _tables_of(database) == {"registry_meta"}


def test_a_counterfeit_same_marker_schema_is_refused_byte_preserved(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite3"
    connection = sqlite3.connect(database)
    with connection:
        connection.execute(f"PRAGMA application_id={0x4C4D4153}")
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT"
        )
        connection.execute(
            "INSERT INTO registry_meta (key, value) VALUES "
            "('schema', 'lm-atelier-shared-asset-registry-v1')"
        )
        # The marker matches but the claims table is unconstrained.
        connection.execute(
            "CREATE TABLE package_claims (claim_id TEXT, consumer_id TEXT,"
            " package_digest TEXT, state TEXT)"
        )
        connection.execute(
            "INSERT INTO package_claims VALUES ('x', ?, ?, 'counterfeit')",
            (_consumer("mallory"), _digest("p")),
        )
    connection.close()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    with pytest.raises(SharedAssetRegistryError):
        claims_for_consumer(database=database, consumer_id=_consumer("mallory"))
    with pytest.raises(SharedAssetRegistryError):
        reserve_claim(database=database, consumer_id=_consumer("a"), package_digest=_digest("p"))
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "claims_sql",
    [
        # Missing the state CHECK constraint.
        "CREATE TABLE package_claims (claim_id TEXT PRIMARY KEY, consumer_id TEXT"
        " NOT NULL, package_digest TEXT NOT NULL, state TEXT NOT NULL,"
        " UNIQUE (consumer_id, package_digest)) STRICT",
        # Missing the per-consumer uniqueness.
        "CREATE TABLE package_claims (claim_id TEXT PRIMARY KEY, consumer_id TEXT"
        " NOT NULL, package_digest TEXT NOT NULL, state TEXT NOT NULL CHECK"
        " (state IN ('provisional', 'final'))) STRICT",
        # Not STRICT.
        "CREATE TABLE package_claims (claim_id TEXT PRIMARY KEY, consumer_id TEXT"
        " NOT NULL, package_digest TEXT NOT NULL, state TEXT NOT NULL CHECK"
        " (state IN ('provisional', 'final')), UNIQUE (consumer_id, package_digest))",
    ],
)
def test_weakened_constraints_under_the_right_marker_are_refused(
    tmp_path: Path, claims_sql: str
) -> None:
    database = tmp_path / "index.sqlite3"
    connection = sqlite3.connect(database)
    with connection:
        connection.execute(f"PRAGMA application_id={0x4C4D4153}")
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT"
        )
        connection.execute(
            "INSERT INTO registry_meta (key, value) VALUES "
            "('schema', 'lm-atelier-shared-asset-registry-v1')"
        )
        connection.execute(claims_sql)
    connection.close()
    with pytest.raises(SharedAssetRegistryError):
        claims_for_consumer(database=database, consumer_id=_consumer("a"))


def test_a_missing_application_identity_is_refused(tmp_path: Path) -> None:
    genuine = tmp_path / "genuine.sqlite3"
    reserve_claim(database=genuine, consumer_id=_consumer("a"), package_digest=_digest("p"))
    copied = tmp_path / "index.sqlite3"
    copied.write_bytes(genuine.read_bytes())
    connection = sqlite3.connect(copied)
    connection.execute("PRAGMA application_id=0")
    connection.close()
    with pytest.raises(SharedAssetRegistryError):
        claims_for_consumer(database=copied, consumer_id=_consumer("a"))


@pytest.mark.parametrize(
    "stowaway_sql",
    [
        "CREATE TABLE stowaway (x TEXT)",
        "CREATE VIEW stowaway_view AS SELECT 1 AS x",
    ],
)
def test_an_extra_foreign_object_inside_a_valid_registry_is_refused(
    tmp_path: Path, stowaway_sql: str
) -> None:
    database = tmp_path / "index.sqlite3"
    reserve_claim(database=database, consumer_id=_consumer("a"), package_digest=_digest("p"))
    connection = sqlite3.connect(database)
    with connection:
        connection.execute(stowaway_sql)
    connection.close()
    with pytest.raises(SharedAssetRegistryError):
        claims_for_consumer(database=database, consumer_id=_consumer("a"))


def test_a_tampered_marker_on_an_otherwise_exact_registry_is_refused(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite3"
    reserve_claim(database=database, consumer_id=_consumer("a"), package_digest=_digest("p"))
    connection = sqlite3.connect(database)
    with connection:
        connection.execute(
            "UPDATE registry_meta SET value = 'foreign-take-over' WHERE key = 'schema'"
        )
    connection.close()
    with pytest.raises(SharedAssetRegistryError):
        package_is_claimed(database=database, package_digest=_digest("p"))


def test_creation_stamps_identity_and_wal(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    reserve_claim(database=database, consumer_id=_consumer("a"), package_digest=_digest("p"))
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0x4C4D4153
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()


def test_a_foreign_schema_marker_refuses_rather_than_adopting(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO registry_meta (key, value) VALUES ('schema', 'other-app-v9')"
        )
    connection.close()
    with pytest.raises(SharedAssetRegistryError):
        reserve_claim(
            database=database,
            consumer_id=_consumer("a"),
            package_digest=_digest("x"),
        )


def test_a_corrupt_database_file_refuses_with_the_fixed_message(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite3"
    database.write_bytes(b"this is not a sqlite database, honest")
    with pytest.raises(SharedAssetRegistryError) as caught:
        claims_for_consumer(database=database, consumer_id=_consumer("a"))
    assert str(caught.value) == "shared asset registry is invalid"


def test_concurrent_reservers_of_one_package_converge_per_consumer(
    tmp_path: Path,
) -> None:
    database = tmp_path / "index.sqlite3"
    consumer = _consumer("profile-a")
    digest = _digest("package-1")
    results: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(4, timeout=30)

    def run() -> None:
        try:
            barrier.wait()
            claim = reserve_claim(database=database, consumer_id=consumer, package_digest=digest)
            results.append(claim.claim_id)
        except BaseException as caught:  # noqa: BLE001 - surfaced below
            errors.append(caught)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert len(set(results)) == 1
    assert len(claims_for_consumer(database=database, consumer_id=consumer)) == 1


def _make_link_dir(link: Path, target: Path) -> bool:
    """Point `link` at `target`, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def test_a_junctioned_parent_is_refused_and_the_foreign_directory_gains_nothing(
    tmp_path: Path,
) -> None:
    """A junctioned parent is refused and the foreign directory gains nothing.

    This is kept as a standing control. An earlier implementation accepted
    it: reserve_claim on <junction>/index.sqlite3 returned WITHOUT an
    exception and the foreign directory gained index.sqlite3. No race was
    needed - the parent was a junction before the call, and a syntax-only
    path check walked straight through it.
    """

    outside = tmp_path / "outside"
    outside.mkdir()
    redirect = tmp_path / "library"
    if not _make_link_dir(redirect, outside):
        pytest.skip("this host does not allow directory links")

    with pytest.raises(SharedAssetRegistryError) as refusal:
        reserve_claim(
            database=redirect / "index.sqlite3",
            consumer_id=_consumer("profile-a"),
            package_digest=_digest("package-1"),
        )

    assert str(refusal.value) == INVALID_REGISTRY
    assert list(outside.iterdir()) == []


def test_every_entry_point_refuses_a_junctioned_parent(tmp_path: Path) -> None:
    """Creation is not the only door into the directory.

    reserve_claim is the one that creates, so it is the one the original
    probe used. The others open the same database by the same path, and a
    successor that anchored only the creating path would leave five ways in.
    """

    outside = tmp_path / "outside"
    outside.mkdir()
    redirect = tmp_path / "library"
    if not _make_link_dir(redirect, outside):
        pytest.skip("this host does not allow directory links")

    database = redirect / "index.sqlite3"
    consumer = _consumer("profile-a")
    claim = uuid.uuid4().hex
    calls = (
        lambda: reserve_claim(database=database, consumer_id=consumer, package_digest=_digest("p")),
        lambda: finalize_claim(database=database, consumer_id=consumer, claim_id=claim),
        lambda: release_claim(database=database, consumer_id=consumer, claim_id=claim),
        lambda: claims_for_consumer(database=database, consumer_id=consumer),
        lambda: package_is_claimed(database=database, package_digest=_digest("p")),
    )
    for call in calls:
        with pytest.raises(SharedAssetRegistryError):
            call()
    assert list(outside.iterdir()) == []


def test_a_link_planted_under_the_registry_name_is_refused_not_followed(
    tmp_path: Path,
) -> None:
    """The leaf is an entry too, and a link there redirects the database.

    The parent walk says nothing about the last component: it is opened
    through the anchor rather than walked, so the refusal has to come from
    opening the entry itself.
    """

    outside = tmp_path / "outside"
    outside.mkdir()
    library = tmp_path / "library"
    library.mkdir()
    if not _make_link_dir(library / "index.sqlite3", outside):
        pytest.skip("this host does not allow directory links")

    with pytest.raises(SharedAssetRegistryError):
        reserve_claim(
            database=library / "index.sqlite3",
            consumer_id=_consumer("profile-a"),
            package_digest=_digest("package-1"),
        )
    assert list(outside.iterdir()) == []


def test_a_missing_parent_directory_is_refused_rather_than_created(tmp_path: Path) -> None:
    """The registry creates a database, never the directory that holds it.

    Creating an absent ancestor would mean deciding by path that a directory
    the caller never named should exist, which is how the predecessor's
    staging reached a directory nothing had verified.
    """

    with pytest.raises(SharedAssetRegistryError):
        reserve_claim(
            database=tmp_path / "absent" / "index.sqlite3",
            consumer_id=_consumer("profile-a"),
            package_digest=_digest("package-1"),
        )
    assert not (tmp_path / "absent").exists()


def test_no_staging_entry_survives_a_successful_creation(tmp_path: Path) -> None:
    """Publication leaves the directory holding exactly the registry."""

    database = tmp_path / "index.sqlite3"
    reserve_claim(
        database=database,
        consumer_id=_consumer("profile-a"),
        package_digest=_digest("package-1"),
    )
    assert [entry.name for entry in tmp_path.iterdir()] == ["index.sqlite3"]


def test_the_published_database_is_complete_from_the_first_byte(tmp_path: Path) -> None:
    """A reader never sees a half-built registry.

    The predecessor built the database in place at a sibling path and linked
    it afterwards, so the bytes existed in the shared directory while SQLite
    was still writing them. Here the finished image is written to a staging
    entry and published whole, so the published name is complete the moment
    it exists.
    """

    database = tmp_path / "index.sqlite3"
    reserve_claim(
        database=database,
        consumer_id=_consumer("profile-a"),
        package_digest=_digest("package-1"),
    )
    read_only = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        assert read_only.execute("PRAGMA application_id").fetchone()[0] == 0x4C4D4153
        assert read_only.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        read_only.close()


def test_a_symlink_to_a_valid_registry_is_refused_rather_than_written_through(
    tmp_path: Path,
) -> None:
    """The leaf guard, measured where validation cannot cover for it.

    A directory junction under the registry name is refused twice over: the
    entry guard declines to follow it, and a directory is not a database
    anyway. A symlink to a REAL registry is the case that separates them. The
    target validates - it is genuinely ours, with the right application id,
    schema and tables - so every check downstream of the entry open says yes,
    and the writable connection lands in a database belonging to somebody
    else. Only refusing to follow the link stops it.
    """

    outside = tmp_path / "outside"
    outside.mkdir()
    foreign = outside / "their.sqlite3"
    owner = _consumer("their-profile")
    reserve_claim(database=foreign, consumer_id=owner, package_digest=_digest("theirs"))
    before = foreign.read_bytes()

    library = tmp_path / "library"
    library.mkdir()
    try:
        os.symlink(foreign, library / "index.sqlite3")
    except OSError:
        pytest.skip("this host does not allow file symlinks")

    with pytest.raises(SharedAssetRegistryError) as refusal:
        reserve_claim(
            database=library / "index.sqlite3",
            consumer_id=_consumer("profile-a"),
            package_digest=_digest("package-1"),
        )

    assert str(refusal.value) == INVALID_REGISTRY
    assert foreign.read_bytes() == before
    assert claims_for_consumer(database=foreign, consumer_id=owner) != []
    assert not package_is_claimed(database=foreign, package_digest=_digest("package-1"))


@pytest.mark.parametrize("holder", ["has#hash", "has%percent", "plain", "space and #2"])
def test_a_uri_significant_character_in_the_path_is_data_not_syntax(
    tmp_path: Path, holder: str
) -> None:
    r"""A registry under a directory whose name contains "#" has to work.

    Measured before this was fixed: the file was CREATED - creation goes
    through the anchor and never builds a URI - and then every call refused,
    because validation dropped the path into `file:...?mode=ro` unescaped and
    "#" started a URI fragment. A registry on disk that nothing can open, with
    a fixed refusal message and no way forward.

    The library root is chosen by the user, and `C:\My Stuff #2\library` is an
    unremarkable thing to pick.
    """

    library = tmp_path / holder
    library.mkdir()
    database = library / "index.sqlite3"
    consumer = _consumer("profile-a")
    digest = _digest("package-1")

    reserved = reserve_claim(database=database, consumer_id=consumer, package_digest=digest)
    assert package_is_claimed(database=database, package_digest=digest)
    finalize_claim(database=database, consumer_id=consumer, claim_id=reserved.claim_id)
    assert claims_for_consumer(database=database, consumer_id=consumer) != []
    release_claim(database=database, consumer_id=consumer, claim_id=reserved.claim_id)
    assert not package_is_claimed(database=database, package_digest=digest)


def _plant_row(
    database: Path,
    *,
    claim_id: str,
    consumer_id: str,
    package_digest: str,
    state: str = "provisional",
) -> None:
    """Write a row the public API could never have written.

    `ignore_check_constraints` is the point of this helper: it is exactly how
    a hostile or careless writer gets past the CHECK constraints the schema
    carries, so a fixture that only used a table without them would be testing
    a weaker attacker than the real one.
    """

    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(
            "INSERT INTO package_claims (claim_id, consumer_id, package_digest, state)"
            " VALUES (?, ?, ?, ?)",
            (claim_id, consumer_id, package_digest, state),
        )
        connection.commit()
    finally:
        connection.close()


def _seeded_registry(tmp_path: Path) -> tuple[Path, str, str]:
    database = tmp_path / "index.sqlite3"
    consumer = _consumer("profile-a")
    digest = _digest("package-1")
    reserve_claim(database=database, consumer_id=consumer, package_digest=digest)
    return database, consumer, digest


MALFORMED_ROWS = [
    ("claim id is not hex", "not-a-hex-claim-id", _consumer("x"), _digest("y")),
    ("claim id is the wrong length", "a" * 31, _consumer("x"), _digest("y")),
    ("claim id is uppercase", "A" * 32, _consumer("x"), _digest("y")),
    ("consumer id is too short", "b" * 32, "c" * 31, _digest("y")),
    ("consumer id is uppercase", "b" * 32, "C" * 32, _digest("y")),
    ("digest is the wrong length", "b" * 32, _consumer("x"), "d" * 63),
    ("digest is uppercase", "b" * 32, _consumer("x"), "D" * 64),
    ("digest is not hex", "b" * 32, _consumer("x"), "z" * 64),
]


@pytest.mark.parametrize(
    ("label", "claim_id", "consumer_id", "package_digest"),
    MALFORMED_ROWS,
    ids=[row[0] for row in MALFORMED_ROWS],
)
def test_every_entry_point_refuses_a_registry_holding_a_malformed_claim(
    tmp_path: Path, label: str, claim_id: str, consumer_id: str, package_digest: str
) -> None:
    """Every entry point refuses a registry holding a malformed claim.

    This is kept as a standing control. The exact schema constrained
    nullability, state and per-consumer uniqueness, and left the three
    identifier formats to the entry points. So
    an otherwise exact registry could hold a normal SQL row with a valid
    consumer and state and a claim id no public call could have produced.
    claims_for_consumer handed it back as a PackageClaim while finalize_claim
    and release_claim correctly refused its format: a claim that is accepted
    and can never be addressed.

    That is worse than untidy, because a claim IS the deletion authority. On a
    well-formed digest, an unaddressable claim blocks collection of those
    bytes permanently.
    """

    database, good_consumer, good_digest = _seeded_registry(tmp_path)
    _plant_row(
        database,
        claim_id=claim_id,
        consumer_id=consumer_id,
        package_digest=package_digest,
    )
    before = database.read_bytes()

    calls = (
        lambda: reserve_claim(
            database=database, consumer_id=good_consumer, package_digest=good_digest
        ),
        lambda: claims_for_consumer(database=database, consumer_id=good_consumer),
        lambda: package_is_claimed(database=database, package_digest=good_digest),
        lambda: finalize_claim(database=database, consumer_id=good_consumer, claim_id="b" * 32),
        lambda: release_claim(database=database, consumer_id=good_consumer, claim_id="b" * 32),
    )
    for call in calls:
        with pytest.raises(SharedAssetRegistryError) as refusal:
            call()
        assert str(refusal.value) == INVALID_REGISTRY

    assert database.read_bytes() == before


def test_the_schema_itself_refuses_a_malformed_claim_without_the_bypass(
    tmp_path: Path,
) -> None:
    """The row check is the second line, not the only one.

    An ordinary writer - one that does not reach for
    ignore_check_constraints - cannot create the row at all, because the
    formats are database constraints now rather than a rule the entry points
    remember.
    """

    database, _consumer_id, _digest_value = _seeded_registry(tmp_path)
    connection = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO package_claims (claim_id, consumer_id, package_digest, state)"
                " VALUES (?, ?, ?, ?)",
                ("not-a-hex-claim-id", _consumer("z"), _digest("z"), "provisional"),
            )
    finally:
        connection.close()


def test_a_registry_whose_claims_table_lacks_the_format_constraints_is_foreign(
    tmp_path: Path,
) -> None:
    """The predecessor's own schema is refused, by identity rather than by row.

    sqlite_master stores the literal CREATE text and validation compares it,
    so a database built with the older table is not merely permissive - it is
    not this schema. Worth pinning: it is what stops a downgrade attack that
    recreates the table without the constraints and then fills it freely.
    """

    database = tmp_path / "index.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute(f"PRAGMA application_id={0x4C4D4153}")
        connection.execute("PRAGMA user_version=1")
        connection.execute(
            "CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT"
        )
        connection.execute(
            "CREATE TABLE package_claims (claim_id TEXT PRIMARY KEY, "
            "consumer_id TEXT NOT NULL, package_digest TEXT NOT NULL, "
            "state TEXT NOT NULL CHECK (state IN ('provisional', 'final')), "
            "UNIQUE (consumer_id, package_digest)) STRICT"
        )
        connection.execute(
            "INSERT INTO registry_meta (key, value) VALUES ('schema', ?)",
            ("lm-atelier-shared-asset-registry-v1",),
        )
        connection.commit()
    finally:
        connection.close()
    before = database.read_bytes()

    with pytest.raises(SharedAssetRegistryError) as refusal:
        package_is_claimed(database=database, package_digest=_digest("p"))
    assert str(refusal.value) == INVALID_REGISTRY
    assert database.read_bytes() == before


def _relax_then_restore_schema(database: Path, rows: list[tuple[object, ...]]) -> bytes:
    """Plant rows that the exact schema would never have accepted.

    This is how such a database actually arrives, and it is worth spelling out
    because NOT NULL looks like it should already prevent it. It does prevent
    every ordinary write: measured, a NULL is refused on all four columns even
    with `PRAGMA ignore_check_constraints=ON`, because NOT NULL is a column
    constraint rather than a CHECK, and STRICT makes the primary key NOT NULL
    too.

    What defeats it is rewriting the schema itself. `PRAGMA writable_schema`
    swaps the stored CREATE text for one without the constraints, the rows go
    in, and the exact text is put back. The rows survive the swap, and the text
    this module compares against sqlite_master is then byte-identical to ours -
    so schema identity says yes while the contents are unusable.

    Returns the exact CREATE text, so a caller can assert it was restored.
    """

    connection = sqlite3.connect(database)
    try:
        exact = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'package_claims'"
        ).fetchone()[0]
    finally:
        connection.close()

    def rewrite(sql: str, version: int) -> None:
        handle = sqlite3.connect(database)
        try:
            handle.execute("PRAGMA writable_schema=ON")
            handle.execute("UPDATE sqlite_master SET sql = ? WHERE name = 'package_claims'", (sql,))
            # The schema cookie must move or other connections keep the cached
            # definition and the planted rows never land.
            handle.execute(f"PRAGMA schema_version={version}")
            handle.commit()
        finally:
            handle.close()

    rewrite(
        "CREATE TABLE package_claims (claim_id TEXT PRIMARY KEY, consumer_id TEXT,"
        " package_digest TEXT, state TEXT, UNIQUE (consumer_id, package_digest))",
        99,
    )
    handle = sqlite3.connect(database)
    try:
        handle.executemany(
            "INSERT INTO package_claims (claim_id, consumer_id, package_digest, state)"
            " VALUES (?, ?, ?, ?)",
            rows,
        )
        handle.commit()
    finally:
        handle.close()
    rewrite(exact, 100)
    return exact.encode("utf-8")


NULL_ROWS = [
    ("every column null", ("a" * 32, None, None, None)),
    ("consumer id null", ("b" * 32, None, _digest("y"), "provisional")),
    ("digest null", ("c" * 32, _consumer("x"), None, "provisional")),
    ("state null", ("d" * 32, _consumer("x"), _digest("y"), None)),
]


@pytest.mark.parametrize(("label", "row"), NULL_ROWS, ids=[entry[0] for entry in NULL_ROWS])
def test_a_stored_null_does_not_escape_the_malformed_check(
    tmp_path: Path, label: str, row: tuple[object, ...]
) -> None:
    """A stored NULL does not escape the malformed-claim check.

    This is kept as a standing control. The check was
    `WHERE NOT (conditions)`. SQL is three-valued, so a NULL
    column made the comparison NULL, `NOT NULL` NULL, and a NULL WHERE clause
    matches nothing - the offending row counted as fine and the registry
    validated. COALESCE(..., 0) is what makes the predicate total.
    """

    database, consumer, digest = _seeded_registry(tmp_path)
    exact = _relax_then_restore_schema(database, [row])

    connection = sqlite3.connect(database)
    try:
        stored = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'package_claims'"
        ).fetchone()[0]
        assert stored.encode("utf-8") == exact, "the fixture must restore the exact schema"
    finally:
        connection.close()

    before = database.read_bytes()
    calls = (
        lambda: reserve_claim(database=database, consumer_id=consumer, package_digest=digest),
        lambda: claims_for_consumer(database=database, consumer_id=consumer),
        lambda: package_is_claimed(database=database, package_digest=digest),
        lambda: finalize_claim(database=database, consumer_id=consumer, claim_id="b" * 32),
        lambda: release_claim(database=database, consumer_id=consumer, claim_id="b" * 32),
    )
    for call in calls:
        with pytest.raises(SharedAssetRegistryError) as refusal:
            call()
        assert str(refusal.value) == INVALID_REGISTRY

    assert database.read_bytes() == before


def test_not_null_alone_does_not_stop_an_ordinary_writer_being_enough(
    tmp_path: Path,
) -> None:
    """Why the row check cannot lean on NOT NULL.

    Recorded as a measurement rather than an argument: every ordinary insert of
    a NULL is refused, including under the pragma that defeats CHECK. That is
    exactly why the fixture above has to rewrite the schema, and why the
    predicate still has to be total - the constraint holds for writers, not for
    databases that arrive already populated.
    """

    database, _consumer_id, _digest_value = _seeded_registry(tmp_path)
    for column_values in (
        (None, _consumer("x"), _digest("y"), "provisional"),
        ("b" * 32, None, _digest("y"), "provisional"),
        ("b" * 32, _consumer("x"), None, "provisional"),
        ("b" * 32, _consumer("x"), _digest("y"), None),
    ):
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA ignore_check_constraints=ON")
            with pytest.raises(sqlite3.IntegrityError, match="NOT NULL"):
                connection.execute(
                    "INSERT INTO package_claims"
                    " (claim_id, consumer_id, package_digest, state) VALUES (?, ?, ?, ?)",
                    column_values,
                )
        finally:
            connection.close()
