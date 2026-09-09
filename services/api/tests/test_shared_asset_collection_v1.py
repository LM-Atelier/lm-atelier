from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import local_lm.shared_asset_collection_v1 as collection
from local_lm.shared_asset_collection_v1 import (
    INVALID_COLLECTION,
    SharedAssetCollectionError,
    collect_unreferenced_object,
)
from local_lm.shared_asset_leases_v1 import (
    _release_all_for_testing,
    acquire_read_lease,
    release_read_lease,
)
from local_lm.shared_asset_package_v1 import publish_package
from local_lm.shared_asset_registry_v1 import (
    finalize_claim,
    release_claim,
    reserve_claim,
)
from local_lm.shared_asset_store_v1 import object_path, publish_file


def _consumer(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()[:32]


def _published(tmp_path: Path) -> tuple[Path, Path, str, Path]:
    root = tmp_path / "library"
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable shared bytes")
    digest = publish_file(root=root, source=source)
    return root, root / "index.sqlite3", digest, object_path(root=root, digest=digest)


def test_collects_one_verified_object_without_a_claim_or_lease(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)

    collect_unreferenced_object(root=root, database=database, package_digest=digest)

    assert not published.exists()
    assert database.is_file()


def test_any_claim_refuses_collection_without_exposing_its_owner(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)
    reserve_claim(database=database, consumer_id=_consumer("owner"), package_digest=digest)

    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=digest)

    assert published.read_bytes() == b"immutable shared bytes"


def test_a_lease_still_refuses_after_its_claim_releases(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)
    consumer = _consumer("owner")
    claim = reserve_claim(database=database, consumer_id=consumer, package_digest=digest)
    finalize_claim(database=database, consumer_id=consumer, claim_id=claim.claim_id)
    lease = acquire_read_lease(
        database=database,
        consumer_id=consumer,
        package_digest=digest,
        now=1,
        ttl=10,
    )
    release_claim(database=database, consumer_id=consumer, claim_id=claim.claim_id)
    try:
        with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
            collect_unreferenced_object(root=root, database=database, package_digest=digest)
        assert published.read_bytes() == b"immutable shared bytes"
    finally:
        release_read_lease(database=database, consumer_id=consumer, lease_id=lease.lease_id)
        _release_all_for_testing()


def test_missing_and_digest_drift_both_refuse(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)
    missing = "f" * 64

    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=missing)

    published.write_bytes(b"different bytes")
    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=digest)
    assert published.read_bytes() == b"different bytes"


def test_registry_must_be_the_exact_database_inside_the_store(tmp_path: Path) -> None:
    root, _database, digest, published = _published(tmp_path)

    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(
            root=root,
            database=tmp_path / "other" / "index.sqlite3",
            package_digest=digest,
        )

    assert published.is_file()


def test_a_link_in_the_object_slot_is_refused_without_following_it(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    published.unlink()
    try:
        os.symlink(outside, published)
    except OSError:
        pytest.skip("this host does not allow file links")

    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=digest)

    assert outside.read_bytes() == b"outside"
    assert published.is_symlink()


def test_replacement_before_the_anchored_rename_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, database, digest, published = _published(tmp_path)
    original = published.parent / "original-away"
    real_rename = collection.rename_entry

    def replace_then_rename(anchor, source: str, destination: str, *, replace: bool) -> None:
        if source == digest:
            published.rename(original)
            published.write_bytes(b"replacement")
        real_rename(anchor, source, destination, replace=replace)

    monkeypatch.setattr(collection, "rename_entry", replace_then_rename)

    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=digest)

    assert original.read_bytes() == b"immutable shared bytes"
    assert published.read_bytes() == b"replacement"


def test_package_claim_protects_each_member(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)
    package = publish_package(root=root, members={"unet": digest})
    reserve_claim(database=database, consumer_id=_consumer("owner"), package_digest=package)

    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=digest)

    assert published.read_bytes() == b"immutable shared bytes"


def test_package_lease_protects_members_after_claim_release(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)
    package = publish_package(root=root, members={"unet": digest})
    consumer = _consumer("owner")
    claim = reserve_claim(database=database, consumer_id=consumer, package_digest=package)
    finalize_claim(database=database, consumer_id=consumer, claim_id=claim.claim_id)
    lease = acquire_read_lease(
        database=database, consumer_id=consumer, package_digest=package, now=1, ttl=10
    )
    release_claim(database=database, consumer_id=consumer, claim_id=claim.claim_id)
    try:
        with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
            collect_unreferenced_object(root=root, database=database, package_digest=digest)
        assert published.read_bytes() == b"immutable shared bytes"
    finally:
        release_read_lease(database=database, consumer_id=consumer, lease_id=lease.lease_id)
        _release_all_for_testing()


def test_member_collects_only_after_the_last_containing_package_releases(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)
    first_package = publish_package(root=root, members={"unet": digest})
    second_package = publish_package(root=root, members={"lora": digest})
    first = reserve_claim(
        database=database, consumer_id=_consumer("first"), package_digest=first_package
    )
    second = reserve_claim(
        database=database, consumer_id=_consumer("second"), package_digest=second_package
    )
    release_claim(database=database, consumer_id=first.consumer_id, claim_id=first.claim_id)
    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=digest)
    assert published.is_file()

    release_claim(database=database, consumer_id=second.consumer_id, claim_id=second.claim_id)
    collect_unreferenced_object(root=root, database=database, package_digest=digest)
    assert not published.exists()


def test_a_claim_on_an_outer_package_protects_nested_members(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)
    inner = publish_package(root=root, members={"unet": digest})
    outer = publish_package(root=root, members={"unet": inner})
    reserve_claim(database=database, consumer_id=_consumer("owner"), package_digest=outer)

    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=digest)

    assert published.is_file()


def test_legacy_package_holds_block_until_membership_is_verified(tmp_path: Path) -> None:
    root, database, digest, published = _published(tmp_path)
    source = tmp_path / "legacy-package.json"
    source.write_text(
        json.dumps(
            {
                "members": {"unet": digest},
                "schema": "lm-atelier-shared-asset-package-v1",
                "version": 1,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="ascii",
    )
    package = publish_file(root=root, source=source)
    reserve_claim(database=database, consumer_id=_consumer("owner"), package_digest=package)
    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=digest)
    assert published.is_file()

    source.write_bytes(b"unrelated unclaimed bytes")
    unrelated = publish_file(root=root, source=source)
    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=unrelated)

    assert publish_package(root=root, members={"unet": digest}) == package
    collect_unreferenced_object(root=root, database=database, package_digest=unrelated)
    assert not object_path(root=root, digest=unrelated).exists()
    with pytest.raises(SharedAssetCollectionError, match=INVALID_COLLECTION):
        collect_unreferenced_object(root=root, database=database, package_digest=digest)
    assert published.is_file()
