from __future__ import annotations

from pathlib import Path

import pytest

from local_lm.shared_asset_claims_v1 import (
    INDEX_NAME,
    INVALID_CLAIM,
    SharedAssetClaimError,
    claim_count_for_digest,
    claim_object,
    claims_for_consumer,
    release_claim,
)
from local_lm.shared_asset_store_v1 import object_path, publish_file

FIRST_CONSUMER = "a" * 32
SECOND_CONSUMER = "b" * 32


def _published(tmp_path: Path, payload: bytes = b"claimed-bytes") -> tuple[Path, str]:
    root = tmp_path / "packages"
    source = tmp_path / "model.bin"
    source.write_bytes(payload)
    digest = publish_file(root=root, source=source)
    return root, digest


def test_two_consumers_claim_one_object_without_sharing_rows(tmp_path: Path) -> None:
    root, digest = _published(tmp_path)
    first = claim_object(root=root, consumer_id=FIRST_CONSUMER, digest=digest)
    again = claim_object(root=root, consumer_id=FIRST_CONSUMER, digest=digest)
    second = claim_object(root=root, consumer_id=SECOND_CONSUMER, digest=digest)
    assert first == again
    assert first != second
    assert claim_count_for_digest(root=root, digest=digest) == 2
    owned = claims_for_consumer(root=root, consumer_id=FIRST_CONSUMER)
    assert [claim.claim_id for claim in owned] == [first]
    assert [claim.digest for claim in owned] == [digest]
    other = claims_for_consumer(root=root, consumer_id=SECOND_CONSUMER)
    assert [claim.claim_id for claim in other] == [second]
    assert first not in {claim.claim_id for claim in other}


def test_release_drops_only_that_consumer_and_keeps_bytes(tmp_path: Path) -> None:
    root, digest = _published(tmp_path)
    first = claim_object(root=root, consumer_id=FIRST_CONSUMER, digest=digest)
    second = claim_object(root=root, consumer_id=SECOND_CONSUMER, digest=digest)
    stored = object_path(root=root, digest=digest)
    release_claim(root=root, consumer_id=FIRST_CONSUMER, claim_id=first)
    assert stored.is_file()
    assert stored.read_bytes() == b"claimed-bytes"
    assert claim_count_for_digest(root=root, digest=digest) == 1
    assert claims_for_consumer(root=root, consumer_id=FIRST_CONSUMER) == ()
    remaining = claims_for_consumer(root=root, consumer_id=SECOND_CONSUMER)
    assert [claim.claim_id for claim in remaining] == [second]
    with pytest.raises(SharedAssetClaimError, match=INVALID_CLAIM):
        release_claim(root=root, consumer_id=FIRST_CONSUMER, claim_id=second)


def test_reads_do_not_create_the_index(tmp_path: Path) -> None:
    root, digest = _published(tmp_path)
    assert claims_for_consumer(root=root, consumer_id=FIRST_CONSUMER) == ()
    assert claim_count_for_digest(root=root, digest=digest) == 0
    assert not (root / INDEX_NAME).exists()


def test_claim_refuses_missing_or_drifted_objects(tmp_path: Path) -> None:
    root, digest = _published(tmp_path)
    missing = "ab" * 32
    with pytest.raises(SharedAssetClaimError, match=INVALID_CLAIM):
        claim_object(root=root, consumer_id=FIRST_CONSUMER, digest=missing)
    stored = object_path(root=root, digest=digest)
    stored.write_bytes(b"corrupted")
    with pytest.raises(SharedAssetClaimError, match=INVALID_CLAIM):
        claim_object(root=root, consumer_id=FIRST_CONSUMER, digest=digest)


def test_claim_refuses_unc_relative_and_invalid_ids(tmp_path: Path) -> None:
    root, digest = _published(tmp_path)
    with pytest.raises(SharedAssetClaimError, match=INVALID_CLAIM):
        claim_object(
            root=Path(r"\\server\share\packages"),
            consumer_id=FIRST_CONSUMER,
            digest=digest,
        )
    with pytest.raises(SharedAssetClaimError, match=INVALID_CLAIM):
        claim_object(root=Path("relative-packages"), consumer_id=FIRST_CONSUMER, digest=digest)
    with pytest.raises(SharedAssetClaimError, match=INVALID_CLAIM):
        claim_object(root=root, consumer_id="not-a-token", digest=digest)
    with pytest.raises(SharedAssetClaimError, match=INVALID_CLAIM):
        claim_object(root=root, consumer_id=FIRST_CONSUMER, digest="../../evil")
    with pytest.raises(SharedAssetClaimError, match=INVALID_CLAIM):
        claims_for_consumer(root=root, consumer_id="C" * 32)
    with pytest.raises(SharedAssetClaimError, match=INVALID_CLAIM):
        release_claim(root=root, consumer_id=FIRST_CONSUMER, claim_id="d" * 32)
