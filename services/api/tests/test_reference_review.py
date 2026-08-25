"""The transition that makes a reference usable, and what it costs to claim it.

`test_promotion_requires_the_image_itself` is the one to read. `usable` is the
difference between a reference that conditions a generation and one that does
not, so it is the state most worth setting carelessly - and the reason
`UNCHECKED` exists as a separate value is that an unlooked-at image is not a
checked one. Promotion here reads and measures the image, so the state cannot
come to mean "someone clicked".

The end-to-end attach-review-bind loop closes once the conditioning binding
lands; its integration test rides with that foundation rather than importing
modules trunk does not have.
"""

from __future__ import annotations

import hashlib
import io
import warnings
import zlib
from collections.abc import Generator
from itertools import count
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import IntegrityError, PendingRollbackError
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.models import (
    Artifact,
    ReferenceAsset,
    ReferenceAssetReviewEvent,
    ReferenceSubject,
)
from local_lm.reference_review import (
    MIN_SHORT_EDGE,
    ReviewedAsset,
    ReviewOutcome,
    ReviewRefusal,
    ReviewRefused,
    _measure,
    _png_complete_and_consumed,
    review_asset,
)
from local_lm.references import ReferencePurpose, ValidationState


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _enforce(connection, _record):  # type: ignore[no-untyped-def]
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


_slugs = count(1)


def png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (128, 128, 128)).save(buffer, format="PNG")
    return buffer.getvalue()


def reader(payload: bytes):  # type: ignore[no-untyped-def]
    def read(_artifact_id: str) -> bytes:
        return payload

    return read


def subject(session: Session) -> ReferenceSubject:
    slug = f"subject-{next(_slugs)}"
    row = ReferenceSubject(name=slug, mention_slug=slug, kind="person")
    session.add(row)
    session.flush()
    return row


def asset(
    session: Session,
    owner: ReferenceSubject,
    seed: int,
    payload: bytes | None = None,
) -> ReferenceAsset:
    # A settling review proves the payload hash against the artifact row,
    # so a test that settles must seed the REAL digest of the bytes its
    # reader returns; refusal-path tests may keep the synthetic one.
    digest = hashlib.sha256(payload).hexdigest() if payload else f"{seed:064x}"
    session.add(
        Artifact(
            id=f"sha256:{digest}",
            sha256=digest,
            kind="image",
            media_type="image/png",
            size_bytes=1,
            relative_path=f"{seed}/a.png",
        )
    )
    row = ReferenceAsset(
        reference_subject_id=owner.id,
        artifact_id=f"sha256:{digest}",
        purpose=ReferencePurpose.IDENTITY.value,
        validation_state=ValidationState.UNCHECKED.value,
        sort_order=0,
    )
    session.add(row)
    session.flush()
    return row


def test_promotion_requires_the_image_itself(session: Session) -> None:
    """No reader, no promotion. The read is the evidence, not advice.

    This is the inverse of `attach_asset`, where a failed read must never block
    the attachment - and correctly so, because there the read only informs. Here
    it is the whole basis of the claim being made.
    """

    person = subject(session)
    image = asset(session, person, 1)

    result = review_asset(session, person, asset_id=image.id, outcome=ReviewOutcome.USABLE)

    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.NO_READER
    assert image.validation_state == ValidationState.UNCHECKED.value


def test_a_measured_image_is_promoted_and_its_size_recorded(session: Session) -> None:
    """The measurement is kept, on columns nothing had ever written."""

    person = subject(session)
    payload = png(1024, 768)
    image = asset(session, person, 2, payload)
    asset_id = image.id
    assert (image.width, image.height) == (None, None)

    result = review_asset(
        session,
        person,
        asset_id=asset_id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(payload),
    )

    assert isinstance(result, ReviewedAsset)
    assert result.validation_state is ValidationState.USABLE
    # Re-read rather than reusing the narrowed reference above: the point is
    # that the row now carries what the review measured.
    stored = session.get(ReferenceAsset, asset_id)
    assert stored is not None
    assert (stored.width, stored.height) == (1024, 768)
    # The landed schema's update guard requires the version to advance with
    # the settle; 1 means unchecked, the first decision makes it 2.
    assert stored.review_version == 2


def test_an_image_too_small_to_ground_refuses_with_the_number(session: Session) -> None:
    """The mechanism grounds at 1024px; below half that, identity does not survive.

    The measurement comes back with the refusal because it is the actionable
    part - "too small" alone leaves somebody guessing how much bigger.
    """

    person = subject(session)
    payload = png(MIN_SHORT_EDGE - 1, 2000)
    image = asset(session, person, 3, payload)

    result = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(payload),
    )

    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.TOO_SMALL
    assert result.measured == (MIN_SHORT_EDGE - 1, 2000)
    assert image.validation_state == ValidationState.UNCHECKED.value


def test_bytes_that_are_not_an_image_refuse(session: Session) -> None:
    person = subject(session)
    image = asset(session, person, 4, b"this is not a picture")

    result = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(b"this is not a picture"),
    )

    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNREADABLE_IMAGE


def test_a_missing_file_refuses_rather_than_raising(session: Session) -> None:
    """The artifact row can outlive its bytes, and a review is not a crash site."""

    person = subject(session)
    image = asset(session, person, 5)

    def missing(_artifact_id: str) -> bytes:
        raise OSError("gone")

    result = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=missing,
    )

    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNREADABLE_IMAGE


def test_a_judgement_against_an_image_has_to_say_why(session: Session) -> None:
    """`weak` with no reason is unreadable later, so it is refused now.

    Nobody can tell from the state alone whether the image was blurry, cropped
    or of the wrong person - and so nobody can tell whether replacing it helps.
    """

    person = subject(session)
    payload = png(64, 64)
    image = asset(session, person, 6, payload)

    bare = review_asset(session, person, asset_id=image.id, outcome=ReviewOutcome.WEAK)
    assert isinstance(bare, ReviewRefused)
    assert bare.refusal is ReviewRefusal.REASONS_REQUIRED

    # The landed schema binds every settle to a measurement, so a negative
    # judgement needs the image too: unreadable bytes cannot be settled in
    # either direction and the asset stays unchecked.
    unreadable = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.REJECTED,
        reasons=("face is turned away",),
        read_bytes=reader(b"not a picture"),
    )
    assert isinstance(unreadable, ReviewRefused)
    assert unreadable.refusal is ReviewRefusal.UNREADABLE_IMAGE

    given = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.REJECTED,
        reasons=("face is turned away",),
        read_bytes=reader(png(64, 64)),
    )
    assert isinstance(given, ReviewedAsset)
    assert image.validation_reasons_json == ["face is turned away"]
    assert image.review_version == 2


@pytest.mark.parametrize("reasons", [("",), ("x" * 300,), (b"bytes",), 1, ["ok"] * 40, (None,)])
def test_malformed_reasons_refuse_rather_than_being_tidied(
    session: Session, reasons: object
) -> None:
    person = subject(session)
    image = asset(session, person, 7)

    result = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.REJECTED,
        reasons=reasons,  # type: ignore[arg-type]
    )

    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.REASONS_MALFORMED


def test_a_review_cannot_reach_into_another_subject(session: Session) -> None:
    """Membership is checked against the subject named, not looked up from the asset."""

    person = subject(session)
    stranger = subject(session)
    theirs = asset(session, stranger, 8)

    result = review_asset(
        session,
        person,
        asset_id=theirs.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(png(1024, 1024)),
    )

    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.ASSET_NOT_FOUND
    assert theirs.validation_state == ValidationState.UNCHECKED.value


def test_a_concurrent_settle_loses_with_a_typed_refusal(tmp_path: Path) -> None:
    """The ordinary two-reviewer race escaped raw: a session holding an
    unchecked snapshot whose rival settled mid-read hit the landed guard as
    a bare IntegrityError, poisoning the caller's session (codex/R1708).
    The loser now refuses ALREADY_SETTLED, the winner's decision stands,
    and the losing session stays usable."""

    engine_one = create_engine(f"sqlite:///{tmp_path / 'race.sqlite3'}")
    Base.metadata.create_all(engine_one)
    engine_two = create_engine(f"sqlite:///{tmp_path / 'race.sqlite3'}")

    with Session(engine_one) as loser, Session(engine_two) as winner:
        person = subject(loser)
        payload = png(1024, 1024)
        contested = asset(loser, person, 14, payload)
        spare = asset(loser, person, 15, png(64, 64))
        loser.commit()
        person_id, contested_id, spare_id = person.id, contested.id, spare.id
        # Re-load so the loser holds a pre-race snapshot in its identity map.
        stale_subject = loser.get(ReferenceSubject, person_id)
        assert stale_subject is not None
        snapshot = loser.get(ReferenceAsset, contested_id)
        assert snapshot is not None
        assert snapshot.validation_state == ValidationState.UNCHECKED.value

        def racing_reader(_artifact_id: str) -> bytes:
            rival_subject = winner.get(ReferenceSubject, person_id)
            assert rival_subject is not None
            settled = review_asset(
                winner,
                rival_subject,
                asset_id=contested_id,
                outcome=ReviewOutcome.USABLE,
                read_bytes=reader(payload),
            )
            assert isinstance(settled, ReviewedAsset)
            winner.commit()
            return payload

        result = review_asset(
            loser,
            stale_subject,
            asset_id=contested_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=racing_reader,
        )
        assert isinstance(result, ReviewRefused)
        assert result.refusal is ReviewRefusal.ALREADY_SETTLED

        # The winner's committed decision is untouched.
        loser.expire_all()
        stored = loser.get(ReferenceAsset, contested_id)
        assert stored is not None
        assert stored.validation_state == ValidationState.USABLE.value
        assert (stored.width, stored.height) == (1024, 1024)
        assert stored.review_version == 2

        # The losing session is still usable: an ordinary settle succeeds.
        follow_up = review_asset(
            loser,
            stale_subject,
            asset_id=spare_id,
            outcome=ReviewOutcome.REJECTED,
            reasons=("blurry",),
            read_bytes=reader(png(64, 64)),
        )
        assert isinstance(follow_up, ReviewedAsset)
        loser.commit()


def test_unrelated_dirty_state_is_never_flushed_or_mislabeled(
    tmp_path: Path,
) -> None:
    """Caller-owned pending garbage must not fail inside this call.

    A session carrying an unrelated pending change the landed trigger
    rejects used to have that change flushed from inside review_asset -
    first by query-invoked autoflush, then by the savepoint - and the
    failure came back mislabeled ALREADY_SETTLED with the session dead
    (codex/R1717). Now the call never flushes the caller's state: the
    review succeeds, the garbage stays pending, and it fails at the
    caller's own commit wearing its own name."""

    engine = create_engine(f"sqlite:///{tmp_path / 'dirty.sqlite3'}")
    Base.metadata.create_all(engine)
    observer_engine = create_engine(f"sqlite:///{tmp_path / 'dirty.sqlite3'}")

    with Session(engine) as caller, Session(observer_engine) as observer:
        person = subject(caller)
        target = asset(caller, person, 16, png(1024, 1024))
        bystander = asset(caller, person, 17)
        person_id, target_id, bystander_id = person.id, target.id, bystander.id
        # Commit AFTER capturing ids so person and target stay expired: the
        # membership check inside review_asset must lazy-load them, which is
        # exactly the query the no-autoflush containment exists to protect.
        caller.commit()

        # An unrelated pending change the landed review trigger rejects:
        # a settle-shaped write with no review_version increment.
        bystander.validation_state = ValidationState.USABLE.value
        bystander.width, bystander.height = 640, 480

        result = review_asset(
            caller,
            person,
            asset_id=target_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=reader(png(1024, 1024)),
        )
        assert isinstance(result, ReviewedAsset)
        assert result.validation_state is ValidationState.USABLE
        assert caller.is_active

        # Nothing of the caller's leaked mid-call: an independent session
        # sees the bystander untouched (and the settle itself still
        # unflushed-to-commit, exactly the flush-only contract).
        spied = observer.get(ReferenceAsset, bystander_id)
        assert spied is not None
        assert spied.validation_state == ValidationState.UNCHECKED.value
        assert spied.width is None and spied.height is None
        uncommitted = observer.get(ReferenceAsset, target_id)
        assert uncommitted is not None
        assert uncommitted.validation_state == ValidationState.UNCHECKED.value

        # The garbage fails where it belongs: at the caller's own commit.
        with pytest.raises(IntegrityError):
            caller.commit()
        caller.rollback()
        assert caller.is_active

        # And the session is genuinely usable: with the garbage rolled
        # back, an ordinary settle of the target completes and commits.
        person_again = caller.get(ReferenceSubject, person_id)
        assert person_again is not None
        retried = review_asset(
            caller,
            person_again,
            asset_id=target_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=reader(png(1024, 1024)),
        )
        assert isinstance(retried, ReviewedAsset)
        caller.commit()
        observer.expire_all()
        settled = observer.get(ReferenceAsset, target_id)
        assert settled is not None
        assert settled.validation_state == ValidationState.USABLE.value
        assert settled.review_version == 2


def test_an_asset_deleted_mid_review_refuses_not_found(tmp_path: Path) -> None:
    """A guarded miss is read back, not presumed settled.

    The conflict classifier distinguishes a row that settled from a row
    that vanished: only the former is ALREADY_SETTLED (codex/R1717)."""

    engine = create_engine(f"sqlite:///{tmp_path / 'gone.sqlite3'}")
    Base.metadata.create_all(engine)
    rival_engine = create_engine(f"sqlite:///{tmp_path / 'gone.sqlite3'}")

    with Session(engine) as caller, Session(rival_engine) as rival:
        person = subject(caller)
        target = asset(caller, person, 18, png(800, 600))
        caller.commit()
        person_id, target_id = person.id, target.id
        stale_subject = caller.get(ReferenceSubject, person_id)
        assert stale_subject is not None
        snapshot = caller.get(ReferenceAsset, target_id)
        assert snapshot is not None

        def deleting_reader(_artifact_id: str) -> bytes:
            doomed = rival.get(ReferenceAsset, target_id)
            assert doomed is not None
            rival.delete(doomed)
            rival.commit()
            return png(800, 600)

        result = review_asset(
            caller,
            stale_subject,
            asset_id=target_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=deleting_reader,
        )
        assert isinstance(result, ReviewRefused)
        assert result.refusal is ReviewRefusal.ASSET_NOT_FOUND
        assert caller.is_active


def test_an_asset_reparented_mid_review_refuses_not_found(tmp_path: Path) -> None:
    """Authority rides in the write predicate, not just the pre-check.

    An asset re-parented to another subject between the membership check
    and the write used to be settled under the OLD subject's authority:
    the settle statement does not change reference_subject_id, so no
    trigger fires, and the conflict readback only runs after a miss
    (codex/R1720). The subject guard in the predicate makes this race
    miss, and the miss classifies as not-found."""

    engine = create_engine(f"sqlite:///{tmp_path / 'moved.sqlite3'}")
    Base.metadata.create_all(engine)
    rival_engine = create_engine(f"sqlite:///{tmp_path / 'moved.sqlite3'}")

    with Session(engine) as caller, Session(rival_engine) as rival:
        person = subject(caller)
        other = subject(caller)
        target = asset(caller, person, 19, png(1024, 1024))
        spare = asset(caller, person, 20, png(1025, 1024))
        caller.commit()
        person_id, other_id = person.id, other.id
        target_id, spare_id = target.id, spare.id
        stale_subject = caller.get(ReferenceSubject, person_id)
        assert stale_subject is not None
        snapshot = caller.get(ReferenceAsset, target_id)
        assert snapshot is not None

        def reparenting_reader(_artifact_id: str) -> bytes:
            moved = rival.get(ReferenceAsset, target_id)
            assert moved is not None
            moved.reference_subject_id = other_id
            rival.commit()
            return png(1024, 1024)

        result = review_asset(
            caller,
            stale_subject,
            asset_id=target_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=reparenting_reader,
        )
        assert isinstance(result, ReviewRefused)
        assert result.refusal is ReviewRefusal.ASSET_NOT_FOUND

        # The new ownership is preserved and the row was never settled.
        rival.expire_all()
        stored = rival.get(ReferenceAsset, target_id)
        assert stored is not None
        assert stored.reference_subject_id == other_id
        assert stored.validation_state == ValidationState.UNCHECKED.value
        assert stored.review_version == 1
        assert stored.width is None and stored.height is None

        # The caller session is still usable: an ordinary settle succeeds.
        follow_up = review_asset(
            caller,
            stale_subject,
            asset_id=spare_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=reader(png(1025, 1024)),
        )
        assert isinstance(follow_up, ReviewedAsset)
        caller.commit()


def test_an_artifact_swapped_mid_review_refuses_not_found(tmp_path: Path) -> None:
    """The settle binds the exact bytes it judged, by predicate.

    The identity trigger permits rebinding an unchecked row's artifact,
    so an artifact swapped between the read and the write used to get
    the OLD bytes' measurement recorded against a replacement the review
    never saw (codex/R1722). The artifact binding in the predicate makes
    the swap miss, and the miss classifies as not-found."""

    engine = create_engine(f"sqlite:///{tmp_path / 'swapped.sqlite3'}")
    Base.metadata.create_all(engine)
    rival_engine = create_engine(f"sqlite:///{tmp_path / 'swapped.sqlite3'}")

    with Session(engine) as caller, Session(rival_engine) as rival:
        person = subject(caller)
        target = asset(caller, person, 23, png(1024, 768))
        spare = asset(caller, person, 24, png(1024, 1024))
        caller.commit()
        person_id = person.id
        target_id, spare_id = target.id, spare.id
        replacement_digest = f"{999:064x}"
        replacement_id = f"sha256:{replacement_digest}"
        stale_subject = caller.get(ReferenceSubject, person_id)
        assert stale_subject is not None
        snapshot = caller.get(ReferenceAsset, target_id)
        assert snapshot is not None

        def swapping_reader(_artifact_id: str) -> bytes:
            rival.add(
                Artifact(
                    id=replacement_id,
                    sha256=replacement_digest,
                    kind="image",
                    media_type="image/png",
                    size_bytes=1,
                    relative_path="999/a.png",
                )
            )
            rebound = rival.get(ReferenceAsset, target_id)
            assert rebound is not None
            rebound.artifact_id = replacement_id
            rival.commit()
            return png(1024, 768)

        result = review_asset(
            caller,
            stale_subject,
            asset_id=target_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=swapping_reader,
        )
        assert isinstance(result, ReviewRefused)
        assert result.refusal is ReviewRefusal.ASSET_NOT_FOUND

        # The replacement binding is preserved and the row was never
        # settled with the old bytes' measurement.
        rival.expire_all()
        stored = rival.get(ReferenceAsset, target_id)
        assert stored is not None
        assert stored.artifact_id == replacement_id
        assert stored.validation_state == ValidationState.UNCHECKED.value
        assert stored.review_version == 1
        assert stored.width is None and stored.height is None

        # The caller session is still usable: an ordinary settle commits.
        follow_up = review_asset(
            caller,
            stale_subject,
            asset_id=spare_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=reader(png(1024, 1024)),
        )
        assert isinstance(follow_up, ReviewedAsset)
        caller.commit()


def test_the_reason_cap_is_the_schema_trigger_cap(session: Session) -> None:
    """The landed update trigger refuses more than 16 reasons; a boundary
    cap above it let 17 pass validation and explode at flush as an
    IntegrityError with the session left failed (codex/R1705). Sixteen
    settle; seventeen refuse typed, before any write."""

    person = subject(session)
    image = asset(session, person, 12, png(64, 64))

    over = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.REJECTED,
        reasons=tuple(f"reason {index}" for index in range(17)),
        read_bytes=reader(png(64, 64)),
    )
    assert isinstance(over, ReviewRefused)
    assert over.refusal is ReviewRefusal.REASONS_MALFORMED
    assert image.validation_state == ValidationState.UNCHECKED.value

    at_cap = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.REJECTED,
        reasons=tuple(f"reason {index}" for index in range(16)),
        read_bytes=reader(png(64, 64)),
    )
    assert isinstance(at_cap, ReviewedAsset)
    assert len(image.validation_reasons_json) == 16


def test_a_truncated_image_with_an_intact_header_refuses(
    session: Session,
) -> None:
    """The lazy open measured a 1024x768 PNG from 48 bytes; the review is
    the evidence boundary, so header-only bytes must refuse without
    settling (codex/R1705)."""

    person = subject(session)
    whole = png(1024, 768)

    # Pillow's verifier in fact tolerates up to FOUR missing bytes of the
    # trailing IEND CRC (codex/R1730 corrected the earlier one-byte note);
    # the byte-level framer refuses every one of those cuts, and the
    # exhaustive proof lives in the completeness sweep. Each cut here
    # binds its OWN asset so its digest gate passes and the refusal
    # proven is the container check's, not the digest check's.
    for offset, cut in enumerate((48, len(whole) // 2, len(whole) - 8)):
        cut_asset = asset(session, person, 130 + offset, whole[:cut])
        result = review_asset(
            session,
            person,
            asset_id=cut_asset.id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=reader(whole[:cut]),
        )
        assert isinstance(result, ReviewRefused), cut
        assert result.refusal is ReviewRefusal.UNREADABLE_IMAGE, cut
        assert cut_asset.validation_state == ValidationState.UNCHECKED.value
        assert (cut_asset.width, cut_asset.height) == (None, None)

    image = asset(session, person, 13, whole)
    intact = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(whole),
    )
    assert isinstance(intact, ReviewedAsset)


def test_a_verifier_refusal_is_a_typed_review_refusal(session: Session) -> None:
    """The production reader raises ValueError for checksum, size, and path
    violations - the exact corruption and tamper cases UNREADABLE_IMAGE
    documents - and those raised straight out of the authoritative path as
    a 500 instead of a refusal (grok/R1178, the reject's headline)."""

    person = subject(session)
    image = asset(session, person, 9)

    def tampered(_artifact_id: str) -> bytes:
        raise ValueError("checksum mismatch")

    result = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=tampered,
    )

    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNREADABLE_IMAGE
    assert image.validation_state == ValidationState.UNCHECKED.value


def test_a_bomb_warning_range_image_refuses_without_a_warning(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Between MAX_IMAGE_PIXELS and twice it Pillow only WARNS, so a small
    crafted header escaped with a warning instead of refusing; past twice
    it errors, which was already caught (grok/R1178)."""

    person = subject(session)
    payload = png(40, 40)  # 1600 pixels
    # The digest gate runs before the decode, so this pin must bind the
    # real payload digest or the refusal it proves is the digest check's,
    # not the bomb promotion's.
    image = asset(session, person, 10, payload)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # warn range: 1000-2000

    # Record, never promote: an error filter HERE would do the module's
    # job for it and the promotion mutant survived exactly that way. The
    # refusal must come from the module's own promotion, and no warning
    # may leak to the caller.
    with warnings.catch_warnings(record=True) as leaked:
        warnings.simplefilter("always")
        result = review_asset(
            session,
            person,
            asset_id=image.id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=reader(payload),
        )

    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNREADABLE_IMAGE
    assert leaked == []
    assert image.validation_state == ValidationState.UNCHECKED.value

    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 700)  # error range: > 1400
    hard = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(payload),
    )
    assert isinstance(hard, ReviewRefused)
    assert hard.refusal is ReviewRefusal.UNREADABLE_IMAGE


def test_a_value_equal_impostor_outcome_refuses(session: Session) -> None:
    """The promotion gate is checked by identity but was written by value,
    so a ValidationState equal to "usable" skipped the measurement and
    still landed as usable. The boundary validates outcome at runtime like
    every other input (grok/R1178)."""

    person = subject(session)
    image = asset(session, person, 11)

    result = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ValidationState.USABLE,  # type: ignore[arg-type]
        read_bytes=reader(png(1024, 1024)),
    )

    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.OUTCOME_INVALID
    assert image.validation_state == ValidationState.UNCHECKED.value
    assert (image.width, image.height) == (None, None)


def jpeg(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (128, 128, 128)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_a_settle_writes_its_immutable_event_in_the_same_transaction(
    tmp_path: Path,
) -> None:
    """No settle without its evidence; both land or neither does.

    review_asset settled the asset but never wrote the landed
    ReferenceAssetReviewEvent, so a committed review left the schema's
    promised evidence blank (codex/R1726)."""

    engine = create_engine(f"sqlite:///{tmp_path / 'evented.sqlite3'}")
    Base.metadata.create_all(engine)
    observer_engine = create_engine(f"sqlite:///{tmp_path / 'evented.sqlite3'}")

    with Session(engine) as caller, Session(observer_engine) as observer:
        person = subject(caller)
        payload = png(1024, 768)
        target = asset(caller, person, 26, payload)
        caller.commit()
        target_id = target.id
        artifact_id = target.artifact_id

        result = review_asset(
            caller,
            person,
            asset_id=target_id,
            outcome=ReviewOutcome.REJECTED,
            reasons=("blurry", "cropped"),
            read_bytes=reader(payload),
        )
        assert isinstance(result, ReviewedAsset)
        events = list(caller.scalars(select(ReferenceAssetReviewEvent)))
        assert len(events) == 1
        event = events[0]
        assert event.reference_asset_id == target_id
        assert event.artifact_id == artifact_id
        assert event.artifact_sha256 == hashlib.sha256(payload).hexdigest()
        assert event.expected_state == ValidationState.UNCHECKED.value
        assert event.expected_version == 1
        assert event.result_version == 2
        assert event.decision == ValidationState.REJECTED.value
        assert event.reasons_json == ["blurry", "cropped"]
        assert (event.width, event.height) == (1024, 768)
        assert event.id == f"refreview:sha256:{event.decision_sha256}"

        # The event rides the caller's transaction: a rollback takes the
        # settle AND its evidence, never one without the other.
        caller.rollback()
        assert observer.scalars(select(ReferenceAssetReviewEvent)).first() is None

        person_again = caller.get(ReferenceSubject, person.id)
        assert person_again is not None
        retried = review_asset(
            caller,
            person_again,
            asset_id=target_id,
            outcome=ReviewOutcome.REJECTED,
            reasons=("blurry", "cropped"),
            read_bytes=reader(payload),
        )
        assert isinstance(retried, ReviewedAsset)
        caller.commit()
        observer.expire_all()
        durable = list(observer.scalars(select(ReferenceAssetReviewEvent)))
        assert len(durable) == 1
        settled = observer.get(ReferenceAsset, target_id)
        assert settled is not None
        assert settled.review_version == durable[0].result_version == 2


def test_caller_pending_changes_on_the_reviewed_row_survive(
    tmp_path: Path,
) -> None:
    """Review refreshes only the attributes it owns.

    An unqualified expire after the direct write erased the caller's own
    pending caption, purpose, and sort order on the reviewed row along
    with their ORM histories (codex/R1726)."""

    engine = create_engine(f"sqlite:///{tmp_path / 'pending.sqlite3'}")
    Base.metadata.create_all(engine)
    observer_engine = create_engine(f"sqlite:///{tmp_path / 'pending.sqlite3'}")

    with Session(engine) as caller, Session(observer_engine) as observer:
        person = subject(caller)
        target = asset(caller, person, 27, png(1024, 1024))
        target.caption = "old"
        caller.commit()
        target_id = target.id

        target.caption = "caller pending"
        target.purpose = "pose"
        target.sort_order = 99

        result = review_asset(
            caller,
            person,
            asset_id=target_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=reader(png(1024, 1024)),
        )
        assert isinstance(result, ReviewedAsset)
        assert target.caption == "caller pending"
        assert target.purpose == "pose"
        assert target.sort_order == 99
        assert target in caller.dirty

        # The review-owned attributes read back settled from the write.
        assert target.validation_state == ValidationState.USABLE.value
        assert target.review_version == 2

        caller.commit()
        observer.expire_all()
        stored = observer.get(ReferenceAsset, target_id)
        assert stored is not None
        assert stored.caption == "caller pending"
        assert stored.purpose == "pose"
        assert stored.sort_order == 99
        assert stored.validation_state == ValidationState.USABLE.value


def test_bytes_that_do_not_match_the_artifact_refuse(session: Session) -> None:
    """The judged bytes must BE the artifact addressed.

    A reader returning a valid image whose hash differs from the seeded
    artifact digest used to promote the asset while the event recorded
    the row digest, not the judged payload digest (codex/R1727). The
    payload is hashed at this boundary and a mismatch refuses before a
    pixel is decoded."""

    person = subject(session)
    image = asset(session, person, 28)  # synthetic digest, no payload

    result = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(png(1024, 768)),
    )
    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNREADABLE_IMAGE
    assert image.validation_state == ValidationState.UNCHECKED.value
    assert session.scalars(select(ReferenceAssetReviewEvent)).first() is None


def test_an_event_write_failure_takes_the_settle_with_it(tmp_path: Path) -> None:
    """Settle and event are failure-atomic: both land or neither does.

    An event-insert failure used to propagate while the preceding settle
    stayed committable - a caller commit then produced a settled asset
    with zero review events (codex/R1727). The connection is now
    invalidated on any event-write error, which kills the whole wire
    transaction; the caller's rollback reconnects and the session is
    usable again."""

    engine = create_engine(f"sqlite:///{tmp_path / 'atomic.sqlite3'}")
    Base.metadata.create_all(engine)
    observer_engine = create_engine(f"sqlite:///{tmp_path / 'atomic.sqlite3'}")

    with Session(engine) as caller, Session(observer_engine) as observer:
        person = subject(caller)
        payload = png(1024, 768)
        target = asset(caller, person, 29, payload)
        caller.commit()
        person_id, target_id = person.id, target.id

        observer.connection().exec_driver_sql(
            "CREATE TRIGGER hostile_event_block "
            "BEFORE INSERT ON reference_asset_review_events "
            "BEGIN SELECT RAISE(ABORT, 'event write refused'); END"
        )
        observer.commit()

        with pytest.raises(IntegrityError):
            review_asset(
                caller,
                person,
                asset_id=target_id,
                outcome=ReviewOutcome.USABLE,
                read_bytes=reader(payload),
            )
        # The half-done settle must not be committable: the invalidated
        # connection surfaces as PendingRollbackError at commit.
        with pytest.raises(PendingRollbackError):
            caller.commit()
        caller.rollback()

        observer.expire_all()
        stored = observer.get(ReferenceAsset, target_id)
        assert stored is not None
        assert stored.validation_state == ValidationState.UNCHECKED.value
        assert stored.review_version == 1
        assert observer.scalars(select(ReferenceAssetReviewEvent)).first() is None

        # With the hostile trigger gone, the recovered session settles
        # and commits the pair normally.
        observer.connection().exec_driver_sql("DROP TRIGGER hostile_event_block")
        observer.commit()
        person_again = caller.get(ReferenceSubject, person_id)
        assert person_again is not None
        retried = review_asset(
            caller,
            person_again,
            asset_id=target_id,
            outcome=ReviewOutcome.USABLE,
            read_bytes=reader(payload),
        )
        assert isinstance(retried, ReviewedAsset)
        caller.commit()
        observer.expire_all()
        assert len(list(observer.scalars(select(ReferenceAssetReviewEvent)))) == 1


def gif(width: int, height: int, frames: int) -> bytes:
    images = [Image.new("RGB", (width, height), (index * 40, 100, 100)) for index in range(frames)]
    buffer = io.BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def apng(width: int, height: int, frames: int) -> bytes:
    images = [Image.new("RGB", (width, height), (index * 60, 100, 100)) for index in range(frames)]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PNG", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def test_formats_without_a_completeness_proof_refuse(session: Session) -> None:
    """Only PNG and JPEG are accepted; GIF refuses even when whole.

    A three-frame GIF cut exactly at a frame boundary presents as a
    complete one-frame still: five of its 641 truncated prefixes gained
    review authority on the previous generation (codex/R1728). GIF has
    no container-completeness proof, so the format refuses outright -
    including the exact frame-boundary prefix and a genuinely
    single-frame GIF."""

    person = subject(session)
    whole = gif(128, 96, 3)

    # The exact frame-boundary family: a prefix that ends after the
    # first frame's image data and decodes as a well-formed still.
    boundary = whole[:216]
    boundary_asset = asset(session, person, 137, boundary)
    result = review_asset(
        session,
        person,
        asset_id=boundary_asset.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(boundary),
    )
    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNSUPPORTED_FORMAT
    assert boundary_asset.validation_state == ValidationState.UNCHECKED.value
    assert session.scalars(select(ReferenceAssetReviewEvent)).first() is None

    single = gif(128, 96, 1)
    single_asset = asset(session, person, 138, single)
    result = review_asset(
        session,
        person,
        asset_id=single_asset.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(single),
    )
    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNSUPPORTED_FORMAT


def test_only_the_exact_complete_payload_measures() -> None:
    """Byte-level completeness: every proper prefix refuses, only the
    whole payload measures, and nothing may ride behind it.

    54 of 641 GIF prefixes escaped _measure as raw exceptions
    (codex/R1728), and Pillow then accepted PNGs missing one through
    four bytes of the mandatory IEND CRC plus payloads with trailing
    junk or a concatenated second image (codex/R1730). The framers
    prove the whole submitted byte string is exactly one complete
    container."""

    gif_bytes = gif(128, 96, 3)
    for cut in range(1, len(gif_bytes) + 1):
        outcome = _measure(gif_bytes[:cut])
        assert isinstance(outcome, ReviewRefusal), cut

    for payload in (png(600, 600), jpeg(600, 600)):
        for cut in range(1, len(payload)):
            outcome = _measure(payload[:cut])
            assert isinstance(outcome, ReviewRefusal), cut
        assert _measure(payload) == (600, 600)
        assert isinstance(_measure(payload + b"junk"), ReviewRefusal)
        assert isinstance(_measure(payload + payload), ReviewRefusal)

    apng_bytes = apng(64, 64, 3)
    for cut in range(1, len(apng_bytes) + 1):
        outcome = _measure(apng_bytes[:cut])
        assert isinstance(outcome, (tuple, ReviewRefusal)), cut


def test_a_multi_frame_image_refuses(session: Session) -> None:
    """A reference is a single still; frames beyond the first refuse.

    The multi-frame check lives on for formats the whitelist accepts:
    an APNG is PNG-format with visible extra frames, and load() decodes
    only the current one (codex/R1727, codex/R1728). A truncated APNG
    fails the chunk-structure verify instead."""

    person = subject(session)
    whole = apng(64, 64, 3)

    intact_asset = asset(session, person, 134, whole)
    result = review_asset(
        session,
        person,
        asset_id=intact_asset.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(whole),
    )
    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.MULTI_FRAME
    assert intact_asset.validation_state == ValidationState.UNCHECKED.value

    half = whole[: len(whole) // 2]
    half_asset = asset(session, person, 135, half)
    result = review_asset(
        session,
        person,
        asset_id=half_asset.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(half),
    )
    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNREADABLE_IMAGE


def _walk_to_idat(payload: bytes) -> tuple[int, int]:
    offset = 8
    while True:
        length = int.from_bytes(payload[offset : offset + 4], "big")
        if payload[offset + 4 : offset + 8] == b"IDAT":
            return offset, length
        offset += 8 + length + 4


def test_each_defense_layer_catches_what_only_it_can_see() -> None:
    """Framing, verify, and load each have irreplaceable work.

    With byte-level framing subsuming truncation, the structural verify
    and the full decode need corruption INSIDE a complete container to
    prove themselves: a valid IDAT whose CRC is flipped is caught only
    by verify (Pillow's load tolerates a bad CRC over decodable data),
    and corrupted IDAT data behind a recomputed CRC is caught only by
    the decode (zlib fails where the container looks perfect). Both must
    refuse typed."""

    whole = png(600, 600)
    offset, length = _walk_to_idat(whole)
    crc_position = offset + 8 + length

    crc_flipped = bytearray(whole)
    crc_flipped[crc_position] ^= 0xFF
    outcome = _measure(bytes(crc_flipped))
    assert isinstance(outcome, ReviewRefusal)
    assert outcome is ReviewRefusal.UNREADABLE_IMAGE

    # The consumption checker verifies stream termination and decoded
    # LENGTH, not filter validity: an invalid filter-type byte behind a
    # well-formed stream of the right length passes framing, the
    # consumption proof, and verify - only the full decode refuses it.
    decoded = bytearray(zlib.decompress(whole[offset + 8 : offset + 8 + length]))
    decoded[0] = 99
    bad_filter = _rebuild_idat(whole, zlib.compress(bytes(decoded)))
    outcome = _measure(bad_filter)
    assert isinstance(outcome, ReviewRefusal)
    assert outcome is ReviewRefusal.UNREADABLE_IMAGE


def _rebuild_idat(payload: bytes, data: bytes) -> bytes:
    offset, length = _walk_to_idat(payload)
    crc = zlib.crc32(b"IDAT" + data)
    return (
        payload[:offset]
        + len(data).to_bytes(4, "big")
        + b"IDAT"
        + data
        + crc.to_bytes(4, "big")
        + payload[offset + 8 + length + 4 :]
    )


def test_jpeg_settles_at_the_physical_eoi_bar(session: Session) -> None:
    """JPEG is accepted again, by owner decision, at physical-EOI framing.

    A truncated JPEG refuses at the framer, an intact one settles, and
    legal 0xFF marker-fill bytes before the terminal EOI are accepted -
    the strict walker falsely refused files Pillow accepts
    (codex/R1731's note). The injected-bytes residual behind this bar
    is a documented, owner-accepted low-priority successor."""

    person = subject(session)
    whole = jpeg(1024, 768)
    half = whole[: len(whole) // 2]

    cut_asset = asset(session, person, 142, half)
    result = review_asset(
        session,
        person,
        asset_id=cut_asset.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(half),
    )
    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNREADABLE_IMAGE

    image = asset(session, person, 143, whole)
    intact = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(whole),
    )
    assert isinstance(intact, ReviewedAsset)
    assert (intact.width, intact.height) == (1024, 768)

    filled = whole[:-2] + b"\xff" + whole[-2:]
    fill_asset = asset(session, person, 144, filled)
    accepted = review_asset(
        session,
        person,
        asset_id=fill_asset.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(filled),
    )
    assert isinstance(accepted, ReviewedAsset)
    assert (accepted.width, accepted.height) == (1024, 768)


def test_bytes_the_decoder_does_not_consume_refuse(session: Session) -> None:
    """A validly framed IDAT may not smuggle unconsumed bytes.

    Appending junk after the completed zlib stream inside IDAT, with the
    chunk length increased and its CRC recomputed, passed the chunk
    walk, Pillow verify and load, and the public authority path with
    unchanged pixels (codex/R1731). The consumption proof requires the
    stream to terminate with zero unused bytes and decode to exactly
    the declared scanline structure."""

    person = subject(session)
    whole = png(600, 600)
    offset, length = _walk_to_idat(whole)
    attack = _rebuild_idat(whole, whole[offset + 8 : offset + 8 + length] + b"Z" * 4096)

    outcome = _measure(attack)
    assert outcome is ReviewRefusal.UNREADABLE_IMAGE

    image = asset(session, person, 141, attack)
    result = review_asset(
        session,
        person,
        asset_id=image.id,
        outcome=ReviewOutcome.USABLE,
        read_bytes=reader(attack),
    )
    assert isinstance(result, ReviewRefused)
    assert result.refusal is ReviewRefusal.UNREADABLE_IMAGE
    assert image.validation_state == ValidationState.UNCHECKED.value

    # An interlaced header refuses AT THE CHECKER: the stream here is
    # untouched non-interlaced data behind a lying flag, so its decoded
    # length still equals the non-interlaced expectation and the
    # interlace condition is the ONLY check that can refuse it.
    header_length = int.from_bytes(whole[8:12], "big")
    header = bytearray(whole[16 : 16 + header_length])
    header[12] = 1
    header_crc = zlib.crc32(b"IHDR" + bytes(header))
    interlaced = (
        whole[:16] + bytes(header) + header_crc.to_bytes(4, "big") + whole[16 + header_length + 4 :]
    )
    assert not _png_complete_and_consumed(interlaced)
    assert _measure(interlaced) is ReviewRefusal.UNREADABLE_IMAGE

    # A cleanly terminated stream that decodes SHORT refuses at the
    # checker: eof true, nothing unused or unconsumed - only the
    # declared-scanline-length comparison can see the missing rows.
    decoded_all = zlib.decompress(whole[offset + 8 : offset + 8 + length])
    row_size = len(decoded_all) // 600
    short = _rebuild_idat(whole, zlib.compress(decoded_all[:-row_size]))
    assert not _png_complete_and_consumed(short)
    assert _measure(short) is ReviewRefusal.UNREADABLE_IMAGE
