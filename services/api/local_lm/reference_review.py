"""Promoting a reference image to something a generation may condition on.

This is the piece the accept path named as missing. `attach_asset` writes
`unchecked` and is the only writer of `validation_state`, so nothing has ever
been eligible and `bind_conditioning` refuses for every subject in the product.
This is the authoritative transition that changes that - and it is deliberately
the *only* other writer, so "reviewed" continues to mean one thing.

`usable` is not a flag somebody sets; it is a claim that has to be paid for.
Promotion reads the image and measures it, because the alternative is a state
that means "someone clicked" - and the whole reason `UNCHECKED` exists as
distinct from `USABLE` is that an unlooked-at image is not a checked one.
That inverts `attach_asset`'s rule, correctly: there the read is advice, and
advice that can fail must not block the thing it advises on. Here the read *is*
the evidence, so an image that cannot be read cannot be promoted.

The size floor is not arbitrary. The mechanism grounds at `grounding_px` 1024,
so a short edge under half that is being upscaled past the point where identity
survives - the run would succeed and return a picture of nobody in particular,
which is the failure this boundary exists to prevent. Refusing is actionable:
the image is too small, and a bigger one fixes it.

Measuring also fills `ReferenceAsset.width`/`height`, which no writer has ever
populated. A rule that read them without populating them would refuse every
asset forever, which is how a bound comes to reject all real data.

What this does not record is *who* reviewed. The application has no notion of a
user, so inventing a reviewer field would be recording a fiction. `updated_at`
carries when, and that is the honest limit of what is known.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from PIL import Image
from sqlalchemy import Connection, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .models import (
    Artifact,
    ReferenceAsset,
    ReferenceAssetReviewEvent,
    ReferenceSubject,
)
from .references import ValidationState

#: Half the grounding resolution. Below this the reference is upscaled past
#: where identity survives, so `usable` would be a claim the pixels cannot keep.
MIN_SHORT_EDGE: Final = 512
#: A refusal needs a reason, and a bounded number of bounded ones. The bound
#: is the landed review trigger's: it refuses more than 16, and a boundary
#: cap above it let 17 reasons pass validation and then explode at flush as
#: an IntegrityError with the session left failed.
MAX_REASONS: Final = 16
MAX_REASON_LENGTH: Final = 200
#: The container formats this boundary accepts. GIF is refused: a
#: frame-boundary cut presents as a complete still and five truncated
#: prefixes gained authority. PNG carries a full consumption proof:
#: exact chunk walk plus a zlib stream terminating with zero unused
#: bytes at the declared scanline structure. JPEG is accepted at the
#: physical-EOI bar instead, because exact decoder-consumption proof
#: needs a standards-complete MCU walker and this is not one. The
#: injected-bytes residual such a walker would close is a known
#: residual, recorded here rather than overlooked.
STILL_FORMATS: Final = frozenset({"PNG", "JPEG"})
#: The attributes this module owns and may refresh after its direct write.
#: An unqualified expire also erased the caller's own pending edits to the
#: same row - caption, purpose, view label, sort order - along with their
#: ORM histories. Review touches only these.
REVIEW_OWNED_ATTRIBUTES: Final = (
    "validation_state",
    "validation_reasons_json",
    "review_version",
    "width",
    "height",
    "updated_at",
)


class ReviewOutcome(StrEnum):
    """What a review can conclude. `UNCHECKED` is deliberately absent.

    Un-reviewing would erase the record of a decision rather than replace it,
    and there is already a way back: an image promoted by mistake is marked
    rejected, which says what happened instead of pretending it never did.
    """

    USABLE = "usable"
    WEAK = "weak"
    REJECTED = "rejected"


class ReviewRefusal(StrEnum):
    """Why a review did not happen. Closed and typed."""

    #: The asset is not this subject's, no longer exists, or no longer binds
    #: the artifact whose bytes were read - the asset as addressed is gone.
    ASSET_NOT_FOUND = "asset_not_found"
    #: A judgement against an image needs to say what was wrong with it.
    REASONS_REQUIRED = "reasons_required"
    REASONS_MALFORMED = "reasons_malformed"
    #: Promotion needs the image, and no reader was supplied.
    NO_READER = "no_reader"
    #: The bytes are gone, or are not an image any decoder here recognizes.
    UNREADABLE_IMAGE = "unreadable_image"
    #: Readable, and too small for the mechanism to ground identity on.
    TOO_SMALL = "too_small"
    #: The outcome is not a ReviewOutcome. A module whose whole point is that
    #: usability cannot be asserted must not itself trust an annotation at
    #: runtime while validating every other input: a value-equal
    #: impostor such as a ValidationState would skip the measurement
    #: and still land as usable.
    OUTCOME_INVALID = "outcome_invalid"
    #: The bytes decode, but not as a format this boundary can prove
    #: complete (PNG or JPEG). A frame-boundary GIF truncation presents
    #: as a well-formed still, so animation-capable formats without a
    #: container-completeness proof refuse outright.
    UNSUPPORTED_FORMAT = "unsupported_format"
    #: The bytes are a readable image with more than one visible frame.
    #: A reference is a single still: Pillow's load() decodes only the
    #: CURRENT frame, so a three-frame GIF truncated after frame one lost
    #: two thirds of its bytes and still measured. Frames beyond the
    #: first refuse explicitly rather than silently judging frame one.
    MULTI_FRAME = "multi_frame"
    #: The asset moved between our snapshot and our write - a concurrent
    #: reviewer settled it, or the caller's view was stale - and the row was
    #: read back to prove it. The standing decision holds and this one was
    #: never recorded.
    ALREADY_SETTLED = "already_settled"


@dataclass(frozen=True, slots=True)
class ReviewedAsset:
    """The transition that happened, as it was written."""

    reference_asset_id: str
    validation_state: ValidationState
    reasons: tuple[str, ...] = ()
    width: int | None = None
    height: int | None = None

    def provenance(self) -> dict[str, object]:
        return {
            "reference_asset_id": self.reference_asset_id,
            "validation_state": self.validation_state.value,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ReviewRefused:
    """No transition, and why - as a closed reason and nothing else."""

    refusal: ReviewRefusal
    #: Present only for `TOO_SMALL`, where the number is the actionable part.
    measured: tuple[int, int] | None = None

    def provenance(self) -> dict[str, object]:
        record: dict[str, object] = {"reviewed": False, "refused": self.refusal.value}
        if self.measured is not None:
            record["measured"] = list(self.measured)
        return record


def _clean_reasons(reasons: Sequence[str]) -> tuple[str, ...] | None:
    if type(reasons) not in (tuple, list) or len(reasons) > MAX_REASONS:
        return None
    cleaned: list[str] = []
    for reason in reasons:
        if type(reason) is not str:
            return None
        text = reason.strip()
        if not text or len(text) > MAX_REASON_LENGTH:
            return None
        cleaned.append(text)
    return tuple(cleaned)


_PNG_CHANNELS: Final = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _png_complete_and_consumed(payload: bytes) -> bool:
    """The bytes are ONE complete PNG whose stream the decoder consumes
    in full - framing alone was not enough.

    Pillow accepts a PNG missing one through four bytes of the IEND CRC,
    reads through trailing junk, and also ignores bytes appended INSIDE
    a validly framed IDAT after the zlib stream ends, with the chunk
    length and CRC recomputed. So beyond walking every chunk to an exact
    IEND CRC at physical end-of-file, the concatenated IDAT stream must
    terminate with zero unused or unconsumed bytes and decode to exactly
    the scanline structure the header declares. Interlaced files and
    non-zero compression or filter methods refuse: only what can be
    proven is accepted.
    """

    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    offset = 8
    header: bytes | None = None
    idat = bytearray()
    framed = False
    while True:
        if offset + 8 > len(payload):
            return False
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        end = offset + 8 + length + 4
        if end > len(payload):
            return False
        if header is None:
            if chunk_type != b"IHDR" or length != 13:
                return False
            header = payload[offset + 8 : offset + 8 + 13]
        elif chunk_type == b"IDAT":
            idat += payload[offset + 8 : offset + 8 + length]
        if chunk_type == b"IEND":
            framed = end == len(payload) and length == 0 and payload[end - 4 : end] == b"\xaeB`\x82"
            break
        offset = end
    if not framed or header is None or not idat:
        return False
    width = int.from_bytes(header[0:4], "big")
    height = int.from_bytes(header[4:8], "big")
    bit_depth = header[8]
    channels = _PNG_CHANNELS.get(header[9])
    compression, filter_method, interlace = header[10], header[11], header[12]
    if (
        width <= 0
        or height <= 0
        or channels is None
        or compression != 0
        or filter_method != 0
        or interlace != 0
    ):
        return False
    row_bytes = -(-width * channels * bit_depth // 8)
    expected = height * (1 + row_bytes)
    decompressor = zlib.decompressobj()
    try:
        # The ceiling bounds a decompression bomb to one byte past the
        # declared scanline structure.
        decoded = decompressor.decompress(bytes(idat), expected + 1)
    except zlib.error:
        return False
    return (
        len(decoded) == expected
        and decompressor.eof
        and decompressor.unused_data == b""
        and decompressor.unconsumed_tail == b""
    )


def _jpeg_frames_exactly(payload: bytes) -> bool:
    """The bytes are ONE complete JPEG: terminal EOI at physical EOF.

    Entropy-coded scans are walked (stuffed 0xFF00 and restart markers
    skipped), length-delimited segments - including an EXIF thumbnail
    with its own embedded EOI - are stepped over whole, and legal 0xFF
    marker-fill runs are consumed, so a truncation, trailing junk, or a
    concatenated second image refuse while every file Pillow accepts is
    accepted, marker fill included.

    KNOWN RESIDUAL: bytes injected inside the final entropy scan
    immediately before the terminal EOI are not detected. Proving exact
    decoder consumption needs a standards-complete MCU walker, which
    this is not. Physical-EOI framing is the bar this function holds,
    and the residual is recorded here rather than overlooked.
    """

    if not payload.startswith(b"\xff\xd8"):
        return False
    offset = 2
    while offset + 2 <= len(payload):
        if payload[offset] != 0xFF:
            return False
        marker = payload[offset + 1]
        if marker == 0xFF:
            # Legal marker fill: any run of 0xFF bytes may pad before a
            # marker; refusing them refused files Pillow accepts, so
            # they are consumed rather than rejected.
            offset += 1
            continue
        if marker == 0xD9:
            return offset + 2 == len(payload)
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if offset + 4 > len(payload):
            return False
        segment_length = int.from_bytes(payload[offset + 2 : offset + 4], "big")
        if segment_length < 2 or offset + 2 + segment_length > len(payload):
            return False
        offset += 2 + segment_length
        if marker == 0xDA:
            while True:
                found = payload.find(b"\xff", offset)
                if found == -1 or found + 1 >= len(payload):
                    return False
                following = payload[found + 1]
                if following == 0x00 or 0xD0 <= following <= 0xD7:
                    offset = found + 2
                    continue
                offset = found
                break
    return False


_CONTAINER_FRAMERS: Final = {
    "PNG": _png_complete_and_consumed,
    "JPEG": _jpeg_frames_exactly,
}


def _measure(payload: bytes) -> tuple[int, int] | ReviewRefusal:
    """The image's size, or the typed refusal these bytes have earned.

    `Image.open` alone is lazy and reads only the header, so it would accept
    a truncated file whose header survived - the read here IS the evidence,
    so every pixel is decoded before the measurement counts.
    """

    try:
        with warnings.catch_warnings():
            # Between MAX_IMAGE_PIXELS and twice it, Pillow only WARNS; a
            # 13 KB crafted header would sail through with a warning nobody
            # is required to see. Promote it to the same refusal the
            # outright bomb error gets.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                # Format first: every later check trusts Pillow's parser,
                # and that trust is only justified where the container has
                # a completeness proof.
                if image.format not in STILL_FORMATS:
                    return ReviewRefusal.UNSUPPORTED_FORMAT
                # Byte-level completeness before any deeper parser
                # trust: the whole submitted byte string must be exactly
                # one complete container, proven by these bytes and not
                # by what Pillow tolerates - PNG to the consumption
                # proof, JPEG to the physical-EOI bar described above.
                if not _CONTAINER_FRAMERS[image.format](payload):
                    return ReviewRefusal.UNREADABLE_IMAGE
                # The lazy open reads only the header, so a truncated file
                # with an intact header still measured.
                # verify() walks the container structure - chunk lengths
                # and CRCs - which load() alone would tolerate at the tail.
                image.verify()
            # And verify() is not whole-stream proof either: a JPEG cut to
            # half its bytes still verified and gained usable authority.
            # load() decodes every pixel - the bomb guard above bounds
            # what that decode can cost - and it needs a fresh parse
            # because verify() consumes the first one.
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                # load() proves only the frame it decoded. A reference is
                # a single still, so visible frames beyond the first
                # refuse explicitly - which also catches a multi-frame
                # file truncated after its first frame, whose remaining
                # frame count is still above one.
                if getattr(image, "n_frames", 1) != 1:
                    return ReviewRefusal.MULTI_FRAME
    except Exception:
        # Deliberately everything: enumerating parser internals lost
        # twice - SyntaxError escaped a PNG chunk cut, then IndexError
        # and struct.error escaped GIF frame cuts, 54 raw exceptions
        # across 641 truncated prefixes. The boundary's contract is that
        # bytes it cannot prove are a typed refusal, never someone else's
        # traceback; there is no parser exception that should propagate
        # from here.
        return ReviewRefusal.UNREADABLE_IMAGE
    if width <= 0 or height <= 0:
        return ReviewRefusal.UNREADABLE_IMAGE
    return width, height


def _conflict_refusal(
    connection: Connection,
    *,
    asset_id: str,
    subject_id: str,
    expected_artifact_id: str,
    expected_version: int,
) -> ReviewRefused | None:
    """What a guarded settle's miss actually was - proven, not presumed.

    ALREADY_SETTLED is a claim about the row, so the row is read back rather
    than inferred from the failure: a conflict is the loser's refusal only
    when the target really settled or moved past the expected version. A row
    that vanished, changed subjects, or swapped its artifact binding is
    not-found - the asset as it was addressed and read no longer exists -
    and anything else is not this module's failure to report.
    """

    row = connection.execute(
        select(
            ReferenceAsset.validation_state,
            ReferenceAsset.review_version,
            ReferenceAsset.reference_subject_id,
            ReferenceAsset.artifact_id,
        ).where(ReferenceAsset.id == asset_id)
    ).one_or_none()
    if (
        row is None
        or row.reference_subject_id != subject_id
        or row.artifact_id != expected_artifact_id
    ):
        return ReviewRefused(ReviewRefusal.ASSET_NOT_FOUND)
    if (
        row.validation_state != ValidationState.UNCHECKED.value
        or row.review_version != expected_version
    ):
        return ReviewRefused(ReviewRefusal.ALREADY_SETTLED)
    return None


def review_asset(
    session: Session,
    subject: ReferenceSubject,
    *,
    asset_id: str,
    outcome: ReviewOutcome,
    reasons: Sequence[str] = (),
    read_bytes: Callable[[str], bytes] | None = None,
) -> ReviewedAsset | ReviewRefused:
    """Record a decision about one image, and for `usable`, verify it first.

    Membership is checked against the subject that was named rather than looked
    up from the asset, so a review cannot reach into a subject the caller did
    not address - the same rule `resolve_reference_requests` applies to pinned
    assets, for the same reason.

    The write is scoped to the reviewed asset and the caller's session is
    never flushed here: unrelated pending state the caller owns stays pending
    and fails, if it fails, at the caller's own flush - not inside this call
    wearing this module's label.
    """

    # The gate below compares by identity (`outcome is ReviewOutcome.USABLE`)
    # but the write went through `outcome.value`, so a value-equal impostor
    # skipped the measurement and still landed as usable. Strict mypy makes
    # that unreachable in checked code, but this boundary validates every
    # other input at runtime and must not trust this one.
    if type(outcome) is not ReviewOutcome:
        return ReviewRefused(ReviewRefusal.OUTCOME_INVALID)

    # Every query this call makes runs without autoflush: a lazy load or get
    # would otherwise flush the caller's unrelated dirty rows into the landed
    # triggers mid-call, surfacing the caller's own failure from inside this
    # one - before any containment exists.
    with session.no_autoflush:
        asset = session.get(ReferenceAsset, asset_id)
        if asset is None or asset.reference_subject_id != subject.id:
            return ReviewRefused(ReviewRefusal.ASSET_NOT_FOUND)

        cleaned = _clean_reasons(reasons)
        if cleaned is None:
            return ReviewRefused(ReviewRefusal.REASONS_MALFORMED)
        # A negative judgement without a reason is unreadable later: nobody
        # can tell whether the image was blurry, cropped, or of the wrong
        # person, so nobody can tell whether replacing it would help.
        if outcome is not ReviewOutcome.USABLE and not cleaned:
            return ReviewRefused(ReviewRefusal.REASONS_REQUIRED)

        # The landed review schema binds every settle to a measurement: the
        # update guard refuses NULL dimensions and a non-incremented
        # review_version for ANY decision, because a settled review event
        # must bind the exact pixels it judged. So every outcome now
        # requires the image, not only promotion - an image that cannot be
        # read cannot be settled either way; it stays unchecked with the
        # refusal as the record. TOO_SMALL stays a promotion-only refusal:
        # smallness can be exactly the reason a reviewer records weak or
        # rejected.
        if read_bytes is None:
            return ReviewRefused(ReviewRefusal.NO_READER)
        asset_key = asset.id
        # The artifact binding is captured once, beside the row key: the
        # bytes about to be read and judged belong to THIS artifact, and
        # the write below binds it so a mid-read artifact swap cannot get
        # the old bytes' measurement recorded against a replacement the
        # review never saw.
        artifact_key = asset.artifact_id
        expected_version = asset.review_version
        try:
            payload = read_bytes(artifact_key)
        except (OSError, KeyError, ValueError):
            # The production reader raises ValueError for checksum mismatch,
            # size mismatch, non-canonical, symlinked, and store-escaping
            # paths - exactly the corruption and tamper cases
            # UNREADABLE_IMAGE documents. Without it those raised out of the
            # authoritative path as a 500 instead of the typed refusal
            # this boundary promises.
            return ReviewRefused(ReviewRefusal.UNREADABLE_IMAGE)
        # The bytes judged must BE the artifact addressed: the event this
        # settle writes advertises exact-pixel evidence, and a callback
        # convention cannot establish that - the payload is hashed HERE
        # and must match the artifact row's digest, or the review refuses
        # before a single pixel is decoded.
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        connection = session.connection()
        stored_sha256 = connection.execute(
            select(Artifact.sha256).where(Artifact.id == artifact_key)
        ).scalar_one_or_none()
        if not payload or stored_sha256 != payload_sha256:
            return ReviewRefused(ReviewRefusal.UNREADABLE_IMAGE)
        measured = _measure(payload)
        if isinstance(measured, ReviewRefusal):
            return ReviewRefused(measured)
        if outcome is ReviewOutcome.USABLE and min(measured) < MIN_SHORT_EDGE:
            return ReviewRefused(ReviewRefusal.TOO_SMALL, measured=measured)

        settled = ValidationState(outcome.value)
        # One guarded UPDATE, scoped to exactly the reviewed row and never
        # the session's flush machinery: a session flush would carry the
        # caller's unrelated pending state along and a failure would poison
        # the caller's transaction. Containment for a single
        # statement is the database's own statement atomicity -
        # a trigger abort rolls back only this UPDATE and keeps the caller's
        # transaction open. A savepoint here would be worse than nothing:
        # pysqlite defers BEGIN past SELECTs, so releasing a savepoint
        # opened before any DML can COMMIT the caller's transaction from
        # inside this call. The WHERE clause makes the ordinary two-reviewer
        # race a first-class miss rather than a trigger explosion.
        try:
            written = connection.execute(
                update(ReferenceAsset)
                .where(
                    ReferenceAsset.id == asset_key,
                    # The subject whose membership authorized this review is
                    # part of the predicate, not just the pre-check: an asset
                    # re-parented mid-call would otherwise be settled under
                    # the OLD subject's authority - the settle statement does
                    # not change reference_subject_id, so no trigger stops
                    # it, and the conflict readback only runs after a miss.
                    # The guard makes the re-parent race miss.
                    ReferenceAsset.reference_subject_id == subject.id,
                    # And the artifact binding: the identity trigger permits
                    # rebinding an unchecked row, so without this a mid-read
                    # artifact swap would record the OLD bytes' measurement
                    # against the replacement.
                    ReferenceAsset.artifact_id == artifact_key,
                    ReferenceAsset.validation_state == ValidationState.UNCHECKED.value,
                )
                .values(
                    width=measured[0],
                    height=measured[1],
                    validation_state=settled.value,
                    validation_reasons_json=list(cleaned),
                    review_version=expected_version + 1,
                )
            )
        except IntegrityError:
            session.expire(asset, REVIEW_OWNED_ATTRIBUTES)
            refusal = _conflict_refusal(
                connection,
                asset_id=asset_key,
                subject_id=subject.id,
                expected_artifact_id=artifact_key,
                expected_version=expected_version,
            )
            if refusal is None:
                # The row is exactly as this call found it, so the conflict
                # was not a settle race and mislabeling it would bury a real
                # failure. It stays whose it was.
                raise
            return refusal
        if written.rowcount != 1:
            session.expire(asset, REVIEW_OWNED_ATTRIBUTES)
            refusal = _conflict_refusal(
                connection,
                asset_id=asset_key,
                subject_id=subject.id,
                expected_artifact_id=artifact_key,
                expected_version=expected_version,
            )
            # A guarded miss can only mean the row settled, changed
            # subjects, or vanished - all of which classify.
            assert refusal is not None
            return refusal
        # The decision is not durable until its immutable event is: a
        # settle that commits without one leaves the schema's promised
        # review evidence blank. The event rides the SAME caller
        # transaction as the guarded settle, and the digest it
        # records is the hash of the judged payload, already proven equal
        # to the artifact row's digest before a pixel was decoded - so the
        # landed insert guard's row-digest comparison and the advertised
        # exact-pixel evidence are the same claim.
        decision_material = json.dumps(
            {
                "reference_asset_id": asset_key,
                "artifact_sha256": payload_sha256,
                "expected_state": ValidationState.UNCHECKED.value,
                "expected_version": expected_version,
                "decision": settled.value,
                "reasons": list(cleaned),
                "width": measured[0],
                "height": measured[1],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        decision_sha256 = hashlib.sha256(decision_material.encode("utf-8")).hexdigest()
        try:
            connection.execute(
                insert(ReferenceAssetReviewEvent).values(
                    id=f"refreview:sha256:{decision_sha256}",
                    reference_asset_id=asset_key,
                    artifact_id=artifact_key,
                    artifact_sha256=payload_sha256,
                    reviewer_kind="local-human",
                    expected_state=ValidationState.UNCHECKED.value,
                    expected_version=expected_version,
                    result_version=expected_version + 1,
                    decision=settled.value,
                    reasons_json=list(cleaned),
                    width=measured[0],
                    height=measured[1],
                    decision_sha256=decision_sha256,
                )
            )
        except BaseException:
            # The settle already succeeded in this transaction; if its
            # event cannot be written, the pair must be failure-atomic -
            # a raw raise alone left the session committable and an
            # observer then saw a settled asset with zero events.
            # Invalidating the connection kills the wire transaction,
            # so settle and event vanish together; the caller's
            # rollback reconnects and the session is usable again
            # afterwards.
            connection.invalidate()
            raise
        session.expire(asset, REVIEW_OWNED_ATTRIBUTES)
    return ReviewedAsset(
        reference_asset_id=asset_key,
        validation_state=settled,
        reasons=cleaned,
        width=measured[0],
        height=measured[1],
    )
