"""Build a neutral grown store and time the artifact reference checks on it.

A long-lived installation holds about a hundred thousand artifacts and tens of
thousands of messages, and every check that proves an artifact unreferenced
reads all of them. This script builds a store of that shape from synthetic rows,
so those checks can be timed without anyone's data:

    build         make a store at a given scale in a new or empty folder
    engine-facts  record how this SQLite build behaves where the checks rely on it
    calibrate     time the reference checks on a store this script built

Every value is synthetic: generated identifiers, seeded hexadecimal padding and
numbers, and every artifact file is sixteen random bytes. The schema comes from
the application's own migrations, and rows go in through the application's own
database engine with every trigger and foreign key enforced, so a row its
database would refuse is refused here as well. Output is counts and timings
only.

A build refuses a folder that already holds anything. Both commands refuse the
folder an installed application keeps its data in, however it is spelled, and
any folder inside it or around it. Calibrate also refuses a folder a build did
not finish, and a store whose database or files are reached through a link, so
neither command can open an installation's data.
"""

from __future__ import annotations

import argparse
import bisect
import functools
import hashlib
import json
import os
import platform
import random
import shutil
import sqlite3
import stat
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from local_lm.config import Settings
    from sqlalchemy import Engine

REPOSITORY = Path(__file__).resolve().parents[1]

#: Written last by a finished build. Calibrate opens no folder without it.
MARKER = "reference-benchmark.json"
#: Raised whenever the shape changes, so calibrate refuses a store of an older one.
MARKER_FORMAT = 2
#: The store's own folders and database. Calibrate refuses a store in which any
#: of them is a link, or leads outside the store's folder.
STORE_FOLDERS = ("state", "artifacts", "artifacts/sha256")
DATABASE = "state/local-lm.sqlite3"
#: The files SQLite keeps beside a database it has open.
DATABASE_COMPANIONS = ("-wal", "-shm", "-journal")

# Rows at scale 1.0, population by population. Every other scale multiplies
# each count and rounds it to whole rows, keeping at least one, so every source
# the reference checks read holds a row at any scale. Totals follow from these.

#: Pictures no job made, such as edits saved in place, held only by their
#: library entries.
FULL_OTHER_IMAGES = 1_250
#: Videos. Each names its poster, a thumbnail made after it, through its own
#: metadata and nowhere else.
FULL_VIDEOS = 242
#: Uploads something names, and the few that nothing names.
FULL_REFERENCED_INPUTS = 103
FULL_UNREFERENCED_INPUTS = 5
#: Temporary previews, which nothing names, by age in days: rows, youngest, oldest.
FULL_PREVIEWS = (
    (700, 0.01, 0.95),
    (16_000, 1.05, 6.95),
    (31_000, 7.05, 29.95),
    (31_163, 30.05, 89.95),
)
FULL_CHATS = 2_879
#: Studio sessions among those chats, each opened on a picture it names.
FULL_STUDIO_CHATS = 138
#: A person's message and the work plan it started.
FULL_TURNS = 17_005
#: A step of a turn's plan, with its run, its work step and its own reply.
FULL_RUNS = 17_789
#: Runs that made a second picture, each with a job of its own.
FULL_SECOND_OUTPUTS = 1_174
#: Runs whose provenance lists their resolved dependencies; the oldest do not.
FULL_DEPENDENCY_RUNS = 17_761
#: Runs, and the work steps that ran them, with a mask in their settings.
FULL_MASKS = 12
#: Uploads attached to a turn's message, which its first run takes as input.
FULL_ATTACHMENTS = 182
FULL_MESSAGE_PARTS = 52_686
FULL_REVISIONS = 17_788
FULL_REVISION_PARTS = 35_374
FULL_REVISION_MEDIA_PARTS = 17_665
#: Sources a grown store barely uses or does not use at all. Each still gets at
#: least one row, so the checks read every source they would read anywhere.
FULL_MESSAGE_REFERENCES = 0
FULL_RUN_CONTEXTS = 1
FULL_REFERENCE_ASSETS = 1
FULL_SETUP_VERIFICATIONS = 0
FULL_DRAFT_ATTACHMENTS = 0
FULL_REGISTRY_REVIEWS = 0

#: How far back the turns reach, and how they are timed.
HISTORY = timedelta(days=365)
RUN_SECONDS = timedelta(seconds=60)
STEP_SPACING = timedelta(seconds=75)

#: A run's provenance document, padded to this many characters: about 100 MB
#: across the runs of a full-scale store.
DEFAULT_PROVENANCE_BYTES = 5_600
#: How many strings that padding holds. The checks pay per JSON node far more
#: than per character, so this sets their cost while the size stays put.
DEFAULT_PROVENANCE_ITEMS = 385
#: The settings a run and its work step carry, padded to this many characters.
DEFAULT_SETTINGS_BYTES = 1_200
#: How many strings the settings padding holds.
DEFAULT_SETTINGS_ITEMS = 15
#: Every picture's and video's metadata document, padded to this many
#: characters: the application stores a copy of the settings that made one
#: beside it. Previews, posters and uploads keep the few keys the application
#: writes for them.
DEFAULT_METADATA_BYTES = 10_700
#: How many strings the metadata padding holds.
DEFAULT_METADATA_ITEMS = 345
#: The kinds whose metadata is padded.
PADDED_KINDS = ("image", "video")
#: The most strings one padding object holds, under the 4,096 members the
#: application allows in any one JSON container.
MAX_PAD_ITEMS = 4_000

ARTIFACT_BYTES = 16

#: The tables whose rows the reference walk counts against its row bound.
COUNTED_TABLES = (
    "message_parts",
    "response_revision_parts",
    "reference_subjects",
    "reference_assets",
    "setup_verifications",
    "artifact_library_entries",
    "message_references",
    "comfy_registry_source_artifact_reviews",
    "runs",
    "run_context_artifacts",
    "work_steps",
    "chats",
    "chat_composer_draft_attachments",
    "jobs",
)

#: The tables a build fills, in the order their foreign keys allow.
FILLED_TABLES = (
    "artifacts",
    "chats",
    "chat_composer_drafts",
    "chat_composer_draft_attachments",
    "messages",
    "message_parts",
    "reference_subjects",
    "reference_assets",
    "message_references",
    "work_plans",
    "work_steps",
    "runs",
    "response_revisions",
    "response_revision_parts",
    "jobs",
    "run_context_snapshots",
    "run_context_artifacts",
    "model_installs",
    "model_profiles",
    "setup_verifications",
    "comfy_registry_source_artifact_reviews",
    "artifact_library_entries",
)


class StoreRefused(RuntimeError):
    """The folder cannot hold or supply a benchmark store."""


@dataclass(frozen=True)
class StoreShape:
    """How many rows of each kind a store holds at one scale."""

    other_images: int
    videos: int
    referenced_inputs: int
    unreferenced_inputs: int
    previews: tuple[int, ...]
    chats: int
    studio_chats: int
    turns: int
    runs: int
    second_outputs: int
    dependency_runs: int
    masks: int
    attachments: int
    message_parts: int
    revisions: int
    revision_parts: int
    revision_media_parts: int
    message_references: int
    run_contexts: int
    reference_assets: int
    setup_verifications: int
    draft_attachments: int
    registry_reviews: int

    @classmethod
    def at(cls, scale: float) -> StoreShape:
        if not 0 < scale <= 4:
            raise StoreRefused("the scale must be above 0 and at most 4")

        def scaled(full: int) -> int:
            return max(1, round(full * scale))

        return cls(
            other_images=scaled(FULL_OTHER_IMAGES),
            videos=scaled(FULL_VIDEOS),
            referenced_inputs=scaled(FULL_REFERENCED_INPUTS),
            unreferenced_inputs=scaled(FULL_UNREFERENCED_INPUTS),
            previews=tuple(scaled(rows) for rows, _youngest, _oldest in FULL_PREVIEWS),
            chats=scaled(FULL_CHATS),
            studio_chats=scaled(FULL_STUDIO_CHATS),
            turns=scaled(FULL_TURNS),
            runs=scaled(FULL_RUNS),
            second_outputs=scaled(FULL_SECOND_OUTPUTS),
            dependency_runs=scaled(FULL_DEPENDENCY_RUNS),
            masks=scaled(FULL_MASKS),
            attachments=scaled(FULL_ATTACHMENTS),
            message_parts=scaled(FULL_MESSAGE_PARTS),
            revisions=scaled(FULL_REVISIONS),
            revision_parts=scaled(FULL_REVISION_PARTS),
            revision_media_parts=scaled(FULL_REVISION_MEDIA_PARTS),
            message_references=scaled(FULL_MESSAGE_REFERENCES),
            run_contexts=scaled(FULL_RUN_CONTEXTS),
            reference_assets=scaled(FULL_REFERENCE_ASSETS),
            setup_verifications=scaled(FULL_SETUP_VERIFICATIONS),
            draft_attachments=scaled(FULL_DRAFT_ATTACHMENTS),
            registry_reviews=scaled(FULL_REGISTRY_REVIEWS),
        )

    @property
    def jobs(self) -> int:
        """A job for every output: one per run, and another for each second picture."""
        return self.runs + self.second_outputs

    @property
    def images(self) -> int:
        return self.jobs - self.videos + self.other_images

    @property
    def inputs(self) -> int:
        return self.referenced_inputs + self.unreferenced_inputs

    @property
    def thumbnails(self) -> int:
        """A poster for every video, and every temporary preview."""
        return self.videos + sum(self.previews)

    @property
    def reply_extra_parts(self) -> int:
        """Replies with a part besides their media, once each turn has its text part."""
        return self.message_parts - self.attachments - self.runs - self.turns

    def check(self) -> None:
        refusals = (
            (self.videos >= self.runs, "more videos than runs after the first"),
            (self.second_outputs > self.runs - self.videos, "more second pictures than runs"),
            (self.masks >= self.runs - self.videos, "more masks than picture runs"),
            (not self.turns <= self.runs <= 2 * self.turns, "runs its turns cannot hold"),
            (self.studio_chats >= self.chats, "no chat besides its studio sessions"),
            (self.studio_chats > self.jobs - self.videos, "more studio sessions than pictures"),
            (self.attachments > self.turns, "more attachments than turns"),
            (not 0 <= self.reply_extra_parts <= self.runs, "parts its messages cannot hold"),
            (self.revisions > self.runs, "more revisions than runs"),
            (
                not self.revision_media_parts
                <= self.revisions
                <= self.revision_parts
                <= self.revisions + self.revision_media_parts,
                "revision parts its revisions cannot hold",
            ),
            (self.dependency_runs > self.runs, "more listed dependencies than runs"),
            (self.message_references > self.turns, "more message references than turns"),
            (self.run_contexts > self.attachments, "more kept contexts than attachments"),
            (
                max(self.reference_assets, self.registry_reviews) > self.referenced_inputs,
                "more reference assets or registry reviews than uploads",
            ),
            (
                self.draft_attachments > self.chats - self.studio_chats,
                "more drafts than chats",
            ),
        )
        for refused, reason in refusals:
            if refused:
                raise StoreRefused(f"this scale has {reason}")


@dataclass
class _Artifact:
    digest: str
    content: bytes
    kind: str
    media_type: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)
    #: A video's poster, which the video names by an update once both are stored.
    poster: _Artifact | None = None

    @property
    def id(self) -> str:
        return f"sha256:{self.digest}"


@dataclass
class _Chat:
    chat_id: str
    created_at: datetime
    #: The picture a studio session was opened on, and the chat that made it.
    source: _Artifact | None = None
    source_chat_id: str | None = None
    turns: int = 0


@dataclass
class _Turn:
    """A person's message in a chat, and the work plan it started."""

    plan_id: str
    message_id: str
    chat_id: str
    sequence: int
    created_at: datetime
    part_id: str
    attachment: _Artifact | None = None
    attachment_part_id: str | None = None


@dataclass
class _Revision:
    revision_id: str
    media_part_id: str | None
    other_part_id: str | None


@dataclass
class _Run:
    """A step of a turn's plan: its run, work step, jobs and reply."""

    run_id: str
    step_id: str
    message_id: str
    turn: _Turn
    ordinal: int
    created_at: datetime
    outputs: list[_Artifact]
    job_ids: list[str]
    inputs: list[str]
    mask: str | None
    seed: int
    steps: int
    media_part_id: str
    lists_dependencies: bool = True
    other_part_id: str | None = None
    revision: _Revision | None = None

    @property
    def finished_at(self) -> datetime:
        return self.created_at + RUN_SECONDS

    @property
    def operation(self) -> str:
        if self.outputs[0].kind == "video":
            return "image_to_video"
        return "image_to_image" if self.inputs else "text_to_image"


@dataclass
class _Plan:
    now: datetime
    artifacts: list[_Artifact]
    chats: list[_Chat]
    runs: list[_Run]
    library: list[_Artifact]
    references: list[tuple[_Turn, _Artifact]]
    contexts: list[_Run]
    verifications: list[_Artifact]
    drafts: list[tuple[_Chat, _Artifact]]
    assets: list[_Artifact]
    reviews: list[_Artifact]


class _Rotation:
    """Hands out a pool's members in order, starting over when it runs out.

    Every member is named at least once as long as the pool is handed out at
    least as many times as it has members, which is what keeps every upload
    that should be retained named by something.
    """

    def __init__(self, pool: Sequence[_Artifact]) -> None:
        self._pool = pool
        self.handed = 0

    def take(self) -> _Artifact:
        member = self._pool[self.handed % len(self._pool)]
        self.handed += 1
        return member

    @property
    def covered(self) -> bool:
        return self.handed >= len(self._pool)


class _Identifiers:
    """Identifiers shaped like the application's: a prefix and thirty-two hex digits.

    The digits are seeded random ones, so rows spread across each index the
    way the application's random identifiers do, and they end in a counter,
    so no two collide.
    """

    def __init__(self, source: random.Random) -> None:
        self._source = source
        self._counts: dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        count = self._counts.get(prefix, 0)
        self._counts[prefix] = count + 1
        return f"{prefix}_{self._source.getrandbits(96):024x}{count:08x}"


def _stamp(moment: datetime) -> str:
    """A UTC time written the way the application's database layer writes one."""
    return moment.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


def _hex(source: random.Random, length: int) -> str:
    return source.randbytes((length + 1) // 2).hex()[:length]


def _padding(
    document: dict[str, object], source: random.Random, *, size: int, items: int
) -> dict[str, object]:
    """`document` with seeded hexadecimal strings added until its JSON is `size` characters.

    The padding is one flat object of up to `items` strings, so `items` sets
    how many JSON nodes the document carries while `size` sets its length.
    When that many strings cannot fit, as many as fit are used, and a document
    with no room left gets no padding at all.
    """

    base = len(json.dumps({**document, "pad": {}}))
    # Every member is '"v0000": ""' plus its characters, eleven fixed
    # characters, and members after the first add a comma and a space. A
    # string carries at least one character, so a member needs fourteen.
    count = max(0, min(items, MAX_PAD_ITEMS, (size - base + 2) // 14))
    if count == 0:
        return dict(document)
    characters = size - base - 13 * count + 2
    padded: dict[str, object] = {
        **document,
        "pad": {
            f"v{slot:04d}": _hex(source, characters // count + (slot < characters % count))
            for slot in range(count)
        },
    }
    if len(json.dumps(padded)) != size:
        raise RuntimeError("the padding did not reach its size")
    return padded


def _plan_store(shape: StoreShape, *, seed: int, now: datetime) -> _Plan:
    """Decide every row of the store before anything is written."""

    shape.check()
    clock = random.Random(f"{seed}:clock")
    choices = random.Random(f"{seed}:choices")
    files = random.Random(f"{seed}:files")
    identifier = _Identifiers(random.Random(f"{seed}:names"))
    digests: set[str] = set()

    def artifact(kind: str, media_type: str, created_at: datetime, **metadata: object) -> _Artifact:
        while True:
            content = files.randbytes(ARTIFACT_BYTES)
            digest = hashlib.sha256(content).hexdigest()
            if digest not in digests:
                digests.add(digest)
                return _Artifact(digest, content, kind, media_type, created_at, metadata)

    def days_ago(youngest: float, oldest: float) -> datetime:
        return now - timedelta(days=clock.uniform(youngest, oldest))

    first_turn = now - HISTORY
    # Every upload something names was made before the first turn, so it is
    # older than anything that names it.
    named_uploads = [
        artifact(
            "input",
            "image/png",
            first_turn - timedelta(hours=clock.uniform(1, 23)),
            uploaded=True,
        )
        for _ in range(shape.referenced_inputs)
    ]
    stray_uploads = [
        artifact("input", "image/png", days_ago(0.5, 365), uploaded=True)
        for _ in range(shape.unreferenced_inputs)
    ]
    other_images = [
        artifact("image", "image/png", days_ago(0.5, 365)) for _ in range(shape.other_images)
    ]
    uploads = _Rotation(named_uploads)
    chats = [
        _Chat(identifier("chat"), days_ago(0.5, 365))
        for _ in range(shape.chats - shape.studio_chats)
    ]

    attached = set(choices.sample(range(shape.turns), shape.attachments))
    two_steps = set(choices.sample(range(shape.turns), shape.runs - shape.turns))
    # The first run makes a picture, so every later video and masked edit has
    # an earlier picture to start from.
    videos = set(choices.sample(range(1, shape.runs), shape.videos))
    pictures = [index for index in range(shape.runs) if index not in videos]
    twins = set(choices.sample(pictures, shape.second_outputs))
    masked = set(choices.sample(pictures[1:], shape.masks))

    runs: list[_Run] = []
    produced: list[_Artifact] = []
    made_in: dict[str, str] = {}
    latest_picture: _Artifact | None = None
    slot = (HISTORY - timedelta(hours=2)) / shape.turns
    for number in range(shape.turns):
        # Each turn starts in the first half of its own slot of the history,
        # so it has finished before the next one starts.
        created_at = first_turn + slot * (number + clock.random() / 2)
        chat = chats[choices.randrange(len(chats))]
        chat.turns += 1
        if chat.turns == 1:
            chat.created_at = created_at - timedelta(minutes=1)
        turn = _Turn(
            plan_id=identifier("plan"),
            message_id=identifier("msg"),
            chat_id=chat.chat_id,
            sequence=chat.turns,
            created_at=created_at,
            part_id=identifier("part"),
        )
        if number in attached:
            turn.attachment = uploads.take()
            turn.attachment_part_id = identifier("part")
        for ordinal in range(1, 3 if number in two_steps else 2):
            index = len(runs)
            run_id = identifier("run")
            started = created_at + timedelta(seconds=2) + (ordinal - 1) * STEP_SPACING
            finished = started + RUN_SECONDS
            outputs: list[_Artifact]
            if index in videos:
                video = artifact("video", "video/mp4", finished, run_id=run_id)
                poster = artifact(
                    "thumbnail",
                    "image/jpeg",
                    finished + timedelta(seconds=2),
                    run_id=run_id,
                    poster_for=video.id,
                )
                video.poster = poster
                outputs = [video]
                produced += [video, poster]
            else:
                outputs = [
                    artifact(
                        "image", "image/png", finished + timedelta(seconds=output), run_id=run_id
                    )
                    for output in range(2 if index in twins else 1)
                ]
                produced += outputs
            if turn.attachment is not None and ordinal == 1:
                inputs = [turn.attachment.id]
            elif (index in videos or index in masked) and latest_picture is not None:
                inputs = [latest_picture.id]
            else:
                inputs = []
            runs.append(
                _Run(
                    run_id=run_id,
                    step_id=identifier("step"),
                    message_id=identifier("msg"),
                    turn=turn,
                    ordinal=ordinal,
                    created_at=started,
                    outputs=outputs,
                    job_ids=[identifier("job") for _ in outputs],
                    inputs=inputs,
                    mask=uploads.take().id if index in masked else None,
                    seed=choices.getrandbits(31),
                    steps=choices.randrange(4, 60),
                    media_part_id=identifier("part"),
                )
            )
            for output in outputs:
                made_in[output.id] = chat.chat_id
            if outputs[-1].kind == "image":
                latest_picture = outputs[-1]

    # The oldest runs predate listing resolved dependencies; none of them took
    # an input, so every run with one still lists it.
    unlisted = shape.runs - shape.dependency_runs
    for run in runs:
        if unlisted and not run.inputs:
            run.lists_dependencies = False
            unlisted -= 1
    if unlisted:
        raise StoreRefused("this scale has too few runs without inputs")

    # Every reply has its media part; the earliest also carry a second part.
    # Revisions follow the same rule, and those without media keep one part.
    both = shape.revision_parts - shape.revisions
    for index, run in enumerate(runs):
        if index < shape.reply_extra_parts:
            run.other_part_id = identifier("part")
        if index < shape.revisions:
            media = index < shape.revision_media_parts
            run.revision = _Revision(
                revision_id=identifier("rev"),
                media_part_id=identifier("rpart") if media else None,
                other_part_id=identifier("rpart") if not media or index < both else None,
            )

    job_pictures = [output for run in runs for output in run.outputs if output.kind == "image"]
    studios = [
        _Chat(
            identifier("chat"),
            source.created_at + timedelta(minutes=1),
            source=source,
            source_chat_id=made_in[source.id],
        )
        for source in choices.sample(job_pictures, shape.studio_chats)
    ]

    # A preview names the run it was made for, the latest one started before it.
    starts = [run.created_at for run in runs]
    previews: list[_Artifact] = []
    for (_rows, youngest, oldest), rows in zip(FULL_PREVIEWS, shape.previews, strict=True):
        for _ in range(rows):
            created_at = days_ago(youngest, oldest)
            maker = runs[max(0, bisect.bisect_right(starts, created_at) - 1)]
            previews.append(
                artifact(
                    "thumbnail",
                    "image/jpeg",
                    created_at,
                    temporary_preview=True,
                    run_id=maker.run_id,
                )
            )

    first_runs = [run for run in runs if run.ordinal == 1 and run.turn.attachment is not None]
    turns = [run.turn for run in runs if run.ordinal == 1]
    references = [(turns[number], uploads.take()) for number in range(shape.message_references)]
    verifications = [uploads.take() for _ in range(shape.setup_verifications)]
    drafts = [(chats[-1 - number], uploads.take()) for number in range(shape.draft_attachments)]
    assets = [uploads.take() for _ in range(shape.reference_assets)]
    reviews = [uploads.take() for _ in range(shape.registry_reviews)]
    if not uploads.covered:
        raise StoreRefused("this scale leaves uploads that nothing names")

    artifacts = sorted(
        named_uploads + stray_uploads + other_images + produced + previews,
        key=lambda item: item.created_at,
    )
    return _Plan(
        now=now,
        artifacts=artifacts,
        chats=chats + studios,
        runs=runs,
        library=[item for item in artifacts if item.kind in ("image", "video")],
        references=references,
        contexts=first_runs[: shape.run_contexts],
        verifications=verifications,
        drafts=drafts,
        assets=assets,
        reviews=reviews,
    )


@dataclass(frozen=True)
class Padding:
    """How large stored documents are, and how many JSON nodes they carry."""

    provenance_bytes: int = DEFAULT_PROVENANCE_BYTES
    provenance_items: int = DEFAULT_PROVENANCE_ITEMS
    settings_bytes: int = DEFAULT_SETTINGS_BYTES
    settings_items: int = DEFAULT_SETTINGS_ITEMS
    metadata_bytes: int = DEFAULT_METADATA_BYTES
    metadata_items: int = DEFAULT_METADATA_ITEMS


Row = tuple[object, ...]


def _statements(
    plan: _Plan,
    relative: Callable[[str], str],
    *,
    seed: int,
    padding: Padding,
) -> list[tuple[str, str, Iterable[Row]]]:
    """Every write, in foreign-key order, with rows produced as they are written.

    Each padded document draws from a source of its own, so the same document
    comes out whenever and however often it is built.
    """

    identifier = _Identifiers(random.Random(f"{seed}:single-names"))
    values = random.Random(f"{seed}:single-values")
    subject_id = identifier("subj")
    install_id = identifier("install")
    profile_id = identifier("profile")
    first_turn = plan.runs[0].turn.created_at
    early = _stamp(first_turn - timedelta(minutes=30))
    recent = _stamp(plan.now - timedelta(hours=1))

    def metadata(item: _Artifact) -> dict[str, object]:
        if item.kind not in PADDED_KINDS:
            return dict(item.metadata)
        return _padding(
            item.metadata,
            random.Random(f"{seed}:metadata:{item.digest}"),
            size=padding.metadata_bytes,
            items=padding.metadata_items,
        )

    def settings(run: _Run) -> str:
        # A run and the step that ran it carry the same settings.
        document: dict[str, object] = {
            "seed": run.seed,
            "steps": run.steps,
            "width": 1024,
            "height": 1024,
        }
        if run.mask is not None:
            document["mask"] = {"artifact_id": run.mask}
        source = random.Random(f"{seed}:settings:{run.run_id}")
        padded = _padding(
            document, source, size=padding.settings_bytes, items=padding.settings_items
        )
        return json.dumps(padded)

    def provenance(run: _Run) -> str:
        document: dict[str, object] = {"input_artifact_ids": run.inputs}
        if run.lists_dependencies:
            document["resolved_dependency_artifact_ids"] = run.inputs
        # A video's poster is left out here: only the video's own metadata
        # names it, so the checks reach it through that link alone.
        document["outputs"] = [
            {
                "artifact_id": output.id,
                "poster_artifact_id": None,
                "browser_proxy_artifact_id": None,
            }
            for output in run.outputs
        ]
        source = random.Random(f"{seed}:provenance:{run.run_id}")
        padded = _padding(
            document, source, size=padding.provenance_bytes, items=padding.provenance_items
        )
        return json.dumps(padded)

    def artifacts() -> Iterator[Row]:
        for item in plan.artifacts:
            moment = _stamp(item.created_at)
            yield (
                item.id,
                item.digest,
                item.kind,
                item.media_type,
                len(item.content),
                relative(item.digest),
                None,
                json.dumps(metadata(item)),
                moment,
                moment,
                0,
            )

    def posters() -> Iterator[Row]:
        # As the application does: the video is stored, then its poster, and
        # then the video's metadata is updated to name the poster.
        for item in plan.artifacts:
            if item.poster is not None:
                named = {**metadata(item), "poster_artifact_id": item.poster.id}
                yield (json.dumps(named), _stamp(item.poster.created_at), item.id)

    def chats() -> Iterator[Row]:
        for number, chat in enumerate(plan.chats):
            moment = _stamp(chat.created_at)
            title = f"c{number:06d}"
            if chat.source is None:
                yield (chat.chat_id, title, 0, "standard", "auto", 1, "{}", moment, moment)
            else:
                origin = {
                    "source_artifact_id": chat.source.id,
                    "source_chat_id": chat.source_chat_id,
                }
                yield (
                    chat.chat_id,
                    title,
                    1,
                    "studio",
                    "image",
                    0,
                    json.dumps(origin),
                    moment,
                    moment,
                )

    def drafts() -> Iterator[Row]:
        for chat, _upload in plan.drafts:
            yield (chat.chat_id, recent, recent)

    def draft_attachments() -> Iterator[Row]:
        for chat, upload in plan.drafts:
            yield (chat.chat_id, upload.id)

    def messages() -> Iterator[Row]:
        for run in plan.runs:
            turn = run.turn
            if run.ordinal == 1:
                moment = _stamp(turn.created_at)
                yield (turn.message_id, turn.chat_id, None, "user", moment, moment)
            yield (
                run.message_id,
                turn.chat_id,
                turn.message_id,
                "assistant",
                _stamp(run.created_at),
                _stamp(run.finished_at),
            )

    def parts() -> Iterator[Row]:
        for run in plan.runs:
            turn = run.turn
            if run.ordinal == 1:
                moment = _stamp(turn.created_at)
                yield (turn.part_id, turn.message_id, 0, "text", "", None, moment, moment)
                if turn.attachment is not None:
                    yield (
                        turn.attachment_part_id,
                        turn.message_id,
                        1,
                        "image",
                        None,
                        turn.attachment.id,
                        moment,
                        moment,
                    )
            moment = _stamp(run.finished_at)
            media = run.outputs[0]
            yield (run.media_part_id, run.message_id, 0, media.kind, None, media.id, moment, moment)
            if run.other_part_id is not None:
                yield (
                    run.other_part_id,
                    run.message_id,
                    1,
                    "generation_metadata",
                    None,
                    None,
                    moment,
                    moment,
                )

    def subjects() -> Iterator[Row]:
        yield (subject_id, "n000000", "s000000", early, early)

    def assets() -> Iterator[Row]:
        for position, upload in enumerate(plan.assets):
            yield (identifier("asset"), subject_id, upload.id, position, early, early)

    def references() -> Iterator[Row]:
        for turn, upload in plan.references:
            moment = _stamp(turn.created_at)
            yield (
                identifier("msgref"),
                turn.message_id,
                subject_id,
                "s000000",
                "n000000",
                json.dumps([upload.id]),
                moment,
                moment,
            )

    def work_plans() -> Iterator[Row]:
        for run in plan.runs:
            if run.ordinal == 1:
                turn = run.turn
                moment = _stamp(turn.created_at)
                yield (turn.plan_id, turn.chat_id, turn.sequence, moment, moment)

    def steps() -> Iterator[Row]:
        for run in plan.runs:
            bindings = [{"type": "explicit_artifact", "artifact_id": value} for value in run.inputs]
            yield (
                run.step_id,
                run.turn.plan_id,
                run.ordinal,
                run.operation,
                settings(run),
                json.dumps(bindings),
                _stamp(run.created_at),
                _stamp(run.finished_at),
            )

    def runs() -> Iterator[Row]:
        for run in plan.runs:
            turn = run.turn
            yield (
                run.run_id,
                turn.chat_id,
                turn.message_id,
                run.message_id,
                turn.plan_id,
                run.step_id,
                run.operation,
                settings(run),
                provenance(run),
                _stamp(run.created_at + timedelta(seconds=1)),
                _stamp(run.finished_at),
                int((RUN_SECONDS - timedelta(seconds=1)).total_seconds() * 1000),
                _stamp(run.created_at),
                _stamp(run.finished_at),
            )

    def step_runs() -> Iterator[Row]:
        for run in plan.runs:
            yield (run.run_id, run.step_id)

    def revisions() -> Iterator[Row]:
        for run in plan.runs:
            if run.revision is not None:
                moment = _stamp(run.finished_at)
                yield (run.revision.revision_id, run.message_id, run.run_id, moment, moment)

    def revision_parts() -> Iterator[Row]:
        for run in plan.runs:
            revision = run.revision
            if revision is None:
                continue
            moment = _stamp(run.finished_at)
            media = run.outputs[0]
            if revision.media_part_id is not None:
                yield (
                    revision.media_part_id,
                    revision.revision_id,
                    0,
                    media.kind,
                    media.id,
                    moment,
                    moment,
                )
            if revision.other_part_id is not None:
                position = 0 if revision.media_part_id is None else 1
                yield (
                    revision.other_part_id,
                    revision.revision_id,
                    position,
                    "generation_metadata",
                    None,
                    moment,
                    moment,
                )

    def jobs() -> Iterator[Row]:
        # A media job's own documents: which output of the run it makes, and
        # the artifact it made.
        for run in plan.runs:
            queued = _stamp(run.created_at)
            finished = _stamp(run.finished_at)
            for index, (job_id, output) in enumerate(zip(run.job_ids, run.outputs, strict=True)):
                payload = {
                    "operation": run.operation,
                    "output_index": index,
                    "output_count": len(run.outputs),
                }
                yield (
                    job_id,
                    output.kind,
                    run.run_id,
                    run.turn.plan_id,
                    run.step_id,
                    queued,
                    json.dumps(payload),
                    json.dumps({"artifact_ids": [output.id]}),
                    _stamp(run.created_at + timedelta(seconds=1)),
                    finished,
                    queued,
                    finished,
                )

    def contexts() -> Iterator[Row]:
        for run in plan.contexts:
            payload = json.dumps({"artifact_ids": run.inputs})
            yield (run.run_id, payload, hashlib.sha256(payload.encode()).hexdigest())

    def context_artifacts() -> Iterator[Row]:
        for run in plan.contexts:
            yield (run.run_id, run.inputs[0])

    def installs() -> Iterator[Row]:
        yield (install_id, "m000000", f"models/{install_id}", early, early)

    def profiles() -> Iterator[Row]:
        yield (profile_id, install_id, "p000000", early, early)

    def verifications() -> Iterator[Row]:
        for upload in plan.verifications:
            yield (
                identifier("verify"),
                _hex(values, 64),
                install_id,
                profile_id,
                upload.id,
                recent,
                recent,
                recent,
                recent,
            )

    def reviews() -> Iterator[Row]:
        for upload in plan.reviews:
            declaration = _hex(values, 64)
            yield (
                identifier("review"),
                declaration,
                hashlib.sha256(declaration.encode()).hexdigest(),
                f"repo_{_hex(values, 32)}",
                _hex(values, 40),
                upload.id,
                upload.digest,
                len(upload.content),
                f"wheel_{_hex(values, 32)}",
                f"dist_{_hex(values, 32)}",
                "0.0.1",
                _hex(values, 64),
                recent,
                recent,
                recent,
            )

    def library() -> Iterator[Row]:
        for item in plan.library:
            moment = _stamp(item.created_at)
            yield (f"libentry:sha256:{item.digest}", item.id, item.digest, moment, moment)

    return [
        (
            "artifacts",
            "INSERT INTO artifacts (id, sha256, kind, media_type, size_bytes, relative_path, "
            "original_name, metadata_json, created_at, updated_at, favorite) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            artifacts(),
        ),
        (
            "artifact-posters",
            "UPDATE artifacts SET metadata_json = ?, updated_at = ? WHERE id = ?",
            posters(),
        ),
        (
            "chats",
            "INSERT INTO chats (id, title, archived, pinned, scope, draft_prompt, routing_mode, "
            "confirm_uncertain_media, generation_settings_json, generation_preset_ids_json, "
            "vision_settings_json, web_settings_json, origin_json, created_at, updated_at) "
            "VALUES (?, ?, ?, 0, ?, '', ?, ?, '{}', '{}', '{}', '{}', ?, ?, ?)",
            chats(),
        ),
        (
            "chat_composer_drafts",
            "INSERT INTO chat_composer_drafts (chat_id, revision, text, prompt_source_json, "
            "mode, output_count, mentions_json, template_settings_json, created_at, updated_at) "
            "VALUES (?, 1, '', NULL, 'auto', 1, '[]', NULL, ?, ?)",
            drafts(),
        ),
        (
            "chat_composer_draft_attachments",
            "INSERT INTO chat_composer_draft_attachments (chat_id, position, artifact_id, kind, "
            "origin) VALUES (?, 0, ?, 'image', 'uploaded')",
            draft_attachments(),
        ),
        (
            "messages",
            "INSERT INTO messages (id, chat_id, parent_id, role, status, transcript_visible, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, 'complete', 1, ?, ?)",
            messages(),
        ),
        (
            "message_parts",
            "INSERT INTO message_parts (id, message_id, position, type, text, artifact_id, "
            "metadata_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, '{}', ?, ?)",
            parts(),
        ),
        (
            "reference_subjects",
            "INSERT INTO reference_subjects (id, name, mention_slug, kind, description, "
            "aliases_json, tags_json, cover_artifact_id, favorite, archived, created_at, "
            "updated_at) VALUES (?, ?, ?, 'other', NULL, '[]', '[]', NULL, 0, 0, ?, ?)",
            subjects(),
        ),
        (
            "reference_assets",
            "INSERT INTO reference_assets (id, reference_subject_id, artifact_id, caption, "
            "purpose, view_label, sort_order, validation_state, validation_reasons_json, width, "
            "height, created_at, updated_at, review_version) "
            "VALUES (?, ?, ?, NULL, 'other', NULL, ?, 'unchecked', '[]', NULL, NULL, ?, ?, 1)",
            assets(),
        ),
        (
            "message_references",
            "INSERT INTO message_references (id, message_id, position, reference_subject_id, "
            "mention_slug, subject_name, subject_kind, source, reference_asset_ids_json, "
            "artifact_ids_json, created_at, updated_at) "
            "VALUES (?, ?, 0, ?, ?, ?, 'other', 'mention', '[]', ?, ?, ?)",
            references(),
        ),
        (
            "work_plans",
            "INSERT INTO work_plans (id, chat_id, source_action, persistence_scope, status, "
            "transcript_sequence, priority, planner_version, failure_policy, summary_json, "
            "created_at, updated_at) VALUES (?, ?, 'send', 'durable', 'complete', ?, 0, "
            "'legacy-turn-v1', 'stop_dependents', '{}', ?, ?)",
            work_plans(),
        ),
        (
            "work_steps",
            "INSERT INTO work_steps (id, plan_id, ordinal, operation, status, prompt, "
            "settings_json, input_bindings_json, output_contract_json, queue_class, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, 'complete', '', ?, ?, '[]', 'media_compute', ?, ?)",
            steps(),
        ),
        (
            "runs",
            "INSERT INTO runs (id, chat_id, user_message_id, assistant_message_id, "
            "work_plan_id, work_step_id, operation, status, standalone_prompt, settings_json, "
            "provenance_json, started_at, completed_at, duration_ms, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'complete', '', ?, ?, ?, ?, ?, ?, ?)",
            runs(),
        ),
        (
            "work-step-runs",
            "UPDATE work_steps SET run_id = ? WHERE id = ?",
            step_runs(),
        ),
        (
            "response_revisions",
            "INSERT INTO response_revisions (id, message_id, run_id, sequence, status, "
            "created_at, updated_at) VALUES (?, ?, ?, 1, 'complete', ?, ?)",
            revisions(),
        ),
        (
            "response_revision_parts",
            "INSERT INTO response_revision_parts (id, response_revision_id, position, type, "
            "text, artifact_id, metadata_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, NULL, ?, '{}', ?, ?)",
            revision_parts(),
        ),
        (
            "jobs",
            "INSERT INTO jobs (id, kind, status, run_id, work_plan_id, work_step_id, progress, "
            "phase, progress_json, queue_priority, enqueued_at, payload_json, result_json, "
            "attempt, cancellable, started_at, completed_at, created_at, updated_at) "
            "VALUES (?, ?, 'complete', ?, ?, ?, 1.0, 'complete', '{}', 0, ?, ?, ?, 1, 0, ?, ?, "
            "?, ?)",
            jobs(),
        ),
        (
            "run_context_snapshots",
            "INSERT INTO run_context_snapshots (run_id, payload_json, sha256) VALUES (?, ?, ?)",
            contexts(),
        ),
        (
            "run_context_artifacts",
            "INSERT INTO run_context_artifacts (run_id, artifact_id) VALUES (?, ?)",
            context_artifacts(),
        ),
        (
            "model_installs",
            "INSERT INTO model_installs (id, source_id, name, role, engine, local_path, "
            "size_bytes, compatibility, manifest_json, active, created_at, updated_at) "
            "VALUES (?, NULL, ?, 'image', 'mock', ?, 16, 'likely', '{}', 0, ?, ?)",
            installs(),
        ),
        (
            "model_profiles",
            "INSERT INTO model_profiles (id, model_install_id, name, role, engine, "
            "load_settings_json, request_settings_json, is_default, created_at, updated_at) "
            "VALUES (?, ?, ?, 'image', 'mock', '{}', '{}', 0, ?, ?)",
            profiles(),
        ),
        (
            "setup_verifications",
            "INSERT INTO setup_verifications (id, role, evidence_key, state, model_install_id, "
            "profile_id, input_artifact_id, started_at, completed_at, created_at, updated_at) "
            "VALUES (?, 'image', ?, 'ready', ?, ?, ?, ?, ?, ?, ?)",
            verifications(),
        ),
        (
            "comfy_registry_source_artifact_reviews",
            "INSERT INTO comfy_registry_source_artifact_reviews (id, source_declaration, "
            "source_declaration_sha256, repository, source_commit, artifact_id, artifact_sha256, "
            "artifact_size_bytes, wheel_filename, wheel_distribution, wheel_version, "
            "evidence_json, reviewer_kind, review_sha256, reviewed_at, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', 'local-human', ?, ?, ?, ?)",
            reviews(),
        ),
        (
            "artifact_library_entries",
            "INSERT INTO artifact_library_entries (id, artifact_id, display_name, favorite, "
            "state, version, created_at, updated_at) VALUES (?, ?, ?, 0, 'visible', 1, ?, ?)",
            library(),
        ),
    ]


def _is_link(path: Path) -> bool:
    """Whether `path` is a symbolic link, a junction or another reparse point.

    The entry itself is inspected and never followed. An absent entry is not a link.
    """

    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    return stat.S_ISLNK(info.st_mode) or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _comparable(path: Path) -> tuple[str, ...]:
    """The parts of `path` as the system finds it, ready to compare with another's.

    Links are followed as far as the path exists, a verbatim prefix such as
    \\\\?\\ is removed, and case is folded where the system ignores it. This
    reads only where each folder leads, never what it holds.
    """

    text = os.path.realpath(path)
    for prefix, replacement in (("\\\\?\\UNC\\", "\\\\"), ("\\\\?\\", "")):
        if text.startswith(prefix):
            text = replacement + text[len(prefix) :]
            break
    return Path(os.path.normcase(text)).parts


def _identities(path: Path) -> set[tuple[int, int]]:
    """The volume and file number of `path` and of every folder above it that exists.

    A file system that reports no file number is left out rather than matched.
    """

    found: set[tuple[int, int]] = set()
    for candidate in (path, *path.parents):
        try:
            info = os.stat(candidate)
        except OSError:
            continue
        if info.st_ino:
            found.add((info.st_dev, info.st_ino))
    return found


def _overlaps(first: Path, second: Path) -> bool:
    """Whether one folder is the other or lies inside it, however each is spelled.

    Spellings are compared once resolved, and so are the folders themselves by
    volume and file number, which also sees an alias no spelling reveals, such
    as a share or a mount of the same folder.
    """

    one, other = _comparable(first), _comparable(second)
    shorter = min(len(one), len(other))
    if one[:shorter] == other[:shorter]:
        return True
    for inner, outer in ((first, second), (second, first)):
        try:
            info = os.stat(outer)
        except OSError:
            continue
        if info.st_ino and (info.st_dev, info.st_ino) in _identities(inner):
            return True
    return False


def _inside(entry: Path, folder: Path) -> bool:
    """Whether `entry` lies within `folder` once both are resolved."""

    inner, outer = _comparable(entry), _comparable(folder)
    return len(inner) > len(outer) and inner[: len(outer)] == outer


def _refuse_occupied(target: Path) -> None:
    """Refuse anything but a new folder or an empty real one, before touching it."""

    if _is_link(target):
        raise StoreRefused("the data folder must not be a link")
    if target.exists():
        if not target.is_dir():
            raise StoreRefused("the data folder must be a folder")
        if any(target.iterdir()):
            raise StoreRefused("the data folder already holds files; give a new or empty one")


def _refuse_installed(target: Path, *, installed: Path | None = None) -> None:
    """Refuse the folder an installed application keeps its data in, and any folder in or around it.

    `installed` is that folder; when it is not given, the one an installed
    application uses for this account.
    """

    if installed is None:
        from local_lm.desktop import default_data_dir

        installed = default_data_dir()
    if _overlaps(target, installed):
        raise StoreRefused("the data folder overlaps an installed application's data")


def _settings(data_dir: Path) -> Settings:
    from local_lm.config import Settings

    return Settings(
        data_dir=data_dir,
        comfy_directory=None,
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )


def _count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def build_store(
    data_dir: Path,
    *,
    scale: float,
    seed: int = 1,
    padding: Padding | None = None,
    report: Callable[[str], None] = print,
    installed: Path | None = None,
) -> dict[str, int]:
    """Build a store of the shape at `scale` in `data_dir` and return its row counts.

    The folder must be new or empty, and it is named before anything is
    written, so a build that stops part way is easy to find. The schema comes
    from the application's migrations, and rows are inserted through its
    engine, so its foreign keys and every trigger judge each row. `installed`
    stands in for an installed application's data folder.
    """

    _refuse_installed(data_dir, installed=installed)
    _refuse_occupied(data_dir)
    padding = padding or Padding()
    shape = StoreShape.at(scale)
    shape.check()
    report(f"data-dir {data_dir}")
    started = time.perf_counter()
    plan = _plan_store(shape, seed=seed, now=datetime.now(UTC))
    report(
        f"plan scale={scale:g} seed={seed} artifacts={len(plan.artifacts)} "
        f"images={shape.images} inputs={shape.inputs} videos={shape.videos} "
        f"thumbnails={shape.thumbnails} previews={sum(shape.previews)} chats={shape.chats} "
        f"studio-chats={shape.studio_chats} turns={shape.turns} runs={shape.runs} "
        f"jobs={shape.jobs} masks={shape.masks} seconds={time.perf_counter() - started:.2f}"
    )

    from local_lm.artifacts import ArtifactStore
    from local_lm.database_migrations import upgrade_database
    from local_lm.db import create_database_engine

    settings = _settings(data_dir)
    settings.prepare()
    moment = time.perf_counter()
    upgrade_database(settings)
    report(f"migrate seconds={time.perf_counter() - moment:.2f}")

    moment = time.perf_counter()
    store = ArtifactStore(settings)
    folders: set[Path] = set()
    for item in plan.artifacts:
        path = store._destination(item.digest)
        if path.parent not in folders:
            path.parent.mkdir(parents=True, exist_ok=True)
            folders.add(path.parent)
        with open(path, "xb") as stored:
            stored.write(item.content)
    report(
        f"artifact-files files={len(plan.artifacts)} folders={len(folders)} "
        f"seconds={time.perf_counter() - moment:.2f}"
    )

    def relative(digest: str) -> str:
        return store._relative(store._destination(digest))

    engine = create_database_engine(settings)
    try:
        raw = engine.raw_connection()
        try:
            driver = raw.driver_connection
            if not isinstance(driver, sqlite3.Connection):
                raise RuntimeError("the application engine did not open SQLite")
            enforced = int(driver.execute("PRAGMA foreign_keys").fetchone()[0])
            installed_triggers = driver.execute(
                "SELECT count(*) FROM sqlite_master WHERE type = 'trigger'"
            )
            triggers = int(installed_triggers.fetchone()[0])
            if enforced != 1 or triggers == 0:
                raise RuntimeError("the application engine is not enforcing its schema")
            report(f"schema foreign_keys={enforced} triggers={triggers}")
            for label, statement, rows in _statements(plan, relative, seed=seed, padding=padding):
                moment = time.perf_counter()
                driver.execute("BEGIN IMMEDIATE")
                try:
                    cursor = driver.executemany(statement, rows)
                    changed = cursor.rowcount
                    driver.commit()
                except BaseException:
                    driver.rollback()
                    raise
                report(f"{label} rows={changed} seconds={time.perf_counter() - moment:.2f}")
            driver.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            counts = {table: _count(driver, table) for table in FILLED_TABLES}
            counted = sum(_count(driver, table) for table in COUNTED_TABLES)
        finally:
            raw.close()
    finally:
        engine.dispose()

    database = data_dir / DATABASE
    report(" ".join(f"{table}={count}" for table, count in counts.items()))
    # Whatever the shape adds up to; at full scale this is above the walk's row bound.
    report(
        f"counted-rows={counted} database-bytes={database.stat().st_size} "
        f"total-seconds={time.perf_counter() - started:.2f}"
    )
    marker = {
        "format": MARKER_FORMAT,
        "scale": scale,
        "seed": seed,
        "padding": asdict(padding),
        "rows": counts,
        "counted_rows": counted,
    }
    with open(data_dir / MARKER, "x", encoding="utf-8") as record:
        record.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    return counts


def _built_store(data_dir: Path) -> dict[str, object]:
    """The record a finished build left in `data_dir`, or a refusal.

    The store's state and artifact folders and its database must be its own:
    plain entries inside the folder, never links to entries anywhere else, so
    timing a store cannot open another installation's database.
    """

    if _is_link(data_dir) or not data_dir.is_dir():
        raise StoreRefused("calibrate needs the folder a build made")
    try:
        record = json.loads((data_dir / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StoreRefused("this folder holds no finished benchmark store") from error
    if not isinstance(record, dict) or record.get("format") != MARKER_FORMAT:
        raise StoreRefused("this folder holds no finished benchmark store")
    database = data_dir / DATABASE
    entries = [data_dir / name for name in STORE_FOLDERS]
    entries.append(database)
    entries.extend(
        companion
        for suffix in DATABASE_COMPANIONS
        if os.path.lexists(companion := database.with_name(database.name + suffix))
    )
    for entry in entries:
        if not os.path.lexists(entry):
            raise StoreRefused("this store is missing its state, artifacts or database")
        if _is_link(entry) or not _inside(entry, data_dir):
            raise StoreRefused("this store's folders or database lead outside it")
    # A second name for the database or a companion, made with a hard link, is
    # a way in from elsewhere.
    if any(entry.stat().st_nlink != 1 for entry in entries if entry.is_file()):
        raise StoreRefused("this store's database has another name elsewhere")
    return record


@contextmanager
def _caps_lifted() -> Iterator[None]:
    """Lift the reference walk's store-size refusals inside this process only.

    A store of the full shape counts more rows than the walk accepts, so
    without this every timed walk would stop at its first count.
    """

    from local_lm import artifact_library

    with (
        mock.patch.object(artifact_library, "MAX_REFERENCE_ROWS", sys.maxsize),
        mock.patch.object(artifact_library, "MAX_REFERENCE_VALUES", sys.maxsize),
    ):
        yield


def _stored_checks() -> dict[str, str]:
    """The stored-document check, one statement per table it reads.

    Built from the same generators as the whole check, and refused if the
    pieces no longer join back into exactly the statement deletion runs.
    """

    from local_lm import artifact_library_schema as schema

    checks: dict[str, str] = {}
    for table, columns in schema._TABLE_COLUMNS.items():
        structural = [
            schema._common_invalid(f"{table}.{column}", root_type, depth=schema._member_depth)
            for column, root_type in columns
        ]
        semantic = list(schema._table_invalid(table, table))
        invalid = (
            f"CASE WHEN {' OR '.join(structural)} THEN 1 "
            f"WHEN {' OR '.join(semantic)} THEN 1 ELSE 0 END"
        )
        scope = "scope = 'studio' AND " if table == "chats" else ""
        checks[table] = f"EXISTS (SELECT 1 FROM {table} WHERE {scope}({invalid}))"
    joined = " OR\n    ".join(checks.values())
    if f"SELECT CASE WHEN {joined} THEN 1 ELSE 0 END" != schema.STORED_JSON_INVALID_SQL:
        raise RuntimeError("the stored-document check no longer divides by table")
    return {table: f"SELECT CASE WHEN {check} THEN 1 ELSE 0 END" for table, check in checks.items()}


def _timed[T](trials: int, run: Callable[[], T]) -> tuple[list[float], T]:
    seconds: list[float] = []
    results: list[T] = []
    for _ in range(trials):
        started = time.perf_counter()
        results.append(run())
        seconds.append(time.perf_counter() - started)
    return seconds, results[-1]


def _line(label: str, seconds: list[float], *, rows: int | None = None, extra: str = "") -> str:
    median = statistics.median(seconds)
    parts = [label]
    if rows is not None:
        parts.append(f"rows={rows}")
    if extra:
        parts.append(extra)
    parts.append("trials-s=" + ",".join(f"{value:.3f}" for value in seconds))
    parts.append(f"median-s={median:.3f}")
    if rows:
        parts.append(f"per-row-us={median / rows * 1e6:.1f}")
    return " ".join(parts)


def calibrate(
    data_dir: Path,
    *,
    trials: int = 3,
    report: Callable[[str], None] = print,
    installed: Path | None = None,
) -> dict[str, list[float]]:
    """Time the reference checks on a store a build finished in `data_dir`.

    W is the reference walk; S is the scan of every artifact's metadata for
    the artifacts it names, which deletion adds to the walk; V is the check of
    every stored reference document deletion runs, whole and table by table;
    L loads every artifact the way the retention sweep does; and F is the whole
    walk deletion takes under the writer, split at the stored-document check.
    Nothing is written: the reads hold no writer, and F gives its writer back
    without a change. `installed` stands in for an installed application's
    data folder.
    """

    _refuse_installed(data_dir, installed=installed)
    record = _built_store(data_dir)
    if trials < 1:
        raise StoreRefused("calibrate needs at least one trial")

    from local_lm import artifact_library
    from local_lm.artifact_library_schema import (
        ARTIFACT_METADATA_REFERENCE_KEYS,
        STORED_JSON_INVALID_SQL,
    )
    from local_lm.db import create_database_engine
    from local_lm.models import Artifact
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    engine: Engine = create_database_engine(_settings(data_dir))
    timings: dict[str, list[float]] = {}
    try:
        with engine.connect() as connection:
            driver = connection.connection.driver_connection
            if not isinstance(driver, sqlite3.Connection):
                raise RuntimeError("the application engine did not open SQLite")
            counted = sum(_count(driver, table) for table in COUNTED_TABLES)
            # The rows each table's share of the stored-document check reads.
            rows = {
                table: _count(driver, table)
                for table in ("artifacts", "runs", "jobs", "work_steps", "message_references")
            }
            studio = driver.execute("SELECT count(*) FROM chats WHERE scope = 'studio'")
            rows["chats"] = int(studio.fetchone()[0])
        report(
            f"calibrate scale={record.get('scale')} artifacts={rows['artifacts']} "
            f"counted-rows={counted} trials={trials}"
        )

        def walk() -> AbstractSet[str]:
            with Session(engine) as session:
                return artifact_library.walk_references(session)[0]

        def referrers() -> int:
            named: dict[str, set[str]] = {}
            with Session(engine) as session:
                statement = select(Artifact.id, Artifact.metadata_json)
                for other_id, value in session.execute(statement):
                    metadata = artifact_library._mapping(value)
                    for key in ARTIFACT_METADATA_REFERENCE_KEYS:
                        if key in metadata:
                            for name in artifact_library._optional_id(metadata[key]):
                                named.setdefault(name, set()).add(other_id)
            return len(named)

        def stored(statement: str) -> Callable[[], object]:
            def run() -> object:
                with engine.connect() as connection:
                    return connection.exec_driver_sql(statement).scalar_one()

            return run

        def load() -> int:
            with Session(engine) as session:
                ordered = session.scalars(select(Artifact).order_by(Artifact.created_at)).all()
                return len(ordered)

        with _caps_lifted():
            seconds, retained = _timed(trials, walk)
            timings["W"] = seconds
            report(_line("W walk", seconds, extra=f"retained={len(retained)}"))

            seconds, named = _timed(trials, referrers)
            timings["S"] = seconds
            report(_line("S referrers", seconds, rows=rows["artifacts"], extra=f"named={named}"))

            seconds, invalid = _timed(trials, stored(STORED_JSON_INVALID_SQL))
            timings["V"] = seconds
            report(_line("V stored", seconds, extra=f"invalid={invalid}"))
            for table, statement in _stored_checks().items():
                seconds, invalid = _timed(trials, stored(statement))
                timings[f"V {table}"] = seconds
                report(_line(f"V {table}", seconds, rows=rows[table], extra=f"invalid={invalid}"))

            seconds, loaded = _timed(trials, load)
            timings["L"] = seconds
            report(_line("L load", seconds, rows=loaded))

            before_check: list[float] = []
            in_check: list[float] = []
            whole: list[float] = []
            for _ in range(trials):
                total, before, inside = _deletion_walk(engine)
                whole.append(total)
                before_check.append(before)
                in_check.append(inside)
            timings["F"] = whole
            report(
                _line(
                    "F deletion-walk",
                    whole,
                    extra=(
                        f"before-check-median-s={statistics.median(before_check):.3f} "
                        f"check-median-s={statistics.median(in_check):.3f}"
                    ),
                )
            )
    finally:
        engine.dispose()
    return timings


def _deletion_walk(engine: Engine) -> tuple[float, float, float]:
    """One walk with deletion authority, timed whole and at the stored-document check.

    The walk itself is timed, not the reference index that now answers in its
    place, so the figures stay comparable with those taken before the index.
    """

    from local_lm import artifact_library
    from local_lm.artifact_library_schema import STORED_JSON_INVALID_SQL
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    marks: list[float] = []

    def mark(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _many: bool,
    ) -> None:
        if statement == STORED_JSON_INVALID_SQL:
            marks.append(time.perf_counter())

    event.listen(engine, "before_cursor_execute", mark)
    event.listen(engine, "after_cursor_execute", mark)
    try:
        with Session(engine) as session:
            started = time.perf_counter()
            artifact_library.begin_artifact_write_fence(session)
            artifact_library.walk_references(session, for_deletion=True)
            finished = time.perf_counter()
            session.rollback()
    finally:
        event.remove(engine, "before_cursor_execute", mark)
        event.remove(engine, "after_cursor_execute", mark)
    if len(marks) != 2:
        raise RuntimeError("the deletion walk did not reach the stored-document check")
    return finished - started, marks[0] - started, marks[1] - marks[0]


def _cookie(connection: sqlite3.Connection, schema: str = "main") -> int:
    return int(connection.execute(f"PRAGMA {schema}.schema_version").fetchone()[0])


def _fact_version() -> str:
    """Which SQLite this is, on which Python, and whether it can report page use."""

    with closing(sqlite3.connect(":memory:")) as connection:
        options = {row[0] for row in connection.execute("PRAGMA compile_options")}
    dbstat = int("ENABLE_DBSTAT_VTAB" in options)
    return (
        f"E1 sqlite-version={sqlite3.sqlite_version} python-version={platform.python_version()} "
        f"platform={sys.platform} dbstat={dbstat}"
    )


def _fact_cookies(folder: Path) -> str:
    """Whether a backup copy, VACUUM or temporary DDL moves the schema cookie."""

    source_path = folder / "cookie-source.sqlite3"
    with closing(sqlite3.connect(source_path)) as source:
        source.execute("PRAGMA journal_mode=WAL")
        source.executescript(
            "CREATE TABLE first_rows (id INTEGER PRIMARY KEY);"
            "CREATE TABLE second_rows (id INTEGER PRIMARY KEY, value INTEGER);"
            "CREATE INDEX second_rows_value ON second_rows (value);"
        )
        source_cookie = _cookie(source)
        handle, name = tempfile.mkstemp(prefix="backup-", suffix=".partial", dir=folder)
        os.close(handle)
        with closing(sqlite3.connect(name)) as copy:
            source.backup(copy)
        with closing(sqlite3.connect(name)) as reopened:
            backup_cookie = _cookie(reopened)
        before_vacuum = _cookie(source)
        source.execute("VACUUM")
        after_vacuum = _cookie(source)
        source.execute("CREATE TEMP TABLE scratch_rows (id INTEGER)")
        after_temporary = _cookie(source)
    return (
        f"E2 source-cookie={source_cookie} backup-copy-cookie={backup_cookie} "
        f"vacuum-cookie-before={before_vacuum} vacuum-cookie-after={after_vacuum} "
        f"temporary-ddl-cookie-after={after_temporary}"
    )


def _fact_change_counter() -> str:
    """Whether trigger and cascade rows count, and what a delete guard sees.

    Each parent is named for how many children it has, and one statement
    deletes all three, so each reading shows which earlier rows the counter
    already held when that row's guard ran.
    """

    with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
        observed: list[tuple[str, int]] = []

        def observe(key: object) -> int:
            observed.append((str(key), connection.total_changes))
            return 1

        connection.create_function("observe_changes", 1, observe)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            "CREATE TABLE parents (id TEXT PRIMARY KEY);"
            "CREATE TABLE children (id TEXT PRIMARY KEY,"
            " parent_id TEXT NOT NULL REFERENCES parents (id) ON DELETE CASCADE);"
            "CREATE TABLE notes (id INTEGER PRIMARY KEY, parent_id TEXT);"
            "CREATE TRIGGER parents_noted AFTER INSERT ON parents"
            " BEGIN INSERT INTO notes (parent_id) VALUES (NEW.id); END;"
            "CREATE TRIGGER parents_observed BEFORE DELETE ON parents"
            " BEGIN SELECT observe_changes(OLD.id); END;"
        )
        before_insert = connection.total_changes
        connection.execute("INSERT INTO parents VALUES ('two')")
        insert_delta = connection.total_changes - before_insert
        connection.execute("INSERT INTO parents VALUES ('none')")
        connection.execute("INSERT INTO parents VALUES ('one')")
        connection.execute(
            "INSERT INTO children VALUES ('c1', 'two'), ('c2', 'two'), ('c3', 'one')"
        )
        before_delete = connection.total_changes
        connection.execute("DELETE FROM parents")
        delete_delta = connection.total_changes - before_delete
    seen = ",".join(f"{key}:{value - before_delete}" for key, value in observed)
    return (
        f"E3 insert-with-trigger-delta={insert_delta} "
        f"delete-three-parents-three-children-delta={delete_delta} guard-saw={seen}"
    )


def _fact_foreign_key_actions() -> str:
    """Whether a cascade fires child delete triggers, and what SET NULL fires."""

    with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(
            "CREATE TABLE plans (id TEXT PRIMARY KEY);"
            "CREATE TABLE steps (id TEXT PRIMARY KEY,"
            " plan_id TEXT NOT NULL REFERENCES plans (id) ON DELETE CASCADE);"
            "CREATE TABLE runs (id TEXT PRIMARY KEY);"
            "CREATE TABLE jobs (id TEXT PRIMARY KEY,"
            " run_id TEXT REFERENCES runs (id) ON DELETE SET NULL,"
            " payload_json TEXT NOT NULL DEFAULT '{}');"
            "CREATE TABLE fired (name TEXT NOT NULL);"
            "CREATE TRIGGER steps_deleted AFTER DELETE ON steps"
            " BEGIN INSERT INTO fired VALUES ('step-delete'); END;"
            "CREATE TRIGGER jobs_payload_updated AFTER UPDATE OF payload_json ON jobs"
            " BEGIN INSERT INTO fired VALUES ('job-payload-update'); END;"
            "CREATE TRIGGER jobs_updated AFTER UPDATE ON jobs"
            " BEGIN INSERT INTO fired VALUES ('job-update'); END;"
            "INSERT INTO plans VALUES ('p');"
            "INSERT INTO steps VALUES ('s1', 'p'), ('s2', 'p');"
            "INSERT INTO runs VALUES ('r');"
            "INSERT INTO jobs (id, run_id) VALUES ('j', 'r');"
            "DELETE FROM plans;"
            "DELETE FROM runs;"
        )
        fired = dict(connection.execute("SELECT name, count(*) FROM fired GROUP BY name"))
        cleared = connection.execute("SELECT count(*) FROM jobs WHERE run_id IS NULL").fetchone()
    return (
        f"E4 cascade-child-delete-triggers={fired.get('step-delete', 0)} "
        f"set-null-update-of-payload-triggers={fired.get('job-payload-update', 0)} "
        f"set-null-update-triggers={fired.get('job-update', 0)} set-null-rows={cleared[0]}"
    )


def _fact_conflict_clauses() -> str:
    """How an outer conflict clause reaches an INSERT OR IGNORE inside a trigger.

    Every change row starts pending with its audit flag set. Each statement
    below writes a source row whose trigger inserts the same key with the
    flag clear, so afterwards whether the change row is still there, and its
    flag, show which conflict clause the trigger's insert ran under. The two
    are reported apart: a row that was replaced is still there.
    """

    with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
        connection.executescript(
            "CREATE TABLE source_rows (id TEXT PRIMARY KEY, value INTEGER);"
            "CREATE TABLE change_rows (source INTEGER NOT NULL, source_key TEXT NOT NULL,"
            " audit INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (source, source_key))"
            " WITHOUT ROWID;"
            "CREATE INDEX change_rows_audit ON change_rows (audit);"
            "CREATE TRIGGER source_rows_changed AFTER INSERT ON source_rows BEGIN"
            " INSERT OR IGNORE INTO change_rows (source, source_key, audit)"
            " VALUES (1, NEW.id, 0); END;"
            "INSERT INTO change_rows VALUES (1, 'a', 1), (1, 'b', 1), (1, 'c', 1), (1, 'd', 1);"
        )

        def change_row(name: str, key: str) -> str:
            row = connection.execute(
                "SELECT audit FROM change_rows WHERE source = 1 AND source_key = ?", (key,)
            ).fetchone()
            audit = "none" if row is None else str(row[0])
            return f"{name}-row-present={int(row is not None)} {name}-audit={audit}"

        connection.execute("INSERT INTO source_rows VALUES ('a', 1)")
        plain = change_row("plain-insert", "a")
        connection.execute("INSERT OR REPLACE INTO source_rows VALUES ('b', 1)")
        replaced = change_row("outer-replace", "b")
        try:
            connection.execute("INSERT OR ABORT INTO source_rows VALUES ('c', 1)")
            abort_failed = 0
        except sqlite3.IntegrityError:
            abort_failed = 1
        aborted = change_row("outer-abort", "c")
        connection.execute(
            "REPLACE INTO change_rows (source, source_key, audit) VALUES (1, 'd', 5)"
        )
        rows_after = int(connection.execute("SELECT count(*) FROM change_rows").fetchone()[0])
    return (
        f"E5 {plain} {replaced} outer-abort-failed={abort_failed} {aborted} "
        f"replace-without-rowid-rows-after={rows_after}-of-4"
    )


def _fact_row_count() -> str:
    """Whether a cursor's row count includes the rows its triggers wrote."""

    with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
        connection.executescript(
            "CREATE TABLE items (id INTEGER PRIMARY KEY);"
            "CREATE TABLE item_notes (item_id INTEGER);"
            "CREATE TRIGGER items_noted AFTER DELETE ON items BEGIN"
            " INSERT INTO item_notes VALUES (OLD.id); INSERT INTO item_notes VALUES (OLD.id);"
            " END;"
            "INSERT INTO items VALUES (1), (2), (3);"
        )
        before = connection.total_changes
        cursor = connection.execute("DELETE FROM items WHERE id <= 2")
        delta = connection.total_changes - before
        return f"E6 cursor-rowcount={cursor.rowcount} total-changes-delta={delta}"


def _fact_json() -> str:
    """Whether JSON text a strict reader refuses still passes json_valid."""

    with closing(sqlite3.connect(":memory:")) as connection:
        valid = connection.execute("SELECT json_valid('NaN'), json_valid('{a:1}')").fetchone()

        def type_of(text: str) -> str:
            try:
                return str(connection.execute("SELECT json_type(?)", (text,)).fetchone()[0])
            except sqlite3.Error:
                return "refused"

        return (
            f"E7 json-valid-nan={valid[0]} json-valid-unquoted-key={valid[1]} "
            f"json-type-nan={type_of('NaN')} json-type-unquoted-key={type_of('{a:1}')}"
        )


def _fact_variable_limit() -> str:
    with closing(sqlite3.connect(":memory:")) as connection:
        limit = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    return f"E8 variable-number-limit={limit}"


def _fact_trusted_schema() -> str:
    with closing(sqlite3.connect(":memory:")) as connection:
        trusted = int(connection.execute("PRAGMA trusted_schema").fetchone()[0])
    return f"E9 trusted-schema={trusted}"


def engine_facts() -> list[str]:
    """Each SQLite behaviour the reference checks rely on, measured on throwaway databases."""

    with tempfile.TemporaryDirectory(
        prefix="lma-engine-facts-", ignore_cleanup_errors=True
    ) as folder:
        return [
            _fact_version(),
            _fact_cookies(Path(folder)),
            _fact_change_counter(),
            _fact_foreign_key_actions(),
            _fact_conflict_clauses(),
            _fact_row_count(),
            _fact_json(),
            _fact_variable_limit(),
            _fact_trusted_schema(),
        ]


@contextmanager
def _application_environment() -> Iterator[None]:
    """Import the application with every default folder inside a throwaway one.

    Importing the database module reads the environment and creates the
    default data folder's layout. Pointing those defaults somewhere disposable
    first means nothing an installation or a developer configured is touched.
    """

    scratch = Path(tempfile.mkdtemp(prefix="lma-reference-benchmark-import-"))
    os.environ.update(
        LOCAL_LM_DATA_DIR=str(scratch / "data"),
        LOCAL_LM_COMFY_DIRECTORY=str(scratch / "comfy"),
        LOCAL_LM_HOST="127.0.0.1",
        LOCAL_LM_DEV="true",
        LOCAL_LM_CHAT_ENGINE="mock",
        LOCAL_LM_MEDIA_ENGINE="mock",
    )
    sys.path.insert(0, str(REPOSITORY / "services" / "api"))
    try:
        yield
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build a store in a new or empty folder")
    build.add_argument("--scale", type=float, default=1.0, help="1.0 is the full shape")
    build.add_argument("--seed", type=int, default=1)
    build.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="a new or empty folder; a new temporary folder when omitted",
    )
    build.add_argument("--provenance-bytes", type=int, default=DEFAULT_PROVENANCE_BYTES)
    build.add_argument("--provenance-items", type=int, default=DEFAULT_PROVENANCE_ITEMS)
    build.add_argument("--settings-bytes", type=int, default=DEFAULT_SETTINGS_BYTES)
    build.add_argument("--settings-items", type=int, default=DEFAULT_SETTINGS_ITEMS)
    build.add_argument("--metadata-bytes", type=int, default=DEFAULT_METADATA_BYTES)
    build.add_argument("--metadata-items", type=int, default=DEFAULT_METADATA_ITEMS)
    commands.add_parser("engine-facts", help="measure the SQLite behaviours the checks rely on")
    timing = commands.add_parser("calibrate", help="time the reference checks on a built store")
    timing.add_argument("--data-dir", type=Path, required=True)
    timing.add_argument("--trials", type=int, default=3)
    arguments = parser.parse_args(argv)
    # Every line goes out as soon as it is written, so a build that stops
    # part way has already said where it was writing.
    report = functools.partial(print, flush=True)

    try:
        if arguments.command == "engine-facts":
            for line in engine_facts():
                report(line)
            return 0
        if arguments.command == "build":
            # A scale the shape refuses is refused before any folder is made.
            StoreShape.at(arguments.scale).check()
            if arguments.data_dir is None:
                target = Path(tempfile.mkdtemp(prefix="lma-reference-benchmark-")) / "data"
            else:
                target = arguments.data_dir.absolute()
            # Decided before the application is imported, because importing it
            # is the first thing that creates folders. The installed folder is
            # refused first, so it is never even listed.
            _refuse_installed(target)
            _refuse_occupied(target)
            with _application_environment():
                build_store(
                    target,
                    scale=arguments.scale,
                    seed=arguments.seed,
                    padding=Padding(
                        provenance_bytes=arguments.provenance_bytes,
                        provenance_items=arguments.provenance_items,
                        settings_bytes=arguments.settings_bytes,
                        settings_items=arguments.settings_items,
                        metadata_bytes=arguments.metadata_bytes,
                        metadata_items=arguments.metadata_items,
                    ),
                    report=report,
                )
            return 0
        target = arguments.data_dir.absolute()
        _refuse_installed(target)
        _built_store(target)
        with _application_environment():
            calibrate(target, trials=arguments.trials, report=report)
        return 0
    except StoreRefused as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
