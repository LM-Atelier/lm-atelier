"""How long retention keeps media nothing uses, as chosen in Settings.

Two windows decide when retention may clear a stored file: how long something
has gone unused, and how old a preview or in-between step is. The installation's
configuration supplies both until somebody chooses otherwise. A choice is stored
in the workspace database and from then on governs every clearing pass.

Every change names the revision it was based on, and a change based on anything
older is refused rather than applied, so two open windows cannot silently undo
each other's choice. The refusal names the current revision.

Retention reads the windows through `windows_for`, after it has taken the
database's writer reservation, so a choice saved before a clearing pass began is
the one that pass uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .config import Settings
from .domain import utcnow
from .models import RetentionPolicy
from .schemas import RetentionPolicyOut, RetentionWindowsIn

#: The one thing a policy row governs today: media retention.
MEDIA_SCOPE = "media"


class RetentionWindows(NamedTuple):
    media_days: int
    temporary_hours: int


@dataclass(frozen=True)
class RetentionPolicyStale(Exception):
    """The retention choice changed after the writer read it."""

    current_revision: int


def _defaults(settings: Settings) -> RetentionWindows:
    return RetentionWindows(settings.artifact_retention_days, settings.temporary_retention_hours)


def _stored(session: Session) -> RetentionPolicy | None:
    # Always from the database, never from objects this session loaded before
    # a write, which may since have been replaced underneath them.
    return session.get(RetentionPolicy, MEDIA_SCOPE, populate_existing=True)


def read_policy(session: Session, settings: Settings) -> RetentionPolicyOut:
    """The windows in force, the revision they are at, and the installation's own."""

    defaults = _defaults(settings)
    row = _stored(session)
    windows = RetentionWindows(row.media_days, row.temporary_hours) if row else defaults
    return RetentionPolicyOut(
        media_days=windows.media_days,
        temporary_hours=windows.temporary_hours,
        revision=row.revision if row else 0,
        default_media_days=defaults.media_days,
        default_temporary_hours=defaults.temporary_hours,
    )


def windows_for(session: Session, settings: Settings) -> RetentionWindows:
    """The windows a clearing pass uses, read in the pass's own transaction."""

    row = _stored(session)
    if row is None:
        return _defaults(settings)
    return RetentionWindows(row.media_days, row.temporary_hours)


def write_policy(
    session: Session,
    settings: Settings,
    expected_revision: int,
    windows: RetentionWindowsIn,
) -> RetentionPolicyOut:
    """Choose both windows, if the choice is still at `expected_revision`.

    Commits nothing: the route commits, so a refusal leaves the transaction to be
    rolled back whole.
    """

    now = utcnow()
    values = {
        "media_days": windows.media_days,
        "temporary_hours": windows.temporary_hours,
        "updated_at": now,
    }
    if expected_revision == 0:
        # The first choice creates the row. One that already exists means
        # another writer chose first, which the statement itself refuses.
        result = session.execute(
            sqlite_insert(RetentionPolicy)
            .values(scope=MEDIA_SCOPE, revision=1, created_at=now, **values)
            .on_conflict_do_nothing(index_elements=["scope"])
        )
    else:
        # The revision check and the write are one statement, so two writers
        # that read the same revision cannot both succeed.
        result = session.execute(
            update(RetentionPolicy)
            .where(
                RetentionPolicy.scope == MEDIA_SCOPE,
                RetentionPolicy.revision == expected_revision,
            )
            .values(revision=expected_revision + 1, **values)
            .execution_options(synchronize_session=False)
        )
    if getattr(result, "rowcount", 0) != 1:
        current = session.scalar(
            select(RetentionPolicy.revision).where(RetentionPolicy.scope == MEDIA_SCOPE)
        )
        raise RetentionPolicyStale(current or 0)
    return read_policy(session, settings)
