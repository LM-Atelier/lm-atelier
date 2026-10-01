"""Select bounded library identities before loading their complete records."""

from __future__ import annotations

from collections.abc import Callable
from itertools import batched, islice

from sqlalchemy import Select, select
from sqlalchemy.orm import InstrumentedAttribute, Session

from .models import ModelInstall
from .schemas import CatalogInstallMatches


def read_library_page[LibraryRow](
    session: Session,
    statement: Select[tuple[LibraryRow]],
    identity: InstrumentedAttribute[str],
    name: InstrumentedAttribute[str],
    *,
    limit: int | None,
    offset: int,
    search: str,
) -> list[LibraryRow]:
    needle = search.strip().casefold()
    if not needle:
        statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
        return list(session.scalars(statement))
    names = statement.with_only_columns(identity, name).execution_options(yield_per=200)
    with session.execute(names) as rows:
        matches = (row_id for row_id, row_name in rows if needle in row_name.casefold())
        identities = list(islice(islice(matches, offset, None), limit))
    result: list[LibraryRow] = []
    for start in range(0, len(identities), 200):
        result.extend(
            session.scalars(statement.where(identity.in_(identities[start : start + 200])))
        )
    return result


def read_library_matches[LibraryRow, Match](
    session: Session,
    statement: Select[tuple[LibraryRow]],
    identity: InstrumentedAttribute[str],
    name: InstrumentedAttribute[str],
    *,
    match: Callable[[LibraryRow], Match | None],
    limit: int | None,
    offset: int,
    search: str,
) -> list[Match]:
    needle = search.strip().casefold()
    names = statement.with_only_columns(identity, name).execution_options(yield_per=200)
    result: list[Match] = []
    with session.execute(names) as rows:
        identities = (row_id for row_id, row_name in rows if needle in row_name.casefold())
        for batch in batched(identities, 200):
            with session.scalars(statement.where(identity.in_(batch))) as candidates:
                for candidate in candidates:
                    matched = match(candidate)
                    if matched is None:
                        continue
                    if offset:
                        offset -= 1
                        continue
                    result.append(matched)
                    if limit is not None and len(result) == limit:
                        return result
    return result


def catalog_install_matches(
    session: Session,
    role: str,
    remote_ids: list[str],
    workflow_template_ids: list[str],
) -> CatalogInstallMatches:
    remote_matches: set[str] = set()
    for field in ("remote_id", "source_remote_id"):
        if remote_ids:
            value = ModelInstall.manifest_json[field].as_string()
            remote_matches.update(
                session.scalars(
                    select(value)
                    .where(
                        ModelInstall.active.is_(True),
                        ModelInstall.role == role,
                        value.in_(remote_ids),
                    )
                    .distinct()
                )
            )
    template_matches: list[str] = []
    if workflow_template_ids:
        value = ModelInstall.manifest_json["workflow_template_id"].as_string()
        template_matches = sorted(
            session.scalars(
                select(value)
                .where(
                    ModelInstall.active.is_(True),
                    ModelInstall.role == role,
                    value.in_(workflow_template_ids),
                )
                .distinct()
            )
        )
    return CatalogInstallMatches(
        remote_ids=sorted(remote_matches), workflow_template_ids=template_matches
    )
