"""Which artifacts one stored document names, read the same way everywhere.

The reference walk, its pending-write counterpart, the recovery graph and the
reference index all ask this question of the same six kinds of row. They share
these readers so they cannot answer it differently: a reader that drifted from
the walk would let the index call an artifact free that the walk still holds.

Every reader fails closed. A document that does not have the shape its column
promises raises `ArtifactReferenceDataError` rather than yielding the ids it
could still find, because a reference skipped is an artifact deleted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final, NoReturn

from .artifact_deletion_authority import ArtifactReferenceDataError
from .artifact_library_schema import ARTIFACT_METADATA_REFERENCE_KEYS

MAX_REFERENCE_LIST = 4_096
MAX_REFERENCE_VALUES = 100_000
MAX_REFERENCE_DEPTH = 16
REFERENCE_CORRUPT = "Stored artifact reference data is invalid."

#: The tables whose stored documents name artifacts, by the number the index
#: records for each. The numbers are stored, so they never change meaning.
SOURCE_JOBS: Final = 1
SOURCE_RUNS: Final = 2
SOURCE_WORK_STEPS: Final = 3
SOURCE_MESSAGE_REFERENCES: Final = 4
SOURCE_CHATS: Final = 5
SOURCE_ARTIFACTS: Final = 6

#: Each source's table and the columns its documents are read from, in the
#: order `row_edges` receives them.
SOURCE_COLUMNS: Final[Mapping[int, tuple[str, tuple[str, ...]]]] = {
    SOURCE_JOBS: ("jobs", ("payload_json", "result_json")),
    SOURCE_RUNS: ("runs", ("provenance_json", "settings_json")),
    SOURCE_WORK_STEPS: ("work_steps", ("input_bindings_json", "settings_json")),
    SOURCE_MESSAGE_REFERENCES: ("message_references", ("artifact_ids_json",)),
    SOURCE_CHATS: ("chats", ("scope", "origin_json")),
    SOURCE_ARTIFACTS: ("artifacts", ("metadata_json",)),
}

#: What an index records for a row whose documents cannot be read. No reader
#: accepts an empty id, so it can never be mistaken for an artifact.
UNDERIVABLE: Final = ""


def _fail() -> NoReturn:
    raise ArtifactReferenceDataError(REFERENCE_CORRUPT)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail()
    if len(value) > MAX_REFERENCE_LIST:
        _fail()
    return value


def _ids(value: object) -> set[str]:
    if not isinstance(value, list) or len(value) > MAX_REFERENCE_LIST:
        _fail()
    if any(not isinstance(item, str) or not item or len(item) > 80 for item in value):
        _fail()
    return set(value)


def _optional_id(value: object) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, str) or not value or len(value) > 80:
        _fail()
    return {value}


def _run_ids(value: object) -> set[str]:
    row = _mapping(value)
    found: set[str] = set()
    for key in ("input_artifact_ids", "resolved_dependency_artifact_ids"):
        if key in row:
            found.update(_ids(row[key]))
    outputs = row.get("outputs")
    if outputs is not None:
        if not isinstance(outputs, list) or len(outputs) > MAX_REFERENCE_LIST:
            _fail()
        for output in outputs:
            item = _mapping(output)
            for key in ("artifact_id", "poster_artifact_id", "browser_proxy_artifact_id"):
                if key in item:
                    found.update(_optional_id(item[key]))
    return found


def _work_step_ids(value: object) -> set[str]:
    if not isinstance(value, list) or len(value) > MAX_REFERENCE_LIST:
        _fail()
    found: set[str] = set()
    for raw in value:
        item = _mapping(raw)
        if "artifact_id" in item:
            found.update(_optional_id(item["artifact_id"]))
    return found


def _settings_ids(value: object) -> set[str]:
    row = _mapping(value)
    mask = row.get("mask")
    if mask is None:
        return set()
    mask_row = _mapping(mask)
    if "artifact_id" not in mask_row:
        _fail()
    return _optional_id(mask_row["artifact_id"])


_SCALAR_ARTIFACT_KEYS = {
    "artifact_id",
    "source_artifact_id",
    "result_artifact_id",
    "input_artifact_id",
    "poster_artifact_id",
    "browser_proxy_artifact_id",
}
_LIST_ARTIFACT_KEYS = {"artifact_ids", "input_artifact_ids"}


def _job_ids(value: object, *, depth: int = 0, budget: list[int] | None = None) -> set[str]:
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > MAX_REFERENCE_VALUES or depth > MAX_REFERENCE_DEPTH:
        _fail()
    found: set[str] = set()
    if isinstance(value, Mapping):
        if len(value) > MAX_REFERENCE_LIST:
            _fail()
        for key, child in value.items():
            if not isinstance(key, str):
                _fail()
            if key in _SCALAR_ARTIFACT_KEYS:
                found.update(_optional_id(child))
            elif key in _LIST_ARTIFACT_KEYS:
                found.update(_ids(child))
            else:
                found.update(_job_ids(child, depth=depth + 1, budget=budget))
    elif isinstance(value, list):
        if len(value) > MAX_REFERENCE_LIST:
            _fail()
        for child in value:
            found.update(_job_ids(child, depth=depth + 1, budget=budget))
    elif value is not None and not isinstance(value, str | int | float | bool):
        _fail()
    return found


def _metadata_ids(value: object) -> set[str]:
    metadata = _mapping(value)
    found: set[str] = set()
    for key in ARTIFACT_METADATA_REFERENCE_KEYS:
        if key in metadata:
            found.update(_optional_id(metadata[key]))
    return found


def _unique_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    members: dict[str, object] = {}
    for key, value in pairs:
        if key in members:
            # SQLite's readers see every copy of a repeated name and Python's
            # keeps only the last, so the two would disagree about what this
            # document names. Neither answer can be trusted.
            _fail()
        members[key] = value
    return members


def _refuse_constant(_name: str) -> NoReturn:
    # NaN and the infinities are not JSON, and SQLite's check refuses them.
    _fail()


def parse_stored_json(text: object) -> object:
    """Read one stored document as canonical JSON, refusing what only Python accepts.

    The application stores what `json.dumps` writes, which never repeats a
    member name and never writes NaN or infinity. A stored document that does
    either was written some other way, and the readers above cannot be trusted
    with it, so it is refused rather than read leniently. SQL NULL reads as
    None, as the column type returns it.
    """

    if text is None:
        return None
    if not isinstance(text, str):
        _fail()
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_members,
            parse_constant=_refuse_constant,
        )
    except ArtifactReferenceDataError:
        raise
    except (RecursionError, ValueError) as exc:
        raise ArtifactReferenceDataError(REFERENCE_CORRUPT) from exc


def _encodable(found: set[str]) -> set[str]:
    for artifact_id in found:
        try:
            artifact_id.encode("utf-8")
        except UnicodeEncodeError:
            # A lone surrogate survives json.loads but cannot be stored as
            # text, so the index could not record it as a name.
            _fail()
    return found


def document_edges(source: int, values: tuple[object, ...]) -> set[str]:
    """The artifact ids one row's already parsed documents name, as the walk reads them.

    `values` are the row's columns in `SOURCE_COLUMNS` order. Raises
    `ArtifactReferenceDataError` exactly where the walk would.
    """

    if source == SOURCE_JOBS:
        payload, result = values
        return _encodable(_job_ids(payload) | _job_ids(result))
    if source == SOURCE_RUNS:
        provenance, settings = values
        return _encodable(_run_ids(provenance) | _settings_ids(settings))
    if source == SOURCE_WORK_STEPS:
        bindings, settings = values
        return _encodable(_work_step_ids(bindings) | _settings_ids(settings))
    if source == SOURCE_MESSAGE_REFERENCES:
        (artifact_ids,) = values
        return _encodable(_ids(artifact_ids))
    if source == SOURCE_CHATS:
        scope, origin = values
        if scope != "studio":
            return set()
        return _encodable(_optional_id(_mapping(origin).get("source_artifact_id")))
    if source == SOURCE_ARTIFACTS:
        (metadata,) = values
        return _encodable(_metadata_ids(metadata))
    raise ValueError(f"unknown artifact reference source {source}")


def row_edges(source: int, columns: tuple[object, ...]) -> set[str]:
    """The artifact ids one stored row names, or the underivable marker alone.

    `columns` are the row's stored values in `SOURCE_COLUMNS` order, as text.
    A row any reader refuses names `UNDERIVABLE` instead, so whatever reads
    the index refuses with it rather than seeing a row that names nothing.
    """

    try:
        if source == SOURCE_CHATS:
            scope, origin = columns
            if scope != "studio":
                # Only a studio chat's origin names an artifact, so another
                # chat's origin is not read at all, however it is stored.
                return set()
            values: tuple[object, ...] = (scope, parse_stored_json(origin))
        else:
            values = tuple(parse_stored_json(column) for column in columns)
        return document_edges(source, values)
    except ArtifactReferenceDataError:
        return {UNDERIVABLE}
