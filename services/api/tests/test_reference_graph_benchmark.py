"""The benchmark store builder makes the shape it is asked for, and only where it may.

The builder is loaded through runpy, as the repository's other scripts are.
A store at a thousandth of full scale takes a few seconds, so the counts and
the application's own reference checks are read from a real build rather
than from a description of one. Every test that could reach an installed
application's data folder names a stand-in for it, so none looks at a real one.
"""

from __future__ import annotations

import itertools
import json
import os
import runpy
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from local_lm.artifact_library import (
    begin_artifact_write_fence,
    metadata_referrers,
    referenced_artifact_ids,
)
from local_lm.artifact_library_schema import PREMIGRATION_INVALID_SQL
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import create_database_engine
from local_lm.models import Artifact

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "reference-graph-benchmark.py"

#: Rows at scale 0.001: each full-scale population times a thousandth, rounded
#: to whole rows, keeping at least one row of every population.
EXPECTED_ROWS = {
    # 19 pictures (18 made by jobs and one other), 2 uploads, a video, and 80
    # thumbnails: the video's poster and 79 previews.
    "artifacts": 102,
    # Two chats and a studio session.
    "chats": 3,
    "chat_composer_drafts": 1,
    "chat_composer_draft_attachments": 1,
    # 17 turns and 18 runs, each with a message of its own.
    "messages": 35,
    "message_parts": 53,
    "reference_subjects": 1,
    "reference_assets": 1,
    "message_references": 1,
    "work_plans": 17,
    "work_steps": 18,
    "runs": 18,
    "response_revisions": 18,
    "response_revision_parts": 35,
    # A job for each run, and another for the run that made a second picture.
    "jobs": 19,
    "run_context_snapshots": 1,
    "run_context_artifacts": 1,
    "model_installs": 1,
    "model_profiles": 1,
    "setup_verifications": 1,
    "comfy_registry_source_artifact_reviews": 1,
    # Every picture and every video.
    "artifact_library_entries": 20,
}
KINDS = {"image": 19, "input": 2, "thumbnail": 80, "video": 1}
#: Previews by age in days: under one, one to seven, seven to thirty, thirty to ninety.
PREVIEW_AGES = {"0-1": 1, "1-7": 16, "7-30": 31, "30-90": 31}
#: What the reference walk counts against its row bound in that store.
COUNTED_ROWS = 173

#: Every place the reference walk or the stored-document check finds an
#: artifact named, as a query counting the names each holds, and how many
#: each holds at that scale.
SOURCES = {
    "reference assets": ("SELECT count(artifact_id) FROM reference_assets", 1),
    "run context artifacts": ("SELECT count(artifact_id) FROM run_context_artifacts", 1),
    "setup verifications": ("SELECT count(input_artifact_id) FROM setup_verifications", 1),
    "library entries": ("SELECT count(artifact_id) FROM artifact_library_entries", 20),
    "registry reviews": (
        "SELECT count(artifact_id) FROM comfy_registry_source_artifact_reviews",
        1,
    ),
    "draft attachments": ("SELECT count(artifact_id) FROM chat_composer_draft_attachments", 1),
    # Every reply's media and the one attached upload.
    "message parts": ("SELECT count(artifact_id) FROM message_parts", 19),
    "revision parts": (
        "SELECT count(part.artifact_id) FROM response_revision_parts AS part"
        " JOIN response_revisions AS revision ON revision.id = part.response_revision_id",
        18,
    ),
    "message references": (
        "SELECT count(*) FROM message_references, json_each(artifact_ids_json)",
        1,
    ),
    "run outputs": (
        "SELECT count(*) FROM runs, json_each(provenance_json, '$.outputs') AS output"
        " WHERE json_type(output.value, '$.artifact_id') = 'text'",
        19,
    ),
    "run masks": (
        "SELECT count(*) FROM runs WHERE json_type(settings_json, '$.mask.artifact_id') = 'text'",
        1,
    ),
    "work step masks": (
        "SELECT count(*) FROM work_steps"
        " WHERE json_type(settings_json, '$.mask.artifact_id') = 'text'",
        1,
    ),
    "studio sessions": (
        "SELECT count(*) FROM chats WHERE scope = 'studio'"
        " AND json_type(origin_json, '$.source_artifact_id') = 'text'",
        1,
    ),
    "job results": ("SELECT count(*) FROM jobs, json_each(result_json, '$.artifact_ids')", 19),
    "artifact metadata": (
        "SELECT count(*) FROM artifacts"
        " WHERE json_type(metadata_json, '$.poster_artifact_id') = 'text'",
        1,
    ),
}
#: The run inputs, which every run that takes one lists three times over: as
#: an input, as a resolved dependency, and in its work step's bindings.
INPUT_SOURCES = (
    "SELECT count(*) FROM runs, json_each(provenance_json, '$.input_artifact_ids')",
    "SELECT count(*) FROM runs, json_each(provenance_json, '$.resolved_dependency_artifact_ids')",
    "SELECT count(*) FROM work_steps, json_each(input_bindings_json) AS binding"
    " WHERE json_type(binding.value, '$.artifact_id') = 'text'",
)
#: Every stored document or column that could name a poster besides its video.
POSTER_NAMERS = (
    "SELECT count(*) FROM runs"
    " WHERE instr(provenance_json, :poster) OR instr(settings_json, :poster)",
    "SELECT count(*) FROM work_steps"
    " WHERE instr(settings_json, :poster) OR instr(input_bindings_json, :poster)",
    "SELECT count(*) FROM jobs WHERE instr(payload_json, :poster) OR instr(result_json, :poster)",
    "SELECT count(*) FROM message_references WHERE instr(artifact_ids_json, :poster)",
    "SELECT count(*) FROM chats WHERE instr(origin_json, :poster)",
    "SELECT count(*) FROM message_parts WHERE artifact_id = :poster",
    "SELECT count(*) FROM response_revision_parts WHERE artifact_id = :poster",
    "SELECT count(*) FROM artifact_library_entries WHERE artifact_id = :poster",
    "SELECT count(*) FROM run_context_artifacts WHERE artifact_id = :poster",
    "SELECT count(*) FROM artifacts WHERE kind != 'video' AND instr(metadata_json, :poster)",
)


def _script() -> dict[str, Any]:
    return runpy.run_path(str(SCRIPT))


def _scalar(connection: sqlite3.Connection, statement: str, parameters: Any = ()) -> int:
    return int(connection.execute(statement, parameters).fetchone()[0])


def _settings(data_dir: Path) -> Settings:
    return Settings(
        data_dir=data_dir,
        comfy_directory=None,
        dev=True,
        chat_engine="mock",
        media_engine="mock",
    )


def _environment(tmp_path: Path, **extra: str) -> dict[str, str]:
    """This process's environment, with an installed application's data folder under tmp_path."""

    stand_in = tmp_path / "installed-root"
    return dict(os.environ, LOCALAPPDATA=str(stand_in), XDG_DATA_HOME=str(stand_in), **extra)


def _link_folder(link: Path, target: Path) -> None:
    """Point `link` at `target`: a junction on Windows, which needs no privilege, else a symlink."""

    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=True,
        )
    else:
        os.symlink(target, link, target_is_directory=True)
    assert link.is_junction() or link.is_symlink()


def test_a_small_store_holds_every_population_it_was_asked_for(tmp_path: Path) -> None:
    data_dir = tmp_path / "store"
    installed = tmp_path / "installed" / "data"
    reported: list[str] = []
    namespace = _script()

    rows = namespace["build_store"](
        data_dir, scale=0.001, seed=7, report=reported.append, installed=installed
    )

    assert rows == EXPECTED_ROWS
    # The folder is named before the build writes anything.
    assert reported[0] == f"data-dir {data_dir}"
    assert any(line.startswith(f"counted-rows={COUNTED_ROWS} ") for line in reported)
    with closing(sqlite3.connect(data_dir / "state" / "local-lm.sqlite3")) as connection:
        for table, count in EXPECTED_ROWS.items():
            assert _scalar(connection, f"SELECT count(*) FROM {table}") == count, table
        kinds = dict(connection.execute("SELECT kind, count(*) FROM artifacts GROUP BY kind"))
        assert kinds == KINDS
        # Previews, posters and uploads keep the keys the application writes
        # for them; only pictures and videos carry the padded document.
        keys = {
            kind: {
                tuple(sorted(json.loads(value)))
                for (value,) in connection.execute(
                    "SELECT metadata_json FROM artifacts WHERE kind = ?", (kind,)
                )
            }
            for kind in KINDS
        }
        assert keys == {
            "image": {("pad", "run_id"), ("pad",)},
            "input": {("uploaded",)},
            "thumbnail": {("run_id", "temporary_preview"), ("poster_for", "run_id")},
            "video": {("pad", "poster_artifact_id", "run_id")},
        }
        for source, (statement, count) in SOURCES.items():
            assert _scalar(connection, statement) == count, source
        inputs = [_scalar(connection, statement) for statement in INPUT_SOURCES]
        assert inputs[0] >= 1
        assert inputs == [inputs[0]] * 3
        # The video was stored first, then its poster, and then the video's
        # metadata was updated to name it.
        assert (
            _scalar(
                connection,
                "SELECT count(*) FROM artifacts AS video JOIN artifacts AS poster"
                " ON poster.id = json_extract(video.metadata_json, '$.poster_artifact_id')"
                " WHERE video.kind = 'video' AND poster.kind = 'thumbnail'"
                " AND poster.rowid > video.rowid AND poster.created_at > video.created_at"
                " AND video.updated_at = poster.created_at"
                " AND json_extract(poster.metadata_json, '$.poster_for') = video.id",
            )
            == KINDS["video"]
        )
        posters = [
            row[0]
            for row in connection.execute(
                "SELECT json_extract(metadata_json, '$.poster_artifact_id') FROM artifacts"
                " WHERE kind = 'video'"
            )
        ]
        # Nothing but its video names a poster, so only that link retains it.
        for poster in posters:
            for statement in POSTER_NAMERS:
                assert _scalar(connection, statement, {"poster": poster}) == 0, statement
        assert (
            _scalar(
                connection,
                "SELECT count(*) FROM artifacts"
                " WHERE json_type(metadata_json, '$.browser_proxy_artifact_id') = 'text'",
            )
            == 0
        )
        ages = dict(
            connection.execute(
                "SELECT CASE WHEN age < 1 THEN '0-1' WHEN age < 7 THEN '1-7'"
                " WHEN age < 30 THEN '7-30' ELSE '30-90' END, count(*)"
                " FROM (SELECT julianday('now') - julianday(created_at) AS age FROM artifacts"
                "  WHERE json_extract(metadata_json, '$.temporary_preview') = 1)"
                " GROUP BY 1"
            )
        )
        assert ages == PREVIEW_AGES
        assert (
            _scalar(connection, "SELECT count(*) FROM runs WHERE length(provenance_json) = 5600")
            == EXPECTED_ROWS["runs"]
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        # The checks a migration runs before trusting stored documents: valid
        # JSON within every bound, and every artifact a document names exists.
        assert _scalar(connection, PREMIGRATION_INVALID_SQL) == 0

    settings = _settings(data_dir)
    store = ArtifactStore(settings)
    engine = create_database_engine(settings)
    now = datetime.now(UTC)
    try:
        with Session(engine) as session:
            artifacts = session.scalars(select(Artifact)).all()
            for artifact in artifacts:
                assert artifact.size_bytes == 16
                # Hashes the file it finds where the store keeps that artifact.
                store.verified_path(artifact)
            retained = referenced_artifact_ids(session)
            unreferenced = {artifact.id for artifact in artifacts if artifact.id not in retained}
            previews = {
                artifact.id
                for artifact in artifacts
                if artifact.metadata_json.get("temporary_preview") is True
            }
            strays = {
                artifact.id
                for artifact in artifacts
                if artifact.kind == "input" and artifact.id not in retained
            }
            assert len(strays) == 1
            assert unreferenced == previews | strays
            assert set(posters) <= set(retained)
            # Each poster has one referrer, its video, which is what a deletion
            # of either has to reckon with.
            videos = {artifact.id for artifact in artifacts if artifact.kind == "video"}
            referrers = metadata_referrers(artifacts)
            assert set(referrers) == set(posters)
            assert all(len(named) == 1 and named <= videos for named in referrers.values())

            # Deletion's walk runs the stored-document check as well, and
            # refuses a store with any document it cannot accept.
            begin_artifact_write_fence(session)
            assert set(referenced_artifact_ids(session, for_deletion=True)) == set(retained)
            session.rollback()

            sweep = store.cleanup_retention(
                session, retention_days=30, temporary_hours=24, dry_run=True, now=now
            )
        assert sweep.examined_count == EXPECTED_ROWS["artifacts"]
        # The upload nothing names is marked; every preview a day old goes.
        assert sweep.marked_count == 1
        assert sweep.removed_count == sum(PREVIEW_AGES.values()) - PREVIEW_AGES["0-1"]
        assert sweep.removed_orphan_file_count == 0
    finally:
        engine.dispose()

    timed: list[str] = []
    namespace["calibrate"](data_dir, trials=1, report=timed.append, installed=installed)
    labels = [" ".join(line.split()[:2]) for line in timed[1:]]
    assert labels == [
        "W walk",
        "S referrers",
        "V stored",
        "V jobs",
        "V runs",
        "V work_steps",
        "V message_references",
        "V chats",
        "V artifacts",
        "L load",
        "F deletion-walk",
    ]
    assert "V chats rows=1 invalid=0 " in "\n".join(timed)
    assert all("invalid=0" in line for line in timed if line.startswith("V "))


def test_a_build_refuses_a_folder_that_already_holds_something(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "kept").write_bytes(b"kept")
    namespace = _script()

    with pytest.raises(namespace["StoreRefused"], match="already holds files"):
        namespace["build_store"](
            occupied, scale=0.001, report=lambda _line: None, installed=tmp_path / "installed"
        )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--scale", "0.001", "--data-dir", str(occupied)],
        capture_output=True,
        text=True,
        env=_environment(tmp_path),
        check=False,
        timeout=120,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "already holds files" in result.stderr
    assert [entry.name for entry in occupied.iterdir()] == ["kept"]
    assert (occupied / "kept").read_bytes() == b"kept"


def test_the_installed_data_folder_is_refused_however_it_is_spelled(tmp_path: Path) -> None:
    namespace = _script()
    refuse = namespace["_refuse_installed"]
    refused = namespace["StoreRefused"]
    installed = tmp_path / "Local" / "LMAtelier" / "data"
    installed.mkdir(parents=True)
    inside = installed / "bench"
    spellings = [
        installed,
        inside,
        # A folder holding it.
        installed.parent,
        installed / "unused" / ".." / "bench",
    ]
    if os.name == "nt":
        spellings += [
            Path("\\\\?\\" + str(inside)),
            Path(str(inside).swapcase()),
            Path(str(inside).replace("\\", "/")),
        ]
    alias = tmp_path / "alias"
    _link_folder(alias, installed.parent)
    spellings.append(alias / "data" / "bench")
    # The installed folder's own path may be the one spelled another way.
    installed_spellings = [alias / "data", installed / "unused" / ".."]
    if os.name == "nt":
        installed_spellings += [
            Path("\\\\?\\" + str(installed)),
            Path(str(installed).swapcase()),
        ]

    for spelling in spellings:
        with pytest.raises(refused, match="installed application"):
            refuse(spelling, installed=installed)
    for other in installed_spellings:
        with pytest.raises(refused, match="installed application"):
            refuse(inside, installed=other)
    if os.name == "nt":
        # Before an installation makes its folder, nothing resolves its case,
        # and Windows takes either spelling as the same folder.
        future = tmp_path / "Local" / "Later" / "data"
        with pytest.raises(refused, match="installed application"):
            refuse(Path(str(future / "bench").swapcase()), installed=future)
    # A folder beside it, even one whose name begins the same way, is fine.
    refuse(installed.parent / "data-copy" / "bench", installed=installed)
    refuse(tmp_path / "elsewhere", installed=installed)

    # The build refuses before it writes anything.
    with pytest.raises(refused, match="installed application"):
        namespace["build_store"](
            spellings[-1], scale=0.001, report=lambda _line: None, installed=installed
        )
    assert list(installed.iterdir()) == []


def test_an_alias_no_spelling_reveals_is_still_the_installed_folder(tmp_path: Path) -> None:
    # A share or a mount of the same folder resolves to a different spelling;
    # the folder itself, by volume and file number, is the same one.
    namespace = _script()
    installed = tmp_path / "installed"
    (installed / "inside").mkdir(parents=True)
    spelled = namespace["_comparable"]
    calls = itertools.count()
    # Every spelling differs from every other, so only the folders' identity can match.
    namespace["_comparable"] = lambda path: (f"alias {next(calls)}", *spelled(path))
    try:
        assert namespace["_overlaps"](installed / "inside" / "store", installed)
        assert namespace["_overlaps"](tmp_path, installed)
        assert not namespace["_overlaps"](tmp_path / "elsewhere", installed)
    finally:
        namespace["_comparable"] = spelled


def test_the_installed_folder_is_refused_before_it_is_listed(tmp_path: Path) -> None:
    namespace = _script()
    installed = tmp_path / "installed"
    (installed / "state").mkdir(parents=True)
    with pytest.raises(namespace["StoreRefused"], match="installed application"):
        namespace["build_store"](
            installed, scale=0.001, report=lambda _line: None, installed=installed
        )


def test_calibrate_refuses_a_store_reached_through_links(tmp_path: Path) -> None:
    namespace = _script()
    refused = namespace["StoreRefused"]
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "state").mkdir(parents=True)
    (elsewhere / "sha256").mkdir()
    (elsewhere / "state" / "local-lm.sqlite3").write_bytes(b"reached through a link")
    # Its own file, so the hard-link case cannot be refused for the link case's reason.
    held = tmp_path / "held-elsewhere.sqlite3"
    held.write_bytes(b"held")

    def finished(name: str) -> Path:
        store = tmp_path / name
        (store / "state").mkdir(parents=True)
        (store / "artifacts" / "sha256").mkdir(parents=True)
        (store / "state" / "local-lm.sqlite3").write_bytes(b"")
        marker = {"format": namespace["MARKER_FORMAT"]}
        (store / namespace["MARKER"]).write_text(json.dumps(marker), encoding="utf-8")
        return store

    # A store's own folders pass.
    assert namespace["_built_store"](finished("plain")) is not None

    state = finished("linked-state")
    (state / "state" / "local-lm.sqlite3").unlink()
    (state / "state").rmdir()
    _link_folder(state / "state", elsewhere / "state")
    artifacts = finished("linked-artifacts")
    (artifacts / "artifacts" / "sha256").rmdir()
    _link_folder(artifacts / "artifacts" / "sha256", elsewhere / "sha256")
    named_twice = finished("linked-database")
    (named_twice / "state" / "local-lm.sqlite3").unlink()
    os.link(held, named_twice / "state" / "local-lm.sqlite3")

    for store, reason in (
        (state, "lead outside it"),
        (artifacts, "lead outside it"),
        (named_twice, "another name elsewhere"),
    ):
        with pytest.raises(refused, match=reason):
            namespace["calibrate"](
                store, trials=1, report=lambda _line: None, installed=tmp_path / "installed"
            )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "calibrate", "--trials", "1", "--data-dir", str(state)],
        capture_output=True,
        text=True,
        env=_environment(tmp_path),
        check=False,
        timeout=120,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert result.stderr.startswith("refused: ")
    # Nothing was opened through any of them.
    assert sorted(entry.name for entry in (elsewhere / "state").iterdir()) == ["local-lm.sqlite3"]
    assert (elsewhere / "state" / "local-lm.sqlite3").read_bytes() == b"reached through a link"
    assert held.read_bytes() == b"held"


def test_a_default_build_names_its_folder_before_it_writes(tmp_path: Path) -> None:
    temporary = tmp_path / "temporary"
    temporary.mkdir()

    # Provenance over the application's limit for one stored document: the
    # database refuses the runs once the build is well under way.
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "build",
            "--scale",
            "0.001",
            "--provenance-bytes",
            "1100000",
        ],
        capture_output=True,
        text=True,
        env=_environment(tmp_path, TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary)),
        check=False,
        timeout=300,
    )

    assert result.returncode == 1, result.stdout + result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("data-dir ")
    assert any(line.startswith("artifacts rows=") for line in lines)
    folder = Path(lines[0].removeprefix("data-dir "))
    assert folder.resolve().is_relative_to(temporary.resolve())
    assert (folder / "state" / "local-lm.sqlite3").is_file()
    assert not (folder / _script()["MARKER"]).exists()


def test_engine_facts_measure_every_behaviour_in_order() -> None:
    facts = _script()["engine_facts"]()

    assert [line.split()[0] for line in facts] == [f"E{number}" for number in range(1, 10)]
    assert all(len(line.split()) > 1 for line in facts)
    conflicts = {
        name: value for name, value in (field.split("=", 1) for field in facts[4].split()[1:])
    }
    assert list(conflicts) == [
        "plain-insert-row-present",
        "plain-insert-audit",
        "outer-replace-row-present",
        "outer-replace-audit",
        "outer-abort-failed",
        "outer-abort-row-present",
        "outer-abort-audit",
        "replace-without-rowid-rows-after",
    ]
    # Whichever clause the trigger's insert ran under, the pending change row
    # is still there; only its audit flag can differ, and it is reported apart.
    assert [conflicts[f"{name}-row-present"] for name in ("plain-insert", "outer-replace")] == [
        "1",
        "1",
    ]
    assert conflicts["outer-abort-row-present"] == "1"
    assert {conflicts[f"{name}-audit"] for name in ("plain-insert", "outer-replace")} <= {"0", "1"}
