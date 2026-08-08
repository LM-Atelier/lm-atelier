from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.models import AdapterPromptGrammar, ModelAssetInstall

ORIGINAL = "a" * 64
REPLACEMENT = "b" * 64
SOURCE = "c" * 64
GRAMMAR_DIGEST = "d" * 64


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def _install(session: Session) -> ModelAssetInstall:
    install = ModelAssetInstall(name="adapter", kind="lora", local_path="adapter.safetensors")
    session.add(install)
    session.flush()
    return install


def _grammar(install_id: str, **changes: object) -> AdapterPromptGrammar:
    values: dict[str, object] = {
        "model_asset_install_id": install_id,
        "asset_sha256": ORIGINAL,
        "source_identity": "sidecar document distributed with the adapter",
        "source_sha256": SOURCE,
        "schema_version": 1,
        "grammar_json": {"trigger": "TRIGGERWORD"},
        "grammar_sha256": GRAMMAR_DIGEST,
        "approved_prose_json": [],
        "verified_values_json": {},
        "compiler_version": "visual-prompt-compiler-v1",
        "compiler_ceiling": 900,
        "fits": True,
    }
    values.update(changes)
    return AdapterPromptGrammar(**values)


def test_a_grammar_records_the_file_it_describes(session: Session) -> None:
    """An install row is mutable and its file can be replaced underneath it. If
    the grammar were bound to the row alone, a new file would silently inherit
    the previous file's description of how to prompt it."""

    install = _install(session)
    session.add(_grammar(install.id))
    session.commit()

    stored = session.query(AdapterPromptGrammar).one()
    assert stored.asset_sha256 == ORIGINAL

    # The file is replaced; the row is untouched and still points at it.
    install.local_path = "adapter-v2.safetensors"
    session.commit()
    assert stored.model_asset_install_id == install.id
    assert stored.asset_sha256 != REPLACEMENT, (
        "the grammar still describes the file it was written about, which is what "
        "makes it recognisable as stale rather than silently wrong"
    )


def test_only_the_digest_of_the_source_document_is_kept(session: Session) -> None:
    """The document is third-party text on its way to a model that rewrites what
    the user wrote. There is deliberately nowhere here to put its contents."""

    install = _install(session)
    session.add(_grammar(install.id))
    session.commit()

    columns = {column.name for column in AdapterPromptGrammar.__table__.columns}
    assert "source_sha256" in columns
    assert not {"source_text", "source_body", "raw_json"} & columns


def test_one_grammar_per_install(session: Session) -> None:
    """A second would raise the question of which one a rewriter believed, and
    the answer has to be that there is only one."""

    install = _install(session)
    session.add(_grammar(install.id))
    session.commit()

    session.add(_grammar(install.id, asset_sha256=REPLACEMENT))
    with pytest.raises(IntegrityError):
        session.commit()


def test_prose_is_unapproved_until_its_exact_text_is_named(session: Session) -> None:
    """Approval is a list of content digests rather than a flag, because a flag
    approves a field while the danger is in the characters. Nothing is approved
    until some exact text has been."""

    install = _install(session)
    session.add(_grammar(install.id))
    session.commit()

    stored = session.query(AdapterPromptGrammar).one()
    assert stored.approved_prose_json == []
    assert stored.verified_values_json == {}
    assert stored.reviewed_at is None


@pytest.mark.parametrize("column", ["asset_sha256", "source_sha256"])
def test_a_digest_must_look_like_one(session: Session, column: str) -> None:
    install = _install(session)
    session.add(_grammar(install.id, **{column: "NOTAHASH"}))
    with pytest.raises(IntegrityError):
        session.commit()
