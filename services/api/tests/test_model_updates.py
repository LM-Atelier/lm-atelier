"""Staleness comparison: honest identity in, honest verdicts out."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.model_updates import (
    InstalledCivitaiIdentity,
    installed_civitai_identities,
    newer_version,
)
from local_lm.models import ModelAssetInstall


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value


def _asset(name: str, metadata: Any) -> ModelAssetInstall:
    return ModelAssetInstall(
        name=name,
        kind="lora",
        family="sdxl",
        local_path=f"C:/models/{name}.safetensors",
        size_bytes=10,
        manifest_json={"metadata": metadata} if metadata is not None else {},
        active=False,
    )


def _identity(**updates: Any) -> InstalledCivitaiIdentity:
    value = InstalledCivitaiIdentity(
        install_id="asset-1",
        name="portrait-lora",
        kind="lora",
        model_id="101",
        version_id="201",
        version_name="v1",
        published_at="2026-06-01T00:00:00Z",
    )
    return InstalledCivitaiIdentity(**{**value.__dict__, **updates})


def _versions(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"model_id": "101", "model_name": "Portrait", "versions": list(entries)}


def test_only_rows_naming_an_exact_version_are_comparable(session: Session) -> None:
    session.add(
        _asset(
            "civitai-lora",
            {
                "provider": "civitai",
                "source_model_id": "101",
                "source_version_id": "201",
                "version_name": "v1",
                "published_at": "2026-06-01T00:00:00Z",
            },
        )
    )
    session.add(_asset("huggingface-lora", {"provider": "huggingface"}))
    session.add(_asset("unversioned", {"provider": "civitai", "source_model_id": "102"}))
    session.add(_asset("no-metadata", None))
    session.commit()

    identities = installed_civitai_identities(session)

    assert [identity.name for identity in identities] == ["civitai-lora"]
    assert identities[0].model_id == "101"
    assert identities[0].version_id == "201"


def test_the_newest_strictly_later_version_wins(session: Session) -> None:
    candidate = newer_version(
        _identity(),
        _versions(
            {"version_id": "203", "version_name": "v3", "published_at": "2026-07-20T00:00:00Z"},
            {
                "version_id": "204",
                "version_name": "v4",
                "published_at": "2026-07-30T00:00:00Z",
                "base_model": "SDXL 1.0",
                "changelog": "Sharper hands",
            },
            {"version_id": "201", "version_name": "v1", "published_at": "2026-06-01T00:00:00Z"},
        ),
    )

    assert candidate is not None
    assert candidate.version_id == "204"
    assert candidate.changelog == "Sharper hands"
    assert candidate.base_model == "SDXL 1.0"


def test_reordering_without_anything_newer_is_not_an_update(session: Session) -> None:
    assert (
        newer_version(
            _identity(),
            _versions(
                {"version_id": "200", "published_at": "2026-05-01T00:00:00Z"},
                {"version_id": "201", "published_at": "2026-06-01T00:00:00Z"},
            ),
        )
        is None
    )


def test_equal_timestamps_are_not_newer(session: Session) -> None:
    assert (
        newer_version(
            _identity(),
            _versions(
                {"version_id": "201", "published_at": "2026-06-01T00:00:00Z"},
                {"version_id": "202", "published_at": "2026-06-01T00:00:00Z"},
            ),
        )
        is None
    )


def test_a_delisted_install_compares_by_its_manifest_timestamp(session: Session) -> None:
    candidate = newer_version(
        _identity(),
        _versions({"version_id": "205", "published_at": "2026-07-01T00:00:00Z"}),
    )
    assert candidate is not None
    assert candidate.version_id == "205"


def test_no_timestamp_anywhere_means_no_honest_order(session: Session) -> None:
    assert (
        newer_version(
            _identity(published_at=None),
            _versions({"version_id": "205", "published_at": "2026-07-01T00:00:00Z"}),
        )
        is None
    )


def test_the_provider_list_refreshes_the_installed_timestamp(session: Session) -> None:
    """The provider's timestamp for the installed version outranks the manifest's."""
    candidate = newer_version(
        _identity(published_at="2020-01-01T00:00:00Z"),
        _versions(
            {"version_id": "201", "published_at": "2026-07-15T00:00:00Z"},
            {"version_id": "202", "published_at": "2026-07-01T00:00:00Z"},
        ),
    )
    assert candidate is None


def test_malformed_provider_entries_are_ignored(session: Session) -> None:
    candidate = newer_version(
        _identity(),
        _versions(
            {"published_at": "2026-08-01T00:00:00Z"},
            {"version_id": "", "published_at": "2026-08-01T00:00:00Z"},
            {"version_id": "206", "published_at": "not a date"},
            {"version_id": "207", "published_at": "2026-07-10T00:00:00Z"},
        ),
    )
    assert candidate is not None
    assert candidate.version_id == "207"
