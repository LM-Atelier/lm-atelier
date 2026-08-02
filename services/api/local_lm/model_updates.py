"""Staleness detection for installed models with first-class version identity.

An installed row is comparable only when its manifest records which provider
version it is - today that is CivitAI auxiliary assets, whose file metadata
carries `source_model_id` and `source_version_id`. Rows without that identity
are absent from the report rather than guessed at: a wrong "up to date" would
teach the user to stop checking, and a wrong "update available" would teach
them to stop believing it.

The comparison itself never touches the network; callers fetch the provider's
version list and pass it in. Choosing what "newer" means stays in one place:
strictly later `published_at` when both sides have one, and never mere
difference - a provider re-ordering its list must not read as an update.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ModelAssetInstall


@dataclass(frozen=True)
class InstalledCivitaiIdentity:
    """One installed row that names its exact provider version."""

    install_id: str
    name: str
    kind: str
    model_id: str
    version_id: str
    version_name: str | None
    published_at: str | None


@dataclass(frozen=True)
class ModelUpdateCandidate:
    """The newest general-audience version ahead of an installed one."""

    installed: InstalledCivitaiIdentity
    version_id: str
    version_name: str | None
    published_at: str
    base_model: str | None
    changelog: str | None


def installed_civitai_identities(session: Session) -> tuple[InstalledCivitaiIdentity, ...]:
    """Installed assets whose manifests name an exact CivitAI version."""
    identities: list[InstalledCivitaiIdentity] = []
    assets = session.scalars(select(ModelAssetInstall).order_by(ModelAssetInstall.name)).all()
    for asset in assets:
        metadata = asset.manifest_json.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("provider") != "civitai":
            continue
        model_id = metadata.get("source_model_id")
        version_id = metadata.get("source_version_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        if not isinstance(version_id, str) or not version_id:
            continue
        version_name = metadata.get("version_name")
        published_at = metadata.get("published_at")
        identities.append(
            InstalledCivitaiIdentity(
                install_id=asset.id,
                name=asset.name,
                kind=asset.kind,
                model_id=model_id,
                version_id=version_id,
                version_name=version_name if isinstance(version_name, str) else None,
                published_at=published_at if isinstance(published_at, str) else None,
            )
        )
    return tuple(identities)


def newer_version(
    installed: InstalledCivitaiIdentity,
    provider_versions: Any,
) -> ModelUpdateCandidate | None:
    """The newest version strictly ahead of the installed one, if any.

    `provider_versions` is the `versions()` summary from the catalog adapter.
    The installed version's own timestamp comes from the provider list when
    present there - the list is fresher than what install time recorded - and
    from the install manifest otherwise. An installed version the provider no
    longer lists (deleted, or no longer general-audience) compares by the
    manifest timestamp alone; without any timestamp there is no honest order,
    so there is no update.
    """
    entries = provider_versions.get("versions") if isinstance(provider_versions, dict) else None
    if not isinstance(entries, list):
        return None
    listed = {
        str(entry.get("version_id")): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("version_id")
    }
    current = listed.get(installed.version_id)
    baseline = _timestamp(
        current.get("published_at") if isinstance(current, dict) else installed.published_at
    )
    if baseline is None:
        return None
    best: ModelUpdateCandidate | None = None
    best_at: datetime | None = None
    for entry in listed.values():
        version_id = str(entry.get("version_id"))
        if version_id == installed.version_id:
            continue
        published = entry.get("published_at")
        published_at = _timestamp(published)
        if published_at is None or published_at <= baseline:
            continue
        if best_at is None or published_at > best_at:
            best_at = published_at
            version_name = entry.get("version_name")
            base_model = entry.get("base_model")
            changelog = entry.get("changelog")
            best = ModelUpdateCandidate(
                installed=installed,
                version_id=version_id,
                version_name=version_name if isinstance(version_name, str) else None,
                published_at=str(published),
                base_model=base_model if isinstance(base_model, str) else None,
                changelog=changelog if isinstance(changelog, str) else None,
            )
    return best


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
