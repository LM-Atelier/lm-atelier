"""Remap imported workflow references without restoring local execution authority."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Base
from .project_dependencies import ImportedDependencies
from .workflow_recovery_schema import WORKFLOW_IDENTITY_KEYS

_DEFINITION_KEYS = frozenset({"workflow_definition_id", "workflow_id"})
_REVISION_KEYS = frozenset({"workflow_revision_id", "source_workflow_revision_id"})
_LOCAL_KEYS = frozenset(
    {"workflow_family_id", "workflow_activation_id", "workflow_install_offer_id"}
)
_GENERIC_KEYS = frozenset(
    {"id", "definition_id", "revision_id", "family_id", "activation_id", "offer_id"}
)


def remap_workflow_provenance(
    session: Session,
    provenance: dict[str, Any],
    dependencies: ImportedDependencies | None,
) -> None:
    """Map portable identities and clear local workflow references at every depth."""
    definitions = dependencies.workflow_ids if dependencies else {}
    revisions = dependencies.revision_ids if dependencies else {}
    mappings = {**definitions, **revisions}
    imported_definitions = set(definitions.values())
    imported_revisions = set(revisions.values())
    imported_ids = imported_definitions | imported_revisions
    nodes: list[dict[str, Any]] = []
    pending: list[Any] = [provenance]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            nodes.append(value)
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)

    # Generic keys also describe models, presets and artifacts. Only a known
    # workflow identity may change under one of those keys.
    generic_ids = sorted(
        {
            value
            for node in nodes
            for key, value in node.items()
            if key in _GENERIC_KEYS and isinstance(value, str)
        }
    )
    local_ids: set[str] = set()
    for offset in range(0, len(generic_ids), 500):
        candidates = generic_ids[offset : offset + 500]
        for name in (
            "workflow_families",
            "workflow_definitions",
            "workflow_revisions",
            "workflow_activations",
            "workflow_install_offers",
        ):
            identifiers = Base.metadata.tables[name].c.id
            local_ids.update(
                str(identifier)
                for identifier in session.scalars(
                    select(identifiers).where(identifiers.in_(candidates))
                )
            )

    identity_keys = set(WORKFLOW_IDENTITY_KEYS) | _DEFINITION_KEYS | _GENERIC_KEYS
    for node in nodes:
        for key in node.keys() & identity_keys:
            value = node[key]
            if key in _LOCAL_KEYS:
                node[key] = None
            elif key in _DEFINITION_KEYS:
                node[key] = (
                    value
                    if isinstance(value, str) and value in imported_definitions
                    else definitions.get(value)
                    if isinstance(value, str)
                    else None
                )
            elif key in _REVISION_KEYS:
                node[key] = (
                    value
                    if isinstance(value, str) and value in imported_revisions
                    else revisions.get(value)
                    if isinstance(value, str)
                    else None
                )
            elif isinstance(value, str) and key in _GENERIC_KEYS:
                if value in mappings:
                    node[key] = mappings[value]
                elif value in local_ids and value not in imported_ids:
                    node[key] = None
