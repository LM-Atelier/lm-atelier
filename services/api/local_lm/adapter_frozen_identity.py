"""The file digest an adapter was bound to, or nothing at all.

Grammar review is bound to a specific adapter file, so consulting a review at
run time needs the digest of the file that will actually load. The obvious
source is wrong: `ModelAssetInstall.manifest_json["sha256"]` is mutable and can
be rewritten under a row that never changed, so a review would appear to still
apply to a file it was never about.

The authoritative record is the activation's own dependency binding, whose
`resource_identity_json` was materialized and digest-verified when the
activation was built. That record only exists for adapters selected as workflow
dependencies, and an adapter chosen from turn settings need not be one. So this
returns nothing rather than guessing, and every caller must treat nothing as
"do not apply the grammar" rather than as "apply it anyway".

There is deliberately no fallback. A fallback here would be correct in the
common case and silently wrong in exactly the case the review exists to catch.
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import WorkflowDependencyBinding


def frozen_adapter_digest(
    session: Session,
    *,
    workflow_activation_id: str,
    model_asset_install_id: str,
) -> str | None:
    """The verified digest this activation bound for one adapter, or `None`.

    `None` covers three different situations on purpose - no binding, more than
    one, and a binding whose identity does not carry a usable digest - because
    the caller does the same thing in all three: declines to apply a grammar.
    Distinguishing them here would invite a caller to treat one of them as
    recoverable.
    """

    if not workflow_activation_id or not model_asset_install_id:
        return None
    bindings = session.scalars(
        select(WorkflowDependencyBinding).where(
            WorkflowDependencyBinding.workflow_activation_id == workflow_activation_id,
            WorkflowDependencyBinding.model_asset_install_id == model_asset_install_id,
        )
    ).all()
    # More than one binding means the activation used this adapter for more than
    # one requirement. Nothing here can say which file a grammar was reviewed
    # against, and picking either would be a guess wearing a digest.
    if len(bindings) != 1:
        return None
    identity = bindings[0].resource_identity_json
    if not isinstance(identity, dict):
        return None
    digest = identity.get("sha256")
    if not isinstance(digest, str):
        return None
    cleaned = digest.strip().lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        return None
    return cleaned


def frozen_adapter_digests(
    session: Session,
    *,
    workflow_activation_id: str,
    model_asset_install_ids: Iterable[str],
) -> dict[str, str | None]:
    """The same answer for a whole selected stack, keyed by install.

    Every requested install appears in the result, including the ones with no
    answer. A caller iterating the returned mapping therefore cannot skip an
    adapter by forgetting it was absent.
    """

    return {
        install_id: frozen_adapter_digest(
            session,
            workflow_activation_id=workflow_activation_id,
            model_asset_install_id=install_id,
        )
        for install_id in dict.fromkeys(model_asset_install_ids)
    }
