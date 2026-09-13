"""Suggest well-rated general-audience LoRAs for the model a workflow runs.

Suggestions are fetched from CivitAI when asked for, filtered to the base
models that fit the workflow's own model family, and never installed or
applied here: a person chooses one, and the ordinary catalog preflight decides
whether it can be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

from .auxiliary_assets import workflow_model_family
from .model_updates import installed_civitai_identities
from .models import WorkflowRevision
from .schemas import CatalogModel, CatalogPage

MAX_LORA_SUGGESTIONS = 12

# Local model families, as installs record them, to the base-model labels
# CivitAI files LoRAs under. Both SDXL spellings appear: repository inspection
# records "stable-diffusion-xl" and component inspection records "sdxl".
# A family missing here gets no suggestions rather than LoRAs for another model.
CIVITAI_BASE_MODELS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "stable-diffusion-xl": ("SDXL 1.0",),
    "sdxl": ("SDXL 1.0",),
    "stable-diffusion": ("SD 1.5",),
    "flux": ("Flux.1 D", "Flux.1 S"),
}

LoraSuggestionGap = Literal["family_unknown", "family_unsupported"]


@dataclass(frozen=True)
class LoraSuggestionScope:
    """What a revision's suggestions are for, decided before any network request."""

    family: str | None
    base_models: tuple[str, ...]
    gap: LoraSuggestionGap | None
    installed_model_ids: frozenset[str]


def lora_suggestion_scope(session: Session, revision: WorkflowRevision) -> LoraSuggestionScope:
    family = workflow_model_family(session, revision)
    installed = frozenset(identity.model_id for identity in installed_civitai_identities(session))
    if family is None:
        return LoraSuggestionScope(None, (), "family_unknown", installed)
    base_models = CIVITAI_BASE_MODELS_BY_FAMILY.get(family.casefold())
    if base_models is None:
        return LoraSuggestionScope(family, (), "family_unsupported", installed)
    return LoraSuggestionScope(family, base_models, None, installed)


def suggested_loras(scope: LoraSuggestionScope, page: CatalogPage) -> list[CatalogModel]:
    """Keep one installable, not yet installed card per LoRA, in the provider's order."""

    suggestions: list[CatalogModel] = []
    seen: set[str] = set()
    for item in page.items:
        model_id = item.parent_model_id or item.remote_id
        if (
            item.provider != "civitai"
            or item.content_rating != "general"
            or item.compatibility == "unsupported"
            or model_id in seen
            or model_id in scope.installed_model_ids
        ):
            continue
        seen.add(model_id)
        suggestions.append(item)
        if len(suggestions) == MAX_LORA_SUGGESTIONS:
            break
    return suggestions
