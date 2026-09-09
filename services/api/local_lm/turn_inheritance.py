"""Internal source configuration passed between edited-turn admission stages."""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from .accepted_turn_context import AcceptedProfile, AcceptedWorkflow, resolve_accepted_profile
from .domain import Operation
from .models import ModelProfile
from .schemas import TurnRequest


@dataclass(frozen=True)
class TurnInheritance:
    profile: AcceptedProfile | None = None
    vision_profile: AcceptedProfile | None = None
    workflow: AcceptedWorkflow | None = None
    image_edit_strength: dict[str, Any] | None = None


TurnSourceResolver = Callable[
    [Session, TurnRequest, Operation, int | None],
    Awaitable[tuple[TurnRequest, TurnInheritance]],
]


def inherited_profile_configuration(session: Session, accepted: AcceptedProfile) -> ModelProfile:
    """Validate availability and project the accepted settings for admission."""
    if accepted.install is not None:
        try:
            return resolve_accepted_profile(session, accepted)[0]
        except RuntimeError as exc:
            raise ValueError("Accepted model installation is unavailable.") from exc
    current = session.get(ModelProfile, accepted.id)
    if (
        current is None
        or current.role != accepted.role
        or current.engine != accepted.engine
        or current.model_install_id is not None
    ):
        raise ValueError("Accepted model configuration is unavailable.")
    return ModelProfile(
        id=accepted.id,
        name=accepted.name,
        role=accepted.role,
        engine=accepted.engine,
        load_settings_json=copy.deepcopy(accepted.load_settings_json),
        request_settings_json=copy.deepcopy(accepted.request_settings_json),
    )


def inherited_edit_strength(
    strength: object, request: TurnRequest, operation: Operation
) -> dict[str, Any] | None:
    """Keep automatic strength unless a strength control or preset was selected."""
    if (
        operation != Operation.IMAGE_TO_IMAGE
        or not isinstance(strength, dict)
        or strength.get("mode") != "auto"
        or request.preset_id is not None
    ):
        return None
    parameter = strength.get("parameter")
    parameter = parameter if isinstance(parameter, str) else "denoise"
    if parameter in request.settings or "_image_edit_strength_mode" in request.settings:
        return None
    return copy.deepcopy(strength)
