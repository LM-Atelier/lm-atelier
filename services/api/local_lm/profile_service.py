from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import ModelInstall, ModelProfile
from .settings_registry import CHAT_SETTINGS, IMAGE_SETTINGS, VIDEO_SETTINGS

AUTO_PROFILE_ID = "__auto__"


def ensure_profile_for_install(
    session: Session,
    install: ModelInstall,
    *,
    default_settings: dict[str, object] | None = None,
) -> ModelProfile:
    existing = session.scalar(
        select(ModelProfile)
        .where(ModelProfile.model_install_id == install.id)
        .order_by(ModelProfile.created_at, ModelProfile.id)
        .limit(1)
    )
    if existing:
        return existing

    fields = {
        "chat": CHAT_SETTINGS,
        "image": IMAGE_SETTINGS,
        "video": VIDEO_SETTINGS,
    }[install.role]
    values = default_settings or {}
    load_keys = {field.key for field in fields if field.scope == "load"}
    request_keys = {field.key for field in fields if field.scope != "load"}
    profile = ModelProfile(
        name=install.name,
        use_case="",
        role=install.role,
        engine=install.engine,
        model_install_id=install.id,
        load_settings_json={key: value for key, value in values.items() if key in load_keys},
        request_settings_json={key: value for key, value in values.items() if key in request_keys},
        is_default=False,
    )
    session.add(profile)
    session.flush()
    return profile
