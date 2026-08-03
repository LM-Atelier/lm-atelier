from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from .domain import new_id
from .models import AppSetting, Chat, ModelInstall, ModelProfile
from .schemas import SettingField
from .settings_registry import (
    CHAT_SETTINGS,
    IMAGE_SETTINGS,
    VIDEO_SETTINGS,
    compatible_stored_settings,
)
from .workflow_compatibility import (
    ChatSelectorCapability,
    ensure_legacy_profile_workflow,
    mirror_legacy_chat_workflow_selections,
)

AUTO_PROFILE_ID = "__auto__"
LAST_CHAT_PROFILE_KEY = "workers.last_chat_profile_id"


def validate_profile_install(
    session: Session,
    *,
    model_install_id: str | None,
    role: str,
    engine: str,
) -> ModelInstall | None:
    """Return a usable bound install, rejecting stale or incompatible bindings."""

    if not model_install_id:
        return None
    install = session.get(ModelInstall, model_install_id)
    if not install:
        raise LookupError("model install not found")
    if not install.active:
        raise ValueError("model install is inactive")
    if install.role != role:
        raise ValueError(f"model install role is {install.role}, not {role}")
    if install.engine != engine:
        raise ValueError(f"model install engine is {install.engine}, not {engine}")
    return install


def validate_profile_binding(session: Session, profile: ModelProfile) -> ModelInstall | None:
    return validate_profile_install(
        session,
        model_install_id=profile.model_install_id,
        role=profile.role,
        engine=profile.engine,
    )


def retire_profiles_for_installs(session: Session, install_ids: Iterable[str]) -> list[str]:
    """Remove inactive install profiles from defaults and explicit chat selections."""

    ids = {install_id for install_id in install_ids if install_id}
    if not ids:
        return []
    profiles = list(
        session.scalars(select(ModelProfile).where(ModelProfile.model_install_id.in_(ids))).all()
    )
    profile_ids = {profile.id for profile in profiles}
    affected_roles = {profile.role for profile in profiles}
    for profile in profiles:
        profile.is_default = False
    _reset_profile_selections(session, profile_ids)
    _restore_unbound_defaults(session, affected_roles)
    return sorted(profile_ids)


def reconcile_profile_bindings(session: Session) -> list[str]:
    """Retire persisted profile bindings that no longer describe a usable install."""

    invalid_profile_ids: set[str] = set()
    for profile in session.scalars(
        select(ModelProfile).where(ModelProfile.model_install_id.is_not(None))
    ).all():
        try:
            validate_profile_binding(session, profile)
        except (LookupError, ValueError):
            profile.is_default = False
            invalid_profile_ids.add(profile.id)
    _reset_profile_selections(session, invalid_profile_ids)
    _restore_unbound_defaults(session, {"chat", "image", "video"})
    return sorted(invalid_profile_ids)


def _reset_profile_selections(session: Session, profile_ids: set[str]) -> None:
    if not profile_ids:
        return
    chats = session.scalars(
        select(Chat).where(
            or_(
                Chat.active_chat_profile_id.in_(profile_ids),
                Chat.active_vision_profile_id.in_(profile_ids),
                Chat.active_image_profile_id.in_(profile_ids),
                Chat.active_video_profile_id.in_(profile_ids),
            )
        )
    ).all()
    for chat in chats:
        changed: list[ChatSelectorCapability] = []
        if chat.active_chat_profile_id in profile_ids:
            chat.active_chat_profile_id = AUTO_PROFILE_ID
            changed.append("chat")
        if chat.active_vision_profile_id in profile_ids:
            chat.active_vision_profile_id = AUTO_PROFILE_ID
            changed.append("vision")
        if chat.active_image_profile_id in profile_ids:
            chat.active_image_profile_id = AUTO_PROFILE_ID
            changed.append("image")
        if chat.active_video_profile_id in profile_ids:
            chat.active_video_profile_id = AUTO_PROFILE_ID
            changed.append("video")
        mirror_legacy_chat_workflow_selections(session, chat, changed)
    last_chat_profile = session.get(AppSetting, LAST_CHAT_PROFILE_KEY)
    if (
        last_chat_profile
        and isinstance(last_chat_profile.value_json, str)
        and last_chat_profile.value_json in profile_ids
    ):
        last_chat_profile.value_json = None


def _restore_unbound_defaults(session: Session, roles: Iterable[str]) -> None:
    for role in set(roles):
        existing_default = session.scalar(
            select(ModelProfile.id).where(
                ModelProfile.role == role,
                ModelProfile.is_default.is_(True),
            )
        )
        if existing_default:
            continue
        replacement = session.scalar(
            select(ModelProfile)
            .where(
                ModelProfile.role == role,
                ModelProfile.model_install_id.is_(None),
            )
            .order_by(ModelProfile.created_at, ModelProfile.id)
            .limit(1)
        )
        if replacement:
            replacement.is_default = True


def ensure_profile_for_install(
    session: Session,
    install: ModelInstall,
    *,
    default_settings: dict[str, object] | None = None,
    fields: Iterable[SettingField] | None = None,
) -> ModelProfile:
    validate_profile_install(
        session,
        model_install_id=install.id,
        role=install.role,
        engine=install.engine,
    )
    existing = session.scalar(
        select(ModelProfile)
        .where(
            ModelProfile.model_install_id == install.id,
            ModelProfile.role == install.role,
            ModelProfile.engine == install.engine,
        )
        .order_by(ModelProfile.created_at, ModelProfile.id)
        .limit(1)
    )
    if existing:
        ensure_legacy_profile_workflow(session, existing)
        return existing

    profile = build_profile_for_install(
        install,
        default_settings=default_settings,
        fields=fields,
    )
    session.add(profile)
    session.flush()
    ensure_legacy_profile_workflow(session, profile)
    return profile


def build_profile_for_install(
    install: ModelInstall,
    *,
    default_settings: dict[str, object] | None = None,
    fields: Iterable[SettingField] | None = None,
) -> ModelProfile:
    """Build an unpersisted profile for a provisional activation probe."""

    profile_fields = (
        list(fields)
        if fields is not None
        else {
            "chat": CHAT_SETTINGS,
            "image": IMAGE_SETTINGS,
            "video": VIDEO_SETTINGS,
        }[install.role]
    )
    values = default_settings or {}
    load_fields = [field for field in profile_fields if field.scope == "load"]
    request_fields = [field for field in profile_fields if field.scope != "load"]
    profile = ModelProfile(
        id=new_id("profile"),
        name=install.name,
        use_case="",
        role=install.role,
        engine=install.engine,
        model_install_id=install.id,
        load_settings_json=compatible_stored_settings(values, load_fields),
        request_settings_json=compatible_stored_settings(values, request_fields),
        is_default=False,
    )
    return profile
