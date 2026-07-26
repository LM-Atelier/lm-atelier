from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from .db import SessionLocal
from .models import AppSetting, ModelInstall, ModelProfile
from .profile_service import LAST_CHAT_PROFILE_KEY, validate_profile_binding

if TYPE_CHECKING:
    from .main import Services

logger = logging.getLogger(__name__)


def remember_chat_profile(profile_id: str) -> None:
    with SessionLocal() as session:
        setting = session.get(AppSetting, LAST_CHAT_PROFILE_KEY)
        if setting:
            setting.value_json = profile_id
        else:
            session.add(AppSetting(key=LAST_CHAT_PROFILE_KEY, value_json=profile_id))
        session.commit()


def chat_profile_to_restore() -> tuple[ModelProfile, ModelInstall] | None:
    with SessionLocal() as session:
        setting = session.get(AppSetting, LAST_CHAT_PROFILE_KEY)
        profile = (
            session.get(ModelProfile, setting.value_json)
            if setting and isinstance(setting.value_json, str)
            else None
        )
        if not _restorable_chat_profile(profile):
            profile = session.scalar(
                select(ModelProfile)
                .join(ModelInstall, ModelInstall.id == ModelProfile.model_install_id)
                .where(
                    ModelProfile.role == "chat",
                    ModelProfile.engine == "llama.cpp",
                    ModelInstall.active.is_(True),
                    ModelInstall.role == ModelProfile.role,
                    ModelInstall.engine == ModelProfile.engine,
                )
                .order_by(ModelProfile.updated_at.desc(), ModelProfile.id)
            )
        if not profile or not profile.model_install_id:
            return None
        try:
            install = validate_profile_binding(session, profile)
        except (LookupError, ValueError):
            return None
        if not install:
            return None
        session.expunge(profile)
        session.expunge(install)
        return profile, install


def _restorable_chat_profile(profile: ModelProfile | None) -> bool:
    return bool(
        profile
        and profile.role == "chat"
        and profile.engine == "llama.cpp"
        and profile.model_install_id
    )


def _media_worker_should_restore(services: Services) -> bool:
    settings = services.settings
    if settings.comfy_executable and settings.comfy_directory:
        return True
    with SessionLocal() as session:
        return (
            session.scalar(
                select(ModelInstall.id)
                .where(
                    ModelInstall.active.is_(True),
                    ModelInstall.engine == "comfyui",
                    ModelInstall.role.in_(("image", "video")),
                )
                .limit(1)
            )
            is not None
        )


async def restore_configured_workers(services: Services) -> None:
    """Restore local workers, provisioning their supported runtimes when needed."""

    settings = services.settings
    if settings.media_engine == "comfyui" and _media_worker_should_restore(services):
        try:
            async with services.scheduler.lease("primary"):
                await services.processes.start_media()
                refreshed = await services.downloads.refresh_installed_media_workflows()
            logger.info("Restored the configured media worker")
            if refreshed:
                logger.info("Refreshed %s installed media workflows", refreshed)
        except Exception:
            logger.exception("Could not restore the configured media worker")

    if settings.chat_engine == "llama.cpp":
        selected = chat_profile_to_restore()
        if not selected:
            logger.info("No installed llama.cpp chat profile is available to restore")
            return
        profile, install = selected
        try:
            async with services.scheduler.lease("primary"):
                await services.processes.load_chat(profile, install)
            remember_chat_profile(profile.id)
            logger.info("Restored chat worker profile %s", profile.id)
        except Exception:
            logger.exception("Could not restore chat worker profile %s", profile.id)
