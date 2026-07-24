from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import select

from .db import SessionLocal
from .models import AppSetting, ModelInstall, ModelProfile

if TYPE_CHECKING:
    from .main import Services

logger = logging.getLogger(__name__)

LAST_CHAT_PROFILE_KEY = "workers.last_chat_profile_id"


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
                )
                .order_by(ModelProfile.updated_at.desc(), ModelProfile.id)
            )
        if not profile or not profile.model_install_id:
            return None
        install = session.get(ModelInstall, profile.model_install_id)
        if not install or not install.active:
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


async def restore_configured_workers(services: Services) -> None:
    """Start configured local engines without delaying API availability."""

    settings = services.settings
    if (
        settings.media_engine == "comfyui"
        and settings.comfy_executable
        and settings.comfy_directory
    ):
        try:
            await services.processes.start_media()
            logger.info("Restored the configured media worker")
            refreshed = await services.downloads.refresh_installed_media_workflows()
            if refreshed:
                logger.info("Refreshed %s installed media workflows", refreshed)
        except Exception:
            logger.exception("Could not restore the configured media worker")

    if settings.chat_engine == "llama.cpp" and settings.llama_executable:
        selected = chat_profile_to_restore()
        if not selected:
            logger.info("No installed llama.cpp chat profile is available to restore")
            return
        profile, install = selected
        try:
            await services.processes.load_chat(profile, install)
            remember_chat_profile(profile.id)
            logger.info("Restored chat worker profile %s", profile.id)
        except Exception:
            logger.exception("Could not restore chat worker profile %s", profile.id)
