from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from local_lm.db import SessionLocal
from local_lm.models import AppSetting, ModelInstall, ModelProfile
from local_lm.worker_startup import LAST_CHAT_PROFILE_KEY, restore_configured_workers


async def test_configured_workers_restore_with_last_chat_profile(
    client, settings, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    del client
    install = ModelInstall(
        id="model_restore",
        name="Restored model",
        role="chat",
        engine="llama.cpp",
        local_path=str(tmp_path / "model.gguf"),
        active=True,
    )
    profile = ModelProfile(
        id="profile_restore",
        model_install_id=install.id,
        name="Restored profile",
        role="chat",
        engine="llama.cpp",
    )
    with SessionLocal() as session:
        session.add_all(
            [
                install,
                profile,
                AppSetting(key=LAST_CHAT_PROFILE_KEY, value_json=profile.id),
            ]
        )
        session.commit()

    settings.chat_engine = "llama.cpp"
    settings.llama_executable = tmp_path / "llama-server"
    settings.media_engine = "comfyui"
    settings.comfy_executable = tmp_path / "python"
    settings.comfy_directory = tmp_path / "ComfyUI"
    processes = SimpleNamespace(start_media=AsyncMock(), load_chat=AsyncMock())
    services = SimpleNamespace(settings=settings, processes=processes)

    await restore_configured_workers(services)  # type: ignore[arg-type]

    processes.start_media.assert_awaited_once_with()
    restored_profile, restored_install = processes.load_chat.await_args.args
    assert restored_profile.id == profile.id
    assert restored_install.id == install.id


async def test_worker_restore_failure_does_not_prevent_other_worker(
    client, settings, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    del client
    settings.chat_engine = "mock"
    settings.media_engine = "comfyui"
    settings.comfy_executable = tmp_path / "python"
    settings.comfy_directory = tmp_path / "ComfyUI"
    processes = SimpleNamespace(
        start_media=AsyncMock(side_effect=RuntimeError("worker unavailable")),
        load_chat=AsyncMock(),
    )

    await restore_configured_workers(  # type: ignore[arg-type]
        SimpleNamespace(settings=settings, processes=processes)
    )

    processes.start_media.assert_awaited_once_with()
    processes.load_chat.assert_not_awaited()
