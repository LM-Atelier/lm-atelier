from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from local_lm.adapters.conformance import probe_chat_adapter, probe_media_adapter
from local_lm.adapters.discovery import AdapterLoadError, load_external_adapter
from local_lm.adapters.mock import MockChatAdapter, MockMediaAdapter
from local_lm.config import Settings
from local_lm.engines import EngineRegistry
from local_lm.schemas import EngineCapabilities
from local_lm.settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS


async def test_builtin_adapters_pass_public_conformance_probes() -> None:
    chat = MockChatAdapter()
    media = MockMediaAdapter()
    try:
        (await probe_chat_adapter(chat)).assert_passed()
        (await probe_media_adapter(media)).assert_passed()
    finally:
        await chat.close()
        await media.close()


async def test_builtin_media_capabilities_isolate_role_settings() -> None:
    adapter = MockMediaAdapter()
    try:
        capabilities = await adapter.capabilities()
    finally:
        await adapter.close()

    assert capabilities.settings_by_role == {
        "image": IMAGE_SETTINGS,
        "video": VIDEO_SETTINGS,
    }
    assert capabilities.settings == [*IMAGE_SETTINGS, *VIDEO_SETTINGS]


def test_role_settings_are_additive_for_legacy_adapter_capabilities() -> None:
    capabilities = EngineCapabilities(
        engine="legacy-media",
        version="1",
        roles=["image", "video"],
        operations=["text_to_image", "text_to_video"],
        formats=["legacy"],
        devices=["cpu:0"],
        streaming=False,
        tool_calling=False,
        settings=IMAGE_SETTINGS,
        healthy=True,
    )

    assert capabilities.settings_by_role == {}


@dataclass
class FakeEntryPoint:
    name: str
    factory: Any

    def load(self) -> Any:
        return self.factory


def test_external_adapter_requires_matching_version_and_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def factory(_settings: Settings) -> MockChatAdapter:
        return MockChatAdapter()

    factory.lm_atelier_contract_version = 1  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "local_lm.adapters.discovery.entry_points",
        lambda *, group: [FakeEntryPoint("example", factory)] if "chat" in group else [],
    )
    settings = Settings(data_dir=tmp_path, chat_engine="example")
    registry = EngineRegistry(settings)
    assert isinstance(registry.chat, MockChatAdapter)


def test_unknown_or_incompatible_adapter_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("local_lm.adapters.discovery.entry_points", lambda *, group: [])
    with pytest.raises(AdapterLoadError, match="unknown chat engine"):
        EngineRegistry(Settings(data_dir=tmp_path, chat_engine="typo"))

    def old_factory(_settings: Settings) -> MockMediaAdapter:
        return MockMediaAdapter()

    old_factory.lm_atelier_contract_version = 0  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "local_lm.adapters.discovery.entry_points",
        lambda *, group: [FakeEntryPoint("old-media", old_factory)] if "media" in group else [],
    )
    with pytest.raises(AdapterLoadError, match="requires 1"):
        load_external_adapter("media", "old-media", Settings(data_dir=tmp_path))
