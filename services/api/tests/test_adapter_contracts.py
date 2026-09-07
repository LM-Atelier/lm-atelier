from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import local_lm.adapters.contracts as contracts
from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.adapters.conformance import probe_chat_adapter, probe_media_adapter
from local_lm.adapters.contracts import (
    AdapterContractError,
    GuardedChatAdapter,
    capability_contract_errors,
)
from local_lm.adapters.discovery import AdapterLoadError, load_external_adapter
from local_lm.adapters.llama_cpp import LlamaCppAdapter
from local_lm.adapters.mock import MockChatAdapter, MockMediaAdapter
from local_lm.adapters.vllm import VllmAdapter
from local_lm.config import Settings
from local_lm.engines import EngineRegistry
from local_lm.schemas import EngineCapabilities, SettingField
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
    assert capabilities.input_modalities == ["text", "image"]


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


def test_capabilities_reject_invalid_or_missing_input_modalities() -> None:
    base = EngineCapabilities(
        engine="vision-chat",
        version="1",
        roles=["chat"],
        operations=["text"],
        formats=["gguf"],
        devices=["gpu:0"],
        streaming=True,
        tool_calling=True,
        settings=[],
        healthy=True,
    )

    assert base.input_modalities == ["text"]
    assert not capability_contract_errors(base, "chat")
    assert not capability_contract_errors(
        base.model_copy(update={"input_modalities": ["text", "image"]}),
        "chat",
    )
    assert "capabilities must not duplicate input modalities" in capability_contract_errors(
        base.model_copy(update={"input_modalities": ["text", "text"]}),
        "chat",
    )
    assert "capabilities include unsupported input modalities" in capability_contract_errors(
        base.model_copy(update={"input_modalities": ["text", "filesystem"]}),
        "chat",
    )
    assert "capabilities include unsupported input modalities" in capability_contract_errors(
        base.model_copy(update={"input_modalities": ["text", "video"]}),
        "chat",
    )
    assert "capabilities must include the text input modality" in capability_contract_errors(
        base.model_copy(update={"input_modalities": ["image"]}),
        "chat",
    )


async def test_comfyui_inactivity_timeout_is_validated_and_wired(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, comfy_inactivity_seconds=29)

    registry = EngineRegistry(
        Settings(
            data_dir=tmp_path,
            media_engine="comfyui",
            comfy_inactivity_seconds=321,
        )
    )
    try:
        assert isinstance(registry.media, ComfyUIAdapter)
        assert registry.media.inactivity_seconds == 321
    finally:
        await registry.close()


async def test_llama_inactivity_timeout_is_validated_and_wired(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, llama_inactivity_seconds=29)

    registry = EngineRegistry(
        Settings(
            data_dir=tmp_path,
            chat_engine="llama.cpp",
            llama_inactivity_seconds=321,
        )
    )
    try:
        assert registry.chat.inactivity_seconds == 321  # type: ignore[attr-defined]
    finally:
        await registry.close()


async def test_managed_chat_registry_switches_adapter_with_active_runtime(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path, chat_engine="llama.cpp")
    registry = EngineRegistry(settings)
    try:
        assert isinstance(registry.chat, LlamaCppAdapter)
        assert not isinstance(registry.chat, VllmAdapter)
        settings.chat_engine = "vllm"
        assert isinstance(registry.chat, VllmAdapter)
        vllm_settings = await registry.settings_for_role(
            "chat",
            engine="vllm",
            allow_inactive=True,
        )
        assert vllm_settings
    finally:
        await registry.close()


async def test_loopback_adapter_clients_ignore_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    llama = LlamaCppAdapter("http://127.0.0.1:12341")
    comfy = ComfyUIAdapter("http://127.0.0.1:8188")
    try:
        assert llama._client._trust_env is False  # type: ignore[attr-defined]
        assert comfy._client._trust_env is False  # type: ignore[attr-defined]
    finally:
        await llama.close()
        await comfy.close()


async def test_comfyui_managed_output_cleanup_is_wired_only_for_managed_runtime(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "python"
    directory = tmp_path / "comfyui"
    managed = EngineRegistry(
        Settings(
            data_dir=tmp_path / "managed-data",
            media_engine="comfyui",
            comfy_executable=executable,
            comfy_directory=directory,
        )
    )
    external = EngineRegistry(
        Settings(
            data_dir=tmp_path / "external-data",
            media_engine="comfyui",
            comfy_directory=directory,
        )
    )
    try:
        assert isinstance(managed.media, ComfyUIAdapter)
        assert (
            managed.media.managed_output_root
            == (tmp_path / "managed-data" / "state" / "comfy-output").resolve()
        )
        assert isinstance(external.media, ComfyUIAdapter)
        assert external.media.managed_output_root is None
    finally:
        await managed.close()
        await external.close()


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
    assert isinstance(registry.chat, GuardedChatAdapter)


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


async def test_external_adapter_factory_does_not_receive_download_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    received_tokens = ("not-called", "not-called")

    def factory(settings: Settings) -> MockChatAdapter:
        nonlocal received_tokens
        received_tokens = (settings.hf_token or "", settings.civitai_token or "")
        return MockChatAdapter()

    factory.lm_atelier_contract_version = 1  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "local_lm.adapters.discovery.entry_points",
        lambda *, group: [FakeEntryPoint("credential-safe", factory)] if "chat" in group else [],
    )
    settings = Settings(
        data_dir=tmp_path,
        chat_engine="credential-safe",
        hf_token="hf_private_value",
        civitai_token="civitai_private_value",
    )
    adapter = load_external_adapter("chat", "credential-safe", settings)
    try:
        assert received_tokens == ("", "")
        assert settings.hf_token == "hf_private_value"
        assert settings.civitai_token == "civitai_private_value"
    finally:
        await adapter.close()


def test_external_adapter_load_and_initialization_errors_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "hf_secret_that_must_not_escape"

    @dataclass
    class BrokenEntryPoint:
        name: str

        def load(self) -> Any:
            raise RuntimeError(secret)

    monkeypatch.setattr(
        "local_lm.adapters.discovery.entry_points",
        lambda *, group: [BrokenEntryPoint("broken-load")],
    )
    with pytest.raises(AdapterLoadError) as load_error:
        load_external_adapter("chat", "broken-load", Settings(data_dir=tmp_path))
    assert secret not in str(load_error.value)

    def broken_factory(_settings: Settings) -> MockChatAdapter:
        raise RuntimeError(secret)

    broken_factory.lm_atelier_contract_version = 1  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "local_lm.adapters.discovery.entry_points",
        lambda *, group: [FakeEntryPoint("broken-init", broken_factory)],
    )
    with pytest.raises(AdapterLoadError) as init_error:
        load_external_adapter("chat", "broken-init", Settings(data_dir=tmp_path))
    assert secret not in str(init_error.value)


def test_external_adapter_names_and_contract_versions_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(AdapterLoadError, match="adapter names"):
        load_external_adapter("chat", "bad\nname", Settings(data_dir=tmp_path))

    def factory(_settings: Settings) -> MockChatAdapter:
        return MockChatAdapter()

    factory.lm_atelier_contract_version = True  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "local_lm.adapters.discovery.entry_points",
        lambda *, group: [FakeEntryPoint("boolean-version", factory)],
    )
    with pytest.raises(AdapterLoadError, match="requires 1"):
        load_external_adapter("chat", "boolean-version", Settings(data_dir=tmp_path))


async def test_guarded_chat_adapter_adds_a_redacted_terminal_error() -> None:
    secret = "adapter-private-error"

    class BrokenChatAdapter(MockChatAdapter):
        async def stream(self, _request: ChatRequest):  # type: ignore[no-untyped-def]
            yield ChatEvent(type="delta", text="partial")
            raise RuntimeError(secret)

    adapter = GuardedChatAdapter(BrokenChatAdapter(), "test-adapter")
    events = [
        event
        async for event in adapter.stream(
            ChatRequest(
                run_id="broken-external",
                messages=[{"role": "user", "content": "test"}],
            )
        )
    ]

    assert [event.type for event in events] == ["delta", "error"]
    assert secret not in events[-1].data["error"]


async def test_guarded_chat_adapter_times_out_a_silent_third_party_stream() -> None:
    class SilentChatAdapter(MockChatAdapter):
        async def stream(self, _request: ChatRequest):  # type: ignore[no-untyped-def]
            await asyncio.Event().wait()
            yield ChatEvent(type="complete")

    adapter = GuardedChatAdapter(
        SilentChatAdapter(),
        "test-adapter",
        inactivity_seconds=0.01,
    )
    events = [
        event async for event in adapter.stream(ChatRequest(run_id="silent-external", messages=[]))
    ]

    assert [event.type for event in events] == ["error"]


async def test_guarded_chat_adapter_stops_events_after_terminal() -> None:
    class PostTerminalAdapter(MockChatAdapter):
        async def stream(self, _request: ChatRequest):  # type: ignore[no-untyped-def]
            yield ChatEvent(type="complete")
            yield ChatEvent(type="delta", text="must not escape")

    adapter = GuardedChatAdapter(PostTerminalAdapter(), "test-adapter")
    events = [
        event async for event in adapter.stream(ChatRequest(run_id="post-terminal", messages=[]))
    ]

    assert [event.type for event in events] == ["complete"]


def test_role_aware_capabilities_require_complete_exact_legacy_definitions() -> None:
    missing_video = EngineCapabilities(
        engine="incomplete-media",
        version="1",
        roles=["image", "video"],
        operations=["text_to_image", "text_to_video"],
        formats=["test"],
        devices=[],
        streaming=False,
        tool_calling=False,
        settings=IMAGE_SETTINGS,
        settings_by_role={"image": IMAGE_SETTINGS},
        healthy=True,
    )
    errors = capability_contract_errors(missing_video, "media")
    assert "settings_by_role is missing advertised roles" in errors

    mismatched = IMAGE_SETTINGS[0].model_copy(update={"default": "different"})
    wrong_definition = missing_video.model_copy(
        update={
            "roles": ["image"],
            "operations": ["text_to_image"],
            "settings_by_role": {"image": [mismatched]},
        }
    )
    errors = capability_contract_errors(wrong_definition, "media")
    assert any("missing its exact definition" in error for error in errors)


def test_role_aware_capabilities_reject_ambiguous_and_weakened_fields() -> None:
    first = SettingField(
        key="adapter_mode",
        label="Adapter mode",
        type="string",
        default="request",
        scope="request",
    )
    second = first.model_copy(update={"default": "workflow", "scope": "workflow"})
    weakened_width = next(field for field in IMAGE_SETTINGS if field.key == "width").model_copy(
        update={"maximum": 8192}
    )
    capabilities = EngineCapabilities(
        engine="unsafe-media",
        version="1",
        roles=["image"],
        operations=["text_to_image"],
        formats=["test"],
        devices=[],
        streaming=False,
        tool_calling=False,
        settings=[first, second, weakened_width],
        settings_by_role={"image": [first, second, weakened_width]},
        healthy=True,
    )

    errors = capability_contract_errors(capabilities, "media")

    assert "settings_by_role[image] duplicates key adapter_mode" in errors
    assert "settings for role image conflict with LM Atelier's schema" in errors


async def test_conformance_probe_reports_events_after_terminal() -> None:
    class PostTerminalAdapter(MockChatAdapter):
        async def stream(self, _request: ChatRequest):  # type: ignore[no-untyped-def]
            yield ChatEvent(type="complete")
            yield ChatEvent(type="delta", text="late")

    report = await probe_chat_adapter(PostTerminalAdapter())
    assert not report.passed
    assert "stream emitted events after its terminal event" in report.errors


async def test_conformance_probe_reports_oversized_events() -> None:
    class OversizedEventAdapter(MockChatAdapter):
        async def stream(self, _request: ChatRequest):  # type: ignore[no-untyped-def]
            yield ChatEvent(type="delta", text="x" * (1024 * 1024 + 1))
            yield ChatEvent(type="complete")

    report = await probe_chat_adapter(OversizedEventAdapter())
    assert not report.passed
    assert "chat event text must be UTF-8 and no larger than 1 MiB" in report.errors


async def test_guarded_capability_errors_do_not_echo_third_party_exceptions() -> None:
    secret = "third-party-secret"

    class BrokenCapabilities(MockChatAdapter):
        async def capabilities(self) -> EngineCapabilities:
            raise RuntimeError(secret)

    adapter = GuardedChatAdapter(BrokenCapabilities(), "test-adapter")
    with pytest.raises(AdapterContractError) as captured:
        await adapter.capabilities()
    assert secret not in str(captured.value)


class _CloseThatIgnoresCancellation:
    """A producer whose `aclose` swallows the cancellation aimed at it.

    Not a straw man. Any `aclose` with a `try/finally` that awaits, or an
    `except asyncio.CancelledError` that tidies up before re-raising, can
    outlive the cancellation. The bound has to hold for those too.

    It ignores the bound once and then finishes, rather than looping forever:
    a control that leaves an immortal task hangs the suite at loop teardown
    instead of testing anything. Mine did, before this.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.swallowed_cancellation = False
        self.finished = False

    def __aiter__(self) -> _CloseThatIgnoresCancellation:
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.swallowed_cancellation = True
        self.finished = True


class _CloseThatRaisesCancellation:
    """A producer whose `aclose` raises `CancelledError` of its own accord."""

    def __aiter__(self) -> _CloseThatRaisesCancellation:
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration

    async def aclose(self) -> None:
        raise asyncio.CancelledError


async def test_a_close_that_ignores_cancellation_does_not_hold_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound must not depend on the close agreeing to be bounded.

    `asyncio.timeout` bounds by cancelling the task that awaits it. Awaited in
    the caller's own task, a close that catches that cancellation and keeps
    waiting defeats the bound entirely, and a stale producer holds the
    dispatch for as long as it likes.
    """

    monkeypatch.setattr(contracts, "CLOSE_SECONDS", 0.05)
    producer = _CloseThatIgnoresCancellation()

    await asyncio.wait_for(contracts.close_iterator(producer), timeout=2)

    assert producer.entered.is_set(), "the close was never attempted"
    # The discriminating assertion: the caller is released while the close is
    # still going. On the parent this line is never reached, because the
    # caller is still inside a close that will not stop.
    assert not producer.finished, "the caller waited for the close after all"

    # Let the abandoned close settle, so the control leaves nothing running
    # and its own premise is checked rather than assumed.
    for _ in range(200):
        if producer.finished:
            break
        await asyncio.sleep(0.01)
    assert producer.swallowed_cancellation, "this control never exercised an uncooperative close"


async def test_a_close_that_raises_cancellation_does_not_replace_the_refusal() -> None:
    """A cancellation from the close must not displace the reason for the close.

    `suppress(Exception)` never caught `CancelledError`, which is a
    `BaseException`. A close raising it while a refusal was already in flight
    replaced that refusal, and the caller saw a cancellation instead of the
    lost claim it actually had - the one thing the surrounding design is
    careful never to do.
    """

    class Refusal(RuntimeError):
        pass

    async def stop_consuming_and_close() -> None:
        try:
            raise Refusal("the row is no longer this execution's")
        finally:
            await contracts.close_iterator(_CloseThatRaisesCancellation())

    with pytest.raises(Refusal):
        await stop_consuming_and_close()


async def test_an_ordinary_close_is_still_awaited_and_its_failure_still_dropped() -> None:
    """The half that must not change.

    A close that finishes is still waited for, so a producer that tidies up
    quickly is not abandoned mid-way; and a close that simply fails is still
    not the caller's problem.
    """

    closed: list[str] = []

    class Ordinary:
        def __aiter__(self) -> object:
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            closed.append("done")

    class Failing(Ordinary):
        async def aclose(self) -> None:
            closed.append("tried")
            raise RuntimeError("the close failed")

    await contracts.close_iterator(Ordinary())
    await contracts.close_iterator(Failing())

    assert closed == ["done", "tried"]


async def test_cancelling_the_caller_still_asks_the_close_to_stop() -> None:
    """A cancelled caller must not leave the producer running.

    `asyncio.wait` does not cancel what it waits on when the WAITER is
    cancelled. Awaiting the close in the caller's own task did, so moving it
    to its own task removed that behaviour, and a cancelled generation left a
    producer running with an unretrieved task behind it. The bound and the
    caller's cancellation are two different ways out and both have to abandon
    the close.
    """

    entered = asyncio.Event()
    asked_to_stop = asyncio.Event()

    class SlowClose:
        def __aiter__(self) -> object:
            return self

        async def __anext__(self) -> object:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            entered.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                asked_to_stop.set()
                raise

    producer = SlowClose()
    waiting = asyncio.ensure_future(contracts.close_iterator(producer))
    await asyncio.wait_for(entered.wait(), timeout=2)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    await asyncio.wait_for(asked_to_stop.wait(), timeout=2)
