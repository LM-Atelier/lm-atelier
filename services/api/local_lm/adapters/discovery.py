from __future__ import annotations

import re
from importlib.metadata import entry_points
from typing import Any, Literal

from ..config import Settings
from .contracts import (
    ADAPTER_CONTRACT_VERSION,
    GuardedChatAdapter,
    GuardedMediaAdapter,
)

CHAT_ADAPTER_GROUP = "lm_atelier.chat_adapters"
MEDIA_ADAPTER_GROUP = "lm_atelier.media_adapters"
_ADAPTER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")

_REQUIRED_METHODS = {
    "chat": ("capabilities", "count_tokens", "stream", "cancel", "close"),
    "media": ("capabilities", "validate_workflow", "generate", "cancel", "close"),
}


class AdapterLoadError(ValueError):
    pass


def load_external_adapter(kind: Literal["chat", "media"], name: str, settings: Settings) -> Any:
    """Load an explicitly selected, installed adapter factory.

    Loading an entry point executes third-party Python code. Callers only reach
    this function after the user selects its exact entry-point name in settings.
    """

    if not _ADAPTER_NAME.fullmatch(name):
        raise AdapterLoadError(
            "adapter names must contain 1-64 letters, numbers, dots, dashes, or underscores"
        )
    group = CHAT_ADAPTER_GROUP if kind == "chat" else MEDIA_ADAPTER_GROUP
    matches = [item for item in entry_points(group=group) if item.name == name]
    if not matches:
        raise AdapterLoadError(
            f"unknown {kind} engine {name!r}; install a {group!r} entry point or select a built-in"
        )
    if len(matches) > 1:
        raise AdapterLoadError(f"multiple installed packages provide {group}:{name}")

    try:
        factory = matches[0].load()
    except Exception:
        raise AdapterLoadError(f"could not load selected adapter {group}:{name}") from None
    version = getattr(factory, "lm_atelier_contract_version", None)
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or (version != ADAPTER_CONTRACT_VERSION)
    ):
        raise AdapterLoadError(
            f"adapter {group}:{name} declares contract {version!r}; "
            f"LM Atelier requires {ADAPTER_CONTRACT_VERSION}"
        )
    if not callable(factory):
        raise AdapterLoadError(f"adapter {group}:{name} does not expose a callable factory")
    # Adapters are explicitly trusted in-process Python extensions, but their
    # engine contract does not require access to download credentials.
    adapter_settings = settings.model_copy(update={"hf_token": None})
    try:
        adapter = factory(adapter_settings)
    except Exception:
        raise AdapterLoadError(f"could not initialize selected adapter {group}:{name}") from None
    missing = [
        method for method in _REQUIRED_METHODS[kind] if not callable(getattr(adapter, method, None))
    ]
    if missing:
        raise AdapterLoadError(f"adapter {group}:{name} is missing: {', '.join(missing)}")
    label = f"{group}:{name}"
    if kind == "chat":
        return GuardedChatAdapter(
            adapter,
            label,
            inactivity_seconds=settings.llama_inactivity_seconds,
        )
    return GuardedMediaAdapter(
        adapter,
        label,
        settings.max_generated_output_bytes,
        inactivity_seconds=settings.comfy_inactivity_seconds,
    )
