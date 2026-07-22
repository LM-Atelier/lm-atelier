from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Literal

from ..config import Settings

ADAPTER_CONTRACT_VERSION = 1
CHAT_ADAPTER_GROUP = "lm_atelier.chat_adapters"
MEDIA_ADAPTER_GROUP = "lm_atelier.media_adapters"

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

    group = CHAT_ADAPTER_GROUP if kind == "chat" else MEDIA_ADAPTER_GROUP
    matches = [item for item in entry_points(group=group) if item.name == name]
    if not matches:
        raise AdapterLoadError(
            f"unknown {kind} engine {name!r}; install a {group!r} entry point or select a built-in"
        )
    if len(matches) > 1:
        raise AdapterLoadError(f"multiple installed packages provide {group}:{name}")

    factory = matches[0].load()
    version = getattr(factory, "lm_atelier_contract_version", None)
    if version != ADAPTER_CONTRACT_VERSION:
        raise AdapterLoadError(
            f"adapter {group}:{name} declares contract {version!r}; "
            f"LM Atelier requires {ADAPTER_CONTRACT_VERSION}"
        )
    adapter = factory(settings)
    missing = [
        method for method in _REQUIRED_METHODS[kind] if not callable(getattr(adapter, method, None))
    ]
    if missing:
        raise AdapterLoadError(f"adapter {group}:{name} is missing: {', '.join(missing)}")
    return adapter
