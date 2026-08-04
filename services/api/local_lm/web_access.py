"""Whether a conversation may reach the internet, and who decided.

Two gates that both have to be open. The installation-wide one is a
deployment fact - somewhere that must never egress pins it shut and nothing
below can argue. The per-chat one is the person's, made deliberately, and it
is never inherited: a new chat starts closed however the last one was set,
because permission that spreads by default is permission nobody granted.
"""

from __future__ import annotations

from typing import Any

FETCH_KEY = "allow_url_fetch"
SEARCH_KEY = "allow_search"
UNATTENDED_SEARCH_KEY = "allow_search_without_asking"


def may_fetch_urls(*, installation_enabled: bool, chat_settings: dict[str, Any] | None) -> bool:
    """Whether this chat may retrieve an address the person pasted."""
    return installation_enabled and _allowed(chat_settings, FETCH_KEY)


def may_search(*, installation_enabled: bool, chat_settings: dict[str, Any] | None) -> bool:
    """Whether this chat may send a composed query to a search provider."""
    return installation_enabled and _allowed(chat_settings, SEARCH_KEY)


def must_confirm_each_query(chat_settings: dict[str, Any] | None) -> bool:
    """Whether each outbound query is shown and confirmed before it leaves.

    Confirming is the default. Turning it off is an explicit per-chat choice
    and still leaves the query visible and cancellable while it is being
    dispatched - what it removes is the wait, not the disclosure.
    """
    return not _allowed(chat_settings, UNATTENDED_SEARCH_KEY)


def _allowed(chat_settings: dict[str, Any] | None, key: str) -> bool:
    if not isinstance(chat_settings, dict):
        return False
    # Only an explicit true opens a gate. A truthy string or a stray number
    # is a malformed setting, and a malformed permission is no permission.
    return chat_settings.get(key) is True
