"""The configured provider's identity, without granting permission to search."""

from __future__ import annotations

import hashlib
import json

from .config import Settings
from .web_search import CrwSearchProvider


def configured_search_provider(settings: Settings) -> CrwSearchProvider | None:
    if not settings.crw_endpoint:
        return None
    return CrwSearchProvider(settings.crw_endpoint, settings.crw_token)


def search_provider_revision(provider: CrwSearchProvider) -> str:
    """Bind consent to the same destination and account across a restart.

    This opaque digest is a configuration identity, never a credential or an
    authentication decision. A token replacement invalidates older approvals.
    """
    encoded = json.dumps(
        ["crw-search-provider-v1", provider.endpoint, provider.token],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
