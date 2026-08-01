from __future__ import annotations

import os
import ssl
from functools import lru_cache

import httpx


@lru_cache(maxsize=8)
def _tls_context_for_environment(
    trust_environment: bool,
    certificate_file: str | None,
    certificate_directory: str | None,
) -> ssl.SSLContext:
    # httpx honors these environment variables while constructing its default
    # context. Including them in the key prevents a later configuration change
    # from silently reusing the wrong trust roots.
    _ = certificate_file, certificate_directory
    return httpx.create_ssl_context(trust_env=trust_environment)


def shared_tls_context(*, trust_environment: bool = True) -> ssl.SSLContext:
    """Reuse immutable trust roots across independent outbound client pools."""

    return _tls_context_for_environment(
        trust_environment,
        os.environ.get("SSL_CERT_FILE") if trust_environment else None,
        os.environ.get("SSL_CERT_DIR") if trust_environment else None,
    )
