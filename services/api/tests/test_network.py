from __future__ import annotations

import ssl
from unittest.mock import Mock, call

import httpx

from local_lm import network


def test_tls_context_is_reused_until_trust_environment_changes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    first = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    second = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    isolated = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    factory = Mock(side_effect=[first, second, isolated])
    monkeypatch.setattr(httpx, "create_ssl_context", factory)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    network._tls_context_for_environment.cache_clear()

    try:
        assert network.shared_tls_context() is first
        assert network.shared_tls_context() is first

        monkeypatch.setenv("SSL_CERT_FILE", "alternate-trust-roots.pem")
        assert network.shared_tls_context() is second
        assert network.shared_tls_context(trust_environment=False) is isolated
        monkeypatch.setenv("SSL_CERT_FILE", "another-trust-root.pem")
        assert network.shared_tls_context(trust_environment=False) is isolated
        assert factory.call_args_list == [
            call(trust_env=True),
            call(trust_env=True),
            call(trust_env=False),
        ]
    finally:
        network._tls_context_for_environment.cache_clear()
