from __future__ import annotations

from typing import Any

import pytest
from httpx2 import AsyncClient

import local_lm.credentials as credentials_module
from local_lm.credentials import (
    CredentialProvider,
    CredentialStore,
    CredentialVaultUnavailable,
)


class FakeBackend:
    priority = 1


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_keyring(self) -> FakeBackend:
        return FakeBackend()

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        self.values.pop((service, account), None)


def test_credential_store_uses_vault_without_echoing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = FakeKeyring()
    monkeypatch.setattr(credentials_module, "keyring", vault)
    store = CredentialStore()

    assert store.state().source == "none"
    store.set_token("  hf_example_secret  ")
    assert store.token() == "hf_example_secret"
    assert store.state().source == "credential_vault"
    assert "hf_example_secret" not in repr(store.state())
    store.delete_token()
    assert store.token() is None


def test_environment_tokens_are_provider_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(credentials_module, "keyring", FakeKeyring())
    store = CredentialStore(
        "hf_environment",
        environment_tokens={"civitai": "civitai_environment"},
    )

    assert store.token() == "hf_environment"
    assert store.token("civitai") == "civitai_environment"
    assert store.state().source == "environment"
    assert store.state("civitai").source == "environment"
    with pytest.raises(ValueError, match="unset LOCAL_LM_HF_TOKEN"):
        store.set_token("hf_other")
    with pytest.raises(ValueError, match="unset LOCAL_LM_CIVITAI_TOKEN"):
        store.delete_token("civitai")


def test_provider_credentials_use_separate_vault_accounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = FakeKeyring()
    monkeypatch.setattr(credentials_module, "keyring", vault)
    store = CredentialStore()

    store.set_token("hf_secret", "huggingface")
    store.set_token("civitai_secret", "civitai")

    assert store.token("huggingface") == "hf_secret"
    assert store.token("civitai") == "civitai_secret"
    assert len(vault.values) == 2
    store.delete_token("civitai")
    assert store.token("huggingface") == "hf_secret"
    assert store.token("civitai") is None


def test_unavailable_vault_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials_module, "keyring", None)
    store = CredentialStore()

    assert store.state().vault_available is False
    with pytest.raises(CredentialVaultUnavailable):
        store.set_token("hf_secret")


def test_vault_availability_does_not_read_a_stored_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = FakeKeyring()
    monkeypatch.setattr(credentials_module, "keyring", vault)
    monkeypatch.setattr(
        vault,
        "get_password",
        lambda *_args: pytest.fail("availability probe read a credential"),
    )

    assert CredentialStore().vault_available() is True


async def test_credential_api_separates_providers_without_echoing_secrets(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    values: dict[CredentialProvider, str] = {}

    def state(self: CredentialStore, provider: CredentialProvider = "huggingface") -> Any:
        return credentials_module.CredentialState(
            configured=provider in values,
            source="credential_vault" if provider in values else "none",
            vault_available=True,
        )

    def token(self: CredentialStore, provider: CredentialProvider = "huggingface") -> str | None:
        return values.get(provider)

    def set_token(
        self: CredentialStore,
        value: str,
        provider: CredentialProvider = "huggingface",
    ) -> None:
        values[provider] = value.strip()

    def delete_token(self: CredentialStore, provider: CredentialProvider = "huggingface") -> None:
        values.pop(provider, None)

    monkeypatch.setattr(CredentialStore, "state", state)
    monkeypatch.setattr(CredentialStore, "token", token)
    monkeypatch.setattr(CredentialStore, "set_token", set_token)
    monkeypatch.setattr(CredentialStore, "delete_token", delete_token)

    for provider in ("huggingface", "civitai"):
        status = await client.get(f"/api/credentials/{provider}")
        assert status.json() == {
            "provider": provider,
            "configured": False,
            "source": "none",
            "vault_available": True,
        }

    hf_saved = await client.put("/api/credentials/huggingface", json={"token": "hf_runtime"})
    civitai_saved = await client.put("/api/credentials/civitai", json={"token": "civitai_runtime"})
    assert hf_saved.status_code == civitai_saved.status_code == 200
    assert hf_saved.json()["configured"] is True
    assert civitai_saved.json()["configured"] is True
    assert "hf_runtime" not in hf_saved.text
    assert "civitai_runtime" not in civitai_saved.text
    assert values == {
        "huggingface": "hf_runtime",
        "civitai": "civitai_runtime",
    }

    diagnostics = await client.post("/api/diagnostics")
    archive = await client.get(diagnostics.json()["url"])
    assert b"hf_runtime" not in archive.content
    assert b"civitai_runtime" not in archive.content

    removed = await client.delete("/api/credentials/civitai")
    assert removed.status_code == 200
    assert removed.json()["configured"] is False
    assert values == {"huggingface": "hf_runtime"}

    invalid = await client.get("/api/credentials/unknown")
    assert invalid.status_code == 404
