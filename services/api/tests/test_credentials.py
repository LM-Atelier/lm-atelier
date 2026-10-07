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


def test_search_credentials_are_isolated_between_installation_namespaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = FakeKeyring()
    monkeypatch.setattr(credentials_module, "keyring", vault)
    first = CredentialStore(service="installation-alpha")
    second = CredentialStore(service="installation-beta")
    first.set_token("constructed-alpha-token", "crw")
    second.set_token("constructed-beta-token", "crw")
    assert first.token("crw") == "constructed-alpha-token"
    assert second.token("crw") == "constructed-beta-token"
    assert first.token("huggingface") is None and first.token("civitai") is None
    first.delete_token("crw")
    assert first.token("crw") is None
    assert second.token("crw") == "constructed-beta-token"
    assert ("installation-beta", "crw-token") in vault.values


def test_search_environment_credentials_cannot_be_changed_through_the_vault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = FakeKeyring()
    monkeypatch.setattr(credentials_module, "keyring", vault)
    store = CredentialStore(environment_tokens={"crw": "constructed-environment-token"})
    assert store.token("crw") == "constructed-environment-token"
    with pytest.raises(ValueError, match="LOCAL_LM_CRW_TOKEN"):
        store.set_token("constructed-replacement", "crw")
    with pytest.raises(ValueError, match="LOCAL_LM_CRW_TOKEN"):
        store.delete_token("crw")
    assert not vault.values


@pytest.mark.parametrize(
    "token",
    ["x" * 4097, "two words", "two\nlines", "non-ascii-\u00e9"],
    ids=["too-long", "space", "newline", "non-ascii"],
)
def test_invalid_search_token_is_refused_without_writing_the_vault(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    vault = FakeKeyring()
    monkeypatch.setattr(credentials_module, "keyring", vault)
    store = CredentialStore()
    with pytest.raises(ValueError, match="search credential is invalid"):
        store.set_token(token, "crw")
    assert not vault.values


async def test_search_credential_api_does_not_change_catalog_credentials_or_diagnostics(
    client: AsyncClient,
    app: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import zipfile

    vault = FakeKeyring()
    monkeypatch.setattr(credentials_module, "keyring", vault)
    services = app.state.services
    services.settings.hf_token = "constructed-hf-runtime"
    services.settings.civitai_token = "constructed-civitai-runtime"
    services.settings.web_access_enabled = False
    saved = await client.put("/api/credentials/crw", json={"token": "constructed-api-search-token"})
    assert saved.status_code == 200
    assert services.settings.crw_token == "constructed-api-search-token"
    assert services.settings.hf_token == "constructed-hf-runtime"
    assert services.settings.civitai_token == "constructed-civitai-runtime"
    assert services.settings.web_access_enabled is False
    assert saved.json() == {
        "provider": "crw",
        "configured": True,
        "source": "credential_vault",
        "vault_available": True,
    }
    assert "constructed-api-search-token" not in saved.text
    for provider in ("huggingface", "civitai"):
        status = await client.get(f"/api/credentials/{provider}")
        assert status.status_code == 200 and status.json()["configured"] is False
    diagnostic = await client.post("/api/diagnostics")
    assert diagnostic.status_code == 201
    archive = await client.get(diagnostic.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        for name in bundle.namelist():
            assert b"constructed-api-search-token" not in bundle.read(name)
    removed = await client.delete("/api/credentials/crw")
    assert removed.status_code == 200 and removed.json()["configured"] is False
    assert services.settings.crw_token is None
