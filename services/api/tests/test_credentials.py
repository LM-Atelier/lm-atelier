from __future__ import annotations

from typing import Any

import pytest
from httpx2 import AsyncClient

import local_lm.credentials as credentials_module
from local_lm.credentials import CredentialStore, CredentialVaultUnavailable


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


def test_environment_token_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials_module, "keyring", FakeKeyring())
    store = CredentialStore("hf_environment")

    assert store.token() == "hf_environment"
    assert store.state().source == "environment"
    with pytest.raises(ValueError, match="unset LOCAL_LM_HF_TOKEN"):
        store.set_token("hf_other")
    with pytest.raises(ValueError, match="unset LOCAL_LM_HF_TOKEN"):
        store.delete_token()


def test_unavailable_vault_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials_module, "keyring", None)
    store = CredentialStore()

    assert store.state().vault_available is False
    with pytest.raises(CredentialVaultUnavailable):
        store.set_token("hf_secret")


async def test_huggingface_credential_api_updates_runtime_clients(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    values: dict[str, str] = {}

    def state(self: CredentialStore) -> Any:
        return credentials_module.CredentialState(
            configured="token" in values,
            source="credential_vault" if "token" in values else "none",
            vault_available=True,
        )

    def token(self: CredentialStore) -> str | None:
        return values.get("token")

    def set_token(self: CredentialStore, value: str) -> None:
        values["token"] = value.strip()

    def delete_token(self: CredentialStore) -> None:
        values.pop("token", None)

    monkeypatch.setattr(CredentialStore, "state", state)
    monkeypatch.setattr(CredentialStore, "token", token)
    monkeypatch.setattr(CredentialStore, "set_token", set_token)
    monkeypatch.setattr(CredentialStore, "delete_token", delete_token)

    status = await client.get("/api/credentials/huggingface")
    assert status.json() == {
        "provider": "huggingface",
        "configured": False,
        "source": "none",
        "vault_available": True,
    }

    saved = await client.put("/api/credentials/huggingface", json={"token": "hf_runtime"})
    assert saved.status_code == 200
    assert saved.json()["configured"] is True
    assert "hf_runtime" not in saved.text

    diagnostics = await client.post("/api/diagnostics")
    archive = await client.get(diagnostics.json()["url"])
    assert b"hf_runtime" not in archive.content

    removed = await client.delete("/api/credentials/huggingface")
    assert removed.status_code == 200
    assert removed.json()["configured"] is False
