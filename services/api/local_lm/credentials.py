from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import keyring
from keyring.errors import KeyringError


class CredentialVaultUnavailable(RuntimeError):
    pass


CredentialProvider = Literal["huggingface", "civitai"]
CREDENTIAL_PROVIDERS: tuple[CredentialProvider, ...] = ("huggingface", "civitai")


@dataclass(frozen=True)
class CredentialState:
    configured: bool
    source: Literal["none", "environment", "credential_vault"]
    vault_available: bool


class CredentialStore:
    SERVICE = "lm-atelier"
    ACCOUNTS: Mapping[CredentialProvider, str] = {
        "huggingface": "huggingface-token",
        "civitai": "civitai-token",
    }
    ENVIRONMENT_VARIABLES: Mapping[CredentialProvider, str] = {
        "huggingface": "LOCAL_LM_HF_TOKEN",
        "civitai": "LOCAL_LM_CIVITAI_TOKEN",
    }

    def __init__(
        self,
        environment_token: str | None = None,
        *,
        environment_tokens: Mapping[CredentialProvider, str | None] | None = None,
    ) -> None:
        configured = dict(environment_tokens or {})
        if environment_token:
            configured["huggingface"] = environment_token
        self.environment_tokens = {
            provider: configured.get(provider) or None for provider in CREDENTIAL_PROVIDERS
        }

    def token(self, provider: CredentialProvider = "huggingface") -> str | None:
        environment_token = self.environment_tokens[provider]
        if environment_token:
            return environment_token
        if keyring is None:
            return None
        try:
            return keyring.get_password(self.SERVICE, self.ACCOUNTS[provider]) or None
        except KeyringError:
            return None

    def state(self, provider: CredentialProvider = "huggingface") -> CredentialState:
        if self.environment_tokens[provider]:
            return CredentialState(
                configured=True,
                source="environment",
                vault_available=self.vault_available(),
            )
        value = self.token(provider)
        return CredentialState(
            configured=bool(value),
            source="credential_vault" if value else "none",
            vault_available=self.vault_available(),
        )

    def vault_available(self) -> bool:
        """Report whether an OS credential backend is usable without reading a secret."""
        return self._available()

    def set_token(self, token: str, provider: CredentialProvider = "huggingface") -> None:
        if self.environment_tokens[provider]:
            variable = self.ENVIRONMENT_VARIABLES[provider]
            raise ValueError(f"unset {variable} before managing the token in the interface")
        normalized = token.strip()
        if not normalized:
            raise ValueError("token cannot be empty")
        if keyring is None:
            raise CredentialVaultUnavailable("the operating-system credential vault is unavailable")
        try:
            keyring.set_password(self.SERVICE, self.ACCOUNTS[provider], normalized)
        except KeyringError as exc:
            raise CredentialVaultUnavailable(
                "the operating-system credential vault rejected the token"
            ) from exc

    def delete_token(self, provider: CredentialProvider = "huggingface") -> None:
        if self.environment_tokens[provider]:
            variable = self.ENVIRONMENT_VARIABLES[provider]
            raise ValueError(f"unset {variable} to remove the environment token")
        if keyring is None:
            raise CredentialVaultUnavailable("the operating-system credential vault is unavailable")
        try:
            keyring.delete_password(self.SERVICE, self.ACCOUNTS[provider])
        except KeyringError as exc:
            # Deleting an absent token is intentionally idempotent. Backends use
            # different exception types, so confirm absence before surfacing it.
            try:
                if keyring.get_password(self.SERVICE, self.ACCOUNTS[provider]) is None:
                    return
            except KeyringError:
                pass
            raise CredentialVaultUnavailable(
                "the operating-system credential vault could not remove the token"
            ) from exc

    @staticmethod
    def _available() -> bool:
        if keyring is None:
            return False
        try:
            backend = keyring.get_keyring()
            return float(getattr(backend, "priority", 0)) > 0
        except (KeyringError, TypeError, ValueError):
            return False


def credential_provider(value: str) -> CredentialProvider:
    if value not in CREDENTIAL_PROVIDERS:
        raise ValueError("credential provider must be huggingface or civitai")
    return value
