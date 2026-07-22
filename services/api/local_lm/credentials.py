from __future__ import annotations

from dataclasses import dataclass

import keyring
from keyring.errors import KeyringError


class CredentialVaultUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CredentialState:
    configured: bool
    source: str
    vault_available: bool


class CredentialStore:
    SERVICE = "lm-atelier"
    HUGGINGFACE_ACCOUNT = "huggingface-token"

    def __init__(self, environment_token: str | None = None) -> None:
        self.environment_token = environment_token or None

    def token(self) -> str | None:
        if self.environment_token:
            return self.environment_token
        if keyring is None:
            return None
        try:
            return keyring.get_password(self.SERVICE, self.HUGGINGFACE_ACCOUNT) or None
        except KeyringError:
            return None

    def state(self) -> CredentialState:
        if self.environment_token:
            return CredentialState(
                configured=True, source="environment", vault_available=self._available()
            )
        value = self.token()
        return CredentialState(
            configured=bool(value),
            source="credential_vault" if value else "none",
            vault_available=self._available(),
        )

    def set_token(self, token: str) -> None:
        if self.environment_token:
            raise ValueError("unset LOCAL_LM_HF_TOKEN before managing the token in the interface")
        normalized = token.strip()
        if not normalized:
            raise ValueError("token cannot be empty")
        if keyring is None:
            raise CredentialVaultUnavailable("the operating-system credential vault is unavailable")
        try:
            keyring.set_password(self.SERVICE, self.HUGGINGFACE_ACCOUNT, normalized)
        except KeyringError as exc:
            raise CredentialVaultUnavailable(
                "the operating-system credential vault rejected the token"
            ) from exc

    def delete_token(self) -> None:
        if self.environment_token:
            raise ValueError("unset LOCAL_LM_HF_TOKEN to remove the environment token")
        if keyring is None:
            raise CredentialVaultUnavailable("the operating-system credential vault is unavailable")
        try:
            keyring.delete_password(self.SERVICE, self.HUGGINGFACE_ACCOUNT)
        except KeyringError as exc:
            # Deleting an absent token is intentionally idempotent. Backends use
            # different exception types, so confirm absence before surfacing it.
            try:
                if keyring.get_password(self.SERVICE, self.HUGGINGFACE_ACCOUNT) is None:
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
