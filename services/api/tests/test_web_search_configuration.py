"""Configuration identity remains stable without exposing the credential."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from local_lm.config import Settings


def _settings(tmp_path: Path, **values: object) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, **values)


def test_absent_provider_configuration_does_not_create_a_default_destination(
    tmp_path: Path,
) -> None:
    configuration = importlib.import_module("local_lm.web_search_configuration")
    assert configuration.configured_search_provider(_settings(tmp_path)) is None


def test_configured_provider_does_not_enable_search_or_expose_its_token(tmp_path: Path) -> None:
    configuration = importlib.import_module("local_lm.web_search_configuration")
    settings = _settings(
        tmp_path,
        crw_endpoint="https://search.example.test",
        crw_token="constructed-crw-token",
    )
    provider = configuration.configured_search_provider(settings)
    assert provider is not None
    assert provider.endpoint == "https://search.example.test"
    assert provider.token == "constructed-crw-token"
    assert settings.web_access_enabled is False
    assert "constructed-crw-token" not in repr(settings)
    assert "constructed-crw-token" not in repr(provider)
    assert "crw_token" not in settings.model_dump()


@pytest.mark.parametrize("change", ["endpoint", "token"])
def test_provider_identity_survives_restart_but_changes_with_destination_or_account(
    tmp_path: Path,
    change: str,
) -> None:
    configuration = importlib.import_module("local_lm.web_search_configuration")
    values = {"crw_endpoint": "https://search.example.test", "crw_token": "constructed-first-token"}
    original = configuration.configured_search_provider(_settings(tmp_path, **values))
    restarted = configuration.configured_search_provider(_settings(tmp_path, **values))
    assert original is not None and restarted is not None
    identity = configuration.search_provider_revision(original)
    assert identity == configuration.search_provider_revision(restarted)
    if change == "endpoint":
        values["crw_endpoint"] = "https://other.example.test"
    else:
        values["crw_token"] = "constructed-second-token"
    replacement = configuration.configured_search_provider(_settings(tmp_path, **values))
    assert replacement is not None
    assert configuration.search_provider_revision(replacement) != identity
    assert len(identity) == 64 and "constructed-first-token" not in identity
