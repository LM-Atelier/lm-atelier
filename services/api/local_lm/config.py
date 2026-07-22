from __future__ import annotations

import ipaddress
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOCAL_LM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("data"))
    web_dist_dir: Path | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=12340, ge=1024, le=65535)
    dev: bool = False
    allow_lan: bool = False
    chat_engine: str = "mock"
    media_engine: str = "mock"
    llama_url: str = "http://127.0.0.1:12341"
    comfy_url: str = "http://127.0.0.1:8188"
    llama_executable: Path | None = None
    comfy_executable: Path | None = None
    comfy_directory: Path | None = None
    worker_startup_seconds: float = Field(default=60, ge=1, le=600)
    worker_shutdown_seconds: float = Field(default=10, ge=1, le=60)
    auto_unload_chat_for_media: bool = True
    hf_token: str | None = None
    max_upload_bytes: int = 100 * 1024 * 1024
    max_concurrent_downloads: int = Field(default=2, ge=1, le=8)
    event_history_size: int = Field(default=2_000, ge=100, le=50_000)

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        if value == "localhost":
            return value
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ValueError("host must be localhost or an IP address") from exc
        return str(address)

    def prepare(self) -> None:
        if not self.allow_lan and self.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("non-loopback binding requires LOCAL_LM_ALLOW_LAN=true")
        for path in (
            self.data_dir,
            self.state_dir,
            self.artifact_dir,
            self.download_dir,
            self.model_dir,
            self.workflow_dir,
            self.export_dir,
            self.backup_dir,
            self.catalog_cache_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts" / "sha256"

    @property
    def download_dir(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def workflow_dir(self) -> Path:
        return self.data_dir / "workflows"

    @property
    def export_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{(self.state_dir / 'local-lm.sqlite3').resolve()}"

    @property
    def session_secret_path(self) -> Path:
        return self.state_dir / "session-secret"

    @property
    def backup_dir(self) -> Path:
        return self.data_dir / "backups"

    @property
    def catalog_cache_dir(self) -> Path:
        return self.data_dir / "catalog-cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare()
    return settings
