from __future__ import annotations

import ipaddress
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

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
    chat_engine: str = "mock"
    media_engine: str = "mock"
    llama_url: str = "http://127.0.0.1:12341"
    comfy_url: str = "http://127.0.0.1:8188"
    llama_executable: Path | None = None
    vllm_executable: Path | None = None
    comfy_executable: Path | None = None
    comfy_directory: Path | None = None
    llama_inactivity_seconds: float = Field(default=600, ge=30, le=7200)
    comfy_inactivity_seconds: float = Field(default=600, ge=30, le=7200)
    worker_startup_seconds: float = Field(default=60, ge=1, le=600)
    worker_shutdown_seconds: float = Field(default=10, ge=1, le=60)
    auto_unload_chat_for_media: bool = True
    hf_token: str | None = None
    civitai_token: str | None = None
    max_upload_bytes: int = 100 * 1024 * 1024
    vision_max_images: int = Field(default=4, ge=1, le=16)
    vision_max_image_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=1024,
        le=100 * 1024**2,
    )
    vision_max_total_bytes: int = Field(
        default=32 * 1024 * 1024,
        ge=1024,
        le=256 * 1024**2,
    )
    vision_max_pixels: int = Field(default=40_000_000, ge=1_000_000, le=200_000_000)
    vision_max_video_frames: int = Field(default=6, ge=3, le=16)
    vision_max_video_duration_seconds: int = Field(default=3600, ge=1, le=86_400)
    vision_max_frame_dimension: int = Field(default=1280, ge=256, le=4096)
    vision_max_frame_bytes: int = Field(default=5 * 1024 * 1024, ge=64 * 1024, le=20 * 1024**2)
    vision_sampler_timeout_seconds: int = Field(default=60, ge=5, le=300)
    vision_bridge_max_tokens: int = Field(default=512, ge=64, le=2048)
    max_generated_output_bytes: int = Field(
        default=512 * 1024**2,
        ge=1024**2,
        le=16 * 1024**3,
    )
    max_media_outputs_per_plan: int = Field(default=8, ge=1, le=16)
    max_media_plan_work_units: int = Field(default=4_000_000_000, ge=1)
    max_media_plan_duration_seconds: int = Field(default=300, ge=1, le=3600)
    max_media_plan_estimated_bytes: int = Field(
        default=8 * 1024**3,
        ge=1024**2,
        le=128 * 1024**3,
    )
    max_project_import_bytes: int = Field(default=2 * 1024**3, ge=1024**2)
    max_project_archive_entries: int = Field(default=20_000, ge=10, le=100_000)
    artifact_retention_days: int = Field(default=30, ge=1, le=3650)
    temporary_retention_hours: int = Field(default=24, ge=1, le=168)
    storage_warning_free_bytes: int = Field(default=10 * 1024**3, ge=0)
    video_confirmation_work_units: int = Field(default=500_000_000, ge=0)
    backup_daily_count: int = Field(default=7, ge=1, le=90)
    backup_weekly_count: int = Field(default=4, ge=0, le=52)
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

    @field_validator("llama_url", "comfy_url")
    @classmethod
    def validate_worker_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("worker URLs must be plain loopback HTTP origins")
        try:
            loopback = (
                parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
            )
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("worker URLs must use a loopback host")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("worker URLs must use a valid port") from exc
        return value.rstrip("/")

    def prepare(self) -> None:
        if self.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("the public preview supports loopback binding only")
        for path in (
            self.data_dir,
            self.state_dir,
            self.artifact_dir,
            self.download_dir,
            self.model_dir,
            self.workflow_dir,
            self.custom_node_dir,
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
    def comfy_output_dir(self) -> Path:
        return self.state_dir / "comfy-output"

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
    def custom_node_dir(self) -> Path:
        if self.comfy_directory:
            return self.comfy_directory / "custom_nodes"
        return self.data_dir / "custom-nodes"

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
