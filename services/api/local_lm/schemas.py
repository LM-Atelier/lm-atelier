from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .domain import Operation, RoutingMode


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ProjectCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10_000)
    instructions: str = Field(default="", max_length=100_000)


class ProjectUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=10_000)
    instructions: str | None = Field(default=None, max_length=100_000)
    archived: bool | None = None


class ProjectOut(ApiModel):
    id: str
    name: str
    description: str
    instructions: str
    archived: bool
    created_at: datetime
    updated_at: datetime


class ChatCreate(ApiModel):
    title: str = Field(default="New chat", min_length=1, max_length=240)
    project_id: str | None = None
    routing_mode: RoutingMode = RoutingMode.AUTO


class ChatUpdate(ApiModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    project_id: str | None = None
    archived: bool | None = None
    routing_mode: RoutingMode | None = None
    confirm_uncertain_media: bool | None = None
    active_chat_profile_id: str | None = None
    active_image_profile_id: str | None = None
    active_video_profile_id: str | None = None


class ArtifactOut(ApiModel):
    id: str
    sha256: str
    kind: str
    media_type: str
    size_bytes: int
    original_name: str | None
    metadata_json: dict[str, Any]
    created_at: datetime
    url: str | None = None


class ArtifactLibraryItem(ArtifactOut):
    reference_count: int = 0
    chat_ids: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)


class ArtifactStorageInfo(ApiModel):
    total_bytes: int
    total_count: int
    referenced_bytes: int
    referenced_count: int
    unreferenced_bytes: int
    unreferenced_count: int
    temporary_bytes: int
    temporary_count: int
    eligible_bytes: int
    eligible_count: int
    disk_free_bytes: int
    warning: bool
    retention_days: int
    temporary_retention_hours: int


class ArtifactCleanupRequest(ApiModel):
    dry_run: bool = True


class ArtifactCleanupResult(ApiModel):
    dry_run: bool
    marked_count: int
    removed_count: int
    reclaimed_bytes: int


class MessagePartOut(ApiModel):
    id: str
    position: int
    type: str
    text: str | None
    artifact_id: str | None
    metadata_json: dict[str, Any]
    artifact: ArtifactOut | None = None


class MessageOut(ApiModel):
    id: str
    chat_id: str
    parent_id: str | None
    role: str
    status: str
    parts: list[MessagePartOut]
    created_at: datetime
    updated_at: datetime


class ChatOut(ApiModel):
    id: str
    project_id: str | None
    title: str
    archived: bool
    routing_mode: str
    confirm_uncertain_media: bool
    active_chat_profile_id: str | None
    active_image_profile_id: str | None
    active_video_profile_id: str | None
    active_head_message_id: str | None
    created_at: datetime
    updated_at: datetime


class ChatDetail(ChatOut):
    messages: list[MessageOut]


class TurnRequest(ApiModel):
    text: str = Field(min_length=1, max_length=200_000)
    mode: RoutingMode | None = None
    parent_message_id: str | None = None
    input_artifact_ids: list[str] = Field(default_factory=list, max_length=16)
    settings: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, max_length=200)


class RegenerateRequest(ApiModel):
    settings: dict[str, Any] = Field(default_factory=dict)


class RoutingPlan(ApiModel):
    operation: Operation
    standalone_prompt: str
    negative_prompt: str | None = None
    input_artifact_ids: list[str] = Field(default_factory=list)
    profile_id: str | None = None
    workflow_id: str | None = None
    parameter_overrides: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    reason: str


class RunOut(ApiModel):
    id: str
    idempotency_key: str | None
    chat_id: str
    user_message_id: str
    assistant_message_id: str
    operation: str
    status: str
    standalone_prompt: str
    profile_id: str | None
    workflow_revision_id: str | None
    settings_json: dict[str, Any]
    provenance_json: dict[str, Any]
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    duration_ms: int | None
    created_at: datetime
    updated_at: datetime


class TurnAccepted(ApiModel):
    run: RunOut
    user_message: MessageOut
    assistant_message: MessageOut


class JobOut(ApiModel):
    id: str
    kind: str
    status: str
    run_id: str | None
    progress: float
    phase: str
    payload_json: dict[str, Any]
    result_json: dict[str, Any]
    error: str | None
    attempt: int
    cancellable: bool
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ModelSourceOut(ApiModel):
    id: str
    provider: str
    remote_id: str
    revision: str
    metadata_json: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ModelInstallOut(ApiModel):
    id: str
    source_id: str | None
    name: str
    role: str
    engine: str
    local_path: str
    size_bytes: int
    compatibility: str
    manifest_json: dict[str, Any]
    active: bool
    created_at: datetime
    updated_at: datetime


class ModelStorageInfo(ApiModel):
    installed_bytes: int
    partial_download_bytes: int
    catalog_cache_bytes: int
    installed_count: int
    partial_download_count: int


class StorageCleanupResult(ApiModel):
    removed_count: int
    reclaimed_bytes: int


class ModelImport(ApiModel):
    name: str = Field(min_length=1, max_length=300)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    local_path: str = Field(min_length=1, max_length=4_096)


class ModelProfileCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    model_install_id: str | None = None
    load_settings: dict[str, Any] = Field(default_factory=dict)
    request_settings: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class ModelProfileUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    load_settings: dict[str, Any] | None = None
    request_settings: dict[str, Any] | None = None
    is_default: bool | None = None


class ModelProfileClone(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)


class ModelProfileBundle(ApiModel):
    format: Literal["lm-atelier-profile"] = "lm-atelier-profile"
    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    model_install_id: str | None = None
    load_settings: dict[str, Any] = Field(default_factory=dict)
    request_settings: dict[str, Any] = Field(default_factory=dict)


class ModelProfileOut(ApiModel):
    id: str
    model_install_id: str | None
    name: str
    role: str
    engine: str
    load_settings_json: dict[str, Any]
    request_settings_json: dict[str, Any]
    is_default: bool
    created_at: datetime
    updated_at: datetime


class PresetCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    settings: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class PresetUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    settings: dict[str, Any] | None = None
    is_default: bool | None = None


class PresetClone(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)


class PresetBundle(ApiModel):
    format: Literal["lm-atelier-preset"] = "lm-atelier-preset"
    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    settings: dict[str, Any] = Field(default_factory=dict)


class PresetOut(ApiModel):
    id: str
    name: str
    role: str
    settings_json: dict[str, Any]
    is_default: bool
    created_at: datetime
    updated_at: datetime


class WorkflowCreate(ApiModel):
    name: str = Field(min_length=1, max_length=240)
    operation: Operation
    description: str = Field(default="", max_length=10_000)
    engine: str = "comfyui"
    engine_version: str | None = None
    ui_graph: dict[str, Any] = Field(default_factory=dict)
    api_graph: dict[str, Any]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)
    trusted: bool = False


class WorkflowRevisionCreate(ApiModel):
    engine_version: str | None = None
    ui_graph: dict[str, Any] = Field(default_factory=dict)
    api_graph: dict[str, Any]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)
    trusted: bool = False


class WorkflowUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = Field(default=None, max_length=10_000)


class WorkflowClone(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=240)


class WorkflowBundle(ApiModel):
    format: Literal["lm-atelier-workflow"] = "lm-atelier-workflow"
    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=240)
    operation: Operation
    description: str = Field(default="", max_length=10_000)
    engine: str = "comfyui"
    engine_version: str | None = None
    ui_graph: dict[str, Any] = Field(default_factory=dict)
    api_graph: dict[str, Any]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)
    trusted: bool = False
    source_revision: int | None = None


class WorkflowRevisionOut(ApiModel):
    id: str
    workflow_id: str
    version: int
    engine: str
    engine_version: str | None
    ui_graph_json: dict[str, Any]
    api_graph_json: dict[str, Any]
    input_schema_json: dict[str, Any]
    dependencies_json: dict[str, Any]
    trusted: bool
    created_at: datetime


class WorkflowOut(ApiModel):
    id: str
    name: str
    operation: str
    description: str
    current_revision_id: str | None
    revisions: list[WorkflowRevisionOut]
    created_at: datetime
    updated_at: datetime


class CatalogModel(ApiModel):
    provider: str = "huggingface"
    remote_id: str
    name: str
    author: str | None = None
    pipeline_tag: str | None = None
    tags: list[str] = Field(default_factory=list)
    downloads: int | None = None
    likes: int | None = None
    trending_score: float | None = None
    created_at: datetime | None = None
    last_modified: datetime | None = None
    gated: bool | str | None = None
    private: bool = False
    library_name: str | None = None
    architecture: str | None = None
    formats: list[str] = Field(default_factory=list)
    quantizations: list[str] = Field(default_factory=list)
    parameter_count: int | None = None
    license_id: str | None = None
    total_size_bytes: int | None = None
    compatibility: str
    compatibility_reasons: list[str] = Field(default_factory=list)


class CatalogPage(ApiModel):
    items: list[CatalogModel]
    next_cursor: str | None = None


class CatalogDetail(ApiModel):
    model: CatalogModel
    revision: str
    files: list[dict[str, Any]]


class CatalogPreflightRequest(ApiModel):
    revision: str = Field(default="main", min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    selected_files: list[str] = Field(default_factory=list, max_length=512)


class CatalogPreflightCheck(ApiModel):
    id: str
    label: str
    status: Literal["pass", "warn", "block"]
    detail: str


class CatalogPreflight(ApiModel):
    remote_id: str
    revision: str
    selected_files: list[str]
    download_bytes: int
    available_disk_bytes: int
    estimated_ram_bytes: int | None = None
    estimated_vram_bytes: int | None = None
    can_install: bool
    checks: list[CatalogPreflightCheck]


class DownloadRequest(ApiModel):
    remote_id: str = Field(min_length=1, max_length=500)
    revision: str = Field(default="main", min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    allow_patterns: list[str] = Field(default_factory=list)
    expected_sha256: dict[str, str] = Field(default_factory=dict)
    recipe_id: str | None = None
    recipe_version: int | None = None
    comfy_paths: dict[str, str] = Field(default_factory=dict)
    workflow_path: str | None = None
    default_settings: dict[str, Any] = Field(default_factory=dict)


class RecipeFile(ApiModel):
    path: str
    size_bytes: int | None = None
    sha256: str | None = None


class RecipeHardware(ApiModel):
    tier: Literal["cpu", "midrange-gpu", "high-end-gpu"]
    minimum_ram_gb: int
    recommended_ram_gb: int
    minimum_vram_gb: int | None = None
    recommended_vram_gb: int | None = None
    guidance: str


class ReferenceRecipe(ApiModel):
    id: str
    version: int
    name: str
    summary: str
    role: Literal["chat", "image", "video"]
    engine: Literal["llama.cpp", "comfyui"]
    operations: list[str]
    license_id: str
    status: Literal["reference-candidate", "certified"]
    certified: bool
    remote_id: str
    revision: str
    files: list[RecipeFile]
    total_size_bytes: int | None
    hardware: RecipeHardware
    default_settings: dict[str, Any]
    workflow_path: str | None = None
    node_policy: str | None = None
    notes: list[str] = Field(default_factory=list)


class SettingField(ApiModel):
    key: str
    label: str
    type: Literal["boolean", "integer", "number", "string", "enum", "array", "object"]
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: list[Any] = Field(default_factory=list)
    scope: Literal["load", "request", "workflow"]
    visibility: Literal["basic", "advanced", "expert"] = "advanced"
    restart_required: bool = False
    available: bool = True
    unavailable_reason: str | None = None
    help: str = ""


class EngineCapabilities(ApiModel):
    engine: str
    version: str
    roles: list[str]
    operations: list[str]
    formats: list[str]
    devices: list[str]
    streaming: bool
    tool_calling: bool
    settings: list[SettingField]
    healthy: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ToolCapabilityProbe(ApiModel):
    engine: str
    version: str
    advertised: bool
    passed: bool
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    error: str | None = None


class DeviceInfo(ApiModel):
    id: str
    name: str
    kind: str
    total_memory_bytes: int | None = None
    available_memory_bytes: int | None = None
    backend: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PlatformMatrixEntry(ApiModel):
    id: str
    name: str
    status: Literal["target", "experimental"]
    operating_systems: list[str]
    architectures: list[str]
    accelerator: str
    workloads: list[str]
    vram_tiers_gb: list[int] = Field(default_factory=list)
    evidence: str
    notes: list[str] = Field(default_factory=list)


class PlatformAssessment(ApiModel):
    platform_status: Literal["target", "experimental", "unsupported"]
    platform_label: str
    accelerator_status: Literal["primary", "experimental", "cpu-only"]
    accelerator_label: str
    certification_status: Literal["hardware-pending", "experimental", "unsupported"]
    chat_ready: bool
    reference_media_ready: bool
    vram_tier_gb: int | None = None
    messages: list[str] = Field(default_factory=list)


class SystemInfo(ApiModel):
    platform: str
    platform_release: str
    distribution: str
    distribution_version: str
    architecture: str
    python_version: str
    cpu_count: int
    memory_total_bytes: int
    memory_available_bytes: int
    disk_total_bytes: int
    disk_free_bytes: int
    ffmpeg_available: bool
    devices: list[DeviceInfo]
    support: PlatformAssessment


class WorkerStatus(ApiModel):
    name: Literal["chat", "media"]
    state: Literal["stopped", "starting", "ready", "exited"] = "stopped"
    managed: bool
    running: bool
    pid: int | None = None
    profile_id: str | None = None
    command: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    estimated_memory_bytes: int | None = None
    current_memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    active_jobs: int = 0
    queued_jobs: int = 0


class BackupInfo(ApiModel):
    name: str
    size_bytes: int
    sha256: str
    created_at: datetime
    verified: bool = False
    restore_pending: bool = False
    media_included: bool = False
    media_size_bytes: int = 0


class EventOut(ApiModel):
    sequence: int
    type: str
    entity_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class HealthOut(ApiModel):
    status: Literal["ok", "degraded"]
    version: str
    database: bool
    engines: list[EngineCapabilities]
