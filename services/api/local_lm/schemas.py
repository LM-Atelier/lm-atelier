from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .domain import Operation, RoutingMode


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


GenerationSettingsByRole = dict[
    Literal["chat", "image", "video"],
    dict[str, Any],
]
GenerationPresetIdsByRole = dict[
    Literal["chat", "image", "video"],
    str | None,
]


class VisionSettings(ApiModel):
    max_images: int = Field(default=4, ge=1, le=16)
    max_video_frames: int = Field(default=6, ge=3, le=16)
    include_prior_visual: bool = True


class ProjectCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10_000)
    instructions: str = Field(default="", max_length=100_000)
    image_workflow_revision_id: str | None = None
    video_workflow_revision_id: str | None = None
    generation_settings_json: GenerationSettingsByRole = Field(default_factory=dict)
    generation_preset_ids_json: GenerationPresetIdsByRole = Field(default_factory=dict)


class ProjectUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=10_000)
    instructions: str | None = Field(default=None, max_length=100_000)
    archived: bool | None = None
    image_workflow_revision_id: str | None = None
    video_workflow_revision_id: str | None = None
    generation_settings_json: GenerationSettingsByRole | None = None
    generation_preset_ids_json: GenerationPresetIdsByRole | None = None


class ProjectOut(ApiModel):
    id: str
    name: str
    description: str
    instructions: str
    archived: bool
    image_workflow_revision_id: str | None
    video_workflow_revision_id: str | None
    generation_settings_json: GenerationSettingsByRole
    generation_preset_ids_json: GenerationPresetIdsByRole
    created_at: datetime
    updated_at: datetime


class ChatCreate(ApiModel):
    title: str = Field(default="New chat", min_length=1, max_length=240)
    project_id: str | None = None
    routing_mode: RoutingMode = RoutingMode.AUTO
    generation_settings_json: GenerationSettingsByRole = Field(default_factory=dict)
    generation_preset_ids_json: GenerationPresetIdsByRole = Field(default_factory=dict)
    vision_settings_json: VisionSettings = Field(default_factory=VisionSettings)


class ChatUpdate(ApiModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    project_id: str | None = None
    archived: bool | None = None
    routing_mode: RoutingMode | None = None
    confirm_uncertain_media: bool | None = None
    active_chat_profile_id: str | None = None
    active_vision_profile_id: str | None = None
    active_image_profile_id: str | None = None
    active_video_profile_id: str | None = None
    generation_settings_json: GenerationSettingsByRole | None = None
    generation_preset_ids_json: GenerationPresetIdsByRole | None = None
    vision_settings_json: VisionSettings | None = None


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
    retention_pending_count: int
    disk_free_bytes: int
    warning: bool
    retention_days: int
    temporary_retention_hours: int


class ArtifactCleanupRequest(ApiModel):
    dry_run: bool = True


class ArtifactCleanupResult(ApiModel):
    dry_run: bool
    marked_count: int
    retention_pending_count: int
    removed_count: int
    reclaimed_bytes: int


class ArtifactDeleteResult(ApiModel):
    artifact_id: str
    reference_count: int
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


class ResponseRevisionOut(ApiModel):
    id: str
    message_id: str
    run_id: str | None
    sequence: int
    status: str
    parts: list[MessagePartOut]
    created_at: datetime
    updated_at: datetime


class MessageOut(ApiModel):
    id: str
    chat_id: str
    parent_id: str | None
    role: str
    status: str
    transcript_visible: bool
    active_response_revision_id: str | None
    parts: list[MessagePartOut]
    response_revisions: list[ResponseRevisionOut] = Field(default_factory=list)
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
    active_vision_profile_id: str | None
    active_image_profile_id: str | None
    active_video_profile_id: str | None
    active_head_message_id: str | None
    generation_settings_json: GenerationSettingsByRole
    generation_preset_ids_json: GenerationPresetIdsByRole
    vision_settings_json: VisionSettings
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
    ordered_settings: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=3)
    output_count: int | None = Field(default=None, ge=1, le=16)
    confirm_media: bool = False
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
    output_count: int = Field(default=1, ge=1, le=16)
    confidence: float = Field(ge=0, le=1)
    reason: str


class GenerationOfferItem(ApiModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    mode: Literal["image", "video"]
    prompt: str = Field(min_length=1, max_length=20_000)


class GenerationOffer(ApiModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")

    message: str = Field(min_length=1, max_length=1_000)
    items: list[GenerationOfferItem] = Field(min_length=1, max_length=8)


class OrderedStepInput(ApiModel):
    source_step_id: str = Field(
        min_length=1,
        max_length=40,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    kind: Literal["text_context", "artifact"]


class OrderedStepIntent(ApiModel):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]*$")
    mode: Literal["text", "image", "video"]
    prompt: str = Field(min_length=1, max_length=20_000)
    depends_on: list[str] = Field(default_factory=list, max_length=8)
    inputs: list[OrderedStepInput] = Field(default_factory=list, max_length=8)


class OrderedWorkIntent(ApiModel):
    planner_version: Literal["ordered-work-v1"] = "ordered-work-v1"
    steps: list[OrderedStepIntent] = Field(min_length=2, max_length=8)
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1, max_length=1_000)
    requires_confirmation: bool = False


class RunOut(ApiModel):
    id: str
    idempotency_key: str | None
    chat_id: str
    user_message_id: str
    assistant_message_id: str
    work_plan_id: str | None
    work_step_id: str | None
    operation: str
    status: str
    standalone_prompt: str
    profile_id: str | None
    vision_profile_id: str | None
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


class ProgressV2(ApiModel):
    version: Literal[2] = 2
    stage: str
    stage_progress: float | None = Field(default=None, ge=0, le=1)
    overall_progress: float | None = Field(default=None, ge=0, le=1)
    completed_units: int | None = Field(default=None, ge=0)
    total_units: int | None = Field(default=None, ge=0)
    unit: str | None = None
    bytes_reused: int = Field(default=0, ge=0)
    rate_bytes_per_second: float | None = Field(default=None, ge=0)
    eta_seconds: int | None = Field(default=None, ge=0)
    file_index: int | None = Field(default=None, ge=1)
    file_count: int | None = Field(default=None, ge=1)
    queue_resource: str | None = None
    queue_position: int | None = Field(default=None, ge=0)
    queue_length: int | None = Field(default=None, ge=0)
    blocked_by: list[str] = Field(default_factory=list)
    indeterminate: bool = False
    updated_at: datetime


class JobOut(ApiModel):
    id: str
    kind: str
    status: str
    run_id: str | None
    work_plan_id: str | None
    work_step_id: str | None
    progress: float
    phase: str
    progress_json: dict[str, Any]
    queue_resource: str | None
    queue_group: str | None
    queue_priority: int
    queue_ticket: str | None
    enqueued_at: datetime | None
    claim_expires_at: datetime | None
    heartbeat_at: datetime | None
    payload_json: dict[str, Any]
    result_json: dict[str, Any]
    error: str | None
    attempt: int
    cancellable: bool
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class WorkStepOut(ApiModel):
    id: str
    plan_id: str
    run_id: str | None
    ordinal: int
    display_group: str | None
    operation: str
    status: str
    prompt: str
    profile_id: str | None
    workflow_revision_id: str | None
    settings_json: dict[str, Any]
    input_bindings_json: list[dict[str, Any]]
    output_contract_json: list[dict[str, Any]]
    queue_class: str
    error: str | None
    created_at: datetime
    updated_at: datetime


class WorkPlanOut(ApiModel):
    id: str
    chat_id: str
    idempotency_key: str | None
    source_action: str
    persistence_scope: str
    status: str
    context_head_message_id: str | None
    transcript_sequence: int
    priority: int
    planner_version: str
    failure_policy: str
    summary_json: dict[str, Any]
    steps: list[WorkStepOut]
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


class InstallArtifact(ApiModel):
    path: str = Field(min_length=1, max_length=1_000)
    kind: str = Field(min_length=1, max_length=40)
    target_folder: str = Field(min_length=1, max_length=80)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    required: bool = True
    reuse: Literal["download", "installed", "verified-cache"] = "download"


class InstallPlanOut(ApiModel):
    id: str
    provider: str
    remote_id: str
    revision: str
    role: str
    engine: str
    architecture: str | None
    family: str | None
    plan_hash: str
    resolver_version: str
    compatibility: str
    artifacts_json: list[dict[str, Any]]
    runtime_contract_json: dict[str, Any]
    activation_probe_json: dict[str, Any]
    status: str
    failure_code: str | None
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime


class ModelCapabilityEvidenceOut(ApiModel):
    id: str
    model_install_id: str
    evidence_key: str
    result: str
    component_hashes_json: dict[str, str]
    runtime_build: str
    adapter_contract_version: int
    launch_contract_version: str
    workflow_contract_version: str | None
    hardware_class: str
    probe_version: str
    failure_code: str | None
    failure_reason: str | None
    details_json: dict[str, Any]
    probed_at: datetime


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
    readiness: Literal["ready", "unverified", "unsupported"] = "unverified"
    capability_evidence: ModelCapabilityEvidenceOut | None = None
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
    use_case: str = Field(default="", max_length=1_000)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    model_install_id: str | None = None
    load_settings: dict[str, Any] = Field(default_factory=dict)
    request_settings: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class ModelProfileUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    use_case: str | None = Field(default=None, max_length=1_000)
    load_settings: dict[str, Any] | None = None
    request_settings: dict[str, Any] | None = None
    is_default: bool | None = None


class ModelProfileClone(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)


class ModelProfileBundle(ApiModel):
    format: Literal["lm-atelier-profile"] = "lm-atelier-profile"
    version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=200)
    use_case: str = Field(default="", max_length=1_000)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    model_install_id: str | None = None
    load_settings: dict[str, Any] = Field(default_factory=dict)
    request_settings: dict[str, Any] = Field(default_factory=dict)


class ModelProfileOut(ApiModel):
    id: str
    model_install_id: str | None
    name: str
    use_case: str
    role: str
    engine: str
    load_settings_json: dict[str, Any]
    request_settings_json: dict[str, Any]
    is_default: bool
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
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


class WorkflowOpenTarget(ApiModel):
    url: str
    filename: str
    ui_graph: dict[str, Any]


class CustomNodeInstallRequest(ApiModel):
    name: str = Field(min_length=1, max_length=240)
    source_url: str = Field(min_length=1, max_length=1000)
    revision: str = Field(min_length=40, max_length=40)


class CustomNodeUpdateRequest(ApiModel):
    revision: str = Field(min_length=40, max_length=40)


class CustomNodeTrustRequest(ApiModel):
    trusted: bool


class CustomNodeOut(ApiModel):
    id: str
    name: str
    source_url: str
    revision: str
    previous_revision: str | None
    tree_hash: str
    trusted: bool
    active: bool
    security_json: dict[str, Any]
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
    required_runtime: str | None = None


class CatalogPage(ApiModel):
    items: list[CatalogModel]
    next_cursor: str | None = None
    stale: bool = False


class CatalogDetail(ApiModel):
    model: CatalogModel
    revision: str
    files: list[dict[str, Any]]


class CatalogPreflightRequest(ApiModel):
    revision: str = Field(default="main", min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    selected_files: list[str] = Field(default_factory=list, max_length=512)
    auxiliary_kind: (
        Literal[
            "lora",
            "vae",
            "controlnet",
            "upscaler",
            "embedding",
            "ip_adapter",
        ]
        | None
    ) = None


class CatalogPreflightCheck(ApiModel):
    id: str
    label: str
    status: Literal["pass", "warn", "block"]
    detail: str


class CatalogFileSource(ApiModel):
    remote_id: str = Field(min_length=1, max_length=500)
    revision: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=1_000)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")


class CatalogPreflight(ApiModel):
    remote_id: str
    source_remote_id: str | None = None
    revision: str
    selected_files: list[str]
    expected_sha256: dict[str, str] = Field(default_factory=dict)
    file_sources: dict[str, CatalogFileSource] = Field(default_factory=dict)
    comfy_paths: dict[str, str] = Field(default_factory=dict)
    workflow_template_id: str | None = None
    workflow_template_sha256: str | None = None
    download_bytes: int
    available_disk_bytes: int
    estimated_ram_bytes: int | None = None
    estimated_vram_bytes: int | None = None
    can_install: bool
    checks: list[CatalogPreflightCheck]
    install_plan: InstallPlanOut | None = None
    auxiliary_kind: str | None = None


class DownloadRequest(ApiModel):
    install_plan_id: str | None = Field(default=None, max_length=40)
    remote_id: str = Field(min_length=1, max_length=500)
    source_remote_id: str | None = Field(default=None, min_length=1, max_length=500)
    revision: str = Field(default="main", min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    allow_patterns: list[str] = Field(default_factory=list)
    expected_sha256: dict[str, str] = Field(default_factory=dict)
    file_sources: dict[str, CatalogFileSource] = Field(default_factory=dict)
    recipe_id: str | None = None
    recipe_version: int | None = None
    comfy_paths: dict[str, str] = Field(default_factory=dict)
    workflow_path: str | None = None
    workflow_template_id: str | None = None
    workflow_template_sha256: str | None = None
    default_settings: dict[str, Any] = Field(default_factory=dict)
    auxiliary_kind: (
        Literal[
            "lora",
            "vae",
            "controlnet",
            "upscaler",
            "embedding",
            "ip_adapter",
        ]
        | None
    ) = None


class ModelAssetOut(ApiModel):
    id: str
    source_id: str | None
    name: str
    kind: str
    family: str | None
    size_bytes: int
    manifest_json: dict[str, Any]
    active: bool
    verified_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ModelAssetUpdate(ApiModel):
    active: bool


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
    engine: Literal["llama.cpp", "vllm", "comfyui"]
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
    multiple_of: float | None = None
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
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    formats: list[str]
    devices: list[str]
    streaming: bool
    tool_calling: bool
    settings: list[SettingField]
    settings_by_role: dict[str, list[SettingField]] = Field(default_factory=dict)
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
    cpu_model: str
    cpu_count: int
    memory_total_bytes: int
    memory_available_bytes: int
    disk_total_bytes: int
    disk_free_bytes: int
    ffmpeg_available: bool
    devices: list[DeviceInfo]
    support: PlatformAssessment


class ApplicationInfo(ApiModel):
    version: str
    data_directory: str
    log_directory: str


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
    failure_detail: str | None = None
    stderr_tail: str | None = None
    log_path: str | None = None


class RuntimeStatus(ApiModel):
    engine: Literal["llama.cpp", "vllm", "comfyui"]
    release: str
    state: Literal["missing", "installing", "ready", "failed", "unsupported"]
    supported: bool
    managed: bool = False
    progress: float = 0
    progress_json: ProgressV2 | None = None
    downloaded_bytes: int = 0
    size_bytes: int | None = None
    distribution: str
    license: str
    security_status: Literal["checksum-pinned", "blocked"] = "checksum-pinned"
    security_message: str = ""
    message: str = ""


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


class CredentialStatus(ApiModel):
    provider: Literal["huggingface"] = "huggingface"
    configured: bool
    source: Literal["none", "environment", "credential_vault"]
    vault_available: bool


class CredentialSet(ApiModel):
    token: str = Field(min_length=1, max_length=10_000)
