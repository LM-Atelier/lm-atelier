from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_serializer,
    field_validator,
)

from .comfy_workflow_packages import WorkflowPackageIssueCode
from .domain import (
    ArtifactKind,
    JobKind,
    JobStatus,
    MaskMode,
    MessageStatus,
    Operation,
    PartType,
    ResourceKind,
    RoutingMode,
    RunStatus,
)
from .install_plan_types import InstallPlanFailureCode
from .model_asset_types import BoundWorkflowAssetKind, InstalledAssetKind
from .model_asset_types import WorkflowAssetKind as WorkflowAssetKind
from .references import (
    MAX_REFERENCES_PER_TURN,
    MAX_ROLE,
    MentionSource,
    ReferenceKind,
    ValidationState,
)
from .saved_settings import GenerationSettingsByRole, SavedRoleSettings
from .studio_capabilities import StudioToolKind
from .worker_failures import WorkerFailureCode


class ApiModel(BaseModel):
    # extra="forbid": a client typo in a request field must be a 422, not a
    # silently applied default. Response construction is unaffected - servers
    # build these from exact attributes.
    model_config = ConfigDict(from_attributes=True, extra="forbid")


ContentRating = Literal["general", "mature", "unknown"]
#: What a compute device is. Produced only by hardware.py, which builds every
#: DeviceInfo and passes one of these three literals; there is no stored column
#: behind it and so no constraint to derive. Bound to those producers by test.
DeviceKind = Literal["accelerator", "cpu", "gpu"]

GenerationPresetIdsByRole = dict[
    Literal["chat", "image", "video"],
    str | None,
]


class VisionSettings(ApiModel):
    max_images: int = Field(default=4, ge=1, le=16)
    max_video_frames: int = Field(default=6, ge=3, le=16)
    include_prior_visual: bool = True
    verify_image_edits: bool = False
    compile_visual_prompts: bool = True


def new_chat_vision_settings() -> VisionSettings:
    """Enable edit review for new chats without changing legacy settings defaults."""
    return VisionSettings(verify_image_edits=True)


class WebSettings(ApiModel):
    """Whether this conversation may reach the internet.

    Off unless someone turned it on for this chat specifically. A new chat
    never inherits it, so permission cannot spread by being nearby.
    """

    allow_url_fetch: bool = False
    allow_search: bool = Field(default=False, strict=True)
    allow_search_without_asking: bool = Field(default=False, strict=True)


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
    pinned: bool | None = None
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
    pinned: bool
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
    vision_settings_json: VisionSettings = Field(default_factory=new_chat_vision_settings)


class ChatUpdate(ApiModel):
    title: str | None = Field(default=None, min_length=1, max_length=240)
    project_id: str | None = None
    archived: bool | None = None
    pinned: bool | None = None
    routing_mode: RoutingMode | None = None
    confirm_uncertain_media: bool | None = None
    active_chat_profile_id: str | None = None
    active_vision_profile_id: str | None = None
    active_image_profile_id: str | None = None
    active_video_profile_id: str | None = None
    generation_settings_json: GenerationSettingsByRole | None = None
    generation_preset_ids_json: GenerationPresetIdsByRole | None = None
    vision_settings_json: VisionSettings | None = None
    web_settings_json: WebSettings | None = None


class GenerationIdentityOut(ApiModel):
    model_profile_name: str | None = None
    workflow_family_name: str | None = None
    workflow_definition_name: str | None = None
    workflow_version: int | None = None


class ArtifactOut(ApiModel):
    id: str
    sha256: str
    kind: ArtifactKind
    media_type: str
    size_bytes: int
    original_name: str | None
    metadata_json: dict[str, Any]
    favorite: bool = False
    created_at: datetime
    url: str | None = None
    generation_identity: GenerationIdentityOut | None = None


class ArtifactUpdate(ApiModel):
    favorite: bool


class ArtifactLibraryItem(ArtifactOut):
    reference_count: int = 0
    chat_ids: list[str] = Field(default_factory=list)
    project_ids: list[str] = Field(default_factory=list)


class ArtifactLibraryEntrySummary(ApiModel):
    id: str = Field(min_length=1, max_length=80)
    artifact_id: str = Field(min_length=1, max_length=80)
    version: int = Field(ge=1)
    state: Literal["visible", "trashed"]
    display_name: str = Field(min_length=1, max_length=500)
    favorite: bool
    kind: Literal["image", "video"]
    media_type: str = Field(min_length=1, max_length=120)
    size_bytes: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class ArtifactLibraryPage(ApiModel):
    items: list[ArtifactLibraryEntrySummary]
    next_cursor: str | None = Field(default=None, min_length=1, max_length=2_048)


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
    # A real run is one bounded batch; True means eligible artifacts remain
    # and another call continues from where this one stopped.
    truncated: bool = False


#: The installation settings' own bounds, so a window chosen in Settings is
#: always one the configuration could have set.
MAX_RETENTION_DAYS = 3650
MAX_TEMPORARY_RETENTION_HOURS = 168


class RetentionWindowsIn(ApiModel):
    """How long media nothing uses is kept, and how long previews are kept."""

    media_days: StrictInt = Field(ge=1, le=MAX_RETENTION_DAYS)
    temporary_hours: StrictInt = Field(ge=1, le=MAX_TEMPORARY_RETENTION_HOURS)


class RetentionPolicyWrite(RetentionWindowsIn):
    expected_revision: StrictInt = Field(ge=0, le=9_223_372_036_854_775_807)


class RetentionPolicyOut(ApiModel):
    media_days: int
    temporary_hours: int
    # 0 until somebody chooses, and both windows are then the installation's.
    revision: int = Field(ge=0)
    default_media_days: int
    default_temporary_hours: int


class ArtifactDeleteResult(ApiModel):
    artifact_id: str
    reference_count: int
    removed_count: int
    reclaimed_bytes: int


class MessagePartOut(ApiModel):
    id: str
    position: int
    type: PartType
    text: str | None
    artifact_id: str | None
    metadata_json: dict[str, Any]
    artifact: ArtifactOut | None = None


class ResponseRevisionOut(ApiModel):
    id: str
    message_id: str
    run_id: str | None
    sequence: int
    status: MessageStatus
    parts: list[MessagePartOut]
    feedback: Literal["up", "down"] | None = None
    activity: ChatActivityReferenceOut | None = None
    created_at: datetime
    updated_at: datetime


class MessageReferenceOut(ApiModel):
    """What one turn referred to, as it stood when the turn was accepted.

    The name and mention are the recorded ones, not the subject's current
    values, and the subject id carries no promise that the subject still
    exists. That is the point: a renamed subject must not rewrite an old
    message, and a deleted one must not erase the record that it was used.
    """

    reference_subject_id: str
    mention_slug: str
    subject_name: str
    subject_kind: ReferenceKind
    role: str | None = None
    strength: float | None = None
    source: str
    # The _json suffix matches the columns and the shape every other
    # reference field already takes in this API.
    reference_asset_ids_json: list[str] = Field(default_factory=list)
    artifact_ids_json: list[str] = Field(default_factory=list)


class MessageOut(ApiModel):
    id: str
    chat_id: str
    parent_id: str | None
    role: str
    status: MessageStatus
    transcript_visible: bool
    content_removed_at: datetime | None
    active_response_revision_id: str | None
    parts: list[MessagePartOut]
    # Empty for every message that named nothing, which is almost all of them.
    references: list[MessageReferenceOut] = Field(default_factory=list)
    response_revisions: list[ResponseRevisionOut] = Field(default_factory=list)
    feedback: Literal["up", "down"] | None = None
    created_at: datetime
    updated_at: datetime

    @field_serializer("content_removed_at", "created_at", "updated_at", when_used="json")
    def serialize_timestamp_as_utc(self, value: datetime | None) -> str | None:
        """Keep SQLite-naive UTC instants explicit at the browser boundary."""
        if value is None:
            return None
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")


class ResponseFeedbackUpdate(ApiModel):
    """Set or clear one verdict; null rating clears. A click stores a local
    preference for evaluation and reranking - it never trains weights."""

    rating: Literal["up", "down"] | None
    response_revision_id: str | None = None


class ResponseFeedbackOut(ApiModel):
    message_id: str
    response_revision_id: str | None
    rating: Literal["up", "down"] | None


class ChatActivityReferenceOut(ApiModel):
    id: str
    sequence: int = Field(ge=1)
    message_id: str
    response_revision_id: str
    occurred_at: datetime

    @field_serializer("occurred_at", when_used="json")
    def serialize_timestamp_as_utc(self, value: datetime) -> str:
        return value.replace(tzinfo=UTC).isoformat() if value.tzinfo is None else value.isoformat()


class ChatActivityOut(ApiModel):
    active_work_count: int = Field(ge=0)
    unresolved_failed_count: int = Field(ge=0)
    last_output: ChatActivityReferenceOut | None
    last_failure: ChatActivityReferenceOut | None


class ChatSummaryOut(ApiModel):
    id: str
    project_id: str | None
    title: str
    archived: bool
    pinned: bool
    created_at: datetime
    updated_at: datetime
    activity: ChatActivityOut


class ChatOut(ApiModel):
    id: str
    project_id: str | None
    title: str
    archived: bool
    pinned: bool
    routing_mode: RoutingMode
    confirm_uncertain_media: bool
    active_chat_profile_id: str | None
    active_vision_profile_id: str | None
    active_image_profile_id: str | None
    active_video_profile_id: str | None
    active_head_message_id: str | None
    generation_settings_json: GenerationSettingsByRole
    generation_preset_ids_json: GenerationPresetIdsByRole
    vision_settings_json: VisionSettings
    web_settings_json: WebSettings = Field(default_factory=WebSettings)
    # Empty for a chat created directly; carries the source chat and message
    # when this thread was forked from one.
    origin_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class WebSearchResultOut(ApiModel):
    url: str = Field(max_length=2_000)
    title: str = Field(max_length=200)
    snippet: str = Field(max_length=2_000)


class WebSearchOut(ApiModel):
    run_id: str
    assistant_message_id: str
    job_id: str | None = None
    revision: int | None = None
    state: Literal[
        "awaiting_approval",
        "scheduled",
        "approved",
        "declined",
        "cancelled",
        "dispatching",
        "complete",
        "failed",
        "uncertain",
    ]
    query: str = Field(min_length=1, max_length=2_000)
    provider: Literal["CRW"] = "CRW"
    provider_endpoint: str = Field(max_length=2_000)
    dispatch_after: datetime | None = None
    results: list[WebSearchResultOut] = Field(default_factory=list, max_length=5)
    result_count: int = Field(default=0, ge=0, le=5)
    truncated: bool = False
    error_code: (
        Literal[
            "search_provider_invalid",
            "search_query_invalid",
            "search_credentials_refused",
            "search_redirect_refused",
            "search_rate_limited",
            "search_unavailable",
            "search_timeout",
            "search_response_invalid",
            "search_response_too_large",
            "search_dispatch_uncertain",
            "search_permission_revoked",
            "search_provider_changed",
            "search_work_unavailable",
        ]
        | None
    ) = None


class WebSearchDecisionRequest(ApiModel):
    revision: int = Field(strict=True, ge=1, le=2**63 - 1)
    action: Literal["approve", "decline", "cancel"]


class WebSearchEditRequest(ApiModel):
    revision: int = Field(strict=True, ge=1, le=2**63 - 1)
    query: str = Field(strict=True, min_length=1, max_length=2_000)


class WebSearchConfiguration(ApiModel):
    installation_enabled: bool
    configured: bool
    provider: Literal["CRW"] = "CRW"
    provider_endpoint: str | None = None
    error_code: (
        Literal["search_not_configured", "search_provider_invalid", "search_credentials_invalid"]
        | None
    ) = None


class ChatDetail(ChatOut):
    messages: list[MessageOut]
    web_searches: list[WebSearchOut] = Field(default_factory=list)


class ChatMessageWindow(ApiModel):
    """One page of a conversation, and whether more of it exists either side.

    A whole transcript is not a page size that scales: a long conversation
    answers this endpoint in the same bounded time as a short one, which the
    endpoint that returns every message cannot do. ``has_older`` and
    ``has_newer`` are what let a reader ask for the next page without guessing
    whether there is one.
    """

    chat_id: str
    messages: list[MessageOut]
    has_older: bool
    has_newer: bool


class ExchangeDeletionOut(ApiModel):
    chat_id: str
    user_message_id: str
    message_ids: list[str]
    run_ids: list[str]
    job_ids: list[str]
    work_plan_ids: list[str]
    released_artifact_ids: list[str]
    retained_artifact_ids: list[str]
    new_head_message_id: str | None = None


class ChatItemRemovalReferenceOut(ApiModel):
    id: str
    subject_name: str
    mention_slug: str
    subject_kind: ReferenceKind


class ChatItemRemovalImpactOut(ApiModel):
    chat_id: str
    message_id: str
    message_revision_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    role: str
    already_removed: bool
    has_replies: bool
    source_backs_regeneration: bool
    detached_message_part_count: int
    detached_response_revision_part_count: int
    detached_reference_count: int
    detached_references: list[ChatItemRemovalReferenceOut]
    detached_references_truncated: bool
    released_artifact_count: int
    released_artifact_ids: list[str]
    released_artifacts_truncated: bool
    retained_artifact_count: int
    retained_artifact_ids: list[str]
    retained_artifacts_truncated: bool
    retained_witness_classes: list[str]
    forensic_erasure: Literal[False]
    execute_authorized: Literal[False]


class ChatItemRemovalExecute(ApiModel):
    expected_message_id: str = Field(min_length=1, max_length=40)
    expected_revision_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation_key: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$",
    )


class ChatItemRemovalExecutionOut(ApiModel):
    operation_key: str
    chat_id: str
    message_id: str
    message_revision_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_removed_at: datetime
    replayed: bool


class StudioSessionCreate(ApiModel):
    """Open the studio over one image; the session is found or created."""

    source_artifact_id: str = Field(min_length=1, max_length=80)
    # When the studio is entered from a chat, its profile and settings
    # snapshot carry over so applies run with the same models.
    source_chat_id: str | None = Field(default=None, max_length=40)


class PromptHelperCreate(ApiModel):
    source_chat_id: str = Field(min_length=1, max_length=40)
    draft_prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000),
    ]


class PromptHelperUpdate(ApiModel):
    draft_prompt: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000),
    ]


class PromptHelperDetail(ChatDetail):
    draft_prompt: str


class PromptTemplateCreate(ApiModel):
    idempotency_key: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$",
    )
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4_000)
    contract: dict[str, Any]


class PromptTemplateUpdate(ApiModel):
    expected_current_revision_id: str = Field(min_length=1, max_length=40)
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$",
    )
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=4_000)
    archived: bool | None = None
    contract: dict[str, Any] | None = None


class PromptTemplateRestore(ApiModel):
    expected_current_revision_id: str = Field(min_length=1, max_length=40)
    idempotency_key: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$",
    )


class PromptTemplateRevisionOut(ApiModel):
    id: str
    prompt_template_id: str
    version: int
    schema_version: int
    contract_json: dict[str, Any]
    contract_sha256: str
    created_at: datetime

    @field_serializer("created_at", when_used="json")
    def serialize_timestamp_as_utc(self, value: datetime) -> str:
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")


class PromptTemplateDefinitionOut(ApiModel):
    id: str
    name: str
    description: str
    archived: bool
    current_revision_id: str
    created_at: datetime
    updated_at: datetime

    @field_serializer("created_at", "updated_at", when_used="json")
    def serialize_timestamps_as_utc(self, value: datetime) -> str:
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")


class PromptTemplateDetailOut(PromptTemplateDefinitionOut):
    current_revision: PromptTemplateRevisionOut


class PromptTemplateWriteOut(ApiModel):
    template: PromptTemplateDetailOut
    revision: PromptTemplateRevisionOut
    idempotent: bool


class PromptTemplatePortableWorkflowDescriptorOut(ApiModel):
    descriptor_version: Literal[1]
    operation: Literal["text_to_image"]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_contract_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PromptTemplatePortableWorkflowBindingOut(ApiModel):
    key: str = Field(pattern=r"^workflow_[1-9][0-9]{0,2}$")
    descriptor: PromptTemplatePortableWorkflowDescriptorOut


class PromptTemplatePortableTemplateOut(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(max_length=4_000)
    contract: dict[str, Any]


class PromptTemplatePortableBundleOut(ApiModel):
    kind: Literal["lm-atelier-prompt-template"]
    bundle_version: Literal[1]
    template: PromptTemplatePortableTemplateOut
    workflows: list[PromptTemplatePortableWorkflowBindingOut] = Field(max_length=16)
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PromptTemplateImportWorkflowSuggestionOut(ApiModel):
    local_ref: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=240)
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_receipt: str = Field(min_length=1, max_length=2_048)


class PromptTemplateImportCandidateResolve(ApiModel):
    bundle_json: StrictStr = Field(min_length=1, max_length=524_288)
    preview_receipt: StrictStr = Field(min_length=1, max_length=2_048)
    binding_key: StrictStr = Field(pattern=r"^workflow_[1-9][0-9]{0,2}$")
    local_ref: StrictStr = Field(min_length=1, max_length=40)


class PromptTemplateImportWorkflowBindingIn(ApiModel):
    binding_key: StrictStr = Field(pattern=r"^workflow_[1-9][0-9]{0,2}$")
    local_ref: StrictStr = Field(min_length=1, max_length=40)
    candidate_receipt: StrictStr = Field(min_length=1, max_length=2_048)


class PromptTemplateImportLoraConfirmationIn(ApiModel):
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_receipt: StrictStr = Field(min_length=1, max_length=2_048)


class PromptTemplateImportCommit(ApiModel):
    idempotency_key: StrictStr = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$",
    )
    bundle_json: StrictStr = Field(min_length=1, max_length=524_288)
    preview_receipt: StrictStr = Field(min_length=1, max_length=2_048)
    confirmed_bundle_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    destination_name: StrictStr = Field(min_length=1, max_length=200)
    workflow_bindings: list[PromptTemplateImportWorkflowBindingIn] = Field(max_length=16)
    lora_confirmations: list[PromptTemplateImportLoraConfirmationIn] = Field(max_length=64)


class PromptTemplateImportCommitOut(ApiModel):
    template_id: str = Field(min_length=1, max_length=40)
    revision_id: str = Field(min_length=1, max_length=40)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotent: bool


class PromptTemplateImportWorkflowRequirementOut(ApiModel):
    kind: Literal["workflow"]
    binding_key: str = Field(pattern=r"^workflow_[1-9][0-9]{0,2}$")
    descriptor: PromptTemplatePortableWorkflowDescriptorOut
    suggestions: list[PromptTemplateImportWorkflowSuggestionOut] = Field(max_length=20)


class PromptTemplateImportLoraRequirementOut(ApiModel):
    kind: Literal["lora"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    available: StrictBool
    authority_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmation_receipt: str = Field(min_length=1, max_length=2_048)


class PromptTemplateImportPreviewOut(ApiModel):
    bundle: PromptTemplatePortableBundleOut
    requirements: list[
        PromptTemplateImportWorkflowRequirementOut | PromptTemplateImportLoraRequirementOut
    ]
    receipt: str = Field(min_length=1, max_length=2_048)
    expires_at: StrictInt = Field(ge=0)


class PromptTemplatePageOut(ApiModel):
    items: list[PromptTemplateDefinitionOut]
    total: int
    limit: int
    offset: int


class PromptExpansionCreate(ApiModel):
    idempotency_key: StrictStr = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$",
    )
    template_revision_id: StrictStr = Field(min_length=1, max_length=40)
    contract_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    item_count: StrictInt = Field(ge=1, le=16)
    selection_seed: StrictInt = Field(ge=0, lt=2_147_483_648)
    inputs: dict[StrictStr, StrictStr | list[StrictStr]] = Field(default_factory=dict)


class PromptExpansionItemUpdate(ApiModel):
    expected_review_version: StrictInt = Field(ge=1)
    expected_plan_version: StrictInt = Field(ge=1)
    reviewed_prompt: StrictStr = Field(min_length=1, max_length=32_000)
    selected: StrictBool


class PromptExpansionQueue(ApiModel):
    idempotency_key: StrictStr = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$",
    )
    expected_plan_version: StrictInt = Field(ge=1)
    expected_plan_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class PromptExpansionItemOut(ApiModel):
    id: str = Field(min_length=1, max_length=40)
    ordinal: int = Field(ge=1, le=16)
    rendered_prompt: str = Field(min_length=1, max_length=32_000)
    rendered_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_prompt: str = Field(min_length=1, max_length=32_000)
    reviewed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected: bool
    review_version: int = Field(ge=1)
    reroll_count: int = Field(ge=0)
    work_step_id: str | None = Field(default=None, min_length=1, max_length=40)
    run_id: str | None = Field(default=None, min_length=1, max_length=40)
    media_seed: int | None = Field(default=None, ge=0, lt=2_147_483_648)


class PromptExpansionBatchOut(ApiModel):
    id: str = Field(min_length=1, max_length=40)
    chat_id: str = Field(min_length=1, max_length=40)
    prompt_template_id: str = Field(min_length=1, max_length=40)
    prompt_template_revision_id: str = Field(min_length=1, max_length=40)
    schema_version: Literal[1]
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    codec_version: Literal[2, 3]
    requested_count: int = Field(ge=1, le=16)
    unfilled_ordinals: list[Annotated[int, Field(ge=1, le=16)]] = Field(
        default_factory=list, max_length=16
    )
    selection_seed: int = Field(ge=0, lt=2_147_483_648)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state: Literal["draft", "queued"]
    plan_version: int = Field(ge=1)
    queue_idempotency_key: str | None = Field(default=None, min_length=1, max_length=200)
    work_plan_id: str | None = Field(default=None, min_length=1, max_length=40)
    queued_at: datetime | None = None
    items: list[PromptExpansionItemOut] = Field(min_length=1, max_length=16)
    replayed: bool


class TurnReferenceIn(ApiModel):
    """One explicitly structured Reference attached to a turn."""

    reference_subject_id: str = Field(min_length=1, max_length=80)
    role: str | None = Field(default=None, min_length=1, max_length=MAX_ROLE)
    selected_asset_ids: list[str] = Field(default_factory=list, max_length=16)
    strength: float | None = Field(default=None, ge=0.0, le=2.0)
    source: MentionSource = MentionSource.MENTION


class PromptComposerSourceIn(ApiModel):
    """One reviewed Prompt Library item the composer was populated from."""

    version: Literal[1] = 1
    batch_id: str = Field(min_length=1, max_length=40)
    expected_plan_version: int = Field(ge=1)
    expected_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_id: str = Field(min_length=1, max_length=40)
    expected_review_version: int = Field(ge=1)
    expected_reviewed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_template_id: str = Field(min_length=1, max_length=40)
    prompt_template_revision_id: str = Field(min_length=1, max_length=40)
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TurnWorkflowDefaultIn(ApiModel):
    selector_capability: Literal["chat", "image", "video"]
    mode: Literal["default", "automatic"]


class TurnWorkflowFamilyIn(ApiModel):
    selector_capability: Literal["chat", "image", "video"]
    mode: Literal["family"]
    workflow_family_id: str = Field(min_length=1, max_length=64)


class TurnWorkflowRevisionIn(ApiModel):
    selector_capability: Literal["chat", "image", "video"]
    mode: Literal["revision"]
    workflow_revision_id: str = Field(min_length=1, max_length=40)


TurnWorkflowSelectionIn = Annotated[
    TurnWorkflowDefaultIn | TurnWorkflowFamilyIn | TurnWorkflowRevisionIn,
    Field(discriminator="mode"),
]


class TurnRoleOverrides(ApiModel):
    """Deliberate choices for whichever steps route to this role."""

    settings: dict[str, Any] = Field(default_factory=dict)
    preset_id: str | None = Field(default=None, min_length=1, max_length=40)
    profile_id: str | None = Field(default=None, min_length=1, max_length=40)
    vision_profile_id: str | None = Field(default=None, min_length=1, max_length=40)
    workflow_revision_id: str | None = Field(default=None, min_length=1, max_length=40)
    workflow_selection: TurnWorkflowSelectionIn | None = None


class TurnRequest(ApiModel):
    text: str = Field(min_length=1, max_length=200_000)
    preset_id: str | None = Field(default=None, min_length=1, max_length=40)
    profile_id: str | None = Field(default=None, min_length=1, max_length=40)
    vision_profile_id: str | None = Field(default=None, min_length=1, max_length=40)
    mode: RoutingMode | None = None
    parent_message_id: str | None = None
    input_artifact_ids: list[str] = Field(default_factory=list, max_length=16)
    references: list[TurnReferenceIn] = Field(
        default_factory=list, max_length=MAX_REFERENCES_PER_TURN
    )
    prompt_source: PromptComposerSourceIn | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    ordered_settings: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=3)
    role_overrides: dict[str, TurnRoleOverrides] = Field(default_factory=dict, max_length=3)
    output_count: int | None = Field(default=None, ge=1, le=16)
    # The workflow a recipe recorded. A recipe that stored which workflow made
    # a result and then ran against whichever one happens to be current is not
    # a recipe; it is the instruction with extra fields. An id that does not
    # match this operation, engine, or install is not honored - the turn
    # refuses rather than quietly substituting.
    workflow_revision_id: str | None = Field(default=None, max_length=40)
    workflow_selection: TurnWorkflowSelectionIn | None = None
    confirm_media: bool = False
    idempotency_key: str | None = Field(default=None, max_length=200)

    @field_validator("role_overrides")
    @classmethod
    def validate_role_overrides(
        cls, value: dict[str, TurnRoleOverrides]
    ) -> dict[str, TurnRoleOverrides]:
        for role, override in value.items():
            if role not in {"chat", "image", "video"}:
                raise ValueError("Turn overrides contain an unsupported role.")
            if override.workflow_selection is not None and (
                override.workflow_selection.selector_capability != role
            ):
                raise ValueError("Turn workflow selection has a different role.")
        return value

    def for_role(self, role: str, *, ordered: bool = False) -> Self:
        """Resolve one role without changing the routing request or another role."""
        override = self.role_overrides.get(role)
        if override is None and not ordered:
            return self
        values = override.model_dump(exclude_unset=True) if override is not None else {}
        if "workflow_selection" in values:
            values.setdefault("workflow_revision_id", None)
        elif "workflow_revision_id" in values:
            values["workflow_selection"] = None
        values["settings"] = {
            **(self.ordered_settings.get(role, {}) if ordered else self.settings),
            **values.get("settings", {}),
        }
        # Keep validated workflow models; model_copy deliberately does not reparse.
        if override is not None and "workflow_selection" in override.model_fields_set:
            values["workflow_selection"] = override.workflow_selection
        return self.model_copy(update=values)


#: How much a draft's template settings may hold. A template's settings are a
#: handful of numbers and names; this bounds a stored draft, not a real template.
MAX_DRAFT_TEMPLATE_SETTINGS = 64
MAX_DRAFT_TEMPLATE_SETTINGS_BYTES = 64_000


class ChatComposerDraftAttachmentIn(ApiModel):
    """One file attached to an unsent draft, as the composer shows it."""

    artifact_id: str = Field(min_length=1, max_length=80)
    kind: Literal["image", "video"]
    origin: Literal["uploaded", "generated", "edited"]


class ChatComposerDraftMentionIn(ApiModel):
    """One Reference mentioned in an unsent draft, and the text that names it."""

    reference_subject_id: str = Field(min_length=1, max_length=80)
    mention_slug: str = Field(min_length=1, max_length=80)


class ChatComposerDraftTemplateIn(ApiModel):
    """The one-click edit template applied to an unsent draft."""

    name: str = Field(min_length=1, max_length=240)
    settings: dict[str, Any] = Field(default_factory=dict, max_length=MAX_DRAFT_TEMPLATE_SETTINGS)


class ChatComposerDraftIn(ApiModel):
    """Everything an unsent message would be sent with, bounded like the send itself."""

    text: str = Field(default="", max_length=200_000)
    prompt_source: PromptComposerSourceIn | None = None
    mode: RoutingMode = RoutingMode.AUTO
    output_count: int = Field(default=1, ge=1, le=16)
    attachments: list[ChatComposerDraftAttachmentIn] = Field(default_factory=list, max_length=16)
    mentions: list[ChatComposerDraftMentionIn] = Field(
        default_factory=list, max_length=MAX_REFERENCES_PER_TURN
    )
    template_settings: ChatComposerDraftTemplateIn | None = None


class ChatComposerDraftWrite(ApiModel):
    """Replace a chat's draft, but only if it is still the revision the writer read."""

    expected_revision: int = Field(ge=0)
    draft: ChatComposerDraftIn


class ChatComposerDraftOut(ChatComposerDraftIn):
    """A chat's unsent draft. Revision 0 means the chat has never had one."""

    chat_id: str
    revision: int
    updated_at: datetime | None = None


class PriorTurnEditRequest(TurnRequest):
    """An exact retryable edit; omitted collection fields inherit the source."""

    step_overrides: dict[str, TurnRoleOverrides] = Field(default_factory=dict, max_length=64)

    idempotency_key: StrictStr = Field(min_length=1, max_length=200)
    source_run_id: str | None = Field(default=None, min_length=1, max_length=40)
    source_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PriorTurnEditConfiguration(ApiModel):
    image_edit_strength: dict[str, Any] | None = None
    operation: str
    profile_engine: str | None = None
    settings: dict[str, Any]
    resolved_settings: dict[str, Any]
    settings_role: str
    output_count: int
    profile_id: str | None
    vision_profile_id: str | None
    preset_id: str | None
    preset: dict[str, Any] | None
    model_selection: dict[str, Any]
    workflow_selection: WorkflowSelectionOut
    workflow_revision_id: str | None
    workflow_schema: dict[str, Any] | None
    profile_settings: dict[str, Any] = Field(default_factory=dict)


class PriorTurnEditStepSource(PriorTurnEditConfiguration):
    step_id: str
    ordinal: int
    source_run_id: str
    depends_on: list[str] = Field(default_factory=list)


class PriorTurnEditSource(PriorTurnEditConfiguration):
    source_user_message_id: str
    source_run_id: str
    source_snapshot_sha256: str
    chat_id: str
    text: str
    mode: RoutingMode
    original_mode: RoutingMode | None = None
    plan_kind: Literal["single", "ordered"] = "single"
    steps: list[PriorTurnEditStepSource] = Field(default_factory=list)
    input_artifact_ids: list[str]
    input_artifacts: list[ArtifactOut]
    references: list[MessageReferenceOut]
    context_messages: list[dict[str, str]]
    context_visual_artifacts: list[ArtifactOut] = Field(default_factory=list)
    prompt_source: dict[str, Any] | None


class PriorTurnEditBinding(ApiModel):
    source_message_id: str = Field(min_length=1, max_length=40)
    source_run_id: str = Field(min_length=1, max_length=40)
    source_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DraftClassificationRequest(ApiModel):
    """An unsent composer draft, classified with the router the turn will use."""

    text: str = Field(default="", max_length=200_000)
    mode: RoutingMode | None = None
    parent_message_id: str | None = None
    edit_source: PriorTurnEditBinding | None = None


class DraftClassification(ApiModel):
    references_prior_visual: bool


class VerifiedSetup(ApiModel):
    """A working setup, described so another machine can resolve it."""

    version: int
    role: str
    engine: str
    model: dict[str, Any]
    workflow: dict[str, Any] | None
    settings: dict[str, Any]
    hardware: dict[str, Any] | None
    attestation: dict[str, Any]
    digest: str


class ResolvedSetupComponent(ApiModel):
    target_folder: str
    sha256: str
    present: bool


class ResolvedSetup(ApiModel):
    """What an imported setup finds on this machine, and what it still needs."""

    version: int
    digest: str | None
    components: list[ResolvedSetupComponent]
    missing_components: list[dict[str, str]]
    hardware_compatible: bool
    # Provenance from the artifact, kept separate from anything earned here.
    verified_elsewhere: bool
    verified_here: bool
    requires_approval: bool
    ready_to_verify: bool


class TrustDerivation(ApiModel):
    """Whether this machine could vouch for a workflow by rebuilding it."""

    version: int
    trusted: bool
    reason: str
    message: str


class RegenerateRequest(ApiModel):
    settings: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: StrictStr | None = Field(default=None, min_length=1, max_length=200)


class RoutingReasonCode(StrEnum):
    EXPLICIT_TEXT_MODE = "explicit_text_mode"
    EXPLICIT_IMAGE_MODE = "explicit_image_mode"
    EXPLICIT_VIDEO_MODE = "explicit_video_mode"
    REPEAT_LAST_GENERATION = "repeat_last_generation"
    ASSISTANT_SUGGESTION_SELECTED = "assistant_suggestion_selected"
    DISCUSSION = "discussion"
    TEXT_EDIT = "text_edit"
    TEXT_MEDIA_TASK = "text_media_task"
    VIDEO_CREATION = "video_creation"
    PRIOR_IMAGE_EDIT = "prior_image_edit"
    IMAGE_CREATION = "image_creation"
    TEXT_TASK = "text_task"
    DEFAULT_TEXT = "default_text"
    MODEL_PLANNER = "model_planner"
    GENERATION_OFFER_ACCEPTED = "generation_offer_accepted"


class RoutingPlan(ApiModel):
    operation: Operation
    standalone_prompt: str
    # The chat passage this request is asking to depict, when it is asking for
    # one. Carried apart from `standalone_prompt` so a media prompt can be
    # compiled from the request and its source rather than their concatenation.
    text_context: str | None = None
    negative_prompt: str | None = None
    input_artifact_ids: list[str] = Field(default_factory=list)
    profile_id: str | None = None
    workflow_id: str | None = None
    # Cost projections computed at admission. These were previously stuffed
    # into a `parameter_overrides` dict under underscore-prefixed keys and
    # read by the browser through that internal marker; they are contract, so
    # they are named fields like the sibling ordered-plan 409 already used.
    generation_estimate: dict[str, Any] | None = None
    media_plan_estimate: dict[str, Any] | None = None
    output_count: int = Field(default=1, ge=1, le=16)
    confidence: float = Field(ge=0, le=1)
    reason_code: RoutingReasonCode
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
    status: RunStatus
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


class PriorTurnEditAccepted(TurnAccepted):
    source_message_id: str
    source_run_id: str
    work_plan_id: str
    branch_head_message_id: str
    branch_activated: Literal[False] = False
    accepted_context_sha256: str


class ProgressStageTiming(ApiModel):
    stage: str
    duration_ms: int = Field(ge=0)


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
    stage_started_at: datetime | None = None
    stage_elapsed_ms: int = Field(default=0, ge=0)
    completed_stages: list[ProgressStageTiming] = Field(default_factory=list)
    updated_at: datetime


class JobOut(ApiModel):
    id: str
    kind: JobKind
    status: JobStatus
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


class JobActivityOut(ApiModel):
    active: list[JobOut]
    active_count: int = Field(ge=0)
    recent_issues: list[JobOut]


class QueueLaneCountsOut(ApiModel):
    generation: int = Field(default=0, ge=0)
    transfer: int = Field(default=0, ge=0)
    install: int = Field(default=0, ge=0)


class QueueControlCommand(ApiModel):
    expected_revision: StrictInt = Field(ge=0, le=9_223_372_036_854_775_807)
    idempotency_key: StrictStr = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class QueueControlResultOut(ApiModel):
    owner_id: str
    control_state: Literal["eligible", "held"]
    control_revision: int = Field(ge=1)
    eligible_since: datetime | None


class GenerationQueuePolicyOut(ApiModel):
    lane: Literal["generation"]
    dispatch_state: Literal["open", "draining", "paused"]
    revision: int = Field(ge=0)
    running_jobs: int = Field(ge=0)
    allowed_actions: list[Literal["pause_after_current", "resume"]]


class TransferQueuePolicyOut(ApiModel):
    lane: Literal["transfer"]
    dispatch_state: Literal["open", "draining", "paused"]
    revision: int = Field(ge=0)
    running_jobs: int = Field(ge=0)
    allowed_actions: list[Literal["pause_after_current", "resume"]]


class QueueActivityItemOut(ApiModel):
    owner_type: Literal["work_plan", "job"]
    owner_id: str
    label: str
    lane: Literal["generation", "transfer", "install"]
    status: Literal["running", "queued", "paused", "blocked"]
    chat_id: str | None
    chat_title: str | None
    created_at: datetime
    updated_at: datetime
    step_count: int = Field(ge=0)
    completed_steps: int = Field(ge=0)
    blocked_steps: int = Field(ge=0)
    active_jobs: int = Field(ge=0)
    running_jobs: int = Field(ge=0)
    queued_jobs: int = Field(ge=0)
    paused_jobs: int = Field(ge=0)
    progress: float | None = Field(ge=0, le=1)

    control_state: Literal["eligible", "held"] | None = None
    control_revision: int | None = Field(default=None, ge=0)
    allowed_actions: list[Literal["hold", "release"]] = Field(default_factory=list)


class QueueActivityPageOut(ApiModel):
    items: list[QueueActivityItemOut]
    total: int = Field(ge=0)
    lane_counts: QueueLaneCountsOut
    next_cursor: str | None
    observed_at: datetime


class WorkStepImport(ApiModel):
    """Portable step input; unknown historical statuses normalize during import."""

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


WorkStepStatus = JobStatus | Literal["blocked"]
WorkPlanStatus = WorkStepStatus | Literal["partial"]


class QueueStepOut(ApiModel):
    id: str
    ordinal: int
    label: str
    status: WorkStepStatus
    blocked_by: int = Field(ge=0)
    progress: float | None = Field(default=None, ge=0, le=1)
    progress_scope: Literal["overall", "stage"] | None = None
    recorded_media_outputs: int | None = Field(default=None, ge=0)


class QueuePlanStepsOut(ApiModel):
    plan_id: str
    items: list[QueueStepOut]
    total: int = Field(ge=0)
    next_offset: int | None
    observed_at: datetime


class WorkStepOut(WorkStepImport):
    status: WorkStepStatus


class _WorkPlanFields(ApiModel):
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
    created_at: datetime
    updated_at: datetime


class WorkPlanImport(_WorkPlanFields):
    """Portable plan input, independent of the live response vocabulary."""

    steps: list[WorkStepImport]


class WorkPlanOut(_WorkPlanFields):
    status: WorkPlanStatus
    steps: list[WorkStepOut]


class EditedBranchOut(ApiModel):
    source_message_id: str
    source_run_id: str
    branch_head_message_id: str
    source_available: bool
    can_continue: bool
    plan: WorkPlanOut
    jobs: list[JobOut]


class EditedBranchPage(ApiModel):
    items: list[EditedBranchOut]
    next_cursor: str | None


class EditedBranchActivationRequest(ApiModel):
    expected_active_head_message_id: str | None = Field(max_length=40)


class EditedBranchActivationOut(ApiModel):
    chat_id: str
    active_head_message_id: str


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
    status: Literal["planned", "downloading", "activated", "failed", "cancelled"]
    failure_code: InstallPlanFailureCode | None
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime


class ModelCapabilityEvidenceOut(ApiModel):
    id: str
    model_install_id: str
    evidence_key: str
    component_hashes_json: dict[str, str]
    runtime_build: str
    adapter_contract_version: int
    launch_contract_version: str
    workflow_contract_version: str | None
    hardware_class: str
    probe_version: str
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


class ModelUpdateOut(ApiModel):
    """One installed asset's staleness verdict against its provider.

    `state` is "update_available", "current", or "unknown" - unknown means the
    provider could not answer or the comparison baseline is unavailable.
    Update fields are set only with
    "update_available"; installing the candidate goes through the normal
    verified catalog flow for its version id.
    """

    install_id: str
    name: str
    kind: InstalledAssetKind
    model_id: str
    installed_version_id: str
    installed_version_name: str | None
    state: Literal["update_available", "current", "unknown"]
    update_version_id: str | None = None
    update_version_name: str | None = None
    update_published_at: str | None = None
    update_base_model: str | None = None
    update_changelog: str | None = None


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
    request_settings: SavedRoleSettings = Field(default_factory=dict)
    is_default: bool = False


class ModelProfileModelUpdate(ApiModel):
    expected_install_id: str = Field(min_length=1, max_length=40)
    download_job_id: str = Field(min_length=1, max_length=40)


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
    use_case_derived: bool = False
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    model_install_id: str | None = None
    load_settings: dict[str, Any] = Field(default_factory=dict)
    request_settings: SavedRoleSettings = Field(default_factory=dict)


class ModelProfileOut(ApiModel):
    id: str
    model_install_id: str | None
    name: str
    use_case: str
    use_case_derived: bool = False
    role: str
    engine: str
    load_settings_json: dict[str, Any]
    request_settings_json: SavedRoleSettings
    is_default: bool
    input_modalities: list[str] = Field(default_factory=lambda: ["text"])
    created_at: datetime
    updated_at: datetime


class PresetCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    settings: SavedRoleSettings = Field(default_factory=dict)
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
    settings: SavedRoleSettings = Field(default_factory=dict)


class PresetOut(ApiModel):
    id: str
    name: str
    role: str
    settings_json: SavedRoleSettings
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


class WorkflowRevisionCreate(ApiModel):
    engine_version: str | None = None
    ui_graph: dict[str, Any] = Field(default_factory=dict)
    api_graph: dict[str, Any]
    input_schema: dict[str, Any] = Field(default_factory=dict)
    dependencies: dict[str, Any] = Field(default_factory=dict)


class ModelAssetAdopt(ApiModel):
    """A file already in the runtime's folder, offered for registration."""

    kind: InstalledAssetKind
    comfy_name: str
    name: str | None = Field(default=None, min_length=1, max_length=300)
    family: str | None = Field(default=None, min_length=1, max_length=100)
    use_case: str | None = Field(default=None, max_length=10_000)


class WorkflowRevisionReviewRequest(ApiModel):
    action: Literal["approve", "revoke"]
    subject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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


class StudioToolCapability(ApiModel):
    """Whether one studio tool can run here, and what would fix it."""

    kind: StudioToolKind
    workflow_class: str
    available: bool
    reason: str | None
    workflow_revision_id: str | None = None
    adapter_asset_id: str | None = None


class StudioCapabilityReport(ApiModel):
    tools: list[StudioToolCapability]


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
    dependency_contract_sha256: str | None = None
    trusted: bool
    created_at: datetime


# The ratio presets a chat can offer. The ids are the ratios they name, and the
# workflow decides which of them it can express exactly.
WorkflowOutputGeometryPresetId = Literal["1:1", "3:4", "2:3", "9:16", "4:3", "3:2", "16:9"]
WorkflowOutputGeometryOperation = Literal["text_to_image", "text_to_video", "image_to_video"]


class WorkflowOutputGeometryBindingOut(ApiModel):
    key: Literal["width", "height"]
    node_id: str
    input_name: Literal["width", "height"]
    default: int
    minimum: int
    maximum: int
    multiple_of: int


class WorkflowOutputGeometryCapabilityOut(ApiModel):
    version: Literal[1]
    available: bool
    reason: Literal["unsupported_workflow_geometry"] | None
    revision_id: str | None
    workflow_id: str | None
    artifact_sha256: str | None
    operation: WorkflowOutputGeometryOperation | None
    engine: Literal["comfyui"] | None
    size_modes: list[Literal["exact", "preset"]]
    preset_ids: list[WorkflowOutputGeometryPresetId]
    width: WorkflowOutputGeometryBindingOut | None
    height: WorkflowOutputGeometryBindingOut | None
    latent_node_id: str | None
    sampler_node_ids: list[str]
    decode_node_ids: list[str]
    save_node_ids: list[str]
    capability: dict[str, Any] | None
    graph_binding_verified: bool
    request_authorized: Literal[False]


class WorkflowOutputGeometryResolutionOut(ApiModel):
    version: Literal[1]
    workflow_id: str
    revision_id: str
    artifact_sha256: str
    operation: WorkflowOutputGeometryOperation
    engine: Literal["comfyui"]
    mode: Literal["image", "video"]
    size_mode: Literal["exact", "preset"]
    preset_id: WorkflowOutputGeometryPresetId | None
    width: int
    height: int
    graph_binding_verified: Literal[True]
    request_authorized: Literal[False]


class WorkflowSummaryOut(ApiModel):
    id: str
    family_id: str | None = None
    name: str
    operation: str
    description: str
    current_revision_id: str | None
    revision_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class WorkflowRevisionChoiceOut(ApiModel):
    revision_id: str
    workflow_id: str
    workflow_name: str
    operation: str
    version: int


class WorkflowRevisionSchemaOut(ApiModel):
    revision_id: str
    workflow_id: str
    operation: str
    input_schema_json: dict[str, Any]


WorkflowLoraEditability = Literal[
    "editable",
    "required_locked",
    "detected_read_only",
]
WorkflowLoraStrengthMode = Literal["separate", "coupled", "model_only", "unknown"]
WorkflowLoraEditableField = Literal["enabled", "model_strength", "clip_strength"]
WorkflowLoraEvidenceGap = Literal[
    "dependency_contract_unavailable",
    "dependency_contract_invalid",
    "active_activation_unavailable",
    "active_activation_invalid",
    "ui_graph_provenance_unavailable",
    "core_runtime_evidence_unavailable",
    "core_graph_binding_unavailable",
    "package_binding_evidence_unavailable",
    "package_graph_binding_unavailable",
]


class WorkflowLoraAssetBindingOut(ApiModel):
    dependency_slot: str = Field(min_length=1, max_length=100)
    requirement_key: str = Field(min_length=1, max_length=100)
    resource_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_reference: str = Field(min_length=1, max_length=1_000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkflowLoraControlSlotOut(ApiModel):
    slot_id: str = Field(pattern=r"^wflora_[0-9a-f]{64}$")
    position: int = Field(ge=0, lt=64)
    loader_type: str = Field(min_length=1, max_length=200)
    loader_contract: str | None = Field(default=None, max_length=200)
    loader_authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    editability: WorkflowLoraEditability
    read_only_reason: str | None = Field(default=None, max_length=100)
    dependency_required: bool | None
    observed_runtime_reference: str | None = Field(default=None, max_length=1_000)
    asset_binding: WorkflowLoraAssetBindingOut | None
    default_enabled: bool | None
    default_model_strength: float | None
    default_clip_strength: float | None
    strength_mode: WorkflowLoraStrengthMode
    editable_fields: list[WorkflowLoraEditableField] = Field(max_length=3)


class WorkflowLoraStrengthBoundsOut(ApiModel):
    minimum: float
    maximum: float


class WorkflowLoraOverrideTargetWitnessOut(ApiModel):
    workflow_family_id: str | None = Field(min_length=1, max_length=64)
    workflow_definition_id: str = Field(min_length=1, max_length=64)
    workflow_variant_key: str | None = Field(min_length=1, max_length=100)
    workflow_revision_id: str = Field(min_length=1, max_length=40)
    slot_contract_version: Literal[1]
    revision_scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    api_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    activation_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    activation_witness_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkflowLoraControlsOut(ApiModel):
    version: Literal[1]
    override_contract_version: Literal[1]
    strength_bounds: WorkflowLoraStrengthBoundsOut
    override_target: WorkflowLoraOverrideTargetWitnessOut | None
    revision_scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    api_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    activation_binding_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    ordering_authority: Literal["presentation_only"]
    evidence_gaps: list[WorkflowLoraEvidenceGap] = Field(max_length=9)
    base_model_family: str | None = Field(default=None, max_length=64)
    slots: list[WorkflowLoraControlSlotOut] = Field(max_length=64)


class WorkflowOut(ApiModel):
    id: str
    family_id: str | None = None
    name: str
    operation: str
    description: str
    current_revision_id: str | None
    revisions: list[WorkflowRevisionOut]
    created_at: datetime
    updated_at: datetime


WorkflowSelectorCapability = Literal["chat", "vision", "image", "video"]
#: ResourceKind itself, not a restatement of it. The models that consume this
#: alias use ApiModel, whose config is not strict, so the enum validates the
#: plain strings the wire carries. An earlier revision kept a hand-maintained
#: Literal here and claimed the copy was unavoidable; a focused compatibility
#: probe proved otherwise. Only the strict parser in workflow_dependencies still
#: needs its own Literal. See test_closed_storage_vocabularies.
WorkflowDependencyResourceKind = ResourceKind
WorkflowVariantReadiness = Literal[
    "ready",
    "setup_required",
    "review_required",
    "unavailable",
]
WorkflowSelectionResponseMode = Literal[
    "default",
    "inherit",
    "automatic",
    "family",
    "revision",
    "legacy",
]


class WorkflowFamilyVariantOut(ApiModel):
    id: str
    variant_key: str
    name: str
    operation: Operation
    current_revision_id: str | None
    current_revision_version: int | None
    engine: str | None
    capabilities: list[str] = Field(default_factory=list)
    trusted: bool
    readiness: WorkflowVariantReadiness
    readiness_reason: str | None = None
    setup_resolution: Literal["reviewed_download_available", "attention_required"] | None = None
    install_offer: WorkflowInstallOfferOut | None = None
    install_progress: WorkflowInstallProgressOut | None = None


class WorkflowFamilyPreferenceOut(ApiModel):
    selector_capability: WorkflowSelectorCapability
    enabled: bool
    is_default: bool
    sort_order: int


class WorkflowFamilyDependencySummaryOut(ApiModel):
    dependency_count: int = Field(ge=0)
    names: list[str] = Field(default_factory=list)


class WorkflowFamilyOut(ApiModel):
    id: str
    name: str
    description: str
    use_case: str
    use_case_derived: bool = False
    tags: list[str] = Field(default_factory=list)
    enabled: bool
    archived: bool
    compatibility: bool
    variants: list[WorkflowFamilyVariantOut] = Field(default_factory=list)
    preferences: list[WorkflowFamilyPreferenceOut] = Field(default_factory=list)
    dependency_summary: WorkflowFamilyDependencySummaryOut | None = None
    created_at: datetime
    updated_at: datetime


class WorkflowFamilyUpdate(ApiModel):
    name: str | None = Field(default=None, min_length=1, max_length=240)
    description: str | None = Field(default=None, max_length=10_000)
    use_case: str | None = Field(default=None, max_length=10_000)
    tags: list[str] | None = Field(default=None, max_length=100)
    enabled: bool | None = None
    archived: bool | None = None


class WorkflowFamilyPreferenceUpdate(ApiModel):
    enabled: bool = True
    is_default: bool = False
    sort_order: int = Field(default=0, ge=-1_000_000, le=1_000_000)


class WorkflowDependencyImpactOut(ApiModel):
    resource_kind: ResourceKind
    resource_id: str
    resource_name: str
    binding_count: int
    revision_count: int
    current_revision: bool
    shared: bool
    other_workflow_count: int
    other_family_ids: list[str] = Field(default_factory=list)


class WorkflowFamilyRemovalImpactOut(ApiModel):
    family_id: str
    removal_strategy: Literal["archive"] = "archive"
    archive_blocked: bool
    revision_count: int
    current_revision_count: int
    chat_selection_count: int
    project_selection_count: int
    project_revision_pin_count: int
    active_run_count: int
    queued_step_count: int
    historical_run_count: int
    active_activation_count: int
    default_for: list[WorkflowSelectorCapability] = Field(default_factory=list)
    dependencies: list[WorkflowDependencyImpactOut] = Field(default_factory=list)


class WorkflowResourceConsumerOut(ApiModel):
    workflow_id: str
    workflow_name: str
    workflow_family_id: str | None = None
    workflow_family_name: str | None = None
    revision_ids: list[str] = Field(default_factory=list)
    binding_count: int
    current_revision: bool


class WorkflowResourceConsumersOut(ApiModel):
    resource_kind: WorkflowDependencyResourceKind
    resource_id: str
    resource_name: str
    consumers: list[WorkflowResourceConsumerOut] = Field(default_factory=list)


class ChatWorkflowDefaultSelectionIn(ApiModel):
    mode: Literal["default"]


class ChatWorkflowAutomaticSelectionIn(ApiModel):
    mode: Literal["automatic"]


class ChatWorkflowFamilySelectionIn(ApiModel):
    mode: Literal["family"]
    workflow_family_id: str = Field(min_length=1, max_length=64)


ChatWorkflowSelectionIn = Annotated[
    ChatWorkflowDefaultSelectionIn
    | ChatWorkflowAutomaticSelectionIn
    | ChatWorkflowFamilySelectionIn,
    Field(discriminator="mode"),
]


class ProjectWorkflowInheritSelectionIn(ApiModel):
    mode: Literal["inherit"]


class ProjectWorkflowAutomaticSelectionIn(ApiModel):
    mode: Literal["automatic"]


class ProjectWorkflowFamilySelectionIn(ApiModel):
    mode: Literal["family"]
    workflow_family_id: str = Field(min_length=1, max_length=64)


class ProjectWorkflowRevisionSelectionIn(ApiModel):
    mode: Literal["revision"]
    workflow_revision_id: str = Field(min_length=1, max_length=40)


ProjectWorkflowSelectionIn = Annotated[
    ProjectWorkflowInheritSelectionIn
    | ProjectWorkflowAutomaticSelectionIn
    | ProjectWorkflowFamilySelectionIn
    | ProjectWorkflowRevisionSelectionIn,
    Field(discriminator="mode"),
]


class WorkflowSelectionOut(ApiModel):
    selector_capability: WorkflowSelectorCapability
    mode: WorkflowSelectionResponseMode
    workflow_family_id: str | None = None
    workflow_revision_id: str | None = None
    legacy_profile_id: str | None = None


class WorkflowOpenTarget(ApiModel):
    url: str
    filename: str
    ui_graph: dict[str, Any]


class WorkflowEditorSessionOut(ApiModel):
    id: str
    protocol_version: int
    workflow_id: str
    base_revision_id: str
    base_graph_sha256: str
    base_prompt_sha256: str
    created_at: datetime
    expires_at: datetime
    ui_graph: dict[str, Any]
    nonce: str


class WorkflowEditorCancelIn(ApiModel):
    nonce: str = Field(min_length=1, max_length=200)


class WorkflowEditorConsumeIn(ApiModel):
    nonce: str = Field(min_length=1, max_length=200)
    base_revision_id: str = Field(min_length=1, max_length=40)
    ui_graph: dict[str, Any]
    api_prompt: dict[str, Any]


class WorkflowEditorGraphDeltaOut(ApiModel):
    node_count_delta: int
    link_count_delta: int
    added_node_types: list[str]
    removed_node_types: list[str]
    added_asset_filenames: list[str]
    removed_asset_filenames: list[str]


class WorkflowEditorReturnOut(ApiModel):
    validated_return_id: str
    session_id: str
    workflow_id: str
    base_revision_id: str
    current_revision_id: str
    base_graph_sha256: str
    returned_graph_sha256: str
    base_prompt_sha256: str
    returned_prompt_sha256: str
    changed: bool
    forked: bool
    delta: WorkflowEditorGraphDeltaOut
    expires_at: datetime


class WorkflowEditorDraftCreateIn(ApiModel):
    validated_return_id: str = Field(min_length=1, max_length=200)


class WorkflowEditorDraftOut(ApiModel):
    workflow_id: str
    base_revision_id: str
    draft_revision_id: str
    current_revision_id: str | None
    version: int
    created: bool
    forked: bool
    trusted: Literal[False]
    review_required: Literal[True]


class EditTemplateCreate(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)
    instruction: str = Field(min_length=1, max_length=20_000)
    settings_json: dict[str, Any] = Field(default_factory=dict)
    # When given, the recipe is read from what this run actually did rather
    # than from whatever is current when Save is pressed.
    from_run_id: str | None = Field(default=None, max_length=40)


class EditTemplateOut(ApiModel):
    id: str
    name: str
    description: str
    instruction: str
    operation: str
    settings_json: dict[str, Any]
    workflow_revision_id: str | None
    model_profile_id: str | None
    mask_mode: MaskMode
    trigger_words_json: list[str]
    content_rating: ContentRating
    builtin: bool
    enabled: bool


class RegistryInstallReviewOut(ApiModel):
    """What staging found, so that trusting a package can be an informed act.

    Trust is what lets this code run. Asking someone to confirm they reviewed
    a package while showing them nothing to review makes the confirmation a
    formality, so these are the things that decide the answer: code that runs
    on install, code that runs at startup, compiled binaries, and the files
    that declare what else gets pulled in.
    """

    file_count: int
    expanded_bytes: int
    python_file_count: int
    install_scripts: list[str] = Field(default_factory=list, max_length=64)
    startup_hooks: list[str] = Field(default_factory=list, max_length=64)
    native_files: list[str] = Field(default_factory=list, max_length=64)
    dependency_manifests: list[str] = Field(default_factory=list, max_length=64)
    top_level_entries: list[str] = Field(default_factory=list, max_length=64)
    # Carried from the resolution: why this package needs looking at, in the
    # resolver's words rather than restated here.
    registry_warnings: list[str] = Field(default_factory=list, max_length=32)


class RegistryInstallOut(ApiModel):
    """One prepared package and the two explicit decisions it is waiting for."""

    id: str
    package_id: str
    package_version: str
    node_types: list[str]
    archive_sha256: str
    manifest_sha256: str
    wheel_closure_sha256: str | None
    wheel_environment_sha256: str | None
    disk_status: Literal[
        "ready",
        "node_files_missing",
        "wheel_environment_missing",
        "files_missing",
    ]
    node_files_present: bool
    wheel_environment_present: bool
    trusted: bool
    active: bool
    reviewed_at: str | None
    activated_at: str | None
    review: RegistryInstallReviewOut | None = None


class RegistryInstallReviewRequest(ApiModel):
    trusted: bool


class WorkflowAssetSelectionIn(ApiModel):
    """One explicit choice: this missing file comes from that plan artifact."""

    reference_filename: str = Field(min_length=1, max_length=1_000)
    install_plan_id: str = Field(min_length=1, max_length=40)
    artifact_path: str = Field(min_length=1, max_length=1_000)


class WorkflowAssetReviewRequest(ApiModel):
    """Review selections against a freshly re-analyzed graph.

    The browser sends the graph and its choices - never a report, digest,
    size, kind, or bound asset. Everything else is rebuilt server-side.
    """

    ui_graph: dict[str, Any]
    selections: list[WorkflowAssetSelectionIn] = Field(default_factory=list, max_length=64)


class WorkflowAssetQueueRequest(WorkflowAssetReviewRequest):
    """Queue exactly the reviewed binding, confirmed by its hash."""

    binding_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class BoundWorkflowAssetOut(ApiModel):
    reference_filename: str
    kind: BoundWorkflowAssetKind
    install_plan_id: str
    install_plan_hash: str
    provider: str
    remote_id: str
    revision: str
    artifact_path: str
    artifact_kind: InstalledAssetKind
    target_folder: str
    size_bytes: int
    sha256: str


class WorkflowAssetReviewOut(ApiModel):
    """What the browser may show: the binding, its hash, and the cost."""

    binding_plan_hash: str
    assets: list[BoundWorkflowAssetOut]
    download_count: int
    total_bytes: int


WorkflowInstallStatus = Literal["ready", "queued", "invalidated", "completed", "expired"]
WorkflowInstallAttentionCode = Literal[
    "download-acceptance-unavailable",
    "download-acceptance-changed",
    "download-result-unavailable",
    "download-result-changed",
    "download-plan-changed",
    "workflow-download-failed",
    "workflow-dependencies-need-selection",
    "download-results-need-binding",
    "workflow-install-offer-changed",
    "workflow-review-required",
    "workflow-completion-unavailable",
    "workflow-runtime-plan-unavailable",
    "workflow-runtime-plan-changed",
    "workflow-extension-review-required",
    "workflow-media-restore-failed",
    "workflow-install-cancelled",
]


WorkflowInstallPhase = Literal[
    "ready",
    "downloading",
    "paused",
    "verifying",
    "needs_attention",
    "completed",
    "invalidated",
    "expired",
]


class WorkflowInstallProgressOut(ApiModel):
    id: str
    workflow_revision_id: str
    status: WorkflowInstallStatus
    phase: WorkflowInstallPhase
    total_downloads: int
    completed_downloads: int
    failed_downloads: int
    cancelled_downloads: int
    paused_downloads: int
    pending_downloads: int
    unavailable_downloads: int
    attention_code: WorkflowInstallAttentionCode | None
    # The stopped installation job a retry would resume, when there is one.
    retry_job_id: str | None = None


class WorkflowInstallOfferCreate(ApiModel):
    """Explicit plan choices for one persisted workflow revision."""

    selections: list[WorkflowAssetSelectionIn] = Field(
        min_length=1,
        max_length=64,
    )


class WorkflowInstallOfferOut(ApiModel):
    id: str
    workflow_revision_id: str
    workflow_artifact_sha256: str
    dependency_contract_sha256: str
    binding_plan_sha256: str
    offer_sha256: str
    assets: list[BoundWorkflowAssetOut]
    plan_count: int
    total_bytes: int
    status: Literal["ready", "queued", "invalidated", "completed", "expired"]
    queued_at: datetime | None
    completed_at: datetime | None
    invalidated_at: datetime | None
    invalidation_code: str | None
    invalidation_reason: str | None


class WorkflowPackageImportRequest(ApiModel):
    """Import a fully resolved ComfyUI package as an untrusted workflow.

    The user confirms the name and operation - the analyzer's guess prefills
    the form, but nothing is silently decided for them.
    """

    ui_graph: dict[str, Any]
    name: str = Field(min_length=1, max_length=240)
    operation: Operation
    description: str = Field(default="", max_length=10_000)
    # Only an explicit portable declaration supplies dependency slots. An
    # omitted contract remains unknown rather than declaring no dependencies.
    dependencies: dict[str, Any] = Field(default_factory=dict)
    # A package that needed preparation is first persisted as a deliberately
    # non-executable revision. Supplying both identities lets import finalize
    # that exact draft instead of creating a second, unrelated workflow.
    draft_workflow_id: str | None = Field(default=None, max_length=64)
    draft_revision_id: str | None = Field(default=None, max_length=40)


class WorkflowPackageDraftRequest(ApiModel):
    """Persist one exact source graph before resolving its dependencies."""

    ui_graph: dict[str, Any]
    name: str = Field(min_length=1, max_length=240)
    operation: Operation
    description: str = Field(default="", max_length=10_000)


class WorkflowPackagePrepareRequest(ApiModel):
    # Preparation binds to one exact identity; an unpinned request has nothing
    # to verify against and the resolver refuses it anyway.
    package_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=200)
    # The server re-analyzes the source graph and derives the exact node types.
    # A browser-provided node list would only be another unverified claim.
    ui_graph: dict[str, Any]
    # Required before an omitted source dependency can be recorded against the
    # install. Re-analyzing a submitted graph proves that graph is internally
    # consistent; it does not bind it to anything stored, and a proof about a
    # graph nobody saved is a proof about nothing.
    workflow_revision_id: str | None = Field(default=None, max_length=40)


class WorkflowPackageAnalyzeRequest(ApiModel):
    ui_graph: dict[str, Any]


class WorkflowPackageRequirementOut(ApiModel):
    package_id: str
    versions: list[str]
    node_types: list[str]
    locally_resolved: bool


class WorkflowSourceCandidateOut(ApiModel):
    """One source link the package author wrote down, already validated.

    A suggestion of what to preflight - never a download instruction. The
    normal immutable-plan path still resolves it, and the browser cannot
    substitute a URL of its own.
    """

    provider: str
    remote_id: str
    revision: str | None
    filename: str | None
    url: str


# What the workflow analyzer can say a referenced file is. One definition, so
# a caller naming an exact file cannot name a kind the analyzer never emits.
AuxiliaryAssetKind = Literal[
    "lora",
    "vae",
    "controlnet",
    "upscaler",
    "embedding",
    "ip_adapter",
]


class WorkflowAssetReferenceOut(ApiModel):
    filename: str
    suffix: str
    policy: Literal["supported", "blocked", "unsupported"]
    kind: WorkflowAssetKind
    source_url: str | None
    present_locally: bool
    # Populated only when the author's own text names this exact file; a link
    # that names no file is reported once in the analysis instead of guessed
    # onto an asset here.
    source_candidates: list[WorkflowSourceCandidateOut] = []


class WorkflowPackageIssueOut(ApiModel):
    code: WorkflowPackageIssueCode
    count: int
    node_types: list[str]
    severity: Literal["blocking", "advisory"]


class WorkflowMissingNodeOut(ApiModel):
    node_type: str
    count: int
    package_id: str | None


class WorkflowPackageAnalysisOut(ApiModel):
    """The analyzer report, field names frozen with the analyzer.

    `ready` is the one trust/activation gate the browser obeys; it is computed
    by the analyzer, never re-derived client-side from list emptiness.
    `node_inventory_available` is this endpoint's own honesty flag: when the
    media runtime cannot enumerate its nodes, `missing_node_types` covers every
    runtime node and must be presented as "unknown", not as "missing".
    """

    format_version: str
    frontend_version: str | None
    node_count: int
    link_count: int
    subgraph_count: int
    operation_guess: Literal["image", "unknown", "video"]
    truncated: bool
    required_node_types: list[str]
    frontend_node_types: list[str]
    missing_node_types: list[str]
    missing_nodes: list[WorkflowMissingNodeOut]
    custom_packages: list[WorkflowPackageRequirementOut]
    asset_references: list[WorkflowAssetReferenceOut]
    issues: list[WorkflowPackageIssueOut]
    ready: bool
    runtime_nodes_available: bool
    dependencies_resolved: bool
    # Links the author recorded that name no particular file. Most authors
    # write display names rather than filenames, so this is the common case,
    # and it is offered for the user to assign rather than matched by guess.
    source_candidates: list[WorkflowSourceCandidateOut] = []
    node_inventory_available: bool


class CustomNodeInstallRequest(ApiModel):
    name: str = Field(min_length=1, max_length=240)
    source_url: str = Field(min_length=1, max_length=1000)
    revision: str = Field(min_length=40, max_length=40)


class CustomNodeUpdateRequest(ApiModel):
    revision: str = Field(min_length=40, max_length=40)


class CustomNodeTrustRequest(ApiModel):
    trusted: bool
    node_types: list[str] = Field(default_factory=list, max_length=4_096)


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
    # A CivitAI card is one *version*, because a version is what installs and
    # what the download path is bound to. These say which model it is a version
    # of, so the library can list versions under one parent without giving up
    # the version identity that install depends on. Absent for providers where
    # the repository is already the installable thing, and a card with no
    # parent renders exactly as it does now.
    parent_model_id: str | None = None
    parent_model_name: str | None = None
    # How many versions this card stands for. One means the card is the whole
    # story; more means it must open a chooser rather than install, because
    # the point of version identity is that the person picked one.
    version_count: int = 1
    # How many of them are already here, or `None` when that cannot be known.
    # Only kinds that record a provider version can answer; a checkpoint
    # install stores none, so "how many are installed" has no truthful number
    # and a guess of zero invites reinstalling what is already on disk.
    installed_version_count: int | None = None
    # Set only on workflow catalog cards: one repository can ship several
    # official workflows, and the card must say which one it is.
    workflow_template_id: str | None = None
    operation: str | None = None
    # Neutral provider-declared rating. The public app renders nothing from
    # it (general-only source); it exists so install provenance is honest and
    # downstream consumers inherit one labeling mechanism.
    content_rating: ContentRating = "unknown"


class CatalogVersionRow(ApiModel):
    """One installable version of a catalogue model, as the chooser sees it."""

    version_id: str
    version_name: str | None = None
    published_at: str | None = None
    base_model: str | None = None
    size_bytes: int = 0
    changelog: str | None = None
    # True for an exact recorded installation. An unmatched version is false
    # only when this model has another recorded version identity; otherwise
    # it remains unknown. This applies to checkpoints and auxiliary assets.
    installed: bool | None = None
    installed_as: str | None = None


class CatalogVersions(ApiModel):
    model_id: str
    model_name: str | None = None
    versions: list[CatalogVersionRow] = Field(default_factory=list)


class CatalogPage(ApiModel):
    items: list[CatalogModel]
    next_cursor: str | None = None
    stale: bool = False


class LoraSuggestionsOut(ApiModel):
    """Well-rated general-audience LoRAs for the model family a workflow runs."""

    family: str | None = Field(default=None, max_length=64)
    gap: Literal["family_unknown", "family_unsupported"] | None = None
    items: list[CatalogModel] = Field(default_factory=list, max_length=12)
    next_cursor: str | None = None
    stale: bool = False


class WorkflowCatalogGraphOut(ApiModel):
    """A discovered workflow's graph, fetched so it can be reviewed like a local file."""

    version_id: str
    ui_graph: dict[str, Any]


class CatalogDetail(ApiModel):
    model: CatalogModel
    revision: str
    files: list[dict[str, Any]]


class CatalogFileVariant(ApiModel):
    """One immutable choice behind an ambiguous filename.

    Everything here is the server's: the identity, the name, and the size come
    from a freshly fetched version detail. No URL and no hash, because the
    browser has no business carrying either - it names a choice, and the
    planner re-resolves it.
    """

    source_file_id: str
    filename: str
    size_bytes: int | None
    precision: str | None


class CatalogPreflightRequest(ApiModel):
    revision: str = Field(default="main", min_length=1, max_length=200)
    role: Literal["chat", "image", "video"]
    engine: str = Field(min_length=1, max_length=32)
    selected_files: list[str] = Field(default_factory=list, max_length=512)
    # Immutable provider file identities, for the case a filename cannot
    # settle: one CivitAI version can publish the same safetensors name five
    # times at different precisions, and preflight rightly refuses to guess.
    # It used to ask the caller to choose a variant and give it no way to say
    # which. Filename-only callers are unchanged.
    selected_file_ids: list[str] = Field(default_factory=list, max_length=512)
    # The exact workflow variant the user chose from the catalog. Absent for
    # repository-only callers, which keep the ranked fallback.
    workflow_template_id: str | None = Field(default=None, max_length=200)
    # A workflow named this exact file, so plan that file and nothing else.
    # Absent means an ordinary repository install, which keeps template
    # ranking. Present means the caller already knows what it needs, and
    # ranking a repository's official bundle over it would install several
    # gigabytes nobody asked for.
    workflow_reference_kind: WorkflowAssetKind | None = None
    auxiliary_kind: AuxiliaryAssetKind | None = None


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
    # CivitAI identity; the download manager derives its URL from these
    # server-side and never consumes a catalog-supplied one.
    source_version_id: str | None = Field(default=None, pattern=r"^[1-9][0-9]{0,11}$")
    source_file_id: str | None = Field(default=None, pattern=r"^[1-9][0-9]{0,11}$")


HardwareFitStatus = Literal["recommended", "likely", "tight", "unsupported", "unknown"]
HardwareFitBasis = Literal["unknown", "calculated", "declared", "measured", "tested", "certified"]


class HardwareFitReasonOut(ApiModel):
    code: Literal[
        "accelerator_backend_missing",
        "accelerator_memory_below_minimum",
        "accelerator_memory_busy",
        "accelerator_memory_declared",
        "accelerator_memory_estimated",
        "accelerator_memory_measured",
        "accelerator_memory_unknown",
        "accelerator_missing",
        "architecture_unsupported",
        "cpu_capabilities_unknown",
        "cpu_capability_missing",
        "evidence_stale",
        "platform_unsupported",
        "runtime_backend_missing",
        "system_memory_below_minimum",
        "system_memory_busy",
        "system_memory_declared",
        "system_memory_estimated",
        "system_memory_measured",
        "system_memory_unknown",
    ]
    severity: Literal["info", "warning", "block"]
    message: str


class HardwareFitAlternativeOut(ApiModel):
    code: Literal[
        "choose_compatible_backend",
        "choose_cpu_compatible_variant",
        "choose_smaller_variant",
        "free_current_memory",
        "install_supported_runtime",
        "use_safer_settings",
    ]
    message: str


class HardwareFitResourceOut(ApiModel):
    kind: Literal["system", "accelerator"]
    capacity_bytes: int = Field(ge=0)
    available_bytes: int | None = Field(ge=0)
    required_bytes: int = Field(ge=0)
    status: HardwareFitStatus
    basis: HardwareFitBasis
    immediate_pressure: bool


class HardwareFitSettingOut(ApiModel):
    key: str
    label: str
    unit: str
    minimum: int
    maximum: int
    advisory_only: bool
    preserves_user_override: bool


class HardwareFitAdviceOut(ApiModel):
    status: HardwareFitStatus
    basis: HardwareFitBasis
    evidence_label: Literal["tested", "certified"] | None
    reasons: list[HardwareFitReasonOut]
    alternatives: list[HardwareFitAlternativeOut]
    resources: list[HardwareFitResourceOut]
    settings: list[HardwareFitSettingOut]


class CatalogHardwareAlternative(ApiModel):
    selected_files: list[str]
    download_bytes: int = Field(ge=0)
    download_size_complete: bool
    hardware_fit: HardwareFitAdviceOut


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
    download_size_complete: bool = False
    available_disk_bytes: int
    estimated_ram_bytes: int | None = None
    estimated_vram_bytes: int | None = None
    hardware_fit: HardwareFitAdviceOut | None = None
    hardware_alternatives: list[CatalogHardwareAlternative] = Field(default_factory=list)
    can_install: bool
    checks: list[CatalogPreflightCheck]
    install_plan: InstallPlanOut | None = None
    auxiliary_kind: AuxiliaryAssetKind | None = None
    # The choices behind any filename this version could not settle, so a
    # refusal arrives with the answer to it. Asking someone to pick a variant
    # and then making them go and find the variants is not a choice, it is a
    # riddle.
    file_variants: dict[str, list[CatalogFileVariant]] = Field(default_factory=dict)
    # Copied server-side from the catalog detail, never client-supplied.
    content_rating: ContentRating = "unknown"


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
    content_rating: ContentRating = "unknown"
    default_settings: dict[str, Any] = Field(default_factory=dict)
    auxiliary_kind: AuxiliaryAssetKind | None = None
    # A dependency owned by one reviewed workflow binding. Unlike an
    # auxiliary asset it is never offered for auto-application or activated as
    # a standalone profile.
    workflow_asset_kind: InstalledAssetKind | None = None


class ModelAssetOut(ApiModel):
    id: str
    source_id: str | None
    name: str
    kind: InstalledAssetKind
    family: str | None
    size_bytes: int
    manifest_json: dict[str, Any]
    active: bool
    use_case: str
    auto_apply: bool
    default_model_strength: float
    default_clip_strength: float
    verified_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ModelAssetUpdate(ApiModel):
    active: bool | None = None
    #: The base model the asset is for. An empty string clears it; a file
    #: registered without declaring one otherwise never gets one.
    family: str | None = Field(default=None, max_length=100)
    use_case: str | None = Field(default=None, max_length=1_000)
    auto_apply: bool | None = None
    default_model_strength: float | None = Field(default=None, ge=-4, le=4)
    default_clip_strength: float | None = Field(default=None, ge=-4, le=4)


class AdapterPromptGrammarReview(ApiModel):
    """A reviewer recording how one adapter must be prompted.

    `verified_values` is the reviewer asserting what they have *seen work here*,
    not what the source document claims. That separation is the point of the
    whole record: one published vocabulary already turned out to contain a value
    the model does not implement, and prompting it degraded silently to the stem
    rather than failing.

    `approve_prose` carries exact text rather than a flag, and is accepted only
    when that text actually appears in `source_text`. Approving prose nobody
    read would mean nothing.
    """

    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity: str = Field(max_length=1_000)
    source_text: str = Field(max_length=20_000)
    grammar: dict[str, Any]
    approve_prose: list[str] = Field(default_factory=list, max_length=16)
    verified_values: dict[str, list[str]] = Field(default_factory=dict)


class AdapterPromptGrammarOut(ApiModel):
    """What was recorded. Digests and decisions only - never the source text."""

    id: str
    model_asset_install_id: str
    asset_sha256: str
    source_identity: str
    source_sha256: str
    schema_version: int
    grammar_sha256: str
    approved_prose_json: list[str]
    verified_values_json: dict[str, list[str]]
    compiler_version: str
    compiler_ceiling: int
    fits: bool
    reviewed_at: datetime | None


class ReferenceSubjectCreate(ApiModel):
    """A new subject. The mention is derived unless one is asked for by name.

    Leaving `mention_slug` unset is the ordinary path and collides gracefully -
    two real people can share a name, so a derived mention is suffixed rather
    than refused. Asking for one explicitly is refused on collision instead,
    because the user asked for that exact mention and quietly giving them a
    different one would be worse than saying no.
    """

    name: str = Field(min_length=1, max_length=120)
    kind: str
    mention_slug: str | None = Field(default=None, max_length=64)
    description: str | None = Field(default=None, max_length=4_000)
    aliases: list[str] = Field(default_factory=list, max_length=32)
    tags: list[str] = Field(default_factory=list, max_length=32)


class ReferenceSubjectUpdate(ApiModel):
    """Rename, archive or favourite. Every field is optional and independent.

    `follow_mention` defaults to false: a rename does not move the mention,
    because a live chat draft may already hold the old one and silently
    breaking it is worse than a mention that no longer matches the name.
    """

    name: str | None = Field(default=None, min_length=1, max_length=120)
    follow_mention: bool = False
    archived: bool | None = None
    favorite: bool | None = None
    # Omitting one leaves it alone; sending an empty string or an empty list
    # clears it. Those are different instructions and a single nullable value
    # could not carry both.
    description: str | None = Field(default=None, max_length=4_000)
    aliases: list[str] | None = Field(default=None, max_length=32)
    tags: list[str] | None = Field(default=None, max_length=32)


class ReferenceCoverIn(ApiModel):
    """Which of a reference's images stands for it.

    Its own request rather than a field on the update above, because "leave the
    cover alone" and "remove the cover" are different intentions and one
    optional field cannot say both. Clearing is the DELETE.
    """

    artifact_id: str = Field(min_length=1, max_length=80)


class ReferenceSubjectOut(ApiModel):
    id: str
    name: str
    mention_slug: str
    kind: ReferenceKind
    description: str | None
    aliases_json: list[str]
    tags_json: list[str]
    cover_artifact_id: str | None
    favorite: bool
    archived: bool


class ReferenceSubjectPage(ApiModel):
    items: list[ReferenceSubjectOut]
    total: int
    limit: int
    offset: int


class ReferenceDeletionImpact(ApiModel):
    """What deleting would destroy, offered before it is done.

    `exclusive_artifact_ids` counts only images nobody else references. A
    photograph showing two subjects belongs to both, and removing one of them
    is not permission to delete the picture.
    """

    reference_subject_id: str
    name: str
    asset_count: int
    exclusive_artifact_ids: list[str]


class ReferenceAssetAttach(ApiModel):
    artifact_id: str = Field(min_length=1, max_length=80)
    caption: str | None = Field(default=None, max_length=2_000)
    purpose: str = "other"
    view_label: str | None = Field(default=None, max_length=60)


class ReferenceAssetOut(ApiModel):
    id: str
    reference_subject_id: str
    artifact_id: str
    caption: str | None
    purpose: str
    view_label: str | None
    sort_order: int
    validation_state: ValidationState


ReferenceReviewReason = Annotated[
    StrictStr,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]


class ReferenceAssetReview(ApiModel):
    outcome: StrictStr = Field(min_length=1, max_length=30)
    reasons: list[ReferenceReviewReason] = Field(default_factory=list, max_length=16)


class ReferenceAssetReviewed(ApiModel):
    asset: ReferenceAssetOut
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    review_version: int = Field(gt=1)


class ReferenceSimilarAsset(ApiModel):
    """An image already held that closely resembles the one just added."""

    reference_asset_id: str
    artifact_id: str
    mean_absolute_difference: float


class ReferenceAssetAttached(ApiModel):
    """The attachment, plus anything the caller should weigh up afterwards.

    `similar` is advice rather than a refusal: two close shots of one subject
    are often deliberate, so the person adding them decides. An empty list means
    nothing resembled it *or* the comparison could not run - never a claim that
    the image is definitely new.
    """

    asset: ReferenceAssetOut
    similar: list[ReferenceSimilarAsset]


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
    kind: DeviceKind
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
    artifact_directory: str
    # Present only when construction followed a filesystem link to reach
    # artifact_directory. A relative path or a case change is not a link.
    artifact_directory_requested: str | None = None
    max_media_outputs_per_plan: int = Field(ge=1, le=16)
    # The installation-wide gate. When this is false no chat can open its
    # own, and the UI says so rather than offering a switch that does
    # nothing.
    web_access_enabled: bool = False


class ThirdPartyNoticesOut(ApiModel):
    # Both null when not running from a release, which carries them.
    text: str | None
    license_folder: str | None


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
    startup_duration_ms: int | None = Field(default=None, ge=0)
    current_memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    active_jobs: int = 0
    queued_jobs: int = 0
    # How long ago the worker last reported forward motion on a job it is
    # running. `state` cannot express this: a worker that has stopped
    # progressing still reports `ready` with an active job, so a live stall
    # is invisible without a measurement of the engine's own reports.
    #
    # None means NO MEASUREMENT, which covers two situations and deliberately
    # does not distinguish them: nothing is running, or something is running that
    # has not reported yet. A worker still loading, and a worker between jobs,
    # are both "no answer" rather than an age. Reading None as "idle" is
    # therefore wrong - only `active_jobs` says that. An age invented for the
    # not-yet-reported case would be indistinguishable at any value from a
    # real stall of the same length, making a worker that is merely starting
    # up look wedged.
    #
    # Deliberately an AGE rather than a `stuck` flag. Where the line falls
    # depends on the workflow - a video step legitimately reports nothing for far
    # longer than an image step - so the number is reported and the threshold is
    # left to whoever has that context, instead of being frozen into the wire
    # format.
    progress_age_seconds: float | None = Field(default=None, ge=0)
    failure_detail: str | None = None
    # What kind of failure this was, and what the user can do about it. Both are
    # derived from the same output `stderr_tail` carries; neither replaces it.
    failure_code: WorkerFailureCode | None = None
    failure_remedy: str | None = None
    stderr_tail: str | None = None
    log_path: str | None = None


class WorkerSettings(ApiModel):
    # Bounds mirror Settings.worker_startup_seconds so a value accepted here is
    # never rejected when the process restarts and reads it back from disk.
    worker_startup_seconds: float = Field(ge=1, le=600)


class WorkerResetResult(ApiModel):
    worker: WorkerStatus
    cancelled_jobs: int


class WorkerLogTail(ApiModel):
    name: Literal["chat", "media"]
    text: str
    truncated: bool
    log_bytes: int


class WorkerLogLocation(ApiModel):
    path: str


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
    #: Another version's managed release that the configuration still uses, when
    #: this build's pinned release is not installed.
    installed_release: str | None = None


SetupReadinessCode = Literal[
    "activation_ready",
    "activation_required",
    "activation_stale",
    "generation_verification_failed",
    "generation_verification_paused",
    "generation_verification_pausing",
    "generation_verification_queued",
    "generation_verification_required",
    "generation_verification_running",
    "generation_verified",
    "install_failed",
    "install_in_progress",
    "model_missing",
    "model_ready",
    "model_unsupported",
    "profile_missing",
    "profile_ready",
    "runtime_external",
    "runtime_failed",
    "runtime_installing",
    "runtime_missing",
    "runtime_other_version",
    "runtime_ready",
    "runtime_unsupported",
    "worker_failed",
    "worker_not_loaded",
    "worker_ready",
    "worker_starting",
    "worker_status_unavailable",
    "workflow_activation_not_ready",
    "workflow_invalid",
    "workflow_missing",
    "workflow_ready",
    "workflow_untrusted",
]


class SetupReadinessCheck(ApiModel):
    code: SetupReadinessCode = Field(min_length=1, max_length=80)
    status: Literal["pass", "pending", "fail"]
    message: str = Field(min_length=1, max_length=240)
    action: str | None = Field(default=None, min_length=1, max_length=80)


class SetupRoleReadiness(ApiModel):
    role: Literal["chat", "image", "video"]
    state: Literal["ready", "in_progress", "action_required"]
    verification_level: Literal["generation_probe"] = "generation_probe"
    engine: str | None = None
    job_id: str | None = None
    verification_id: str | None = None
    install_id: str | None = None
    profile_id: str | None = None
    workflow_revision_id: str | None = None
    next_action: str | None = None
    checks: list[SetupReadinessCheck] = Field(default_factory=list)


class SetupReadinessReport(ApiModel):
    version: Literal[2] = 2
    state: Literal["ready", "in_progress", "action_required"]
    roles: list[SetupRoleReadiness]


class SetupVerificationOut(ApiModel):
    id: str
    role: Literal["chat", "image", "video"]
    state: Literal["queued", "running", "ready", "failed"]
    job_id: str | None
    failure_code: (
        Literal[
            "application_restarted",
            "empty_generation",
            "generation_cancelled",
            "generation_failed",
            "generation_not_started",
        ]
        | None
    )
    started_at: datetime | None
    completed_at: datetime | None


class BackupInfo(ApiModel):
    name: str
    size_bytes: int
    sha256: str
    created_at: datetime
    verified: bool = False
    restore_pending: bool = False
    media_included: bool = False
    media_size_bytes: int = 0


def _utc_instant(value: datetime) -> str:
    """SQLite keeps these naive and they are UTC; say so at the browser boundary."""

    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return normalized.isoformat().replace("+00:00", "Z")


class EmptyChatEntryOut(ApiModel):
    """One empty chat, described by the decision rather than by its content.

    No title, no draft text, no prompt. A maintenance list that shows titles is
    a list of what somebody wrote, and the reasons are what the decision needs.
    """

    id: str
    classification: Literal["strict_blank", "configured_blank", "inconsistent"]
    created_at: datetime
    updated_at: datetime
    age_hours: float
    reasons: list[str] = Field(default_factory=list)
    #: `inconsistent` is never offered for deletion, and says so here rather
    #: than leaving the surface to infer it from the classification.
    deletable: bool

    @field_serializer("created_at", "updated_at", when_used="json")
    def serialize_timestamp_as_utc(self, value: datetime) -> str:
        return _utc_instant(value)


class EmptyChatPageOut(ApiModel):
    entries: list[EmptyChatEntryOut] = Field(default_factory=list)
    next_cursor: str | None = None
    #: Of this page, not of the library. A count over everything would be a
    #: second query answering a different question from the rows shown.
    counts: dict[str, int] = Field(default_factory=dict)
    evaluated_at: datetime

    @field_serializer("evaluated_at", when_used="json")
    def serialize_evaluated_at_as_utc(self, value: datetime) -> str:
        return _utc_instant(value)


#: The most chats one cleanup may name: a page's worth, so a selection is always
#: something a person could have looked at.
EMPTY_CHAT_SELECTION_LIMIT = 200


class EmptyChatPreviewIn(ApiModel):
    """The exact ids chosen, and the filter state they were chosen under."""

    chat_ids: list[Annotated[str, Field(min_length=1, max_length=40)]] = Field(
        min_length=1, max_length=EMPTY_CHAT_SELECTION_LIMIT
    )
    min_age_hours: Annotated[float, Field(ge=0)] = 24.0
    include_archived: bool = False
    include_configured: bool = False


class EmptyChatConflictOut(ApiModel):
    chat_id: str
    reason: Literal[
        "missing",
        "out_of_scope",
        "not_empty",
        "too_young",
        "archived_excluded",
        "inconsistent",
        "filtered_out",
    ]


class EmptyChatPreviewOut(ApiModel):
    #: Names this issuance. Deleting must send it back, because the deadline
    #: below belongs to this preview and not to the selection in general.
    preview_id: str
    digest: str
    expires_at: datetime
    strict_count: int
    configured_count: int
    conflicts: list[EmptyChatConflictOut] = Field(default_factory=list)

    @field_serializer("expires_at", when_used="json")
    def serialize_expires_at_as_utc(self, value: datetime) -> str:
        return _utc_instant(value)


class EmptyChatExecuteIn(EmptyChatPreviewIn):
    #: Client-generated before asking, and the same across a retry of one
    #: decision, so a repeat returns the first result instead of deleting again.
    operation_id: str = Field(min_length=1, max_length=80)
    preview_id: str = Field(min_length=1, max_length=64)
    digest: str = Field(min_length=64, max_length=64)
    acknowledged_count: Annotated[int, Field(ge=0)]
    acknowledged_configured: bool = False


class EmptyChatDeletionOut(ApiModel):
    operation_id: str
    deleted_ids: list[str] = Field(default_factory=list)
    deleted_at: datetime
    #: True when this returned an earlier operation's result rather than deleting now.
    replayed: bool

    @field_serializer("deleted_at", when_used="json")
    def serialize_deleted_at_as_utc(self, value: datetime) -> str:
        return _utc_instant(value)


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
    provider: Literal["huggingface", "civitai", "crw"]
    configured: bool
    source: Literal["none", "environment", "credential_vault"]
    vault_available: bool


class CredentialSet(ApiModel):
    token: str = Field(min_length=1, max_length=10_000)
