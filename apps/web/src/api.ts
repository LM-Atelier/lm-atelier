import type { QueueLane, QueueOrderCommand, QueueOrderPage, QueueOrderResult } from "./queueOrderTypes";
import type { GenerationExperiment, GenerationExperimentBlindEvaluationCreate, GenerationExperimentBlindView, GenerationExperimentCreate, GenerationExperimentEvaluationCreate, GenerationExperimentPreflight, GenerationExperimentRecipeDraft, GenerationExperimentRequest, GenerationExperimentStart } from "./generationExperimentTypes";
import { workflowFamilyQuery, workflowReadQuery, type WorkflowFamilyReadOptions, type WorkflowReadPageOptions } from "./workflowReadQuery";
import type { WorkflowRecipeTarget, WorkflowUseCase, WorkflowUseCaseChoice, WorkflowUseCaseDefault, WorkflowUseCasePreset, WorkflowUseCasePresetCreate } from "./workflowUseCaseTypes";
import type { EnlargementPreview } from "./studioEnlargement";
import { buildTurnRequest, SOURCE_FIT_BINDING_ERROR, type TurnRequestPayload } from "./turnRequest";
import { defaultOutputShapes } from "./outputShapePreferences";
export { buildTurnRequest } from "./turnRequest";
import type { SourceFitCapability, SourceFitIntent, SourceFitPreviewResult, SourceFitSelection } from "./sourceFit";
import type { TurnReference } from "./mentionDraft";
import type { ComposerPromptSource } from "./composerPromptSource";
import type { InstallQueueAction, InstallQueuePolicy } from "./installationQueueTypes";
import {
  parseArtifactLibraryPage,
  type ArtifactLibraryFilters,
} from "./artifactLibraryPage";
import type {
  WebSearch,
  WebSearchConfiguration,
  StudioCapabilityReport,
  StudioLocalEditRequest,
  ApplicationInfo,
  AppEvent,
  Artifact,
  ArtifactCleanupResult,
  ArtifactDeleteResult,
  ArtifactLibraryItem,
  ArtifactStorageInfo,
  BackupInfo,
  RetentionPolicy,
  ThirdPartyNotices,
  LoraSuggestions,
  WorkflowLoraControls,
  EmptyChatDeletion,
  EmptyChatPage,
  EmptyChatPreview,
  ReferenceAsset,
  ReferenceAssetAttached,
  ReferenceAssetReview,
  ReferenceAssetReviewed,
  ReferenceDeletionImpact,
  ReferenceSubject,
  ReferenceSubjectPage,
  CatalogModel,
  CatalogPage,
  CatalogDetail,
  CatalogInstallMatches,
  CatalogPreflight,
  CatalogVersions,
  Chat,
  ChatSearchPage,
  ChatTranscriptContext,
  ChatEditLineagePage,
  ChatSummary,
  ChatComposerDraft,
  ChatComposerDraftInput,
  ChatItemRemovalExecution,
  ChatItemRemovalImpact,
  ContentRating,
  ExchangeDeletion,
  ChatDetail,
  ChatMessageWindow,
  DraftClassification,
  PriorTurnEditBinding,
  CustomNodeInstall,
  CredentialProvider,
  CredentialStatus,
  EngineCapabilities,
  GenerationPreset,
  GenerationPresetBundle,
  Job,
  JobActivity,
  QueueActivityItem,
  QueueControlCommand,
  GenerationQueueAction,
  GenerationQueuePolicy,
  TransferQueueAction,
  TransferQueuePolicy,
  QueueControlResult,
  QueueActivityPage,
  QueuePlanSteps,
  Message,
  ModelAssetInstall,
  ModelInstall,
  ModelStorageInfo,
  ModelUpdate,
  ModelProfile,
  UseCaseSuggestionOut,
  ModelProfileModelUpdate,
  ModelProfileBundle,
  OutputRatioPresetId,
  PlatformMatrixEntry,
  PromptHelperDetail,
  PromptBatchCreateInput,
  PromptBatchItemUpdateInput,
  PromptBatchQueueInput,
  PromptTemplateCreateInput,
  PromptTemplateDetail,
  PromptTemplatePage,
  PromptTemplateRevision,
  PromptTemplateUpdateInput,
  PromptTemplateWriteResult,
  Project,
  ReferenceRecipe,
  RegistryInstall,
  RoutingMode,
  RuntimeStatus,
  SetupReadinessReport,
  SetupVerification,
  SystemInfo,
  ToolCapabilityProbe,
  TurnAccepted,
  Run,
  PriorTurnEditRequest,
  PriorTurnEditAccepted,
  PriorTurnEditSource,
  EditTemplate,
  Workflow,
  WorkflowOutputGeometryCapability,
  WorkflowOutputGeometryResolution,
  WorkflowBundle,
  WorkflowCatalogGraph,
  WorkflowAssetReview,
  WorkflowPackageAnalysis,
  WorkflowRevisionChoice,
  WorkflowReadyRevision,
  WorkflowRevisionSchema,
  WorkflowSummary,
  WorkflowRevision,
  WorkflowRevisionReview,
  WorkflowActivation,
  WorkflowActivationPreparation,
  WorkflowActivationRequest,
  WorkflowEditorDraft,
  WorkflowEditorReturn,
  WorkflowEditorSession,
  WorkerLogLocation,
  WorkerLogTail,
  WorkerResetResult,
  WorkerSettings,
  WorkerStatus,
  EditedBranchPage,
  EditedBranchActivation,
  WorkPlan,
  WorkStep,
  WorkflowDependencyResourceKind,
  WorkflowFamily,
  WorkflowInstallProgress,
  WorkflowFamilyPreference,
  WorkflowFamilyPreferenceUpdate,
  WorkflowFamilyRemovalImpact,
  WorkflowFamilyUpdate,
  WorkflowResourceConsumers,
  WorkflowSelection,
  WorkflowSelectorCapability,
  ChatWorkflowSelectionInput,
  ProjectWorkflowSelectionInput,
} from "./types";

export type TurnConfirmationRequest = {
  kind: "ordered_plan";
  title: string;
  question: string;
  confirmLabel: string;
  details: {
    sequence: string[];
    videoDurationSeconds?: number;
    estimatedWorkingBytes?: number;
  };
} | {
  kind: "media_route";
  title: string;
  question: string;
  confirmLabel: string;
  details: {
    operation: "image" | "video";
    durationSeconds?: number;
    estimatedIntermediateBytes?: number;
  };
};

export type TurnConfirmationHandler = (
  request: TurnConfirmationRequest,
) => Promise<boolean>;

/** Decode only the server's supported confirmation responses. */
export function turnConfirmationForError(error: unknown): { confirmation: TurnConfirmationRequest; mode: RoutingMode } | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  const detail = error.detail && typeof error.detail === "object" ? error.detail as Record<string, unknown> : null;
  const plan = detail?.plan && typeof detail.plan === "object" ? detail.plan as Record<string, unknown> : null;
  const estimate = detail?.estimate && typeof detail.estimate === "object" ? detail.estimate as Record<string, unknown> : null;
  const orderedSteps = Array.isArray(plan?.steps) ? plan.steps.filter((step): step is Record<string, unknown> => (
    Boolean(step) && typeof step === "object"
  )) : [];
  if (detail?.code === "ordered_plan_confirmation_required" && orderedSteps.length >= 2) {
    return { mode: "auto", confirmation: {
      kind: "ordered_plan", title: "Start ordered plan?",
      question: `This request will run ${orderedSteps.length} steps in sequence.`, confirmLabel: "Start plan",
      details: {
        sequence: orderedSteps.map((step) => typeof step.mode === "string" ? step.mode : "work"),
        ...(typeof estimate?.video_duration_seconds === "number" && estimate.video_duration_seconds > 0
          ? { videoDurationSeconds: estimate.video_duration_seconds } : {}),
        ...(typeof estimate?.estimated_bytes === "number" && estimate.estimated_bytes > 0
          ? { estimatedWorkingBytes: estimate.estimated_bytes } : {}),
      },
    } };
  }
  const operation = typeof plan?.operation === "string" ? plan.operation : "";
  if (detail?.code === "route_confirmation_required" && (operation.includes("image") || operation.includes("video"))) {
    const selectedOperation = operation.includes("video") ? "video" : "image";
    return { mode: selectedOperation, confirmation: {
      kind: "media_route", title: `Start ${selectedOperation} generation?`,
      question: `Auto mode suggests ${selectedOperation === "image" ? "an" : "a"} ${selectedOperation} generation.`,
      confirmLabel: `Start ${selectedOperation}`,
      details: {
        operation: selectedOperation,
        ...(typeof estimate?.duration_seconds === "number" ? { durationSeconds: estimate.duration_seconds } : {}),
        ...(typeof estimate?.estimated_intermediate_bytes === "number"
          ? { estimatedIntermediateBytes: estimate.estimated_intermediate_bytes } : {}),
      },
    } };
  }
  return null;
}

let csrfToken = "";
let eventEpoch = "";
let eventSequence = 0;
let sessionPromise: Promise<void> | null = null;

function resetSession(): void {
  csrfToken = "";
  sessionPromise = null;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: unknown,
    message: string,
    public readonly code?: string,
    /** The whole error body, for routes whose refusal carries more than a message. */
    public readonly payload?: Record<string, unknown>,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function ensureSession(): Promise<void> {
  if (csrfToken) return;
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const response = await fetch("/api/session", {
        method: "POST",
        credentials: "same-origin",
      });
      if (!response.ok) throw new Error("Could not initialize the local session");
      const payload = (await response.json()) as {
        csrf_token: string;
        event_epoch?: string;
        event_sequence?: number;
      };
      csrfToken = payload.csrf_token;
      eventEpoch = payload.event_epoch ?? "";
      eventSequence = Math.max(0, payload.event_sequence ?? 0);
    })();
  }
  try {
    await sessionPromise;
  } catch (error) {
    sessionPromise = null;
    throw error;
  }
}

/** Whether a 403 is the CSRF guard rather than a genuine refusal.
 *
 * The code first, the prose second. Matching on "CSRF check failed" made this
 * retry depend on wording nobody thought of as contract: rephrasing that
 * sentence would silently turn a recoverable stale token into a hard refusal,
 * and the sentence would have looked entirely safe to change.
 *
 * The prose check stays as a fallback rather than being deleted, because it
 * costs nothing and a response that predates the code still means the same
 * thing.
 */
async function isCsrfFailure(response: Response): Promise<boolean> {
  try {
    const payload = (await response.clone().json()) as { detail?: unknown; code?: unknown };
    if (payload.code === "csrf-invalid") return true;
    return payload.detail === "CSRF check failed";
  } catch {
    return false;
  }
}

/** Send one request and return its successful response, or throw its refusal. */
async function send(
  path: string,
  init: RequestInit = {},
  retrySession = true,
): Promise<Response> {
  if (path !== "/api/session") await ensureSession();
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  if (init.method && !["GET", "HEAD"].includes(init.method.toUpperCase())) {
    headers.set("x-local-lm-csrf", csrfToken);
  }
  const response = await fetch(path, { ...init, headers, credentials: "same-origin" });
  if (path !== "/api/session" && retrySession) {
    // A stale session answers 401, but a stale CSRF token answers 403, and only
    // the first was retried. Since resetSession() clears the token on every
    // socket close, a request that started just before a close went out with an
    // empty header and got a permanent 403 with no way back.
    const staleSession =
      response.status === 401
      || (response.status === 403 && (await isCsrfFailure(response)));
    if (staleSession) {
      resetSession();
      await ensureSession();
      return send(path, init, false);
    }
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let detail: unknown;
    let code: string | undefined;
    let body: Record<string, unknown> | undefined;
    try {
      const payload = (await response.json()) as { detail?: unknown; code?: unknown };
      if (payload && typeof payload === "object" && !Array.isArray(payload)) body = payload as Record<string, unknown>;
      detail = payload.detail;
      if (typeof payload.code === "string") code = payload.code;
      if (typeof detail === "string") message = detail;
      else if (detail && typeof detail === "object" && "message" in detail && typeof detail.message === "string") message = detail.message;
    } catch {
      // Preserve the HTTP status text.
    }
    throw new ApiError(response.status, detail, message, code, body);
  }
  return response;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await send(path, init);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/** A response's exact bytes, for a body whose digest covers them as sent. */
async function requestBytes(path: string, init: RequestInit = {}): Promise<ArrayBuffer> {
  return (await send(path, init)).arrayBuffer();
}

type WorkflowRevisionInput = Pick<
  WorkflowBundle,
  "engine_version" | "api_graph" | "ui_graph" | "input_schema" | "dependencies"
> & { trusted?: never };

type WorkflowCreateInput = WorkflowRevisionInput & Pick<
  WorkflowBundle,
  "name" | "description" | "operation" | "engine"
>;

export const api = {
  searchConfiguration: () => request<WebSearchConfiguration>("/api/web-search/configuration"),
  decideSearch: (jobId: string, revision: number, action: "approve" | "decline" | "cancel") =>
    request<WebSearch>("/api/jobs/" + encodeURIComponent(jobId) + "/search/decision",
      { method: "POST", body: JSON.stringify({ revision, action }) }),
  editSearch: (jobId: string, revision: number, query: string) =>
    request<WebSearch>("/api/jobs/" + encodeURIComponent(jobId) + "/search",
      { method: "PUT", body: JSON.stringify({ revision, query }) }),
  initialize: ensureSession,
  setupReadiness: () => request<SetupReadinessReport>("/api/setup/readiness"),
  verifySetupRole: (role: SetupVerification["role"]) =>
    request<SetupVerification>(`/api/setup/verify/${role}`, { method: "POST" }),
  projects: (includeArchived = false, query = "", options: { limit?: number; offset?: number; projectIds?: string[]; literalSearch?: boolean; signal?: AbortSignal } = {}) => {
    const parameters = new URLSearchParams({ include_archived: String(includeArchived), query });
    if (options.limit !== undefined) parameters.set("limit", String(options.limit));
    if (options.offset !== undefined) parameters.set("offset", String(options.offset));
    for (const id of options.projectIds ?? []) parameters.append("project_id", id);
    if (options.literalSearch) parameters.set("literal_search", "true");
    return request<Project[]>(`/api/projects?${parameters}`, options.signal ? { signal: options.signal } : undefined);
  },
  project: (id: string, signal?: AbortSignal) => request<Project>(`/api/projects/${encodeURIComponent(id)}`, signal ? { signal } : undefined),
  createProject: (name: string) =>
    request<Project>("/api/projects", { method: "POST", body: JSON.stringify({ name }) }),
  updateProject: (id: string, values: Partial<Project>) =>
    request<Project>(`/api/projects/${id}`, { method: "PATCH", body: JSON.stringify(values) }),
  deleteProject: (id: string) => request<void>(`/api/projects/${id}`, { method: "DELETE" }),
  chats: (projectId?: string | null, includeArchived = false, query = "", options: { limit?: number; offset?: number; searchProjects?: boolean; signal?: AbortSignal } = {}) => {
    const parameters = new URLSearchParams({ include_archived: String(includeArchived), query });
    if (projectId) parameters.set("project_id", projectId);
    if (options.limit !== undefined) parameters.set("limit", String(options.limit));
    if (options.offset !== undefined) parameters.set("offset", String(options.offset));
    if (options.searchProjects) parameters.set("search_projects", "true");
    return request<Chat[]>(`/api/chats?${parameters}`, options.signal ? { signal: options.signal } : undefined);
  },
  chatSummaries: (projectId?: string | null, includeArchived = false, query = "", options: { limit?: number; offset?: number; searchProjects?: boolean; signal?: AbortSignal } = {}) => {
    const parameters = new URLSearchParams({ include_archived: String(includeArchived), query });
    if (projectId) parameters.set("project_id", projectId);
    if (options.limit !== undefined) parameters.set("limit", String(options.limit));
    if (options.offset !== undefined) parameters.set("offset", String(options.offset));
    if (options.searchProjects) parameters.set("search_projects", "true");
    return request<ChatSummary[]>(`/api/chats/summaries?${parameters}`, options.signal ? { signal: options.signal } : undefined);
  },
  chat: (id: string) => request<ChatDetail>(`/api/chats/${id}`),
  chatMetadata: (id: string, signal?: AbortSignal) =>
    request<Chat>(`/api/chats/${encodeURIComponent(id)}/metadata`, signal ? { signal } : undefined),
  chatEditLineage: (id: string, resultId: string, options: {
    before?: string; limit?: number; signal?: AbortSignal;
  } = {}) => {
    const parameters = new URLSearchParams({ limit: String(options.limit ?? 40) });
    if (options.before) parameters.set("before", options.before);
    return request<ChatEditLineagePage>(`/api/chats/${encodeURIComponent(id)}/messages/${encodeURIComponent(resultId)}/lineage?${parameters}`,
      options.signal ? { signal: options.signal } : undefined);
  },
  chatContext: (id: string, headId: string | null, signal?: AbortSignal) => {
    const parameters = new URLSearchParams();
    if (headId !== null) parameters.set("head_id", headId);
    return request<ChatTranscriptContext>(`/api/chats/${encodeURIComponent(id)}/context?${parameters}`,
      signal ? { signal } : undefined);
  },
  chatSearches: (id: string, options: {
    headId?: string | null; oldestMessageId?: string; before?: string;
    pendingOnly?: boolean; limit?: number; signal?: AbortSignal;
  } = {}) => {
    const parameters = new URLSearchParams({ limit: String(options.limit ?? 40) });
    if (options.headId) parameters.set("head_id", options.headId);
    if (options.oldestMessageId !== undefined) parameters.set("oldest_message_id", options.oldestMessageId);
    if (options.before !== undefined) parameters.set("before", options.before);
    if (options.pendingOnly) parameters.set("pending_only", "true");
    return request<ChatSearchPage>(`/api/chats/${encodeURIComponent(id)}/searches?${parameters}`,
      options.signal ? { signal: options.signal } : undefined);
  },
  chatMessages: (id: string, options: {
    headId?: string | null; before?: string; after?: string; around?: string;
    limit?: number; signal?: AbortSignal;
  } = {}) => {
    const parameters = new URLSearchParams({ limit: String(options.limit ?? 40) });
    if (options.headId) parameters.set("head_id", options.headId);
    for (const anchor of ["before", "after", "around"] as const) {
      if (options[anchor] !== undefined) parameters.set(anchor, options[anchor]);
    }
    return request<ChatMessageWindow>(`/api/chats/${encodeURIComponent(id)}/messages?${parameters}`,
      options.signal ? { signal: options.signal } : undefined);
  },
  classifyDraft: (chatId: string, text: string, mode: RoutingMode, editSource?: PriorTurnEditBinding) =>
    request<DraftClassification>(`/api/chats/${chatId}/classify-draft`, {
      method: "POST",
      body: JSON.stringify({ text, mode, ...(editSource ? { edit_source: editSource } : {}) }),
    }),
  createChat: (projectId?: string | null) =>
    request<Chat>("/api/chats", {
      method: "POST",
      body: JSON.stringify({ title: "New chat", project_id: projectId ?? null }),
    }),
  updateChat: (id: string, values: Partial<Chat>) =>
    request<Chat>(`/api/chats/${id}`, { method: "PATCH", body: JSON.stringify(values) }),
  /** The chat's unsent draft as the workspace keeps it; revision 0 when it has none. */
  composerDraft: (chatId: string) =>
    request<ChatComposerDraft>(`/api/chats/${encodeURIComponent(chatId)}/composer-draft`),
  /** Replace the chat's stored draft, refused if another save came first. */
  saveComposerDraft: (chatId: string, expectedRevision: number, draft: ChatComposerDraftInput) =>
    request<ChatComposerDraft>(`/api/chats/${encodeURIComponent(chatId)}/composer-draft`, {
      method: "PUT",
      body: JSON.stringify({ expected_revision: expectedRevision, draft }),
    }),
  deleteChat: (id: string, deleteGeneratedMedia = false) => {
    const parameters = new URLSearchParams({
      delete_generated_media: String(deleteGeneratedMedia),
    });
    return request<void>(`/api/chats/${id}?${parameters}`, { method: "DELETE" });
  },  openStudioSession: (sourceArtifactId: string, sourceChatId: string | null = null) =>
    request<ChatDetail>("/api/studio/sessions", {
      method: "POST",
      body: JSON.stringify({
        source_artifact_id: sourceArtifactId,
        source_chat_id: sourceChatId,
      }),
    }),
  studioSession: (sessionId: string) =>
    request<ChatDetail>(`/api/studio/sessions/${encodeURIComponent(sessionId)}`),
  /** One run, with what it resolved and recorded: how a Studio result's edit is made again. */
  run: (runId: string) => request<Run>(`/api/runs/${encodeURIComponent(runId)}`),
  /** Which of a record's requirements this installation holds; nothing is installed or kept. */
  /** Whether a run generated again from a record came out as the record's output did. */
  replayResult: (runId: string, signal?: AbortSignal) =>
    request<unknown>(`/api/runs/${encodeURIComponent(runId)}/replay-result`, { signal }),
  /** Whether a record could be generated again exactly here; nothing is started or kept. */
  planGenerationReplay: (content: ArrayBuffer) =>
    request<unknown>("/api/output-recipes/replay-plan", {
      method: "POST",
      body: content,
      headers: { "content-type": "application/octet-stream" },
    }),
  /** Generate a record again exactly, as the first turn of a chat with nothing in it. */
  replayGenerationRecord: (chatId: string, content: ArrayBuffer) =>
    request<unknown>(`/api/chats/${encodeURIComponent(chatId)}/replays`, {
      method: "POST",
      body: content,
      headers: { "content-type": "application/octet-stream" },
    }),
  checkGenerationRecord: (content: ArrayBuffer) =>
    // A file's own bytes, not JSON: a bundle with its picture can be far larger
    // than a JSON body may be, and the route bounds what it reads itself.
    request<unknown>("/api/output-recipes/check", {
      method: "POST",
      body: content,
      headers: { "content-type": "application/octet-stream" },
    }),
  /** One output's portable generation record, as the exact bytes its digest covers. */
  generationRecord: (
    runId: string,
    artifactId: string,
    includePrompt: boolean,
    signal?: AbortSignal,
  ) =>
    requestBytes(
      `/api/runs/${encodeURIComponent(runId)}/outputs/${encodeURIComponent(artifactId)}/recipe?prompts=${includePrompt ? "include" : "omit"}`,
      { signal },
    ),
  /** A picture's record, byte for byte as shown and named by its digest, zipped with a clean copy. */
  generationRecordBundle: (
    runId: string,
    artifactId: string,
    includePrompt: boolean,
    digest: string,
    signal?: AbortSignal,
  ) =>
    requestBytes(
      `/api/runs/${encodeURIComponent(runId)}/outputs/${encodeURIComponent(artifactId)}/recipe-bundle?prompts=${includePrompt ? "include" : "omit"}&digest=${encodeURIComponent(digest)}`,
      { signal },
    ),
  preflightGenerationExperiment: (payload: GenerationExperimentRequest) =>
    request<GenerationExperimentPreflight>("/api/generation-experiments/preflight", { method: "POST", body: JSON.stringify(payload) }),
  createGenerationExperiment: (payload: GenerationExperimentCreate) =>
    request<GenerationExperiment>("/api/generation-experiments", { method: "POST", body: JSON.stringify(payload) }),
  generationExperiment: (experimentId: string, signal?: AbortSignal) =>
    request<GenerationExperiment>(`/api/generation-experiments/${encodeURIComponent(experimentId)}`, { signal }),
  startGenerationExperiment: (experimentId: string, payload: GenerationExperimentStart) =>
    request<GenerationExperiment>(`/api/generation-experiments/${encodeURIComponent(experimentId)}/start`, { method: "POST", body: JSON.stringify(payload) }),
  /** Keep which picture is preferred, or a tie, or neither suiting; the answer carries the latest. */
  evaluateGenerationExperiment: (experimentId: string, payload: GenerationExperimentEvaluationCreate) =>
    request<GenerationExperiment>(`/api/generation-experiments/${encodeURIComponent(experimentId)}/evaluations`, { method: "POST", body: JSON.stringify(payload) }),
  /** Begin a viewing of a blind comparison, with its own random order of the pictures. */
  openBlindView: (experimentId: string) =>
    request<GenerationExperimentBlindView>(`/api/generation-experiments/${encodeURIComponent(experimentId)}/blind-views`, { method: "POST" }),
  blindView: (experimentId: string, viewId: string, signal?: AbortSignal) =>
    request<GenerationExperimentBlindView>(`/api/generation-experiments/${encodeURIComponent(experimentId)}/blind-views/${encodeURIComponent(viewId)}`, { signal }),
  /** Keep the preference said in a viewing; the answer carries the reveal. */
  sayBlindPreference: (experimentId: string, viewId: string, payload: GenerationExperimentBlindEvaluationCreate) =>
    request<GenerationExperimentBlindView>(`/api/generation-experiments/${encodeURIComponent(experimentId)}/blind-views/${encodeURIComponent(viewId)}/evaluations`, { method: "POST", body: JSON.stringify(payload) }),
  /** A recipe to review from one choice of a comparison; nothing is saved. */
  generationExperimentRecipeDraft: (experimentId: string, ordinal: number, signal?: AbortSignal) =>
    request<GenerationExperimentRecipeDraft>(`/api/generation-experiments/${encodeURIComponent(experimentId)}/arms/${ordinal}/recipe-draft`, { signal }),
  studioLocalEdit: (sessionId: string, edit: StudioLocalEditRequest) =>
    request<ChatDetail>(`/api/studio/sessions/${encodeURIComponent(sessionId)}/local-edits`, {
      method: "POST",
      body: JSON.stringify(edit),
    }),
  createPromptHelper: (sourceChatId: string, draftPrompt: string) =>
    request<PromptHelperDetail>("/api/prompt-helpers", {
      method: "POST",
      body: JSON.stringify({ source_chat_id: sourceChatId, draft_prompt: draftPrompt }),
    }),
  promptHelper: (id: string) => request<PromptHelperDetail>(`/api/prompt-helpers/${id}`),
  updatePromptHelper: (id: string, draftPrompt: string) =>
    request<PromptHelperDetail>(`/api/prompt-helpers/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ draft_prompt: draftPrompt }),
    }),
  setResponseFeedback: (messageId: string, rating: "up" | "down" | null, revisionId: string | null = null) =>
    request<{ message_id: string; response_revision_id: string | null; rating: "up" | "down" | null }>(
      `/api/messages/${messageId}/feedback`,
      {
        method: "PUT",
        body: JSON.stringify({ rating, response_revision_id: revisionId }),
      },
    ),
  deletePromptHelper: (id: string) =>
    request<void>(`/api/prompt-helpers/${id}`, { method: "DELETE" }),
  promptTemplates: (includeArchived = false, limit = 100, offset = 0) => {
    const parameters = new URLSearchParams({
      include_archived: String(includeArchived),
      limit: String(limit),
      offset: String(offset),
    });
    return request<PromptTemplatePage>(`/api/prompt-templates?${parameters}`);
  },
  promptTemplate: (id: string) =>
    request<PromptTemplateDetail>(`/api/prompt-templates/${encodeURIComponent(id)}`),
  createPromptTemplate: (payload: PromptTemplateCreateInput) =>
    request<PromptTemplateWriteResult>("/api/prompt-templates", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updatePromptTemplate: (id: string, payload: PromptTemplateUpdateInput) =>
    request<PromptTemplateWriteResult>(`/api/prompt-templates/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deletePromptTemplate: (id: string, expectedCurrentRevisionId: string) => {
    const parameters = new URLSearchParams({
      expected_current_revision_id: expectedCurrentRevisionId,
    });
    return request<void>(
      `/api/prompt-templates/${encodeURIComponent(id)}?${parameters}`,
      { method: "DELETE" },
    );
  },
  promptTemplateRevisions: (id: string) =>
    request<PromptTemplateRevision[]>(
      `/api/prompt-templates/${encodeURIComponent(id)}/revisions`,
    ),
  restorePromptTemplateRevision: (
    id: string,
    revisionId: string,
    expectedCurrentRevisionId: string,
    idempotencyKey: string,
  ) => request<PromptTemplateWriteResult>(
    `/api/prompt-templates/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revisionId)}/restore`,
    {
      method: "POST",
      body: JSON.stringify({
        expected_current_revision_id: expectedCurrentRevisionId,
        idempotency_key: idempotencyKey,
      }),
    },
  ),
  createPromptBatch: (chatId: string, payload: PromptBatchCreateInput) =>
    request<unknown>(`/api/chats/${encodeURIComponent(chatId)}/prompt-batches`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  promptBatch: (batchId: string) =>
    request<unknown>(`/api/prompt-batches/${encodeURIComponent(batchId)}`),
  updatePromptBatchItem: (
    batchId: string,
    ordinal: number,
    payload: PromptBatchItemUpdateInput,
  ) => request<unknown>(
    `/api/prompt-batches/${encodeURIComponent(batchId)}/items/${ordinal}`,
    { method: "PATCH", body: JSON.stringify(payload) },
  ),
  queuePromptBatch: (batchId: string, payload: PromptBatchQueueInput) =>
    request<unknown>(`/api/prompt-batches/${encodeURIComponent(batchId)}/queue`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  studioCapabilities: () => request<StudioCapabilityReport>("/api/studio/capabilities"),
  /** Which workflow an enlargement of this turn's picture would run, and what it lets the person choose. */
  previewEnlargement: (chatId: string, payload: TurnRequestPayload, signal?: AbortSignal) =>
    request<EnlargementPreview>(`/api/chats/${encodeURIComponent(chatId)}/upscale/preview`, {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    }),
  sendTurn: async (
    chatId: string,
    text: string,
    mode: RoutingMode,
    inputArtifactIds: string[],
    settings: Record<string, unknown>,
    idempotencyKey: string = crypto.randomUUID(),
    endpoint: string = "turns",
    workflowRevisionId?: string,
    // Ids the person chose from the mention picker. Never derived from the
    // text: the server refuses to recover references by reading a prompt,
    // because that binds whoever the words most resemble.
    references: TurnReference[] = [],
    outputCount?: number,
    promptSource?: ComposerPromptSource,
    confirmTurn?: TurnConfirmationHandler,
    sourceFit?: SourceFitSelection,
    upscale?: boolean,
  ) => {
    const payload = buildTurnRequest({
      text, mode, inputArtifactIds, settings, idempotencyKey, workflowRevisionId,
      references, outputCount, promptSource, sourceFit,
      defaultOutputShapes: defaultOutputShapes(),
      upscale,
    });
    const submit = (selectedMode: RoutingMode, confirmed = false) => {
      if (payload.source_fit && selectedMode !== "image" && selectedMode !== "auto") throw new Error(SOURCE_FIT_BINDING_ERROR);
      return request<TurnAccepted>(`/api/chats/${chatId}/${endpoint}`, {
        method: "POST",
        body: JSON.stringify({ ...payload, mode: selectedMode, confirm_media: confirmed }),
      });
    };
    try {
      return await submit(mode);
    } catch (error) {
      const requested = turnConfirmationForError(error);
      if (!requested || !confirmTurn || !await confirmTurn(requested.confirmation)) throw error;
      return submit(requested.mode, true);
    }
  },
  stopAndSendTurn: (
    chatId: string,
    text: string,
    mode: RoutingMode,
    inputArtifactIds: string[],
    settings: Record<string, unknown>,
    idempotencyKey: string = crypto.randomUUID(),
    references: TurnReference[] = [],
    outputCount?: number,
    promptSource?: ComposerPromptSource,
    confirmTurn?: TurnConfirmationHandler,
    sourceFit?: SourceFitSelection,
  ) => api.sendTurn(
    chatId,
    text,
    mode,
    inputArtifactIds,
    settings,
    idempotencyKey,
    "stop-and-send",
    undefined,
    references,
    outputCount,
    promptSource,
    confirmTurn,
    sourceFit,
  ),
  regenerateMessage: (messageId: string, settings: Record<string, unknown>, idempotencyKey?: string) =>
    request<TurnAccepted>(`/api/messages/${messageId}/regenerate`, {
      method: "POST",
      body: JSON.stringify({ settings, idempotency_key: idempotencyKey }),
    }),
  forkThread: (messageId: string) =>
    request<Chat>(`/api/messages/${messageId}/fork`, { method: "POST" }),
  deleteExchange: (messageId: string) =>
    request<ExchangeDeletion>(`/api/messages/${messageId}/exchange`, { method: "DELETE" }),
  chatItemRemovalImpact: (messageId: string) =>
    request<ChatItemRemovalImpact>(`/api/messages/${messageId}/removal-impact`),
  removeChatItemContent: (
    messageId: string,
    expectedRevisionId: string,
    operationKey: string,
  ) => request<ChatItemRemovalExecution>(`/api/messages/${messageId}/remove-content`, {
    method: "POST",
    body: JSON.stringify({
      expected_message_id: messageId,
      expected_revision_id: expectedRevisionId,
      operation_key: operationKey,
    }),
  }),
  selectResponseRevision: (messageId: string, revisionId: string) =>
    request<Message>(`/api/messages/${messageId}/revisions/${revisionId}/select`, {
      method: "POST",
    }),
  getPriorTurnEditSource: (messageId: string, sourceRunId?: string) => {
    const query = sourceRunId === undefined
      ? ""
      : `?${new URLSearchParams({ source_run_id: sourceRunId })}`;
    return request<PriorTurnEditSource>(
      `/api/messages/${encodeURIComponent(messageId)}/edit-source${query}`,
    );
  },
  queueEditedMessage: (messageId: string, payload: PriorTurnEditRequest) =>
    request<PriorTurnEditAccepted>(`/api/messages/${encodeURIComponent(messageId)}/edits`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  branchMessage: (
    messageId: string,
    text: string,
    mode: RoutingMode,
    settings: Record<string, unknown>,
  ) =>
    request<TurnAccepted>(`/api/messages/${messageId}/branch`, {
      method: "POST",
      body: JSON.stringify({
        text,
        mode,
        input_artifact_ids: [],
        settings,
        idempotency_key: crypto.randomUUID(),
      }),
    }),
  cancelChat: (chatId: string) =>
    request<Job>(`/api/chats/${chatId}/cancel`, { method: "POST" }),
  jobs: () => request<Job[]>("/api/jobs"),
  queuePlanSteps: (planId: string, offset: number, signal?: AbortSignal) =>
    request<QueuePlanSteps>("/api/queue/plans/" + encodeURIComponent(planId)
      + "/steps?limit=50&offset=" + String(offset), { signal }).then((value) => {
      if (value.plan_id !== planId) throw new Error("The submitted work steps could not be read.");
      return value;
    }),
  installQueuePolicy: (signal?: AbortSignal) =>
    request<InstallQueuePolicy>("/api/queue/lanes/install", { signal }),
  installQueueControl: (action: InstallQueueAction, command: QueueControlCommand) =>
    request<InstallQueuePolicy>("/api/queue/lanes/install/"
      + (action === "pause_after_current" ? "pause-after-current" : "resume"),
    { method: "POST", body: JSON.stringify(command) }),
  transferQueuePolicy: (signal?: AbortSignal) =>
    request<TransferQueuePolicy>("/api/queue/lanes/transfer", { signal }),
  transferQueueControl: (action: TransferQueueAction, command: QueueControlCommand) =>
    request<TransferQueuePolicy>("/api/queue/lanes/transfer/"
      + (action === "pause_after_current" ? "pause-after-current" : "resume"),
    { method: "POST", body: JSON.stringify(command) }),
  generationQueuePolicy: (signal?: AbortSignal) =>
    request<GenerationQueuePolicy>("/api/queue/lanes/generation", { signal }),
  generationQueueControl: (action: GenerationQueueAction, command: QueueControlCommand) =>
    request<GenerationQueuePolicy>("/api/queue/lanes/generation/"
      + (action === "pause_after_current" ? "pause-after-current" : "resume"),
    { method: "POST", body: JSON.stringify(command) }),
  queueControl: (planId: string, action: "hold" | "release", command: QueueControlCommand) =>
    request<QueueControlResult>("/api/queue/items/" + encodeURIComponent(planId) + "/" + action,
      { method: "POST", body: JSON.stringify(command) }),
  queueOrder: (lane: QueueLane, options: { cursor?: string | null; limit: number }, signal?: AbortSignal) => {
    const params = new URLSearchParams({ limit: String(options.limit) });
    if (options.cursor) params.set("cursor", options.cursor);
    return request<QueueOrderPage>("/api/queue/lanes/" + lane + "/order?" + params.toString(), { signal });
  },
  reorderQueue: (lane: QueueLane, command: QueueOrderCommand) =>
    request<QueueOrderResult>("/api/queue/lanes/" + lane + "/reorder",
      { method: "POST", body: JSON.stringify(command) }),
  queueActivity: (options: { lane?: QueueActivityItem["lane"]; cursor?: string | null; limit: number }, signal?: AbortSignal) => {
    const params = new URLSearchParams({ limit: String(options.limit) });
    if (options.lane) params.set("lane", options.lane);
    if (options.cursor) params.set("cursor", options.cursor);
    return request<QueueActivityPage>("/api/queue/activity?" + params.toString(), { signal });
  },
  jobActivity: (activeLimit: number) =>
    request<JobActivity>(`/api/jobs/activity?active_limit=${activeLimit}`),
  workPlans: (chatId?: string) =>
    request<WorkPlan[]>(
      `/api/work-plans${chatId ? `?chat_id=${encodeURIComponent(chatId)}` : ""}`,
    ),
  editedBranches: (chatId: string, cursor: string | null = null, signal?: AbortSignal) => {
    const query = new URLSearchParams({ limit: "50" });
    if (cursor !== null) query.set("cursor", cursor);
    return request<EditedBranchPage>(
      `/api/chats/${encodeURIComponent(chatId)}/edited-branches?${query}`, { signal },
    );
  },
  activateEditedBranch: (chatId: string, planId: string, expectedHead: string | null) =>
    request<EditedBranchActivation>(
      `/api/chats/${encodeURIComponent(chatId)}/edited-branches/${encodeURIComponent(planId)}/activate`,
      { method: "POST", body: JSON.stringify({ expected_active_head_message_id: expectedHead }) },
    ),
  workPlan: (id: string) => request<WorkPlan>(`/api/work-plans/${id}`),
  workStep: (id: string) => request<WorkStep>(`/api/work-steps/${id}`),
  cancelWorkPlan: (id: string) =>
    request<WorkPlan>(`/api/work-plans/${id}/cancel`, { method: "POST" }),
  retryWorkPlan: (id: string) =>
    request<WorkPlan>(`/api/work-plans/${id}/retry`, { method: "POST" }),
  cancelWorkStep: (id: string) =>
    request<Job>(`/api/work-steps/${id}/cancel`, { method: "POST" }),
  retryWorkStep: (id: string) =>
    request<Job>(`/api/work-steps/${id}/retry`, { method: "POST" }),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  retryJob: (id: string) =>
    request<Job>(`/api/jobs/${encodeURIComponent(id)}/retry`, { method: "POST" }),
  pauseDownload: (id: string) =>
    request<Job>(`/api/downloads/${id}/pause`, { method: "POST" }),
  resumeDownload: (id: string) =>
    request<Job>(`/api/downloads/${id}/resume`, { method: "POST" }),
  activateModel: (id: string) =>
    request<Job>(`/api/models/${id}/activate`, { method: "POST" }),
  engines: () => request<EngineCapabilities[]>("/api/engines"),
  probeChatTools: () =>
    request<ToolCapabilityProbe>("/api/engines/chat/tool-probe", { method: "POST" }),
  system: () => request<SystemInfo>("/api/system"),
  about: () => request<ApplicationInfo>("/api/about"),
  thirdPartyNotices: () => request<ThirdPartyNotices>("/api/about/third-party-notices"),
  platforms: () => request<PlatformMatrixEntry[]>("/api/platforms"),
  createDiagnostics: () => request<{ url: string }>("/api/diagnostics", { method: "POST" }),
  credentialStatus: (provider: CredentialProvider) =>
    request<CredentialStatus>(`/api/credentials/${encodeURIComponent(provider)}`),
  setCredentialToken: (provider: CredentialProvider, token: string) =>
    request<CredentialStatus>(`/api/credentials/${encodeURIComponent(provider)}`, {
      method: "PUT",
      body: JSON.stringify({ token }),
    }),
  deleteCredentialToken: (provider: CredentialProvider) =>
    request<CredentialStatus>(`/api/credentials/${encodeURIComponent(provider)}`, {
      method: "DELETE",
    }),
  models: () => request<ModelInstall[]>("/api/models"),
  modelsPage: (options: {
    limit: number; offset?: number; search?: string; role?: "chat" | "image" | "video";
    chatCapability?: "text" | "vision"; modelIds?: string[];
  }) => {
    const parameters = new URLSearchParams({ limit: String(options.limit), offset: String(options.offset ?? 0) });
    if (options.search) parameters.set("search", options.search);
    if (options.role) parameters.set("role", options.role);
    if (options.chatCapability) parameters.set("chat_capability", options.chatCapability);
    for (const id of options.modelIds ?? []) parameters.append("model_id", id);
    return request<ModelInstall[]>(`/api/models?${parameters}`);
  },
  catalogInstallMatches: (options: {
    role: "chat" | "image" | "video"; remoteIds: string[]; workflowTemplateIds: string[];
  }) => {
    const parameters = new URLSearchParams({ role: options.role });
    for (const id of options.remoteIds) parameters.append("remote_id", id);
    for (const id of options.workflowTemplateIds) parameters.append("workflow_template_id", id);
    return request<CatalogInstallMatches>(`/api/models/catalog-matches?${parameters}`);
  },
  modelInstall: async (installId: string) => {
    const installs = await request<ModelInstall[]>("/api/models?install_id=" + encodeURIComponent(installId));
    return installs.find((install) => install.id === installId) ?? null;
  },
  modelStorage: () => request<ModelStorageInfo>("/api/models/storage"),
  modelUpdates: () => request<ModelUpdate[]>("/api/models/updates"),
  catalogItemDetail: (source: string, itemId: string, role: string | null) => {
    const parameters = new URLSearchParams({ source, id: itemId });
    if (role) parameters.set("role", role);
    return request<CatalogDetail>(`/api/catalog/item?${parameters}`);
  },
  deleteModel: (id: string, deleteProfiles = false) =>
    request<void>(
      `/api/models/${id}?${new URLSearchParams({ delete_profiles: String(deleteProfiles) })}`,
      { method: "DELETE" },
    ),
  cleanupDownloads: () =>
    request<{ removed_count: number; reclaimed_bytes: number }>("/api/downloads/cleanup", {
      method: "POST",
    }),
  profiles: () => request<ModelProfile[]>("/api/profiles"),
  profilesPage: (options: {
    limit: number; offset?: number; search?: string; role?: string; engine?: string;
    inputModality?: "text" | "image"; installIds?: string[]; profileIds?: string[]; defaultsOnly?: boolean;
  }) => {
    const parameters = new URLSearchParams({ limit: String(options.limit), offset: String(options.offset ?? 0) });
    if (options.search) parameters.set("search", options.search);
    if (options.role) parameters.set("role", options.role);
    if (options.inputModality) parameters.set("input_modality", options.inputModality);
    if (options.engine) parameters.set("engine", options.engine);
    if (options.defaultsOnly !== undefined) parameters.set("defaults_only", String(options.defaultsOnly));
    for (const id of options.installIds ?? []) parameters.append("install_id", id);
    for (const id of options.profileIds ?? []) parameters.append("profile_id", id);
    return request<ModelProfile[]>(`/api/profiles?${parameters}`);
  },
  createProfile: (model: ModelInstall, isDefault = false) =>
    request<ModelProfile>("/api/profiles", {
      method: "POST",
      body: JSON.stringify({
        name: model.name,
        role: model.role,
        engine: model.engine,
        model_install_id: model.id,
        load_settings: {},
        request_settings: {},
        is_default: isDefault,
      }),
    }),
  downloadJob: (id: string) => request<Job>(`/api/downloads/${encodeURIComponent(id)}`),
  updateProfileModel: (id: string, values: ModelProfileModelUpdate) =>
    request<ModelProfile>(`/api/profiles/${encodeURIComponent(id)}/model-update`, {
      method: "POST", body: JSON.stringify(values),
    }),
  updateProfile: (
    id: string,
    values: {
      name?: string;
      use_case?: string;
      use_case_derived?: boolean;
      expected_use_case?: string;
      load_settings?: Record<string, unknown>;
      request_settings?: Record<string, unknown>;
      is_default?: boolean;
    },
  ) => request<ModelProfile>(`/api/profiles/${id}`, { method: "PATCH", body: JSON.stringify(values) }),
  cloneProfile: (id: string, name?: string) =>
    request<ModelProfile>(`/api/profiles/${id}/clone`, {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  suggestProfileUseCase: (id: string, expectedUseCase: string, signal?: AbortSignal) =>
    request<UseCaseSuggestionOut>(`/api/profiles/${encodeURIComponent(id)}/use-case-suggestion`, {
      method: "POST", body: JSON.stringify({ expected_use_case: expectedUseCase }), signal,
    }),
  suggestLoraUseCase: (id: string, expectedUseCase: string, signal?: AbortSignal) =>
    request<UseCaseSuggestionOut>(`/api/model-assets/${encodeURIComponent(id)}/use-case-suggestion`, {
      method: "POST", body: JSON.stringify({ expected_use_case: expectedUseCase }), signal,
    }),
  resetProfile: (id: string) =>
    request<ModelProfile>(`/api/profiles/${id}/reset`, { method: "POST" }),
  deleteProfile: (id: string) => request<void>(`/api/profiles/${id}`, { method: "DELETE" }),
  exportProfile: (id: string) => request<ModelProfileBundle>(`/api/profiles/${id}/export`),
  importProfile: (bundle: ModelProfileBundle) =>
    request<ModelProfile>("/api/profiles/import", { method: "POST", body: JSON.stringify(bundle) }),
  presets: () => request<GenerationPreset[]>("/api/presets"),
  presetsPage: (options: {
    limit: number; offset?: number; search?: string; role?: string; presetIds?: string[]; defaultsOnly?: boolean;
  }) => {
    const parameters = new URLSearchParams({ limit: String(options.limit), offset: String(options.offset ?? 0) });
    if (options.search) parameters.set("search", options.search);
    if (options.role) parameters.set("role", options.role);
    if (options.defaultsOnly !== undefined) parameters.set("defaults_only", String(options.defaultsOnly));
    for (const id of options.presetIds ?? []) parameters.append("preset_id", id);
    return request<GenerationPreset[]>(`/api/presets?${parameters}`);
  },
  createPreset: (role: GenerationPreset["role"], name: string) =>
    request<GenerationPreset>("/api/presets", {
      method: "POST",
      body: JSON.stringify({ role, name, settings: {} }),
    }),
  updatePreset: (
    id: string,
    values: { name?: string; settings?: Record<string, unknown>; is_default?: boolean },
  ) => request<GenerationPreset>(`/api/presets/${id}`, { method: "PATCH", body: JSON.stringify(values) }),
  clonePreset: (id: string, name?: string) =>
    request<GenerationPreset>(`/api/presets/${id}/clone`, {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  resetPreset: (id: string) =>
    request<GenerationPreset>(`/api/presets/${id}/reset`, { method: "POST" }),
  exportPreset: (id: string) => request<GenerationPresetBundle>(`/api/presets/${id}/export`),
  importPreset: (bundle: GenerationPresetBundle) =>
    request<GenerationPreset>("/api/presets/import", { method: "POST", body: JSON.stringify(bundle) }),
  deletePreset: (id: string) => request<void>(`/api/presets/${id}`, { method: "DELETE" }),
  workers: () => request<WorkerStatus[]>("/api/workers"),
  workerSettings: () => request<WorkerSettings>("/api/workers/settings"),
  updateWorkerSettings: (values: WorkerSettings) =>
    request<WorkerSettings>("/api/workers/settings", { method: "PUT", body: JSON.stringify(values) }),
  runtimes: () => request<RuntimeStatus[]>("/api/runtimes"),
  installRuntime: (engine: RuntimeStatus["engine"]) =>
    request<RuntimeStatus>(`/api/runtimes/${engine}/install`, { method: "POST" }),
  loadChatWorker: (profileId: string) =>
    request<WorkerStatus>(`/api/workers/chat/load/${profileId}`, { method: "POST" }),
  startMediaWorker: () => request<WorkerStatus>("/api/workers/media/start", { method: "POST" }),
  stopWorker: (name: "chat" | "media") =>
    request<WorkerStatus>(`/api/workers/${name}/stop`, { method: "POST" }),
  restartWorker: (name: "chat" | "media") =>
    request<WorkerStatus>(`/api/workers/${name}/restart`, { method: "POST" }),
  resetWorker: (name: "chat" | "media") =>
    request<WorkerResetResult>(`/api/workers/${name}/reset`, { method: "POST" }),
  workerLogTail: (name: "chat" | "media") =>
    request<WorkerLogTail>(`/api/workers/${name}/log-tail`),
  workerLogLocation: () => request<WorkerLogLocation>("/api/workers/log-location"),
  // Empty-chat cleanup. The list and the check only read; the delete spends a
  // check and is refused whole if anything it bound has changed.
  emptyChats: (options: {
    include_archived?: boolean;
    include_configured?: boolean;
    cursor?: string;
  } = {}) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(options)) {
      if (value !== undefined) query.set(key, String(value));
    }
    const suffix = query.toString();
    return request<EmptyChatPage>(`/api/maintenance/empty-chats${suffix ? `?${suffix}` : ""}`);
  },
  previewEmptyChats: (body: {
    chat_ids: string[];
    include_archived: boolean;
    include_configured: boolean;
  }) =>
    request<EmptyChatPreview>("/api/maintenance/empty-chats/preview", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteEmptyChats: (body: {
    operation_id: string;
    preview_id: string;
    digest: string;
    acknowledged_count: number;
    acknowledged_configured: boolean;
    chat_ids: string[];
    include_archived: boolean;
    include_configured: boolean;
  }) =>
    request<EmptyChatDeletion>("/api/maintenance/empty-chats/execute", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  backups: () => request<BackupInfo[]>("/api/backups"),
  createBackup: (includeMedia = false) =>
    request<BackupInfo>(`/api/backups?${new URLSearchParams({ include_media: String(includeMedia) })}`, { method: "POST" }),
  verifyBackup: (name: string) =>
    request<BackupInfo>(`/api/backups/${encodeURIComponent(name)}/verify`, { method: "POST" }),
  restoreBackup: (name: string) =>
    request<BackupInfo>(`/api/backups/${encodeURIComponent(name)}/restore`, { method: "POST" }),
  deleteBackup: (name: string) =>
    request<void>(`/api/backups/${encodeURIComponent(name)}`, { method: "DELETE" }),
  exportProject: (projectId: string, includeMedia = true) =>
    request<{ url: string }>(`/api/projects/${projectId}/export?${new URLSearchParams({ include_media: String(includeMedia) })}`, { method: "POST" }),
  importProject: async (file: File) => {
    await ensureSession();
    const form = new FormData();
    form.append("archive", file);
    return request<Project>("/api/projects/import", {
      method: "POST",
      headers: { "x-local-lm-csrf": csrfToken },
      body: form,
    });
  },
  references: (search = "", includeArchived = false, limit = 50, offset = 0) => {
    const parameters = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (search) parameters.set("search", search);
    if (includeArchived) parameters.set("include_archived", "true");
    return request<ReferenceSubjectPage>(`/api/references?${parameters}`);
  },
  createReference: (body: { name: string; kind: string; description?: string }) =>
    request<ReferenceSubject>("/api/references", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updateReference: (
    id: string,
    // Omitting a field leaves it alone; sending "" or [] clears it. Those are
    // different instructions, so the optional fields are genuinely optional
    // rather than nullable.
    body: {
      name?: string;
      follow_mention?: boolean;
      archived?: boolean;
      favorite?: boolean;
      description?: string;
      aliases?: string[];
      tags?: string[];
    },
  ) =>
    request<ReferenceSubject>(`/api/references/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  referenceDeletionImpact: (id: string) =>
    request<ReferenceDeletionImpact>(
      `/api/references/${encodeURIComponent(id)}/deletion-impact`,
    ),
  // The acknowledgement is required by the server: it refuses to delete
  // something other than what the caller was shown.
  deleteReference: (id: string, acknowledgedAssets: number) =>
    request<void>(
      `/api/references/${encodeURIComponent(id)}?acknowledged_assets=${acknowledgedAssets}`,
      { method: "DELETE" },
    ),
  referenceAssets: (id: string) =>
    request<ReferenceAsset[]>(`/api/references/${encodeURIComponent(id)}/assets`),
  reviewReferenceAsset: (id: string, assetId: string, body: ReferenceAssetReview) =>
    request<ReferenceAssetReviewed>(
      `/api/references/${encodeURIComponent(id)}/assets/${encodeURIComponent(assetId)}/review`,
      { method: "POST", body: JSON.stringify(body) },
    ),
  attachReferenceAsset: (id: string, body: { artifact_id: string; purpose?: string }) =>
    request<ReferenceAssetAttached>(`/api/references/${encodeURIComponent(id)}/assets`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  detachReferenceAsset: (id: string, assetId: string) =>
    request<void>(
      `/api/references/${encodeURIComponent(id)}/assets/${encodeURIComponent(assetId)}`,
      { method: "DELETE" },
    ),
  // Setting and clearing are separate calls rather than one nullable field,
  // because "leave the cover alone" and "remove it" are different intentions.
  setReferenceCover: (id: string, artifactId: string) =>
    request<ReferenceSubject>(`/api/references/${encodeURIComponent(id)}/cover`, {
      method: "PUT",
      body: JSON.stringify({ artifact_id: artifactId }),
    }),
  clearReferenceCover: (id: string) =>
    request<ReferenceSubject>(`/api/references/${encodeURIComponent(id)}/cover`, {
      method: "DELETE",
    }),
  artifacts: (
    kind = "", query = "", favorites = false,
    page?: { limit: number; offset: number }, signal?: AbortSignal,
  ) => {
    const parameters = new URLSearchParams({ query });
    if (kind) parameters.set("kind", kind);
    if (favorites) parameters.set("favorites", "true");
    if (page) {
      parameters.set("limit", String(page.limit));
      parameters.set("offset", String(page.offset));
    }
    return request<ArtifactLibraryItem[]>(`/api/artifacts?${parameters}`, { signal });
  },
  artifactLibrary: async (
    filters: ArtifactLibraryFilters,
    cursor: string | null,
    limit: number,
    signal?: AbortSignal,
  ) => {
    const parameters = new URLSearchParams({
      limit: String(limit),
      query: filters.query,
      state: "visible",
    });
    if (filters.kind) parameters.set("kind", filters.kind);
    if (filters.favorite) parameters.set("favorite", "true");
    if (cursor !== null) parameters.set("cursor", cursor);
    const payload = await request<unknown>(`/api/artifact-library?${parameters}`, { signal });
    const page = parseArtifactLibraryPage(payload, limit);
    if (cursor !== null && page.next_cursor === cursor) {
      throw new Error("The Media Library response was invalid.");
    }
    return page;
  },
  generationRetryPolicy: () =>
    request<import("./generationRetryTypes").GenerationRetryPolicy>("/api/settings/generation-retries"),
  updateGenerationRetryPolicy: (maxRetries: number, expectedRevision: number) =>
    request<import("./generationRetryTypes").GenerationRetryPolicy>("/api/settings/generation-retries", {
      method: "PUT",
      body: JSON.stringify({ max_retries: maxRetries, expected_revision: expectedRevision }),
    }),
  artifact: (artifactId: string) =>
    request<Artifact>(`/api/artifacts/${encodeURIComponent(artifactId)}`),
  favoriteArtifact: (artifactId: string, favorite: boolean) =>
    request<Artifact>(`/api/artifacts/${encodeURIComponent(artifactId)}`, {
      method: "PATCH",
      body: JSON.stringify({ favorite }),
    }),
  artifactStorage: () => request<ArtifactStorageInfo>("/api/artifacts/storage"),
  cleanupArtifacts: (dryRun: boolean) =>
    request<ArtifactCleanupResult>("/api/artifacts/cleanup", {
      method: "POST",
      body: JSON.stringify({ dry_run: dryRun }),
    }),
  retentionPolicy: () => request<RetentionPolicy>("/api/artifacts/retention"),
  chooseRetention: (expectedRevision: number, mediaDays: number, temporaryHours: number) =>
    request<RetentionPolicy>("/api/artifacts/retention", {
      method: "PUT",
      body: JSON.stringify({
        expected_revision: expectedRevision,
        media_days: mediaDays,
        temporary_hours: temporaryHours,
      }),
    }),
  previewRetention: (mediaDays: number, temporaryHours: number) =>
    request<ArtifactCleanupResult>("/api/artifacts/retention/preview", {
      method: "POST",
      body: JSON.stringify({ media_days: mediaDays, temporary_hours: temporaryHours }),
    }),
  deleteArtifact: (artifactId: string) =>
    request<ArtifactDeleteResult>(`/api/artifacts/${encodeURIComponent(artifactId)}`, {
      method: "DELETE",
    }),
  /** Workflows on a remote source.
   *
   * Its own method rather than an option on `catalog`, because the two ask
   * different questions: the model search carries role, quantization and
   * parameter filters that a workflow cannot answer, and folding them together
   * would make every caller carry parameters that mean nothing for half of
   * them. The server draws the same line.
   */
  workflowCatalog: (
    query: string,
    sort: string,
    cursor?: string | null,
    source = "civitai",
  ) => {
    const parameters = new URLSearchParams({ query, sort, source });
    if (cursor) parameters.set("cursor", cursor);
    return request<CatalogPage>(`/api/workflow-catalog?${parameters.toString()}`);
  },
  /** One discovered workflow's graph, for the same review an imported file gets. */
  workflowCatalogGraph: (versionId: string, source = "civitai") =>
    request<WorkflowCatalogGraph>(
      `/api/workflow-catalog/versions/${encodeURIComponent(versionId)}/graph?${new URLSearchParams({ source })}`,
    ),
  catalog: (
    query: string,
    role: string,
    sort: string,
    cursor?: string | null,
    filters: Record<string, string> = {},
    source = "huggingface",
  ) => {
    const parameters = new URLSearchParams({ query, role, sort });
    if (source !== "huggingface") parameters.set("source", source);
    if (cursor) parameters.set("cursor", cursor);
    for (const [key, value] of Object.entries(filters)) if (value) parameters.set(key, value);
    return request<CatalogPage>(`/api/catalog?${parameters.toString()}`);
  },
  workflowCatalogModels: (role: string) =>
    request<CatalogModel[]>(`/api/catalog/workflow-models?${new URLSearchParams({ role })}`),
  catalogDetail: (remoteId: string, role: string, revision = "main") =>
    request<CatalogDetail>(`/api/catalog/${remoteId}?${new URLSearchParams({ role, revision })}`),
  catalogPreflight: (
    remoteId: string,
    role: string,
    engine: string,
    revision: string,
    selectedFiles: string[],
    auxiliaryKind: string | null = null,
    workflowTemplateId: string | null = null,
    provider = "huggingface",
    workflowReferenceKind: string | null = null,
    // Named when a filename cannot settle the choice: one version can publish
    // the same safetensors name several times at different precisions.
    selectedFileIds: string[] = [],
  ) => {
    const body = JSON.stringify({
      role,
      engine,
      revision,
      selected_files: selectedFiles,
      selected_file_ids: selectedFileIds,
      auxiliary_kind: auxiliaryKind,
      workflow_template_id: workflowTemplateId,
      workflow_reference_kind: workflowReferenceKind,
    });
    if (provider !== "huggingface") {
      const parameters = new URLSearchParams({ source: provider, id: remoteId });
      return request<CatalogPreflight>(`/api/catalog/preflight?${parameters}`, {
        method: "POST",
        body,
      });
    }
    return request<CatalogPreflight>(`/api/catalog/${remoteId}/preflight`, {
      method: "POST",
      body,
    });
  },
  recipes: () => request<ReferenceRecipe[]>("/api/recipes"),
  installRecipe: (recipeId: string) =>
    request<Job>(`/api/recipes/${encodeURIComponent(recipeId)}/install`, { method: "POST" }),
  download: (
    remoteId: string,
    sourceRemoteId: string | null,
    role: string,
    engine: string,
    revision: string,
    allowPatterns: string[] = [],
    expectedSha256: Record<string, string> = {},
    fileSources: NonNullable<CatalogPreflight["file_sources"]> = {},
    comfyPaths: Record<string, string> = {},
    workflowTemplateId: string | null = null,
    workflowTemplateSha256: string | null = null,
    installPlanId: string | null = null,
    auxiliaryKind: string | null = null,
    contentRating: ContentRating = "unknown",
  ) =>
    request<Job>("/api/downloads", {
      method: "POST",
      body: JSON.stringify({
        remote_id: remoteId,
        source_remote_id: sourceRemoteId,
        revision,
        role,
        engine,
        allow_patterns: allowPatterns,
        expected_sha256: expectedSha256,
        file_sources: fileSources,
        comfy_paths: comfyPaths,
        workflow_template_id: workflowTemplateId,
        workflow_template_sha256: workflowTemplateSha256,
        install_plan_id: installPlanId,
        auxiliary_kind: auxiliaryKind,
        content_rating: contentRating,
      }),
    }),
  modelAssets: (kind?: string) =>
    request<ModelAssetInstall[]>(
      `/api/model-assets${kind ? `?${new URLSearchParams({ kind })}` : ""}`,
    ),
  updateModelAsset: (
    id: string,
    values: Partial<Pick<
      ModelAssetInstall,
      "active" | "use_case" | "auto_apply" | "default_model_strength" | "default_clip_strength"
      | "typed_trigger_words" | "use_case_derived"
    >> & {
      /** The base model the asset is for; an empty string clears it. */
      family?: string;
      expected_use_case?: string;
    },
  ) =>
    request<ModelAssetInstall>(`/api/model-assets/${id}`, {
      method: "PATCH",
      body: JSON.stringify(values),
    }),
  deleteModelAsset: (id: string) =>
    request<void>(`/api/model-assets/${id}`, { method: "DELETE" }),
  importModel: (payload: { name: string; role: string; engine: string; local_path: string }) =>
    request<ModelInstall>("/api/models/import", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  workflows: () => request<Workflow[]>("/api/workflows"),
  workflowRevisionOutputGeometry: (revisionId: string) =>
    request<WorkflowOutputGeometryCapability>(
      `/api/workflow-revisions/${encodeURIComponent(revisionId)}/output-geometry`,
    ),
  // Read-only: resolving a shape writes nothing, queues nothing and grants no
  // authority to generate. It exists so the browser can show the exact pixels a
  // choice means instead of working them out itself.
  resolveWorkflowRevisionOutputGeometry: (
    revisionId: string,
    geometry: { mode: "image" | "video"; size_mode: "preset"; preset_id: OutputRatioPresetId },
  ) =>
    request<WorkflowOutputGeometryResolution>(
      `/api/workflow-revisions/${encodeURIComponent(revisionId)}/output-geometry/resolve`,
      { method: "POST", body: JSON.stringify(geometry) },
    ),
  // Read-only as well: the size the revision makes in the exact shape of one
  // picture, as the picture is shown. A shape it cannot make exactly is refused.
  matchWorkflowRevisionOutputGeometryToSource: (revisionId: string, sourceArtifactId: string) =>
    request<WorkflowOutputGeometryResolution>(
      `/api/workflow-revisions/${encodeURIComponent(revisionId)}/output-geometry/match-source`,
      { method: "POST", body: JSON.stringify({ source_artifact_id: sourceArtifactId }) },
    ),
  workflowSummaries: (options: WorkflowReadPageOptions = {}, signal?: AbortSignal) =>
    request<WorkflowSummary[]>("/api/workflow-summaries" + workflowReadQuery(options), { signal }),
  workflow: (id: string, signal?: AbortSignal) =>
    request<Workflow>("/api/workflows/" + encodeURIComponent(id), { signal }).then((value) => {
      if (value.id !== id) throw new Error("The selected workflow could not be read.");
      return value;
    }),
  workflowReadyRevisions: (options: WorkflowReadPageOptions = {}, signal?: AbortSignal) =>
    request<WorkflowReadyRevision[]>("/api/workflow-ready-revisions" + workflowReadQuery(options), { signal }),
  workflowRevisionChoices: (signal?: AbortSignal, options: WorkflowReadPageOptions = {}) =>
    request<WorkflowRevisionChoice[]>("/api/workflow-revision-choices" + workflowReadQuery(options), { signal }),
  /** The LoRAs a workflow revision applies, read-only. */
  workflowLoraControls: (revisionId: string, signal?: AbortSignal) =>
    request<WorkflowLoraControls>(
      "/api/workflow-revisions/" + encodeURIComponent(revisionId) + "/lora-controls", { signal },
    ),
  workflowLoraSuggestions: (revisionId: string, signal?: AbortSignal) =>
    request<LoraSuggestions>(
      "/api/workflow-revisions/" + encodeURIComponent(revisionId) + "/lora-suggestions", { signal },
    ),
  workflowRevisionSchema: (revisionId: string, signal?: AbortSignal) =>
    request<WorkflowRevisionSchema>(
      "/api/workflow-revisions/" + encodeURIComponent(revisionId) + "/settings-schema", { signal },
    ).then((value) => {
      if (value.revision_id !== revisionId) throw new Error("The selected workflow settings could not be read.");
      return value;
    }),
  previewTurnSourceFit: (chatId: string, payload: TurnRequestPayload, signal?: AbortSignal) =>
    request<SourceFitPreviewResult>(
      "/api/chats/" + encodeURIComponent(chatId) + "/source-fit/preview",
      { method: "POST", body: JSON.stringify(payload), signal },
    ),
  previewPriorTurnSourceFit: (messageId: string, payload: PriorTurnEditRequest, signal?: AbortSignal) =>
    request<SourceFitPreviewResult>(
      "/api/messages/" + encodeURIComponent(messageId) + "/edits/source-fit/preview",
      { method: "POST", body: JSON.stringify(payload), signal },
    ),
  workflowRevisionSourceFit: (revisionId: string, signal?: AbortSignal) =>
    request<SourceFitCapability>(
      "/api/workflow-revisions/" + encodeURIComponent(revisionId) + "/source-fit",
      { signal },
    ),
  previewWorkflowRevisionSourceFit: (
    revisionId: string,
    sourceArtifactId: string,
    sourceFit: SourceFitIntent,
    signal?: AbortSignal,
  ) =>
    request<SourceFitPreviewResult>(
      "/api/workflow-revisions/" + encodeURIComponent(revisionId) + "/source-fit/preview",
      {
        method: "POST",
        body: JSON.stringify({ source_artifact_id: sourceArtifactId, source_fit: sourceFit }),
        signal,
      },
    ),
  workflowFamily: (familyId: string, options: WorkflowFamilyReadOptions = {}, signal?: AbortSignal) => {
    const query = workflowFamilyQuery(options).toString();
    return request<WorkflowFamily>(`/api/workflow-families/${encodeURIComponent(familyId)}${query ? `?${query}` : ""}`, { signal });
  },
  updateWorkflowFamily: (familyId: string, changes: WorkflowFamilyUpdate) =>
    request<WorkflowFamily>(`/api/workflow-families/${encodeURIComponent(familyId)}`, {
      method: "PATCH",
      body: JSON.stringify(changes),
    }),
  setWorkflowFamilyPreference: (
    familyId: string,
    capability: WorkflowSelectorCapability,
    preference: WorkflowFamilyPreferenceUpdate,
  ) =>
    request<WorkflowFamilyPreference>(
      `/api/workflow-families/${encodeURIComponent(familyId)}/preferences/${capability}`,
      { method: "PUT", body: JSON.stringify(preference) },
    ),
  workflowFamilyRemovalImpact: (familyId: string) =>
    request<WorkflowFamilyRemovalImpact>(
      `/api/workflow-families/${encodeURIComponent(familyId)}/removal-impact`,
    ),
  workflowResourceConsumers: (kind: WorkflowDependencyResourceKind, resourceId: string) =>
    request<WorkflowResourceConsumers>(
      `/api/workflow-dependencies/${kind}/${encodeURIComponent(resourceId)}/consumers`,
    ),
  workflowFamilies: (capability?: WorkflowSelectorCapability, includeArchived = false, includeDependencies = false, options: WorkflowFamilyReadOptions = {}, signal?: AbortSignal) => {
    const parameters = workflowFamilyQuery(options);
    if (capability) parameters.set("selector_capability", capability);
    if (includeArchived) parameters.set("include_archived", "true");
    if (includeDependencies) parameters.set("include_dependencies", "true");
    const query = parameters.toString();
    return request<WorkflowFamily[]>(`/api/workflow-families${query ? `?${query}` : ""}`, { signal });
  },
  workflowFamilyOperations: (includeArchived = false, signal?: AbortSignal) =>
    request<string[]>(`/api/workflow-family-operations${includeArchived ? "?include_archived=true" : ""}`, { signal }),
  workflowUseCasePresets: (useCase?: WorkflowUseCase, offset = 0, signal?: AbortSignal) => {
    const query = new URLSearchParams({ limit: "200", offset: String(offset) });
    if (useCase) query.set("use_case", useCase);
    return request<WorkflowUseCasePreset[]>(`/api/workflow-use-case-presets?${query}`, { signal });
  },
  createWorkflowUseCasePreset: (payload: WorkflowUseCasePresetCreate) =>
    request<WorkflowUseCasePreset>("/api/workflow-use-case-presets", { method: "POST", body: JSON.stringify(payload) }),
  replaceWorkflowUseCasePreset: (id: string, payload: WorkflowUseCasePresetCreate) =>
    request<WorkflowUseCasePreset>(`/api/workflow-use-case-presets/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify(payload) }),
  deleteWorkflowUseCasePreset: (id: string) =>
    request<void>(`/api/workflow-use-case-presets/${encodeURIComponent(id)}`, { method: "DELETE" }),
  workflowUseCaseDefault: (useCase: WorkflowUseCase, signal?: AbortSignal) =>
    request<WorkflowUseCaseDefault>(`/api/workflow-use-case-defaults/${useCase}`, { signal }),
  setWorkflowUseCaseDefault: (useCase: WorkflowUseCase, payload: WorkflowUseCaseDefault) =>
    request<WorkflowUseCaseDefault>(`/api/workflow-use-case-defaults/${useCase}`, { method: "PUT", body: JSON.stringify(payload) }),
  workflowUseCaseChoice: (scope: WorkflowRecipeTarget, useCase: WorkflowUseCase, signal?: AbortSignal) =>
    request<WorkflowUseCaseChoice>(`/api/${scope.kind === "chat" ? "chats" : "projects"}/${encodeURIComponent(scope.id)}/workflow-use-case-presets/${useCase}`, { signal }),
  setWorkflowUseCaseChoice: (scope: WorkflowRecipeTarget, useCase: WorkflowUseCase, payload: WorkflowUseCaseChoice) =>
    request<WorkflowUseCaseChoice>(`/api/${scope.kind === "chat" ? "chats" : "projects"}/${encodeURIComponent(scope.id)}/workflow-use-case-presets/${useCase}`, { method: "PUT", body: JSON.stringify(payload) }),
  chatWorkflowSelections: (chatId: string) =>
    request<WorkflowSelection[]>(
      `/api/chats/${encodeURIComponent(chatId)}/workflow-selections`,
    ),
  setChatWorkflowSelection: (
    chatId: string,
    capability: WorkflowSelectorCapability,
    selection: ChatWorkflowSelectionInput,
  ) =>
    request<WorkflowSelection>(
      `/api/chats/${encodeURIComponent(chatId)}/workflow-selections/${capability}`,
      { method: "PUT", body: JSON.stringify(selection) },
    ),
  projectWorkflowSelections: (projectId: string) =>
    request<WorkflowSelection[]>(
      `/api/projects/${encodeURIComponent(projectId)}/workflow-selections`,
    ),
  setProjectWorkflowSelection: (
    projectId: string,
    capability: WorkflowSelectorCapability,
    selection: ProjectWorkflowSelectionInput,
  ) =>
    request<WorkflowSelection>(
      `/api/projects/${encodeURIComponent(projectId)}/workflow-selections/${capability}`,
      { method: "PUT", body: JSON.stringify(selection) },
    ),
  createWorkflow: (payload: WorkflowCreateInput) =>
    request<Workflow>("/api/workflows", { method: "POST", body: JSON.stringify(payload) }),
  updateWorkflow: (id: string, payload: Record<string, unknown>) =>
    request<Workflow>(`/api/workflows/${id}`, { method: "PATCH", body: JSON.stringify(payload) }),
  createWorkflowRevision: (id: string, payload: WorkflowRevisionInput) =>
    request<WorkflowRevision>(`/api/workflows/${id}/revisions`, { method: "POST", body: JSON.stringify(payload) }),
  restoreWorkflowRevision: (id: string, revisionId: string) =>
    request<WorkflowRevision>(`/api/workflows/${id}/revisions/${revisionId}/restore`, { method: "POST" }),
  prepareWorkflowActivation: (id: string, revisionId: string, signal?: AbortSignal) =>
    request<WorkflowActivationPreparation>(`/api/workflows/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revisionId)}/activation/prepare`, { signal }),
  activateWorkflowRevision: (id: string, revisionId: string, payload: WorkflowActivationRequest, signal?: AbortSignal) =>
    request<WorkflowActivation>(`/api/workflows/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revisionId)}/activation`, { method: "POST", body: JSON.stringify(payload), signal }),
  previewWorkflowRevisionReview: (id: string, revisionId: string) =>
    request<WorkflowRevisionReview>(`/api/workflows/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revisionId)}/review`),
  decideWorkflowRevisionReview: (id: string, revisionId: string, payload: { action: "approve" | "revoke"; subject_sha256: string }) =>
    request<WorkflowRevisionReview>(`/api/workflows/${encodeURIComponent(id)}/revisions/${encodeURIComponent(revisionId)}/review`, { method: "POST", body: JSON.stringify(payload) }),
  cloneWorkflow: (id: string, name?: string) =>
    request<Workflow>(`/api/workflows/${id}/clone`, { method: "POST", body: JSON.stringify({ name }) }),
  exportWorkflow: (id: string) => request<WorkflowBundle>(`/api/workflows/${id}/export`),
  workflowOpenTarget: (id: string) => request<{ url: string; filename: string; ui_graph: Record<string, unknown> }>(`/api/workflows/${id}/open-target`),
  startWorkflowEditor: (id: string) =>
    request<WorkflowEditorSession>(`/api/workflows/${encodeURIComponent(id)}/editor-sessions`, {
      method: "POST",
    }),
  consumeWorkflowEditor: (
    workflowId: string,
    sessionId: string,
    payload: {
      nonce: string;
      base_revision_id: string;
      ui_graph: Record<string, unknown>;
      api_prompt: Record<string, unknown>;
    },
  ) =>
    request<WorkflowEditorReturn>(
      `/api/workflows/${encodeURIComponent(workflowId)}/editor-sessions/${encodeURIComponent(sessionId)}/consume`,
      { method: "POST", body: JSON.stringify(payload) },
    ),
  createWorkflowEditorDraft: (workflowId: string, validatedReturnId: string) =>
    request<WorkflowEditorDraft>(
      `/api/workflows/${encodeURIComponent(workflowId)}/editor-drafts`,
      { method: "POST", body: JSON.stringify({ validated_return_id: validatedReturnId }) },
    ),
  cancelWorkflowEditor: (workflowId: string, sessionId: string, nonce: string) =>
    request<void>(
      `/api/workflows/${encodeURIComponent(workflowId)}/editor-sessions/${encodeURIComponent(sessionId)}/cancel`,
      { method: "POST", body: JSON.stringify({ nonce }) },
    ),
  importWorkflow: (bundle: WorkflowBundle) =>
    request<Workflow>("/api/workflows/import", { method: "POST", body: JSON.stringify(bundle) }),
  editTemplates: () => request<EditTemplate[]>("/api/edit-templates"),
  createEditTemplate: (payload: {
    name: string;
    description?: string;
    instruction: string;
    settings_json?: Record<string, unknown>;
    /** Read the recipe from what this run did, rather than from the words and settings given here. */
    from_run_id?: string;
  }) =>
    request<EditTemplate>("/api/edit-templates", { method: "POST", body: JSON.stringify(payload) }),
  deleteEditTemplate: (id: string) =>
    request<void>(`/api/edit-templates/${id}`, { method: "DELETE" }),
  ensureWorkflowPackageDraft: (payload: {
    ui_graph: Record<string, unknown>;
    name: string;
    operation: string;
    description?: string;
  }) =>
    request<Workflow>("/api/workflows/packages/drafts", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  prepareWorkflowPackage: (
    packageId: string,
    version: string,
    uiGraph: Record<string, unknown>,
    workflowRevisionId?: string,
  ) =>
    request<Job>("/api/workflows/packages/prepare", {
      method: "POST",
      body: JSON.stringify({
        package_id: packageId,
        version,
        ui_graph: uiGraph,
        workflow_revision_id: workflowRevisionId,
      }),
    }),
  catalogVersions: (modelId: string) =>
    request<CatalogVersions>(`/api/catalog/civitai/${encodeURIComponent(modelId)}/versions`),
  registryInstalls: () => request<RegistryInstall[]>("/api/workflows/packages/installs"),
  reviewRegistryInstall: (installId: string, trusted: boolean) =>
    request<RegistryInstall>(`/api/workflows/packages/installs/${installId}/review`, {
      method: "POST",
      body: JSON.stringify({ trusted }),
    }),
  activateRegistryInstall: (installId: string) =>
    request<RegistryInstall>(`/api/workflows/packages/installs/${installId}/activate`, {
      method: "POST",
    }),
  deactivateRegistryInstall: (installId: string) =>
    request<RegistryInstall>(`/api/workflows/packages/installs/${installId}/deactivate`, {
      method: "POST",
    }),
  renewRegistryInstall: (installId: string) =>
    request<Job>(`/api/workflows/packages/installs/${installId}/renew`, {
      method: "POST",
    }),
  removeRegistryInstall: (installId: string) =>
    request<void>(`/api/workflows/packages/installs/${installId}`, {
      method: "DELETE",
    }),
  analyzeWorkflowPackage: (uiGraph: Record<string, unknown>) =>
    request<WorkflowPackageAnalysis>("/api/workflows/packages/analyze", {
      method: "POST",
      body: JSON.stringify({ ui_graph: uiGraph }),
    }),
  reviewWorkflowAssets: (
    uiGraph: Record<string, unknown>,
    selections: Array<{
      reference_filename: string;
      install_plan_id: string;
      artifact_path: string;
    }>,
  ) =>
    request<WorkflowAssetReview>("/api/workflows/packages/assets/review", {
      method: "POST",
      body: JSON.stringify({ ui_graph: uiGraph, selections }),
    }),
  workflowInstallProgress: (offerId: string, signal?: AbortSignal) =>
    request<WorkflowInstallProgress>(`/api/workflow-install-offers/${encodeURIComponent(offerId)}/progress`, { signal }),
  installWorkflowOffer: (offerId: string) =>
    request<Job[]>(`/api/workflow-install-offers/${encodeURIComponent(offerId)}/install`, { method: "POST" }),
  installWorkflowAssets: (
    uiGraph: Record<string, unknown>,
    selections: Array<{
      reference_filename: string;
      install_plan_id: string;
      artifact_path: string;
    }>,
    bindingPlanHash: string,
  ) =>
    request<Job[]>("/api/workflows/packages/assets/install", {
      method: "POST",
      body: JSON.stringify({
        ui_graph: uiGraph,
        selections,
        binding_plan_hash: bindingPlanHash,
      }),
    }),
  importWorkflowPackage: (payload: {
    ui_graph: Record<string, unknown>;
    name: string;
    operation: string;
    description?: string;
    dependencies?: Record<string, unknown>;
    draft_workflow_id?: string;
    draft_revision_id?: string;
  }) =>
    request<Workflow>("/api/workflows/packages/import", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  customNodes: () => request<CustomNodeInstall[]>("/api/custom-nodes"),
  installCustomNode: (payload: { name: string; source_url: string; revision: string }) =>
    request<CustomNodeInstall>("/api/custom-nodes", { method: "POST", body: JSON.stringify(payload) }),
  updateCustomNode: (id: string, revision: string) =>
    request<CustomNodeInstall>(`/api/custom-nodes/${id}`, { method: "PATCH", body: JSON.stringify({ revision }) }),
  trustCustomNode: (id: string, trusted: boolean, nodeTypes: string[] = []) =>
    request<CustomNodeInstall>(`/api/custom-nodes/${id}/trust`, {
      method: "POST",
      body: JSON.stringify({ trusted, node_types: nodeTypes }),
    }),
  rollbackCustomNode: (id: string) =>
    request<CustomNodeInstall>(`/api/custom-nodes/${id}/rollback`, { method: "POST" }),
  removeCustomNode: (id: string) => request<void>(`/api/custom-nodes/${id}`, { method: "DELETE" }),
  validateWorkflow: (id: string) =>
    request<{ valid: boolean; errors: string[]; warnings: string[]; revision_id: string }>(
      `/api/workflows/${id}/validate`,
      { method: "POST" },
    ),
  upload: async (file: File): Promise<Artifact> => {
    await ensureSession();
    const form = new FormData();
    form.append("file", file);
    const artifact = await request<Artifact>("/api/artifacts", {
      method: "POST",
      headers: { "x-local-lm-csrf": csrfToken },
      body: form,
    });
    return artifact;
  },
};

/** Read one event frame, or nothing if it is not one we can act on. */
export function readEvent(data: unknown): AppEvent | null {
  if (typeof data !== "string") return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object") return null;
  const event = parsed as Partial<AppEvent>;
  if (!Number.isFinite(event.sequence) || typeof event.type !== "string") return null;
  return event as AppEvent;
}

export async function connectEvents(
  onEvent: (event: AppEvent) => void,
  onStatus: (connected: boolean) => void,
  onReconnect?: () => void,
): Promise<() => void> {
  let closed = false;
  let opening = false;
  let socket: WebSocket | null = null;
  let retry: number | undefined;
  let connectedEpoch = eventEpoch;
  let lastSequence = eventSequence;
  let sequenceInitialized = false;
  let hasOpened = false;

  const scheduleRetry = () => {
    if (closed || retry !== undefined) return;
    retry = window.setTimeout(() => {
      retry = undefined;
      void open();
    }, 1_000);
  };

  const open = async () => {
    if (closed || opening) return;
    opening = true;
    try {
      await ensureSession();
      if (closed) return;
      if (!sequenceInitialized) {
        lastSequence = eventSequence;
        connectedEpoch = eventEpoch;
        sequenceInitialized = true;
      } else if (eventEpoch && connectedEpoch && eventEpoch !== connectedEpoch) {
        lastSequence = 0;
        connectedEpoch = eventEpoch;
      } else if (eventSequence < lastSequence) {
        lastSequence = 0;
      }
      const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${scheme}//${window.location.host}/api/events?after=${lastSequence}`);
      socket.onopen = () => {
        onStatus(true);
        if (hasOpened) onReconnect?.();
        hasOpened = true;
      };
      socket.onmessage = (message) => {
        // onmessage is called by the browser long after the try around the
        // connection has returned, so anything thrown here escapes it: the
        // stream would keep its socket open while delivering nothing, and a
        // non-numeric sequence would carry NaN into the ?after= of every
        // later reconnect. Drop the frame instead; the next one still counts.
        const event = readEvent(message.data);
        if (!event) return;
        lastSequence = Math.max(lastSequence, event.sequence);
        eventSequence = lastSequence;
        onEvent(event);
      };
      socket.onclose = () => {
        onStatus(false);
        if (closed) return;
        resetSession();
        scheduleRetry();
      };
    } catch {
      onStatus(false);
      scheduleRetry();
    } finally {
      opening = false;
    }
  };
  await open();
  return () => {
    closed = true;
    if (retry !== undefined) window.clearTimeout(retry);
    socket?.close();
  };
}
