import type {
  ApplicationInfo,
  AppEvent,
  Artifact,
  ArtifactCleanupResult,
  ArtifactDeleteResult,
  ArtifactLibraryItem,
  ArtifactStorageInfo,
  BackupInfo,
  CatalogModel,
  CatalogPage,
  CatalogDetail,
  CatalogPreflight,
  Chat,
  ContentRating,
  ExchangeDeletion,
  ChatDetail,
  DraftClassification,
  CustomNodeInstall,
  CredentialProvider,
  CredentialStatus,
  EngineCapabilities,
  GenerationPreset,
  GenerationPresetBundle,
  Job,
  Message,
  ModelAssetInstall,
  ModelInstall,
  ModelStorageInfo,
  ModelProfile,
  ModelProfileBundle,
  PlatformMatrixEntry,
  PromptHelperDetail,
  Project,
  ReferenceRecipe,
  RoutingMode,
  RuntimeStatus,
  SetupReadinessReport,
  SetupVerification,
  SystemInfo,
  ToolCapabilityProbe,
  TurnAccepted,
  EditTemplate,
  Workflow,
  WorkflowBundle,
  WorkflowPackageAnalysis,
  WorkflowRevision,
  WorkerLogLocation,
  WorkerLogTail,
  WorkerResetResult,
  WorkerSettings,
  WorkerStatus,
  WorkPlan,
  WorkStep,
} from "./types";

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

/** Whether a 403 is the CSRF guard rather than a genuine refusal. */
async function isCsrfFailure(response: Response): Promise<boolean> {
  try {
    const payload = (await response.clone().json()) as { detail?: unknown };
    return payload.detail === "CSRF check failed";
  } catch {
    return false;
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  retrySession = true,
): Promise<T> {
  if (path !== "/api/session") await ensureSession();
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set("content-type", "application/json");
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
      return request<T>(path, init, false);
    }
  }
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    let detail: unknown;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      detail = payload.detail;
      if (typeof detail === "string") message = detail;
      else if (detail && typeof detail === "object" && "message" in detail && typeof detail.message === "string") message = detail.message;
    } catch {
      // Preserve the HTTP status text.
    }
    throw new ApiError(response.status, detail, message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  initialize: ensureSession,
  setupReadiness: () => request<SetupReadinessReport>("/api/setup/readiness"),
  verifySetupRole: (role: SetupVerification["role"]) =>
    request<SetupVerification>(`/api/setup/verify/${role}`, { method: "POST" }),
  projects: (includeArchived = false, query = "") =>
    request<Project[]>(`/api/projects?${new URLSearchParams({ include_archived: String(includeArchived), query })}`),
  createProject: (name: string) =>
    request<Project>("/api/projects", { method: "POST", body: JSON.stringify({ name }) }),
  updateProject: (id: string, values: Partial<Project>) =>
    request<Project>(`/api/projects/${id}`, { method: "PATCH", body: JSON.stringify(values) }),
  deleteProject: (id: string) => request<void>(`/api/projects/${id}`, { method: "DELETE" }),
  chats: (projectId?: string | null, includeArchived = false, query = "") => {
    const parameters = new URLSearchParams({ include_archived: String(includeArchived), query });
    if (projectId) parameters.set("project_id", projectId);
    return request<Chat[]>(`/api/chats?${parameters}`);
  },
  chat: (id: string) => request<ChatDetail>(`/api/chats/${id}`),
  classifyDraft: (chatId: string, text: string, mode: RoutingMode) =>
    request<DraftClassification>(`/api/chats/${chatId}/classify-draft`, {
      method: "POST",
      body: JSON.stringify({ text, mode }),
    }),
  createChat: (projectId?: string | null) =>
    request<Chat>("/api/chats", {
      method: "POST",
      body: JSON.stringify({ title: "New chat", project_id: projectId ?? null }),
    }),
  updateChat: (id: string, values: Partial<Chat>) =>
    request<Chat>(`/api/chats/${id}`, { method: "PATCH", body: JSON.stringify(values) }),
  deleteChat: (id: string, deleteGeneratedMedia = false) => {
    const parameters = new URLSearchParams({
      delete_generated_media: String(deleteGeneratedMedia),
    });
    return request<void>(`/api/chats/${id}?${parameters}`, { method: "DELETE" });
  },  createPromptHelper: (sourceChatId: string, draftPrompt: string) =>
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
  deletePromptHelper: (id: string) =>
    request<void>(`/api/prompt-helpers/${id}`, { method: "DELETE" }),
  sendTurn: async (
    chatId: string,
    text: string,
    mode: RoutingMode,
    inputArtifactIds: string[],
    settings: Record<string, unknown>,
    idempotencyKey: string = crypto.randomUUID(),
    endpoint: string = "turns",
  ) => {
    const submit = (selectedMode: RoutingMode, confirmed = false) => request<TurnAccepted>(`/api/chats/${chatId}/${endpoint}`, {
      method: "POST",
      body: JSON.stringify({
        text,
        mode: selectedMode,
        input_artifact_ids: inputArtifactIds,
        settings,
        confirm_media: confirmed,
        idempotency_key: idempotencyKey,
      }),
    });
    try {
      return await submit(mode);
    } catch (error) {
      const detail = error instanceof ApiError && error.detail && typeof error.detail === "object" ? error.detail as Record<string, unknown> : null;
      const plan = detail?.plan && typeof detail.plan === "object" ? detail.plan as Record<string, unknown> : null;
      const operation = typeof plan?.operation === "string" ? plan.operation : "";
      const estimate = detail?.estimate && typeof detail.estimate === "object" ? detail.estimate as Record<string, unknown> : null;
      const duration = typeof estimate?.duration_seconds === "number" ? `, about ${estimate.duration_seconds} seconds of output` : "";
      const intermediate = typeof estimate?.estimated_intermediate_bytes === "number" ? ` and up to ${Math.ceil(estimate.estimated_intermediate_bytes / 1024 ** 3)} GB of intermediate data` : "";
      const orderedSteps = Array.isArray(plan?.steps)
        ? plan.steps.filter((step): step is Record<string, unknown> => (
          Boolean(step) && typeof step === "object"
        ))
        : [];
      if (
        error instanceof ApiError
        && error.status === 409
        && detail?.code === "ordered_plan_confirmation_required"
        && orderedSteps.length >= 2
      ) {
        const sequence = orderedSteps
          .map((step) => typeof step.mode === "string" ? step.mode : "work")
          .join(" → ");
        const orderedEstimate = detail.estimate && typeof detail.estimate === "object"
          ? detail.estimate as Record<string, unknown>
          : null;
        const videoDuration = typeof orderedEstimate?.video_duration_seconds === "number"
          && orderedEstimate.video_duration_seconds > 0
          ? ` · about ${orderedEstimate.video_duration_seconds} seconds of video`
          : "";
        const workingBytes = typeof orderedEstimate?.estimated_bytes === "number"
          && orderedEstimate.estimated_bytes > 0
          ? ` · up to ${Math.ceil(orderedEstimate.estimated_bytes / 1024 ** 3)} GB working space`
          : "";
        if (window.confirm(
          `${orderedSteps.length}-step plan: ${sequence}${videoDuration}${workingBytes}. Start it?`,
        )) {
          return submit("auto", true);
        }
      }
      if (
        error instanceof ApiError
        && error.status === 409
        && detail?.code === "route_confirmation_required"
        && (operation.includes("image") || operation.includes("video"))
        && window.confirm(`Auto mode suggests a ${operation.includes("video") ? "video" : "image"} generation${duration}${intermediate}. Start it?`)
      ) {
        return submit(operation.includes("video") ? "video" : "image", true);
      }
      throw error;
    }
  },
  stopAndSendTurn: (
    chatId: string,
    text: string,
    mode: RoutingMode,
    inputArtifactIds: string[],
    settings: Record<string, unknown>,
    idempotencyKey: string = crypto.randomUUID(),
  ) => api.sendTurn(
    chatId,
    text,
    mode,
    inputArtifactIds,
    settings,
    idempotencyKey,
    "stop-and-send",
  ),
  regenerateMessage: (messageId: string, settings: Record<string, unknown>) =>
    request<TurnAccepted>(`/api/messages/${messageId}/regenerate`, {
      method: "POST",
      body: JSON.stringify({ settings }),
    }),
  forkThread: (messageId: string) =>
    request<Chat>(`/api/messages/${messageId}/fork`, { method: "POST" }),
  deleteExchange: (messageId: string) =>
    request<ExchangeDeletion>(`/api/messages/${messageId}/exchange`, { method: "DELETE" }),
  selectResponseRevision: (messageId: string, revisionId: string) =>
    request<Message>(`/api/messages/${messageId}/revisions/${revisionId}/select`, {
      method: "POST",
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
  workPlans: (chatId?: string) =>
    request<WorkPlan[]>(
      `/api/work-plans${chatId ? `?chat_id=${encodeURIComponent(chatId)}` : ""}`,
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
  modelStorage: () => request<ModelStorageInfo>("/api/models/storage"),
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
  updateProfile: (
    id: string,
    values: {
      name?: string;
      use_case?: string;
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
  resetProfile: (id: string) =>
    request<ModelProfile>(`/api/profiles/${id}/reset`, { method: "POST" }),
  deleteProfile: (id: string) => request<void>(`/api/profiles/${id}`, { method: "DELETE" }),
  exportProfile: (id: string) => request<ModelProfileBundle>(`/api/profiles/${id}/export`),
  importProfile: (bundle: ModelProfileBundle) =>
    request<ModelProfile>("/api/profiles/import", { method: "POST", body: JSON.stringify(bundle) }),
  presets: () => request<GenerationPreset[]>("/api/presets"),
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
  artifacts: (kind = "", query = "") => {
    const parameters = new URLSearchParams({ query });
    if (kind) parameters.set("kind", kind);
    return request<ArtifactLibraryItem[]>(`/api/artifacts?${parameters}`);
  },
  artifactStorage: () => request<ArtifactStorageInfo>("/api/artifacts/storage"),
  cleanupArtifacts: (dryRun: boolean) =>
    request<ArtifactCleanupResult>("/api/artifacts/cleanup", {
      method: "POST",
      body: JSON.stringify({ dry_run: dryRun }),
    }),
  deleteArtifact: (artifactId: string) =>
    request<ArtifactDeleteResult>(`/api/artifacts/${encodeURIComponent(artifactId)}`, {
      method: "DELETE",
    }),
  catalog: (
    query: string,
    role: string,
    sort: string,
    cursor?: string | null,
    filters: Record<string, string> = {},
  ) => {
    const parameters = new URLSearchParams({ query, role, sort });
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
  ) => request<CatalogPreflight>(`/api/catalog/${remoteId}/preflight`, {
    method: "POST",
    body: JSON.stringify({
      role,
      engine,
      revision,
      selected_files: selectedFiles,
      auxiliary_kind: auxiliaryKind,
      workflow_template_id: workflowTemplateId,
    }),
  }),
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
    >>,
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
  createWorkflow: (payload: Record<string, unknown>) =>
    request<Workflow>("/api/workflows", { method: "POST", body: JSON.stringify(payload) }),
  updateWorkflow: (id: string, payload: Record<string, unknown>) =>
    request<Workflow>(`/api/workflows/${id}`, { method: "PATCH", body: JSON.stringify(payload) }),
  createWorkflowRevision: (id: string, payload: Record<string, unknown>) =>
    request<WorkflowRevision>(`/api/workflows/${id}/revisions`, { method: "POST", body: JSON.stringify(payload) }),
  restoreWorkflowRevision: (id: string, revisionId: string) =>
    request<WorkflowRevision>(`/api/workflows/${id}/revisions/${revisionId}/restore`, { method: "POST" }),
  cloneWorkflow: (id: string, name?: string) =>
    request<Workflow>(`/api/workflows/${id}/clone`, { method: "POST", body: JSON.stringify({ name }) }),
  exportWorkflow: (id: string) => request<WorkflowBundle>(`/api/workflows/${id}/export`),
  workflowOpenTarget: (id: string) => request<{ url: string; filename: string; ui_graph: Record<string, unknown> }>(`/api/workflows/${id}/open-target`),
  importWorkflow: (bundle: WorkflowBundle) =>
    request<Workflow>("/api/workflows/import", { method: "POST", body: JSON.stringify(bundle) }),
  editTemplates: () => request<EditTemplate[]>("/api/edit-templates"),
  createEditTemplate: (payload: { name: string; description?: string; instruction: string; settings_json?: Record<string, unknown> }) =>
    request<EditTemplate>("/api/edit-templates", { method: "POST", body: JSON.stringify(payload) }),
  deleteEditTemplate: (id: string) =>
    request<void>(`/api/edit-templates/${id}`, { method: "DELETE" }),
  analyzeWorkflowPackage: (uiGraph: Record<string, unknown>) =>
    request<WorkflowPackageAnalysis>("/api/workflows/packages/analyze", {
      method: "POST",
      body: JSON.stringify({ ui_graph: uiGraph }),
    }),
  customNodes: () => request<CustomNodeInstall[]>("/api/custom-nodes"),
  installCustomNode: (payload: { name: string; source_url: string; revision: string }) =>
    request<CustomNodeInstall>("/api/custom-nodes", { method: "POST", body: JSON.stringify(payload) }),
  updateCustomNode: (id: string, revision: string) =>
    request<CustomNodeInstall>(`/api/custom-nodes/${id}`, { method: "PATCH", body: JSON.stringify({ revision }) }),
  trustCustomNode: (id: string, trusted: boolean) =>
    request<CustomNodeInstall>(`/api/custom-nodes/${id}/trust`, { method: "POST", body: JSON.stringify({ trusted }) }),
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
        const event = JSON.parse(message.data as string) as AppEvent;
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
