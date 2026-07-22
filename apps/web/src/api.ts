import type {
  AppEvent,
  BackupInfo,
  CatalogPage,
  CatalogDetail,
  CatalogPreflight,
  Chat,
  ChatDetail,
  EngineCapabilities,
  GenerationPreset,
  GenerationPresetBundle,
  Job,
  ModelInstall,
  ModelStorageInfo,
  ModelProfile,
  ModelProfileBundle,
  PlatformMatrixEntry,
  Project,
  ReferenceRecipe,
  RoutingMode,
  SystemInfo,
  ToolCapabilityProbe,
  TurnAccepted,
  Workflow,
  WorkerStatus,
} from "./types";

let csrfToken = "";
let sessionPromise: Promise<void> | null = null;

async function ensureSession(): Promise<void> {
  if (csrfToken) return;
  if (!sessionPromise) {
    sessionPromise = fetch("/api/session", {
      method: "POST",
      credentials: "same-origin",
    }).then(async (response) => {
      if (!response.ok) throw new Error("Could not initialize the local session");
      const payload = (await response.json()) as { csrf_token: string };
      csrfToken = payload.csrf_token;
    });
  }
  await sessionPromise;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  if (path !== "/api/session") await ensureSession();
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set("content-type", "application/json");
  if (init.method && !["GET", "HEAD"].includes(init.method.toUpperCase())) {
    headers.set("x-local-lm-csrf", csrfToken);
  }
  const response = await fetch(path, { ...init, headers, credentials: "same-origin" });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Preserve the HTTP status text.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  initialize: ensureSession,
  projects: () => request<Project[]>("/api/projects"),
  createProject: (name: string) =>
    request<Project>("/api/projects", { method: "POST", body: JSON.stringify({ name }) }),
  updateProject: (id: string, values: Partial<Project>) =>
    request<Project>(`/api/projects/${id}`, { method: "PATCH", body: JSON.stringify(values) }),
  chats: (projectId?: string | null) =>
    request<Chat[]>(`/api/chats${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`),
  chat: (id: string) => request<ChatDetail>(`/api/chats/${id}`),
  createChat: (projectId?: string | null) =>
    request<Chat>("/api/chats", {
      method: "POST",
      body: JSON.stringify({ title: "New chat", project_id: projectId ?? null }),
    }),
  updateChat: (id: string, values: Partial<Chat>) =>
    request<Chat>(`/api/chats/${id}`, { method: "PATCH", body: JSON.stringify(values) }),
  sendTurn: (
    chatId: string,
    text: string,
    mode: RoutingMode,
    inputArtifactIds: string[],
    settings: Record<string, unknown>,
  ) =>
    request<TurnAccepted>(`/api/chats/${chatId}/turns`, {
      method: "POST",
      body: JSON.stringify({
        text,
        mode,
        input_artifact_ids: inputArtifactIds,
        settings,
        idempotency_key: crypto.randomUUID(),
      }),
    }),
  regenerateMessage: (messageId: string) =>
    request<TurnAccepted>(`/api/messages/${messageId}/regenerate`, {
      method: "POST",
      body: JSON.stringify({ settings: {} }),
    }),
  branchMessage: (messageId: string, text: string) =>
    request<TurnAccepted>(`/api/messages/${messageId}/branch`, {
      method: "POST",
      body: JSON.stringify({ text, idempotency_key: crypto.randomUUID() }),
    }),
  cancelChat: (chatId: string) =>
    request<Job>(`/api/chats/${chatId}/cancel`, { method: "POST" }),
  jobs: () => request<Job[]>("/api/jobs"),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  pauseDownload: (id: string) =>
    request<Job>(`/api/downloads/${id}/pause`, { method: "POST" }),
  resumeDownload: (id: string) =>
    request<Job>(`/api/downloads/${id}/resume`, { method: "POST" }),
  engines: () => request<EngineCapabilities[]>("/api/engines"),
  probeChatTools: () =>
    request<ToolCapabilityProbe>("/api/engines/chat/tool-probe", { method: "POST" }),
  system: () => request<SystemInfo>("/api/system"),
  platforms: () => request<PlatformMatrixEntry[]>("/api/platforms"),
  models: () => request<ModelInstall[]>("/api/models"),
  modelStorage: () => request<ModelStorageInfo>("/api/models/storage"),
  deleteModel: (id: string) => request<void>(`/api/models/${id}`, { method: "DELETE" }),
  cleanupDownloads: () =>
    request<{ removed_count: number; reclaimed_bytes: number }>("/api/downloads/cleanup", {
      method: "POST",
    }),
  profiles: () => request<ModelProfile[]>("/api/profiles"),
  createProfile: (model: ModelInstall) =>
    request<ModelProfile>("/api/profiles", {
      method: "POST",
      body: JSON.stringify({
        name: model.name,
        role: model.role,
        engine: model.engine,
        model_install_id: model.id,
        load_settings: {},
        request_settings: {},
        is_default: false,
      }),
    }),
  updateProfile: (
    id: string,
    values: {
      name?: string;
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
  loadChatWorker: (profileId: string) =>
    request<WorkerStatus>(`/api/workers/chat/load/${profileId}`, { method: "POST" }),
  startMediaWorker: () => request<WorkerStatus>("/api/workers/media/start", { method: "POST" }),
  stopWorker: (name: "chat" | "media") =>
    request<WorkerStatus>(`/api/workers/${name}/stop`, { method: "POST" }),
  backups: () => request<BackupInfo[]>("/api/backups"),
  createBackup: () => request<BackupInfo>("/api/backups", { method: "POST" }),
  exportProject: (projectId: string) =>
    request<{ url: string }>(`/api/projects/${projectId}/export`, { method: "POST" }),
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
  catalogDetail: (remoteId: string, role: string, revision = "main") =>
    request<CatalogDetail>(`/api/catalog/${remoteId}?${new URLSearchParams({ role, revision })}`),
  catalogPreflight: (
    remoteId: string,
    role: string,
    engine: string,
    revision: string,
    selectedFiles: string[],
  ) => request<CatalogPreflight>(`/api/catalog/${remoteId}/preflight`, {
    method: "POST",
    body: JSON.stringify({ role, engine, revision, selected_files: selectedFiles }),
  }),
  recipes: () => request<ReferenceRecipe[]>("/api/recipes"),
  installRecipe: (recipeId: string) =>
    request<Job>(`/api/recipes/${encodeURIComponent(recipeId)}/install`, { method: "POST" }),
  download: (remoteId: string, role: string, engine: string, revision: string, allowPatterns: string[] = []) =>
    request<Job>("/api/downloads", {
      method: "POST",
      body: JSON.stringify({
        remote_id: remoteId,
        revision,
        role,
        engine,
        allow_patterns: allowPatterns,
      }),
    }),
  importModel: (payload: { name: string; role: string; engine: string; local_path: string }) =>
    request<ModelInstall>("/api/models/import", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  workflows: () => request<Workflow[]>("/api/workflows"),
  createWorkflow: (payload: Record<string, unknown>) =>
    request<Workflow>("/api/workflows", { method: "POST", body: JSON.stringify(payload) }),
  validateWorkflow: (id: string) =>
    request<{ valid: boolean; errors: string[]; revision_id: string }>(
      `/api/workflows/${id}/validate`,
      { method: "POST" },
    ),
  upload: async (file: File): Promise<string> => {
    await ensureSession();
    const form = new FormData();
    form.append("file", file);
    const artifact = await request<{ id: string }>("/api/artifacts", {
      method: "POST",
      headers: { "x-local-lm-csrf": csrfToken },
      body: form,
    });
    return artifact.id;
  },
};

export async function connectEvents(
  onEvent: (event: AppEvent) => void,
  onStatus: (connected: boolean) => void,
): Promise<() => void> {
  await ensureSession();
  let closed = false;
  let socket: WebSocket | null = null;
  let retry: number | undefined;
  let lastSequence = 0;

  const open = () => {
    if (closed) return;
    const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${scheme}//${window.location.host}/api/events?after=${lastSequence}`);
    socket.onopen = () => onStatus(true);
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data as string) as AppEvent;
      lastSequence = Math.max(lastSequence, event.sequence);
      onEvent(event);
    };
    socket.onclose = () => {
      onStatus(false);
      if (!closed) retry = window.setTimeout(open, 1_000);
    };
  };
  open();
  return () => {
    closed = true;
    if (retry) window.clearTimeout(retry);
    socket?.close();
  };
}
