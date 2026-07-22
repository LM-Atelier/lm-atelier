import type {
  AppEvent,
  BackupInfo,
  CatalogPage,
  CatalogDetail,
  Chat,
  ChatDetail,
  EngineCapabilities,
  Job,
  ModelInstall,
  ModelProfile,
  PlatformMatrixEntry,
  Project,
  ReferenceRecipe,
  RoutingMode,
  SystemInfo,
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
  jobs: () => request<Job[]>("/api/jobs"),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  engines: () => request<EngineCapabilities[]>("/api/engines"),
  system: () => request<SystemInfo>("/api/system"),
  platforms: () => request<PlatformMatrixEntry[]>("/api/platforms"),
  models: () => request<ModelInstall[]>("/api/models"),
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
  catalog: (query: string, role: string, sort: string) =>
    request<CatalogPage>(
      `/api/catalog?query=${encodeURIComponent(query)}&role=${encodeURIComponent(role)}&sort=${encodeURIComponent(sort)}`,
    ),
  catalogDetail: (remoteId: string) =>
    request<CatalogDetail>(`/api/catalog/${remoteId}`),
  recipes: () => request<ReferenceRecipe[]>("/api/recipes"),
  installRecipe: (recipeId: string) =>
    request<Job>(`/api/recipes/${encodeURIComponent(recipeId)}/install`, { method: "POST" }),
  download: (remoteId: string, role: string, engine: string, allowPatterns: string[] = []) =>
    request<Job>("/api/downloads", {
      method: "POST",
      body: JSON.stringify({
        remote_id: remoteId,
        revision: "main",
        role,
        engine,
        allow_patterns: allowPatterns,
      }),
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
