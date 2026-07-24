import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api, connectEvents } from "./api";
import type { EngineCapabilities, SettingField } from "./types";

const imageSetting: SettingField = {
  key: "negative_prompt",
  label: "Negative prompt",
  type: "string",
  default: "",
  minimum: null,
  maximum: null,
  step: null,
  choices: [],
  scope: "workflow",
  visibility: "basic",
  restart_required: false,
  available: true,
  unavailable_reason: null,
  help: "Exclude unwanted image details.",
};

const videoSetting: SettingField = {
  ...imageSetting,
  key: "frames",
  label: "Frames",
  type: "integer",
  default: 49,
  help: "Number of video frames.",
};

const maxTokensSetting: SettingField = {
  ...imageSetting,
  key: "max_tokens",
  label: "Maximum output",
  type: "integer",
  default: 1024,
  minimum: 1,
  maximum: 131072,
  scope: "request",
  help: "Maximum tokens generated for one assistant run.",
};

const contextLengthSetting: SettingField = {
  ...maxTokensSetting,
  key: "context_length",
  label: "Context length",
  default: 8192,
  minimum: 512,
  maximum: 1048576,
  scope: "load",
  restart_required: true,
  help: "Maximum tokens held in the model context.",
};

const roleAwareMediaEngine: EngineCapabilities = {
  engine: "mock",
  version: "1",
  roles: ["image", "video"],
  operations: ["text_to_image", "text_to_video"],
  formats: ["mock"],
  devices: ["cpu:0"],
  streaming: false,
  tool_calling: false,
  settings: [imageSetting, videoSetting],
  settings_by_role: { image: [imageSetting], video: [videoSetting] },
  healthy: true,
  details: {},
};

vi.mock("./api", () => ({
  api: {
    initialize: vi.fn().mockResolvedValue(undefined),
    projects: vi.fn().mockResolvedValue([]),
    chats: vi.fn().mockResolvedValue([]),
    chat: vi.fn(),
    createProject: vi.fn(),
    updateProject: vi.fn(),
    deleteProject: vi.fn(),
    createChat: vi.fn(),
    updateChat: vi.fn(),
    deleteChat: vi.fn(),
    exportProject: vi.fn(),
    importProject: vi.fn(),
    artifacts: vi.fn().mockResolvedValue([]),
    artifactStorage: vi.fn().mockResolvedValue({ total_bytes: 0, total_count: 0, referenced_bytes: 0, referenced_count: 0, unreferenced_bytes: 0, unreferenced_count: 0, temporary_bytes: 0, temporary_count: 0, eligible_bytes: 0, eligible_count: 0, disk_free_bytes: 1024, warning: false, retention_days: 30, temporary_retention_hours: 24 }),
    cleanupArtifacts: vi.fn(),
    sendTurn: vi.fn(),
    regenerateMessage: vi.fn(),
    branchMessage: vi.fn(),
    cancelChat: vi.fn(),
    jobs: vi.fn().mockResolvedValue([]),
    cancelJob: vi.fn(),
    pauseDownload: vi.fn(),
    resumeDownload: vi.fn(),
    engines: vi.fn().mockResolvedValue([
      {
        engine: "mock",
        version: "1",
        roles: ["chat", "image", "video"],
        operations: ["text", "text_to_image", "text_to_video"],
        formats: ["mock"],
        devices: ["cpu:0"],
        streaming: true,
        tool_calling: true,
        settings: [],
        healthy: true,
        details: {},
      },
    ]),
    probeChatTools: vi.fn(),
    system: vi.fn().mockResolvedValue({
      platform: "Linux",
      platform_release: "6.8",
      distribution: "Ubuntu",
      distribution_version: "24.04",
      architecture: "x86_64",
      python_version: "3.12",
      cpu_model: "Test CPU 9000",
      cpu_count: 16,
      memory_total_bytes: 32 * 1024 ** 3,
      memory_available_bytes: 16 * 1024 ** 3,
      disk_total_bytes: 1024 ** 4,
      disk_free_bytes: 512 * 1024 ** 3,
      ffmpeg_available: true,
      devices: [],
      support: {
        platform_status: "target",
        platform_label: "Ubuntu 24.04 LTS x64 target",
        accelerator_status: "cpu-only",
        accelerator_label: "CPU fallback",
        certification_status: "hardware-pending",
        chat_ready: true,
        reference_media_ready: false,
        vram_tier_gb: null,
        messages: ["No primary media accelerator was detected."],
      },
    }),
    platforms: vi.fn().mockResolvedValue([]),
    createDiagnostics: vi.fn(),
    credentialStatus: vi.fn().mockResolvedValue({ provider: "huggingface", configured: false, source: "none", vault_available: true }),
    setHuggingFaceToken: vi.fn(),
    deleteHuggingFaceToken: vi.fn(),
    models: vi.fn(),
    modelStorage: vi.fn().mockResolvedValue({ installed_bytes: 0, partial_download_bytes: 0, catalog_cache_bytes: 0, installed_count: 0, partial_download_count: 0 }),
    deleteModel: vi.fn(),
    cleanupDownloads: vi.fn(),
    profiles: vi.fn().mockResolvedValue([]),
    updateProfile: vi.fn(),
    cloneProfile: vi.fn(),
    resetProfile: vi.fn(),
    deleteProfile: vi.fn(),
    exportProfile: vi.fn(),
    importProfile: vi.fn(),
    presets: vi.fn().mockResolvedValue([]),
    createPreset: vi.fn(),
    updatePreset: vi.fn(),
    clonePreset: vi.fn(),
    resetPreset: vi.fn(),
    exportPreset: vi.fn(),
    importPreset: vi.fn(),
    deletePreset: vi.fn(),
    workers: vi.fn().mockResolvedValue([]),
    backups: vi.fn().mockResolvedValue([]),
    loadChatWorker: vi.fn(),
    startMediaWorker: vi.fn(),
    stopWorker: vi.fn(),
    createBackup: vi.fn(),
    catalog: vi.fn(),
    catalogDetail: vi.fn(),
    catalogPreflight: vi.fn(),
    recipes: vi.fn().mockResolvedValue([]),
    installRecipe: vi.fn(),
    download: vi.fn(),
    importModel: vi.fn(),
    workflows: vi.fn(),
    createWorkflow: vi.fn(),
    updateWorkflow: vi.fn(),
    createWorkflowRevision: vi.fn(),
    restoreWorkflowRevision: vi.fn(),
    cloneWorkflow: vi.fn(),
    exportWorkflow: vi.fn(),
    workflowOpenTarget: vi.fn(),
    importWorkflow: vi.fn(),
    validateWorkflow: vi.fn(),
    customNodes: vi.fn().mockResolvedValue([]),
    installCustomNode: vi.fn(),
    updateCustomNode: vi.fn(),
    trustCustomNode: vi.fn(),
    rollbackCustomNode: vi.fn(),
    removeCustomNode: vi.fn(),
    upload: vi.fn(),
  },
  connectEvents: vi.fn().mockImplementation(async (_onEvent, onStatus) => {
    onStatus(true);
    return () => undefined;
  }),
}));

describe("App", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    vi.mocked(api.profiles).mockResolvedValue([]);
    vi.mocked(api.chats).mockResolvedValue([]);
    vi.mocked(api.workers).mockResolvedValue([]);
    vi.mocked(api.models).mockResolvedValue([]);
    vi.mocked(api.catalog).mockResolvedValue({ items: [], next_cursor: null });
    vi.mocked(api.workflows).mockResolvedValue([]);
    vi.mocked(api.customNodes).mockResolvedValue([]);
  });
  afterEach(cleanup);

  it("renders the local workspace shell without an existing chat", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("LM Atelier")).toBeInTheDocument();
    expect(await screen.findByText("Start a local conversation")).toBeInTheDocument();
    expect(screen.getByText("Model library")).toBeInTheDocument();
    expect(screen.queryByText("Local service connected")).not.toBeInTheDocument();
    expect(screen.getByText("Skip to main content")).toHaveAttribute("href", "#main-content");
    const navigation = screen.getByRole("button", { name: "Toggle navigation" });
    expect(navigation).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(navigation);
    expect(navigation).toHaveAttribute("aria-expanded", "true");
  });

  it("refreshes the visible chat when media generation progress changes", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidate = vi.spyOn(client, "invalidateQueries");
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("LM Atelier")).toBeInTheDocument();
    const onEvent = vi.mocked(connectEvents).mock.calls.at(-1)?.[0];
    expect(onEvent).toBeDefined();

    act(() => {
      onEvent?.({
        sequence: 1,
        type: "generation.progress",
        entity_id: "run-1",
        payload: { progress: 0.5, phase: "sampling", job_id: "job-1" },
        created_at: "2026-07-23T00:00:00Z",
      });
    });

    await waitFor(() => {
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["jobs"] });
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["chat"] });
    });
  });

  it("searches and manages chats from the workspace sidebar", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    vi.mocked(api.projects).mockResolvedValue([{ id: "project-1", name: "Research", description: "", instructions: "", archived: false, image_workflow_revision_id: null, video_workflow_revision_id: null, created_at: stamp, updated_at: stamp }]);
    const chat = { id: "chat-1", project_id: "project-1", title: "Model notes", archived: false, routing_mode: "auto" as const, confirm_uncertain_media: false, active_chat_profile_id: null, active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: null, created_at: stamp, updated_at: stamp };
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    vi.mocked(api.updateChat).mockResolvedValue({ id: "chat-1", project_id: null, title: "Renamed notes", archived: true, routing_mode: "auto", confirm_uncertain_media: false, active_chat_profile_id: null, active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: null, created_at: stamp, updated_at: stamp });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("Model notes")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Search projects and chats"), { target: { value: "notes" } });
    fireEvent.click(screen.getByRole("button", { name: "Manage Model notes" }));
    fireEvent.change(screen.getByDisplayValue("Model notes"), { target: { value: "Renamed notes" } });
    fireEvent.change(screen.getByLabelText("Project"), { target: { value: "" } });
    fireEvent.click(screen.getByRole("checkbox", { name: /Archived/ }));
    fireEvent.click(screen.getByText("Save chat"));
    await waitFor(() => expect(vi.mocked(api.updateChat).mock.calls[0]?.[0]).toBe("chat-1"));
    expect(vi.mocked(api.updateChat).mock.calls[0]?.[1]).toMatchObject({ title: "Renamed notes", project_id: null, archived: true });
  });

  it("contains long chat lists in a dedicated workspace scroll region", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    vi.mocked(api.chats).mockResolvedValue(
      Array.from({ length: 40 }, (_, index) => ({
        id: `chat-${index}`,
        project_id: null,
        title: `Diagnostic chat ${index + 1}`,
        archived: false,
        routing_mode: "auto" as const,
        confirm_uncertain_media: false,
        active_chat_profile_id: null,
        active_image_profile_id: null,
        active_video_profile_id: null,
        active_head_message_id: null,
        created_at: stamp,
        updated_at: stamp,
      })),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const workspace = await screen.findByRole("region", { name: "Projects and chats" });
    expect(workspace).toHaveClass("workspace-tree");
    await waitFor(() => expect(workspace.querySelectorAll(".sidebar-chat-row")).toHaveLength(40));
    expect(workspace).toContainElement(screen.getByText("Diagnostic chat 40"));
    expect(workspace).not.toContainElement(screen.getByLabelText("Search projects and chats"));
    expect(workspace).not.toContainElement(screen.getByRole("button", { name: "Settings" }));
  });

  it("imports portable project archives from the workspace sidebar", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    vi.mocked(api.importProject).mockResolvedValue({ id: "project-imported", name: "Imported", description: "", instructions: "", archived: false, image_workflow_revision_id: null, video_workflow_revision_id: null, created_at: stamp, updated_at: stamp });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    const file = new File(["archive"], "portable.lm-atelier.zip", { type: "application/zip" });
    const input = container.querySelector<HTMLInputElement>('input[accept*=".lm-atelier.zip"]');
    expect(input).not.toBeNull();
    fireEvent.change(input!, { target: { files: [file] } });
    await waitFor(() => expect(vi.mocked(api.importProject).mock.calls[0]?.[0]).toBe(file));
  });

  it("exports projects with or without embedded media", async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    const stamp = "2026-07-22T00:00:00Z";
    vi.mocked(api.projects).mockResolvedValue([{ id: "project-1", name: "Portable", description: "", instructions: "", archived: false, image_workflow_revision_id: null, video_workflow_revision_id: null, created_at: stamp, updated_at: stamp }]);
    vi.mocked(api.exportProject).mockResolvedValue({ url: "/api/artifacts/export/content" });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Manage Portable" }));
    fireEvent.click(await screen.findByText("Export metadata only"));
    await waitFor(() => expect(vi.mocked(api.exportProject).mock.calls[0]).toEqual(["project-1", false]));
    click.mockRestore();
  });

  it("resumes a paused model download from the job panel", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const pausedJob = {
      id: "download-1",
      kind: "download",
      status: "paused",
      run_id: null,
      progress: 0.42,
      phase: "paused",
      payload_json: {},
      result_json: {},
      error: null,
      attempt: 1,
      cancellable: true,
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.jobs).mockResolvedValue([pausedJob]);
    vi.mocked(api.resumeDownload).mockResolvedValue({
      ...pausedJob,
      status: "queued",
      phase: "resume queued",
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByRole("button", { name: "Resume download" }));
    await waitFor(() => expect(vi.mocked(api.resumeDownload).mock.calls[0]?.[0]).toBe("download-1"));
  });

  it("shows useful machine details without platform-status clutter", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByText("Settings"));
    expect(await screen.findByText("Test CPU 9000")).toBeInTheDocument();
    expect(screen.getByText("16 logical processors")).toBeInTheDocument();
    expect(screen.queryByText("Platform support")).not.toBeInTheDocument();
  });

  it("stores a Hugging Face token without echoing it back", async () => {
    vi.mocked(api.setHuggingFaceToken).mockResolvedValue({
      provider: "huggingface",
      configured: true,
      source: "credential_vault",
      vault_available: true,
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Settings"));
    const input = await screen.findByLabelText("Hugging Face access token");
    fireEvent.change(input, { target: { value: "temporary-token" } });
    fireEvent.click(screen.getByText("Save token"));
    await waitFor(() => expect(api.setHuggingFaceToken).toHaveBeenCalledWith("temporary-token"));
    await waitFor(() => expect(input).toHaveValue(""));
    expect(screen.queryByDisplayValue("temporary-token")).not.toBeInTheDocument();
    expect(await screen.findByText("Configured · credential vault")).toBeInTheDocument();
  });

  it("opens a model profile in the schema-driven settings editor", async () => {
    vi.mocked(api.profiles).mockResolvedValue([
      {
        id: "profile-1",
        model_install_id: "model-1",
        name: "Local chat",
        use_case: "",
        role: "chat",
        engine: "mock",
        load_settings_json: {},
        request_settings_json: {},
        is_default: true,
      },
    ]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Settings"));
    expect(await screen.findByText(/Default.*default/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    expect(await screen.findByText("Edit profile")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Local chat")).toBeInTheDocument();
    expect(screen.getByText("Default model")).toBeInTheDocument();
  });

  it("renders Markdown on only the active edited branch", async () => {
    localStorage.setItem("local-lm-chat", "chat-1");
    const stamp = "2026-07-22T00:00:00Z";
    const message = (id: string, parentId: string | null, role: "user" | "assistant", text: string) => ({
      id,
      chat_id: "chat-1",
      parent_id: parentId,
      role,
      status: "complete" as const,
      parts: [
        { id: `${id}-part`, position: 0, type: "text" as const, text, artifact_id: null, metadata_json: {} },
        ...(role === "assistant" ? [{
          id: `${id}-metadata`,
          position: 1,
          type: "generation_metadata" as const,
          text: null,
          artifact_id: null,
          metadata_json: {
            provenance: {
              model_selection: { mode: "auto", profile_name: "Code specialist" },
            },
          },
        }] : []),
      ],
      created_at: stamp,
      updated_at: stamp,
    });
    vi.mocked(api.chat).mockResolvedValue({
      id: "chat-1",
      project_id: null,
      title: "Branches",
      archived: false,
      routing_mode: "auto",
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "a3",
      created_at: stamp,
      updated_at: stamp,
      messages: [
        message("u1", null, "user", "First question"),
        message("a1", "u1", "assistant", "First answer"),
        message("u2", "a1", "user", "Old follow-up"),
        message("a2", "u2", "assistant", "# Old branch"),
        message("u3", "a1", "user", "Edited question"),
        message("a3", "u3", "assistant", "# Edited answer"),
      ],
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByRole("heading", { name: "Edited answer" })).toBeInTheDocument();
    expect(screen.queryByText("Old follow-up")).not.toBeInTheDocument();
    expect(screen.queryByText("Old branch")).not.toBeInTheDocument();
    expect(screen.getAllByText("Auto chose Code specialist")).toHaveLength(2);
    fireEvent.click(screen.getAllByText("Edit and branch").at(-1)!);
    expect(screen.getByDisplayValue("Edited question")).toBeInTheDocument();
  });

  it("keeps cancelled assistant text above subdued cancellation metadata", async () => {
    localStorage.setItem("local-lm-chat", "chat-cancelled");
    const stamp = "2026-07-22T00:00:00Z";
    const chat = {
      id: "chat-cancelled",
      project_id: null,
      title: "Cancelled stream",
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-cancelled",
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({
      ...chat,
      messages: [
        {
          id: "user-cancelled",
          chat_id: chat.id,
          parent_id: null,
          role: "user",
          status: "complete",
          parts: [{ id: "prompt", position: 0, type: "text", text: "Keep counting", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
        {
          id: "assistant-cancelled",
          chat_id: chat.id,
          parent_id: "user-cancelled",
          role: "assistant",
          status: "cancelled",
          parts: [{ id: "partial", position: 0, type: "text", text: "1 2 3 4 5", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
      ],
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("1 2 3 4 5")).toBeInTheDocument();
    const cancellation = screen.getByText("Generation cancelled");
    expect(cancellation.closest(".message-meta")).not.toBeNull();
    expect(cancellation.closest(".message-error")).toBeNull();
  });

  it("keeps failed assistant text above its error", async () => {
    localStorage.setItem("local-lm-chat", "chat-failed");
    const stamp = "2026-07-23T00:00:00Z";
    const chat = {
      id: "chat-failed",
      project_id: null,
      title: "Failed stream",
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-failed",
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({
      ...chat,
      messages: [
        {
          id: "user-failed",
          chat_id: chat.id,
          parent_id: null,
          role: "user",
          status: "complete",
          parts: [{ id: "prompt", position: 0, type: "text", text: "Keep counting", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
        {
          id: "assistant-failed",
          chat_id: chat.id,
          parent_id: "user-failed",
          role: "assistant",
          status: "failed",
          parts: [
            { id: "partial", position: 0, type: "text", text: "1 2 3 4 5", artifact_id: null, metadata_json: {} },
            { id: "error", position: 1, type: "error", text: "llama.cpp stream failed: ReadError", artifact_id: null, metadata_json: {} },
          ],
          created_at: stamp,
          updated_at: stamp,
        },
      ],
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("1 2 3 4 5")).toBeInTheDocument();
    const error = screen.getByText("llama.cpp stream failed: ReadError");
    expect(error.closest(".message-error")).not.toBeNull();
  });

  it("renders an in-progress media preview inside the assistant message", async () => {
    localStorage.setItem("local-lm-chat", "chat-preview");
    const stamp = "2026-07-22T00:00:00Z";
    vi.mocked(api.chat).mockResolvedValue({
      id: "chat-preview",
      project_id: null,
      title: "Preview",
      archived: false,
      routing_mode: "auto",
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-preview",
      created_at: stamp,
      updated_at: stamp,
      messages: [
        {
          id: "user-preview",
          chat_id: "chat-preview",
          parent_id: null,
          role: "user",
          status: "complete",
          parts: [{ id: "prompt", position: 0, type: "text", text: "Create an image", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
        {
          id: "assistant-preview",
          chat_id: "chat-preview",
          parent_id: "user-preview",
          role: "assistant",
          status: "pending",
          parts: [
            { id: "progress", position: 0, type: "progress", text: "Preview", artifact_id: null, metadata_json: { progress: 0.85 } },
            {
              id: "preview",
              position: 1,
              type: "image",
              text: null,
              artifact_id: "sha256:preview",
              metadata_json: { preview: true },
              artifact: {
                id: "sha256:preview",
                sha256: "preview",
                kind: "thumbnail",
                media_type: "image/png",
                size_bytes: 100,
                original_name: "generation-preview",
                metadata_json: { temporary_preview: true },
                created_at: stamp,
                url: "/api/artifacts/sha256:preview/content",
              },
            },
          ],
          created_at: stamp,
          updated_at: stamp,
        },
      ],
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByAltText("Generation preview")).toBeInTheDocument();
    expect(screen.getByText("Generation preview")).toBeInTheDocument();
  });

  it("shows managed worker queue and memory telemetry", async () => {
    vi.mocked(api.workers).mockResolvedValue([
      {
        name: "chat",
        state: "ready",
        managed: true,
        running: true,
        pid: 123,
        profile_id: "profile-1",
        command: ["llama-server"],
        exit_code: null,
        estimated_memory_bytes: 6 * 1024 ** 3,
        current_memory_bytes: 5 * 1024 ** 3,
        peak_memory_bytes: 5.5 * 1024 ** 3,
        active_jobs: 2,
        queued_jobs: 1,
      },
    ]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Settings"));
    expect(await screen.findByText("Ready · PID 123")).toBeInTheDocument();
    expect(screen.getByText("current RAM")).toBeInTheDocument();
    expect(screen.getByText("measured peak")).toBeInTheDocument();
    expect(screen.getByText("estimated load")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Unload" })).toBeDisabled();
  });

  it("runs an executable structured-tool capability probe", async () => {
    vi.mocked(api.probeChatTools).mockResolvedValue({
      engine: "mock",
      version: "1",
      advertised: true,
      passed: true,
      tool_name: "choose_route",
      arguments: { mode: "image", confidence: 1 },
      error: null,
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Settings"));
    fireEvent.click(await screen.findByText("Test structured tools"));
    expect(await screen.findByText("Structured tool schema passed on mock 1.")).toBeInTheDocument();
  });

  it("keeps diagnostic bundle controls out of routine settings", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Settings"));
    expect(screen.queryByText("Download redacted diagnostics")).not.toBeInTheDocument();
    expect(screen.queryByText("FFmpeg")).not.toBeInTheDocument();
  });

  it("shows installed-model storage and partial cleanup controls", async () => {
    vi.mocked(api.models).mockResolvedValue([
      {
        id: "model-1",
        source_id: null,
        name: "Local GGUF",
        role: "chat",
        engine: "llama.cpp",
        local_path: "/models/local.gguf",
        size_bytes: 2048,
        compatibility: "advanced_import",
        manifest_json: {},
        active: true,
        created_at: "2026-07-22T00:00:00Z",
        updated_at: "2026-07-22T00:00:00Z",
      },
    ]);
    vi.mocked(api.modelStorage).mockResolvedValue({
      installed_bytes: 2048,
      partial_download_bytes: 512,
      catalog_cache_bytes: 128,
      installed_count: 1,
      partial_download_count: 2,
    });
    vi.mocked(api.cleanupDownloads).mockResolvedValue({ removed_count: 2, reclaimed_bytes: 512 });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Model library"));
    expect(await screen.findByText("1 installed · 2.0 KB")).toBeInTheDocument();
    expect(screen.getByText("Clean 2 partial")).toBeEnabled();
    expect(await screen.findByText("Local GGUF")).toBeInTheDocument();
    expect(screen.getByTitle("Delete installed model")).toBeEnabled();
  });

  it("opens the advanced local model import form", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Model library"));
    fireEvent.click(await screen.findByText("Import local"));
    expect(await screen.findByRole("heading", { name: "Import a local model" })).toBeInTheDocument();
    expect(screen.getByText(/Pickle-compatible formats are blocked/)).toBeInTheDocument();
    expect(screen.getByPlaceholderText("/path/to/model.gguf")).toBeInTheDocument();
  });

  it("paginates catalog results and exposes compatibility filters", async () => {
    vi.mocked(api.catalog).mockResolvedValue({
      items: [],
      next_cursor: "https://huggingface.co/api/models?cursor=next",
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Model library"));
    expect(await screen.findByLabelText("Compatibility filter")).toBeInTheDocument();
    expect(screen.getByLabelText("Format filter")).toBeInTheDocument();
    fireEvent.click(await screen.findByText("Load more models"));
    await waitFor(() => expect(api.catalog).toHaveBeenCalledTimes(2));
    expect(vi.mocked(api.catalog).mock.calls[1]?.[3]).toContain("cursor=next");
  });

  it("preflights and queues a safe catalog model from one click", async () => {
    const model = {
      provider: "huggingface",
      remote_id: "owner/model-8B-GGUF",
      name: "model-8B-GGUF",
      author: "owner",
      pipeline_tag: "text-generation",
      tags: ["gguf"],
      downloads: 42,
      likes: 3,
      trending_score: 1,
      created_at: "2026-07-22T00:00:00Z",
      last_modified: "2026-07-22T00:00:00Z",
      gated: false,
      private: false,
      library_name: null,
      architecture: "qwen3",
      formats: ["gguf"],
      quantizations: ["q4_k_m"],
      parameter_count: 8_000_000_000,
      license_id: "apache-2.0",
      total_size_bytes: 1024,
      compatibility: "likely",
      compatibility_reasons: ["GGUF artifact detected"],
    };
    vi.mocked(api.catalog).mockResolvedValue({ items: [model], next_cursor: null });
    vi.mocked(api.catalogDetail).mockResolvedValue({
      model,
      revision: "main",
      files: [{ filename: "model-q4.gguf", size: 1024, sha256: "a".repeat(64) }],
    });
    vi.mocked(api.catalogPreflight).mockResolvedValue({
      remote_id: model.remote_id,
      revision: "main",
      selected_files: ["model-q4.gguf"],
      expected_sha256: { "model-q4.gguf": "a".repeat(64) },
      download_bytes: 1024,
      available_disk_bytes: 4096,
      estimated_ram_bytes: 2048,
      estimated_vram_bytes: null,
      can_install: true,
      checks: [
        { id: "checksum", label: "Checksum metadata", status: "pass", detail: "Available." },
        { id: "disk", label: "Disk capacity", status: "pass", detail: "Fits." },
      ],
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Model library"));
    fireEvent.click(await screen.findByText("Install"));
    await waitFor(() => expect(api.catalogPreflight).toHaveBeenCalledWith(
      model.remote_id,
      "chat",
      "llama.cpp",
      "main",
      [],
    ));
    await waitFor(() => expect(api.download).toHaveBeenCalledWith(
      model.remote_id,
      "chat",
      "llama.cpp",
      "main",
      ["model-q4.gguf"],
      { "model-q4.gguf": "a".repeat(64) },
    ));
  });

  it("renders workflow revision history and declared controls", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    vi.mocked(api.workflows).mockResolvedValue([
      {
        id: "workflow-1",
        name: "Studio image",
        operation: "text_to_image",
        description: "A tunable image pipeline",
        current_revision_id: "revision-2",
        revisions: [
          {
            id: "revision-1",
            workflow_id: "workflow-1",
            version: 1,
            engine: "comfyui",
            engine_version: null,
            ui_graph_json: {},
            api_graph_json: { node: { class_type: "Sampler" } },
            input_schema_json: {},
            dependencies_json: {},
            trusted: true,
            created_at: stamp,
          },
          {
            id: "revision-2",
            workflow_id: "workflow-1",
            version: 2,
            engine: "comfyui",
            engine_version: null,
            ui_graph_json: {},
            api_graph_json: { node: { class_type: "SamplerV2" } },
            input_schema_json: { type: "object", properties: { steps: { type: "integer", title: "Steps", default: 20, minimum: 1, maximum: 100 } } },
            dependencies_json: { models: [] },
            trusted: true,
            created_at: stamp,
          },
        ],
      },
    ]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Workflows"));
    fireEvent.click(await screen.findByText("Studio image"));
    expect(await screen.findByText("Declared controls")).toBeInTheDocument();
    expect(screen.getByDisplayValue("20")).toBeInTheDocument();
    expect(screen.getByText("v2 · current")).toBeInTheDocument();
    expect(screen.queryByText("Restore as new revision")).not.toBeInTheDocument();
  });

  it("browses generated media and exposes retention-safe cleanup", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    vi.mocked(api.artifacts).mockResolvedValue([{
      id: "sha256:image",
      sha256: "0123456789abcdef",
      kind: "image",
      media_type: "image/png",
      size_bytes: 2048,
      original_name: "observatory.png",
      metadata_json: {},
      created_at: stamp,
      url: "/api/artifacts/sha256:image/content",
      reference_count: 1,
      chat_ids: ["chat-1"],
      project_ids: [],
    }]);
    vi.mocked(api.artifactStorage).mockResolvedValue({
      total_bytes: 2048,
      total_count: 1,
      referenced_bytes: 2048,
      referenced_count: 1,
      unreferenced_bytes: 0,
      unreferenced_count: 0,
      temporary_bytes: 0,
      temporary_count: 0,
      eligible_bytes: 0,
      eligible_count: 0,
      disk_free_bytes: 1024 ** 3,
      warning: false,
      retention_days: 30,
      temporary_retention_hours: 24,
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Media library"));
    expect(await screen.findByText("observatory.png")).toBeInTheDocument();
    expect(screen.getByText(/2\.0 KB · 1 reference/)).toBeInTheDocument();
    expect(screen.getByText("Clean eligible")).toBeDisabled();
  });

  it("isolates role-aware settings in profile and preset editors", async () => {
    vi.mocked(api.engines).mockResolvedValue([roleAwareMediaEngine]);
    vi.mocked(api.profiles).mockResolvedValue([{
      id: "image-profile",
      model_install_id: null,
      name: "Image profile",
      use_case: "",
      role: "image",
      engine: "mock",
      load_settings_json: {},
      request_settings_json: {},
      is_default: false,
    }]);
    vi.mocked(api.presets).mockResolvedValue([{
      id: "video-preset",
      name: "Video preset",
      role: "video",
      settings_json: {},
      is_default: false,
    }]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Settings"));
    await screen.findByText("Image profile");
    fireEvent.click(screen.getAllByRole("button", { name: "Edit" })[0]);
    expect(await screen.findByText("Negative prompt")).toBeInTheDocument();
    expect(screen.queryByText("Frames")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Close profile editor" }));
    fireEvent.click(screen.getAllByRole("button", { name: "Edit" })[1]);
    expect(await screen.findByText("Frames")).toBeInTheDocument();
    expect(screen.queryByText("Negative prompt")).not.toBeInTheDocument();
  });

  it("keeps load-only controls out of generation presets", async () => {
    vi.mocked(api.engines).mockResolvedValue([{
      ...roleAwareMediaEngine,
      roles: ["chat"],
      operations: ["text"],
      settings: [contextLengthSetting, maxTokensSetting],
      settings_by_role: { chat: [contextLengthSetting, maxTokensSetting] },
    }]);
    vi.mocked(api.presets).mockResolvedValue([{
      id: "chat-preset",
      name: "Chat preset",
      role: "chat",
      settings_json: {},
      is_default: false,
    }]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Settings"));
    await screen.findByText("Chat preset");
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    expect(await screen.findByText("Maximum output")).toBeInTheDocument();
    expect(screen.queryByText("Context length")).not.toBeInTheDocument();
  });

  it("isolates role-aware settings in per-turn controls", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const chat = {
      id: "chat-role-settings",
      project_id: null,
      title: "Role settings",
      archived: false,
      routing_mode: "auto" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      created_at: stamp,
      updated_at: stamp,
    };
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.engines).mockResolvedValue([roleAwareMediaEngine]);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const mode = await screen.findByDisplayValue("Auto");
    fireEvent.change(mode, { target: { value: "image" } });
    fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
    expect(await screen.findByText("Negative prompt")).toBeInTheDocument();
    expect(screen.queryByText("Frames")).not.toBeInTheDocument();

    fireEvent.change(mode, { target: { value: "video" } });
    expect(await screen.findByText("Frames")).toBeInTheDocument();
    expect(screen.queryByText("Negative prompt")).not.toBeInTheDocument();
  });

  it("applies the pinned workflow schema to per-turn controls", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const project = {
      id: "project-workflow-controls",
      name: "Video project",
      description: "",
      instructions: "",
      archived: false,
      image_workflow_revision_id: null,
      video_workflow_revision_id: "revision-video",
      created_at: stamp,
      updated_at: stamp,
    };
    const chat = {
      id: "chat-workflow-controls",
      project_id: project.id,
      title: "Workflow controls",
      archived: false,
      routing_mode: "video" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      created_at: stamp,
      updated_at: stamp,
    };
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.projects).mockResolvedValue([project]);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    vi.mocked(api.engines).mockResolvedValue([roleAwareMediaEngine]);
    vi.mocked(api.workflows).mockResolvedValue([{
      id: "workflow-video",
      name: "Fixed video",
      operation: "text_to_video",
      description: "",
      current_revision_id: "revision-video",
      revisions: [{
        id: "revision-video",
        workflow_id: "workflow-video",
        version: 1,
        engine: "mock",
        engine_version: null,
        ui_graph_json: {},
        api_graph_json: {},
        input_schema_json: {
          type: "object",
          properties: { frames: { type: "integer", const: 81, default: 81 } },
        },
        dependencies_json: {},
        trusted: true,
        created_at: stamp,
      }],
    }]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Turn settings" }));
    const frames = await screen.findByRole("spinbutton", { name: /Frames/ });
    expect(frames).toHaveValue(81);
    expect(frames).toBeDisabled();
    expect(screen.getByText(/Fixed by this workflow at 81/)).toBeInTheDocument();
  });

  it("applies turn controls to send, edit-and-branch, and regenerate actions", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const chat = {
      id: "chat-turn-overrides",
      project_id: null,
      title: "Turn overrides",
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-turn-overrides",
      created_at: stamp,
      updated_at: stamp,
    };
    const userMessage = {
      id: "user-turn-overrides",
      chat_id: chat.id,
      parent_id: null,
      role: "user" as const,
      status: "complete" as const,
      parts: [{ id: "user-part", position: 0, type: "text" as const, text: "Count to 100", artifact_id: null, metadata_json: {} }],
      created_at: stamp,
      updated_at: stamp,
    };
    const assistantMessage = {
      id: "assistant-turn-overrides",
      chat_id: chat.id,
      parent_id: userMessage.id,
      role: "assistant" as const,
      status: "complete" as const,
      parts: [{ id: "assistant-part", position: 0, type: "text" as const, text: "1 2 3", artifact_id: null, metadata_json: {} }],
      created_at: stamp,
      updated_at: stamp,
    };
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.engines).mockResolvedValue([{
      ...roleAwareMediaEngine,
      roles: ["chat"],
      operations: ["text"],
      settings: [contextLengthSetting, maxTokensSetting],
      settings_by_role: { chat: [contextLengthSetting, maxTokensSetting] },
    }]);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [userMessage, assistantMessage] });
    vi.mocked(api.sendTurn).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.branchMessage).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.regenerateMessage).mockReturnValue(new Promise(() => {}));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Turn settings" }));
    expect(screen.queryByRole("spinbutton", { name: /Context length/ })).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("spinbutton", { name: /Maximum output/ }), { target: { value: "4096" } });
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));

    fireEvent.change(screen.getByPlaceholderText(/Ask anything/), { target: { value: "Count to 1000" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(api.sendTurn).toHaveBeenCalledWith(chat.id, "Count to 1000", "text", [], { max_tokens: 4096 }));

    fireEvent.click(screen.getByText("Edit and branch"));
    fireEvent.change(screen.getByLabelText("Edit message"), { target: { value: "Count to 1000" } });
    fireEvent.click(screen.getByText("Send edited message"));
    await waitFor(() => expect(api.branchMessage).toHaveBeenCalledWith(userMessage.id, "Count to 1000", { max_tokens: 4096 }));

    fireEvent.click(screen.getByRole("button", { name: "Regenerate response" }));
    await waitFor(() => expect(api.regenerateMessage).toHaveBeenCalledWith(assistantMessage.id, { max_tokens: 4096 }));
  });

  it("keeps turn controls isolated to their chat", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const chat = (id: string, title: string) => ({
      id,
      project_id: null,
      title,
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      created_at: stamp,
      updated_at: stamp,
    });
    const firstChat = chat("chat-settings-one", "Settings chat one");
    const secondChat = chat("chat-settings-two", "Settings chat two");
    localStorage.setItem("local-lm-chat", firstChat.id);
    vi.mocked(api.engines).mockResolvedValue([{
      ...roleAwareMediaEngine,
      roles: ["chat"],
      operations: ["text"],
      settings: [maxTokensSetting],
      settings_by_role: { chat: [maxTokensSetting] },
    }]);
    vi.mocked(api.chats).mockResolvedValue([firstChat, secondChat]);
    vi.mocked(api.chat).mockImplementation(async (id) => ({
      ...(id === firstChat.id ? firstChat : secondChat),
      messages: [],
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Turn settings" }));
    fireEvent.change(screen.getByRole("spinbutton", { name: /Maximum output/ }), { target: { value: "4096" } });
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));

    fireEvent.click(screen.getByText(secondChat.title));
    await screen.findByRole("heading", { name: secondChat.title });
    fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
    expect(screen.getByRole("spinbutton", { name: /Maximum output/ })).toHaveValue(1024);
    fireEvent.change(screen.getByRole("spinbutton", { name: /Maximum output/ }), { target: { value: "2048" } });
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));

    fireEvent.click(screen.getByText(firstChat.title));
    await screen.findByRole("heading", { name: firstChat.title });
    fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
    expect(screen.getByRole("spinbutton", { name: /Maximum output/ })).toHaveValue(4096);
  });
});
