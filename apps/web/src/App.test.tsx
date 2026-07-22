import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";

vi.mock("./api", () => ({
  api: {
    initialize: vi.fn().mockResolvedValue(undefined),
    projects: vi.fn().mockResolvedValue([]),
    chats: vi.fn().mockResolvedValue([]),
    chat: vi.fn(),
    createProject: vi.fn(),
    createChat: vi.fn(),
    sendTurn: vi.fn(),
    regenerateMessage: vi.fn(),
    branchMessage: vi.fn(),
    cancelChat: vi.fn(),
    jobs: vi.fn().mockResolvedValue([]),
    cancelJob: vi.fn(),
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
    models: vi.fn(),
    profiles: vi.fn().mockResolvedValue([]),
    updateProfile: vi.fn(),
    cloneProfile: vi.fn(),
    resetProfile: vi.fn(),
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
    recipes: vi.fn().mockResolvedValue([]),
    installRecipe: vi.fn(),
    download: vi.fn(),
    workflows: vi.fn(),
    createWorkflow: vi.fn(),
    validateWorkflow: vi.fn(),
    upload: vi.fn(),
  },
  connectEvents: vi.fn().mockImplementation(async (_onEvent, onStatus) => {
    onStatus(true);
    return () => undefined;
  }),
}));

describe("App", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.mocked(api.profiles).mockResolvedValue([]);
    vi.mocked(api.chats).mockResolvedValue([]);
    vi.mocked(api.workers).mockResolvedValue([]);
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
    expect(screen.getByText("Local service connected")).toBeInTheDocument();
  });

  it("shows the current machine against the approved platform matrix", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(screen.getByText("Settings"));
    expect(await screen.findByText("Platform support")).toBeInTheDocument();
    expect(await screen.findByText("Ubuntu 24.04 LTS x64 target")).toBeInTheDocument();
    expect(screen.getByText("hardware pending")).toBeInTheDocument();
  });

  it("opens a model profile in the schema-driven settings editor", async () => {
    vi.mocked(api.profiles).mockResolvedValue([
      {
        id: "profile-1",
        model_install_id: "model-1",
        name: "Local chat",
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
    expect(await screen.findByText("Local chat · default")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    expect(await screen.findByText("Edit profile")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Local chat")).toBeInTheDocument();
    expect(screen.getByText("Default chat profile")).toBeInTheDocument();
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
      parts: [{ id: `${id}-part`, position: 0, type: "text" as const, text, artifact_id: null, metadata_json: {} }],
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
    fireEvent.click(screen.getAllByText("Edit and branch").at(-1)!);
    expect(screen.getByDisplayValue("Edited question")).toBeInTheDocument();
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
});
