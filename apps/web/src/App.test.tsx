import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
    importWorkflow: vi.fn(),
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
    vi.clearAllMocks();
    localStorage.clear();
    vi.mocked(api.profiles).mockResolvedValue([]);
    vi.mocked(api.chats).mockResolvedValue([]);
    vi.mocked(api.workers).mockResolvedValue([]);
    vi.mocked(api.models).mockResolvedValue([]);
    vi.mocked(api.catalog).mockResolvedValue({ items: [], next_cursor: null });
    vi.mocked(api.workflows).mockResolvedValue([]);
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
    expect(screen.getByText("Skip to main content")).toHaveAttribute("href", "#main-content");
    const navigation = screen.getByRole("button", { name: "Toggle navigation" });
    expect(navigation).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(navigation);
    expect(navigation).toHaveAttribute("aria-expanded", "true");
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

  it("downloads a redacted diagnostic bundle", async () => {
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    vi.mocked(api.createDiagnostics).mockResolvedValue({ url: "/api/artifacts/diagnostics/content" });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Settings"));
    fireEvent.click(await screen.findByText("Download redacted diagnostics"));
    await waitFor(() => expect(api.createDiagnostics).toHaveBeenCalledOnce());
    expect(click).toHaveBeenCalledOnce();
    click.mockRestore();
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

  it("shows catalog install checks before queueing a model", async () => {
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
      files: [{ filename: "model-q4.gguf", size: 1024, sha256: null }],
    });
    vi.mocked(api.catalogPreflight).mockResolvedValue({
      remote_id: model.remote_id,
      revision: "main",
      selected_files: ["model-q4.gguf"],
      download_bytes: 1024,
      available_disk_bytes: 4096,
      estimated_ram_bytes: 2048,
      estimated_vram_bytes: null,
      can_install: true,
      checks: [{ id: "disk", label: "Disk capacity", status: "pass", detail: "Fits." }],
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Model library"));
    fireEvent.click(await screen.findByText("Choose files"));
    expect(await screen.findByRole("heading", { name: model.remote_id })).toBeInTheDocument();
    expect(await screen.findByText("Disk capacity")).toBeInTheDocument();
    expect(screen.getByText("apache-2.0")).toBeInTheDocument();
    expect(screen.getByText("Queue download")).toBeEnabled();
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
});
