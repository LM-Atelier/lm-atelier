import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { api, connectEvents } from "./api";
import type { BackupInfo, Chat, ChatDetail, EngineCapabilities, SettingField, TurnAccepted } from "./types";

const clipboardWrite = vi.fn();

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
    deleteArtifact: vi.fn(),
    sendTurn: vi.fn(),
    stopAndSendTurn: vi.fn(),
    regenerateMessage: vi.fn(),
    selectResponseRevision: vi.fn(),
    branchMessage: vi.fn(),
    cancelChat: vi.fn(),
    jobs: vi.fn().mockResolvedValue([]),
    workPlans: vi.fn().mockResolvedValue([]),
    workPlan: vi.fn(),
    workStep: vi.fn(),
    cancelWorkPlan: vi.fn(),
    retryWorkPlan: vi.fn(),
    cancelWorkStep: vi.fn(),
    retryWorkStep: vi.fn(),
    cancelJob: vi.fn(),
    retryJob: vi.fn(),
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
    about: vi.fn().mockResolvedValue({
      version: "0.1.7",
      data_directory: "C:\\LM Atelier\\data",
      log_directory: "C:\\LM Atelier\\data\\logs",
    }),
    platforms: vi.fn().mockResolvedValue([]),
    createDiagnostics: vi.fn(),
    credentialStatus: vi.fn().mockResolvedValue({ provider: "huggingface", configured: false, source: "none", vault_available: true }),
    setHuggingFaceToken: vi.fn(),
    deleteHuggingFaceToken: vi.fn(),
    models: vi.fn(),
    modelAssets: vi.fn().mockResolvedValue([]),
    updateModelAsset: vi.fn(),
    deleteModelAsset: vi.fn(),
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
    runtimes: vi.fn().mockResolvedValue([]),
    installRuntime: vi.fn(),
    backups: vi.fn().mockResolvedValue([]),
    loadChatWorker: vi.fn(),
    startMediaWorker: vi.fn(),
    stopWorker: vi.fn(),
    createBackup: vi.fn(),
    verifyBackup: vi.fn(),
    restoreBackup: vi.fn(),
    deleteBackup: vi.fn(),
    catalog: vi.fn(),
    workflowCatalogModels: vi.fn(),
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
    clipboardWrite.mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: clipboardWrite },
    });
    localStorage.clear();
    sessionStorage.clear();
    vi.mocked(api.projects).mockResolvedValue([]);
    vi.mocked(api.profiles).mockResolvedValue([]);
    vi.mocked(api.presets).mockResolvedValue([]);
    vi.mocked(api.chats).mockResolvedValue([]);
    vi.mocked(api.engines).mockResolvedValue([
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
    ]);
    vi.mocked(api.workers).mockResolvedValue([]);
    vi.mocked(api.runtimes).mockResolvedValue([]);
    vi.mocked(api.jobs).mockResolvedValue([]);
    vi.mocked(api.workPlans).mockResolvedValue([]);
    vi.mocked(api.backups).mockResolvedValue([]);
    vi.mocked(api.models).mockResolvedValue([]);
    vi.mocked(api.modelAssets).mockResolvedValue([]);
    vi.mocked(api.catalog).mockResolvedValue({ items: [], next_cursor: null });
    vi.mocked(api.workflowCatalogModels).mockResolvedValue([]);
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
    const modelLibrary = screen.getByRole("button", { name: "Model library" });
    fireEvent.click(modelLibrary);
    expect(modelLibrary).toHaveAttribute("aria-current", "page");
    await waitFor(() => expect(document.getElementById("main-content")).toHaveFocus());
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

  it("applies durable job snapshots immediately without waiting for polling", async () => {
    const stamp = "2026-07-26T00:00:00Z";
    const queued = {
      id: "job-live",
      kind: "download",
      status: "queued",
      run_id: null,
      progress: 0,
      phase: "queued",
      payload_json: {},
      result_json: {},
      error: null,
      attempt: 0,
      cancellable: true,
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.jobs).mockResolvedValue([queued]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidate = vi.spyOn(client, "invalidateQueries");
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("queued")).toBeInTheDocument();
    const onEvent = vi.mocked(connectEvents).mock.calls.at(-1)?.[0];
    invalidate.mockClear();

    act(() => {
      onEvent?.({
        sequence: 2,
        type: "job.progress",
        entity_id: queued.id,
        payload: {
          job: {
            ...queued,
            status: "running",
            phase: "downloading model",
            progress: 0.25,
            updated_at: "2026-07-26T00:00:01Z",
          },
        },
        created_at: "2026-07-26T00:00:01Z",
      });
    });

    expect(await screen.findByText("downloading model · 25%")).toBeInTheDocument();
    expect(invalidate).not.toHaveBeenCalledWith({ queryKey: ["jobs"] });
  });

  it("coalesces reconnect and replay-gap reconciliation across authoritative data", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const invalidate = vi.spyOn(client, "invalidateQueries");
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    expect(await screen.findByText("LM Atelier")).toBeInTheDocument();
    const eventConnection = vi.mocked(connectEvents).mock.calls.at(-1);
    const onEvent = eventConnection?.[0];
    const onReconnect = eventConnection?.[2];
    expect(onReconnect).toBeDefined();
    invalidate.mockClear();

    act(() => {
      onReconnect?.();
      onEvent?.({
        sequence: 2,
        type: "events.replay_gap",
        entity_id: null,
        payload: { oldest_sequence: 10 },
        created_at: "2026-07-23T00:00:00Z",
      });
    });

    await waitFor(() => expect(invalidate).toHaveBeenCalledTimes(1));
    const options = invalidate.mock.calls[0]?.[0] as {
      predicate?: (query: { queryKey: readonly unknown[] }) => boolean;
    };
    expect(options.predicate?.({ queryKey: ["chat", "chat-1"] })).toBe(true);
    expect(options.predicate?.({ queryKey: ["backups"] })).toBe(true);
    expect(options.predicate?.({ queryKey: ["workers"] })).toBe(true);
    expect(options.predicate?.({ queryKey: ["unrelated"] })).toBe(false);
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

  it("offers only runtime-verified vision profiles in the per-chat vision selector", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const chat = {
      id: "chat-vision-selector",
      project_id: null,
      title: "Vision selector",
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_vision_profile_id: "__auto__",
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      vision_settings_json: {
        max_images: 4,
        max_video_frames: 6,
        include_prior_visual: true,
      },
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    vi.mocked(api.profiles).mockResolvedValue([
      {
        id: "profile-text-only",
        model_install_id: "model-text-only",
        name: "Text only",
        use_case: "",
        role: "chat",
        engine: "mock",
        load_settings_json: {},
        request_settings_json: {},
        input_modalities: ["text"],
        is_default: false,
      },
      {
        id: "profile-vision",
        model_install_id: "model-vision",
        name: "Visual observer",
        use_case: "",
        role: "chat",
        engine: "mock",
        load_settings_json: {},
        request_settings_json: {},
        input_modalities: ["text", "image"],
        is_default: false,
      },
    ]);
    vi.mocked(api.updateChat).mockResolvedValue({
      ...chat,
      active_vision_profile_id: "profile-vision",
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const selector = await screen.findByRole("combobox", { name: "vision" });
    expect(selector).toHaveValue("__auto__");
    const optionNames = Array.from(selector.querySelectorAll("option"), (option) => option.textContent);
    expect(optionNames).toContain("Visual observer");
    expect(optionNames).not.toContain("Text only");
    expect(optionNames).toContain("Off");

    fireEvent.change(selector, { target: { value: "profile-vision" } });
    await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(
      chat.id,
      { active_vision_profile_id: "profile-vision" },
    ));
  });

  it("removes a deleted chat immediately while the API request is pending", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const chats = ["First chat", "Second chat"].map((title, index) => ({
      id: `chat-${index + 1}`,
      project_id: null,
      title,
      archived: false,
      routing_mode: "auto" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      created_at: stamp,
      updated_at: stamp,
    }));
    vi.mocked(api.chats).mockResolvedValue(chats);
    vi.mocked(api.chat).mockImplementation(async (id) => ({
      ...chats.find((chat) => chat.id === id)!,
      messages: [],
    }));
    let finishDelete: (() => void) | undefined;
    vi.mocked(api.deleteChat).mockImplementation(
      () => new Promise<void>((resolve) => { finishDelete = resolve; }),
    );
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Manage First chat" }));
    expect(screen.getByText("Ask before Auto mode starts an image or video when the planner is unsure.")).toBeVisible();
    fireEvent.click(screen.getByRole("checkbox", { name: /Delete generated media with chat/ }));
    fireEvent.click(screen.getByRole("button", { name: "Delete chat" }));

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Manage First chat" })).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Manage Second chat" })).toBeInTheDocument();
      expect(localStorage.getItem("local-lm-chat")).toBe("chat-2");
    });
    expect(vi.mocked(api.deleteChat)).toHaveBeenCalledWith("chat-1", true);
    finishDelete?.();
    confirm.mockRestore();
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

  it("persists schema-driven project defaults by role without leaking values", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const project = {
      id: "project-defaults",
      name: "Creative work",
      description: "",
      instructions: "",
      archived: false,
      image_workflow_revision_id: null,
      video_workflow_revision_id: null,
      generation_settings_json: {
        chat: { max_tokens: 2048 },
        image: { negative_prompt: "noise" },
        video: { frames: 49 },
      },
      generation_preset_ids_json: {},
      created_at: stamp,
      updated_at: stamp,
    };
    const imagePreset = {
      id: "project-image-preset",
      name: "Project image",
      role: "image" as const,
      settings_json: { negative_prompt: "blur" },
      is_default: false,
    };
    vi.mocked(api.projects).mockResolvedValue([project]);
    vi.mocked(api.engines).mockResolvedValue([{
      ...roleAwareMediaEngine,
      roles: ["chat", "image", "video"],
      settings: [maxTokensSetting, imageSetting, videoSetting],
      settings_by_role: {
        chat: [maxTokensSetting],
        image: [imageSetting],
        video: [videoSetting],
      },
    }]);
    vi.mocked(api.presets).mockResolvedValue([imagePreset]);
    vi.mocked(api.updateProject).mockResolvedValue(project);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Manage Creative work" }));
    const dialog = screen.getByRole("dialog", { name: "Manage project" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(screen.getByRole("spinbutton", { name: /Maximum output/ })).toHaveValue(2048);
    fireEvent.change(screen.getByRole("spinbutton", { name: /Maximum output/ }), {
      target: { value: "4096" },
    });

    fireEvent.click(screen.getByRole("button", { name: "image" }));
    expect(screen.getByLabelText(/Negative prompt/)).toHaveValue("noise");
    fireEvent.change(screen.getByRole("combobox", { name: "image project preset" }), {
      target: { value: imagePreset.id },
    });
    fireEvent.change(screen.getByLabelText(/Negative prompt/), {
      target: { value: "fog" },
    });

    fireEvent.click(screen.getByRole("button", { name: "video" }));
    expect(screen.getByRole("spinbutton", { name: /Frames/ })).toHaveValue(49);
    fireEvent.change(screen.getByRole("spinbutton", { name: /Frames/ }), {
      target: { value: "81" },
    });
    fireEvent.click(screen.getByRole("button", { name: "chat" }));
    expect(screen.getByRole("spinbutton", { name: /Maximum output/ })).toHaveValue(4096);
    fireEvent.click(screen.getByRole("button", { name: "Save project" }));

    await waitFor(() => expect(api.updateProject).toHaveBeenCalledWith(
      project.id,
      expect.objectContaining({
        generation_settings_json: {
          chat: { max_tokens: 4096 },
          image: { negative_prompt: "fog" },
          video: { frames: 81 },
        },
        generation_preset_ids_json: { image: imagePreset.id },
      }),
    ));
  });

  it("clears only the selected project role or all project generation defaults", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const project = {
      id: "project-reset",
      name: "Reset scopes",
      description: "",
      instructions: "",
      archived: false,
      image_workflow_revision_id: null,
      video_workflow_revision_id: null,
      generation_settings_json: {
        chat: { max_tokens: 2048 },
        image: { negative_prompt: "noise" },
        video: { frames: 81 },
      },
      generation_preset_ids_json: {
        image: "project-image-preset",
        video: "project-video-preset",
      },
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.projects).mockResolvedValue([project]);
    vi.mocked(api.engines).mockResolvedValue([{
      ...roleAwareMediaEngine,
      roles: ["chat", "image", "video"],
      settings_by_role: {
        chat: [maxTokensSetting],
        image: [imageSetting],
        video: [videoSetting],
      },
    }]);
    vi.mocked(api.presets).mockResolvedValue([
      {
        id: "project-image-preset",
        name: "Image",
        role: "image",
        settings_json: {},
        is_default: false,
      },
      {
        id: "project-video-preset",
        name: "Video",
        role: "video",
        settings_json: {},
        is_default: false,
      },
    ]);
    vi.mocked(api.updateProject).mockResolvedValue(project);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Manage Reset scopes" }));
    fireEvent.click(screen.getByRole("button", { name: "video" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear role defaults" }));
    fireEvent.click(screen.getByRole("button", { name: "Save project" }));
    await waitFor(() => expect(api.updateProject).toHaveBeenLastCalledWith(
      project.id,
      expect.objectContaining({
        generation_settings_json: {
          chat: { max_tokens: 2048 },
          image: { negative_prompt: "noise" },
        },
        generation_preset_ids_json: { image: "project-image-preset" },
      }),
    ));

    fireEvent.click(screen.getByRole("button", { name: "Manage Reset scopes" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear all" }));
    fireEvent.click(screen.getByRole("button", { name: "Save project" }));
    await waitFor(() => expect(api.updateProject).toHaveBeenLastCalledWith(
      project.id,
      expect.objectContaining({
        generation_settings_json: {},
        generation_preset_ids_json: {},
      }),
    ));
  });

  it("shows inherited project defaults without copying them into chat scope", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const project = {
      id: "project-inherited",
      name: "Inherited defaults",
      description: "",
      instructions: "",
      archived: false,
      image_workflow_revision_id: null,
      video_workflow_revision_id: null,
      generation_settings_json: { image: { negative_prompt: "project value" } },
      generation_preset_ids_json: { image: "project-image-preset" },
      created_at: stamp,
      updated_at: stamp,
    };
    const chat = {
      id: "chat-inherited",
      project_id: project.id,
      title: "Inherited chat",
      archived: false,
      routing_mode: "image" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      generation_settings_json: {},
      generation_preset_ids_json: {},
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.projects).mockResolvedValue([project]);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    vi.mocked(api.engines).mockResolvedValue([roleAwareMediaEngine]);
    vi.mocked(api.presets).mockResolvedValue([
      {
        id: "global-image-preset",
        name: "Global image",
        role: "image",
        settings_json: { negative_prompt: "global value" },
        is_default: true,
      },
      {
        id: "project-image-preset",
        name: "Project image",
        role: "image",
        settings_json: { negative_prompt: "project preset value" },
        is_default: false,
      },
    ]);
    vi.mocked(api.updateChat).mockImplementation(async (_id, values) => ({ ...chat, ...values }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Turn settings" }));
    expect(screen.getByRole("combobox", { name: "image preset" }))
      .toHaveDisplayValue("Inherit · Project image");
    expect(screen.getByLabelText(/Negative prompt/)).toHaveValue("project value");
    fireEvent.change(screen.getByLabelText(/Negative prompt/), {
      target: { value: "chat value" },
    });

    await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(chat.id, {
      generation_settings_json: { image: { negative_prompt: "chat value" } },
    }));
    expect(project.generation_settings_json.image.negative_prompt).toBe("project value");
  });

  it("traps modal focus, closes on Escape, and restores the opener", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const chat = {
      id: "chat-dialog",
      project_id: null,
      title: "Keyboard dialog",
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
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const opener = await screen.findByRole("button", { name: "Manage Keyboard dialog" });
    opener.focus();
    fireEvent.click(opener);
    const dialog = screen.getByRole("dialog", { name: "Manage chat" });
    const close = screen.getByRole("button", { name: "Close chat manager" });
    const save = screen.getByRole("button", { name: "Save chat" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(close).toHaveFocus();

    save.focus();
    fireEvent.keyDown(dialog, { key: "Tab" });
    expect(close).toHaveFocus();
    fireEvent.keyDown(dialog, { key: "Tab", shiftKey: true });
    expect(save).toHaveFocus();

    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Manage chat" })).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("treats the settings drawer as a labelled modal and restores focus", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const chat = {
      id: "chat-settings-dialog",
      project_id: null,
      title: "Settings dialog",
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
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const opener = await screen.findByRole("button", { name: "Turn settings" });
    opener.focus();
    fireEvent.click(opener);
    const dialog = screen.getByRole("dialog", { name: "Chat settings" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(screen.getByRole("button", { name: "Close settings" })).toHaveFocus();
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Chat settings" })).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("uses installed LoRAs through a picker instead of raw JSON", async () => {
    const stamp = "2026-07-26T00:00:00Z";
    const chat = {
      id: "chat-lora",
      project_id: null,
      title: "LoRA controls",
      archived: false,
      routing_mode: "image" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_vision_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      generation_settings_json: {},
      generation_preset_ids_json: {},
      vision_settings_json: {},
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    vi.mocked(api.engines).mockResolvedValue([roleAwareMediaEngine]);
    vi.mocked(api.workflows).mockResolvedValue([{
      id: "workflow-lora",
      name: "LoRA image",
      operation: "text_to_image",
      description: "",
      current_revision_id: "revision-lora",
      revisions: [{
        id: "revision-lora",
        workflow_id: "workflow-lora",
        version: 1,
        engine: "comfyui",
        engine_version: null,
        ui_graph_json: {},
        api_graph_json: {},
        input_schema_json: {
          type: "object",
          properties: {
            loras: { type: "array", title: "LoRAs", default: [], maxItems: 8 },
          },
        },
        dependencies_json: {
          extensions: { lora: { model: ["1", 0], clip: ["1", 1] } },
        },
        trusted: true,
        created_at: stamp,
      }],
    }]);
    vi.mocked(api.modelAssets).mockResolvedValue([{
      id: "asset-ink",
      source_id: "source-ink",
      name: "Atelier Ink",
      kind: "lora",
      family: "sdxl",
      size_bytes: 1024,
      manifest_json: {
        metadata: { trigger_words: ["ink wash"] },
      },
      active: true,
      verified_at: stamp,
      created_at: stamp,
      updated_at: stamp,
    }]);
    vi.mocked(api.updateChat).mockImplementation(async (_id, values) => ({ ...chat, ...values }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Turn settings" }));
    fireEvent.click(screen.getByRole("button", { name: "advanced" }));
    const addLora = await screen.findByRole("button", { name: "Add LoRA" });
    await waitFor(() => expect(addLora).toBeEnabled());
    fireEvent.click(addLora);

    expect(screen.getByRole("combobox", { name: "LoRA 1" })).toHaveDisplayValue("Atelier Ink");
    expect(screen.getByLabelText("LoRA 1 model strength")).toHaveValue(1);
    expect(screen.getByText("sdxl · ink wash")).toBeInTheDocument();
    await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(chat.id, {
      generation_settings_json: {
        image: {
          loras: [{
            asset_id: "asset-ink",
            model_strength: 1,
            clip_strength: 1,
            enabled: true,
          }],
        },
      },
    }));
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

  it("shows truthful structured progress without a forced minimum percentage", async () => {
    const stamp = new Date().toISOString();
    vi.mocked(api.jobs).mockResolvedValue([
      {
        id: "download-progress",
        kind: "download",
        status: "running",
        run_id: null,
        progress: 0.73,
        phase: "downloading",
        progress_json: {
          version: 2,
          stage: "downloading",
          stage_progress: 0.73,
          overall_progress: null,
          completed_units: 730,
          total_units: 1_000,
          unit: "bytes",
          bytes_reused: 0,
          rate_bytes_per_second: 40 * 1024 * 1024,
          eta_seconds: 7,
          file_index: 1,
          file_count: 1,
          queue_resource: "network",
          queue_position: null,
          queue_length: null,
          blocked_by: [],
          indeterminate: false,
          updated_at: stamp,
        },
        payload_json: {},
        result_json: {},
        error: null,
        attempt: 1,
        cancellable: true,
        created_at: stamp,
        updated_at: stamp,
      },
      {
        id: "queued-indeterminate",
        kind: "image",
        status: "queued",
        run_id: "run-queued",
        progress: 0,
        phase: "queued",
        progress_json: {
          version: 2,
          stage: "queued",
          stage_progress: null,
          overall_progress: null,
          completed_units: null,
          total_units: null,
          unit: null,
          bytes_reused: 0,
          rate_bytes_per_second: null,
          eta_seconds: null,
          file_index: null,
          file_count: null,
          queue_resource: "media_compute",
          queue_position: 2,
          queue_length: 3,
          blocked_by: [],
          indeterminate: true,
          updated_at: stamp,
        },
        payload_json: {},
        result_json: {},
        error: null,
        attempt: 0,
        cancellable: true,
        created_at: stamp,
        updated_at: stamp,
      },
    ]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("downloading · 73% · 40 MB/s · about 7 sec"))
      .toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "download progress" }))
      .toHaveAttribute("aria-valuenow", "73");
    expect(screen.getByText("queued · 2 ahead")).toBeInTheDocument();
    const indeterminate = screen.getByRole("progressbar", { name: "image progress" });
    expect(indeterminate).not.toHaveAttribute("aria-valuenow");
    expect(indeterminate.firstElementChild).toHaveClass("indeterminate");
  });

  it("shows a bounded list of unsuccessful jobs and retries one", async () => {
    const failedJobs = Array.from({ length: 5 }, (_, index) => ({
      id: `job-${index + 1}`,
      kind: `image-${index + 1}`,
      status: index % 2 ? "interrupted" : "failed",
      run_id: `run-${index + 1}`,
      progress: 0.5,
      phase: `loading phase ${index + 1}`,
      payload_json: {},
      result_json: {},
      error: `loader ${index + 1} crashed`,
      attempt: 1,
      cancellable: false,
      created_at: `2026-07-2${index + 1}T00:00:00Z`,
      updated_at: `2026-07-2${index + 1}T00:00:00Z`,
    }));
    vi.mocked(api.jobs).mockResolvedValue(failedJobs);
    vi.mocked(api.retryJob).mockResolvedValue({
      ...failedJobs[4],
      status: "queued",
      phase: "retry queued",
      error: null,
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("loader 5 crashed")).toBeInTheDocument();
    expect(screen.queryByText("loader 1 crashed")).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /Retry image-\d job/ })).toHaveLength(3);
    fireEvent.click(screen.getByRole("button", { name: "Retry image-5 job" }));
    await waitFor(() => expect(vi.mocked(api.retryJob).mock.calls[0]?.[0]).toBe("job-5"));
  });

  it("clears recent job issues without deleting history and keeps them hidden", async () => {
    const dismissedJob = {
      id: "job-dismissed",
      kind: "image",
      status: "failed",
      run_id: "run-dismissed",
      progress: 0.5,
      phase: "loading",
      payload_json: {},
      result_json: {},
      error: "old loader failure",
      attempt: 1,
      cancellable: false,
      created_at: "2026-07-24T00:00:00Z",
      updated_at: "2026-07-24T00:00:00Z",
    };
    vi.mocked(api.jobs).mockResolvedValue([dismissedJob]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const firstRender = render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Clear recent job issues" }));
    await waitFor(() => expect(screen.queryByText("old loader failure")).not.toBeInTheDocument());
    expect(localStorage.getItem("lm-atelier-dismissed-job-issues-before")).toBe(
      String(Date.parse(dismissedJob.updated_at)),
    );
    expect(api.retryJob).not.toHaveBeenCalled();
    firstRender.unmount();

    const newJob = {
      ...dismissedJob,
      id: "job-new",
      run_id: "run-new",
      error: "new loader failure",
      updated_at: "2026-07-25T00:00:00Z",
    };
    vi.mocked(api.jobs).mockResolvedValue([dismissedJob, newJob]);
    const refreshedClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    render(
      <QueryClientProvider client={refreshedClient}>
        <App />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("new loader failure")).toBeInTheDocument();
    expect(screen.queryByText("old loader failure")).not.toBeInTheDocument();
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
    expect(screen.getByText("CPU model")).toBeInTheDocument();
    expect(screen.queryByText(/logical processors/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Platform support")).not.toBeInTheDocument();
  });

  it("verifies, schedules restore, and deletes recovery backups", async () => {
    const backup = (name: string, createdAt: string): BackupInfo => ({
      name,
      size_bytes: 2048,
      sha256: "abcdef0123456789",
      created_at: createdAt,
      verified: false,
      restore_pending: false,
      media_included: false,
      media_size_bytes: 0,
    });
    const first = backup("local-lm-first.sqlite3", "2026-07-25T12:00:00Z");
    const second = { ...backup("local-lm-second.sqlite3", "2026-07-24T12:00:00Z"), verified: true };
    vi.mocked(api.backups).mockResolvedValue([first, second]);
    vi.mocked(api.verifyBackup).mockResolvedValue({ ...first, verified: true });
    vi.mocked(api.restoreBackup).mockResolvedValue({
      ...first,
      verified: true,
      restore_pending: true,
    });
    vi.mocked(api.deleteBackup).mockResolvedValue(undefined);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Settings"));
    expect(await screen.findByText(first.name)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: `Verify backup ${first.name}` }));
    expect(await screen.findByText("Backup verified.")).toBeInTheDocument();
    expect(vi.mocked(api.verifyBackup).mock.calls[0]?.[0]).toBe(first.name);

    fireEvent.click(screen.getByRole("button", { name: `Restore backup ${first.name} on restart` }));
    await waitFor(() => expect(vi.mocked(api.restoreBackup).mock.calls[0]?.[0]).toBe(first.name));
    expect(await screen.findByText("Restore scheduled. Restart LM Atelier to apply this backup.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: `Restore backup ${first.name} on restart` })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: `Delete backup ${second.name}` }));
    await waitFor(() => expect(vi.mocked(api.deleteBackup).mock.calls[0]?.[0]).toBe(second.name));
    expect(screen.queryByText(second.name)).not.toBeInTheDocument();
    expect(await screen.findByText("Backup deleted.")).toBeInTheDocument();
    confirm.mockRestore();
  });

  it("prevents duplicate backup creation while a snapshot is pending", async () => {
    let finishBackup: ((backup: BackupInfo) => void) | undefined;
    vi.mocked(api.createBackup).mockImplementation(
      () => new Promise<BackupInfo>((resolve) => { finishBackup = resolve; }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Settings"));
    const create = await screen.findByRole("button", { name: "Back up state" });
    fireEvent.click(create);
    expect(await screen.findByRole("button", { name: "Backing up…" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Backing up…" }));
    expect(api.createBackup).toHaveBeenCalledTimes(1);

    await act(async () => {
      finishBackup?.({
        name: "local-lm-new.sqlite3",
        size_bytes: 1024,
        sha256: "1234567890abcdef",
        created_at: "2026-07-25T12:00:00Z",
        verified: true,
        restore_pending: false,
        media_included: false,
        media_size_bytes: 0,
      });
    });
    expect(await screen.findByText("Backup created.")).toBeInTheDocument();
  });

  it("offers pinned external runtime setup without bundling ComfyUI", async () => {
    const missingRuntime = {
      engine: "llama.cpp" as const,
      release: "b9637",
      state: "missing" as const,
      supported: true,
      managed: false,
      progress: 0,
      downloaded_bytes: 0,
      size_bytes: 16_906_751,
      distribution: "external",
      license: "MIT",
      message: "Installs automatically when first used.",
    };
    vi.mocked(api.runtimes).mockResolvedValue([
      missingRuntime,
      {
        ...missingRuntime,
        engine: "comfyui",
        release: "v0.28.0",
        state: "unsupported",
        supported: false,
        size_bytes: null,
        distribution: "external-gpl-3.0",
        license: "GPL-3.0-only",
        security_status: "blocked",
        security_message: "Automatic setup is paused pending dependency security updates.",
        message: "Automatic setup is paused pending dependency security updates.",
      },
    ]);
    vi.mocked(api.installRuntime).mockResolvedValue({
      ...missingRuntime,
      state: "installing",
      message: "Preparing download.",
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Settings"));
    expect(await screen.findByText("v0.28.0 · Manual setup required")).toBeInTheDocument();
    expect(
      screen.getByText(
        "GPL-3.0-only · Automatic setup is paused pending dependency security updates.",
      ),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Install · 16 MB/ }));
    await waitFor(() => expect(api.installRuntime).toHaveBeenCalledWith("llama.cpp"));
  });

  it("shows support paths and copies only allowlisted technical details", async () => {
    vi.mocked(api.about).mockResolvedValue({
      version: "0.1.7",
      data_directory: "C:\\Users\\someone\\LM Atelier\\data",
      log_directory: "C:\\Users\\someone\\LM Atelier\\data\\logs",
    });
    vi.mocked(api.system).mockResolvedValue({
      platform: "Windows",
      platform_release: "11",
      distribution: "Windows",
      distribution_version: "11",
      architecture: "AMD64",
      python_version: "3.12.10",
      cpu_model: "Test CPU 9000",
      cpu_count: 16,
      memory_total_bytes: 32 * 1024 ** 3,
      memory_available_bytes: 16 * 1024 ** 3,
      disk_total_bytes: 1024 ** 4,
      disk_free_bytes: 512 * 1024 ** 3,
      ffmpeg_available: true,
      devices: [{
        id: "cuda:0",
        name: "Test GPU",
        kind: "gpu",
        total_memory_bytes: 16 * 1024 ** 3,
        available_memory_bytes: 12 * 1024 ** 3,
        backend: "cuda",
        details: {
          credential: "must-not-copy",
          prompt: "private chat content",
          driver_version: "999.1",
        },
      }],
      support: {
        platform_status: "target",
        platform_label: "Windows 11 x64 target",
        accelerator_status: "primary",
        accelerator_label: "CUDA",
        certification_status: "hardware-pending",
        chat_ready: true,
        reference_media_ready: true,
        vram_tier_gb: 16,
        messages: [],
      },
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Settings"));
    expect(await screen.findByText("Version 0.1.7")).toBeInTheDocument();
    expect(screen.getByText("C:\\Users\\someone\\LM Atelier\\data")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Issues" })).toHaveAttribute(
      "href",
      "https://github.com/ajccarlson/lm-atelier/issues",
    );
    expect(screen.getByRole("link", { name: "Security" })).toHaveAttribute(
      "href",
      "https://github.com/ajccarlson/lm-atelier/blob/main/SECURITY.md",
    );
    expect(screen.getByRole("link", { name: "Support" })).toHaveAttribute(
      "href",
      "https://github.com/ajccarlson/lm-atelier/blob/main/SUPPORT.md",
    );
    expect(screen.getByRole("link", { name: "Privacy" })).toHaveAttribute(
      "href",
      "https://github.com/ajccarlson/lm-atelier/blob/main/docs/PRIVACY.md",
    );

    fireEvent.click(screen.getByRole("button", { name: "Copy data folder" }));
    await waitFor(() => {
      expect(clipboardWrite).toHaveBeenLastCalledWith("C:\\Users\\someone\\LM Atelier\\data");
    });

    fireEvent.click(screen.getByRole("button", { name: "Copy technical details" }));
    await waitFor(() => expect(clipboardWrite).toHaveBeenCalledTimes(2));
    const technicalDetails = String(clipboardWrite.mock.calls.at(-1)?.[0]);
    expect(technicalDetails).toContain("LM Atelier: 0.1.7");
    expect(technicalDetails).toContain("Runtime: Python 3.12.10");
    expect(technicalDetails).toContain("CPU: Test CPU 9000");
    expect(technicalDetails).toContain("Test GPU (gpu; cuda; 16 GB)");
    expect(technicalDetails).toContain("mock 1 (chat, image, video)");
    expect(technicalDetails).not.toContain("C:\\Users");
    expect(technicalDetails).not.toContain("must-not-copy");
    expect(technicalDetails).not.toContain("private chat content");
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
    expect(await screen.findByText(/Local chat.*default/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Edit profile: Local chat" }));
    expect(await screen.findByText("Edit profile")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Local chat")).toBeInTheDocument();
    expect(screen.getByText("Default chat model")).toBeInTheDocument();
    const detailLevel = screen.getByRole("group", { name: "Profile setting detail" });
    expect(detailLevel).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "basic" })).toHaveAttribute("aria-pressed", "true");
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
              model_selection: {
                mode: "auto",
                profile_name: "Code specialist",
                matched_terms: ["code", "python"],
                fallback: false,
              },
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
    expect(screen.getAllByText("Auto chose Code specialist · matched code, python")).toHaveLength(2);
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

  it("copies complete messages and fenced code blocks", async () => {
    localStorage.setItem("local-lm-chat", "chat-copy");
    const stamp = "2026-07-22T00:00:00Z";
    const response = [
      "Run this command:",
      "",
      "```powershell",
      "Get-Date",
      "```",
      "",
      "![tracking pixel](https://telemetry.invalid/pixel.png)",
      "",
      "[Documentation](https://example.com/docs)",
    ].join("\n");
    vi.mocked(api.chats).mockResolvedValue([{
      id: "chat-copy",
      project_id: null,
      title: "Copy text",
      archived: false,
      routing_mode: "text",
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-copy",
      created_at: stamp,
      updated_at: stamp,
    }]);
    vi.mocked(api.chat).mockResolvedValue({
      id: "chat-copy",
      project_id: null,
      title: "Copy text",
      archived: false,
      routing_mode: "text",
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-copy",
      created_at: stamp,
      updated_at: stamp,
      messages: [
        {
          id: "user-copy",
          chat_id: "chat-copy",
          parent_id: null,
          role: "user",
          status: "complete",
          parts: [{ id: "user-text", position: 0, type: "text", text: "Show a command", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
        {
          id: "assistant-copy",
          chat_id: "chat-copy",
          parent_id: "user-copy",
          role: "assistant",
          status: "complete",
          parts: [{ id: "assistant-text", position: 0, type: "text", text: response, artifact_id: null, metadata_json: {} }],
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

    fireEvent.click(await screen.findByRole("button", { name: "Copy assistant message" }));
    await waitFor(() => expect(clipboardWrite).toHaveBeenCalledWith(response));

    fireEvent.click(screen.getByRole("button", { name: "Copy code block" }));
    await waitFor(() => expect(clipboardWrite).toHaveBeenCalledWith("Get-Date"));
    expect(screen.queryByRole("img", { name: "tracking pixel" })).not.toBeInTheDocument();
    expect(screen.getByText("[Image: tracking pixel]")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Documentation" })).toHaveAttribute(
      "rel",
      "noopener noreferrer",
    );
  });

  it("shows the persisted chat startup phase before the first token", async () => {
    localStorage.setItem("local-lm-chat", "chat-starting");
    const stamp = new Date().toISOString();
    const chat = {
      id: "chat-starting",
      project_id: null,
      title: "Starting response",
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-starting",
      created_at: stamp,
      updated_at: stamp,
    };
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({
      ...chat,
      messages: [
        {
          id: "user-starting",
          chat_id: chat.id,
          parent_id: null,
          role: "user",
          status: "complete",
          parts: [{ id: "prompt", position: 0, type: "text", text: "Hello", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
        {
          id: "assistant-starting",
          chat_id: chat.id,
          parent_id: "user-starting",
          role: "assistant",
          status: "pending",
          parts: [
            { id: "empty-text", position: 0, type: "text", text: "", artifact_id: null, metadata_json: {} },
            { id: "chat-progress", position: 1, type: "progress", text: "Preparing chat model", artifact_id: null, metadata_json: { activity: "chat", progress: 0, phase: "preparing chat model" } },
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

    const status = await screen.findByRole("status");
    expect(status).toHaveTextContent(/Preparing chat model/);
    expect(status).toHaveTextContent(/· 0s/);
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
    expect(screen.getByRole("button", { name: "Unload chat worker" })).toBeDisabled();
  });

  it("shows concise redacted worker failure diagnostics and the local log", async () => {
    vi.mocked(api.workers).mockResolvedValue([
      {
        name: "chat",
        state: "exited",
        managed: true,
        running: false,
        pid: null,
        profile_id: "profile-1",
        command: ["[local path]", "--model", "[data folder]/models/model.gguf"],
        exit_code: 1,
        estimated_memory_bytes: null,
        current_memory_bytes: null,
        peak_memory_bytes: null,
        active_jobs: 0,
        queued_jobs: 0,
        failure_detail: "chat worker exited with code 1.",
        stderr_tail: "model loader: unsupported architecture",
        log_path: "logs/chat-worker.log",
      },
    ]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Settings"));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("chat worker exited with code 1.");
    expect(alert).toHaveTextContent("model loader: unsupported architecture");
    expect(alert).toHaveTextContent("Log · Data folder/logs/chat-worker.log");
    expect(screen.getByLabelText("chat worker error output")).toBeInTheDocument();
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
    expect(await screen.findByRole("img", { name: "mock engine: ready" })).toBeInTheDocument();
    expect(screen.queryByText("Download redacted diagnostics")).not.toBeInTheDocument();
    expect(screen.queryByText("FFmpeg")).not.toBeInTheDocument();
    expect(screen.queryByText("memory free")).not.toBeInTheDocument();
    expect(screen.queryByText(/available$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/RAM is measured for the managed process tree/)).not.toBeInTheDocument();
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
        readiness: "unverified",
        capability_evidence: null,
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
    expect(screen.getByRole("button", { name: "Delete Local GGUF" })).toBeEnabled();
  });

  it("omits the partial-download cleanup action when there is nothing to clean", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Model library"));
    expect(await screen.findByRole("heading", { name: "Model library" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clean 0 partial" })).not.toBeInTheDocument();
  });

  it("distinguishes installed text-only and vision-capable chat models", async () => {
    const installedBase = {
      source_id: null,
      role: "chat" as const,
      engine: "llama.cpp",
      size_bytes: 4096,
      compatibility: "likely",
      manifest_json: {},
      active: true,
      readiness: "ready" as const,
      capability_evidence: null,
      created_at: "2026-07-22T00:00:00Z",
      updated_at: "2026-07-22T00:00:00Z",
    };
    vi.mocked(api.models).mockResolvedValue([
      {
        ...installedBase,
        id: "model-text",
        name: "Text specialist",
        local_path: "/models/text.gguf",
      },
      {
        ...installedBase,
        id: "model-vision",
        name: "Visual observer",
        local_path: "/models/vision.gguf",
      },
    ]);
    vi.mocked(api.profiles).mockResolvedValue([
      {
        id: "profile-text",
        model_install_id: "model-text",
        name: "Text specialist",
        use_case: "",
        role: "chat",
        engine: "llama.cpp",
        load_settings_json: {},
        request_settings_json: {},
        is_default: false,
        input_modalities: ["text"],
      },
      {
        id: "profile-vision",
        model_install_id: "model-vision",
        name: "Visual observer",
        use_case: "",
        role: "chat",
        engine: "llama.cpp",
        load_settings_json: {},
        request_settings_json: {},
        is_default: false,
        input_modalities: ["text", "image"],
      },
    ]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Model library"));
    const capability = await screen.findByRole("combobox", {
      name: "Installed chat capability",
    });
    fireEvent.change(capability, { target: { value: "vision" } });
    expect(screen.getByRole("button", { name: "Delete Visual observer" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Delete Text specialist" })).not.toBeInTheDocument();

    fireEvent.change(capability, { target: { value: "text" } });
    expect(screen.getByRole("button", { name: "Delete Text specialist" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Delete Visual observer" })).not.toBeInTheDocument();
  });

  it("edits Auto Mode use cases beside installed models", async () => {
    vi.mocked(api.models).mockResolvedValue([
      {
        id: "model-use-case",
        source_id: null,
        name: "Local specialist",
        role: "chat",
        engine: "llama.cpp",
        local_path: "/models/specialist.gguf",
        size_bytes: 4096,
        compatibility: "likely",
        manifest_json: {},
        active: true,
        readiness: "unverified",
        capability_evidence: null,
        created_at: "2026-07-22T00:00:00Z",
        updated_at: "2026-07-22T00:00:00Z",
      },
    ]);
    const profile = {
      id: "profile-use-case",
      model_install_id: "model-use-case",
      name: "Local specialist",
      use_case: "General conversation",
      role: "chat" as const,
      engine: "llama.cpp",
      load_settings_json: {},
      request_settings_json: {},
      is_default: false,
    };
    vi.mocked(api.profiles).mockResolvedValue([profile]);
    vi.mocked(api.updateProfile).mockImplementation(async (_id, values) => {
      const updated = { ...profile, use_case: values.use_case ?? profile.use_case };
      vi.mocked(api.profiles).mockResolvedValue([updated]);
      return updated;
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Model library"));
    expect(await screen.findByText("General conversation")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Edit use case for Local specialist" }));
    fireEvent.change(screen.getByLabelText("Best uses for Local specialist"), {
      target: { value: "Python programming and code review" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(api.updateProfile).toHaveBeenCalledWith(
      "profile-use-case",
      { use_case: "Python programming and code review" },
    ));
    expect(await screen.findByText("Python programming and code review")).toBeInTheDocument();
  });

  it("sets an installed model as the default for its type", async () => {
    const model = {
      id: "model-default-image",
      source_id: null,
      name: "Image specialist",
      role: "image",
      engine: "comfyui",
      local_path: "/models/image-specialist",
      size_bytes: 4096,
      compatibility: "likely",
      manifest_json: {},
      active: true,
      readiness: "ready" as const,
      capability_evidence: null,
      created_at: "2026-07-22T00:00:00Z",
      updated_at: "2026-07-22T00:00:00Z",
    };
    const profile = {
      id: "profile-default-image",
      model_install_id: model.id,
      name: model.name,
      use_case: "",
      role: "image" as const,
      engine: model.engine,
      load_settings_json: {},
      request_settings_json: {},
      is_default: false,
    };
    const updated = { ...profile, is_default: true };
    vi.mocked(api.models).mockResolvedValue([model]);
    vi.mocked(api.profiles).mockResolvedValue([profile]);
    vi.mocked(api.updateProfile).mockImplementation(async () => {
      vi.mocked(api.profiles).mockResolvedValue([updated]);
      return updated;
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Model library"));
    fireEvent.click(await screen.findByRole("button", {
      name: "Set Image specialist as default image model",
    }));

    await waitFor(() => expect(api.updateProfile).toHaveBeenCalledWith(
      profile.id,
      { is_default: true },
    ));
    expect(await screen.findByText("Default")).toBeInTheDocument();
  });

  it("does not mark a catalog model installed from an inactive replacement", async () => {
    const model = {
      provider: "huggingface",
      remote_id: "stabilityai/sdxl-turbo",
      name: "sdxl-turbo",
      author: "stabilityai",
      pipeline_tag: "text-to-image",
      tags: ["diffusers", "safetensors"],
      downloads: 42,
      likes: 3,
      trending_score: 1,
      created_at: "2026-07-22T00:00:00Z",
      last_modified: "2026-07-22T00:00:00Z",
      gated: false,
      private: false,
      library_name: "diffusers",
      architecture: "sdxl",
      formats: ["safetensors"],
      quantizations: [],
      parameter_count: null,
      license_id: "openrail++",
      total_size_bytes: 1024,
      compatibility: "likely",
      compatibility_reasons: ["safetensors artifact detected"],
    };
    vi.mocked(api.models).mockResolvedValue([
      {
        id: "model-inactive",
        source_id: null,
        name: "z_image_turbo",
        role: "image",
        engine: "comfyui",
        local_path: "/models/z-image",
        size_bytes: 2048,
        compatibility: "likely",
        manifest_json: {
          remote_id: "Comfy-Org/z_image_turbo",
          source_remote_id: model.remote_id,
        },
        active: false,
        readiness: "unverified",
        capability_evidence: null,
        created_at: "2026-07-22T00:00:00Z",
        updated_at: "2026-07-22T00:00:00Z",
      },
    ]);
    vi.mocked(api.catalog).mockResolvedValue({ items: [model], next_cursor: null });
    vi.mocked(api.workflowCatalogModels).mockResolvedValue([model]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("Model library"));
    fireEvent.change(screen.getByLabelText("Model role"), { target: { value: "image" } });

    await waitFor(() => expect(api.workflowCatalogModels).toHaveBeenCalledWith("image"));
    expect(await screen.findByRole("button", { name: "Install" })).toBeEnabled();
    expect(screen.getAllByText("sdxl-turbo")).toHaveLength(1);
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
    expect(screen.getByLabelText("Last updated filter")).toBeInTheDocument();
    expect(screen.getByLabelText("Format filter")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Last updated filter"), { target: { value: "30" } });
    await waitFor(() => expect(vi.mocked(api.catalog).mock.calls.at(-1)?.[4]).toMatchObject({
      updated_within_days: "30",
    }));
    fireEvent.click(await screen.findByText("Load more models"));
    await waitFor(() => expect(vi.mocked(api.catalog).mock.calls.at(-1)?.[3]).toContain("cursor=next"));
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
      source_remote_id: null,
      revision: "main",
      selected_files: ["model-q4.gguf"],
      expected_sha256: { "model-q4.gguf": "a".repeat(64) },
      comfy_paths: {},
      workflow_template_id: null,
      workflow_template_sha256: null,
      download_bytes: 1024,
      available_disk_bytes: 4096,
      estimated_ram_bytes: 2048,
      estimated_vram_bytes: null,
      can_install: true,
      install_plan: {
        id: "plan-chat",
        plan_hash: "b".repeat(64),
        compatibility: "supported",
        family: "qwen",
        failure_code: null,
        failure_reason: null,
      },
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
      null,
      "chat",
      "llama.cpp",
      "main",
      ["model-q4.gguf"],
      { "model-q4.gguf": "a".repeat(64) },
      {},
      null,
      null,
      "plan-chat",
    ));
  });

  it("gates ModelOpt downloads on vLLM readiness and selects vLLM when ready", async () => {
    const model = {
      provider: "huggingface",
      remote_id: "owner/qwen-nvfp4",
      name: "qwen-nvfp4",
      author: "owner",
      pipeline_tag: "image-text-to-text",
      tags: ["modelopt", "nvfp4", "safetensors"],
      downloads: 12,
      likes: 2,
      trending_score: 1,
      created_at: "2026-07-26T00:00:00Z",
      last_modified: "2026-07-26T00:00:00Z",
      gated: false,
      private: false,
      library_name: "transformers",
      architecture: "qwen3_5",
      formats: ["safetensors"],
      quantizations: ["nvfp4"],
      parameter_count: 27_000_000_000,
      license_id: "apache-2.0",
      total_size_bytes: 20 * 1024 ** 3,
      compatibility: "advanced_import",
      compatibility_reasons: ["requires the managed vLLM ModelOpt runtime"],
      required_runtime: "vllm",
    };
    vi.mocked(api.catalog).mockResolvedValue({ items: [model], next_cursor: null });
    vi.mocked(api.runtimes).mockResolvedValue([
      {
        engine: "vllm",
        release: "v0.25.0",
        state: "unsupported",
        supported: false,
        managed: false,
        progress: 0,
        downloaded_bytes: 0,
        size_bytes: null,
        distribution: "external",
        license: "Apache-2.0",
        security_status: "blocked",
        security_message: "Dependency review pending.",
        message: "Dependency review pending.",
      },
    ]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const rendered = render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Model library"));
    expect(await screen.findByRole("button", { name: "Needs vLLM" })).toBeDisabled();

    rendered.unmount();
    vi.mocked(api.runtimes).mockResolvedValue([
      {
        engine: "vllm",
        release: "v0.25.0",
        state: "ready",
        supported: true,
        managed: true,
        progress: 1,
        downloaded_bytes: 0,
        size_bytes: null,
        distribution: "external",
        license: "Apache-2.0",
        security_status: "checksum-pinned",
        security_message: "",
        message: "Ready.",
      },
    ]);
    vi.mocked(api.catalogPreflight).mockResolvedValue({
      remote_id: model.remote_id,
      source_remote_id: null,
      revision: "main",
      selected_files: ["config.json", "hf_quant_config.json", "model.safetensors"],
      expected_sha256: {},
      comfy_paths: {},
      workflow_template_id: null,
      workflow_template_sha256: null,
      download_bytes: model.total_size_bytes,
      available_disk_bytes: model.total_size_bytes * 2,
      estimated_ram_bytes: null,
      estimated_vram_bytes: model.total_size_bytes,
      can_install: true,
      install_plan: {
        id: "plan-nvfp4",
        plan_hash: "b".repeat(64),
        compatibility: "supported",
        family: "qwen",
        failure_code: null,
        failure_reason: null,
      },
      checks: [],
    });
    const readyClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={readyClient}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Model library"));
    fireEvent.click(await screen.findByRole("button", { name: "Install" }));
    await waitFor(() => expect(api.catalogPreflight).toHaveBeenCalledWith(
      model.remote_id,
      "chat",
      "vllm",
      "main",
      [],
    ));
  });

  it("preflights and queues a LoRA as a verified auxiliary asset", async () => {
    const model = {
      provider: "huggingface",
      remote_id: "owner/atelier-ink-lora",
      name: "atelier-ink-lora",
      author: "owner",
      pipeline_tag: null,
      tags: ["lora", "safetensors"],
      downloads: 42,
      likes: 3,
      trending_score: 1,
      created_at: "2026-07-26T00:00:00Z",
      last_modified: "2026-07-26T00:00:00Z",
      gated: false,
      private: false,
      library_name: null,
      architecture: "sdxl",
      formats: ["safetensors"],
      quantizations: [],
      parameter_count: null,
      license_id: "other",
      total_size_bytes: 1024,
      compatibility: "likely",
      compatibility_reasons: ["data-only LoRA candidate"],
    };
    vi.mocked(api.catalog).mockResolvedValue({ items: [model], next_cursor: null });
    vi.mocked(api.catalogPreflight).mockResolvedValue({
      remote_id: model.remote_id,
      source_remote_id: null,
      revision: "main",
      selected_files: ["atelier-ink.safetensors"],
      expected_sha256: { "atelier-ink.safetensors": "a".repeat(64) },
      comfy_paths: { loras: "." },
      workflow_template_id: null,
      workflow_template_sha256: null,
      download_bytes: 1024,
      available_disk_bytes: 4096,
      estimated_ram_bytes: null,
      estimated_vram_bytes: null,
      can_install: true,
      auxiliary_kind: "lora",
      install_plan: {
        id: "plan-lora",
        plan_hash: "b".repeat(64),
        compatibility: "supported",
        family: "sdxl",
        failure_code: null,
        failure_reason: null,
      },
      checks: [],
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Model library"));
    fireEvent.change(screen.getByLabelText("Model role"), { target: { value: "lora" } });
    fireEvent.click(await screen.findByText("Install"));

    await waitFor(() => expect(api.catalogPreflight).toHaveBeenCalledWith(
      model.remote_id,
      "image",
      "comfyui",
      "main",
      [],
      "lora",
    ));
    await waitFor(() => expect(api.download).toHaveBeenCalledWith(
      model.remote_id,
      null,
      "image",
      "comfyui",
      "main",
      ["atelier-ink.safetensors"],
      { "atelier-ink.safetensors": "a".repeat(64) },
      { loras: "." },
      null,
      null,
      "plan-lora",
      "lora",
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
      retention_pending_count: 0,
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
    expect(screen.getByText("Run cleanup")).toBeDisabled();
    vi.mocked(api.deleteArtifact).mockImplementation(async () => {
      vi.mocked(api.artifacts).mockResolvedValue([]);
      return {
        artifact_id: "sha256:image",
        reference_count: 1,
        removed_count: 1,
        reclaimed_bytes: 2048,
      };
    });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: "Delete observatory.png" }));
    expect(confirm).toHaveBeenCalledWith(
      "Permanently delete observatory.png and remove 1 appearance from chats?",
    );
    await waitFor(() => expect(vi.mocked(api.deleteArtifact).mock.calls[0]?.[0]).toBe("sha256:image"));
    await waitFor(() => expect(screen.queryByText("observatory.png")).not.toBeInTheDocument());
  });

  it("explains when cleanup only finds media still in the recovery window", async () => {
    vi.mocked(api.artifactStorage).mockResolvedValue({
      total_bytes: 2048,
      total_count: 1,
      referenced_bytes: 0,
      referenced_count: 0,
      unreferenced_bytes: 2048,
      unreferenced_count: 1,
      temporary_bytes: 0,
      temporary_count: 0,
      eligible_bytes: 0,
      eligible_count: 0,
      retention_pending_count: 1,
      disk_free_bytes: 1024 ** 3,
      warning: false,
      retention_days: 30,
      temporary_retention_hours: 24,
    });
    vi.mocked(api.cleanupArtifacts).mockResolvedValue({
      dry_run: false,
      marked_count: 0,
      retention_pending_count: 1,
      removed_count: 0,
      reclaimed_bytes: 0,
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByText("Media library"));
    fireEvent.click(await screen.findByRole("button", { name: "Run cleanup" }));
    expect(await screen.findByText("1 artifact remains in the recovery window; none are eligible yet.")).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: "Edit profile: Image profile" }));
    expect(await screen.findByText("Negative prompt")).toBeInTheDocument();
    expect(screen.queryByText("Frames")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Close profile editor" }));
    fireEvent.click(screen.getByRole("button", { name: "Edit preset: Video preset" }));
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
    fireEvent.click(screen.getByRole("button", { name: "Edit preset: Chat preset" }));
    expect(await screen.findByText("Maximum output")).toBeInTheDocument();
    expect(screen.queryByText("Context length")).not.toBeInTheDocument();
  });

  it("isolates role-aware settings in per-chat defaults", async () => {
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
    let persistedChat: Chat = { ...chat };
    vi.mocked(api.engines).mockResolvedValue([roleAwareMediaEngine]);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockImplementation(async () => ({ ...persistedChat, messages: [] }));
    vi.mocked(api.updateChat).mockImplementation(async (_id, values) => {
      persistedChat = { ...persistedChat, ...values };
      return persistedChat;
    });
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

  it("shows an Auto submission while model routing is pending", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const chat = {
      id: "chat-auto-routing",
      project_id: null,
      title: "Auto routing",
      archived: false,
      routing_mode: "auto" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      generation_settings_json: { chat: { max_tokens: 4096 } },
      created_at: stamp,
      updated_at: stamp,
    };
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    vi.mocked(api.sendTurn).mockReturnValue(new Promise(() => {}));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const composer = await screen.findByRole("textbox", { name: "Message" });
    expect(screen.getByRole("combobox", { name: "Generation mode" })).toHaveValue("auto");
    fireEvent.change(composer, { target: { value: "Surprise me with a tiny story" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("Surprise me with a tiny story")).toBeVisible();
    expect(api.sendTurn).toHaveBeenCalledWith(
      chat.id,
      "Surprise me with a tiny story",
      "auto",
      [],
      {},
      expect.any(String),
    );
    expect(screen.getByText("Choosing mode and model…")).toBeVisible();
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });

  it("keeps the composer available and orders multiple optimistic submissions", async () => {
    const stamp = "2026-07-26T00:00:00Z";
    const chat = {
      id: "chat-continuous",
      project_id: null,
      title: "Continuous submission",
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      created_at: stamp,
      updated_at: stamp,
    };
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
    vi.mocked(api.sendTurn).mockReturnValue(new Promise(() => {}));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const composer = await screen.findByRole("textbox", { name: "Message" });
    for (const prompt of ["First queued prompt", "Second queued prompt", "Third queued prompt"]) {
      fireEvent.change(composer, { target: { value: prompt } });
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    }

    expect(composer).toBeEnabled();
    await waitFor(() => expect(api.sendTurn).toHaveBeenCalledTimes(3));
    expect(
      Array.from(
        container.querySelectorAll(".message.user.optimistic .message-text"),
        (element) => element.textContent,
      ),
    ).toEqual(["First queued prompt", "Second queued prompt", "Third queued prompt"]);
  });

  it("offers queued cancellation and an explicit Stop and send action", async () => {
    const stamp = new Date().toISOString();
    const chat = {
      id: "chat-stop-and-send",
      project_id: null,
      title: "Stop controls",
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-queued",
      created_at: stamp,
      updated_at: stamp,
    };
    const plan = {
      id: "plan-queued",
      chat_id: chat.id,
      idempotency_key: "queued-key",
      source_action: "send",
      persistence_scope: "durable" as const,
      status: "queued",
      context_head_message_id: null,
      transcript_sequence: 1,
      priority: 10,
      planner_version: "legacy-turn-v1",
      failure_policy: "stop_dependents",
      summary_json: { assistant_message_id: "assistant-queued" },
      steps: [],
      created_at: stamp,
      updated_at: stamp,
    };
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({
      ...chat,
      messages: [
        {
          id: "user-queued",
          chat_id: chat.id,
          parent_id: null,
          role: "user",
          status: "complete",
          parts: [{ id: "prompt", position: 0, type: "text", text: "Original", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
        {
          id: "assistant-queued",
          chat_id: chat.id,
          parent_id: "user-queued",
          role: "assistant",
          status: "pending",
          parts: [{ id: "queued", position: 0, type: "progress", text: "Queued", artifact_id: null, metadata_json: { activity: "chat", progress: 0 } }],
          created_at: stamp,
          updated_at: stamp,
        },
      ],
    });
    vi.mocked(api.workPlans).mockResolvedValue([plan]);
    vi.mocked(api.cancelWorkPlan).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.stopAndSendTurn).mockReturnValue(new Promise(() => {}));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Cancel queued item" }));
    await waitFor(() => expect(api.cancelWorkPlan).toHaveBeenCalledWith(
      plan.id,
      expect.anything(),
    ));
    const composer = screen.getByRole("textbox", { name: "Message" });
    fireEvent.change(composer, { target: { value: "Use this instead" } });
    expect(screen.getByRole("button", { name: "Stop current response" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Stop current response and send" }));
    await waitFor(() => expect(api.stopAndSendTurn).toHaveBeenCalledWith(
      chat.id,
      "Use this instead",
      "text",
      [],
      {},
      expect.any(String),
    ));
  });

  it("shows ordered media output status with per-output controls", async () => {
    const stamp = new Date().toISOString();
    const chat = {
      id: "chat-media-batch",
      project_id: null,
      title: "Media batch",
      archived: false,
      routing_mode: "image" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-output-2",
      created_at: stamp,
      updated_at: stamp,
    };
    const step = (ordinal: number, status: string) => ({
      id: `step-${ordinal}`,
      plan_id: "plan-media-batch",
      run_id: `run-${ordinal}`,
      ordinal,
      display_group: "media_outputs",
      operation: "text_to_image",
      status,
      prompt: "Blue apples",
      profile_id: null,
      workflow_revision_id: null,
      settings_json: {},
      input_bindings_json: [],
      output_contract_json: [{ slot: `output-${ordinal}`, type: "image" }],
      queue_class: "media_compute",
      error: status === "cancelled" ? "Cancelled" : null,
      created_at: stamp,
      updated_at: stamp,
    });
    const plan = {
      id: "plan-media-batch",
      chat_id: chat.id,
      idempotency_key: null,
      source_action: "send",
      persistence_scope: "durable" as const,
      status: "queued",
      context_head_message_id: null,
      transcript_sequence: 1,
      priority: 0,
      planner_version: "media-outputs-v1",
      failure_policy: "stop_dependents",
      summary_json: {
        assistant_message_id: "assistant-output-1",
        assistant_message_ids: ["assistant-output-1", "assistant-output-2"],
      },
      steps: [step(1, "queued"), step(2, "cancelled")],
      created_at: stamp,
      updated_at: stamp,
    };
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({
      ...chat,
      messages: [
        {
          id: "user-media-batch",
          chat_id: chat.id,
          parent_id: null,
          role: "user",
          status: "complete",
          parts: [{ id: "prompt", position: 0, type: "text", text: "Two blue apples", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
        {
          id: "assistant-output-1",
          chat_id: chat.id,
          parent_id: "user-media-batch",
          role: "assistant",
          status: "pending",
          parts: [{ id: "progress-1", position: 0, type: "progress", text: "Queued", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
        {
          id: "assistant-output-2",
          chat_id: chat.id,
          parent_id: "assistant-output-1",
          role: "assistant",
          status: "cancelled",
          parts: [{ id: "progress-2", position: 0, type: "progress", text: "Cancelled", artifact_id: null, metadata_json: {} }],
          created_at: stamp,
          updated_at: stamp,
        },
      ],
    });
    vi.mocked(api.workPlans).mockResolvedValue([plan]);
    vi.mocked(api.cancelWorkStep).mockReturnValue(new Promise(() => {}));
    vi.mocked(api.retryWorkStep).mockReturnValue(new Promise(() => {}));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByText("2 media outputs"));
    expect(screen.getByText("1 queued · 1 cancelled")).toBeInTheDocument();
    expect(screen.getByText("Output 1")).toBeInTheDocument();
    expect(screen.getByText("Output 2")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel output 1" }));
    fireEvent.click(screen.getByRole("button", { name: "Retry output 2" }));
    await waitFor(() => {
      expect(api.cancelWorkStep).toHaveBeenCalledWith("step-1", expect.anything());
      expect(api.retryWorkStep).toHaveBeenCalledWith("step-2", expect.anything());
    });
    client.setQueryData(["work-plans", chat.id], [
      {
        ...plan,
        planner_version: "ordered-work-v1",
        steps: [
          {
            ...plan.steps[0],
            operation: "text",
            output_contract_json: [{ slot: "story", type: "text" }],
          },
          {
            ...plan.steps[1],
            operation: "text_to_image",
            output_contract_json: [{ slot: "visual", type: "image" }],
          },
        ],
      },
    ]);
    expect(await screen.findByText("2-step plan")).toBeInTheDocument();
    fireEvent.click(screen.getByText("2-step plan"));
    expect(screen.getByText("Step 1 · Text")).toBeInTheDocument();
    expect(screen.getByText("Step 2 · Image")).toBeInTheDocument();
  });

  it("keeps a deferred turn and its pending state on the originating chat", async () => {
    const stamp = "2026-07-25T12:00:00Z";
    const makeChat = (id: string, title: string): Chat => ({
      id,
      project_id: null,
      title,
      archived: false,
      routing_mode: "text",
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      created_at: stamp,
      updated_at: stamp,
    });
    const first = makeChat("chat-origin", "Origin chat");
    const second = makeChat("chat-destination", "Destination chat");
    const details = new Map<string, ChatDetail>([
      [first.id, { ...first, messages: [] }],
      [second.id, { ...second, messages: [] }],
    ]);
    const accepted: TurnAccepted = {
      run: {
        id: "run-origin",
        idempotency_key: "turn-origin",
        chat_id: first.id,
        user_message_id: "user-origin",
        assistant_message_id: "assistant-origin",
        operation: "text",
        status: "queued",
        standalone_prompt: "Stay with the first chat",
        profile_id: null,
        workflow_revision_id: null,
        settings_json: {},
        provenance_json: {},
        error: null,
        created_at: stamp,
      },
      user_message: {
        id: "user-origin",
        chat_id: first.id,
        parent_id: null,
        role: "user",
        status: "complete",
        parts: [{
          id: "user-origin-part",
          position: 0,
          type: "text",
          text: "Stay with the first chat",
          artifact_id: null,
          metadata_json: {},
        }],
        created_at: stamp,
        updated_at: stamp,
      },
      assistant_message: {
        id: "assistant-origin",
        chat_id: first.id,
        parent_id: "user-origin",
        role: "assistant",
        status: "pending",
        parts: [],
        created_at: stamp,
        updated_at: stamp,
      },
    };
    let finishTurn: ((turn: TurnAccepted) => void) | undefined;
    vi.mocked(api.sendTurn).mockImplementation(
      () => new Promise<TurnAccepted>((resolve) => { finishTurn = resolve; }),
    );
    vi.mocked(api.chats).mockResolvedValue([first, second]);
    vi.mocked(api.chat).mockImplementation(async (id) => details.get(id)!);
    localStorage.setItem("local-lm-chat", first.id);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const composer = await screen.findByRole("textbox", { name: "Message" });
    fireEvent.change(composer, { target: { value: "Stay with the first chat" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("Stay with the first chat")).toBeVisible();

    fireEvent.click(screen.getByText(second.title));
    expect(await screen.findByRole("heading", { name: second.title })).toBeInTheDocument();
    expect(screen.queryByText("Stay with the first chat")).not.toBeInTheDocument();
    const secondComposer = screen.getByRole("textbox", { name: "Message" });
    expect(secondComposer).toBeEnabled();
    fireEvent.change(secondComposer, { target: { value: "A second-chat message" } });
    expect(screen.getByRole("button", { name: "Send" })).toBeEnabled();

    await act(async () => {
      finishTurn?.(accepted);
    });
    await waitFor(() => {
      const origin = client.getQueryData<ChatDetail>(["chat", first.id]);
      expect(origin?.messages.map((message) => message.id)).toEqual([
        "user-origin",
        "assistant-origin",
      ]);
    });
    expect(client.getQueryData<ChatDetail>(["chat", second.id])?.messages).toEqual([]);
    expect(screen.getByRole("heading", { name: second.title })).toBeInTheDocument();
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

  it("selects prior-image workflow controls only for an explicit visual follow-up", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const chat = {
      id: "chat-prior-image-controls",
      project_id: null,
      title: "Prior image controls",
      archived: false,
      routing_mode: "image" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-prior-image",
      created_at: stamp,
      updated_at: stamp,
    };
    const userMessage = {
      id: "user-prior-image",
      chat_id: chat.id,
      parent_id: null,
      role: "user" as const,
      status: "complete" as const,
      parts: [{ id: "prior-prompt", position: 0, type: "text" as const, text: "Make an apple", artifact_id: null, metadata_json: {} }],
      created_at: stamp,
      updated_at: stamp,
    };
    const assistantMessage = {
      id: "assistant-prior-image",
      chat_id: chat.id,
      parent_id: userMessage.id,
      role: "assistant" as const,
      status: "complete" as const,
      parts: [{ id: "prior-image", position: 0, type: "image" as const, text: null, artifact_id: "sha256:prior", metadata_json: {} }],
      created_at: stamp,
      updated_at: stamp,
    };
    const workflow = (
      id: string,
      operation: "text_to_image" | "image_to_image",
      title: string,
    ) => ({
      id,
      name: title,
      operation,
      description: "",
      current_revision_id: `${id}-revision`,
      revisions: [{
        id: `${id}-revision`,
        workflow_id: id,
        version: 1,
        engine: "mock",
        engine_version: null,
        ui_graph_json: {},
        api_graph_json: {},
        input_schema_json: {
          type: "object",
          properties: {
            negative_prompt: {
              type: "string",
              title,
              default: "",
            },
          },
        },
        dependencies_json: {},
        trusted: true,
        created_at: stamp,
      }],
    });
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.engines).mockResolvedValue([roleAwareMediaEngine]);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({
      ...chat,
      messages: [userMessage, assistantMessage],
    });
    vi.mocked(api.workflows).mockResolvedValue([
      workflow("fresh-image", "text_to_image", "Fresh image exclusion"),
      workflow("edit-image", "image_to_image", "Edit image exclusion"),
    ]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const composer = await screen.findByRole("textbox", { name: "Message" });
    fireEvent.change(composer, { target: { value: "Create an image of a pear" } });
    fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
    expect(screen.getByLabelText(/Fresh image exclusion/)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Edit image exclusion/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));

    fireEvent.change(composer, { target: { value: "Make it green" } });
    fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
    expect(screen.getByLabelText(/Edit image exclusion/)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Fresh image exclusion/)).not.toBeInTheDocument();
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
    let persistedChat: Chat = { ...chat };
    vi.mocked(api.chat).mockImplementation(async () => ({
      ...persistedChat,
      messages: [userMessage, assistantMessage],
    }));
    vi.mocked(api.updateChat).mockImplementation(async (_id, values) => {
      persistedChat = { ...persistedChat, ...values };
      return persistedChat;
    });
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
    await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(chat.id, {
      generation_settings_json: { chat: { max_tokens: 4096 } },
    }));

    fireEvent.click(screen.getByRole("button", { name: "Regenerate response" }));
    await waitFor(() => expect(api.regenerateMessage).toHaveBeenCalledWith(assistantMessage.id, { max_tokens: 4096 }));

    fireEvent.click(screen.getByText("Edit and branch"));
    fireEvent.change(screen.getByLabelText("Edit message"), { target: { value: "Count to 1000" } });
    fireEvent.click(screen.getByText("Send edited message"));
    await waitFor(() => expect(api.branchMessage).toHaveBeenCalledWith(
      userMessage.id,
      "Count to 1000",
      "text",
      { max_tokens: 4096 },
    ));

    fireEvent.change(screen.getByRole("textbox", { name: "Message" }), { target: { value: "Count to 1000" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(api.sendTurn).toHaveBeenCalledWith(
      chat.id,
      "Count to 1000",
      "text",
      [],
      { max_tokens: 4096 },
      expect.any(String),
    ));
  });

  it("switches completed response revisions without branching the chat", async () => {
    const stamp = "2026-07-25T12:00:00Z";
    const chat = {
      id: "chat-revisions",
      project_id: null,
      title: "Response revisions",
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: "assistant-revisions",
      created_at: stamp,
      updated_at: stamp,
    };
    const userMessage = {
      id: "user-revisions",
      chat_id: chat.id,
      parent_id: null,
      role: "user" as const,
      status: "complete" as const,
      parts: [{
        id: "user-revisions-part",
        position: 0,
        type: "text" as const,
        text: "Write a response",
        artifact_id: null,
        metadata_json: {},
      }],
      created_at: stamp,
      updated_at: stamp,
    };
    const firstPart = {
      id: "revision-part-one",
      position: 0,
      type: "text" as const,
      text: "First response",
      artifact_id: null,
      metadata_json: {},
    };
    const secondPart = {
      id: "revision-part-two",
      position: 0,
      type: "text" as const,
      text: "Second response",
      artifact_id: null,
      metadata_json: {},
    };
    const assistantMessage = {
      id: "assistant-revisions",
      chat_id: chat.id,
      parent_id: userMessage.id,
      role: "assistant" as const,
      status: "complete" as const,
      transcript_visible: true,
      active_response_revision_id: "revision-two",
      parts: [secondPart],
      response_revisions: [
        {
          id: "revision-one",
          message_id: "assistant-revisions",
          run_id: "run-one",
          sequence: 1,
          status: "complete" as const,
          parts: [firstPart],
          created_at: stamp,
          updated_at: stamp,
        },
        {
          id: "revision-two",
          message_id: "assistant-revisions",
          run_id: "run-two",
          sequence: 2,
          status: "complete" as const,
          parts: [secondPart],
          created_at: stamp,
          updated_at: stamp,
        },
      ],
      created_at: stamp,
      updated_at: stamp,
    };
    localStorage.setItem("local-lm-chat", chat.id);
    vi.mocked(api.chats).mockResolvedValue([chat]);
    vi.mocked(api.chat).mockResolvedValue({
      ...chat,
      messages: [userMessage, assistantMessage],
    });
    vi.mocked(api.selectResponseRevision).mockResolvedValue({
      ...assistantMessage,
      active_response_revision_id: "revision-one",
      parts: [firstPart],
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    expect(await screen.findByText("Second response")).toBeVisible();
    expect(screen.getByText("2 / 2")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Previous response revision" }));
    await waitFor(() => expect(api.selectResponseRevision).toHaveBeenCalledWith(
      assistantMessage.id,
      "revision-one",
    ));
    expect(api.branchMessage).not.toHaveBeenCalled();
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
    const storedChats = new Map<string, Chat>([
      [firstChat.id, firstChat],
      [secondChat.id, secondChat],
    ]);
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
      ...storedChats.get(id)!,
      messages: [],
    }));
    vi.mocked(api.updateChat).mockImplementation(async (id, values) => {
      const updated = { ...storedChats.get(id)!, ...values };
      storedChats.set(id, updated);
      return updated;
    });
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

  it("persists each chat mode, role defaults, and preset binding without leakage", async () => {
    const stamp = "2026-07-22T00:00:00Z";
    const imagePreset = {
      id: "preset-image-studio",
      name: "Studio image",
      role: "image" as const,
      settings_json: { negative_prompt: "blur" },
      is_default: false,
    };
    const firstChat = {
      id: "chat-persisted-image",
      project_id: null,
      title: "Persisted image chat",
      archived: false,
      routing_mode: "text" as const,
      confirm_uncertain_media: false,
      active_chat_profile_id: null,
      active_image_profile_id: null,
      active_video_profile_id: null,
      active_head_message_id: null,
      generation_settings_json: {},
      generation_preset_ids_json: {},
      created_at: stamp,
      updated_at: stamp,
    };
    const secondChat = {
      ...firstChat,
      id: "chat-persisted-video",
      title: "Persisted video chat",
      routing_mode: "video" as const,
      generation_settings_json: { video: { frames: 81 } },
    };
    const storedChats = new Map<string, Chat>([
      [firstChat.id, firstChat],
      [secondChat.id, secondChat],
    ]);
    localStorage.setItem("local-lm-chat", firstChat.id);
    vi.mocked(api.engines).mockResolvedValue([roleAwareMediaEngine]);
    vi.mocked(api.presets).mockResolvedValue([imagePreset]);
    vi.mocked(api.chats).mockResolvedValue([firstChat, secondChat]);
    vi.mocked(api.chat).mockImplementation(async (id) => ({
      ...storedChats.get(id)!,
      messages: [],
    }));
    vi.mocked(api.updateChat).mockImplementation(async (id, values) => {
      const updated = { ...storedChats.get(id)!, ...values };
      storedChats.set(id, updated);
      return updated;
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>,
    );

    const mode = await screen.findByDisplayValue("Text");
    fireEvent.change(mode, { target: { value: "image" } });
    await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(
      firstChat.id,
      { routing_mode: "image" },
    ));
    fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
    fireEvent.change(screen.getByRole("combobox", { name: "image preset" }), {
      target: { value: imagePreset.id },
    });
    fireEvent.change(screen.getByLabelText(/Negative prompt/), {
      target: { value: "no fog" },
    });
    await waitFor(() => {
      expect(api.updateChat).toHaveBeenCalledWith(firstChat.id, {
        generation_preset_ids_json: { image: imagePreset.id },
      });
      expect(api.updateChat).toHaveBeenCalledWith(firstChat.id, {
        generation_settings_json: { image: { negative_prompt: "no fog" } },
      });
    });
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));

    fireEvent.click(screen.getByText(secondChat.title));
    await screen.findByRole("heading", { name: secondChat.title });
    expect(screen.getByDisplayValue("Video")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
    expect(screen.getByRole("spinbutton", { name: /Frames/ })).toHaveValue(81);
    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));

    fireEvent.click(screen.getByText(firstChat.title));
    await screen.findByRole("heading", { name: firstChat.title });
    expect(screen.getByDisplayValue("Image")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
    expect(screen.getByRole("combobox", { name: "image preset" })).toHaveValue(imagePreset.id);
    expect(screen.getByLabelText(/Negative prompt/)).toHaveValue("no fog");
  });
});
