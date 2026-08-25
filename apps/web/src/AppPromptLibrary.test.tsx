import { createHash } from "node:crypto";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import type { Chat, PromptBatch, PromptBatchCreateInput } from "./types";

vi.mock("./api", () => ({
  api: {
    setupReadiness: vi.fn(),
    projects: vi.fn(),
    chats: vi.fn(),
    chat: vi.fn(),
    workPlans: vi.fn(),
    engines: vi.fn(),
    profiles: vi.fn(),
    presets: vi.fn(),
    workflows: vi.fn(),
    workflowFamilies: vi.fn(),
    chatWorkflowSelections: vi.fn(),
    projectWorkflowSelections: vi.fn(),
    classifyDraft: vi.fn(),
    about: vi.fn(),
    jobs: vi.fn(),
    updateChat: vi.fn(),
    promptTemplates: vi.fn(),
    promptTemplate: vi.fn(),
    promptTemplateRevisions: vi.fn(),
    createPromptBatch: vi.fn(),
    queuePromptBatch: vi.fn(),
    sendTurn: vi.fn(),
  },
  connectEvents: vi.fn().mockResolvedValue(() => undefined),
}));

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  vi.mocked(api.setupReadiness).mockResolvedValue({ version: 2, state: "ready", roles: [] });
  vi.mocked(api.projects).mockResolvedValue([]);
  vi.mocked(api.workPlans).mockResolvedValue([]);
  vi.mocked(api.profiles).mockResolvedValue([]);
  vi.mocked(api.presets).mockResolvedValue([]);
  vi.mocked(api.workflows).mockResolvedValue([]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.classifyDraft).mockResolvedValue({ references_prior_visual: false });
  vi.mocked(api.engines).mockResolvedValue([{
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
  }]);
  vi.mocked(api.about).mockResolvedValue({
    max_media_outputs_per_plan: 8,
    version: "0.1.8",
    web_access_enabled: false,
    data_directory: "C:\\LM Atelier\\data",
    log_directory: "C:\\LM Atelier\\data\\logs",
  });
  vi.mocked(api.jobs).mockResolvedValue([]);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function promptDigest(prompt: string): string {
  return createHash("sha256")
    .update(`prompt-expansion-rendered-v1\0${prompt}`, "utf8")
    .digest("hex");
}

it("queues template prompts from chat without changing the composer or generation settings", async () => {
  const stamp = "2026-08-21T08:00:00Z";
  const chat: Chat = {
    id: "chat-prompt-library",
    project_id: null,
    title: "Prompt Library chat",
    pinned: false,
    archived: false,
    routing_mode: "image",
    confirm_uncertain_media: false,
    active_chat_profile_id: null,
    active_image_profile_id: null,
    active_video_profile_id: null,
    active_head_message_id: null,
    created_at: stamp,
    updated_at: stamp,
  };
  const definition = {
    id: "ptdef-one",
    name: "Portrait variants",
    description: "A controlled image prompt",
    archived: false,
    current_revision_id: "ptrev-one",
    created_at: stamp,
    updated_at: stamp,
  };
  const revision = {
    id: "ptrev-one",
    prompt_template_id: definition.id,
    version: 1,
    schema_version: 1,
    contract_json: {
      schema_version: 1 as const,
      operation: "text_to_image" as const,
      body: "A portrait of {{subject}}.",
      slots: [{ name: "subject", mode: "input" as const, variation_scope: "item" as const }],
      resource_policy: { mode: "inherited" as const },
    },
    contract_sha256: "a".repeat(64),
    created_at: stamp,
  };
  const detail = { ...definition, current_revision: revision };
  let createdBatch: PromptBatch | null = null;

  localStorage.setItem("local-lm-chat", chat.id);
  vi.mocked(api.chats).mockResolvedValue([chat]);
  vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
  vi.mocked(api.promptTemplates).mockResolvedValue({
    items: [definition],
    total: 1,
    limit: 100,
    offset: 0,
  });
  vi.mocked(api.promptTemplate).mockResolvedValue(detail);
  vi.mocked(api.promptTemplateRevisions).mockResolvedValue([revision]);
  vi.mocked(api.createPromptBatch).mockImplementation(
    async (_chatId, payload: PromptBatchCreateInput) => {
      const subjects = payload.inputs.subject as string[];
      createdBatch = {
        id: "prompt-batch-one",
        chat_id: chat.id,
        prompt_template_id: definition.id,
        prompt_template_revision_id: revision.id,
        schema_version: 1,
        contract_sha256: revision.contract_sha256,
        codec_version: 2,
        requested_count: payload.item_count,
        selection_seed: payload.selection_seed,
        plan_sha256: "d".repeat(64),
        state: "draft",
        plan_version: 1,
        queue_idempotency_key: null,
        work_plan_id: null,
        queued_at: null,
        replayed: false,
        items: subjects.map((subject, index) => {
          const prompt = `A portrait of ${subject}.`;
          return {
            id: `prompt-item-${index + 1}`,
            ordinal: index + 1,
            rendered_prompt: prompt,
            rendered_sha256: promptDigest(prompt),
            reviewed_prompt: prompt,
            reviewed_sha256: promptDigest(prompt),
            selected: true,
            review_version: 1,
            reroll_count: 0,
            work_step_id: null,
            run_id: null,
            media_seed: null,
          };
        }),
      };
      return createdBatch;
    },
  );
  vi.mocked(api.queuePromptBatch).mockImplementation(async (_batchId, payload) => ({
    ...createdBatch!,
    state: "queued",
    plan_version: 2,
    queue_idempotency_key: payload.idempotency_key,
    work_plan_id: "work-plan-one",
    queued_at: stamp,
    items: createdBatch!.items.map((item) => ({
      ...item,
      work_step_id: `step-${item.ordinal}`,
      run_id: `run-${item.ordinal}`,
      media_seed: 200 + item.ordinal,
    })),
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);

  expect(await screen.findByRole("heading", { name: chat.title })).toBeVisible();
  const composer = screen.getByRole("textbox", { name: "Message" });
  fireEvent.change(composer, { target: { value: "Keep this composer draft" } });
  const mode = screen.getByRole("combobox", { name: "Generation mode" });
  const outputCount = screen.getByRole("combobox", { name: "Number of outputs" });
  fireEvent.change(outputCount, { target: { value: "3" } });
  const workflow = screen.getByRole("combobox", { name: "Workflow for this request type" });
  const workflowValue = workflow.getAttribute("value") ?? (workflow as HTMLSelectElement).value;

  fireEvent.click(screen.getByRole("button", { name: "Open prompt templates" }));
  fireEvent.click(await screen.findByRole("button", { name: /Portrait variants/ }));
  fireEvent.change(await screen.findByLabelText("Number of prompts"), {
    target: { value: "2" },
  });
  fireEvent.change(screen.getByLabelText("subject for prompt 1"), {
    target: { value: "Ada" },
  });
  fireEvent.change(screen.getByLabelText("subject for prompt 2"), {
    target: { value: "Grace" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Create 2 prompts" }));

  expect(await screen.findByRole("heading", { name: "2 prompts queued" })).toBeVisible();
  expect(api.createPromptBatch).toHaveBeenCalledTimes(1);
  expect(api.queuePromptBatch).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Done" }));

  expect(composer).toHaveValue("Keep this composer draft");
  expect(mode).toHaveValue("image");
  expect(outputCount).toHaveValue("3");
  expect(workflow).toHaveValue(workflowValue);
  expect(screen.queryByText("Prompt Library draft linked")).toBeNull();
  expect(api.updateChat).not.toHaveBeenCalled();
  expect(api.sendTurn).not.toHaveBeenCalled();
  await waitFor(() => expect(client.getQueryData(["prompt-batch", "prompt-batch-one"]))
    .toMatchObject({ state: "queued", requested_count: 2 }));
});
