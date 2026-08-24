import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api";
import type { Chat } from "./types";

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
    promptBatch: vi.fn(),
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

afterEach(cleanup);

it("inserts a reviewed Prompt Library draft into the image composer with removable provenance", async () => {
  const stamp = "2026-08-21T08:00:00Z";
  const chat: Chat = {
    id: "chat-prompt-library",
    project_id: null,
    title: "Prompt Library chat",
    pinned: false,
    archived: false,
    routing_mode: "auto",
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
  const batch = {
    id: "prompt-batch-one",
    chat_id: chat.id,
    prompt_template_id: definition.id,
    prompt_template_revision_id: revision.id,
    schema_version: 1,
    contract_sha256: revision.contract_sha256,
    codec_version: 2 as const,
    requested_count: 2,
    selection_seed: 0,
    plan_sha256: "d".repeat(64),
    state: "draft" as const,
    plan_version: 1,
    queue_idempotency_key: null,
    work_plan_id: null,
    queued_at: null,
    replayed: false,
    items: [{
      id: "prompt-item-one",
      ordinal: 1,
      rendered_prompt: "A portrait of Ada.",
      rendered_sha256: "3ac91354251961158edffac465e3453adac752df1cce37b4e2e8a6a3ee2a392a",
      reviewed_prompt: "A portrait of Ada.",
      reviewed_sha256: "3ac91354251961158edffac465e3453adac752df1cce37b4e2e8a6a3ee2a392a",
      selected: true,
      review_version: 1,
      reroll_count: 0,
      work_step_id: null,
      run_id: null,
      media_seed: null,
    }, {
      id: "prompt-item-two",
      ordinal: 2,
      rendered_prompt: "A portrait of Grace.",
      rendered_sha256: "d8568c8470dfb1f3958ac624cad550beeec3bad35efcd844506708c4263be3cc",
      reviewed_prompt: "A portrait of Grace.",
      reviewed_sha256: "d8568c8470dfb1f3958ac624cad550beeec3bad35efcd844506708c4263be3cc",
      selected: true,
      review_version: 1,
      reroll_count: 0,
      work_step_id: null,
      run_id: null,
      media_seed: null,
    }],
  };
  localStorage.setItem("local-lm-chat", chat.id);
  vi.mocked(api.chats).mockResolvedValue([chat]);
  vi.mocked(api.chat).mockResolvedValue({ ...chat, messages: [] });
  vi.mocked(api.promptTemplates).mockResolvedValue({ items: [definition], total: 1, limit: 50, offset: 0 });
  vi.mocked(api.promptTemplate).mockResolvedValue(detail);
  vi.mocked(api.promptTemplateRevisions).mockResolvedValue([revision]);
  vi.mocked(api.createPromptBatch).mockResolvedValue(batch);
  vi.mocked(api.promptBatch).mockResolvedValue({ ...batch, replayed: true });
  vi.mocked(api.updateChat).mockResolvedValue({ ...chat, routing_mode: "image" });
  vi.mocked(api.sendTurn).mockRejectedValue(new Error("Prompt Library source changed. Refresh and try again."));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);

  fireEvent.click(await screen.findByRole("button", { name: "Prompt library" }));
  await screen.findByRole("heading", { name: "Portrait variants" });
  fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
  fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
  fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
  fireEvent.click(screen.getByRole("button", { name: "Generate drafts" }));
  await screen.findByRole("heading", { name: "Review drafts" });
  const firstCard = screen.getByLabelText("Reviewed prompt for draft 1").closest("article")!;
  fireEvent.click(within(firstCard).getByRole("button", { name: "Insert into composer" }));

  expect(await screen.findByRole("heading", { name: chat.title })).toBeInTheDocument();
  const composer = screen.getByRole("textbox", { name: "Message" });
  expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("A portrait of Ada.");
  expect(screen.getByRole("combobox", { name: "Generation mode" })).toHaveValue("image");
  expect(screen.getByText("Prompt Library draft linked")).toBeVisible();
  await waitFor(() => expect(api.updateChat).toHaveBeenCalledWith(chat.id, { routing_mode: "image" }));

  fireEvent.change(screen.getByRole("combobox", { name: "Number of outputs" }), {
    target: { value: "2" },
  });
  expect(screen.queryByText("Prompt Library draft linked")).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "Number of outputs" }), {
    target: { value: "1" },
  });

  fireEvent.click(screen.getByRole("button", { name: "Prompt library" }));
  await screen.findByRole("heading", { name: "Portrait variants" });
  fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
  fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
  fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
  fireEvent.click(screen.getByRole("button", { name: "Generate drafts" }));
  await screen.findByRole("heading", { name: "Review drafts" });
  fireEvent.click(within(
    screen.getByLabelText("Reviewed prompt for draft 1").closest("article")!,
  ).getByRole("button", { name: "Insert into composer" }));

  fireEvent.change(composer, { target: { value: "A warmer portrait of Ada." } });
  expect(screen.getByText("Prompt Library draft linked")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Remove Prompt Library source" }));
  expect(screen.queryByText("Prompt Library draft linked")).not.toBeInTheDocument();
  expect(composer).toHaveValue("A warmer portrait of Ada.");

  fireEvent.click(screen.getByRole("button", { name: "Prompt library" }));
  await screen.findByRole("heading", { name: "Portrait variants" });
  fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
  fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
  fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
  fireEvent.click(screen.getByRole("button", { name: "Generate drafts" }));
  await screen.findByRole("heading", { name: "Review drafts" });
  fireEvent.click(within(
    screen.getByLabelText("Reviewed prompt for draft 1").closest("article")!,
  ).getByRole("button", { name: "Insert into composer" }));
  fireEvent.click(await screen.findByRole("button", { name: "Send" }));

  await waitFor(() => expect(api.sendTurn).toHaveBeenCalledWith(
    chat.id, "A portrait of Ada.", "image", [], {}, expect.any(String),
    "turns", undefined, [], undefined, {
      version: 1,
      batch_id: batch.id,
      expected_plan_version: batch.plan_version,
      expected_plan_sha256: batch.plan_sha256,
      item_id: batch.items[0].id,
      expected_review_version: batch.items[0].review_version,
      expected_reviewed_sha256: batch.items[0].reviewed_sha256,
      prompt_template_id: batch.prompt_template_id,
      prompt_template_revision_id: batch.prompt_template_revision_id,
      contract_sha256: batch.contract_sha256,
    },
  ));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Prompt Library source changed. Refresh and try again.",
  );
  expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("A portrait of Ada.");
  expect(screen.getByText("Prompt Library draft linked")).toBeVisible();
});
