import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createHash } from "node:crypto";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api } from "./api";
import { PromptLibraryView } from "./PromptLibraryView";
import type {
  PromptTemplateContract,
  PromptBatch,
  ChatDetail,
  PromptTemplateDefinition,
  PromptTemplateDetail,
  PromptTemplateRevision,
  PromptTemplateWriteResult,
} from "./types";

vi.mock("./api", () => ({
  api: {
    promptTemplates: vi.fn(),
    promptTemplate: vi.fn(),
    createPromptTemplate: vi.fn(),
    updatePromptTemplate: vi.fn(),
    promptTemplateRevisions: vi.fn(),
    restorePromptTemplateRevision: vi.fn(),
    chat: vi.fn(),
    createPromptBatch: vi.fn(),
    promptBatch: vi.fn(),
    updatePromptBatchItem: vi.fn(),
    queuePromptBatch: vi.fn(),
  },
}));

const stamp = "2026-08-20T12:00:00Z";
function promptDigest(prompt: string): string {
  return createHash("sha256")
    .update(`prompt-expansion-rendered-v1\0${prompt}`, "utf8")
    .digest("hex");
}
function planDigest(items: PromptBatch["items"]): string {
  return createHash("sha256")
    .update(JSON.stringify(items.map((item) => ({
      ordinal: item.ordinal,
      rendered_sha256: item.reviewed_sha256,
    }))), "utf8")
    .digest("hex");
}
const contract: PromptTemplateContract = {
  schema_version: 1,
  operation: "text_to_image",
  body: "A portrait of {{subject}}.",
  slots: [{ name: "subject", mode: "input", variation_scope: "item" }],
  resource_policy: { mode: "inherited" },
};
const definition: PromptTemplateDefinition = {
  id: "ptdef_one",
  name: "Portrait variants",
  description: "One controlled subject slot",
  archived: false,
  current_revision_id: "ptrev_two",
  created_at: stamp,
  updated_at: stamp,
};
const currentRevision: PromptTemplateRevision = {
  id: "ptrev_two",
  prompt_template_id: definition.id,
  version: 2,
  schema_version: 1,
  contract_json: contract,
  contract_sha256: "b".repeat(64),
  created_at: stamp,
};
const previousRevision: PromptTemplateRevision = {
  ...currentRevision,
  id: "ptrev_one",
  version: 1,
  contract_sha256: "a".repeat(64),
};
const detail: PromptTemplateDetail = { ...definition, current_revision: currentRevision };
const writeResult: PromptTemplateWriteResult = {
  template: detail,
  revision: currentRevision,
  idempotent: false,
};
const chat: ChatDetail = {
  id: "chat-one",
  project_id: null,
  title: "Active chat",
  archived: false,
  pinned: false,
  routing_mode: "auto",
  confirm_uncertain_media: true,
  active_chat_profile_id: null,
  active_image_profile_id: null,
  active_video_profile_id: null,
  active_head_message_id: null,
  created_at: stamp,
  updated_at: stamp,
  messages: [],
};
const batchItems: PromptBatch["items"] = [1, 2].map((ordinal) => {
  const prompt = `A portrait of subject ${ordinal}.`;
  return {
    id: `prompt-item-${ordinal}`,
    ordinal,
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
});
const batch: PromptBatch = {
  id: "prompt-batch-one",
  chat_id: chat.id,
  prompt_template_id: definition.id,
  prompt_template_revision_id: currentRevision.id,
  schema_version: currentRevision.schema_version,
  contract_sha256: currentRevision.contract_sha256,
  codec_version: 2,
  requested_count: 2,
  selection_seed: 0,
  plan_sha256: planDigest(batchItems),
  state: "draft",
  plan_version: 1,
  queue_idempotency_key: null,
  work_plan_id: null,
  queued_at: null,
  replayed: false,
  items: batchItems,
};
let latestBatch: PromptBatch;

function renderLibrary(
  activeChatId: string | null = chat.id,
  onInsertIntoComposer?: Parameters<typeof PromptLibraryView>[0]["onInsertIntoComposer"],
) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><PromptLibraryView activeChatId={activeChatId} onInsertIntoComposer={onInsertIntoComposer} /></QueryClientProvider>);
  return client;
}

async function openAndSubmitExpansion(): Promise<void> {
  await screen.findByRole("heading", { name: "Portrait variants" });
  fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
  fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
  fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
  fireEvent.click(screen.getByRole("button", { name: "Generate drafts" }));
}

beforeEach(() => {
  latestBatch = structuredClone(batch);
  vi.mocked(api.promptTemplates).mockResolvedValue({
    items: [definition],
    total: 1,
    limit: 100,
    offset: 0,
  });
  vi.mocked(api.promptTemplate).mockResolvedValue(detail);
  vi.mocked(api.promptTemplateRevisions).mockResolvedValue([currentRevision, previousRevision]);
  vi.mocked(api.createPromptTemplate).mockResolvedValue(writeResult);
  vi.mocked(api.updatePromptTemplate).mockResolvedValue(writeResult);
  vi.mocked(api.restorePromptTemplateRevision).mockResolvedValue(writeResult);
  vi.mocked(api.chat).mockResolvedValue(chat);
  vi.mocked(api.createPromptBatch).mockImplementation(async () => structuredClone(latestBatch));
  vi.mocked(api.promptBatch).mockImplementation(async () => structuredClone({
    ...latestBatch,
    replayed: true,
  }));
  vi.mocked(api.updatePromptBatchItem).mockImplementation(async (_batchId, ordinal, payload) => {
    const before = latestBatch.items.find((item) => item.ordinal === ordinal)!;
    const promptChanged = before.reviewed_prompt !== payload.reviewed_prompt;
    const nextItems = latestBatch.items.map((item) => item.ordinal === ordinal
      ? {
          ...item,
          reviewed_prompt: payload.reviewed_prompt,
          reviewed_sha256: promptDigest(payload.reviewed_prompt),
          selected: payload.selected,
          review_version: payload.expected_review_version + 1,
        }
      : item);
    latestBatch = {
      ...latestBatch,
      plan_sha256: promptChanged ? planDigest(nextItems) : latestBatch.plan_sha256,
      plan_version: payload.expected_plan_version + 1,
      items: nextItems,
      replayed: false,
    };
    return structuredClone(latestBatch);
  });
  vi.mocked(api.queuePromptBatch).mockImplementation(async (_batchId, payload) => {
    latestBatch = {
      ...latestBatch,
      state: "queued",
      plan_version: latestBatch.plan_version + 1,
      queue_idempotency_key: payload.idempotency_key,
      work_plan_id: "work-plan-prompt-batch",
      queued_at: stamp,
      replayed: false,
      items: latestBatch.items.map((item) => item.selected
        ? {
            ...item,
            work_step_id: `work-step-${item.ordinal}`,
            run_id: `run-prompt-${item.ordinal}`,
            media_seed: 100 + item.ordinal,
          }
        : item),
    };
    return structuredClone(latestBatch);
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Prompt Library Phase 1", () => {
  it("shows the immutable current revision, structured slots, resources, and history", async () => {
    renderLibrary();

    expect(await screen.findByRole("heading", { name: "Portrait variants" })).toBeVisible();
    expect(screen.getByText("A portrait of {{subject}}.")).toBeVisible();
    expect(screen.getByText("{{subject}}")).toBeVisible();
    expect(screen.getByText("Inherited image resources")).toBeVisible();
    expect(screen.getByText("Revision 2")).toBeVisible();
    expect(screen.getByText("Revision 1")).toBeVisible();
  });

  it("creates a new template instead of updating the selected template", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });

    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Template name"), { target: { value: "Mood board" } });
    fireEvent.change(screen.getByLabelText("Slot 1 mode"), { target: { value: "choice" } });
    fireEvent.change(screen.getByLabelText("Slot 1 choices"), { target: { value: "warm\ncool" } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    await waitFor(() => expect(api.createPromptTemplate).toHaveBeenCalledTimes(1));
    expect(api.updatePromptTemplate).not.toHaveBeenCalled();
    expect(api.createPromptTemplate).toHaveBeenCalledWith(expect.objectContaining({
      idempotency_key: expect.any(String),
      name: "Mood board",
      contract: expect.objectContaining({
        slots: [{ name: "subject", mode: "choice", variation_scope: "item", choices: ["warm", "cool"] }],
      }),
    }));
  });

  it("saves edits with optimistic revision authority and archives without minting a body", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });

    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByLabelText("Template body"), { target: { value: "Detailed {{subject}}." } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));
    await waitFor(() => expect(api.updatePromptTemplate).toHaveBeenCalledWith(
      definition.id,
      expect.objectContaining({
        expected_current_revision_id: currentRevision.id,
        idempotency_key: expect.any(String),
        contract: expect.objectContaining({ body: "Detailed {{subject}}." }),
      }),
    ));

    fireEvent.click(screen.getByRole("button", { name: "Archive" }));
    await waitFor(() => expect(api.updatePromptTemplate).toHaveBeenCalledWith(definition.id, {
      expected_current_revision_id: currentRevision.id,
      archived: true,
    }));
  });

  it("keeps edit authority bound to the template that opened the editor", async () => {
    const otherDefinition = {
      ...definition,
      id: "ptdef_other",
      name: "Other template",
      current_revision_id: "ptrev_other",
    };
    const otherRevision = {
      ...currentRevision,
      id: otherDefinition.current_revision_id,
      prompt_template_id: otherDefinition.id,
    };
    vi.mocked(api.promptTemplates).mockResolvedValue({
      items: [definition, otherDefinition],
      total: 2,
      limit: 50,
      offset: 0,
    });
    vi.mocked(api.promptTemplate).mockImplementation(async (id) => id === otherDefinition.id
      ? { ...otherDefinition, current_revision: otherRevision }
      : detail);
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });

    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.click(screen.getByRole("button", { name: /Other template/ }));
    await screen.findByRole("heading", { name: "Other template" });
    fireEvent.change(screen.getByLabelText("Template body"), { target: { value: "Still {{subject}}." } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    await waitFor(() => expect(api.updatePromptTemplate).toHaveBeenCalledWith(
      definition.id,
      expect.objectContaining({
        expected_current_revision_id: currentRevision.id,
        contract: expect.objectContaining({ body: "Still {{subject}}." }),
      }),
    ));
    expect(api.createPromptTemplate).not.toHaveBeenCalled();
  });

  it("authors fixed slots and a verified fixed resource policy without paths or mutable names", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Slot 1 mode"), { target: { value: "fixed" } });
    fireEvent.change(screen.getByLabelText("Slot 1 fixed value"), { target: { value: "studio portrait" } });
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "fixed" } });
    fireEvent.change(screen.getByLabelText("Workflow revision ID"), { target: { value: "workflow-revision-1" } });
    fireEvent.change(screen.getByLabelText("LoRA policy"), { target: { value: "fixed" } });
    fireEvent.change(screen.getByLabelText("LoRA 1 SHA-256"), { target: { value: "c".repeat(64) } });
    fireEvent.change(screen.getByLabelText("LoRA 1 model strength"), { target: { value: "0.8" } });
    fireEvent.change(screen.getByLabelText("LoRA 1 CLIP strength"), { target: { value: "0.7" } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    await waitFor(() => expect(api.createPromptTemplate).toHaveBeenCalledWith(expect.objectContaining({
      contract: expect.objectContaining({
        slots: [{ name: "subject", mode: "fixed", variation_scope: "batch", fixed_value: "studio portrait" }],
        resource_policy: {
          mode: "fixed",
          workflow_revision_id: "workflow-revision-1",
          lora_policy: {
            mode: "fixed",
            stack: [{ sha256: "c".repeat(64), model_strength: 0.8, clip_strength: 0.7 }],
          },
        },
      }),
    })));
  });

  it("restores an old revision by appending through the reviewed restore endpoint", async () => {
    renderLibrary();
    await screen.findByText("Revision 1");

    fireEvent.click(screen.getByRole("button", { name: "Restore" }));
    await waitFor(() => expect(api.restorePromptTemplateRevision).toHaveBeenCalledWith(
      definition.id,
      previousRevision.id,
      currentRevision.id,
      expect.any(String),
    ));
  });

  it("refuses a body/slot mismatch before any write", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });

    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Template body"), { target: { value: "No declared token" } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    expect(await screen.findByText("Place every declared slot exactly once, in the same order as the slot list.")).toBeVisible();
    expect(api.createPromptTemplate).not.toHaveBeenCalled();
  });

  it("reports server failures without echoing private backend text", async () => {
    vi.mocked(api.promptTemplates).mockRejectedValue(new Error("C:\\private\\template-body.txt"));
    renderLibrary();

    expect(await screen.findByText("The Prompt Library could not complete that request. Refresh and try again.")).toBeVisible();
    expect(screen.queryByText(/private\\template-body/)).toBeNull();
  });

  it("reports a save failure inside the open editor without echoing backend text", async () => {
    vi.mocked(api.updatePromptTemplate).mockRejectedValue(new Error("C:\\private\\saved-template.json"));
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    const dialog = screen.getByRole("dialog", { name: "Edit Portrait variants" });
    expect(await within(dialog).findByText("The Prompt Library could not save that revision. Review the template and try again.")).toBeVisible();
    expect(screen.queryByText(/private\\saved-template/)).toBeNull();
  });

  it("pages through a bounded template list", async () => {
    vi.mocked(api.promptTemplates)
      .mockResolvedValueOnce({ items: [definition], total: 51, limit: 50, offset: 0 })
      .mockResolvedValueOnce({ items: [{ ...definition, id: "ptdef_last", name: "Last template" }], total: 51, limit: 50, offset: 50 });
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });

    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(api.promptTemplates).toHaveBeenLastCalledWith(false, 50, 50));
  });

  it("keeps the previous-page recovery control when the last page becomes empty", async () => {
    vi.mocked(api.promptTemplates)
      .mockResolvedValueOnce({ items: [definition], total: 51, limit: 50, offset: 0 })
      .mockResolvedValueOnce({ items: [], total: 50, limit: 50, offset: 50 });
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByRole("button", { name: "Previous" })).toBeEnabled();
    expect(screen.getByText("No items of 50")).toBeVisible();
  });

  it("generates two no-media drafts with exact authored input scope and one idempotent submit", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });

    fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
    fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
    fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
    const generate = screen.getByRole("button", { name: "Generate drafts" });
    fireEvent.click(generate);
    fireEvent.click(generate);

    const reviewHeading = await screen.findByRole("heading", { name: "Review drafts" });
    expect(reviewHeading).toBeVisible();
    await waitFor(() => expect(reviewHeading).toHaveFocus());
    expect(screen.getByRole("status")).toHaveTextContent("2 drafts ready to review.");
    expect(screen.getAllByRole("textbox", { name: /Reviewed prompt for draft/ })).toHaveLength(2);
    expect(api.createPromptBatch).toHaveBeenCalledTimes(1);
    expect(api.createPromptBatch).toHaveBeenCalledWith(chat.id, {
      idempotency_key: expect.any(String),
      template_revision_id: currentRevision.id,
      contract_sha256: currentRevision.contract_sha256,
      item_count: 2,
      selection_seed: 0,
      inputs: { subject: ["Ada", "Grace"] },
    });
    expect(api.createPromptTemplate).not.toHaveBeenCalled();
  });

  it("inserts exactly one selected saved review with its admitted authority", async () => {
    const insert = vi.fn();
    renderLibrary(chat.id, insert);
    await openAndSubmitExpansion();
    await screen.findByRole("heading", { name: "Review drafts" });

    const firstCard = screen.getByLabelText("Reviewed prompt for draft 1").closest("article")!;
    fireEvent.click(within(firstCard).getByRole("button", { name: "Insert into composer" }));

    expect(insert).toHaveBeenCalledTimes(1);
    expect(insert).toHaveBeenCalledWith(chat.id, {
      text: batch.items[0].reviewed_prompt,
      source: {
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
    });
    expect(screen.queryByRole("dialog", { name: "Test Portrait variants" })).toBeNull();
  });

  it("does not insert an unsaved or deselected review", async () => {
    renderLibrary();
    await openAndSubmitExpansion();
    await screen.findByRole("heading", { name: "Review drafts" });
    const firstCard = screen.getByLabelText("Reviewed prompt for draft 1").closest("article")!;
    const insert = within(firstCard).getByRole("button", { name: "Insert into composer" });

    fireEvent.change(screen.getByLabelText("Reviewed prompt for draft 1"), {
      target: { value: "Unsaved edit" },
    });
    expect(insert).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Reviewed prompt for draft 1"), {
      target: { value: batch.items[0].reviewed_prompt },
    });
    fireEvent.click(within(firstCard).getByRole("checkbox", { name: "Selected" }));
    expect(insert).toBeDisabled();
  });

  it("queues every saved selected draft as one exact work plan", async () => {
    const client = renderLibrary();
    await openAndSubmitExpansion();
    await screen.findByRole("heading", { name: "Review drafts" });

    const queue = screen.getByRole("button", { name: "Queue selected drafts" });
    expect(queue).toBeEnabled();
    fireEvent.click(queue);

    await waitFor(() => expect(api.queuePromptBatch).toHaveBeenCalledWith(
      batch.id,
      {
        idempotency_key: expect.any(String),
        expected_plan_version: batch.plan_version,
        expected_plan_sha256: batch.plan_sha256,
      },
    ));
    expect(await screen.findByText("2 selected drafts were queued as one work plan."))
      .toBeVisible();
    expect(screen.queryByRole("button", { name: "Queue selected drafts" })).toBeNull();
    expect(client.getQueryData<PromptBatch>(["prompt-batch", batch.id])).toMatchObject({
      state: "queued",
      work_plan_id: "work-plan-prompt-batch",
      items: [
        { work_step_id: "work-step-1", run_id: "run-prompt-1", media_seed: 101 },
        { work_step_id: "work-step-2", run_id: "run-prompt-2", media_seed: 102 },
      ],
    });
  });

  it("refuses to queue while any review card has unsaved changes", async () => {
    renderLibrary();
    await openAndSubmitExpansion();
    await screen.findByRole("heading", { name: "Review drafts" });

    const queue = screen.getByRole("button", { name: "Queue selected drafts" });
    fireEvent.change(screen.getByLabelText("Reviewed prompt for draft 2"), {
      target: { value: "Unsaved private review" },
    });
    await waitFor(() => expect(queue).toBeDisabled());
    fireEvent.click(queue);
    expect(api.queuePromptBatch).not.toHaveBeenCalled();
  });

  it("collects one batch input and leaves model guidance read-only", async () => {
    const scopedDetail: PromptTemplateDetail = {
      ...detail,
      current_revision: {
        ...currentRevision,
        contract_sha256: "f".repeat(64),
        contract_json: {
          ...contract,
          body: "{{subject}} in {{style}} with {{lighting}}.",
          slots: [
            { name: "subject", mode: "input", variation_scope: "item" },
            { name: "style", mode: "input", variation_scope: "batch" },
            {
              name: "lighting",
              mode: "model",
              variation_scope: "batch",
              guidance: "Choose soft, motivated light.",
            },
          ],
        },
      },
    };
    const scopedBatch = {
      ...batch,
      contract_sha256: scopedDetail.current_revision.contract_sha256,
    };
    vi.mocked(api.promptTemplate).mockResolvedValue(scopedDetail);
    vi.mocked(api.createPromptBatch).mockResolvedValue(scopedBatch);
    vi.mocked(api.promptBatch).mockResolvedValue({ ...scopedBatch, replayed: true });
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));

    const dialog = screen.getByRole("dialog", { name: "Test Portrait variants" });
    expect(within(dialog).getByText("Choose soft, motivated light.")).toBeVisible();
    expect(within(dialog).queryByRole("textbox", { name: /lighting/i })).toBeNull();
    fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
    fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
    fireEvent.change(screen.getByLabelText("style shared across batch"), { target: { value: "Noir" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate drafts" }));

    await waitFor(() => expect(api.createPromptBatch).toHaveBeenCalledWith(
      chat.id,
      expect.objectContaining({
        template_revision_id: currentRevision.id,
        contract_sha256: "f".repeat(64),
        inputs: {
          subject: ["Ada", "Grace"],
          style: "Noir",
        },
      }),
    ));
  });

  it("edits and deselects a draft through optimistic review-version authority", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
    fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
    fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate drafts" }));
    await screen.findByRole("heading", { name: "Review drafts" });
    expect(await screen.findByText(/replayed safely/)).toBeVisible();
    const originalPlanDigest = batch.plan_sha256;

    fireEvent.change(screen.getByLabelText("Reviewed prompt for draft 1"), {
      target: { value: "A reviewed portrait of Ada." },
    });
    const firstCard = screen.getByLabelText("Reviewed prompt for draft 1").closest("article")!;
    fireEvent.click(within(firstCard).getByRole("checkbox", { name: "Selected" }));
    const saveReview = within(firstCard).getByRole("button", { name: "Save review" });
    saveReview.focus();
    fireEvent.click(saveReview);

    await waitFor(() => expect(api.updatePromptBatchItem).toHaveBeenCalledWith(
      batch.id,
      1,
      {
        expected_review_version: 1,
        expected_plan_version: 1,
        reviewed_prompt: "A reviewed portrait of Ada.",
        selected: false,
      },
    ));
    await within(firstCard).findByText("Review saved");
    await waitFor(() => expect(saveReview).toHaveFocus());
    expect(latestBatch.plan_sha256).not.toBe(originalPlanDigest);
    expect(latestBatch.replayed).toBe(false);
    expect(screen.queryByText(/replayed safely/)).toBeNull();
  });

  it("keeps the plan digest stable for a selection-only review", async () => {
    renderLibrary();
    await openAndSubmitExpansion();
    await screen.findByText(/replayed safely/);
    const originalPlanDigest = batch.plan_sha256;
    const firstCard = screen.getByLabelText("Reviewed prompt for draft 1").closest("article")!;
    fireEvent.click(within(firstCard).getByRole("checkbox", { name: "Selected" }));
    fireEvent.click(within(firstCard).getByRole("button", { name: "Save review" }));

    await within(firstCard).findByText("Review saved");
    expect(latestBatch.plan_version).toBe(2);
    expect(latestBatch.plan_sha256).toBe(originalPlanDigest);
    expect(latestBatch.items[0].selected).toBe(false);
  });

  it("accepts an exact POST replay of an already edited batch", async () => {
    const reviewedPrompt = "A previously reviewed portrait.";
    const replayedItems = batch.items.map((item) => item.ordinal === 1
      ? {
          ...item,
          reviewed_prompt: reviewedPrompt,
          reviewed_sha256: promptDigest(reviewedPrompt),
          review_version: 2,
        }
      : item);
    const replayedBatch: PromptBatch = {
      ...batch,
      plan_sha256: planDigest(replayedItems),
      plan_version: 2,
      items: replayedItems,
      replayed: true,
    };
    vi.mocked(api.createPromptBatch).mockResolvedValue(replayedBatch);
    vi.mocked(api.promptBatch).mockResolvedValue(replayedBatch);
    renderLibrary();
    await openAndSubmitExpansion();

    expect(await screen.findByText(/replayed safely/)).toBeVisible();
    expect(screen.getByLabelText("Reviewed prompt for draft 1")).toHaveValue(reviewedPrompt);
    expect(screen.getByText(new RegExp(`Plan ${replayedBatch.plan_sha256.slice(0, 12)}`))).toBeVisible();
  });

  it("carries whole-batch plan authority across sequential sibling edits", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
    fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
    fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate drafts" }));
    await screen.findByRole("heading", { name: "Review drafts" });

    fireEvent.change(screen.getByLabelText("Reviewed prompt for draft 1"), {
      target: { value: "First review." },
    });
    fireEvent.click(screen.getByLabelText("Reviewed prompt for draft 1")
      .closest("article")!
      .querySelector<HTMLButtonElement>("button.secondary")!);
    await waitFor(() => expect(api.updatePromptBatchItem).toHaveBeenCalledTimes(1));
    await within(screen.getByLabelText("Reviewed prompt for draft 1").closest("article")!)
      .findByText("Review saved");

    fireEvent.change(screen.getByLabelText("Reviewed prompt for draft 2"), {
      target: { value: "Second review." },
    });
    const secondCard = screen.getByLabelText("Reviewed prompt for draft 2").closest("article")!;
    fireEvent.click(within(secondCard).getByRole("button", { name: "Save review" }));

    await waitFor(() => expect(api.updatePromptBatchItem).toHaveBeenNthCalledWith(
      2,
      batch.id,
      2,
      {
        expected_review_version: 1,
        expected_plan_version: 2,
        reviewed_prompt: "Second review.",
        selected: true,
      },
    ));
    expect(latestBatch.plan_version).toBe(3);
    expect(latestBatch.items.map((item) => item.review_version)).toEqual([2, 2]);
  });

  it("does not let a deferred version-one GET roll back a version-two review", async () => {
    let resolveRead!: (value: unknown) => void;
    const deferredRead = new Promise<unknown>((resolve) => {
      resolveRead = resolve;
    });
    vi.mocked(api.promptBatch).mockReturnValue(deferredRead);
    renderLibrary();
    await openAndSubmitExpansion();
    await screen.findByRole("heading", { name: "Review drafts" });

    fireEvent.change(screen.getByLabelText("Reviewed prompt for draft 1"), {
      target: { value: "Review saved before the read returns." },
    });
    const firstCard = screen.getByLabelText("Reviewed prompt for draft 1").closest("article")!;
    fireEvent.click(within(firstCard).getByRole("button", { name: "Save review" }));
    await within(firstCard).findByText("Review saved");
    expect(latestBatch.plan_version).toBe(2);

    await act(async () => {
      resolveRead({ ...batch, replayed: true });
      await deferredRead;
    });
    await screen.findByText(/replayed safely/);

    fireEvent.change(screen.getByLabelText("Reviewed prompt for draft 2"), {
      target: { value: "Sibling review after the late read." },
    });
    const secondCard = screen.getByLabelText("Reviewed prompt for draft 2").closest("article")!;
    fireEvent.click(within(secondCard).getByRole("button", { name: "Save review" }));

    await waitFor(() => expect(api.updatePromptBatchItem).toHaveBeenNthCalledWith(
      2,
      batch.id,
      2,
      expect.objectContaining({ expected_plan_version: 2 }),
    ));
    expect(latestBatch.plan_version).toBe(3);
  });

  it("rejects a conflicting same-version GET instead of replacing admitted data", async () => {
    vi.mocked(api.promptBatch).mockResolvedValue({
      ...batch,
      replayed: true,
      items: [{ ...batch.items[0], selected: false }, batch.items[1]],
    });
    renderLibrary();
    await openAndSubmitExpansion();

    expect(await screen.findByText(
      "The saved draft batch could not be loaded. Refresh and try again.",
    )).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Review drafts" })).toBeNull();
  });

  it("fails closed without an active chat and for an archived template", async () => {
    const { unmount } = render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <PromptLibraryView activeChatId={null} />
      </QueryClientProvider>,
    );
    await screen.findByRole("heading", { name: "Portrait variants" });
    expect(screen.getByRole("button", { name: "Test expansion" })).toBeDisabled();
    expect(screen.getByText("Choose or create an active chat to test this template.")).toBeVisible();
    unmount();

    const archived = { ...detail, archived: true };
    vi.mocked(api.promptTemplates).mockResolvedValue({
      items: [archived],
      total: 1,
      limit: 50,
      offset: 0,
    });
    vi.mocked(api.promptTemplate).mockResolvedValue(archived);
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    expect(screen.getByRole("button", { name: "Test expansion" })).toBeDisabled();
    expect(screen.getByText("Archived templates cannot create test drafts.")).toBeVisible();
  });

  it("invalidates stale revision authority while the dialog is open", async () => {
    const client = renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
    expect(screen.getByRole("button", { name: "Generate drafts" })).toBeEnabled();

    act(() => client.setQueryData(["prompt-template", definition.id], {
      ...detail,
      current_revision_id: "ptrev_three",
      current_revision: {
        ...currentRevision,
        id: "ptrev_three",
        version: 3,
        contract_sha256: "e".repeat(64),
      },
    }));

    expect(await screen.findByText("This template or chat changed while the dialog was open. Close and reopen it.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Generate drafts" })).toBeDisabled();
    expect(api.createPromptBatch).not.toHaveBeenCalled();
  });

  it("shows stable expansion failures without echoing backend text", async () => {
    vi.mocked(api.createPromptBatch).mockRejectedValue(new Error("C:\\private\\prompt-inputs.json"));
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
    fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
    fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate drafts" }));

    expect(await screen.findByText("The Prompt Library could not create those drafts. Refresh the template and try again.")).toBeVisible();
    expect(screen.queryByText(/private.*prompt-inputs/)).toBeNull();
  });

  it("refuses a generated batch whose readback authority does not match", async () => {
    vi.mocked(api.createPromptBatch).mockResolvedValue({
      ...batch,
      prompt_template_revision_id: "ptrev_stale",
      contract_sha256: "e".repeat(64),
    });
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Test expansion" }));
    fireEvent.change(screen.getByLabelText("subject for draft 1"), { target: { value: "Ada" } });
    fireEvent.change(screen.getByLabelText("subject for draft 2"), { target: { value: "Grace" } });
    fireEvent.click(screen.getByRole("button", { name: "Generate drafts" }));

    expect(await screen.findByText("The generated batch did not match the selected template revision. Refresh before trying again.")).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Review drafts" })).toBeNull();
    expect(api.promptBatch).not.toHaveBeenCalled();
  });

  it.each([
    ["null", () => null],
    ["an unknown property", () => ({ ...batch, unexpected: "not admitted" })],
    ["a queued state", () => ({ ...batch, state: "queued" })],
    ["a non-draft state", () => ({ ...batch, state: "review" })],
    ["a non-replayed later plan", () => ({ ...batch, plan_version: 2 })],
    ["a zero-based ordinal", () => ({
      ...batch,
      items: [{ ...batch.items[0], ordinal: 0 }, batch.items[1]],
    })],
    ["an oversized item array", () => ({
      ...batch,
      items: Array.from({ length: 17 }, (_, index) => ({
        ...batch.items[0],
        id: `prompt-item-${index + 1}`,
        ordinal: index + 1,
      })),
    })],
    ["an oversized prompt", () => {
      const renderedPrompt = "x".repeat(32_001);
      return {
        ...batch,
        items: [{
          ...batch.items[0],
          rendered_prompt: renderedPrompt,
          rendered_sha256: promptDigest(renderedPrompt),
        }, batch.items[1]],
      };
    }],
    ["a tampered reviewed digest", () => ({
      ...batch,
      items: [{ ...batch.items[0], reviewed_prompt: "Tampered prompt." }, batch.items[1]],
    })],
  ])("rejects %s before rendering a generated readback", async (_label, hostileResponse) => {
    vi.mocked(api.createPromptBatch).mockResolvedValue(hostileResponse());
    renderLibrary();
    await openAndSubmitExpansion();

    expect(await screen.findByText(
      "The generated batch did not match the selected template revision. Refresh before trying again.",
    )).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Review drafts" })).toBeNull();
    expect(api.promptBatch).not.toHaveBeenCalled();
  });

  it("rejects a tampered saved-batch GET readback without retaining it", async () => {
    vi.mocked(api.promptBatch).mockResolvedValue({
      ...batch,
      replayed: true,
      items: [{ ...batch.items[0], rendered_sha256: "0".repeat(64) }, batch.items[1]],
    });
    renderLibrary();
    await openAndSubmitExpansion();

    expect(await screen.findByText(
      "The saved draft batch could not be loaded. Refresh and try again.",
    )).toBeVisible();
    expect(screen.queryByRole("heading", { name: "Review drafts" })).toBeNull();
  });

  it("rejects a hostile full-batch PATCH response and keeps the admitted batch", async () => {
    vi.mocked(api.updatePromptBatchItem).mockImplementation(async (_batchId, _ordinal, payload) => ({
      ...latestBatch,
      chat_id: "hostile-chat",
      plan_version: payload.expected_plan_version + 1,
    }));
    renderLibrary();
    await openAndSubmitExpansion();
    await screen.findByRole("heading", { name: "Review drafts" });
    const textarea = screen.getByLabelText("Reviewed prompt for draft 1");
    fireEvent.change(textarea, { target: { value: "Safe local review." } });
    const firstCard = textarea.closest("article")!;
    fireEvent.click(within(firstCard).getByRole("button", { name: "Save review" }));

    expect(await within(firstCard).findByText(
      "This draft could not be saved. Reload the batch before trying again.",
    )).toBeVisible();
    expect(screen.getByLabelText("Reviewed prompt for draft 2")).toHaveValue(
      batch.items[1].reviewed_prompt,
    );
    expect(screen.queryByText("hostile-chat")).toBeNull();
  });

  it.each([
    ["a prompt edit whose plan digest stayed unchanged", true],
    ["a selection-only edit whose plan digest changed", false],
  ])("rejects %s", async (_label, promptChanged) => {
    vi.mocked(api.updatePromptBatchItem).mockImplementation(async (_batchId, ordinal, payload) => {
      const nextItems = latestBatch.items.map((item) => item.ordinal === ordinal
        ? {
            ...item,
            reviewed_prompt: payload.reviewed_prompt,
            reviewed_sha256: promptDigest(payload.reviewed_prompt),
            selected: payload.selected,
            review_version: payload.expected_review_version + 1,
          }
        : item);
      return {
        ...latestBatch,
        plan_sha256: promptChanged ? latestBatch.plan_sha256 : "f".repeat(64),
        plan_version: payload.expected_plan_version + 1,
        items: nextItems,
        replayed: false,
      };
    });
    renderLibrary();
    await openAndSubmitExpansion();
    await screen.findByText(/replayed safely/);
    const textarea = screen.getByLabelText("Reviewed prompt for draft 1");
    const firstCard = textarea.closest("article")!;
    if (promptChanged) {
      fireEvent.change(textarea, { target: { value: "A changed prompt with a stale plan." } });
    } else {
      fireEvent.click(within(firstCard).getByRole("checkbox", { name: "Selected" }));
    }
    fireEvent.click(within(firstCard).getByRole("button", { name: "Save review" }));

    expect(await within(firstCard).findByText(
      "This draft could not be saved. Reload the batch before trying again.",
    )).toBeVisible();
  });

  it("focuses and closes the accessible expansion dialog with Escape", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    const opener = screen.getByRole("button", { name: "Test expansion" });
    opener.focus();
    fireEvent.click(opener);

    const dialog = screen.getByRole("dialog", { name: "Test Portrait variants" });
    await waitFor(() => expect(within(dialog).getByRole("button", { name: "Close prompt expansion" })).toHaveFocus());
    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(screen.queryByRole("dialog", { name: "Test Portrait variants" })).toBeNull();
    expect(opener).toHaveFocus();
  });
});
