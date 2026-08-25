import { createHash } from "node:crypto";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { ComposerPromptTemplatesAction } from "./ComposerPromptTemplatesAction";
import type {
  PromptBatch,
  PromptBatchCreateInput,
  PromptBatchQueueInput,
  PromptTemplateDetail,
} from "./types";

vi.mock("./api", () => ({
  api: {
    promptTemplates: vi.fn(),
    promptTemplate: vi.fn(),
    createPromptTemplate: vi.fn(),
    createPromptBatch: vi.fn(),
    queuePromptBatch: vi.fn(),
  },
}));

const stamp = "2026-08-21T12:00:00Z";
const template: PromptTemplateDetail = {
  id: "template-one",
  name: "Portrait prompts",
  description: "",
  archived: false,
  current_revision_id: "revision-one",
  created_at: stamp,
  updated_at: stamp,
  current_revision: {
    id: "revision-one",
    prompt_template_id: "template-one",
    version: 1,
    schema_version: 1,
    contract_json: {
      schema_version: 1,
      operation: "text_to_image",
      body: "A portrait of {{subject}}.",
      slots: [{ name: "subject", mode: "input", variation_scope: "item" }],
      resource_policy: { mode: "inherited" },
    },
    contract_sha256: "a".repeat(64),
    created_at: stamp,
  },
};

function digest(prompt: string): string {
  return createHash("sha256")
    .update(`prompt-expansion-rendered-v1\0${prompt}`, "utf8")
    .digest("hex");
}

function draftFor(payload: PromptBatchCreateInput): PromptBatch {
  const subjects = payload.inputs.subject as string[];
  return {
    id: "batch-one",
    chat_id: "chat-one",
    prompt_template_id: template.id,
    prompt_template_revision_id: template.current_revision.id,
    schema_version: template.current_revision.schema_version,
    contract_sha256: template.current_revision.contract_sha256,
    codec_version: 2,
    requested_count: payload.item_count,
    selection_seed: payload.selection_seed,
    plan_sha256: "b".repeat(64),
    state: "draft",
    plan_version: 1,
    queue_idempotency_key: null,
    work_plan_id: null,
    queued_at: null,
    replayed: false,
    items: subjects.map((subject, index) => {
      const prompt = `A portrait of ${subject}.`;
      return {
        id: `item-${index + 1}`,
        ordinal: index + 1,
        rendered_prompt: prompt,
        rendered_sha256: digest(prompt),
        reviewed_prompt: prompt,
        reviewed_sha256: digest(prompt),
        selected: true,
        review_version: 1,
        reroll_count: 0,
        work_step_id: null,
        run_id: null,
        media_seed: null,
      };
    }),
  };
}

function queuedFrom(draft: PromptBatch, payload: PromptBatchQueueInput): PromptBatch {
  return {
    ...draft,
    state: "queued",
    plan_version: 2,
    queue_idempotency_key: payload.idempotency_key,
    work_plan_id: "work-plan-one",
    queued_at: stamp,
    items: draft.items.map((item) => ({
      ...item,
      work_step_id: `step-${item.ordinal}`,
      run_id: `run-${item.ordinal}`,
      media_seed: 100 + item.ordinal,
    })),
  };
}

function renderAction(maximum = 8) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <ComposerPromptTemplatesAction
        chatId="chat-one"
        currentPrompt=""
        maximum={maximum}
      />
    </QueryClientProvider>,
  );
  return client;
}

async function configureAndCreate(count = 2) {
  fireEvent.click(screen.getByRole("button", { name: "Open prompt templates" }));
  fireEvent.click(await screen.findByRole("button", { name: /Portrait prompts/ }));
  fireEvent.change(await screen.findByLabelText("Number of prompts"), {
    target: { value: String(count) },
  });
  for (let index = 1; index <= count; index += 1) {
    fireEvent.change(screen.getByLabelText(`subject for prompt ${index}`), {
      target: { value: `Subject ${index}` },
    });
  }
  const create = screen.getByRole("button", {
    name: count === 1 ? "Create prompt" : `Create ${count} prompts`,
  });
  fireEvent.click(create);
  return create;
}

beforeEach(() => {
  vi.mocked(api.promptTemplates).mockResolvedValue({
    items: [template],
    total: 1,
    limit: 100,
    offset: 0,
  });
  vi.mocked(api.promptTemplate).mockResolvedValue(template);
  vi.mocked(api.createPromptBatch).mockImplementation(
    async (_chatId, payload) => draftFor(payload),
  );
  vi.mocked(api.queuePromptBatch).mockImplementation(
    async (_batchId, payload) => {
      const createPayload = vi.mocked(api.createPromptBatch).mock.calls.at(-1)![1];
      return queuedFrom(draftFor(createPayload), payload);
    },
  );
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("ComposerPromptTemplatesAction", () => {
  it("creates and atomically queues the requested count with one click", async () => {
    const client = renderAction(3);
    await configureAndCreate(3);

    await waitFor(() => expect(api.queuePromptBatch).toHaveBeenCalledTimes(1));
    expect(api.createPromptBatch).toHaveBeenCalledTimes(1);
    const createPayload = vi.mocked(api.createPromptBatch).mock.calls[0][1];
    expect(api.createPromptBatch).toHaveBeenCalledWith("chat-one", {
      idempotency_key: expect.any(String),
      template_revision_id: template.current_revision.id,
      contract_sha256: template.current_revision.contract_sha256,
      item_count: 3,
      selection_seed: expect.any(Number),
      inputs: { subject: ["Subject 1", "Subject 2", "Subject 3"] },
    });
    expect(api.queuePromptBatch).toHaveBeenCalledWith("batch-one", {
      idempotency_key: expect.any(String),
      expected_plan_version: 1,
      expected_plan_sha256: "b".repeat(64),
    });
    expect(await screen.findByRole("heading", { name: "3 prompts queued" })).toBeVisible();
    expect(client.getQueryData<PromptBatch>(["prompt-batch", "batch-one"]))
      .toMatchObject({ state: "queued", requested_count: 3 });
    expect(createPayload.selection_seed).toBeGreaterThanOrEqual(0);
    expect(createPayload.selection_seed).toBeLessThanOrEqual(2_147_483_647);
  });

  it("uses a synchronous guard so a double click creates one attempt", async () => {
    let resolveCreate!: (value: PromptBatch) => void;
    vi.mocked(api.createPromptBatch).mockImplementationOnce((_chatId, payload) =>
      new Promise((resolve) => {
        resolveCreate = resolve;
        queueMicrotask(() => resolveCreate(draftFor(payload)));
      }));
    renderAction();

    const create = await configureAndCreate(1);
    fireEvent.click(create);

    await waitFor(() => expect(api.createPromptBatch).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.queuePromptBatch).toHaveBeenCalledTimes(1));
  });

  it("retries a lost create with the exact same payload and idempotency key", async () => {
    vi.mocked(api.createPromptBatch)
      .mockRejectedValueOnce(new Error("private backend detail"))
      .mockImplementationOnce(async (_chatId, payload) => draftFor(payload));
    renderAction();
    await configureAndCreate(1);

    fireEvent.click(await screen.findByRole("button", { name: "Retry" }));
    await waitFor(() => expect(api.queuePromptBatch).toHaveBeenCalledTimes(1));
    expect(api.createPromptBatch).toHaveBeenCalledTimes(2);
    expect(vi.mocked(api.createPromptBatch).mock.calls[1])
      .toEqual(vi.mocked(api.createPromptBatch).mock.calls[0]);
    expect(screen.queryByText(/private backend detail/)).toBeNull();
  });

  it("retains the admitted draft and queue key across close, reopen, and retry", async () => {
    vi.mocked(api.queuePromptBatch)
      .mockRejectedValueOnce(new Error("lost queue response"))
      .mockImplementationOnce(async (_batchId, payload) => {
        const createPayload = vi.mocked(api.createPromptBatch).mock.calls[0][1];
        return queuedFrom(draftFor(createPayload), payload);
      });
    renderAction();
    await configureAndCreate(2);

    const retry = await screen.findByRole("button", { name: "Retry queue" });
    const firstQueueCall = structuredClone(vi.mocked(api.queuePromptBatch).mock.calls[0]);
    fireEvent.click(screen.getByRole("button", { name: "Close prompt templates" }));
    fireEvent.click(screen.getByRole("button", { name: "Open prompt templates" }));
    expect(await screen.findByRole("button", { name: "Retry queue" })).toBeVisible();
    fireEvent.click(retry.isConnected ? retry : screen.getByRole("button", { name: "Retry queue" }));

    await waitFor(() => expect(api.queuePromptBatch).toHaveBeenCalledTimes(2));
    expect(api.createPromptBatch).toHaveBeenCalledTimes(1);
    expect(vi.mocked(api.queuePromptBatch).mock.calls[1]).toEqual(firstQueueCall);
    expect(await screen.findByRole("heading", { name: "2 prompts queued" })).toBeVisible();
  });

  it("accepts an exact already-queued create replay without queueing it again", async () => {
    const createKey = "00000000-0000-4000-8000-000000000001";
    const queueKey = "00000000-0000-4000-8000-000000000002";
    const uuid = vi.spyOn(crypto, "randomUUID");
    uuid
      .mockReturnValueOnce(createKey)
      .mockReturnValueOnce(queueKey);
    vi.mocked(api.createPromptBatch).mockImplementationOnce(async (_chatId, payload) => ({
      ...queuedFrom(draftFor(payload), {
        idempotency_key: queueKey,
        expected_plan_version: 1,
        expected_plan_sha256: "b".repeat(64),
      }),
      replayed: true,
    }));
    renderAction();
    await configureAndCreate(1);

    expect(await screen.findByRole("heading", { name: "1 prompt queued" })).toBeVisible();
    expect(api.createPromptBatch).toHaveBeenCalledWith(
      "chat-one",
      expect.objectContaining({ idempotency_key: createKey }),
    );
    expect(api.queuePromptBatch).not.toHaveBeenCalled();
    uuid.mockRestore();
  });
});
