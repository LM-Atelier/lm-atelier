import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import type { PromptDirectQueueAttempt } from "./ComposerPromptTemplatesAction";
import { PromptTemplatesDialog } from "./PromptTemplatesDialog";
import type {
  PromptTemplateContract,
  PromptTemplateDefinition,
  PromptTemplateDetail,
  PromptTemplateRevision,
} from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
    promptTemplates: vi.fn(),
    promptTemplate: vi.fn(),
    createPromptTemplate: vi.fn(),
    workflowFamilies: vi.fn(),
    modelAssets: vi.fn(),
    },
  };
});

const stamp = "2026-08-21T12:00:00Z";
const contract: PromptTemplateContract = {
  schema_version: 1,
  operation: "text_to_image",
  body: "A portrait of {{subject}}.",
  slots: [{ name: "subject", mode: "input", variation_scope: "item" }],
  resource_policy: { mode: "inherited" },
};
const definition: PromptTemplateDefinition = {
  id: "template-portrait",
  name: "Portrait starter",
  description: "One subject",
  archived: false,
  current_revision_id: "revision-portrait",
  created_at: stamp,
  updated_at: stamp,
};
const revision: PromptTemplateRevision = {
  id: definition.current_revision_id,
  prompt_template_id: definition.id,
  version: 1,
  schema_version: 1,
  contract_json: contract,
  contract_sha256: "a".repeat(64),
  created_at: stamp,
};
const detail: PromptTemplateDetail = { ...definition, current_revision: revision };

function renderDialog({
  currentPrompt = "",
  maximum = 8,
  onCreate = vi.fn(),
  attempt = null,
}: {
  currentPrompt?: string;
  maximum?: number;
  onCreate?: ReturnType<typeof vi.fn>;
  attempt?: PromptDirectQueueAttempt | null;
} = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <PromptTemplatesDialog
        currentPrompt={currentPrompt}
        maximum={maximum}
        attempt={attempt}
        onClose={vi.fn()}
        onCreate={onCreate}
        onRetry={vi.fn()}
        onDiscard={vi.fn()}
      />
    </QueryClientProvider>,
  );
  return onCreate;
}

beforeEach(() => {
  vi.mocked(api.promptTemplates).mockResolvedValue({
    items: [definition],
    total: 1,
    limit: 100,
    offset: 0,
  });
  vi.mocked(api.promptTemplate).mockResolvedValue(detail);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.modelAssets).mockResolvedValue([]);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("PromptTemplatesDialog", () => {
  it("quick-creates a reusable template from the current composer text", async () => {
    const savedContract: PromptTemplateContract = {
      schema_version: 1,
      operation: "text_to_image",
      body: "A glass terrarium in soft window light",
      slots: [],
      resource_policy: { mode: "inherited" },
    };
    const savedRevision: PromptTemplateRevision = {
      ...revision,
      id: "revision-saved",
      prompt_template_id: "template-saved",
      contract_json: savedContract,
      contract_sha256: "c".repeat(64),
    };
    const saved: PromptTemplateDetail = {
      ...definition,
      id: "template-saved",
      name: "Terrarium light",
      description: "A reusable product scene",
      current_revision_id: savedRevision.id,
      current_revision: savedRevision,
    };
    vi.mocked(api.createPromptTemplate).mockResolvedValue({
      template: saved,
      revision: savedRevision,
      idempotent: false,
    });
    vi.mocked(api.promptTemplate).mockResolvedValue(saved);
    renderDialog({ currentPrompt: "  A glass terrarium in soft window light  " });

    fireEvent.click(await screen.findByRole("button", {
      name: "Save current prompt as a template",
    }));
    fireEvent.change(screen.getByLabelText("Template name"), {
      target: { value: "  Terrarium light  " },
    });
    fireEvent.change(screen.getByLabelText(/Description/), {
      target: { value: "  A reusable product scene  " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save template" }));

    await waitFor(() => expect(api.createPromptTemplate).toHaveBeenCalledWith({
      idempotency_key: expect.any(String),
      name: "Terrarium light",
      description: "A reusable product scene",
      contract: savedContract,
    }));
    expect(await screen.findByRole("heading", { name: "Terrarium light" })).toBeVisible();
  });

  it("explains a duplicate quick-create name without echoing backend text", async () => {
    vi.mocked(api.createPromptTemplate).mockRejectedValue(new ApiError(
      409,
      { detail: "C:\\private\\quick-template.json" },
      "C:\\private\\quick-template.json",
      "prompt-template-name-taken",
    ));
    renderDialog({ currentPrompt: "A glass terrarium in soft window light" });

    fireEvent.click(await screen.findByRole("button", { name: "Save current prompt as a template" }));
    fireEvent.change(screen.getByLabelText("Template name"), { target: { value: "Portrait starter" } });
    fireEvent.click(screen.getByRole("button", { name: "Save template" }));

    expect(await screen.findByText("A template with this name already exists. Choose a different name.")).toBeVisible();
    expect(screen.queryByText(/private\\quick-template/)).toBeNull();
  });

  it.each([
    [
      "prompt-batch-distinct-capacity-exceeded",
      "Distinct choice mode cannot create that many prompts. Request fewer prompts, add more choices, or allow repeats.",
    ],
    [
      "prompt-model-worker-unavailable",
      "Start a ready chat model before using model-guided template slots.",
    ],
    [
      "prompt-model-invocation-failed",
      "The chat model could not fill the template slots. Retry, or use authored inputs and choices instead.",
    ],
  ])("shows actionable direct-run feedback for %s", (errorCode, message) => {
    const attempt: PromptDirectQueueAttempt = {
      template: detail,
      createPayload: {
        idempotency_key: "preview-attempt",
        template_revision_id: revision.id,
        contract_sha256: revision.contract_sha256,
        item_count: 3,
        selection_seed: 7,
        inputs: {},
      },
      queueIdempotencyKey: "queue-attempt",
      batch: null,
      status: "error",
      errorStage: "create",
      errorCode,
    };

    renderDialog({ attempt });
    expect(screen.getByText(message)).toBeVisible();
  });

  it("configures the exact prompt count and creates directly without review or insertion", async () => {
    const onCreate = renderDialog({ maximum: 3 });

    fireEvent.click(await screen.findByRole("button", { name: /Portrait starter/ }));
    const count = await screen.findByLabelText("Number of prompts");
    expect(count).toHaveAttribute("max", "3");
    fireEvent.change(count, { target: { value: "3" } });

    fireEvent.change(screen.getByLabelText("subject for prompt 1"), {
      target: { value: "Ada" },
    });
    fireEvent.change(screen.getByLabelText("subject for prompt 2"), {
      target: { value: "Grace" },
    });
    fireEvent.change(screen.getByLabelText("subject for prompt 3"), {
      target: { value: "Katherine" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Create 3 prompts" }));

    expect(onCreate).toHaveBeenCalledWith({
      template: detail,
      itemCount: 3,
      inputs: { subject: ["Ada", "Grace", "Katherine"] },
    });
    expect(screen.queryByRole("button", { name: /test|use|insert|queue|review/i })).toBeNull();
    expect(screen.queryByLabelText(/selection seed/i)).toBeNull();
  });
});
