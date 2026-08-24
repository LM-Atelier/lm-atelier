import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { api } from "./api";
import { PromptLibraryView } from "./PromptLibraryView";
import type {
  PromptTemplateContract,
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
  },
}));

const stamp = "2026-08-20T12:00:00Z";
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

function renderLibrary() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><PromptLibraryView /></QueryClientProvider>);
}

beforeEach(() => {
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
});
