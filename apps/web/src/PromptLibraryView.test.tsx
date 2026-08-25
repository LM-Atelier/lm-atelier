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
    chat: vi.fn(),
    createPromptBatch: vi.fn(),
    promptBatch: vi.fn(),
    updatePromptBatchItem: vi.fn(),
    queuePromptBatch: vi.fn(),
    workflowFamilies: vi.fn(),
    modelAssets: vi.fn(),
  },
}));

const stamp = "2026-08-20T12:00:00Z";
function readyVariant(name: string, revisionId: string) {
  return {
    id: `wfvar_${revisionId}`,
    variant_key: revisionId,
    name,
    operation: "text_to_image",
    current_revision_id: revisionId,
    current_revision_version: 1,
    engine: "comfyui",
    capabilities: ["image"],
    trusted: true,
    readiness: "ready" as const,
    readiness_reason: null,
  };
}
const imageFamilies = [{
  id: "wffam_one",
  name: "Portrait",
  description: "",
  use_case: "image",
  tags: [],
  enabled: true,
  archived: false,
  compatibility: true,
  variants: [
    readyVariant("Base", "workflow-revision-1"),
    readyVariant("Detail", "workflow-revision-2"),
    readyVariant("Shared", "shared-revision"),
    readyVariant("Same", "same-revision"),
    readyVariant("Pool", "workflow-revision-pool"),
    // Ready, in an image family, and the wrong operation: the selector filters
    // families by preference, so this one must be excluded by operation here.
    { ...readyVariant("Edit", "edit-revision"), operation: "image_to_image" },
    // Right operation, not ready: a pool may only name a revision that can run.
    {
      ...readyVariant("Unready", "unready-revision"),
      readiness: "setup_required" as const,
      readiness_reason: "missing model",
    },
  ],
  preferences: [{ selector_capability: "image" as const, enabled: true, is_default: true, sort_order: 0 }],
  created_at: stamp,
  updated_at: stamp,
}, {
  // Ready, text-to-image, and the user has turned this family off for image.
  // The real selector hides it, so the pool editor must too.
  id: "wffam_two",
  name: "Disabled",
  description: "",
  use_case: "image",
  tags: [],
  enabled: true,
  archived: false,
  compatibility: true,
  variants: [readyVariant("Off", "disabled-family-revision")],
  preferences: [{ selector_capability: "image" as const, enabled: false, is_default: false, sort_order: 1 }],
  created_at: stamp,
  updated_at: stamp,
}];
const installedLoraDigests = [
  "a".repeat(64), "b".repeat(64), "c".repeat(64), "d".repeat(64),
  "1".repeat(64), "2".repeat(64),
  ...Array.from({ length: 70 }, (_, index) => index.toString(16).padStart(64, "0")),
];
const installedLoras = installedLoraDigests.map((sha256, index) => ({
  id: `asset_${index}`,
  source_id: null,
  kind: "lora",
  name: `Portrait LoRA ${index + 1}`,
  family: index < 6 ? "Portrait styles" : null,
  size_bytes: 1,
  active: true,
  use_case: "",
  auto_apply: false,
  verified_at: stamp,
  created_at: stamp,
  updated_at: stamp,
  default_model_strength: 1,
  default_clip_strength: 1,
  manifest_json: { sha256 },
}));
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
  return client;
}

beforeEach(() => {
  vi.mocked(api.promptTemplates).mockResolvedValue({
    items: [definition],
    total: 1,
    limit: 100,
    offset: 0,
  });
  vi.mocked(api.promptTemplate).mockResolvedValue(detail);
  vi.mocked(api.workflowFamilies).mockResolvedValue(imageFamilies);
  vi.mocked(api.modelAssets).mockResolvedValue(installedLoras);
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
    await screen.findByRole("option", { name: "Portrait - Base - revision 1" });
    fireEvent.change(screen.getByLabelText("Workflow"), { target: { value: "workflow-revision-1" } });
    fireEvent.change(screen.getByLabelText("LoRA policy"), { target: { value: "fixed" } });
    await screen.findByRole("option", { name: "Portrait LoRA 3 - Portrait styles" });
    fireEvent.change(screen.getByLabelText("LoRA 1"), { target: { value: "c".repeat(64) } });
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

  it("authors an exact workflow bundle pool with per-option LoRA policies", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });
    fireEvent.change(screen.getByLabelText("Pool strategy"), { target: { value: "random" } });
    fireEvent.change(screen.getByLabelText("Option 1 workflow revision"), { target: { value: "workflow-revision-1" } });
    fireEvent.change(screen.getByLabelText("Option 2 workflow revision"), { target: { value: "workflow-revision-2" } });
    fireEvent.change(screen.getByLabelText("Option 2 LoRA policy"), { target: { value: "fixed" } });
    await screen.findByRole("option", { name: "Portrait LoRA 4 - Portrait styles" });
    fireEvent.change(screen.getByLabelText("Option 2 LoRA 1"), { target: { value: "d".repeat(64) } });
    fireEvent.change(screen.getByLabelText("Option 2 LoRA 1 model strength"), { target: { value: "0.6" } });
    fireEvent.change(screen.getByLabelText("Option 2 LoRA 1 CLIP strength"), { target: { value: "0.5" } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    await waitFor(() => expect(api.createPromptTemplate).toHaveBeenCalledWith(expect.objectContaining({
      contract: expect.objectContaining({
        resource_policy: {
          mode: "pool",
          strategy: "random",
          options: [
            { workflow_revision_id: "workflow-revision-1", lora_policy: { mode: "inherited_auto" } },
            {
              workflow_revision_id: "workflow-revision-2",
              lora_policy: {
                mode: "fixed",
                stack: [{ sha256: "d".repeat(64), model_strength: 0.6, clip_strength: 0.5 }],
              },
            },
          ],
        },
      }),
    })));
  });

  it("accepts one workflow revision paired with two different LoRA policies", async () => {
    // Uniqueness is on the whole bundle, not the revision id, so the same
    // workflow may appear twice when its LoRA policy differs.
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });
    fireEvent.change(screen.getByLabelText("Option 1 workflow revision"), { target: { value: "shared-revision" } });
    fireEvent.change(screen.getByLabelText("Option 2 workflow revision"), { target: { value: "shared-revision" } });
    fireEvent.change(screen.getByLabelText("Option 2 LoRA policy"), { target: { value: "none" } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    await waitFor(() => expect(api.createPromptTemplate).toHaveBeenCalledWith(expect.objectContaining({
      contract: expect.objectContaining({
        resource_policy: {
          mode: "pool",
          strategy: "round_robin",
          options: [
            { workflow_revision_id: "shared-revision", lora_policy: { mode: "inherited_auto" } },
            { workflow_revision_id: "shared-revision", lora_policy: { mode: "none" } },
          ],
        },
      }),
    })));
  });

  it("refuses two identical workflow and LoRA bundles", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });
    fireEvent.change(screen.getByLabelText("Option 1 workflow revision"), { target: { value: "same-revision" } });
    fireEvent.change(screen.getByLabelText("Option 2 workflow revision"), { target: { value: "same-revision" } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    expect(await screen.findByText("Every workflow pool option must be a distinct workflow and LoRA bundle.")).toBeInTheDocument();
    expect(api.createPromptTemplate).not.toHaveBeenCalled();
  });

  it("refuses a pool option with no workflow revision", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });
    fireEvent.change(screen.getByLabelText("Option 1 workflow revision"), { target: { value: "workflow-revision-1" } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    expect(await screen.findByText("Option 2 needs an exact workflow revision.")).toBeInTheDocument();
    expect(api.createPromptTemplate).not.toHaveBeenCalled();
  });

  it("omits a family whose image preference is disabled", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });

    expect(screen.queryByRole("option", { name: /Disabled/ })).not.toBeInTheDocument();
    expect(
      Array.from(screen.getByLabelText("Option 1 workflow revision").querySelectorAll("option"))
        .map((option) => option.getAttribute("value")),
    ).not.toContain("disabled-family-revision");
  });

  it("omits a variant that is not ready", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });

    expect(screen.queryByRole("option", { name: /Unready/ })).not.toBeInTheDocument();
    expect(
      Array.from(screen.getByLabelText("Option 1 workflow revision").querySelectorAll("option"))
        .map((option) => option.getAttribute("value")),
    ).not.toContain("unready-revision");
  });

  it("refuses a pool option whose fixed stack is incomplete", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });
    fireEvent.change(screen.getByLabelText("Option 1 workflow revision"), { target: { value: "workflow-revision-1" } });
    fireEvent.change(screen.getByLabelText("Option 2 workflow revision"), { target: { value: "workflow-revision-2" } });
    // A fixed stack starts with one empty LoRA; leaving its digest blank must
    // refuse rather than submit an unusable stack.
    fireEvent.change(screen.getByLabelText("Option 2 LoRA policy"), { target: { value: "fixed" } });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    expect(await screen.findByText("Choose an installed LoRA for Option 2.")).toBeInTheDocument();
    expect(api.createPromptTemplate).not.toHaveBeenCalled();
  });

  it("treats one workflow with two different fixed stacks as two options", async () => {
    // Uniqueness is on the whole bundle, so the stack has to be part of the key.
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });
    for (const option of [1, 2]) {
      fireEvent.change(screen.getByLabelText(`Option ${option} workflow revision`), { target: { value: "shared-revision" } });
      fireEvent.change(screen.getByLabelText(`Option ${option} LoRA policy`), { target: { value: "fixed" } });
      if (option === 1) await screen.findByRole("option", { name: "Portrait LoRA 5 - Portrait styles" });
      fireEvent.change(screen.getByLabelText(`Option ${option} LoRA 1`), { target: { value: `${option}`.repeat(64) } });
      fireEvent.change(screen.getByLabelText(`Option ${option} LoRA 1 model strength`), { target: { value: "1" } });
      fireEvent.change(screen.getByLabelText(`Option ${option} LoRA 1 CLIP strength`), { target: { value: "1" } });
    }
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    await waitFor(() => expect(api.createPromptTemplate).toHaveBeenCalledWith(expect.objectContaining({
      contract: expect.objectContaining({
        resource_policy: {
          mode: "pool",
          strategy: "round_robin",
          options: [
            {
              workflow_revision_id: "shared-revision",
              lora_policy: { mode: "fixed", stack: [{ sha256: "1".repeat(64), model_strength: 1, clip_strength: 1 }] },
            },
            {
              workflow_revision_id: "shared-revision",
              lora_policy: { mode: "fixed", stack: [{ sha256: "2".repeat(64), model_strength: 1, clip_strength: 1 }] },
            },
          ],
        },
      }),
    })));
  });

  it("omits a ready variant whose operation is not text to image", async () => {
    // workflowFamilies("image") filters FAMILIES by selector preference, so a
    // family can carry an image-to-image variant that is perfectly ready and
    // still wrong for a prompt template.
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });

    expect(screen.queryByRole("option", { name: /Edit/ })).not.toBeInTheDocument();
    const first = screen.getByLabelText("Option 1 workflow revision");
    expect(
      Array.from(first.querySelectorAll("option")).map((option) => option.getAttribute("value")),
    ).toEqual(["", "workflow-revision-1", "workflow-revision-2", "shared-revision", "same-revision", "workflow-revision-pool"]);
  });

  it("does not call a pinned revision stale when the readiness read failed", async () => {
    // An empty ready set means "we could not look", not "your workflow is gone".
    vi.mocked(api.workflowFamilies).mockRejectedValue(new Error("offline"));
    vi.mocked(api.promptTemplate).mockResolvedValue({
      ...detail,
      current_revision: {
        ...currentRevision,
        contract_json: {
          ...contract,
          resource_policy: {
            mode: "pool",
            strategy: "round_robin",
            options: [
              { workflow_revision_id: "retired-revision", lora_policy: { mode: "inherited_auto" } },
              { workflow_revision_id: "workflow-revision-1", lora_policy: { mode: "none" } },
            ],
          },
        },
      },
    });
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));

    expect(await screen.findByText(/Could not read which image workflows are ready/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
    // The pinned values survive, and neither is described as stale.
    expect(screen.getByLabelText("Option 1 workflow revision")).toHaveValue("retired-revision");
    expect(screen.getByLabelText("Option 1 workflow revision")).toHaveTextContent("Previously selected workflow");
    expect(screen.queryByText(/not currently ready/)).not.toBeInTheDocument();
  });

  it("keeps a pinned revision that is no longer ready instead of moving to the tip", async () => {
    // A template authored earlier can name a revision the ready list no longer
    // offers. Dropping it here would silently retarget the template.
    vi.mocked(api.promptTemplate).mockResolvedValue({
      ...detail,
      current_revision: {
        ...currentRevision,
        contract_json: {
          ...contract,
          resource_policy: {
            mode: "pool",
            strategy: "round_robin",
            options: [
              { workflow_revision_id: "retired-revision", lora_policy: { mode: "inherited_auto" } },
              { workflow_revision_id: "workflow-revision-1", lora_policy: { mode: "none" } },
            ],
          },
        },
      },
    });
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });

    const first = screen.getByLabelText("Option 1 workflow revision");
    expect(first).toHaveValue("retired-revision");
    expect(first).toHaveTextContent("Previously selected workflow");
    // The still-ready option is offered by name rather than by raw id.
    expect(screen.getByLabelText("Option 2 workflow revision")).toHaveValue("workflow-revision-1");
  });

  it("refuses a fixed stack on another option once sixty-four LoRAs are pooled", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });
    // Nine options: eight carrying eight LoRAs each, and one still automatic.
    // Eight per stack rather than sixteen on purpose - at sixteen the Add LoRA
    // button is already disabled by the per-stack limit, so the pool cap would
    // be masked and its own guard could be deleted unnoticed.
    for (let option = 2; option < 9; option += 1) {
      fireEvent.click(screen.getByRole("button", { name: "Add option" }));
    }
    for (let option = 1; option <= 8; option += 1) {
      fireEvent.change(screen.getByLabelText(`Option ${option} LoRA policy`), { target: { value: "fixed" } });
      if (option === 1) await screen.findByRole("option", { name: "Portrait LoRA 1 - Portrait styles" });
      for (let lora = 1; lora < 8; lora += 1) {
        fireEvent.click(screen.getAllByRole("button", { name: "Add LoRA" })[option - 1]);
      }
    }
    expect(screen.getByText("9 options · 64 paired LoRAs of 64")).toBeInTheDocument();

    // The ninth option may no longer take a stack, because doing so would mint
    // a sixty-fifth LoRA before any validation ran.
    const ninth = screen.getByLabelText("Option 9 LoRA policy");
    const fixedChoice = Array.from(ninth.querySelectorAll("option"))
      .find((option) => option.getAttribute("value") === "fixed");
    expect(fixedChoice).toBeDisabled();
    // The other way past the cap is growing a stack that already exists, and
    // every one of these holds only eight of its sixteen.
    for (const button of screen.getAllByRole("button", { name: "Add LoRA" })) {
      expect(button).toBeDisabled();
    }
    fireEvent.change(ninth, { target: { value: "fixed" } });
    expect(ninth).toHaveValue("inherited_auto");
    expect(screen.getByText("9 options · 64 paired LoRAs of 64")).toBeInTheDocument();
    // Building a maximal pool is sixty-odd interactions, so this one case needs
    // more than the default budget.
  }, 30_000);

  it("keeps the pool between two and sixteen options and offers no nested LoRA pool", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "pool" } });
    await screen.findAllByRole("option", { name: "Portrait · Base (ready)" });

    // Two options are the floor, so neither may be removed.
    for (const button of screen.getAllByRole("button", { name: "Remove option" })) {
      expect(button).toBeDisabled();
    }
    // A pool option may not itself hold a LoRA pool.
    expect(
      Array.from(screen.getByLabelText("Option 1 LoRA policy").querySelectorAll("option"))
        .map((option) => option.getAttribute("value")),
    ).toEqual(["inherited_auto", "none", "fixed"]);

    const add = screen.getByRole("button", { name: "Add option" });
    for (let index = 2; index < 16; index += 1) fireEvent.click(add);
    expect(screen.getAllByRole("button", { name: "Remove option" })).toHaveLength(16);
    expect(add).toBeDisabled();
  });

  it("authors an exact deterministic LoRA stack pool", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });
    fireEvent.click(screen.getByRole("button", { name: "New template" }));
    fireEvent.change(screen.getByLabelText("Resource policy"), { target: { value: "fixed" } });
    await screen.findByRole("option", { name: "Portrait - Pool - revision 1" });
    fireEvent.change(screen.getByLabelText("Workflow"), {
      target: { value: "workflow-revision-pool" },
    });
    fireEvent.change(screen.getByLabelText("LoRA policy"), { target: { value: "pool" } });
    fireEvent.change(screen.getByLabelText("LoRA pool strategy"), {
      target: { value: "random" },
    });
    await screen.findAllByRole("option", { name: "Portrait LoRA 1 - Portrait styles" });
    fireEvent.change(screen.getByLabelText("Stack 1 LoRA 1"), {
      target: { value: "a".repeat(64) },
    });
    fireEvent.change(screen.getByLabelText("Stack 1 LoRA 1 model strength"), {
      target: { value: "0.8" },
    });
    fireEvent.change(screen.getByLabelText("Stack 1 LoRA 1 CLIP strength"), {
      target: { value: "0.7" },
    });
    fireEvent.change(screen.getByLabelText("Stack 2 LoRA 1"), {
      target: { value: "b".repeat(64) },
    });
    fireEvent.change(screen.getByLabelText("Stack 2 LoRA 1 model strength"), {
      target: { value: "1.1" },
    });
    fireEvent.change(screen.getByLabelText("Stack 2 LoRA 1 CLIP strength"), {
      target: { value: "0.9" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save revision" }));

    await waitFor(() => expect(api.createPromptTemplate).toHaveBeenCalledWith(
      expect.objectContaining({
        contract: expect.objectContaining({
          resource_policy: {
            mode: "fixed",
            workflow_revision_id: "workflow-revision-pool",
            lora_policy: {
              mode: "pool",
              strategy: "random",
              stacks: [
                [{ sha256: "a".repeat(64), model_strength: 0.8, clip_strength: 0.7 }],
                [{ sha256: "b".repeat(64), model_strength: 1.1, clip_strength: 0.9 }],
              ],
            },
          },
        }),
      }),
    ));
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

  it("keeps prompt execution and composer insertion out of the management library", async () => {
    renderLibrary();
    await screen.findByRole("heading", { name: "Portrait variants" });

    expect(screen.queryByRole("button", { name: /test|run|use|insert|queue/i })).toBeNull();
    expect(api.chat).not.toHaveBeenCalled();
    expect(api.createPromptBatch).not.toHaveBeenCalled();
    expect(api.promptBatch).not.toHaveBeenCalled();
    expect(api.updatePromptBatchItem).not.toHaveBeenCalled();
    expect(api.queuePromptBatch).not.toHaveBeenCalled();
  });

});
