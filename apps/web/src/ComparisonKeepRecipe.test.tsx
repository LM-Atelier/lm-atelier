import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ApiError, api } from "./api";
import { ComparisonKeepRecipe } from "./ComparisonKeepRecipe";
import type {
  ExperimentArm, ExperimentOperation, ExperimentTrial, GenerationExperimentRecipeDraft,
} from "./generationExperimentTypes";
import type { Chat, EditTemplate, EngineCapabilities, SettingField, WorkflowRevisionSchema } from "./types";
import type { WorkflowUseCasePreset } from "./workflowUseCaseTypes";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      generationExperimentRecipeDraft: vi.fn(),
      createWorkflowUseCasePreset: vi.fn(),
      workflowSummaries: vi.fn(),
      engines: vi.fn(),
      workflowRevisionSchema: vi.fn(),
      createChat: vi.fn(),
      updateChat: vi.fn(),
      setChatWorkflowSelection: vi.fn(),
      setWorkflowUseCaseChoice: vi.fn(),
      editRecipeDraft: vi.fn(),
      createEditTemplate: vi.fn(),
    },
  };
});

function field(key: string, label: string, value: number): SettingField {
  return { key, label, type: "integer", default: value, minimum: 1, maximum: 4096, step: 1, choices: [], scope: "request",
    visibility: "basic", restart_required: false, available: true, unavailable_reason: null, help: "" };
}

const ENGINE: EngineCapabilities = { engine: "mock", version: "1", roles: ["image"], operations: ["text_to_image"], formats: [],
  devices: [], streaming: false, tool_calling: false, healthy: true, details: {},
  settings: [field("steps", "Steps", 20), field("width", "Width", 1024), field("height", "Height", 1024)] };
const SCHEMA: WorkflowRevisionSchema = { workflow_id: "w2", revision_id: "revision-b", operation: "text_to_image",
  input_schema_json: { properties: {} } };
const TRIAL: ExperimentTrial = { id: "gtrial_b", ordinal: 1, seed: 7, state: "started", work_step_id: "step_b", run_id: "run_b",
  job_id: "job_b", status: "complete" };
const ARM = { id: "garm_b", ordinal: 2, label: "More steps", profile_id: "profile-b", trials: [TRIAL] } as unknown as ExperimentArm;
const DRAFT: GenerationExperimentRecipeDraft = {
  experiment_id: "gexp_one", arm_ordinal: 2, use_case: "image_generation", name: "More steps",
  settings_json: { steps: 20, width: 1024, height: 1024 },
  left_out: [{ setting: "negative_prompt", reason: "recipe-prompt", message: "A recipe never holds the words; each request brings its own." }],
  profile_id: "profile-b", profile_name: "Lighthouse model", workflow_id: "w2", workflow_family_id: "f2",
  workflow_revision_id: "revision-b", workflow_name: "Careful", workflow_version: 5,
};
const SAVED: WorkflowUseCasePreset = { id: "wfuc_saved", name: "More steps", use_case: "image_generation",
  settings_json: { steps: 20, width: 1024, height: 1024 }, enabled: true, is_default: false, builtin: false };

function show(onOpenChat = vi.fn(), operation: ExperimentOperation = "text_to_image", arm = ARM, open = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <ComparisonKeepRecipe experimentId="gexp_one" operation={operation} arm={arm} onOpenChat={onOpenChat} />
  </QueryClientProvider>);
  if (open) fireEvent.click(screen.getByRole("button", { name: "Keep as a recipe" }));
  return onOpenChat;
}

async function saveTheDraft() {
  const dialog = await screen.findByRole("dialog", { name: "Keep More steps as a recipe" });
  await within(dialog).findByRole("spinbutton", { name: "Steps" });
  fireEvent.click(within(dialog).getByRole("button", { name: "Save recipe" }));
  await screen.findByText("Saved the recipe More steps.");
  return dialog;
}

beforeEach(() => {
  vi.mocked(api.generationExperimentRecipeDraft).mockResolvedValue(DRAFT);
  vi.mocked(api.createWorkflowUseCasePreset).mockResolvedValue(SAVED);
  vi.mocked(api.workflowSummaries).mockResolvedValue([{ id: "w2", name: "Careful", operation: "text_to_image", description: "",
    current_revision_id: "revision-b", revision_count: 1, created_at: "", updated_at: "" }]);
  vi.mocked(api.engines).mockResolvedValue([ENGINE]);
  vi.mocked(api.workflowRevisionSchema).mockResolvedValue(SCHEMA);
  vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as Chat);
  vi.mocked(api.updateChat).mockResolvedValue({ id: "chat_new" } as Chat);
  vi.mocked(api.setChatWorkflowSelection).mockResolvedValue({} as never);
  vi.mocked(api.setWorkflowUseCaseChoice).mockResolvedValue({ mode: "preset", preset_id: "wfuc_saved" });
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("drafts the recipe from the choice, names what it ran on and what was left out, and saves it as reviewed", async () => {
  show();
  const dialog = await screen.findByRole("dialog", { name: "Keep More steps as a recipe" });
  expect(api.generationExperimentRecipeDraft).toHaveBeenCalledWith("gexp_one", 2, expect.anything());
  expect(await within(dialog).findByText(/Made with Lighthouse model on Careful \(version 5\)/)).toBeInTheDocument();
  expect(within(dialog).getByText(/each request brings its own/).closest("li")).toHaveTextContent("negative_prompt");
  // The editor starts from the draft, with the choice's own workflow giving the controls.
  expect(within(dialog).getByRole("textbox", { name: "Recipe name" })).toHaveValue("More steps");
  expect(await within(dialog).findByRole("spinbutton", { name: "Steps" })).toHaveValue(20);
  expect(within(dialog).getByRole("combobox", { name: "Reference workflow for settings" })).toHaveValue("w2");

  fireEvent.click(within(dialog).getByRole("button", { name: "Save recipe" }));

  await screen.findByText("Saved the recipe More steps.");
  expect(api.createWorkflowUseCasePreset).toHaveBeenCalledExactlyOnceWith({ name: "More steps", use_case: "image_generation",
    settings_json: { steps: 20, width: 1024, height: 1024 }, enabled: true, is_default: false });
  // Saving changes nothing else: no chat until one is asked for.
  expect(api.createChat).not.toHaveBeenCalled();
});

it("sets up a new chat with the choice's model, workflow and recipe, then shows it", async () => {
  const onOpenChat = show();
  const dialog = await saveTheDraft();

  fireEvent.click(within(dialog).getByRole("button", { name: "Use in a new chat" }));

  await waitFor(() => expect(onOpenChat).toHaveBeenCalledExactlyOnceWith("chat_new"));
  expect(api.createChat).toHaveBeenCalledExactlyOnceWith(null);
  expect(api.updateChat).toHaveBeenCalledExactlyOnceWith("chat_new", { active_image_profile_id: "profile-b" });
  expect(api.setChatWorkflowSelection).toHaveBeenCalledExactlyOnceWith("chat_new", "image",
    { mode: "family", workflow_family_id: "f2" });
  expect(api.setWorkflowUseCaseChoice).toHaveBeenCalledExactlyOnceWith({ kind: "chat", id: "chat_new" }, "image_generation",
    { mode: "preset", preset_id: "wfuc_saved" });
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("says so, and shows no chat, when the new chat could not take the whole setup", async () => {
  vi.mocked(api.updateChat).mockRejectedValue(new Error("That model is no longer installed."));
  const onOpenChat = show();
  const dialog = await saveTheDraft();

  fireEvent.click(within(dialog).getByRole("button", { name: "Use in a new chat" }));

  expect(await within(dialog).findByText(
    "A new chat was made, but not all of this setup could be applied to it. That model is no longer installed.",
  )).toBeInTheDocument();
  expect(onOpenChat).not.toHaveBeenCalled();
  expect(api.setWorkflowUseCaseChoice).not.toHaveBeenCalled();
});

it("says why when no draft can be made from the choice", async () => {
  vi.mocked(api.generationExperimentRecipeDraft).mockRejectedValue(
    new ApiError(409, "The workflow this choice ran on cannot take a recipe now.", "The workflow this choice ran on cannot take a recipe now.",
      "generation-experiment-recipe-unavailable"),
  );
  show();
  const dialog = await screen.findByRole("dialog", { name: "Keep More steps as a recipe" });

  expect(await within(dialog).findByText("The workflow this choice ran on cannot take a recipe now.")).toBeInTheDocument();
  expect(within(dialog).queryByRole("button", { name: "Save recipe" })).toBeNull();
});

it("keeps a choice that changed a picture as an Image Studio recipe, read from the run that made its picture", async () => {
  vi.mocked(api.editRecipeDraft).mockResolvedValue({ run_id: "run_b", instruction: "Warmer evening light" });
  vi.mocked(api.createEditTemplate).mockResolvedValue({ id: "tmpl_1", name: "Heavier touch" } as EditTemplate);
  show(vi.fn(), "image_to_image");
  const dialog = await screen.findByRole("dialog", { name: "Keep this edit as a recipe" });

  expect(await within(dialog).findByRole("textbox", { name: "Words" })).toHaveValue("Warmer evening light");
  expect(api.editRecipeDraft).toHaveBeenCalledWith("run_b", expect.anything());
  expect(api.generationExperimentRecipeDraft).not.toHaveBeenCalled();
  fireEvent.change(within(dialog).getByRole("textbox", { name: "Recipe name" }), { target: { value: "Heavier touch" } });
  fireEvent.click(within(dialog).getByRole("button", { name: "Save recipe" }));

  await screen.findByText("Saved the recipe Heavier touch. Image Studio offers it with its recipes.");
  expect(api.createEditTemplate).toHaveBeenCalledExactlyOnceWith(
    { name: "Heavier touch", instruction: "Warmer evening light", from_run_id: "run_b" });
});

it("offers no recipe for a changed picture until that choice's picture is made", () => {
  show(vi.fn(), "image_to_image", { ...ARM, trials: [{ ...TRIAL, status: "running" }] }, false);

  expect(screen.queryByRole("button", { name: "Keep as a recipe" })).toBeNull();
});
