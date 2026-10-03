import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { RecipeDraftDialog } from "./RecipeDraftDialog";
import type { RecipeDraft } from "./recipeDraftTypes";
import type { Chat, EngineCapabilities, WorkflowRevisionSchema } from "./types";
import type { WorkflowUseCasePreset } from "./workflowUseCaseTypes";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      createWorkflowUseCasePreset: vi.fn(),
      workflowSummaries: vi.fn(),
      engines: vi.fn(),
      workflowRevisionSchema: vi.fn(),
      createChat: vi.fn(),
      updateChat: vi.fn(),
      setChatWorkflowSelection: vi.fn(),
      setWorkflowUseCaseChoice: vi.fn(),
    },
  };
});

const ENGINE: EngineCapabilities = { engine: "mock", version: "1", roles: ["image", "video"],
  operations: ["text_to_image", "image_to_video"], formats: [], devices: [], streaming: false, tool_calling: false,
  healthy: true, details: {}, settings: [] };
const SCHEMA: WorkflowRevisionSchema = { workflow_id: "w3", revision_id: "revision-c", operation: "image_to_video",
  input_schema_json: { properties: {} } };
const DRAFT: RecipeDraft = {
  use_case: "video_animate", name: "Slow pan", settings_json: {},
  left_out: [{ setting: "seed", reason: "recipe-seed", message: "A recipe leaves the seed to each request, so each one comes out new." }],
  profile_id: "profile-v", profile_name: "Harbor motion", workflow_id: "w3", workflow_family_id: "f3",
  workflow_name: "Pan", workflow_version: 2,
};
const SAVED: WorkflowUseCasePreset = { id: "wfuc_video", name: "Slow pan", use_case: "video_animate", settings_json: {},
  enabled: true, is_default: false, builtin: false };

function Draft({ draft, onOpenChat }: { draft: RecipeDraft; onOpenChat?: (chatId: string) => void }) {
  const query = useQuery({ queryKey: ["draft", draft.name], queryFn: () => Promise.resolve(draft) });
  return <RecipeDraftDialog title="Keep it as a recipe" eyebrow="Recipe" draft={query}
    failure={(error) => error.message} onOpenChat={onOpenChat} onClose={() => {}} />;
}

async function save(draft: RecipeDraft, onOpenChat?: (chatId: string) => void) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><Draft draft={draft} onOpenChat={onOpenChat} /></QueryClientProvider>);
  const dialog = await screen.findByRole("dialog", { name: "Keep it as a recipe" });
  fireEvent.click(await within(dialog).findByRole("button", { name: "Save recipe" }));
  await within(dialog).findByText("Saved the recipe Slow pan.");
  return dialog;
}

async function saveAndUse(draft: RecipeDraft) {
  const onOpenChat = vi.fn();
  const dialog = await save(draft, onOpenChat);
  fireEvent.click(within(dialog).getByRole("button", { name: "Use in a new chat" }));
  await waitFor(() => expect(onOpenChat).toHaveBeenCalledExactlyOnceWith("chat_new"));
}

beforeEach(() => {
  vi.mocked(api.createWorkflowUseCasePreset).mockResolvedValue(SAVED);
  vi.mocked(api.workflowSummaries).mockResolvedValue([]);
  vi.mocked(api.engines).mockResolvedValue([ENGINE]);
  vi.mocked(api.workflowRevisionSchema).mockResolvedValue(SCHEMA);
  vi.mocked(api.createChat).mockResolvedValue({ id: "chat_new" } as Chat);
  vi.mocked(api.updateChat).mockResolvedValue({ id: "chat_new" } as Chat);
  vi.mocked(api.setChatWorkflowSelection).mockResolvedValue({} as never);
  vi.mocked(api.setWorkflowUseCaseChoice).mockResolvedValue({ mode: "preset", preset_id: "wfuc_video" });
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

it("sets up a video recipe's chat with the video model and workflow", async () => {
  await saveAndUse(DRAFT);

  expect(api.createWorkflowUseCasePreset).toHaveBeenCalledExactlyOnceWith({ name: "Slow pan", use_case: "video_animate",
    settings_json: {}, enabled: true, is_default: false });
  expect(api.updateChat).toHaveBeenCalledExactlyOnceWith("chat_new", { active_video_profile_id: "profile-v" });
  expect(api.setChatWorkflowSelection).toHaveBeenCalledExactlyOnceWith("chat_new", "video",
    { mode: "family", workflow_family_id: "f3" });
  expect(api.setWorkflowUseCaseChoice).toHaveBeenCalledExactlyOnceWith({ kind: "chat", id: "chat_new" }, "video_animate",
    { mode: "preset", preset_id: "wfuc_video" });
});

it("leaves the chat's model alone when the draft names none", async () => {
  vi.mocked(api.createWorkflowUseCasePreset).mockResolvedValue({ ...SAVED, use_case: "image_generation" });
  await saveAndUse({ ...DRAFT, use_case: "image_generation", profile_id: null, profile_name: null });

  expect(api.updateChat).not.toHaveBeenCalled();
  expect(api.setChatWorkflowSelection).toHaveBeenCalledExactlyOnceWith("chat_new", "image",
    { mode: "family", workflow_family_id: "f3" });
  expect(api.setWorkflowUseCaseChoice).toHaveBeenCalledExactlyOnceWith({ kind: "chat", id: "chat_new" }, "image_generation",
    { mode: "preset", preset_id: "wfuc_video" });
});

it("moves focus to what comes next once the recipe is saved", async () => {
  const dialog = await save(DRAFT, vi.fn());

  expect(within(dialog).getByRole("button", { name: "Use in a new chat" })).toHaveFocus();
});

it("offers no chat without a place to show it, and leaves focus on Done", async () => {
  const dialog = await save(DRAFT);

  expect(within(dialog).queryByRole("button", { name: "Use in a new chat" })).toBeNull();
  expect(within(dialog).getByRole("button", { name: "Done" })).toHaveFocus();
});

it("offers no chat for a recipe saved turned off or for another use case", async () => {
  for (const saved of [{ ...SAVED, enabled: false }, { ...SAVED, use_case: "image_edit" as const }]) {
    vi.mocked(api.createWorkflowUseCasePreset).mockResolvedValue(saved);
    const dialog = await save(DRAFT, vi.fn());

    expect(within(dialog).queryByRole("button", { name: "Use in a new chat" })).toBeNull();
    expect(within(dialog).getByRole("button", { name: "Done" })).toHaveFocus();
    cleanup();
  }
  expect(api.createChat).not.toHaveBeenCalled();
});
