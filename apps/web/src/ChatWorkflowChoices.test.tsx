import { useState, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { TurnEditor } from "./TurnEditor";
import type { ComposerDraft } from "./composerPromptSource";
import type { ChatDetail, RoutingMode, WorkflowSelection } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api,
    workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(),
    projectWorkflowSelections: vi.fn(), setChatWorkflowSelection: vi.fn(),
  } };
});
const stamp = "2026-09-09T00:00:00Z";
const chat: ChatDetail = {
  id: "neutral-chat", project_id: null, title: "Workflow choices", archived: false, pinned: false,
  routing_mode: "auto", confirm_uncertain_media: false, active_chat_profile_id: null,
  active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: null,
  created_at: stamp, updated_at: stamp, messages: [],
};
const ignore = () => {};
let selections: WorkflowSelection[];
const clients: QueryClient[] = [];
function Harness({ workflowControl }: { workflowControl?: ReactNode }) {
  const [mode, setMode] = useState<RoutingMode>("auto");
  const [draft, setDraft] = useState<ComposerDraft>({ text: "", promptSource: null });
  return <TurnEditor chat={{ ...chat, routing_mode: mode }} engines={[]} profiles={[]} workflows={[]}
    presets={[]} stoppable={false} settings={{}} onSettings={ignore} settingsRole="chat"
    onSettingsRole={ignore} presetId={null} onPreset={ignore} onMode={setMode} onSend={ignore}
    onStop={ignore} onStopAndSend={ignore} maxMediaOutputsPerPlan={4}
    draft={draft} onDraftChange={setDraft} workflowControl={workflowControl} workflowSchemaOverride={null} />;
}
function show(workflowControl?: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  return render(<QueryClientProvider client={client}><Harness workflowControl={workflowControl} /></QueryClientProvider>);
}
beforeEach(() => {
  selections = [
    { selector_capability: "chat", mode: "automatic", workflow_family_id: null, workflow_revision_id: null, legacy_profile_id: null },
    { selector_capability: "image", mode: "family", workflow_family_id: "missing-image-family", workflow_revision_id: null, legacy_profile_id: null },
    { selector_capability: "video", mode: "revision", workflow_family_id: null, workflow_revision_id: "retained-video", legacy_profile_id: null },
  ];
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.chatWorkflowSelections).mockImplementation(async () => selections);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.setChatWorkflowSelection).mockImplementation(async (_id, capability, choice) => {
    const next: WorkflowSelection = { selector_capability: capability, mode: choice.mode,
      workflow_family_id: choice.mode === "family" ? choice.workflow_family_id : null,
      workflow_revision_id: null, legacy_profile_id: null };
    selections = selections.map((value) => value.selector_capability === capability ? next : value);
    return next;
  });
});
afterEach(() => { cleanup(); for (const client of clients.splice(0)) client.clear(); vi.resetAllMocks(); });

it("shows all three stored workflow choices in routing Auto and preserves the other types when one changes", async () => {
  show();
  const text = await screen.findByRole("combobox", { name: "Text workflow" });
  const image = screen.getByRole("combobox", { name: "Image workflow" });
  const video = screen.getByRole("combobox", { name: "Video workflow" });
  await waitFor(() => expect(text).toHaveValue("automatic"));
  expect(image).toHaveValue("missing-image-family");
  expect(video).toHaveValue("compatibility:revision");
  expect(screen.getByRole("option", { name: "Selected workflow (unavailable)" })).toBeDisabled();
  expect(screen.getByRole("combobox", { name: "Generation mode" })).toHaveValue("auto");
  expect(api.setChatWorkflowSelection).not.toHaveBeenCalled();
  fireEvent.change(image, { target: { value: "default" } });
  await waitFor(() => expect(image).toHaveValue("default"));
  expect(api.setChatWorkflowSelection).toHaveBeenCalledExactlyOnceWith(chat.id, "image", { mode: "default" });
  expect(text).toHaveValue("automatic");
  expect(video).toHaveValue("compatibility:revision");
  fireEvent.change(screen.getByRole("combobox", { name: "Generation mode" }), { target: { value: "video" } });
  expect(text).toBeVisible();
  expect(image).toHaveValue("default");
  expect(video).toHaveValue("compatibility:revision");
  expect(api.setChatWorkflowSelection).toHaveBeenCalledTimes(1);
});

it("keeps a failed capability read explicit while the other workflow controls remain available", async () => {
  vi.mocked(api.workflowFamilies).mockImplementation(async (capability) => {
    if (capability === "image") throw new Error("Image choices unavailable");
    return [];
  });
  show();
  await screen.findByText("Image choices unavailable");
  const image = screen.getByRole("combobox", { name: "Image workflow" });
  expect(image).toBeDisabled();
  expect(image).toHaveValue("");
  expect(screen.getByRole("combobox", { name: "Text workflow" })).toHaveValue("automatic");
  expect(screen.getByRole("combobox", { name: "Video workflow" })).toHaveValue("compatibility:revision");
  expect(api.setChatWorkflowSelection).not.toHaveBeenCalled();
});

it("retains the caller-supplied exact workflow control for prior-turn editing", () => {
  show(<span>Exact edited workflow</span>);
  expect(screen.getByText("Exact edited workflow")).toBeVisible();
  expect(screen.queryByRole("combobox", { name: /^(Text|Image|Video) workflow$/ })).toBeNull();
  expect(api.chatWorkflowSelections).not.toHaveBeenCalled();
});
