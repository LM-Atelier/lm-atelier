import { useState, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import type { ComposerDraft } from "./composerPromptSource";
import { TurnEditor, type TurnEditorProps, type TurnEditorSubmission } from "./TurnEditor";
import type { ChatDetail } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api,
    profilesPage: vi.fn().mockResolvedValue([]),
    presetsPage: vi.fn(async (options) => (await import("./test/modelLibraryPageFixtures")).presetPages({ presets: async () => [] }, options)),
    workflowFamilies: vi.fn().mockResolvedValue([]),
    chatWorkflowSelections: vi.fn().mockResolvedValue([]),
    projectWorkflowSelections: vi.fn().mockResolvedValue([]),
    classifyDraft: vi.fn(), references: vi.fn(),
  } };
});

const stamp = "2026-09-06T12:00:00Z";
const chat: ChatDetail = {
  id: "garden-purposes", project_id: null, title: "Garden pictures", archived: false, pinned: false,
  routing_mode: "image", confirm_uncertain_media: false, active_chat_profile_id: null,
  active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: null,
  created_at: stamp, updated_at: stamp, messages: [],
};
const ignore = () => {};
const clients: QueryClient[] = [];

function Harness({ send = ignore, accept }: {
  send?: TurnEditorProps["onSend"];
  accept?: (submission: TurnEditorSubmission) => Promise<unknown>;
}) {
  const [draft, setDraft] = useState<ComposerDraft>({ text: "Paint a garden", promptSource: null });
  return <TurnEditor chat={chat} engines={[]} stoppable={false} settings={{}} onSettings={ignore}
    settingsRole="image" onSettingsRole={ignore} presetId={null} onPreset={ignore} onMode={ignore}
    onSend={send} onStop={ignore} onStopAndSend={send} maxMediaOutputsPerPlan={4}
    draft={draft} onDraftChange={setDraft} contextMessages={[]}
    workflowControl={<span>Garden workflow</span>} workflowSchemaOverride={null}
    onAccept={accept} initialState={{ requestId: "garden-request", mode: "image", attachments: [
      { id: "layout", kind: "image", origin: "uploaded" },
      { id: "canvas", kind: "image", origin: "uploaded" },
      { id: "palette", kind: "image", origin: "uploaded" },
    ] }} submitLabel={accept ? "Queue edited version" : "Send"} />;
}

function mount(element: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  return render(<QueryClientProvider client={client}>{element}</QueryClientProvider>);
}

beforeEach(() => {
  vi.mocked(api.profilesPage).mockResolvedValue([]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue([]);
});
afterEach(() => {
  cleanup();
  for (const client of clients.splice(0)) client.clear();
});

it("sends the chosen second canvas and both references without rearranging attachments", () => {
  const send = vi.fn<TurnEditorProps["onSend"]>();
  mount(<Harness send={send} />);
  fireEvent.change(screen.getAllByRole("combobox", { name: "Picture purpose" })[1], { target: { value: "edit_source" } });
  expect(screen.getAllByRole("combobox", { name: "Picture purpose" }).map((item) => (item as HTMLSelectElement).value))
    .toEqual(["reference", "edit_source", "reference"]);
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(send).toHaveBeenCalledWith("Paint a garden", "image", ["layout", "canvas", "palette"], {}, [],
    undefined, undefined, undefined, ["reference", "edit_source", "reference"]);
});

it("returns the whole attachment selection to automatic when that choice is restored", () => {
  const send = vi.fn<TurnEditorProps["onSend"]>();
  mount(<Harness send={send} />);
  fireEvent.change(screen.getAllByRole("combobox", { name: "Picture purpose" })[1], { target: { value: "edit_source" } });
  fireEvent.change(screen.getAllByRole("combobox", { name: "Picture purpose" })[0], { target: { value: "automatic" } });
  expect(screen.getAllByRole("combobox", { name: "Picture purpose" }).map((item) => (item as HTMLSelectElement).value))
    .toEqual(["automatic", "automatic", "automatic"]);
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
  expect(send).toHaveBeenCalledWith("Paint a garden", "image", ["layout", "canvas", "palette"], {}, [],
    undefined, undefined);
});

it("retains selected purposes and request identity when an edited version is refused", async () => {
  const accept = vi.fn<(submission: TurnEditorSubmission) => Promise<unknown>>().mockRejectedValue(new Error("Queue full"));
  mount(<Harness accept={accept} />);
  fireEvent.change(screen.getAllByRole("combobox", { name: "Picture purpose" })[1], { target: { value: "edit_source" } });
  fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Queue full");
  expect(accept.mock.calls[0][0]).toMatchObject({ requestId: "garden-request",
    inputArtifactIds: ["layout", "canvas", "palette"], inputImageRoles: ["reference", "edit_source", "reference"] });
  fireEvent.click(screen.getByRole("button", { name: "Queue edited version" }));
  await waitFor(() => expect(accept).toHaveBeenCalledTimes(2));
  expect(accept.mock.calls[1][0]).toEqual(accept.mock.calls[0][0]);
});
