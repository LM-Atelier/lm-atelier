/** A video made from a picture is offered that picture's shape, in the composer's own settings. */

import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { TurnEditor } from "./TurnEditor";
import type { ComposerDraft } from "./composerPromptSource";
import type { ChatDetail, WorkflowFamily, WorkflowOutputGeometryCapability } from "./types";
import type { TurnEditorState } from "./useTurnEditorState";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, profilesPage: vi.fn().mockResolvedValue([]),
    presetsPage: vi.fn().mockResolvedValue([]), workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(), projectWorkflowSelections: vi.fn(),
    workflowRevisionSchema: vi.fn(), workflowRevisionOutputGeometry: vi.fn(), workflowLoraControls: vi.fn(),
    workflowRevisionSourceFit: vi.fn(), matchWorkflowRevisionOutputGeometryToSource: vi.fn(),
  } };
});

const stamp = "2026-09-30T00:00:00Z";
const chat: ChatDetail = {
  id: "neutral-chat", project_id: null, title: "Frames", archived: false, pinned: false,
  routing_mode: "video", confirm_uncertain_media: false, active_chat_profile_id: null,
  active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: null,
  created_at: stamp, updated_at: stamp, messages: [],
};

const frames: WorkflowFamily = {
  id: "frames", name: "Neutral frames", description: "", use_case: "", tags: [], enabled: true, archived: false,
  compatibility: false,
  variants: [{
    id: "frames-variant", variant_key: "animate", name: "Animate", operation: "image_to_video",
    current_revision_id: "frame-revision", current_revision_version: 1, engine: "comfyui", capabilities: ["video"],
    trusted: true, readiness: "ready", readiness_reason: null,
  }, {
    id: "words-variant", variant_key: "create", name: "Create", operation: "text_to_video",
    current_revision_id: "words-revision", current_revision_version: 1, engine: "comfyui", capabilities: ["video"],
    trusted: true, readiness: "ready", readiness_reason: null,
  }],
  preferences: [{ selector_capability: "video", enabled: true, is_default: true, sort_order: 0 }],
  created_at: stamp, updated_at: stamp,
};

const proof: WorkflowOutputGeometryCapability = {
  version: 1, available: true, reason: null, revision_id: "frame-revision", workflow_id: "frames-workflow",
  artifact_sha256: "a".repeat(64), operation: "image_to_video", engine: "comfyui", size_modes: ["exact", "preset"],
  preset_ids: ["16:9", "9:16"], width: null, height: null, graph_binding_verified: true, request_authorized: false,
};

const clients: QueryClient[] = [];
const ignore = () => {};

function Harness({ initial }: { initial: Partial<TurnEditorState> }) {
  const [draft, setDraft] = useState<ComposerDraft>({ text: "", promptSource: null });
  return <TurnEditor chat={chat} engines={[]} stoppable={false} settings={{}}
    onSettings={ignore} settingsRole="video" onSettingsRole={ignore} presetId={null} onPreset={ignore}
    onMode={ignore} onSend={ignore} onStop={ignore} onStopAndSend={ignore} maxMediaOutputsPerPlan={4}
    draft={draft} onDraftChange={setDraft} initialState={initial} />;
}

function open(initial: Partial<TurnEditorState>) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><Harness initial={initial} /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
}

beforeEach(() => {
  vi.mocked(api.profilesPage).mockResolvedValue([]);
  vi.mocked(api.presetsPage).mockResolvedValue([]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([frames]);
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.workflowRevisionSchema).mockImplementation(async (revisionId) => ({
    revision_id: revisionId, operation: revisionId === "frame-revision" ? "image_to_video" : "text_to_video",
    input_schema_json: { type: "object", properties: {} },
  }) as never);
  vi.mocked(api.workflowRevisionOutputGeometry).mockImplementation(async (revisionId) => ({ ...proof, revision_id: revisionId }));
  vi.mocked(api.workflowLoraControls).mockRejectedValue(new Error("not asked here"));
  vi.mocked(api.workflowRevisionSourceFit).mockRejectedValue(new Error("not asked here"));
});

afterEach(() => {
  cleanup();
  for (const client of clients.splice(0)) client.clear();
  vi.resetAllMocks();
});

it("offers a video the shape of the picture it starts from", async () => {
  vi.mocked(api.matchWorkflowRevisionOutputGeometryToSource).mockRejectedValue(new Error("not a shape it makes"));
  open({ mode: "video", attachments: [{ id: "start-frame", kind: "image", origin: "uploaded" }] });

  fireEvent.click(await screen.findByRole("button", { name: "Match source" }));

  await waitFor(() => expect(api.matchWorkflowRevisionOutputGeometryToSource).toHaveBeenCalledExactlyOnceWith(
    "frame-revision", "start-frame",
  ));
});

it("offers nothing to match when the video starts from no picture", async () => {
  open({ mode: "video", attachments: [] });

  expect(await screen.findByRole("group", { name: "Output aspect ratio" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Match source" })).toBeNull();
});
