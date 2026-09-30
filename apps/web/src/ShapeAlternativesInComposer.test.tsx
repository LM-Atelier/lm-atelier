/** From a workflow that sets its own size to one that takes a shape, in the composer's own settings. */

import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { TurnEditor } from "./TurnEditor";
import type { ComposerDraft } from "./composerPromptSource";
import type { ChatDetail, WorkflowFamily, WorkflowOutputGeometryCapability, WorkflowSelection } from "./types";

vi.mock("./api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./api")>();
  return { ...actual, api: { ...actual.api,
    workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(), projectWorkflowSelections: vi.fn(),
    setChatWorkflowSelection: vi.fn(), workflowRevisionSchema: vi.fn(), workflowRevisionOutputGeometry: vi.fn(),
    workflowLoraControls: vi.fn(), workflowRevisionSourceFit: vi.fn(),
  } };
});

const stamp = "2026-09-30T00:00:00Z";
const chat: ChatDetail = {
  id: "neutral-chat", project_id: null, title: "Shapes", archived: false, pinned: false,
  routing_mode: "image", confirm_uncertain_media: false, active_chat_profile_id: null,
  active_image_profile_id: null, active_video_profile_id: null, active_head_message_id: null,
  created_at: stamp, updated_at: stamp, messages: [],
};

function family(id: string, sortOrder: number): WorkflowFamily {
  return {
    id, name: `Neutral ${id}`, description: "", use_case: "", tags: [], enabled: true, archived: false, compatibility: false,
    variants: [{
      id: `${id}-variant`, variant_key: "create", name: "Create", operation: "text_to_image",
      current_revision_id: `${id}-revision`, current_revision_version: 1, engine: "comfyui", capabilities: ["image"],
      trusted: true, readiness: "ready", readiness_reason: null,
    }],
    preferences: [{ selector_capability: "image", enabled: true, is_default: false, sort_order: sortOrder }],
    created_at: stamp, updated_at: stamp,
  };
}

function proof(revisionId: string): WorkflowOutputGeometryCapability {
  const shaped = revisionId === "shaped-revision";
  return {
    version: 1, available: shaped, reason: shaped ? null : "unsupported_workflow_geometry",
    revision_id: revisionId, workflow_id: null, artifact_sha256: null, operation: shaped ? "text_to_image" : null,
    engine: shaped ? "comfyui" : null, size_modes: shaped ? ["exact", "preset"] : [],
    preset_ids: shaped ? ["1:1", "16:9"] : [], width: null, height: null,
    graph_binding_verified: shaped, request_authorized: false,
  };
}

let selections: WorkflowSelection[];
const clients: QueryClient[] = [];
const ignore = () => {};

function Harness() {
  const [draft, setDraft] = useState<ComposerDraft>({ text: "", promptSource: null });
  return <TurnEditor chat={chat} engines={[]} profiles={[]} presets={[]} stoppable={false} settings={{}}
    onSettings={ignore} settingsRole="image" onSettingsRole={ignore} presetId={null} onPreset={ignore}
    onMode={ignore} onSend={ignore} onStop={ignore} onStopAndSend={ignore} maxMediaOutputsPerPlan={4}
    draft={draft} onDraftChange={setDraft} />;
}

beforeEach(() => {
  selections = [{
    selector_capability: "image", mode: "family", workflow_family_id: "sized",
    workflow_revision_id: null, legacy_profile_id: null,
  }];
  vi.mocked(api.workflowFamilies).mockResolvedValue([family("sized", 0), family("shaped", 1)]);
  vi.mocked(api.chatWorkflowSelections).mockImplementation(async () => selections);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue([]);
  vi.mocked(api.setChatWorkflowSelection).mockImplementation(async (_chat, capability, choice) => {
    const next: WorkflowSelection = {
      selector_capability: capability, mode: choice.mode,
      workflow_family_id: choice.mode === "family" ? choice.workflow_family_id : null,
      workflow_revision_id: null, legacy_profile_id: null,
    };
    selections = [next];
    return next;
  });
  vi.mocked(api.workflowRevisionSchema).mockImplementation(async (revisionId) => ({
    revision_id: revisionId, operation: "text_to_image", input_schema_json: { type: "object", properties: {} },
  }) as never);
  vi.mocked(api.workflowRevisionOutputGeometry).mockImplementation(async (revisionId) => proof(revisionId));
  vi.mocked(api.workflowLoraControls).mockRejectedValue(new Error("not asked here"));
});

afterEach(() => {
  cleanup();
  for (const client of clients.splice(0)) client.clear();
  vi.resetAllMocks();
});

it("offers the chat a workflow that takes a shape, and shows its shapes once chosen", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><Harness /></QueryClientProvider>);

  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  expect(await screen.findByText("This workflow sets the picture size itself.")).toBeVisible();
  fireEvent.click(await screen.findByRole("button", { name: "Use Neutral shaped" }));

  await waitFor(() => expect(api.setChatWorkflowSelection).toHaveBeenCalledExactlyOnceWith(
    "neutral-chat", "image", { mode: "family", workflow_family_id: "shaped" },
  ));
  expect(await screen.findByRole("group", { name: "Output aspect ratio" })).toBeVisible();
  expect(screen.queryByText("This workflow sets the picture size itself.")).toBeNull();
});
