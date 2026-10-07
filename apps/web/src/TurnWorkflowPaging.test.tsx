import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { TurnEditor } from "./TurnEditor";
import { ShapeAlternativeList } from "./ShapeAlternativeList";
import { usePagedShapeAlternatives } from "./usePagedShapeAlternatives";
import { familyFixturePage } from "./workflowFamilyReadFixtures";
import type { ChatDetail, EngineCapabilities, WorkflowFamily, WorkflowSelection } from "./types";

vi.mock("./api", async original => {
  const actual = await original<typeof import("./api")>();
  return { ...actual, api: { ...actual.api, workflowFamilies: vi.fn(), chatWorkflowSelections: vi.fn(),
    projectWorkflowSelections: vi.fn(), workflowRevisionSchema: vi.fn(), workflowLoraControls: vi.fn(),
    workflowRevisionOutputGeometry: vi.fn(), references: vi.fn(), modelAssets: vi.fn(),
    setChatWorkflowSelection: vi.fn(), profilesPage: vi.fn(), presetsPage: vi.fn() } };
});
const stamp = "2026-09-30T00:00:00Z";
const chat: ChatDetail = { id: "chat", title: "Neutral conversation", project_id: null,
  archived: false, pinned: false, routing_mode: "image", confirm_uncertain_media: false,
  active_chat_profile_id: null, active_image_profile_id: null, active_video_profile_id: null,
  active_head_message_id: null, created_at: stamp, updated_at: stamp, messages: [] };
const engine: EngineCapabilities = { engine: "mock", version: "1", roles: ["image"],
  operations: ["text_to_image"], formats: [], devices: [], streaming: false, tool_calling: false,
  healthy: true, details: {}, settings: [{ key: "steps", label: "Steps", type: "integer", default: 20,
    minimum: 1, maximum: 100, step: 1, choices: [], scope: "request", visibility: "basic",
    restart_required: false, available: true, unavailable_reason: null, help: "" }] };
const ignore = () => {};
function family(id: string, count = 1, isDefault = false): WorkflowFamily {
  return { id, name: id, description: "", use_case: "", tags: [], enabled: true, archived: false,
    compatibility: false, created_at: stamp, updated_at: stamp,
    preferences: [{ selector_capability: "image", enabled: true, is_default: isDefault, sort_order: 0 }],
    variants: Array.from({ length: count }, (_, index) => ({ id: `${id}-${index}`, name: `Variant ${index}`,
      variant_key: String(index), operation: "text_to_image", current_revision_id: `${id}-revision-${index}`,
      current_revision_version: 1, engine: "mock", capabilities: ["image"], trusted: true,
      readiness: "ready", readiness_reason: null })) };
}
const chosen = "Selected elsewhere";
const selection: WorkflowSelection = { selector_capability: "image", mode: "family", workflow_family_id: chosen,
  workflow_revision_id: null, legacy_profile_id: null };
let rows: WorkflowFamily[];
beforeEach(() => {
  vi.mocked(api.profilesPage).mockResolvedValue([]);
  vi.mocked(api.presetsPage).mockResolvedValue([]);
  rows = [...Array.from({ length: 50 }, (_, index) => family(`Page ${String(index).padStart(2, "0")}`)), family(chosen, 1, true)];
  vi.mocked(api.workflowFamilies).mockImplementation(async (capability, archived, _dependencies, options) =>
    familyFixturePage(rows, archived, options ?? { limit: 50 }, capability));
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([selection]);
  vi.mocked(api.projectWorkflowSelections).mockResolvedValue([selection]);
  vi.mocked(api.workflowLoraControls).mockResolvedValue({ version: 1, override_contract_version: 1,
    strength_bounds: { minimum: -4, maximum: 4 }, override_target: null, revision_scope_sha256: "a".repeat(64),
    api_graph_sha256: "a".repeat(64), dependency_contract_sha256: "b".repeat(64), activation_binding_sha256: null,
    ordering_authority: "presentation_only", evidence_gaps: [], base_model_family: null,
    accepts_added_loras: false, slots: [] });
  vi.mocked(api.references).mockResolvedValue({ items: [], total: 0, limit: 50, offset: 0 });
  vi.mocked(api.modelAssets).mockResolvedValue([]);
  vi.mocked(api.workflowRevisionSchema).mockImplementation(async id => ({ revision_id: id, workflow_id: "workflow",
    operation: "text_to_image", input_schema_json: { properties: { steps: { type: "integer", title: "Exact steps", default: 9 } } } }));
  vi.mocked(api.workflowRevisionOutputGeometry).mockImplementation(async id => ({ version: 1, available: true,
    reason: null, revision_id: id, workflow_id: "workflow", operation: "text_to_image", engine: "comfyui",
    artifact_sha256: "a".repeat(64), size_modes: ["preset"], preset_ids: ["1:1"], width: null, height: null,
    graph_binding_verified: true, request_authorized: false }));
});
afterEach(() => { cleanup(); vi.resetAllMocks(); });
function Composer({ project = false }: { project?: boolean }) {
  const [settings, setSettings] = useState<Record<string, unknown>>({});
  return <TurnEditor chat={chat} project={project ? { id: "project", name: "Project", description: "",
    instructions: "", archived: false, pinned: false, image_workflow_revision_id: null,
    video_workflow_revision_id: null, created_at: stamp, updated_at: stamp } : undefined}
    engines={[engine]} stoppable={false} settings={settings} onSettings={setSettings}
    settingsRole="image" onSettingsRole={ignore} presetId={null} onPreset={ignore} onMode={ignore}
    onSend={ignore} onStop={ignore} onStopAndSend={ignore} maxMediaOutputsPerPlan={4}
    draft={{ text: "A blue cube", promptSource: null }} onDraftChange={ignore} />;
}
function mount(element: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>{element}</QueryClientProvider>);
  return client;
}
it.each(["chat", "project", "default"])("resolves the off-page %s workflow for composer settings", async scope => {
  if (scope !== "chat") vi.mocked(api.chatWorkflowSelections).mockResolvedValue([]);
  mount(<Composer project={scope === "project"} />);
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  await screen.findByRole("spinbutton", { name: "Exact steps" });
  expect(api.workflowRevisionSchema).toHaveBeenCalledWith(chosen + "-revision-0", expect.any(AbortSignal));
  expect(api.workflowFamilies).toHaveBeenCalledWith("image", false, false, expect.objectContaining({
    limit: 1, variantLimit: 2, operation: "text_to_image", readiness: "ready",
    ...(scope === "default" ? { defaultsOnly: true } : { familyIds: [chosen] }),
  }), expect.any(AbortSignal));
});
it("keeps automatic composer choices unresolved instead of borrowing the default workflow", async () => {
  vi.mocked(api.chatWorkflowSelections).mockResolvedValue([{ ...selection, mode: "automatic", workflow_family_id: null }]);
  mount(<Composer />);
  await waitFor(() => expect(api.chatWorkflowSelections).toHaveBeenCalled());
  fireEvent.click(screen.getByRole("button", { name: "Turn settings" }));
  expect(api.workflowRevisionSchema).not.toHaveBeenCalled();
  expect(screen.queryByRole("spinbutton", { name: "Exact steps" })).not.toBeInTheDocument();
});
function Alternatives() {
  const choices = usePagedShapeAlternatives({ chatId: chat.id, capability: "image", hasAttachments: false,
    currentRevisionId: "current", enabled: true });
  return choices ? <ShapeAlternativeList alternatives={choices} /> : null;
}
it("finds shape alternatives beyond the first page and keeps browsing available on read errors", async () => {
  let fail = true;
  const normal = vi.mocked(api.workflowFamilies).getMockImplementation()!;
  vi.mocked(api.workflowFamilies).mockImplementation(async (capability, archived, dependencies, options, signal) => {
    if (options?.offset === 10 && fail) throw new Error("Later alternatives unavailable");
    return normal(capability, archived, dependencies, options, signal);
  });
  mount(<Alternatives />);
  await screen.findByRole("button", { name: "Use Page 00" });
  fireEvent.click(screen.getByRole("button", { name: "Load more alternative workflows" }));
  await screen.findByText("Later alternatives unavailable");
  fail = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry alternative workflows" }));
  await screen.findByRole("button", { name: "Use Page 19" });
  fireEvent.change(screen.getByRole("searchbox"), { target: { value: chosen } });
  await screen.findByRole("button", { name: "Use " + chosen });
  expect(api.setChatWorkflowSelection).not.toHaveBeenCalled();
});
it("does not offer an ambiguous family when two ready variants fit the operation", async () => {
  rows = [family("Ambiguous", 3)];
  mount(<Alternatives />);
  await waitFor(() => expect(api.workflowFamilies).toHaveBeenCalledWith("image", false, false,
    expect.objectContaining({ variantLimit: 2, readiness: "ready" }), expect.any(AbortSignal)));
  expect(screen.queryByRole("button", { name: "Use Ambiguous" })).not.toBeInTheDocument();
  expect(screen.getByRole("searchbox")).toBeInTheDocument();
  expect(api.workflowRevisionOutputGeometry).not.toHaveBeenCalled();
});
