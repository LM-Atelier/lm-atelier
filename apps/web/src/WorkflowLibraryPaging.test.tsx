import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { WorkflowsView } from "./WorkflowsView";
import { WorkflowFamilyVariants } from "./WorkflowFamilyVariants";
import { useSelectedWorkflowFamily } from "./useWorkflowLibraryReads";
import { familyFixturePage, mockWorkflowFamilyPages } from "./workflowFamilyReadFixtures";
import type { WorkflowFamily } from "./types";

vi.mock("./api", () => ({ api: {
  workflowFamilies: vi.fn(), workflowSummaries: vi.fn(), workflowFamilyOperations: vi.fn(), workflow: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));
afterEach(() => { cleanup(); vi.resetAllMocks(); });
beforeEach(() => {
  vi.mocked(api.workflowSummaries).mockResolvedValue([]);
  vi.mocked(api.workflow).mockImplementation(async id => ({
    id, name: id, description: "", operation: "text_to_image", family_id: "family-0",
    current_revision_id: `revision-${id}`, revisions: [{ id: `revision-${id}`, workflow_id: id,
      version: 1, engine: "mock", engine_version: null, api_graph_json: {}, ui_graph_json: {},
      dependencies_json: {}, input_schema_json: {}, trusted: true, created_at: "2026-09-30" }],
  }));
});

function family(index: number, count = 1): WorkflowFamily {
  return {
    id: `family-${index}`, name: `Family ${String(index).padStart(3, "0")}`, description: "", use_case: "",
    tags: [], enabled: true, archived: false, compatibility: false,
    created_at: "2026-09-30", updated_at: "2026-09-30", preferences: [],
    variants: Array.from({ length: count }, (_, variant) => ({ id: `workflow-${index}-${variant}`,
      name: `Variant ${index}/${variant}`, variant_key: `variant-${variant}`, operation: "text_to_image",
      current_revision_id: `revision-${index}-${variant}`, current_revision_version: 1,
      engine: "mock", capabilities: ["image"], trusted: true, readiness: "ready", readiness_reason: null })),
  };
}
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    {children}
  </QueryClientProvider>;
}
function open() {
  render(<WorkflowsView />, { wrapper });
  return vi.mocked(api.workflow);
}

it("finds off-page family metadata and operation choices without fetching all workflow summaries", async () => {
  const rows = Array.from({ length: 12 }, (_, index) => family(index));
  rows[11].tags = ["Outside the first page"];
  rows[11].variants[0].operation = "text_to_video";
  mockWorkflowFamilyPages(rows);
  open();
  await screen.findByRole("heading", { name: "Family 000" });
  expect(screen.queryByRole("heading", { name: "Family 011" })).not.toBeInTheDocument();
  expect(await screen.findByRole("option", { name: "Text to video" })).toBeInTheDocument();
  fireEvent.change(screen.getByRole("searchbox"), { target: { value: "Outside the first page" } });
  await screen.findByRole("heading", { name: "Family 011" });
  expect(api.workflowFamilies).toHaveBeenCalledWith(undefined, false, true,
    expect.objectContaining({ limit: 10, variantLimit: 5, search: "Outside the first page" }), expect.any(AbortSignal));
  expect(api.workflowSummaries).toHaveBeenCalledWith(
    expect.objectContaining({ limit: 20, ungroupedOnly: true }), expect.any(AbortSignal));
});

it("loads later variants and keeps them selectable without a separate summary page", async () => {
  mockWorkflowFamilyPages([family(0, 7)]);
  const choose = open();
  const region = within(await screen.findByRole("region", { name: "Family 000" }));
  expect(region.queryByText("Variant 0/6")).not.toBeInTheDocument();
  fireEvent.click(region.getByRole("button", { name: "Load more Family 000 variants" }));
  fireEvent.click(await region.findByRole("button", { name: /Variant 0\/6/ }));
  await waitFor(() => expect(choose).toHaveBeenCalledWith("workflow-0-6", expect.any(AbortSignal)));
  expect(api.workflowFamilies).toHaveBeenCalledWith(undefined, true, false,
    expect.objectContaining({ familyIds: ["family-0"], variantLimit: 5, variantOffset: 5 }), expect.any(AbortSignal));
});

it("retries a failed family page without discarding the current families", async () => {
  const rows = Array.from({ length: 12 }, (_, index) => family(index));
  mockWorkflowFamilyPages(rows);
  let failed = true;
  vi.mocked(api.workflowFamilies).mockImplementation(async (_cap, archived, _deps, options) => {
    if (options?.offset === 10 && failed) throw new Error("Family page unavailable");
    return familyFixturePage(rows, archived, options);
  });
  open();
  await screen.findByRole("heading", { name: "Family 000" });
  fireEvent.click(screen.getByRole("button", { name: "Load more workflow families" }));
  await screen.findByText("Family page unavailable");
  expect(screen.getByRole("heading", { name: "Family 000" })).toBeInTheDocument();
  failed = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry workflow families" }));
  await screen.findByRole("heading", { name: "Family 011" });
});

it("resolves the selected variant independently of family and variant pages", async () => {
  const rows = Array.from({ length: 12 }, (_, index) => family(index, 7));
  mockWorkflowFamilyPages(rows);
  const { result } = renderHook(() => useSelectedWorkflowFamily("workflow-11-6"), { wrapper });
  await waitFor(() => expect(result.current.data?.id).toBe("family-11"));
  expect(result.current.data?.variants.map(variant => variant.id)).toEqual(["workflow-11-6"]);
  expect(api.workflowFamilies).toHaveBeenCalledWith(undefined, true, false,
    { workflowIds: ["workflow-11-6"], limit: 1, variantLimit: 1 }, expect.any(AbortSignal));
});

it("pages the operation-variant disclosure beyond the selected variant", async () => {
  const full = family(0, 7);
  mockWorkflowFamilyPages([full]);
  render(<WorkflowFamilyVariants family={{ ...full, variants: [full.variants[6]] }} />, { wrapper });
  expect(api.workflowFamilies).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Show operation variants" }));
  await screen.findByText("Variant 0/0");
  expect(screen.queryByText("Variant 0/6")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load more Family 000 operation variants" }));
  await screen.findByText("Variant 0/6");
  expect(screen.getAllByRole("listitem")).toHaveLength(7);
});
