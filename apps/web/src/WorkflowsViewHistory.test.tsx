import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";
import type { Workflow, WorkflowRevision } from "./types";

vi.mock("./api", () => ({ api: {
  workflows: vi.fn(), workflowFamilies: vi.fn(), restoreWorkflowRevision: vi.fn(),
  updateWorkflow: vi.fn(), cloneWorkflow: vi.fn(),
} }));
vi.mock("./CustomNodesPanel", () => ({ CustomNodesPanel: () => null }));
vi.mock("./RegistryInstallsPanel", () => ({ RegistryInstallsPanel: () => null }));
vi.mock("./WorkflowRevisionReviewPanel", () => ({ WorkflowRevisionReviewPanel: () => null }));

function revision(version: number, createdAt: string): WorkflowRevision {
  return { id: "revision-" + version, workflow_id: "landscape", version, engine: "comfyui",
    engine_version: null, ui_graph_json: {}, api_graph_json: { version },
    input_schema_json: {}, dependencies_json: {}, trusted: false, created_at: createdAt };
}
function workflow(): Workflow {
  return { id: "landscape", name: "Landscape study", description: "Neutral fixture",
    operation: "text_to_image", current_revision_id: "revision-2", revisions: [
      revision(1, "2026-08-01T10:00:00Z"), revision(3, "2026-08-03T10:00:00Z"),
      revision(2, "2026-08-02T10:00:00Z"),
    ] };
}
const clients: QueryClient[] = [];
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.workflows).mockResolvedValue([workflow()]);
  vi.mocked(api.workflowFamilies).mockResolvedValue([]);
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });
async function openHistory() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><WorkflowsView /></QueryClientProvider>);
  fireEvent.click(await screen.findByText("Landscape study"));
  fireEvent.click(screen.getByRole("button", { name: "Show revision history" }));
  return within(screen.getByRole("region", { name: "Revision history" }));
}

describe("workflow revision history", () => {
  it("interprets zone-free stored dates as UTC and preserves explicit offsets", async () => {
    const value = workflow();
    value.revisions = [
      revision(1, "2026-08-01T00:30:00.123456"),
      revision(2, "2026-08-01T00:30:00.123456Z"),
      revision(3, "2026-08-01T02:30:00.123456+02:00"),
    ];
    vi.mocked(api.workflows).mockResolvedValue([value]);
    const history = await openHistory();
    const dates = history.getAllByRole("listitem").map((row) => row.querySelector("time"));
    const expected = new Date("2026-08-01T00:30:00.123456Z").toLocaleString(undefined, {
      year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    });
    expect(dates.map((date) => date?.textContent)).toEqual([expected, expected, expected]);
    expect(dates.map((date) => date?.dateTime)).toEqual([
      "2026-08-01T02:30:00.123456+02:00", "2026-08-01T00:30:00.123456Z",
      "2026-08-01T00:30:00.123456Z",
    ]);
  });

  it("lists newest versions first with recorded creation dates and the actual current revision", async () => {
    const history = await openHistory();
    const rows = history.getAllByRole("listitem");
    expect(rows.map((row) => within(row).getByRole("button").textContent)).toEqual([
      "Inspect v3", "Inspect v2", "Inspect v1",
    ]);
    expect(within(rows[1]).getByText("Current")).toBeInTheDocument();
    expect(within(rows[0]).queryByText("Current")).not.toBeInTheDocument();
    expect(rows.map((row) => row.querySelector("time")?.dateTime)).toEqual([
      "2026-08-03T10:00:00Z", "2026-08-02T10:00:00Z", "2026-08-01T10:00:00Z",
    ]);
    expect(within(rows[1]).getByRole("button")).toHaveAttribute("aria-pressed", "true");
  });

  it("inspects a historical revision without restoring it or changing the current revision", async () => {
    const history = await openHistory();
    fireEvent.click(history.getByRole("button", { name: "Inspect v1" }));
    expect(screen.getByLabelText("Revision")).toHaveValue("revision-1");
    expect(history.getByRole("button", { name: "Inspect v1" })).toHaveAttribute("aria-pressed", "true");
    expect(history.getByRole("button", { name: "Inspect v2" })).toHaveAttribute("aria-pressed", "false");
    expect(within(history.getAllByRole("listitem")[1]).getByText("Current")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Restore as new revision" })).toBeInTheDocument();
    expect(api.restoreWorkflowRevision).not.toHaveBeenCalled();
    expect(api.updateWorkflow).not.toHaveBeenCalled();
    expect(api.cloneWorkflow).not.toHaveBeenCalled();
  });

  it("resets the history disclosure and selection when switching workflows", async () => {
    const second = workflow();
    second.id = "portrait"; second.name = "Portrait study";
    second.current_revision_id = "portrait-revision";
    second.revisions = [{ ...revision(8, "2026-08-04T10:00:00Z"), id: "portrait-revision", workflow_id: "portrait" }];
    vi.mocked(api.workflows).mockResolvedValue([workflow(), second]);
    const history = await openHistory();
    fireEvent.click(history.getByRole("button", { name: "Inspect v1" }));
    fireEvent.click(screen.getByText("Portrait study"));
    expect(screen.getByRole("button", { name: "Show revision history" })).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(screen.getByRole("button", { name: "Show revision history" }));
    const selected = within(screen.getByRole("region", { name: "Revision history" }));
    expect(selected.getAllByRole("listitem")).toHaveLength(1);
    expect(selected.getByRole("button", { name: "Inspect v8" })).toHaveAttribute("aria-pressed", "true");
    expect(selected.queryByRole("button", { name: "Inspect v1" })).not.toBeInTheDocument();
  });

  it("keeps a revision inspectable when its creation date is unavailable", async () => {
    const value = workflow(); value.revisions[0].created_at = "unknown";
    vi.mocked(api.workflows).mockResolvedValue([value]);
    const history = await openHistory();
    expect(history.getByText("Creation date unavailable")).toBeInTheDocument();
    fireEvent.click(history.getByRole("button", { name: "Inspect v1" }));
    expect(screen.getByLabelText("Revision")).toHaveValue("revision-1");
  });
});
