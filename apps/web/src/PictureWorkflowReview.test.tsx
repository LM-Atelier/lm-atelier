import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { PictureFileSettings } from "./PictureFileSettings";
import { api } from "./api";
import type { Workflow, WorkflowPackageAnalysis } from "./types";

vi.mock("./api", () => ({
  api: {
    pictureSettings: vi.fn(),
    pictureWorkflow: vi.fn(),
    analyzeWorkflowPackage: vi.fn(),
    ensureWorkflowPackageDraft: vi.fn(),
    importWorkflowPackage: vi.fn(),
    prepareWorkflowPackage: vi.fn(),
  },
}));
afterEach(() => { cleanup(); vi.resetAllMocks(); });

const artifactId = `sha256:${"b".repeat(64)}`;
const graph = { version: 0.4, nodes: [{ id: 1, type: "Source" }], links: [] };
const analysis: WorkflowPackageAnalysis = {
  format_version: "0.4",
  frontend_version: null,
  node_count: 1,
  link_count: 0,
  subgraph_count: 0,
  operation_guess: "image",
  truncated: false,
  required_node_types: ["Source"],
  frontend_node_types: [],
  missing_node_types: [],
  missing_nodes: [],
  custom_packages: [],
  asset_references: [],
  issues: [],
  ready: true,
  runtime_nodes_available: true,
  dependencies_resolved: true,
  node_inventory_available: true,
  source_candidates: [],
};

function settings(ignored: { name: string; reason: string }[]) {
  return { dialect: "none", claims: [], ignored, warnings: ["no_settings_found"] };
}

async function renderSettings(ignored = [{ name: "workflow", reason: "workflow_graph" }]) {
  vi.mocked(api.pictureSettings).mockResolvedValue(settings(ignored));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <PictureFileSettings artifactId={artifactId} pictureName="Harbor at dusk" />
  </QueryClientProvider>);
  const details = screen.getByText("Settings in the file", { selector: "summary" }).parentElement as HTMLDetailsElement;
  details.open = true;
  fireEvent(details, new Event("toggle"));
  await screen.findByText(/Nothing named in it is fetched/);
}

describe("reviewing the workflow a picture's file carries", () => {
  it("is offered only for a workflow an editor saved", async () => {
    await renderSettings([{ name: "prompt", reason: "workflow_graph" }]);

    expect(screen.queryByRole("button", { name: "Review this picture's workflow" })).toBeNull();
    cleanup();

    await renderSettings();

    expect(screen.getByRole("button", { name: "Review this picture's workflow" })).toBeVisible();
    expect(api.pictureWorkflow).not.toHaveBeenCalled();
  });

  it("opens the workflow review and imports nothing until it is confirmed there", async () => {
    vi.mocked(api.pictureWorkflow).mockResolvedValue({ ui_graph: graph });
    vi.mocked(api.analyzeWorkflowPackage).mockResolvedValue(analysis);
    vi.mocked(api.ensureWorkflowPackageDraft).mockResolvedValue(
      { id: "draft-1", current_revision_id: "draft-revision-1" } as Workflow);
    vi.mocked(api.importWorkflowPackage).mockResolvedValue({ id: "wf-1" } as Workflow);
    await renderSettings();

    fireEvent.click(screen.getByRole("button", { name: "Review this picture's workflow" }));

    expect(await screen.findByRole("dialog", { name: /Review workflow package/ })).toBeVisible();
    expect(api.pictureWorkflow).toHaveBeenCalledWith(artifactId);
    expect(api.analyzeWorkflowPackage).toHaveBeenCalledWith(graph);
    expect(screen.getByDisplayValue("Harbor at dusk workflow")).toBeVisible();
    expect(api.ensureWorkflowPackageDraft).not.toHaveBeenCalled();
    expect(api.importWorkflowPackage).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Import workflow" }));

    await waitFor(() => expect(api.importWorkflowPackage).toHaveBeenCalledWith(expect.objectContaining({
      ui_graph: graph,
      name: "Harbor at dusk workflow",
    })));
    expect(await screen.findByText("Added to Workflows. It runs only after you review it there.")).toBeVisible();
    expect(screen.queryByRole("dialog", { name: /Review workflow package/ })).toBeNull();
  });

  it.each([
    ["the file carries none", () => Promise.reject(Object.assign(new Error("missing"), { code: "picture-workflow-missing" }))],
    ["the answer holds no editor graph", () => Promise.resolve({ ui_graph: { version: 0.4 } })],
  ])("says so when %s, and reviews nothing", async (_case, answer) => {
    vi.mocked(api.pictureWorkflow).mockImplementation(answer);
    await renderSettings();

    fireEvent.click(screen.getByRole("button", { name: "Review this picture's workflow" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This picture's file carries no workflow that can be reviewed.");
    expect(api.analyzeWorkflowPackage).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
