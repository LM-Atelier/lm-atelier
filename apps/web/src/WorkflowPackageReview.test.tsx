import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { api } from "./api";
import { WorkflowPackageReview } from "./WorkflowPackageReview";
import type { Workflow, WorkflowPackageAnalysis } from "./types";

vi.mock("./api", () => ({
  api: {
    prepareWorkflowPackage: vi.fn(),
    importWorkflowPackage: vi.fn(),
  },
}));

const analysis = (overrides: Partial<WorkflowPackageAnalysis> = {}): WorkflowPackageAnalysis => ({
  format_version: "0.4",
  frontend_version: null,
  node_count: 2,
  link_count: 1,
  subgraph_count: 0,
  operation_guess: "video",
  truncated: false,
  required_node_types: ["LoadImage", "WanVideoSampler"],
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
  ...overrides,
});

function renderReview(state: WorkflowPackageAnalysis, onImported = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <WorkflowPackageReview
        analysis={state}
        fileName="harbor-motion.json"
        uiGraph={{ nodes: [] }}
        onImported={onImported}
        onClose={vi.fn()}
      />
    </QueryClientProvider>,
  );
  return onImported;
}

describe("WorkflowPackageReview import", () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("imports a ready package under a confirmed name and operation", async () => {
    vi.mocked(api.importWorkflowPackage).mockResolvedValue({ id: "wf-1" } as Workflow);
    const onImported = renderReview(analysis());

    // Prefilled from the file and the analyzer's guess, both editable.
    expect(screen.getByDisplayValue("harbor-motion")).toBeInTheDocument();
    const operation = screen.getByRole("combobox", { name: /Operation/ });
    expect(operation).toHaveValue("image_to_video");
    fireEvent.change(operation, { target: { value: "text_to_video" } });
    fireEvent.click(screen.getByRole("button", { name: "Import workflow" }));

    await waitFor(() => expect(api.importWorkflowPackage).toHaveBeenCalledWith({
      ui_graph: { nodes: [] },
      name: "harbor-motion",
      operation: "text_to_video",
    }));
    await waitFor(() => expect(onImported).toHaveBeenCalled());
  });

  it("offers no import until everything is resolved", () => {
    renderReview(analysis({ ready: false, dependencies_resolved: false }));

    expect(screen.queryByRole("button", { name: "Import workflow" })).toBeNull();
    expect(api.importWorkflowPackage).not.toHaveBeenCalled();
  });

  it("keeps the server's refusal visible instead of closing over it", async () => {
    vi.mocked(api.importWorkflowPackage).mockRejectedValue(
      new Error("Start the media worker to compile workflows"),
    );
    const onImported = renderReview(analysis());

    fireEvent.click(screen.getByRole("button", { name: "Import workflow" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Start the media worker to compile workflows",
    );
    expect(onImported).not.toHaveBeenCalled();
  });
});
