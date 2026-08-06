import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkflowsView } from "./WorkflowsView";
import { api } from "./api";

vi.mock("./api", () => ({
  api: {
    workflows: vi.fn(),
    workflowFamilies: vi.fn().mockResolvedValue([]),
    validateWorkflow: vi.fn(),
  },
}));

function revision(id: string) {
  return {
    id,
    workflow_id: id,
    version: 1,
    engine: "comfyui",
    api_graph_json: {},
    input_schema_json: {},
    dependencies_json: {},
    trusted: true,
    created_at: "2026-08-06T00:00:00Z",
  };
}

function workflow(id: string, name: string) {
  return {
    id,
    name,
    description: "",
    operation: "text_to_image",
    current_revision_id: id,
    revisions: [revision(id)],
  };
}

function renderView() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <WorkflowsView />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("a verdict belongs to the workflow it was asked about", () => {
  it("does not carry one workflow's validation over to another", async () => {
    vi.mocked(api.workflows).mockResolvedValue([
      workflow("wf-a", "Alpha"),
      workflow("wf-b", "Beta"),
    ] as never);
    vi.mocked(api.validateWorkflow).mockResolvedValue({
      valid: false,
      errors: ["Alpha is missing a node"],
      warnings: [],
    } as never);

    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(await screen.findByRole("button", { name: "Validate" }));
    await waitFor(() => expect(screen.getByText("Alpha is missing a node")).toBeTruthy());

    // The verdict was about Alpha. Beta has not been validated at all.
    fireEvent.click(screen.getByText("Beta"));

    expect(screen.queryByText("Alpha is missing a node")).toBeNull();
  });
});

describe("a list that could not be read", () => {
  it("says so instead of offering an empty shelf", async () => {
    vi.mocked(api.workflows).mockRejectedValue(new Error("workflow library unreachable"));

    renderView();

    await waitFor(() =>
      expect(screen.getByText("workflow library unreachable")).toBeTruthy(),
    );
  });
});
