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
    workflowOpenTarget: vi.fn(),
    cloneWorkflow: vi.fn(),
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

describe("opening a workflow in ComfyUI", () => {
  it("offers the way in as a link rather than a popup", async () => {
    // A window asked for after the click is a popup and gets refused, and
    // `noopener` makes window.open return null either way - so the old code
    // could not tell a blocked tab from an opened one.
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    vi.mocked(api.workflowOpenTarget).mockResolvedValue({
      url: "http://127.0.0.1:8188/",
      filename: "alpha.json",
      ui_graph: {},
    } as never);
    const popup = vi.fn();
    vi.stubGlobal("open", popup);
    // jsdom has no download plumbing; the anchor click is not under test.
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    renderView();
    fireEvent.click(await screen.findByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: /Download UI graph/i }));

    const link = await screen.findByRole("link", { name: "Open ComfyUI" });
    expect(link.getAttribute("href")).toBe("http://127.0.0.1:8188/");
    expect(popup).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });
});

describe("after a workflow changes", () => {
  it("re-asks for the families the change could have moved", async () => {
    // The server derives a family's current revision, engine, capabilities and
    // readiness from the revision that just changed. Refreshing the list alone
    // left the families beside it - and the selectors elsewhere - answering
    // from before the change, with nothing to heal them while mounted.
    vi.mocked(api.workflows).mockResolvedValue([workflow("wf-a", "Alpha")] as never);
    vi.mocked(api.cloneWorkflow).mockResolvedValue(workflow("wf-b", "Alpha copy") as never);

    renderView();
    await screen.findByText("Alpha");
    const familiesReadBefore = vi.mocked(api.workflowFamilies).mock.calls.length;

    fireEvent.click(screen.getByText("Alpha"));
    fireEvent.click(screen.getByRole("button", { name: "Duplicate" }));

    await waitFor(() =>
      expect(vi.mocked(api.workflowFamilies).mock.calls.length).toBeGreaterThan(familiesReadBefore),
    );
  });
});
