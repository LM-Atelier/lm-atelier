import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { api } from "./api";
import { WorkflowPackageReview } from "./WorkflowPackageReview";
import type { Workflow, WorkflowPackageAnalysis } from "./types";

vi.mock("./api", () => ({
  api: {
    ensureWorkflowPackageDraft: vi.fn(),
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
  source_candidates: [],
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
    vi.mocked(api.ensureWorkflowPackageDraft).mockResolvedValue({
      id: "draft-1",
      current_revision_id: "draft-revision-1",
    } as Workflow);
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
      draft_workflow_id: "draft-1",
      draft_revision_id: "draft-revision-1",
    }));
    await waitFor(() => expect(onImported).toHaveBeenCalled());
  });

  it("offers no import until everything is resolved", () => {
    renderReview(analysis({ ready: false, dependencies_resolved: false }));

    expect(screen.queryByRole("button", { name: "Import workflow" })).toBeNull();
    expect(api.importWorkflowPackage).not.toHaveBeenCalled();
  });

  it("persists the exact graph before preparing an unresolved package", async () => {
    vi.mocked(api.ensureWorkflowPackageDraft).mockResolvedValue({
      id: "draft-1",
      current_revision_id: "draft-revision-1",
    } as Workflow);
    vi.mocked(api.prepareWorkflowPackage).mockResolvedValue({ id: "job-1" } as never);
    renderReview(analysis({
      ready: false,
      dependencies_resolved: false,
      custom_packages: [{
        package_id: "rgthree-comfy",
        versions: ["1.2.3"],
        node_types: ["Power Lora Loader"],
        locally_resolved: false,
      }],
    }));

    fireEvent.click(screen.getByRole("button", { name: "Prepare 1.2.3" }));

    await waitFor(() => expect(api.ensureWorkflowPackageDraft).toHaveBeenCalledWith({
      ui_graph: { nodes: [] },
      name: "harbor-motion",
      operation: "image_to_video",
    }));
    await waitFor(() => expect(api.prepareWorkflowPackage).toHaveBeenCalledWith(
      "rgthree-comfy",
      "1.2.3",
      { nodes: [] },
      "draft-revision-1",
    ));
  });

  it("keeps the server's refusal visible instead of closing over it", async () => {
    vi.mocked(api.ensureWorkflowPackageDraft).mockResolvedValue({
      id: "draft-1",
      current_revision_id: "draft-revision-1",
    } as Workflow);
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

describe("WorkflowPackageReview sources", () => {
  afterEach(() => {
    cleanup();
  });

  it("offers the sources the author recorded when they name no file", () => {
    renderReview(
      analysis({
        source_candidates: [
          {
            provider: "civitai",
            remote_id: "3075606",
            revision: "3075606",
            filename: null,
            url: "https://civitai.com/models/1662740/portrait?modelVersionId=3075606",
          },
        ],
      }),
    );

    const link = screen.getByRole("link", { name: "3075606" });
    expect(link).toHaveAttribute("href", "https://civitai.com/models/1662740/portrait?modelVersionId=3075606");
    // Opening someone else's link must not hand them this window.
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
  });

  it("says nothing about sources when the author recorded none", () => {
    renderReview(analysis());
    expect(screen.queryByText("Sources this workflow mentions")).toBeNull();
  });
});

describe("WorkflowPackageReview runtime inventory", () => {
  afterEach(() => {
    cleanup();
  });

  it("does not present inventory-derived package findings as facts while offline", () => {
    renderReview(analysis({
      ready: false,
      runtime_nodes_available: false,
      dependencies_resolved: false,
      node_inventory_available: false,
      issues: [
        {
          code: "unidentified_custom_node_package",
          count: 1,
          node_types: ["KSampler"],
          severity: "blocking",
        },
        {
          code: "unresolved_custom_node_package",
          count: 1,
          node_types: ["RegistrySampler"],
          severity: "blocking",
        },
        {
          code: "unversioned_custom_node_package",
          count: 1,
          node_types: ["UnpinnedNode"],
          severity: "blocking",
        },
      ],
    }));

    expect(screen.getByText(/node availability is unknown/i)).toBeInTheDocument();
    expect(screen.queryByText("Uses custom nodes with no declared package")).toBeNull();
    expect(screen.queryByText("Needs a package version this machine does not have installed")).toBeNull();
    expect(screen.getByText("Uses a package without a pinned version")).toBeInTheDocument();
  });
  it("tells you to re-read a package you already installed, not to fetch it", () => {
    // Reachable by upgrading rather than by doing anything wrong: the reviewed
    // node inventory began being recorded after people had already trusted
    // packages, and until this had its own wording the application told them to
    // install what was already sitting on disk.
    renderReview(analysis({
      ready: false,
      runtime_nodes_available: true,
      dependencies_resolved: false,
      node_inventory_available: true,
      issues: [
        {
          code: "custom_node_package_awaiting_review",
          count: 1,
          node_types: ["GetNode", "SetNode"],
          severity: "blocking",
        },
      ],
    }));

    expect(screen.getByText(/review it again to confirm/i)).toBeInTheDocument();
    expect(
      screen.queryByText("Needs a package version this machine does not have installed"),
    ).toBeNull();
  });

  it("does not claim a package needs re-reading while node availability is unknown", () => {
    // Whether a package resolves is read against the runtime inventory, so
    // without one every package looks unresolved and this would send somebody
    // to re-review packages that are fine.
    renderReview(analysis({
      ready: false,
      runtime_nodes_available: false,
      dependencies_resolved: false,
      node_inventory_available: false,
      issues: [
        {
          code: "custom_node_package_awaiting_review",
          count: 1,
          node_types: ["GetNode"],
          severity: "blocking",
        },
      ],
    }));

    expect(screen.queryByText(/review it again to confirm/i)).toBeNull();
  });
  it("presents a refusal as an error rather than as another note", async () => {
    // Both refusals rendered with the same muted class as ordinary explanatory
    // text, so a blocking failure read as a caption.
    vi.mocked(api.ensureWorkflowPackageDraft).mockRejectedValue(
      Object.assign(new Error("comfyui-videohelpersuite did not load."), { code: "x" }),
    );
    renderReview(analysis({ ready: true, dependencies_resolved: true }), vi.fn());

    fireEvent.click(screen.getByRole("button", { name: /Import workflow/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.className).toContain("callout");
    expect(alert.className).toContain("error");
    expect(alert.className).not.toContain("package-review-note");
  });
});
