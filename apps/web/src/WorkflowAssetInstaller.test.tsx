import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { api } from "./api";
import { WorkflowAssetInstaller } from "./WorkflowAssetInstaller";
import { searchTermFor } from "./workflowAssetSearch";
import type { CatalogModel, WorkflowAssetReference } from "./types";

vi.mock("./api", () => ({
  api: {
    catalog: vi.fn(),
    catalogPreflight: vi.fn(),
    reviewWorkflowAssets: vi.fn(),
    installWorkflowAssets: vi.fn(),
  },
}));

const uiGraph = { nodes: [] };

const missing: WorkflowAssetReference[] = [
  { filename: "detail-slider.safetensors", suffix: ".safetensors", policy: "supported", kind: "lora", source_url: null, present_locally: false , source_candidates: [] },
];

const candidate = {
  provider: "civitai",
  remote_id: "3102245",
  name: "Detail slider",
  author: "creator",
  total_size_bytes: 1024,
  required_runtime: "comfyui",
} as CatalogModel;

function renderInstaller(onInstalled = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <WorkflowAssetInstaller uiGraph={uiGraph} missing={missing} onInstalled={onInstalled} />
    </QueryClientProvider>,
  );
  return onInstalled;
}

describe("workflow asset installer", () => {
  beforeEach(() => {
    vi.mocked(api.catalog).mockResolvedValue({ items: [candidate], next_cursor: null } as never);
    vi.mocked(api.catalogPreflight).mockResolvedValue({
      install_plan: { id: "plan-1", plan_hash: "a".repeat(64), compatibility: "supported" },
      selected_files: ["detail-slider.safetensors"],
      can_install: true,
      checks: [],
    } as never);
    vi.mocked(api.reviewWorkflowAssets).mockResolvedValue({
      binding_plan_hash: "b".repeat(64),
      assets: [{ reference_filename: "detail-slider.safetensors", size_bytes: 1024 } as never],
      download_count: 1,
      total_bytes: 1024,
    } as never);
    vi.mocked(api.installWorkflowAssets).mockResolvedValue([{ id: "job-1" }] as never);
  });
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("turns a filename into a searchable term", () => {
    expect(searchTermFor("Atelier_Portrait_AIO.safetensors")).toBe("Atelier Portrait AIO");
    expect(searchTermFor("subdir/studio_lighting.safetensors")).toBe("studio lighting");
  });

  it("walks search, preflight, review, and queue without guessing", async () => {
    const onInstalled = renderInstaller();

    // Nothing is reviewable until the user picks a source for the file.
    expect(screen.getByRole("button", { name: /Review 0 selected/ })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /Search/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Detail slider" }));

    // The preflight's plan - not the browser - supplies the binding.
    await waitFor(() => expect(screen.getByText("Selected")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /Review 1 selected/ }));

    await waitFor(() =>
      expect(api.reviewWorkflowAssets).toHaveBeenCalledWith(uiGraph, [
        {
          reference_filename: "detail-slider.safetensors",
          install_plan_id: "plan-1",
          artifact_path: "detail-slider.safetensors",
        },
      ]));
    expect(await screen.findByText(/1 download/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Install these files/ }));
    await waitFor(() =>
      expect(api.installWorkflowAssets).toHaveBeenCalledWith(
        uiGraph,
        expect.any(Array),
        "b".repeat(64),
      ));
    await waitFor(() => expect(onInstalled).toHaveBeenCalledWith(1));
    expect(await screen.findByText(/Queued 1 download/)).toBeInTheDocument();
  });

  it("cannot queue without a fresh review of the current selection", async () => {
    renderInstaller();
    fireEvent.click(screen.getByRole("button", { name: /Search/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Detail slider" }));
    await waitFor(() => expect(screen.getByText("Selected")).toBeInTheDocument());

    // Install stays disabled until a review produces a hash to confirm.
    expect(screen.getByRole("button", { name: /Install these files/ })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /Review 1 selected/ }));
    await screen.findByText(/1 download/);
    expect(screen.getByRole("button", { name: /Install these files/ })).toBeEnabled();

    // Changing the selection invalidates the review rather than queueing a
    // stale binding the user never saw.
    fireEvent.click(screen.getByRole("button", { name: "Change" }));
    expect(screen.getByRole("button", { name: /Install these files/ })).toBeDisabled();
    expect(api.installWorkflowAssets).not.toHaveBeenCalled();
  });

  it("names the exact file and its kind rather than letting a bundle be ranked", async () => {
    renderInstaller();
    fireEvent.click(screen.getByRole("button", { name: /Search/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Detail slider" }));

    await waitFor(() => expect(api.catalogPreflight).toHaveBeenCalled());
    const call = vi.mocked(api.catalogPreflight).mock.calls[0];
    expect(call[4]).toEqual(["detail-slider.safetensors"]);
    expect(call[8]).toBe("lora");
  });

  it("refuses a plan holding more than the one file the workflow named", async () => {
    // A repository ranked into its official bundle answers with several files.
    // Binding the first would quietly install the rest.
    vi.mocked(api.catalogPreflight).mockResolvedValue({
      install_plan: { id: "plan-1", plan_hash: "a".repeat(64), compatibility: "supported" },
      selected_files: ["detail-slider.safetensors", "unrelated-vae.safetensors"],
      can_install: true,
      checks: [],
    } as never);
    renderInstaller();

    fireEvent.click(screen.getByRole("button", { name: /Search/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Detail slider" }));

    await waitFor(() => expect(api.catalogPreflight).toHaveBeenCalled());
    expect(screen.queryByText("Selected")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Review 0 selected/ })).toBeDisabled();
  });

  it("says nothing at all when every file is present", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const { container } = render(
      <QueryClientProvider client={client}>
        <WorkflowAssetInstaller uiGraph={uiGraph} missing={[]} />
      </QueryClientProvider>,
    );
    expect(container.querySelector("section")).toBeNull();
  });
});
