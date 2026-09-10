/** Choosing the shape of what comes out, and seeing the pixels before Send. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { OutputRatioControl } from "./OutputRatioControl";
import { ratioOf } from "./outputRatio";
import type { OutputRatioPresetId, WorkflowOutputGeometryCapability } from "./types";

vi.mock("./api", () => ({
  api: {
    workflowRevisionOutputGeometry: vi.fn(),
    resolveWorkflowRevisionOutputGeometry: vi.fn(),
  },
}));

function capability(
  overrides: Partial<WorkflowOutputGeometryCapability> = {},
): WorkflowOutputGeometryCapability {
  return {
    version: 1,
    available: true,
    reason: null,
    revision_id: "rev-1",
    workflow_id: "wf-1",
    artifact_sha256: "a".repeat(64),
    operation: "text_to_image",
    engine: "comfyui",
    size_modes: ["exact", "preset"],
    preset_ids: ["16:9", "1:1", "2:3", "3:2", "3:4", "4:3", "9:16"],
    width: {
      key: "width",
      node_id: "latent",
      input_name: "width",
      default: 1024,
      minimum: 128,
      maximum: 2048,
      multiple_of: 64,
    },
    height: {
      key: "height",
      node_id: "latent",
      input_name: "height",
      default: 768,
      minimum: 128,
      maximum: 2048,
      multiple_of: 64,
    },
    graph_binding_verified: true,
    request_authorized: false,
    ...overrides,
  };
}

function renderControl(
  props: Partial<{
    revisionId: string;
    width: unknown;
    height: unknown;
    onDimensions: (dimensions: { width: number; height: number }) => void;
  }> = {},
) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <OutputRatioControl
        revisionId={props.revisionId ?? "rev-1"}
        width={props.width ?? 1024}
        height={props.height ?? 768}
        onDimensions={props.onDimensions ?? (() => {})}
      />
    </QueryClientProvider>,
  );
}

describe("ratioOf", () => {
  const offered: OutputRatioPresetId[] = ["1:1", "4:3", "16:9"];

  it("names the ratio a pair of pixels already is", () => {
    expect(ratioOf(1024, 576, offered)).toBe("16:9");
    expect(ratioOf(1024, 768, offered)).toBe("4:3");
    expect(ratioOf(896, 896, offered)).toBe("1:1");
  });

  it("names nothing for a ratio this workflow does not offer", () => {
    // 3:2 is a real preset and a real ratio; it is not in this workflow's list,
    // so the control must not show it as the current choice.
    expect(ratioOf(1152, 768, offered)).toBeNull();
  });

  it("classifies rather than divides", () => {
    // The values that would reduce to a fraction of NaN, or to a division by
    // zero, are refused before any arithmetic touches them.
    expect(ratioOf(Number.NaN, 768, offered)).toBeNull();
    expect(ratioOf(1024, Number.NaN, offered)).toBeNull();
    expect(ratioOf(Number.POSITIVE_INFINITY, 768, offered)).toBeNull();
    expect(ratioOf(1024, 0, offered)).toBeNull();
    expect(ratioOf(-1024, -768, offered)).toBeNull();
    expect(ratioOf(1024.5, 768, offered)).toBeNull();
    expect(ratioOf("1024", 768, offered)).toBeNull();
    expect(ratioOf(undefined, undefined, offered)).toBeNull();
  });
});

describe("OutputRatioControl", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("offers only the ratios this workflow can produce exactly", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(
      capability({ preset_ids: ["1:1", "4:3"] }),
    );

    renderControl();

    await waitFor(() => expect(screen.getByRole("group", { name: "Output aspect ratio" })).toBeTruthy());
    const buttons = screen.getAllByRole("button").map((button) => button.textContent);
    expect(buttons).toEqual(["1:1 Square", "4:3 Landscape"]);
  });

  it("shows nothing at all when the workflow expresses no ratio", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(
      capability({ size_modes: ["exact"], preset_ids: [] }),
    );

    const { container } = renderControl();

    await waitFor(() => expect(api.workflowRevisionOutputGeometry).toHaveBeenCalled());
    expect(container.textContent).toBe("");
  });

  it("shows nothing when the revision has no proof that width and height reach its output", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(
      capability({
        available: false,
        reason: "unsupported_workflow_geometry",
        preset_ids: [],
        width: null,
        height: null,
      }),
    );

    const { container } = renderControl();

    await waitFor(() => expect(api.workflowRevisionOutputGeometry).toHaveBeenCalled());
    expect(container.textContent).toBe("");
  });

  it("takes the pixels from the server rather than working them out", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(capability());
    vi.mocked(api.resolveWorkflowRevisionOutputGeometry).mockResolvedValue({
      version: 1,
      workflow_id: "wf-1",
      revision_id: "rev-1",
      artifact_sha256: "a".repeat(64),
      operation: "text_to_image",
      engine: "comfyui",
      mode: "image",
      size_mode: "preset",
      preset_id: "16:9",
      width: 1024,
      height: 576,
      graph_binding_verified: true,
      request_authorized: false,
    });
    const onDimensions = vi.fn();

    renderControl({ onDimensions });

    await waitFor(() => expect(screen.getByRole("button", { name: "16:9 Wide" })).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "16:9 Wide" }));

    await waitFor(() => expect(onDimensions).toHaveBeenCalledWith({ width: 1024, height: 576 }));
    expect(api.resolveWorkflowRevisionOutputGeometry).toHaveBeenCalledWith("rev-1", {
      mode: "image",
      size_mode: "preset",
      preset_id: "16:9",
    });
  });

  it("shows the exact pixels the run will use", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(capability());

    renderControl({ width: 1024, height: 576 });

    await waitFor(() => expect(screen.getByText("Output: 1024 × 576")).toBeTruthy());
  });

  it("marks the ratio the current pixels already are", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(capability());

    renderControl({ width: 1024, height: 576 });

    await waitFor(() => expect(screen.getByRole("button", { name: "16:9 Wide" })).toBeTruthy());
    expect(screen.getByRole("button", { name: "16:9 Wide" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "1:1 Square" }).getAttribute("aria-pressed")).toBe("false");
  });

  it("says so when the revision stops offering a ratio it had offered", async () => {
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(capability());
    vi.mocked(api.resolveWorkflowRevisionOutputGeometry).mockRejectedValue(new Error("422"));
    const onDimensions = vi.fn();

    renderControl({ onDimensions });

    await waitFor(() => expect(screen.getByRole("button", { name: "16:9 Wide" })).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "16:9 Wide" }));

    await waitFor(() =>
      expect(screen.getByText("This workflow no longer offers 16:9.")).toBeTruthy(),
    );
    expect(onDimensions).not.toHaveBeenCalled();
  });
});
