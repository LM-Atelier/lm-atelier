/** Match source: the workflow's own size in the exact shape of the turn's picture. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { OutputRatioControl } from "./OutputRatioControl";
import type { WorkflowOutputGeometryCapability, WorkflowOutputGeometryResolution } from "./types";

vi.mock("./api", () => ({
  api: {
    workflowRevisionOutputGeometry: vi.fn(),
    resolveWorkflowRevisionOutputGeometry: vi.fn(),
    matchWorkflowRevisionOutputGeometryToSource: vi.fn(),
  },
}));

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

const proof: WorkflowOutputGeometryCapability = {
  version: 1, available: true, reason: null, revision_id: "frame-revision", workflow_id: "wf-1",
  artifact_sha256: "a".repeat(64), operation: "image_to_video", engine: "comfyui", size_modes: ["exact", "preset"],
  preset_ids: ["16:9", "9:16"], width: null, height: null, graph_binding_verified: true, request_authorized: false,
};

function answer(width: number, height: number): WorkflowOutputGeometryResolution {
  return {
    version: 1, workflow_id: "wf-1", revision_id: "frame-revision", artifact_sha256: "a".repeat(64),
    operation: "image_to_video", engine: "comfyui", mode: "video", size_mode: "exact", preset_id: null,
    width, height, graph_binding_verified: true, request_authorized: false,
  };
}

/** The row as a settings panel holds it: the chosen size is the panel's own. */
function Row({ source, revisionId = "frame-revision" }: { source: string | null; revisionId?: string }) {
  const [size, setSize] = useState({ width: 1344, height: 768 });
  return <OutputRatioControl revisionId={revisionId} width={size.width} height={size.height}
    sizeIsTheWorkflowsOwn={false} onDimensions={setSize} source={source} />;
}

function show(source: string | null, revisionId?: string) {
  vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(proof);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><Row source={source} revisionId={revisionId} /></QueryClientProvider>);
}

it("sizes the output to the picture's exact shape, and shows that choice as made", async () => {
  vi.mocked(api.matchWorkflowRevisionOutputGeometryToSource).mockResolvedValue(answer(896, 1120));
  show("start-frame");

  const match = await screen.findByRole("button", { name: "Match source" });
  expect(match).toHaveAttribute("aria-pressed", "false");
  fireEvent.click(match);

  await waitFor(() => expect(screen.getByText("Output: 896 × 1120")).toBeInTheDocument());
  expect(api.matchWorkflowRevisionOutputGeometryToSource).toHaveBeenCalledExactlyOnceWith("frame-revision", "start-frame");
  expect(match).toHaveAttribute("aria-pressed", "true");
});

it("stops showing the choice as made once another shape is chosen", async () => {
  vi.mocked(api.matchWorkflowRevisionOutputGeometryToSource).mockResolvedValue(answer(896, 1120));
  vi.mocked(api.resolveWorkflowRevisionOutputGeometry).mockResolvedValue({ ...answer(1344, 756), size_mode: "preset", preset_id: "16:9" });
  show("start-frame");
  fireEvent.click(await screen.findByRole("button", { name: "Match source" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Match source" })).toHaveAttribute("aria-pressed", "true"));

  fireEvent.click(screen.getByRole("button", { name: "16:9 Wide" }));

  await waitFor(() => expect(screen.getByText("Output: 1344 × 756")).toBeInTheDocument());
  expect(screen.getByRole("button", { name: "Match source" })).toHaveAttribute("aria-pressed", "false");
});

it("says so when the workflow cannot make the picture's exact shape, and keeps the size", async () => {
  vi.mocked(api.matchWorkflowRevisionOutputGeometryToSource).mockRejectedValue(new Error("refused"));
  show("start-frame");

  fireEvent.click(await screen.findByRole("button", { name: "Match source" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("This workflow cannot make the source picture's exact shape.");
  expect(screen.getByText("Output: 1344 × 768")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Match source" })).toHaveAttribute("aria-pressed", "false");
});

it("offers no Match source without a picture to match", async () => {
  show(null);

  expect(await screen.findByRole("group", { name: "Output aspect ratio" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Match source" })).toBeNull();
});
