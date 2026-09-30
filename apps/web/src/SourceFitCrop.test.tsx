/** Crop / fill beside Extend: offered only where the workflow can edit a crop, previewed as the kept part. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { useState, type PropsWithChildren } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import type { SourceFitMode, SourceFitPreviewResult, SourceFitSelection } from "./sourceFit";
import { SourceFitControl } from "./SourceFitControl";
import { useSourceFitCanvas } from "./useSourceFitCanvas";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const sourceId = "sha256:" + "a".repeat(64);
const crop: SourceFitSelection = {
  sourceArtifactId: sourceId,
  workflowRevisionId: "revision-one",
  request: { mode: "crop", width: 1200, height: 900 },
};
// A 600 by 800 source keeps its whole width and the middle 450 rows for a 4:3 canvas.
const answer: SourceFitPreviewResult = {
  version: 1, mode: "crop", workflow_revision_id: "revision-one",
  workflow_artifact_sha256: "b".repeat(64), source_artifact_id: sourceId,
  source: { width: 600, height: 800 }, canvas: { width: 1200, height: 900 },
  margins: { left: 0, right: 0, top: 0, bottom: 0 },
  source_rectangle: { x: 0, y: 0, width: 1200, height: 900 },
  kept: { left: 0, top: 175, width: 600, height: 450 },
  request_authorized: false,
};

function answering(modes: SourceFitMode[], preview: SourceFitPreviewResult = answer) {
  vi.spyOn(api, "workflowRevisionSourceFit").mockResolvedValue({
    available: modes.length > 0, reason: modes.length > 0 ? null : "source_fit_workflow_unsupported", modes,
    request_authorized: false,
  });
  return vi.spyOn(api, "previewWorkflowRevisionSourceFit").mockResolvedValue(preview);
}

/** A fresh query cache for each render, so no answer carries over between them. */
function fresh() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

const props = {
  mode: "image" as const, sourceCanvasRevisionId: "revision-one", sourceId,
  families: [], workflowSelection: null, projectSelection: null,
};

describe("a crop's canvas", () => {
  it("is previewed and selected only as a crop", async () => {
    const preview = answering(["crop"]);
    const { result } = renderHook(() => useSourceFitCanvas({ ...props, value: crop, onChange: vi.fn() }), { wrapper: fresh() });

    await waitFor(() => expect(result.current.canPreview).toBe(true));
    expect(result.current.modes).toEqual(["crop"]);
    act(() => result.current.requestPreview());

    await waitFor(() => expect(result.current.selection).toEqual(crop));
    expect(preview).toHaveBeenCalledWith("revision-one", sourceId, { mode: "crop", width: 1200, height: 900 }, expect.any(AbortSignal));
  });

  it("refuses an answer for the other way of fitting", async () => {
    answering(["extend", "crop"], { ...answer, mode: "extend", kept: null });
    const { result } = renderHook(() => useSourceFitCanvas({ ...props, value: crop, onChange: vi.fn() }), { wrapper: fresh() });

    await waitFor(() => expect(result.current.canPreview).toBe(true));
    act(() => result.current.requestPreview());

    await waitFor(() => expect(result.current.error).toBe(true));
    expect(result.current.selection).toBeNull();
  });

  it("cannot be previewed on a workflow that only extends", async () => {
    answering(["extend"]);
    const { result } = renderHook(() => useSourceFitCanvas({ ...props, value: crop, onChange: vi.fn() }), { wrapper: fresh() });

    await waitFor(() => expect(result.current.available).toBe(true));
    expect(result.current.modes).toEqual(["extend"]);
    expect(result.current.canPreview).toBe(false);
  });
});

/** The control as a turn editor holds it: the chosen canvas is the editor's state. */
function Control({ initial = null }: { initial?: SourceFitSelection | null }) {
  const [value, setValue] = useState<SourceFitSelection | null>(initial);
  const canvas = useSourceFitCanvas({ ...props, value, onChange: setValue });
  return <SourceFitControl canvas={canvas} sourceUrl="blob:source" initialWidth={1200} initialHeight={900} />;
}

describe("the source canvas control", () => {
  it("offers Crop / fill only where the workflow can edit a crop", async () => {
    answering(["extend"]);
    const { unmount } = render(<Control />, { wrapper: fresh() });
    expect(await screen.findByRole("button", { name: "Extend / preserve all" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Crop / fill" })).toBeNull();
    unmount();

    answering(["crop"]);
    render(<Control />, { wrapper: fresh() });
    expect(await screen.findByRole("button", { name: "Crop / fill" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Extend / preserve all" })).toBeNull();
  });

  it("keeps the chosen canvas when switching between the two ways", async () => {
    answering(["extend", "crop"]);
    render(<Control initial={{ ...crop, request: { mode: "extend", width: 1400, height: 700 } }} />, { wrapper: fresh() });

    fireEvent.click(await screen.findByRole("button", { name: "Crop / fill" }));

    expect(screen.getByRole("button", { name: "Crop / fill" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Extend / preserve all" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("spinbutton", { name: "Canvas width" })).toHaveValue(1400);
    expect(screen.getByRole("spinbutton", { name: "Canvas height" })).toHaveValue(700);
  });

  it("shows the part of the source a crop keeps", async () => {
    answering(["crop"]);
    render(<Control initial={crop} />, { wrapper: fresh() });

    const request = await screen.findByRole("button", { name: "Preview canvas" });
    await waitFor(() => expect(request).toHaveAttribute("aria-disabled", "false"));
    fireEvent.click(request);

    const shown = await screen.findByRole("img", { name: /Crop preview/ });
    expect(shown).toHaveAccessibleName(
      "Crop preview The outlined 600 by 450 part of the 600 by 800 source fills the 1200 by 900 canvas. The rest of the source is left out.",
    );
    expect(shown).toHaveAttribute("viewBox", "0 0 600 800");
    const outline = shown.querySelector("rect");
    expect(outline).toHaveAttribute("y", "175");
    expect(outline).toHaveAttribute("height", "450");
    expect(screen.getByText("Source: 600 × 800 · Output: 1200 × 900")).toBeInTheDocument();
  });
});
