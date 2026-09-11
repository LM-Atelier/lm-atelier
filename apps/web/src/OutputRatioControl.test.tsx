/** Choosing the shape of what comes out, and seeing the pixels before Send. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { OutputRatioControl } from "./OutputRatioControl";
import { ratioOf } from "./outputRatio";
import type {
  OutputRatioPresetId,
  WorkflowOutputGeometryCapability,
  WorkflowOutputGeometryResolution,
} from "./types";

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

function resolution(
  overrides: Partial<WorkflowOutputGeometryResolution> = {},
): WorkflowOutputGeometryResolution {
  return {
    version: 1,
    workflow_id: "wf-1",
    revision_id: "rev-1",
    artifact_sha256: "a".repeat(64),
    operation: "text_to_image",
    engine: "comfyui",
    mode: "image",
    size_mode: "preset",
    preset_id: "3:4",
    width: 896,
    height: 1152,
    graph_binding_verified: true,
    request_authorized: false,
    ...overrides,
  };
}

/** A resolve call whose answer this test decides when to give.
 *
 * The window between the press and the answer is where the focus defect lived,
 * so it has to be held open and observed rather than waited out.
 */
function deferredResolve(): (value: WorkflowOutputGeometryResolution) => void {
  let answer!: (value: WorkflowOutputGeometryResolution) => void;
  vi.mocked(api.resolveWorkflowRevisionOutputGeometry).mockReturnValue(
    new Promise<WorkflowOutputGeometryResolution>((resolve) => {
      answer = resolve;
    }),
  );
  return answer;
}

function renderControl(
  props: Partial<{
    revisionId: string;
    width: unknown;
    height: unknown;
    sizeIsTheWorkflowsOwn: boolean;
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
        sizeIsTheWorkflowsOwn={props.sizeIsTheWorkflowsOwn ?? false}
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

  it("leaves the choice under the hand that made it", async () => {
    // Somebody who is using the keyboard is standing ON the button when they
    // press it. Taking that button away from them - which the disabled
    // attribute does, because the browser will not leave focus on a disabled
    // control - drops them on the document body, so nothing announces the
    // choice and there is nowhere to carry on from.
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(capability());
    const answer = deferredResolve();

    renderControl();

    await waitFor(() => expect(screen.getByRole("button", { name: "3:4 Portrait" })).toBeTruthy());
    const chosen = screen.getByRole("button", { name: "3:4 Portrait" });
    chosen.focus();
    fireEvent.click(chosen);

    // While the answer is still in flight: the whole point, since this is the
    // window in which the old code had already thrown focus away.
    await waitFor(() => expect(chosen.getAttribute("aria-disabled")).toBe("true"));
    expect(document.activeElement).toBe(chosen);

    answer(resolution());
    await waitFor(() => expect(chosen.getAttribute("aria-disabled")).toBe("false"));
    expect(document.activeElement).toBe(chosen);
  });

  it("ignores a second press rather than taking the button away to prevent one", async () => {
    // Keeping focus must not cost the protection the disabled attribute gave.
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(capability());
    const answer = deferredResolve();

    renderControl();

    await waitFor(() => expect(screen.getByRole("button", { name: "3:4 Portrait" })).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "3:4 Portrait" }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "1:1 Square" }).getAttribute("aria-disabled")).toBe("true"),
    );
    fireEvent.click(screen.getByRole("button", { name: "1:1 Square" }));
    fireEvent.click(screen.getByRole("button", { name: "3:4 Portrait" }));

    expect(api.resolveWorkflowRevisionOutputGeometry).toHaveBeenCalledTimes(1);
    answer(resolution());
  });

  it("says a refusal out loud rather than only on screen", async () => {
    // A failed action that is merely drawn is not reported at all to somebody
    // listening, who is then left believing the shape they chose was taken.
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(capability());
    vi.mocked(api.resolveWorkflowRevisionOutputGeometry).mockRejectedValue(new Error("422"));

    renderControl();

    await waitFor(() => expect(screen.getByRole("button", { name: "16:9 Wide" })).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "16:9 Wide" }));

    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toBe("This workflow no longer offers 16:9."),
    );
  });

  it("does not carry a refusal over to a different revision", async () => {
    // The panel's role tabs change which revision this row describes without
    // remounting it, so a message held across that change contradicts the
    // buttons beside it: the shape it says is gone is offered right there.
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(capability());
    vi.mocked(api.resolveWorkflowRevisionOutputGeometry).mockRejectedValue(new Error("422"));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const tree = (revisionId: string) => (
      <QueryClientProvider client={client}>
        <OutputRatioControl
          revisionId={revisionId}
          width={1024}
          height={768}
          sizeIsTheWorkflowsOwn={false}
          onDimensions={() => {}}
        />
      </QueryClientProvider>
    );

    const { rerender } = render(tree("rev-1"));

    await waitFor(() => expect(screen.getByRole("button", { name: "16:9 Wide" })).toBeTruthy());
    fireEvent.click(screen.getByRole("button", { name: "16:9 Wide" }));
    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());

    rerender(tree("rev-2"));

    // Wait for the new revision's own answer to arrive first. Asserting the
    // message is gone while the row is still empty would pass for a reason
    // that has nothing to do with the refusal being cleared.
    await waitFor(() => expect(screen.getByRole("button", { name: "16:9 Wide" })).toBeTruthy());
    expect(screen.queryByRole("alert")).toBeNull();
  });
});

describe("a workflow that decides its own size", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("says so, rather than leaving the panel silent", async () => {
    // A workflow can declare its width and height as not yours to change. The
    // panel then offers no shape control AND no number boxes, so before this
    // the row for picture size simply was not there - no shapes, no
    // dimensions, no reason, and nothing to distinguish it from a fault.
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(
      capability({ available: false, reason: "unsupported_workflow_geometry", preset_ids: [] }),
    );

    renderControl({ sizeIsTheWorkflowsOwn: true, width: undefined, height: undefined });

    await waitFor(() =>
      expect(screen.getByText("This workflow sets the picture size itself.")).toBeTruthy(),
    );
    expect(screen.queryByRole("group", { name: "Output aspect ratio" })).toBeNull();
  });

  it("stays out of the way while the number boxes can still be used", async () => {
    // The other half of the same condition, and the reason this is not simply
    // "explain whenever there is no proof": where width and height ARE offered
    // they remain the honest way to ask for a size, and a row saying the
    // workflow decides would contradict the boxes directly beneath it.
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(
      capability({ available: false, reason: "unsupported_workflow_geometry", preset_ids: [] }),
    );

    const { container } = renderControl({ sizeIsTheWorkflowsOwn: false });

    await waitFor(() => expect(api.workflowRevisionOutputGeometry).toHaveBeenCalled());
    expect(container.textContent).toBe("");
  });

  it("offers the shapes when a workflow proves it can produce them", async () => {
    // The claim is about what this panel shows, not about the workflow's own
    // nature, so a proven revision is never described as deciding for itself
    // even when the panel happens to be asked the other way.
    vi.mocked(api.workflowRevisionOutputGeometry).mockResolvedValue(capability());

    renderControl({ sizeIsTheWorkflowsOwn: true });

    await waitFor(() => expect(screen.getByRole("button", { name: "16:9 Wide" })).toBeTruthy());
    expect(screen.queryByText("This workflow sets the picture size itself.")).toBeNull();
  });
});
