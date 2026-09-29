/** Changing the canvas: a size, a place for the picture, and a fill for the rest. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioCanvasSizeTool } from "./StudioCanvasSizeTool";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { useStudioImage } from "./useStudioImage";

vi.mock("./api", () => ({
  api: {
    openStudioSession: vi.fn(),
    studioSession: vi.fn(),
    studioLocalEdit: vi.fn(),
    sendTurn: vi.fn(),
    upload: vi.fn(),
    favoriteArtifact: vi.fn(),
    artifact: vi.fn(),
    editTemplates: vi.fn(),
    studioCapabilities: vi.fn(),
  },
}));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

function tool(onChange = vi.fn(), busy = false) {
  render(<StudioCanvasSizeTool size={{ width: 400, height: 200 }} busy={busy} onChange={onChange} />);
  return {
    width: screen.getByRole("spinbutton", { name: "Width" }),
    height: screen.getByRole("spinbutton", { name: "Height" }),
    change: screen.getByRole("button", { name: "Change the canvas" }),
  };
}

describe("the canvas panel", () => {
  it("starts at the picture's size, centered on transparency, and waits for a new size", () => {
    const onChange = vi.fn();
    const { width, height, change } = tool(onChange);

    expect(width).toHaveValue(400);
    expect(height).toHaveValue(200);
    expect(screen.getByRole("button", { name: "Center" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Transparent" })).toHaveAttribute("aria-pressed", "true");
    expect(change).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(change);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("sends the size, where the picture sits, and the fill", () => {
    const onChange = vi.fn();
    const { height, change } = tool(onChange);

    fireEvent.change(height, { target: { value: "400" } });
    fireEvent.click(screen.getByRole("button", { name: "Top" }));
    fireEvent.click(screen.getByRole("button", { name: "White" }));

    expect(screen.getByText("The canvas goes from 400 × 200 to 400 × 400.")).toBeInTheDocument();
    fireEvent.click(change);
    expect(onChange).toHaveBeenCalledWith({ width: 400, height: 400, anchor: "top", fill: "white" });
  });

  it("says when a smaller canvas cuts part of the picture away", () => {
    const { width } = tool();

    fireEvent.change(width, { target: { value: "300" } });

    expect(screen.getByText(/What falls outside it is cut away\./)).toBeInTheDocument();
  });

  it("refuses a size that is not a whole number of pixels, and a press while busy", () => {
    const onChange = vi.fn();
    const { width, change } = tool(onChange);
    fireEvent.change(width, { target: { value: "0" } });
    fireEvent.click(change);
    expect(screen.getByText("Enter a width and a height of at least one pixel.")).toBeInTheDocument();
    cleanup();

    const busy = tool(onChange, true);
    fireEvent.change(busy.width, { target: { value: "500" } });
    fireEvent.click(busy.change);

    expect(screen.getByText("Applying…")).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("changing the canvas in the studio", () => {
  it("sends the new canvas for the picture on screen", async () => {
    const session = { id: "chat-studio", messages: [] } as never;
    vi.mocked(api.openStudioSession).mockResolvedValue(session);
    vi.mocked(api.studioSession).mockResolvedValue(session);
    vi.mocked(api.studioLocalEdit).mockResolvedValue(session);
    vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
    vi.mocked(api.editTemplates).mockResolvedValue([]);
    vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
    vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: /^Change the canvas size/ }));
    expect(screen.queryByRole("button", { name: "Apply edit" })).toBeNull();
    fireEvent.change(screen.getByRole("spinbutton", { name: "Width" }), { target: { value: "600" } });
    fireEvent.click(screen.getByRole("button", { name: "Change the canvas" }));

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1",
      operation: "canvas",
      canvas: { width: 600, height: 200, anchor: "center", fill: "transparent" },
    });
  });
});
