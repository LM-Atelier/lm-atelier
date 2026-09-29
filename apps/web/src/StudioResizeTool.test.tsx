/** Resizing: a width and a height, in proportion unless that is switched off. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioResizeTool } from "./StudioResizeTool";
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

function tool(onResize = vi.fn(), busy = false) {
  render(<StudioResizeTool size={{ width: 400, height: 200 }} busy={busy} onResize={onResize} />);
  return {
    width: screen.getByRole("spinbutton", { name: "Width" }),
    height: screen.getByRole("spinbutton", { name: "Height" }),
    resize: screen.getByRole("button", { name: "Resize" }),
  };
}

describe("the resize tool", () => {
  it("starts at the picture's size and does nothing until the size changes", () => {
    const onResize = vi.fn();
    const { width, height, resize } = tool(onResize);

    expect(width).toHaveValue(400);
    expect(height).toHaveValue(200);
    expect(screen.getByText("Now 400 × 200 pixels.")).toBeInTheDocument();
    expect(resize).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(resize);
    expect(onResize).not.toHaveBeenCalled();
  });

  it("keeps the proportions while either side changes", () => {
    const onResize = vi.fn();
    const { width, height, resize } = tool(onResize);

    fireEvent.change(width, { target: { value: "100" } });
    expect(height).toHaveValue(50);
    fireEvent.change(height, { target: { value: "150" } });
    expect(width).toHaveValue(300);

    expect(screen.getByText("400 × 200 becomes 300 × 150.")).toBeInTheDocument();
    fireEvent.click(resize);
    expect(onResize).toHaveBeenCalledWith({ width: 300, height: 150 });
  });

  it("changes one side alone once the proportions are let go", () => {
    const onResize = vi.fn();
    const { width, height, resize } = tool(onResize);

    fireEvent.click(screen.getByRole("checkbox", { name: "Keep proportions" }));
    fireEvent.change(width, { target: { value: "500" } });

    expect(height).toHaveValue(200);
    // Growing is allowed, and says what it cannot do.
    expect(screen.getByText(/Growing only spreads the pixels it has; Enhance adds detail\./)).toBeInTheDocument();
    fireEvent.click(resize);
    expect(onResize).toHaveBeenCalledWith({ width: 500, height: 200 });
  });

  it("refuses a side that is not a whole number of pixels", () => {
    const onResize = vi.fn();
    const { width, resize } = tool(onResize);

    for (const value of ["0", "12.5", ""]) {
      fireEvent.change(width, { target: { value } });
      expect(screen.getByText("Enter a width and a height of at least one pixel.")).toBeInTheDocument();
      expect(resize).toHaveAttribute("aria-disabled", "true");
      fireEvent.click(resize);
    }
    expect(onResize).not.toHaveBeenCalled();
  });

  it("refuses a press while another edit is arriving", () => {
    const onResize = vi.fn();
    const { width, resize } = tool(onResize, true);

    fireEvent.change(width, { target: { value: "100" } });
    fireEvent.click(resize);

    expect(screen.getByText("Applying…")).toBeInTheDocument();
    expect(onResize).not.toHaveBeenCalled();
  });
});

describe("resizing in the studio", () => {
  it("sends the new size for the picture on screen", async () => {
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

    fireEvent.click(await screen.findByRole("button", { name: /^Resize the picture/ }));
    expect(screen.queryByRole("button", { name: "Apply edit" })).toBeNull();
    fireEvent.change(screen.getByRole("spinbutton", { name: "Width" }), { target: { value: "200" } });
    fireEvent.click(screen.getByRole("button", { name: "Resize" }));

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1",
      operation: "resize",
      size: { width: 200, height: 100 },
    });
  });
});
