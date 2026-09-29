/** Painting a marked area: a color and an opacity, previewed in the marking itself. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioPaintTool } from "./StudioPaintTool";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { createMask, encodeMaskPng, toAlphaImageData } from "./studioMasks";
import { paintRgb } from "./studioPaint";
import { initialToolState, studioToolReducer, toolFor } from "./studioToolState";
import { BrushTool } from "./studioTools";
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
vi.mock("./studioMasks", async (original) => ({
  ...(await original<typeof import("./studioMasks")>()),
  encodeMaskPng: vi.fn(),
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

function marked() {
  const mask = createMask(40, 20);
  mask.data.fill(255, 0, 40 * 10);
  return mask;
}

describe("the paint panel", () => {
  it("offers colors at a press, any other color, and an opacity", () => {
    const onColor = vi.fn();
    const onOpacity = vi.fn();
    render(<StudioPaintTool mask={null} maskVersion={0} featherPx={0} color="#000000" opacity={100}
      busy={false} onColor={onColor} onOpacity={onOpacity} onPaint={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Black" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "Yellow" }));
    fireEvent.change(screen.getByLabelText("Another color"), { target: { value: "#AB12CD" } });
    fireEvent.change(screen.getByRole("slider", { name: "Opacity" }), { target: { value: "40" } });

    expect(onColor).toHaveBeenNthCalledWith(1, "#fdd835");
    expect(onColor).toHaveBeenNthCalledWith(2, "#ab12cd");
    expect(onOpacity).toHaveBeenCalledWith(40);
    expect(screen.getByText("Brush over the part to paint, or select it with another tool first.")).toBeInTheDocument();
  });

  it("hands over the marked area once there is one", async () => {
    const onPaint = vi.fn();
    const selection = new Blob(["mask"], { type: "image/png" });
    vi.mocked(encodeMaskPng).mockResolvedValue(selection);
    render(<StudioPaintTool mask={marked()} maskVersion={1} featherPx={0} color="#000000" opacity={100}
      busy={false} onColor={vi.fn()} onOpacity={vi.fn()} onPaint={onPaint} />);

    fireEvent.click(screen.getByRole("button", { name: "Paint the marked area" }));

    await waitFor(() => expect(onPaint).toHaveBeenCalledWith(selection));
  });
});

describe("the paint's color and cover", () => {
  it("keeps a valid color and an opacity within its range, and marks with the brush", () => {
    let state = studioToolReducer(initialToolState(), { type: "image-changed", width: 40, height: 20 });
    state = studioToolReducer(state, { type: "select-tool", kind: "paint" });
    state = studioToolReducer(state, { type: "set-paint-color", color: "#12ab34" });
    state = studioToolReducer(state, { type: "set-paint-color", color: "red" });
    state = studioToolReducer(state, { type: "set-paint-opacity", opacity: 0 });

    expect(state.paintColor).toBe("#12ab34");
    expect(state.paintOpacity).toBe(1);
    expect(toolFor(state, null)).toBeInstanceOf(BrushTool);
    expect(paintRgb("#12ab34")).toEqual([0x12, 0xab, 0x34]);
  });

  it("shows the marking at the paint's opacity, rounding halves up as the server does", () => {
    const mask = createMask(2, 1);
    mask.data[0] = 255;
    mask.data[1] = 1;

    const shown = toAlphaImageData(mask, [10, 20, 30], 0.5);

    expect(Array.from(shown)).toEqual([10, 20, 30, 128, 10, 20, 30, 1]);
  });
});

describe("painting in the studio", () => {
  it("uploads the marked area and sends it with the color and opacity", async () => {
    const session = { id: "chat-studio", messages: [] } as never;
    vi.mocked(api.openStudioSession).mockResolvedValue(session);
    vi.mocked(api.studioSession).mockResolvedValue(session);
    vi.mocked(api.studioLocalEdit).mockResolvedValue(session);
    vi.mocked(api.upload).mockResolvedValue({ id: "sha256:mask" } as never);
    vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
    vi.mocked(api.editTemplates).mockResolvedValue([]);
    vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
    vi.mocked(encodeMaskPng).mockResolvedValue(new Blob(["mask"], { type: "image/png" }));
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
    vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: /^Paint over part of the picture/ }));
    expect(screen.queryByRole("button", { name: "Apply edit" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Red" }));
    const canvas = screen.getByRole("application");
    fireEvent.keyDown(canvas, { key: "Enter" });
    fireEvent.keyDown(canvas, { key: "Enter" });
    fireEvent.click(await screen.findByRole("button", { name: "Paint the marked area" }));

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1",
      operation: "paint",
      paint: { mask_artifact_id: "sha256:mask", color: "#e53935", opacity: 100 },
    });
  });
});
