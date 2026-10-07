/** Blurring or pixelating a marked area: brushed or selected, then changed with one press. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import type { ComponentProps } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioBlurTool } from "./StudioBlurTool";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { createMask, encodeMaskPng } from "./studioMasks";
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

function panel(props: Partial<ComponentProps<typeof StudioBlurTool>>) {
  return render(
    <StudioBlurTool mask={createMask(40, 20)} maskVersion={0} featherPx={0} style="blur" radius={12}
      block={12} busy={false} onStyle={vi.fn()} onRadius={vi.fn()} onBlock={vi.fn()} onBlur={vi.fn()}
      onPixelate={vi.fn()} {...props} />,
  );
}

function marked() {
  const mask = createMask(40, 20);
  mask.data.fill(255, 0, 40 * 10);
  return mask;
}

describe("the blur panel", () => {
  it("asks for a marked area before it will blur", () => {
    const onBlur = vi.fn();
    panel({ onBlur });

    fireEvent.click(screen.getByRole("button", { name: "Blur the marked area" }));

    expect(screen.getByText("Brush over the part to blur, or select it with another tool first.")).toBeInTheDocument();
    expect(onBlur).not.toHaveBeenCalled();
  });

  it("hands over the marked area as a picture, and says so when it cannot", async () => {
    const onBlur = vi.fn();
    const selection = new Blob(["mask"], { type: "image/png" });
    vi.mocked(encodeMaskPng).mockResolvedValueOnce(selection).mockResolvedValueOnce(null);
    panel({ mask: marked(), maskVersion: 1, onBlur });
    const blur = screen.getByRole("button", { name: "Blur the marked area" });

    fireEvent.click(blur);
    await waitFor(() => expect(onBlur).toHaveBeenCalledWith(selection));
    fireEvent.click(blur);

    expect(await screen.findByRole("alert")).toHaveTextContent("The marked area could not be prepared. Mark it again.");
    expect(onBlur).toHaveBeenCalledTimes(1);
  });

  it("reports the strength as it moves", () => {
    const onRadius = vi.fn();
    panel({ mask: null, radius: 30, onRadius });

    fireEvent.change(screen.getByRole("slider", { name: "Blur strength" }), { target: { value: "45" } });

    expect(screen.getByText("30 px")).toBeInTheDocument();
    expect(onRadius).toHaveBeenCalledWith(45);
  });

  it("switches to pixelating, with its own block size", () => {
    const onStyle = vi.fn();
    const onBlock = vi.fn();
    panel({ mask: null, style: "pixelate", block: 16, onStyle, onBlock });

    fireEvent.change(screen.getByRole("slider", { name: "Block size" }), { target: { value: "24" } });
    fireEvent.click(screen.getByRole("button", { name: "Blur" }));

    expect(screen.getByRole("button", { name: "Pixelate" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("slider", { name: "Blur strength" })).toBeNull();
    expect(screen.getByText("16 px")).toBeInTheDocument();
    expect(onBlock).toHaveBeenCalledWith(24);
    expect(onStyle).toHaveBeenCalledWith("blur");
  });

  it("hands the marked area to the pixelation, not the blur, while pixelating", async () => {
    const onBlur = vi.fn();
    const onPixelate = vi.fn();
    const selection = new Blob(["mask"], { type: "image/png" });
    vi.mocked(encodeMaskPng).mockResolvedValueOnce(selection);
    panel({ mask: marked(), maskVersion: 1, style: "pixelate", onBlur, onPixelate });

    fireEvent.click(screen.getByRole("button", { name: "Pixelate the marked area" }));

    await waitFor(() => expect(onPixelate).toHaveBeenCalledWith(selection));
    expect(onBlur).not.toHaveBeenCalled();
  });
});

describe("marking an area to blur", () => {
  it("marks with the brush and keeps the strength within its range", () => {
    let state = studioToolReducer(initialToolState(), { type: "image-changed", width: 40, height: 20 });
    state = studioToolReducer(state, { type: "select-tool", kind: "blur" });
    state = studioToolReducer(state, { type: "set-blur-radius", radius: 500 });

    expect(state.blurRadius).toBe(100);
    expect(toolFor(state, null)).toBeInstanceOf(BrushTool);
  });

  it("keeps pixelation blocks at two pixels or more", () => {
    let state = studioToolReducer(initialToolState(), { type: "set-blur-style", style: "pixelate" });
    state = studioToolReducer(state, { type: "set-pixel-block", block: 1 });

    expect(state.blurStyle).toBe("pixelate");
    expect(state.pixelBlock).toBe(2);
    expect(studioToolReducer(state, { type: "set-pixel-block", block: 500 }).pixelBlock).toBe(100);
  });
});

function openStudio() {
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
}

describe("blurring in the studio", () => {
  it("uploads the marked area and sends it with the strength", async () => {
    openStudio();

    fireEvent.click(await screen.findByRole("button", { name: /^Blur or pixelate part of the picture/ }));
    expect(screen.queryByRole("button", { name: "Apply edit" })).toBeNull();
    // One dab of the brush in the middle of the picture, from the keyboard.
    const canvas = screen.getByRole("application");
    fireEvent.keyDown(canvas, { key: "Enter" });
    fireEvent.keyDown(canvas, { key: "Enter" });
    fireEvent.click(await screen.findByRole("button", { name: "Blur the marked area" }));

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    const uploaded = vi.mocked(api.upload).mock.calls[0][0] as File;
    expect(uploaded.name).toBe("studio-selection.png");
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1",
      operation: "blur",
      blur: { mask_artifact_id: "sha256:mask", radius: 12 },
    });
  });

  it("sends a pixelation of the marked area with its block size", async () => {
    openStudio();

    fireEvent.click(await screen.findByRole("button", { name: /^Blur or pixelate part of the picture/ }));
    fireEvent.click(screen.getByRole("button", { name: "Pixelate" }));
    const canvas = screen.getByRole("application");
    fireEvent.keyDown(canvas, { key: "Enter" });
    fireEvent.keyDown(canvas, { key: "Enter" });
    fireEvent.click(await screen.findByRole("button", { name: "Pixelate the marked area" }));

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1",
      operation: "pixelate",
      pixelate: { mask_artifact_id: "sha256:mask", block: 12 },
    });
  });
});
