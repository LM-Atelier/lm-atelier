/** Light and color: sliders previewed on the canvas, kept only when applied. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioAdjustTool } from "./StudioAdjustTool";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { adjustPixels, NEUTRAL_ADJUSTMENTS } from "./studioAdjustments";
import { autoAdjustments } from "./studioAutoAdjust";
import { readSourcePixels } from "./studioSourcePixels";
import { initialToolState, studioToolReducer } from "./studioToolState";
import { useAdjustedPreview } from "./useAdjustedPreview";
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
vi.mock("./studioSourcePixels", () => ({ readSourcePixels: vi.fn() }));
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
  vi.unstubAllGlobals();
});

describe("the light and color panel", () => {
  it("offers nothing to reset or apply until a slider moves", () => {
    const onApply = vi.fn();
    const onReset = vi.fn();
    render(<StudioAdjustTool adjustments={NEUTRAL_ADJUSTMENTS} busy={false} onChange={vi.fn()} onReset={onReset} onApply={onApply} />);

    expect(screen.getByText("Move a slider to see the change on the picture.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    fireEvent.click(screen.getByRole("button", { name: "Apply adjustments" }));

    expect(screen.getByRole("button", { name: "Apply adjustments" })).toHaveAttribute("aria-disabled", "true");
    expect(onReset).not.toHaveBeenCalled();
    expect(onApply).not.toHaveBeenCalled();
  });

  it("reports each slider and applies or resets once one has moved", () => {
    const onChange = vi.fn();
    const onApply = vi.fn();
    const onReset = vi.fn();
    render(
      <StudioAdjustTool adjustments={{ ...NEUTRAL_ADJUSTMENTS, warmth: -15 }} busy={false}
        onChange={onChange} onReset={onReset} onApply={onApply} />,
    );

    expect(screen.getByText("-15")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("slider", { name: "Contrast" }), { target: { value: "30" } });
    fireEvent.change(screen.getByRole("slider", { name: "Highlights" }), { target: { value: "-20" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply adjustments" }));
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));

    expect(onChange).toHaveBeenCalledWith("contrast", 30);
    expect(onChange).toHaveBeenCalledWith("highlights", -20);
    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onReset).toHaveBeenCalledTimes(1);
  });

  it("refuses to apply while another edit is arriving", () => {
    const onApply = vi.fn();
    render(
      <StudioAdjustTool adjustments={{ ...NEUTRAL_ADJUSTMENTS, brightness: 5 }} busy
        onChange={vi.fn()} onReset={vi.fn()} onApply={onApply} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Apply adjustments" }));

    expect(screen.getByText("Applying…")).toBeInTheDocument();
    expect(onApply).not.toHaveBeenCalled();
  });

  it("offers Auto only where the picture's colors can be read, and not while an edit arrives", () => {
    const onAuto = vi.fn();
    const panel = (busy: boolean, auto?: () => void) => (
      <StudioAdjustTool adjustments={NEUTRAL_ADJUSTMENTS} busy={busy} onChange={vi.fn()} onReset={vi.fn()}
        onApply={vi.fn()} onAuto={auto} />
    );
    const { rerender } = render(panel(false));
    expect(screen.queryByRole("button", { name: "Auto" })).toBeNull();

    rerender(panel(false, onAuto));
    fireEvent.click(screen.getByRole("button", { name: "Auto" }));
    rerender(panel(true, onAuto));
    fireEvent.click(screen.getByRole("button", { name: "Auto" }));

    expect(onAuto).toHaveBeenCalledTimes(1);
  });
});

describe("where the sliders stand", () => {
  it("keeps whole steps within the range and starts over with each picture", () => {
    let state = studioToolReducer(initialToolState(), { type: "set-adjustment", key: "saturation", value: 40 });
    state = studioToolReducer(state, { type: "set-adjustment", key: "saturation", value: 101 });
    state = studioToolReducer(state, { type: "set-adjustment", key: "brightness", value: 2.5 });
    expect(state.adjustments).toEqual({ ...NEUTRAL_ADJUSTMENTS, saturation: 40 });

    // An applied adjustment is already in the next picture; keeping the
    // sliders would show it twice.
    state = studioToolReducer(state, { type: "image-changed", width: 4, height: 4 });
    expect(state.adjustments).toEqual(NEUTRAL_ADJUSTMENTS);
  });

  it("takes every slider at once, as Auto sets them, only when each is a whole step in range", () => {
    const auto = { ...NEUTRAL_ADJUSTMENTS, brightness: 31, contrast: 12, warmth: -8 };
    const state = studioToolReducer(initialToolState(), { type: "set-adjustments", adjustments: auto });

    expect(state.adjustments).toEqual(auto);
    expect(studioToolReducer(state, { type: "set-adjustments", adjustments: { ...auto, tint: 101 } })).toBe(state);
    expect(studioToolReducer(state, { type: "set-adjustments", adjustments: { ...auto, tint: 1.5 } })).toBe(state);
  });
});

describe("the preview on the canvas", () => {
  it("draws the adjusted pixels on the next frame, and nothing for sliders at zero", () => {
    const drawn: ImageData[] = [];
    vi.stubGlobal("ImageData", class {
      constructor(readonly data: Uint8ClampedArray, readonly width: number, readonly height: number) {}
    });
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      putImageData: (image: ImageData) => drawn.push(image),
    } as never);
    const bitmap = { width: 1, height: 1 } as ImageBitmap;
    const pixels = new Uint8ClampedArray([200, 100, 50, 255]);
    const sliders = { ...NEUTRAL_ADJUSTMENTS, saturation: -100 };

    const { result, rerender } = renderHook(
      ({ adjustments }) => useAdjustedPreview(bitmap, pixels, adjustments),
      { initialProps: { adjustments: sliders } },
    );

    expect(result.current).toBeInstanceOf(HTMLCanvasElement);
    expect(Array.from(drawn[0].data)).toEqual(Array.from(adjustPixels(pixels, 1, sliders)));
    rerender({ adjustments: NEUTRAL_ADJUSTMENTS });
    expect(result.current).toBeNull();
  });

  it("sharpens across rows as wide as the picture", () => {
    const drawn: ImageData[] = [];
    vi.stubGlobal("ImageData", class {
      constructor(readonly data: Uint8ClampedArray, readonly width: number, readonly height: number) {}
    });
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
      putImageData: (image: ImageData) => drawn.push(image),
    } as never);
    // A light pixel in the middle of a dark 3 by 3 picture: only its whole
    // neighborhood, read three pixels to a row, makes it lighter still.
    const bitmap = { width: 3, height: 3 } as ImageBitmap;
    const pixels = new Uint8ClampedArray(Array.from({ length: 9 }, (_, index) => (index === 4 ? [180, 180, 180, 255] : [60, 60, 60, 255])).flat());
    const sliders = { ...NEUTRAL_ADJUSTMENTS, sharpness: 50 };

    renderHook(() => useAdjustedPreview(bitmap, pixels, sliders));

    expect(Array.from(drawn[0].data)).toEqual(Array.from(adjustPixels(pixels, 3, sliders)));
    expect(drawn[0].data[16]).toBeGreaterThan(180);
  });
});

describe("adjusting in the studio", () => {
  it("sends where the sliders stand for the picture on screen", async () => {
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

    fireEvent.click(await screen.findByRole("button", { name: /^Adjust light and color/ }));
    expect(screen.queryByRole("button", { name: "Apply edit" })).toBeNull();
    act(() => {
      fireEvent.change(screen.getByRole("slider", { name: "Brightness" }), { target: { value: "25" } });
      fireEvent.change(screen.getByRole("slider", { name: "Shadows" }), { target: { value: "40" } });
      fireEvent.change(screen.getByRole("slider", { name: "Whites" }), { target: { value: "-25" } });
      fireEvent.change(screen.getByRole("slider", { name: "Blacks" }), { target: { value: "30" } });
      fireEvent.change(screen.getByRole("slider", { name: "Sharpness" }), { target: { value: "-30" } });
      fireEvent.change(screen.getByRole("slider", { name: "Vibrance" }), { target: { value: "15" } });
      fireEvent.change(screen.getByRole("slider", { name: "Vignette" }), { target: { value: "20" } });
      fireEvent.change(screen.getByRole("slider", { name: "Grain" }), { target: { value: "12" } });
    });
    // Grain is added or not: its slider starts at zero.
    expect(screen.getByRole("slider", { name: "Grain" })).toHaveAttribute("min", "0");
    expect(screen.getByRole("slider", { name: "Vignette" })).toHaveAttribute("min", "-100");
    fireEvent.click(screen.getByRole("button", { name: "Apply adjustments" }));

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1",
      operation: "adjust",
      adjustments: {
        brightness: 25, contrast: 0, highlights: 0, shadows: 40, whites: -25, blacks: 30, saturation: 0, warmth: 0,
        tint: 0, sharpness: -30, vibrance: 15, vignette: 20, grain: 12,
      },
    });
  });

  it("sets the sliders from the picture on screen with Auto, and sends where they then stand", async () => {
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
    // A dim grey picture, darker to the left.
    const pixels = new Uint8ClampedArray(400 * 200 * 4);
    for (let at = 0; at < pixels.length; at += 4) {
      const level = 10 + Math.round((160 * ((at / 4) % 400)) / 399);
      pixels.set([level, level, level, 255], at);
    }
    vi.mocked(readSourcePixels).mockReturnValue(pixels);
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: /^Adjust light and color/ }));
    fireEvent.click(screen.getByRole("button", { name: "Auto" }));
    const expected = autoAdjustments(pixels, NEUTRAL_ADJUSTMENTS);

    expect(expected.brightness).toBeGreaterThan(0);
    expect(screen.getByRole("slider", { name: "Brightness" })).toHaveValue(String(expected.brightness));
    fireEvent.click(screen.getByRole("button", { name: "Apply adjustments" }));
    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1", operation: "adjust", adjustments: expected,
    });
  });
});
