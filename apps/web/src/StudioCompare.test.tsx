/** Comparing a result with the picture it was made from, from the controls and through the studio. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioCompare } from "./StudioCompare";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession, type StudioStep } from "./useStudioSession";

vi.mock("./api", () => ({
  api: { favoriteArtifact: vi.fn(), artifact: vi.fn(), editTemplates: vi.fn(), studioCapabilities: vi.fn() },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

/** A press or release of one mouse button.
 *
 * jsdom's pointer events carry no button, which would let a check for the
 * main button pass by default; a mouse event under the pointer's name does.
 */
function pointer(target: Element, type: "pointerdown" | "pointerup", button = 0) {
  fireEvent(target, new MouseEvent(type, { bubbles: true, cancelable: true, button }));
}

describe("the compare controls", () => {
  function controls(overrides: Partial<Parameters<typeof StudioCompare>[0]> = {}) {
    const onHold = vi.fn();
    const onSplit = vi.fn();
    render(<StudioCompare holding={false} onHold={onHold} split={null} onSplit={onSplit} canSplit {...overrides} />);
    return { onHold, onSplit, hold: screen.getByRole("button", { name: "Hold to compare" }) };
  }

  it("shows the earlier picture only while the button is held", () => {
    const { onHold, hold } = controls();
    pointer(hold, "pointerdown");
    expect(onHold).toHaveBeenLastCalledWith(true);
    pointer(hold, "pointerup");
    expect(onHold).toHaveBeenLastCalledWith(false);
  });

  it("holds from the keyboard, and a repeating key does not start it again", () => {
    const { onHold, hold } = controls();
    fireEvent.keyDown(hold, { key: " " });
    fireEvent.keyDown(hold, { key: " ", repeat: true });
    expect(onHold.mock.calls).toEqual([[true]]);
    fireEvent.keyUp(hold, { key: " " });
    expect(onHold).toHaveBeenLastCalledWith(false);
  });

  it("lets go when the pointer is lost or focus moves away", () => {
    const { onHold, hold } = controls();
    pointer(hold, "pointerdown");
    fireEvent.lostPointerCapture(hold, { pointerId: 1 });
    expect(onHold).toHaveBeenLastCalledWith(false);
    fireEvent.keyDown(hold, { key: "Enter" });
    fireEvent.blur(hold);
    expect(onHold).toHaveBeenLastCalledWith(false);
  });

  it("does not hold on a secondary button", () => {
    const { onHold, hold } = controls();
    pointer(hold, "pointerdown", 2);
    expect(onHold).not.toHaveBeenCalled();
  });

  it("splits at the middle, moves the divider, and closes the split", () => {
    const first = controls();
    fireEvent.click(screen.getByRole("button", { name: "Split" }));
    expect(first.onSplit).toHaveBeenCalledWith(0.5);
    cleanup();

    const second = controls({ split: 0.5 });
    expect(screen.getByRole("button", { name: "Split" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.change(screen.getByRole("slider", { name: "Divider position" }), { target: { value: "30" } });
    expect(second.onSplit).toHaveBeenLastCalledWith(0.3);
    fireEvent.click(screen.getByRole("button", { name: "Split" }));
    expect(second.onSplit).toHaveBeenLastCalledWith(null);
  });

  it("offers no split between pictures of different shapes", () => {
    controls({ canSplit: false, split: 0.5 });
    expect(screen.queryByRole("button", { name: "Split" })).toBeNull();
    expect(screen.queryByRole("slider")).toBeNull();
  });
});

describe("comparing in the studio", () => {
  const source: StudioStep = {
    messageId: "source",
    artifactId: "art-source",
    instruction: "",
    beforeArtifactId: null,
    isSource: true,
    generationIdentity: null,
  };
  const first: StudioStep = {
    messageId: "answer-1",
    artifactId: "art-1",
    instruction: "brighter sky",
    beforeArtifactId: "art-source",
    isSource: false,
    generationIdentity: null,
  };
  // Made from the source again, so the picture before it in the strip is not
  // the one it was made from.
  const second: StudioStep = { ...first, messageId: "answer-2", artifactId: "art-2", instruction: "warmer light" };

  function open(steps: StudioStep[], sizes: Record<string, [number, number]>) {
    vi.mocked(api.artifact).mockResolvedValue({ id: "art", favorite: false } as never);
    vi.mocked(api.editTemplates).mockResolvedValue([]);
    vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    // One decoded picture per artifact, as the real hook keeps: a new object
    // on every render would read as a new picture each time.
    const bitmaps = new Map(
      Object.entries(sizes).map(([id, [width, height]]) => [
        id,
        { width, height, close: vi.fn() } as unknown as ImageBitmap,
      ]),
    );
    const reload = vi.fn();
    vi.mocked(useStudioImage).mockImplementation((artifactId: string | null) => ({
      bitmap: (artifactId && bitmaps.get(artifactId)) || null,
      error: null,
      reload,
    }));
    const sessionWith = (current: StudioStep[]) =>
      vi.mocked(useStudioSession).mockReturnValue({
        steps: current,
        previewArtifactId: null,
        sessionId: "chat-studio",
        session: null,
        busy: false,
        error: null,
        apply: vi.fn(),
      } as unknown as ReturnType<typeof useStudioSession>);
    sessionWith(steps);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const [onOpenArtifact, onOpenWorkflows, onClose] = [vi.fn(), vi.fn(), vi.fn()];
    // A new element each time: handed the same one, React would skip the render.
    const studio = () => (
      <QueryClientProvider client={client}>
        <StudioView
          sourceArtifactId="art-source"
          onOpenArtifact={onOpenArtifact}
          onOpenWorkflows={onOpenWorkflows}
          onClose={onClose}
        />
      </QueryClientProvider>
    );
    const view = render(studio());
    // A result arriving: the session answers with more steps.
    return (next: StudioStep[]) => {
      sessionWith(next);
      view.rerender(studio());
    };
  }

  const earlierLayer = () => document.querySelector('canvas[data-layer="before"]') as HTMLCanvasElement | null;

  it("holds and splits the result against the picture it was made from, on the canvas", () => {
    open([source, first, second], { "art-source": [400, 200], "art-1": [400, 200], "art-2": [400, 200] });
    // The newest result is on the canvas, and the picture it was made from is
    // the source rather than the result beside it in the strip.
    expect(useStudioImage).toHaveBeenCalledWith("art-source");
    expect(useStudioImage).not.toHaveBeenCalledWith("art-1");
    const hold = screen.getByRole("button", { name: "Hold to compare" });
    expect(earlierLayer()!.style.clipPath).toBe("inset(0 100% 0 0)");

    pointer(hold, "pointerdown");
    expect(hold).toHaveAttribute("aria-pressed", "true");
    expect(earlierLayer()!.style.clipPath).toBe("inset(0 0% 0 0)");
    pointer(hold, "pointerup");
    expect(earlierLayer()!.style.clipPath).toBe("inset(0 100% 0 0)");

    fireEvent.click(screen.getByRole("button", { name: "Split" }));
    expect(earlierLayer()!.style.clipPath).toBe("inset(0 50% 0 0)");
    fireEvent.change(screen.getByRole("slider", { name: "Divider position" }), { target: { value: "20" } });
    expect(earlierLayer()!.style.clipPath).toBe("inset(0 80% 0 0)");

    // Holding shows the whole earlier picture even with the split open, and
    // letting go returns to the divider where it was left.
    pointer(hold, "pointerdown");
    expect(earlierLayer()!.style.clipPath).toBe("inset(0 0% 0 0)");
    pointer(hold, "pointerup");
    expect(earlierLayer()!.style.clipPath).toBe("inset(0 80% 0 0)");
  });

  it("keeps a hold with the result it began on when a newer one arrives", () => {
    const arrive = open([source, first], { "art-source": [400, 200], "art-1": [400, 200], "art-2": [400, 200] });
    fireEvent.keyDown(screen.getByRole("button", { name: "Hold to compare" }), { key: " " });
    expect(earlierLayer()!.style.clipPath).toBe("inset(0 0% 0 0)");

    // An edit still running finishes while the key is down, and its result
    // takes the canvas.
    arrive([source, first, second]);

    expect(screen.getByRole("button", { name: "Hold to compare" })).toHaveAttribute("aria-pressed", "false");
    expect(earlierLayer()!.style.clipPath).toBe("inset(0 100% 0 0)");
  });

  it("offers nothing to compare on the original", () => {
    open([source, first], { "art-source": [400, 200], "art-1": [400, 200] });
    expect(screen.getByRole("button", { name: "Hold to compare" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /The original image/ }));

    expect(screen.queryByRole("button", { name: "Hold to compare" })).toBeNull();
    expect(earlierLayer()).toBeNull();
  });

  it("keeps to holding when the edit changed the picture's shape", () => {
    // An extended picture: the earlier one no longer lines up across a divider.
    open([source, first], { "art-source": [400, 200], "art-1": [600, 200] });

    expect(screen.getByRole("button", { name: "Hold to compare" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Split" })).toBeNull();
  });

  it("offers no comparison until the earlier picture can be shown", () => {
    // The source cannot be read, so holding would show nothing.
    open([source, first], { "art-1": [400, 200] });

    expect(screen.queryByRole("button", { name: "Hold to compare" })).toBeNull();
    expect(earlierLayer()).toBeNull();
  });

  it("offers no comparison while the result itself cannot be shown", () => {
    // The result did not decode, so the stage says so and there is no canvas
    // for the earlier picture to be laid over.
    open([source, first], { "art-source": [400, 200] });

    expect(screen.queryByRole("button", { name: "Hold to compare" })).toBeNull();
    expect(earlierLayer()).toBeNull();
  });
});
