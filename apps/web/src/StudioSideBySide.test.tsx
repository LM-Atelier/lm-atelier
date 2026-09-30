/** Two pictures next to each other, zoomed and moved as one, from the viewer and through the studio. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StudioSideBySide } from "./StudioSideBySide";
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

function bitmap(width: number, height: number): ImageBitmap {
  return { width, height, close: vi.fn() } as unknown as ImageBitmap;
}

/** The translation and scale a canvas is drawn with. */
function placed(canvas: HTMLCanvasElement): { x: number; y: number; scale: number } {
  const match = /translate\((-?[\d.e-]+)px, (-?[\d.e-]+)px\) scale\(([\d.e-]+)\)/.exec(canvas.style.transform);
  if (!match) throw new Error(`no transform: ${canvas.style.transform}`);
  return { x: Number(match[1]), y: Number(match[2]), scale: Number(match[3]) };
}

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  // Each half is 200 by 200 on screen; jsdom lays nothing out.
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockImplementation(function (this: HTMLElement) {
    return this.classList.contains("studio-side-view") ? 200 : 0;
  });
  vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
    return this.classList.contains("studio-side-view") ? 200 : 0;
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("the side-by-side viewer", () => {
  function show() {
    render(
      <StudioSideBySide
        before={{ image: bitmap(400, 200), label: "What it was made from" }}
        after={{ image: bitmap(800, 400), label: "This result" }}
      />,
    );
    const [left, right] = [...document.querySelectorAll("canvas")] as HTMLCanvasElement[];
    return { left, right, viewer: screen.getByRole("application", { name: /beside This result/ }) };
  }

  /** Where each picture is on screen: its corner and its drawn width. */
  const onScreen = (canvas: HTMLCanvasElement, width: number) => {
    const at = placed(canvas);
    return { x: at.x, y: at.y, width: width * at.scale };
  };

  it("names both pictures and fits each whole in its half", () => {
    const { left, right } = show();

    expect(screen.getByText("What it was made from")).toBeInTheDocument();
    expect(screen.getByText("This result")).toBeInTheDocument();
    expect(onScreen(left, 400)).toEqual({ x: 0, y: 50, width: 200 });
    expect(onScreen(right, 800)).toEqual({ x: 0, y: 50, width: 200 });
  });

  it("zooms and moves both halves as one from the keyboard, and zero shows each whole again", () => {
    const { left, right, viewer } = show();

    fireEvent.keyDown(viewer, { key: "+" });
    fireEvent.keyDown(viewer, { key: "ArrowRight" });

    const zoomedLeft = onScreen(left, 400);
    const zoomedRight = onScreen(right, 800);
    expect(zoomedLeft.width).toBeCloseTo(240);
    expect(zoomedRight.width).toBeCloseTo(zoomedLeft.width);
    expect(zoomedRight.x).toBeCloseTo(zoomedLeft.x);
    // Zoomed about the middle, then moved to show more of the right.
    expect(zoomedLeft.x).toBeCloseTo(-40);

    fireEvent.keyDown(viewer, { key: "0" });
    expect(onScreen(left, 400)).toEqual({ x: 0, y: 50, width: 200 });
  });

  it("zooms both about the pointer when either half is wheeled, and drags both together", () => {
    const { left, right } = show();
    const halves = document.querySelectorAll(".studio-side-view");

    fireEvent.wheel(halves[1], { deltaY: -100, clientX: 100, clientY: 100 });
    const zoomed = onScreen(left, 400);
    expect(zoomed.width).toBeCloseTo(240);
    expect(onScreen(right, 800).width).toBeCloseTo(240);

    fireEvent(halves[0], new MouseEvent("pointerdown", { bubbles: true, button: 0, clientX: 50, clientY: 50 }));
    fireEvent(halves[0], new MouseEvent("pointermove", { bubbles: true, clientX: 80, clientY: 40 }));
    fireEvent(halves[0], new MouseEvent("pointerup", { bubbles: true, clientX: 80, clientY: 40 }));

    expect(onScreen(left, 400).x).toBeCloseTo(zoomed.x + 30);
    expect(onScreen(right, 800).x).toBeCloseTo(zoomed.x + 30);
    expect(onScreen(right, 800).y).toBeCloseTo(zoomed.y - 10);
  });
});

describe("side by side in the studio", () => {
  const source: StudioStep = {
    messageId: "source", artifactId: "art-source", instruction: "", beforeArtifactId: null, isSource: true, generationIdentity: null,
  };
  const first: StudioStep = {
    messageId: "answer-1", artifactId: "art-1", instruction: "brighter sky", beforeArtifactId: "art-source", isSource: false,
    generationIdentity: null,
  };
  const second: StudioStep = { ...first, messageId: "answer-2", artifactId: "art-2", instruction: "warmer light" };

  function open(steps: StudioStep[], sizes: Record<string, [number, number]>) {
    vi.mocked(api.artifact).mockResolvedValue({ id: "art", favorite: false } as never);
    vi.mocked(api.editTemplates).mockResolvedValue([]);
    vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
    const bitmaps = new Map(Object.entries(sizes).map(([id, [width, height]]) => [id, bitmap(width, height)]));
    vi.mocked(useStudioImage).mockImplementation((artifactId: string | null) => ({
      bitmap: (artifactId && bitmaps.get(artifactId)) || null,
      error: null,
      reload: vi.fn(),
    }));
    vi.mocked(useStudioSession).mockReturnValue({
      steps, previewArtifactId: null, sessionId: "chat-studio", session: null, busy: false, error: null, apply: vi.fn(),
    } as unknown as ReturnType<typeof useStudioSession>);
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <StudioView sourceArtifactId="art-source" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
      </QueryClientProvider>,
    );
  }

  const canvas = () => screen.queryByRole("application", { name: /^Image editing canvas/ });
  const sideBySide = () => screen.queryByRole("application", { name: /beside This result/ });

  it("puts a result beside a picture of another shape, which a split cannot line up", () => {
    // An extended picture: wider than the one it was made from.
    open([source, first], { "art-source": [400, 200], "art-1": [600, 200] });
    expect(screen.queryByRole("button", { name: "Split" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /Side by side/ }));

    expect(sideBySide()).toBeInTheDocument();
    expect(canvas()).toBeNull();
    expect(screen.getByText("What it was made from")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Side by side/ })).toHaveAttribute("aria-pressed", "true");
    // Holding lays one picture over the other, which is not what is on show.
    expect(screen.queryByRole("button", { name: "Hold to compare" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /Side by side/ }));
    expect(sideBySide()).toBeNull();
    expect(canvas()).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hold to compare" })).toBeInTheDocument();
  });

  it("names the picture chosen from the strip, and gives way to a split", () => {
    open([source, first, second], { "art-source": [400, 200], "art-1": [400, 200], "art-2": [400, 200] });
    fireEvent.change(screen.getByRole("combobox", { name: "Compare with" }), { target: { value: "art-1" } });
    fireEvent.click(screen.getByRole("button", { name: /Side by side/ }));

    expect(screen.getByText("Step 1 · brighter sky", { selector: "figcaption" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Split" }));
    expect(sideBySide()).toBeNull();
    expect(screen.getByRole("button", { name: /Side by side/ })).toHaveAttribute("aria-pressed", "false");
  });

  it("is not offered while a tool marks the picture, which needs the canvas", () => {
    open([source, first], { "art-source": [400, 200], "art-1": [400, 200] });
    expect(screen.getByRole("button", { name: /Side by side/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^Select part of the picture/ }));

    expect(screen.queryByRole("button", { name: /Side by side/ })).toBeNull();
    expect(canvas()).toBeInTheDocument();
  });
});
