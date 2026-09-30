/** Adding words: drawn in the browser, previewed on the canvas, and added as drawn. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioCaptionTool } from "./StudioCaptionTool";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { captionFontPx, captionLayout, DEFAULT_CAPTION, drawCaption } from "./studioCaption";
import { initialToolState, snapshotBeforeGesture, studioToolReducer, toolFor } from "./studioToolState";
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
vi.mock("./studioCaption", async (original) => ({
  ...(await original<typeof import("./studioCaption")>()),
  drawCaption: vi.fn(),
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

/** One mouse event at a place, under the pointer event's name: jsdom has no PointerEvent. */
function pointer(target: Element, type: string, x: number, y: number) {
  const event = new MouseEvent(type, { clientX: x, clientY: y, button: 0, bubbles: true });
  Object.defineProperty(event, "pointerId", { value: 1 });
  fireEvent(target, event);
}

function drawn(words = new Blob(["words"], { type: "image/png" })) {
  return { toBlob: (done: (blob: Blob | null) => void) => done(words) } as unknown as HTMLCanvasElement;
}

describe("where the words go", () => {
  it("keeps a margin from the edges they are anchored to and places lines as one block", () => {
    // 400 by 200: the margin is four percent of the shorter side, 8 pixels.
    expect(captionLayout(400, 200, 1, 20, "top_left")).toEqual({ x: 8, align: "left", tops: [8] });
    expect(captionLayout(400, 200, 2, 20, "bottom")).toEqual({ x: 200, align: "center", tops: [144, 168] });
    expect(captionLayout(400, 200, 1, 20, "right")).toEqual({ x: 392, align: "right", tops: [88] });
  });

  it("sizes the letters by the picture's height, never below eight pixels", () => {
    expect(captionFontPx(1000, 8)).toBe(80);
    expect(captionFontPx(50, 2)).toBe(8);
  });
});

describe("the words panel", () => {
  it("asks for words before it will add any", () => {
    const onAdd = vi.fn();
    render(<StudioCaptionTool caption={DEFAULT_CAPTION} size={{ width: 400, height: 200 }} busy={false}
      onChange={vi.fn()} onAdd={onAdd} />);

    fireEvent.click(screen.getByRole("button", { name: "Add the words" }));

    expect(screen.getByText("Write the words to add, then choose where they sit.")).toBeInTheDocument();
    expect(onAdd).not.toHaveBeenCalled();
  });

  it("turns the words with a slider and says they can be dragged once written", () => {
    const onChange = vi.fn();
    render(<StudioCaptionTool caption={{ ...DEFAULT_CAPTION, text: "Harbour", turn: 12 }} size={{ width: 400, height: 200 }}
      busy={false} onChange={onChange} onAdd={vi.fn()} />);

    expect(screen.getByText("12°")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("slider", { name: "Turn" }), { target: { value: "-30" } });

    expect(onChange).toHaveBeenCalledWith({ turn: -30 });
    expect(screen.getByText(/Drag them on the picture to move them\./)).toBeInTheDocument();
  });

  it("reports each choice and hands over the words as drawn", async () => {
    const onChange = vi.fn();
    const onAdd = vi.fn();
    const words = new Blob(["words"], { type: "image/png" });
    vi.mocked(drawCaption).mockResolvedValue(drawn(words));
    const caption = { ...DEFAULT_CAPTION, text: "Harbour" };
    render(<StudioCaptionTool caption={caption} size={{ width: 400, height: 200 }} busy={false}
      onChange={onChange} onAdd={onAdd} />);

    fireEvent.change(screen.getByRole("textbox", { name: "Words" }), { target: { value: "Harbour at dawn" } });
    fireEvent.click(screen.getByRole("button", { name: "Serif" }));
    fireEvent.change(screen.getByRole("slider", { name: "Size" }), { target: { value: "12" } });
    fireEvent.click(screen.getByRole("button", { name: "Yellow" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Outline" }));
    fireEvent.click(screen.getByRole("checkbox", { name: "Shadow" }));
    fireEvent.click(screen.getByRole("button", { name: "Top right" }));
    fireEvent.click(screen.getByRole("button", { name: "Add the words" }));

    expect(onChange.mock.calls.map(([patch]) => patch)).toEqual([
      { text: "Harbour at dawn" },
      { font: "serif" },
      { sizePercent: 12 },
      { color: "#fdd835" },
      { outline: false },
      { shadow: true },
      { anchor: "top_right" },
    ]);
    await waitFor(() => expect(onAdd).toHaveBeenCalledWith(words));
    expect(drawCaption).toHaveBeenCalledWith(400, 200, caption);
  });

  it("says so when the words cannot be drawn", async () => {
    vi.mocked(drawCaption).mockResolvedValue(null);
    render(<StudioCaptionTool caption={{ ...DEFAULT_CAPTION, text: "Harbour" }} size={{ width: 40, height: 20 }}
      busy={false} onChange={vi.fn()} onAdd={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Add the words" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("The words could not be drawn. Try again.");
  });
});

describe("the words in the tool's state", () => {
  it("keeps a valid color and size, and clears the words for a new picture", () => {
    let state = studioToolReducer(initialToolState(), { type: "set-caption", patch: { text: "Hi", sizePercent: 90 } });
    state = studioToolReducer(state, { type: "set-caption", patch: { color: "yellow" } });
    expect(state.caption).toEqual({ ...DEFAULT_CAPTION, text: "Hi", sizePercent: 30 });

    // Added words are in the next picture; keeping them would draw them twice.
    state = studioToolReducer(state, { type: "image-changed", width: 4, height: 4 });
    expect(state.caption).toEqual({ ...DEFAULT_CAPTION, sizePercent: 30 });
  });

  it("keeps a drag within the picture and a turn within half a circle, and a new place undoes the drag", () => {
    let state = studioToolReducer(initialToolState(), { type: "set-caption", patch: { shift: { x: 3, y: -0.25 } } });
    expect(state.caption.shift).toEqual({ x: 1, y: -0.25 });
    state = studioToolReducer(state, { type: "set-caption", patch: { turn: 200 } });
    expect(state.caption.turn).toBe(180);
    state = studioToolReducer(state, { type: "set-caption", patch: { turn: -12.6 } });
    expect(state.caption.turn).toBe(-13);

    state = studioToolReducer(state, { type: "set-caption", patch: { anchor: "top" } });

    expect(state.caption).toEqual({ ...DEFAULT_CAPTION, anchor: "top", turn: -13 });
  });

  it("moves the words from where a drag took hold, and a cancel puts them back there", () => {
    let state = studioToolReducer(initialToolState(), { type: "image-changed", width: 400, height: 200 });
    state = studioToolReducer(state, { type: "set-caption", patch: { shift: { x: 0.1, y: 0 } } });
    state = studioToolReducer(state, { type: "hold-caption" });
    state = studioToolReducer(state, { type: "drag-caption", by: { x: 10, y: -5 } });
    state = studioToolReducer(state, { type: "drag-caption", by: { x: 20, y: -10 } });
    expect(state.caption.shift).toEqual({ x: 0.15, y: -0.05 });
    state = studioToolReducer(state, { type: "let-go-caption" });
    expect(state.captionHold).toBeNull();
    // With nothing held, a drag moves nothing.
    expect(studioToolReducer(state, { type: "drag-caption", by: { x: 40, y: 0 } }).caption.shift)
      .toEqual({ x: 0.15, y: -0.05 });

    state = studioToolReducer(state, { type: "hold-caption" });
    state = studioToolReducer(state, { type: "drag-caption", by: { x: -100, y: 0 } });
    state = studioToolReducer(state, { type: "cancel-caption" });

    expect(state.caption.shift).toEqual({ x: 0.15, y: -0.05 });
    expect(state.captionHold).toBeNull();
  });

  it("drags the words, and only them, when the view says where they are", () => {
    let state = studioToolReducer(initialToolState(), { type: "image-changed", width: 400, height: 200 });
    state = studioToolReducer(state, { type: "select-tool", kind: "caption" });
    const words = { hold: vi.fn(), drag: vi.fn(), letGo: vi.fn(), cancel: vi.fn() };

    // With nothing written yet there is nothing to drag, so a drag moves the view.
    expect(toolFor(state, null, vi.fn(), words)).toBeNull();
    state = studioToolReducer(state, { type: "set-caption", patch: { text: "Harbour" } });
    expect(toolFor(state)).toBeNull();
    const tool = toolFor(state, null, vi.fn(), words);
    snapshotBeforeGesture(state);
    tool?.down({ x: 10, y: 10 });
    tool?.up({ x: 50, y: 30 });

    expect(words.hold).toHaveBeenCalledTimes(1);
    expect(words.drag).toHaveBeenLastCalledWith({ x: 40, y: 20 });
    expect(words.letGo).toHaveBeenCalledTimes(1);
    // The selection was not touched, so Undo has nothing of the drag's to take back.
    expect(state.history.canUndo).toBe(false);
  });
});

describe("adding words in the studio", () => {
  it("uploads the drawn words and sends them for the picture on screen", async () => {
    const session = { id: "chat-studio", messages: [] } as never;
    vi.mocked(api.openStudioSession).mockResolvedValue(session);
    vi.mocked(api.studioSession).mockResolvedValue(session);
    vi.mocked(api.studioLocalEdit).mockResolvedValue(session);
    vi.mocked(api.upload).mockResolvedValue({ id: "sha256:words" } as never);
    vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
    vi.mocked(api.editTemplates).mockResolvedValue([]);
    vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
    vi.mocked(drawCaption).mockResolvedValue(drawn());
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
    vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: /^Add text to the picture/ }));
    expect(screen.queryByRole("button", { name: "Apply edit" })).toBeNull();
    fireEvent.change(screen.getByRole("textbox", { name: "Words" }), { target: { value: "Harbour" } });
    fireEvent.click(screen.getByRole("button", { name: "Add the words" }));

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    const uploaded = vi.mocked(api.upload).mock.calls[0][0] as File;
    expect(uploaded.name).toBe("studio-words.png");
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1",
      operation: "caption",
      caption: { overlay_artifact_id: "sha256:words" },
    });
  });

  /** The studio over a 400 by 200 picture with "Harbour" written; the canvas it shows. */
  async function writingWords(): Promise<Element> {
    const session = { id: "chat-studio", messages: [] } as never;
    vi.mocked(api.openStudioSession).mockResolvedValue(session);
    vi.mocked(api.studioSession).mockResolvedValue(session);
    vi.mocked(api.studioLocalEdit).mockResolvedValue(session);
    vi.mocked(api.upload).mockResolvedValue({ id: "sha256:words" } as never);
    vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
    vi.mocked(api.editTemplates).mockResolvedValue([]);
    vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
    vi.mocked(drawCaption).mockResolvedValue(drawn());
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
    vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
    const { container } = render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByRole("button", { name: /^Add text to the picture/ }));
    fireEvent.change(screen.getByRole("textbox", { name: "Words" }), { target: { value: "Harbour" } });
    return container.querySelector(".studio-canvas")!;
  }

  it("moves the words from the keyboard, and Escape puts them back where the hold began", async () => {
    const surface = await writingWords();

    // Each arrow moves the point twenty of the picture's pixels.
    fireEvent.keyDown(surface, { key: "Enter" });
    fireEvent.keyDown(surface, { key: "ArrowRight" });
    fireEvent.keyDown(surface, { key: "ArrowRight" });
    fireEvent.keyDown(surface, { key: "ArrowUp" });
    fireEvent.keyDown(surface, { key: "Enter" });
    await waitFor(() => expect(drawCaption).toHaveBeenLastCalledWith(400, 200,
      expect.objectContaining({ shift: { x: 0.1, y: -0.1 } })));

    fireEvent.keyDown(surface, { key: "Enter" });
    fireEvent.keyDown(surface, { key: "ArrowDown" });
    await waitFor(() => expect(drawCaption).toHaveBeenLastCalledWith(400, 200,
      expect.objectContaining({ shift: { x: 0.1, y: 0 } })));
    fireEvent.keyDown(surface, { key: "Escape" });

    await waitFor(() => expect(drawCaption).toHaveBeenLastCalledWith(400, 200,
      expect.objectContaining({ shift: { x: 0.1, y: -0.1 } })));
  });

  it("moves the words where they are dragged on the picture, shows them there, and adds them there", async () => {
    const surface = await writingWords();
    expect(surface).toHaveClass("moving");
    expect(surface.getAttribute("aria-label")).toContain("Enter takes hold and lets go");

    // jsdom shows the 400 by 200 picture at its own size from the corner, so
    // screen and picture pixels are the same here.
    pointer(surface, "pointerdown", 100, 150);
    pointer(surface, "pointermove", 120, 140);
    await waitFor(() => expect(drawCaption).toHaveBeenLastCalledWith(400, 200,
      expect.objectContaining({ text: "Harbour", shift: { x: 0.05, y: -0.05 } })));
    pointer(surface, "pointermove", 140, 130);
    pointer(surface, "pointerup", 140, 130);
    fireEvent.change(screen.getByRole("slider", { name: "Turn" }), { target: { value: "-8" } });
    fireEvent.click(screen.getByRole("button", { name: "Add the words" }));

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(drawCaption).toHaveBeenLastCalledWith(400, 200,
      expect.objectContaining({ text: "Harbour", shift: { x: 0.1, y: -0.1 }, turn: -8 }));
  });
});
