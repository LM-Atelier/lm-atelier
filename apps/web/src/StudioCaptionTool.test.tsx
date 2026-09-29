/** Adding words: drawn in the browser, previewed on the canvas, and added as drawn. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioCaptionTool } from "./StudioCaptionTool";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { captionFontPx, captionLayout, DEFAULT_CAPTION, drawCaption } from "./studioCaption";
import { initialToolState, studioToolReducer } from "./studioToolState";
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
    fireEvent.click(screen.getByRole("button", { name: "Top right" }));
    fireEvent.click(screen.getByRole("button", { name: "Add the words" }));

    expect(onChange.mock.calls.map(([patch]) => patch)).toEqual([
      { text: "Harbour at dawn" },
      { font: "serif" },
      { sizePercent: 12 },
      { color: "#fdd835" },
      { outline: false },
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
});
