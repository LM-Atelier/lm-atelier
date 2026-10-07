/** Removing something sends the marked part, softened, to be kept from a whole-picture edit. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { encodeMaskPng, type MaskRaster } from "./studioMasks";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: {
    favoriteArtifact: vi.fn(),
    artifact: vi.fn().mockResolvedValue({ id: "art-1", favorite: false }),
    editTemplates: vi.fn().mockResolvedValue([]),
    studioCapabilities: vi.fn().mockResolvedValue({ tools: [] }),
  },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./studioMasks", async (original) => ({
  ...(await original<typeof import("./studioMasks")>()),
  encodeMaskPng: vi.fn(),
}));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

let apply: ReturnType<typeof vi.fn>;
const encoded: Uint8ClampedArray[] = [];

beforeEach(() => {
  // No 2D context in the test DOM; the selection raster is still drawn and kept.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.mocked(encodeMaskPng).mockImplementation(async (mask: MaskRaster) => {
    encoded.push(new Uint8ClampedArray(mask.data));
    return new Blob(["selection"], { type: "image/png" });
  });
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  } as ReturnType<typeof useStudioImage>);
  apply = vi.fn();
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [{ artifactId: "art-1", instruction: null, generationIdentity: null }],
    previewArtifactId: null,
    sessionId: "chat-studio",
    busy: false,
    error: null,
    apply,
  } as unknown as ReturnType<typeof useStudioSession>);
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
});

afterEach(() => {
  cleanup();
  encoded.length = 0;
  vi.restoreAllMocks();
});

/** One dab of the brush in the middle of the picture, from the keyboard. */
function brushOnce() {
  const canvas = screen.getByRole("application");
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.keyDown(canvas, { key: "Enter" });
}

it("removes what is named from the part brushed over, through a softened copy of the marking", async () => {
  fireEvent.click(screen.getByRole("button", { name: "Remove something from the picture" }));
  const remove = screen.getByRole("button", { name: "Remove" });
  // Marked with the brush, so its size is offered.
  expect(screen.getByRole("slider", { name: "Brush size" })).toBeInTheDocument();
  expect(remove).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(remove);
  expect(apply).not.toHaveBeenCalled();

  fireEvent.change(screen.getByRole("textbox", { name: /What to remove/ }), { target: { value: "the lamp post" } });
  // Named but not marked, it would redraw the whole picture without it.
  expect(remove).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(remove);
  expect(apply).not.toHaveBeenCalled();
  brushOnce();
  expect(remove).toHaveAttribute("aria-disabled", "false");

  fireEvent.click(remove);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  const [words, artifactId, mask, settings, workflowRevisionId] = apply.mock.calls[0];
  expect(words).toBe(
    "Remove the lamp post. Fill the space it leaves to match what surrounds it, and leave everything else unchanged.",
  );
  expect(artifactId).toBe("art-1");
  // Kept from the edit rather than handed to the workflow, so any edit workflow can run it.
  expect(mask).toEqual({ blob: expect.any(Blob), featherPx: 4, invert: false, apply: "blend" });
  expect(settings).toBeUndefined();
  expect(workflowRevisionId).toBeUndefined();
  // Softened: the marking's edge fades rather than stepping from nothing to all.
  expect(encoded[0].some((value) => value > 0 && value < 255)).toBe(true);
});

it("grows the marked part before softening it, so the edge of what is removed is redrawn too", async () => {
  fireEvent.click(screen.getByRole("button", { name: "Remove something from the picture" }));
  fireEvent.change(screen.getByRole("textbox", { name: /What to remove/ }), { target: { value: "the blue square" } });
  // One dab at the picture's middle, (200, 100), with the brush's 24 px radius:
  // on that row the marking ends at x = 223.
  brushOnce();
  fireEvent.click(screen.getByRole("button", { name: "Remove" }));

  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  // Grown by the reach a replaced subject is given (8 px on a picture 200 high)
  // before the 4 px softening, so 5 px past the dab is still mostly removed...
  expect(encoded[0][100 * 400 + 228]).toBeGreaterThanOrEqual(128);
  // ...and the growth stops there.
  expect(encoded[0][100 * 400 + 240]).toBe(0);
  // What is sent says the softening the person chose; the growth is not theirs.
  expect(apply.mock.calls[0][2]).toEqual({ blob: expect.any(Blob), featherPx: 4, invert: false, apply: "blend" });
});

it("asks what to remove before it will, however much is marked", () => {
  fireEvent.click(screen.getByRole("button", { name: "Remove something from the picture" }));
  brushOnce();
  const remove = screen.getByRole("button", { name: "Remove" });

  expect(remove).toHaveAttribute("aria-disabled", "true");
  fireEvent.change(screen.getByRole("textbox", { name: /What to remove/ }), { target: { value: "   " } });
  fireEvent.click(remove);

  expect(apply).not.toHaveBeenCalled();
});
