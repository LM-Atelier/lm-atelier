/** Replacing words sends a softened box that is placed back after a whole-picture edit. */

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

function drawBox() {
  const canvas = screen.getByRole("application");
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.keyDown(canvas, { key: "ArrowRight" });
  fireEvent.keyDown(canvas, { key: "ArrowRight" });
  fireEvent.keyDown(canvas, { key: "ArrowDown" });
  fireEvent.keyDown(canvas, { key: "ArrowDown" });
  fireEvent.keyDown(canvas, { key: "Enter" });
}

it("replaces the boxed words through a softened copy of the box", async () => {
  fireEvent.click(screen.getByRole("button", { name: "Replace words in the picture" }));
  const replace = screen.getByRole("button", { name: "Replace words" });
  expect(replace).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(replace);
  expect(apply).not.toHaveBeenCalled();
  expect(screen.queryByRole("slider", { name: "Brush size" })).toBeNull();

  fireEvent.change(screen.getByRole("textbox", { name: /Replace with/ }), { target: { value: "Open late" } });
  // New words without a box would change the whole picture.
  expect(replace).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(replace);
  expect(apply).not.toHaveBeenCalled();
  drawBox();
  fireEvent.change(screen.getByRole("textbox", { name: /Words there now/ }), { target: { value: "Open daily" } });
  expect(replace).toHaveAttribute("aria-disabled", "false");

  fireEvent.click(replace);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  const [words, artifactId, mask, settings, workflowRevisionId] = apply.mock.calls[0];
  expect(words).toBe(
    'Replace the text "Open daily" with "Open late". Keep the same font, color, size and position, and leave everything else unchanged.',
  );
  expect(artifactId).toBe("art-1");
  expect(mask).toEqual({ blob: expect.any(Blob), featherPx: 4, invert: false, apply: "blend" });
  expect(settings).toBeUndefined();
  expect(workflowRevisionId).toBeUndefined();
  // Softened: the box's edge fades rather than stepping from nothing to all.
  expect(encoded[0].some((value) => value > 0 && value < 255)).toBe(true);

  // The box on the canvas stays as drawn, so a second try softens it once, not twice.
  fireEvent.click(replace);
  await waitFor(() => expect(apply).toHaveBeenCalledTimes(2));
  expect(encoded[1]).toEqual(encoded[0]);
});

it("keeps other selection tools handing their selection to the workflow", async () => {
  fireEvent.click(screen.getByRole("button", { name: "Select a rectangle" }));
  drawBox();
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "make it blue" } });

  fireEvent.click(screen.getByRole("button", { name: "Apply to selection" }));

  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  expect(apply.mock.calls[0][2]).toEqual({ blob: expect.any(Blob), featherPx: 4, invert: false });
  expect(encoded[0].every((value) => value === 0 || value === 255)).toBe(true);
});
