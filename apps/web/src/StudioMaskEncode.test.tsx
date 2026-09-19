/** A selection that cannot be prepared is never sent as an edit of the whole picture. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
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
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

let apply: ReturnType<typeof vi.fn>;

beforeEach(() => {
  // No 2D context: the canvas cannot encode the selection, as in a browser that
  // blocks canvas drawing. The selection itself is still drawn and kept.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
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
  vi.restoreAllMocks();
});

it("refuses and says why instead of editing the whole picture", async () => {
  fireEvent.click(screen.getByRole("button", { name: "Brush a selection" }));
  const canvas = screen.getByRole("application");
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.keyDown(canvas, { key: "ArrowRight" });
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "make it blue" } });

  fireEvent.click(screen.getByRole("button", { name: "Apply to selection" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("The selection could not be prepared");
  expect(apply).not.toHaveBeenCalled();
  // The words stay for another try.
  expect(screen.getByRole("textbox")).toHaveValue("make it blue");
});

it("refuses the same way when drawing the selection throws", async () => {
  fireEvent.click(screen.getByRole("button", { name: "Brush a selection" }));
  const canvas = screen.getByRole("application");
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.keyDown(canvas, { key: "ArrowRight" });
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "make it blue" } });
  // Only the encoder's canvas fails: it throws while drawing, so encoding rejects
  // rather than coming back empty.
  vi.mocked(HTMLCanvasElement.prototype.getContext).mockReturnValue({
    putImageData: () => {
      throw new Error("canvas drawing failed");
    },
  } as unknown as CanvasRenderingContext2D);

  fireEvent.click(screen.getByRole("button", { name: "Apply to selection" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("The selection could not be prepared");
  expect(apply).not.toHaveBeenCalled();
});
