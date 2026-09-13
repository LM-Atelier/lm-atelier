/** Undo takes back a whole selection gesture, drawn through the real canvas. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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
vi.mock("./StudioWorkflowSelector", () => ({ StudioWorkflowSelector: () => <div>Workflow chooser</div> }));

const NOTHING = "Nothing selected yet - paint over what you want to change.";

function selection(): string {
  return document.querySelector(".studio-selection-controls small")?.textContent ?? "";
}

beforeEach(() => {
  // The canvas has no 2D context here, so it paints nothing; the selection and
  // the gestures that change it still run exactly as in a browser.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  } as ReturnType<typeof useStudioImage>);
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [{ artifactId: "art-1", instruction: null, generationIdentity: null }],
    previewArtifactId: null,
    sessionId: "chat-studio",
    busy: false,
    error: null,
    apply: vi.fn(),
  } as unknown as ReturnType<typeof useStudioSession>);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

/** Start a gesture, move the caret twice, and finish it, from the keyboard. */
function stroke() {
  const canvas = screen.getByRole("application");
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.keyDown(canvas, { key: "ArrowRight" });
  fireEvent.keyDown(canvas, { key: "ArrowRight" });
  fireEvent.keyDown(canvas, { key: "Enter" });
}

it("undoes a whole brush stroke, where it started too, and redoes it", () => {
  fireEvent.click(screen.getByRole("button", { name: "Brush a selection" }));

  stroke();
  const painted = selection();
  expect(painted).not.toBe(NOTHING);

  fireEvent.click(screen.getByRole("button", { name: "Undo the selection change" }));
  expect(selection()).toBe(NOTHING);
  fireEvent.click(screen.getByRole("button", { name: "Redo the selection change" }));
  expect(selection()).toBe(painted);
});

it("undoes a whole eraser stroke, where it started too", () => {
  fireEvent.click(screen.getByRole("button", { name: "Brush a selection" }));
  stroke();
  const painted = selection();
  fireEvent.click(screen.getByRole("button", { name: "Erase from the selection" }));

  stroke();
  expect(selection()).not.toBe(painted);

  fireEvent.click(screen.getByRole("button", { name: "Undo the selection change" }));
  expect(selection()).toBe(painted);
});

it("undoes a single dab", () => {
  fireEvent.click(screen.getByRole("button", { name: "Brush a selection" }));
  const canvas = screen.getByRole("application");
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.keyDown(canvas, { key: "Enter" });
  expect(selection()).not.toBe(NOTHING);

  fireEvent.click(screen.getByRole("button", { name: "Undo the selection change" }));

  expect(selection()).toBe(NOTHING);
});
