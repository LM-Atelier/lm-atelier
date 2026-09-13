/** The paint bucket and the magic wand, as the studio offers them. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { PointerTool } from "./studioTools";
import { readSourcePixels } from "./studioSourcePixels";
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
vi.mock("./studioSourcePixels", () => ({ readSourcePixels: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({ StudioWorkflowSelector: () => <div>Workflow chooser</div> }));

type CanvasProps = { tool: PointerTool | null; onGestureStart?: () => void; onStrokeEnd?: () => void };
/** What the studio last handed the canvas. The canvas itself needs a real 2D context. */
const canvas: { props: CanvasProps | null } = { props: null };
vi.mock("./StudioCanvas", () => ({
  StudioCanvas: (props: CanvasProps) => {
    canvas.props = props;
    return <div data-testid="studio-canvas" />;
  },
}));

/** A 4 by 2 picture: the left half one orange, the right half blue. */
const PICTURE = { width: 4, height: 2, close: vi.fn() } as unknown as ImageBitmap;
const PIXELS = new Uint8ClampedArray(
  [0, 1, 2, 3, 4, 5, 6, 7].flatMap((index) => (index % 4 < 2 ? [200, 20, 20, 255] : [20, 20, 200, 255])),
);

function renderStudio() {
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
}

/** Click the canvas at a picture point the way the canvas reports one.
 *
 * Press and release are separate events, with a render between them, which is
 * when the studio takes its undo snapshot.
 */
function click(x: number, y: number) {
  const { tool, onGestureStart, onStrokeEnd } = canvas.props!;
  act(() => {
    onGestureStart?.();
    tool!.down({ x, y });
  });
  act(() => {
    if (tool!.up({ x, y })) onStrokeEnd?.();
  });
}

beforeEach(() => {
  canvas.props = null;
  vi.mocked(useStudioImage).mockReturnValue({ bitmap: PICTURE, error: null, reload: vi.fn() } as ReturnType<typeof useStudioImage>);
  vi.mocked(readSourcePixels).mockReturnValue(PIXELS);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("selects the area of similar color with the wand, and takes it away again", () => {
  renderStudio();
  fireEvent.click(screen.getByRole("button", { name: "Select similar colors" }));
  expect(screen.queryByLabelText("Brush size")).toBeNull();
  expect(screen.getByLabelText("Color tolerance")).toHaveValue("32");

  click(0, 0);
  expect(screen.getByText("50.0% of the image selected")).toBeInTheDocument();
  expect(readSourcePixels).toHaveBeenCalledWith(PICTURE);

  fireEvent.click(screen.getByRole("button", { name: "Take from selection" }));
  expect(screen.getByRole("button", { name: "Take from selection" })).toHaveAttribute("aria-pressed", "true");
  click(1, 1);
  expect(screen.getByText("Nothing selected yet - paint over what you want to change.")).toBeInTheDocument();
});

it("fills an area with the paint bucket and undoes it as one step", () => {
  renderStudio();
  fireEvent.click(screen.getByRole("button", { name: "Fill an area of the selection" }));
  expect(screen.getByRole("button", { name: "Add to selection" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.queryByLabelText("Color tolerance")).toBeNull();

  click(2, 1);
  expect(screen.getByText("100.0% of the image selected")).toBeInTheDocument();
  // The bucket needs no picture colors.
  expect(readSourcePixels).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole("button", { name: "Undo the selection change" }));
  expect(screen.getByText("Nothing selected yet - paint over what you want to change.")).toBeInTheDocument();
});

it("says so when the picture's colors cannot be read, and gives the wand nothing to click with", () => {
  vi.mocked(readSourcePixels).mockReturnValue(null);
  renderStudio();
  fireEvent.click(screen.getByRole("button", { name: "Select similar colors" }));

  expect(screen.getByRole("alert")).toHaveTextContent("This picture's colors cannot be read here");
  expect(canvas.props!.tool).toBeNull();
});

it("uses the chosen tolerance for the next click", () => {
  // Two orange columns, a darker orange 60 away in one channel, then blue.
  const shades = [[200, 20, 20, 255], [200, 20, 20, 255], [140, 20, 20, 255], [20, 20, 200, 255]];
  vi.mocked(readSourcePixels).mockReturnValue(
    new Uint8ClampedArray([0, 1, 2, 3, 0, 1, 2, 3].flatMap((column) => shades[column])),
  );
  renderStudio();
  fireEvent.click(screen.getByRole("button", { name: "Select similar colors" }));

  click(0, 0);
  expect(screen.getByText("50.0% of the image selected")).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "Undo the selection change" }));
  fireEvent.change(screen.getByLabelText("Color tolerance"), { target: { value: "64" } });
  click(0, 0);
  expect(screen.getByText("75.0% of the image selected")).toBeInTheDocument();
});
