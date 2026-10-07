/** The history strip names the picture a result was made from when it is not the one before it, and marks the way back from the chosen one. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession, type StudioStep } from "./useStudioSession";

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
vi.mock("./StudioCanvas", () => ({ StudioCanvas: () => <div data-testid="studio-canvas" /> }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function step(artifactId: string, beforeArtifactId: string | null, instruction = ""): StudioStep {
  return {
    messageId: `message-${artifactId}`,
    artifactId,
    instruction,
    beforeArtifactId,
    isSource: beforeArtifactId === null,
    generationIdentity: null,
  };
}

it("names the picture a result was made from when it is not the step before it", () => {
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  } as ReturnType<typeof useStudioImage>);
  vi.mocked(useStudioSession).mockReturnValue({
    // Two changes in a row, then one more made from the first change, then
    // one made from the original.
    steps: [
      step("art-1", null),
      step("art-2", "art-1", "warmer"),
      step("art-3", "art-2", "sharper"),
      step("art-4", "art-2", "cooler"),
      step("art-5", "art-1", "brighter"),
    ],
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

  const strip = within(screen.getByRole("group", { name: "Edit history" }));
  expect(strip.getByRole("button", { name: "Result of step 2 sharper" })).toBeInTheDocument();
  expect(strip.getByRole("button", { name: "Result of step 3 cooler From step 1" })).toBeInTheDocument();
  expect(strip.getByRole("button", { name: "Result of step 4 brighter From the original" })).toBeInTheDocument();
  expect(strip.getByRole("button", { name: "Result of step 1 warmer" })).toBeInTheDocument();
});

it("marks the results the chosen one was made from, and not the other branch", () => {
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  } as ReturnType<typeof useStudioImage>);
  vi.mocked(useStudioSession).mockReturnValue({
    // Two branches from the original, each with one more change on it.
    steps: [
      step("art-1", null),
      step("art-2", "art-1", "warmer"),
      step("art-3", "art-1", "sharper"),
      step("art-4", "art-2", "cooler"),
      step("art-5", "art-3", "brighter"),
    ],
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

  const strip = within(screen.getByRole("group", { name: "Edit history" }));
  const note = "Part of how the chosen result was made";
  // The newest is chosen at first, and it was made on the second branch.
  expect(strip.getByRole("button", { name: "Result of step 4 brighter From step 2" })).toHaveClass("selected");
  expect(strip.getByRole("button", { name: `Result of step 2 sharper From the original ${note}` })).toHaveClass("made-from");
  expect(strip.getByRole("button", { name: `The original image Original ${note}` })).toHaveClass("made-from");
  expect(strip.getByRole("button", { name: "Result of step 1 warmer" })).not.toHaveClass("made-from");
  expect(strip.getByRole("button", { name: "Result of step 3 cooler From step 1" })).not.toHaveClass("made-from");

  // Choosing a result on the first branch moves the marks to its own way back.
  fireEvent.click(strip.getByRole("button", { name: "Result of step 3 cooler From step 1" }));
  expect(strip.getByRole("button", { name: "Result of step 3 cooler From step 1" })).toHaveClass("selected");
  expect(strip.getByRole("button", { name: `Result of step 1 warmer ${note}` })).toHaveClass("made-from");
  expect(strip.getByRole("button", { name: `The original image Original ${note}` })).toHaveClass("made-from");
  expect(strip.getByRole("button", { name: "Result of step 2 sharper From the original" })).not.toHaveClass("made-from");
  expect(strip.getByRole("button", { name: "Result of step 4 brighter From step 2" })).not.toHaveClass("made-from");
});
