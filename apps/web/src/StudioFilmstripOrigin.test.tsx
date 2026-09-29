/** The history strip names the picture a result was made from when it is not the one before it. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
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
