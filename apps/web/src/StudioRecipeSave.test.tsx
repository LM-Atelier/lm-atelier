/** The Studio offers to save the result on screen as a recipe when a model made it. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import type { Message } from "./types";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: {
    favoriteArtifact: vi.fn(),
    artifact: vi.fn().mockResolvedValue({ id: "art-2", favorite: false }),
    editTemplates: vi.fn().mockResolvedValue([]),
    studioCapabilities: vi.fn().mockResolvedValue({ tools: [] }),
  },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({ StudioWorkflowSelector: () => <div>Workflow chooser</div> }));
vi.mock("./StudioCanvas", () => ({ StudioCanvas: () => <div data-testid="studio-canvas" /> }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

const stamp = "2026-09-29T00:00:00Z";

function answer(metadata: Record<string, unknown>): Message {
  return {
    id: "answer-2",
    chat_id: "chat-studio",
    parent_id: null,
    role: "assistant",
    status: "complete",
    parts: [
      { id: "image", position: 0, type: "image", text: null, artifact_id: "art-2", metadata_json: {} },
      { id: "meta", position: 1, type: "generation_metadata", text: null, artifact_id: null, metadata_json: metadata },
    ],
    created_at: stamp,
    updated_at: stamp,
  };
}

function openOn(metadata: Record<string, unknown>, instruction: string) {
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [
      { messageId: "source", artifactId: "art-1", instruction: "", beforeArtifactId: null, isSource: true, generationIdentity: null },
      { messageId: "answer-2", artifactId: "art-2", instruction, beforeArtifactId: "art-1", isSource: false, generationIdentity: null },
    ],
    session: { messages: [answer(metadata)] },
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

it("offers to save the edit on screen as a recipe when a model made it", async () => {
  openOn({ run_id: "run-7", provenance: {} }, "make it a watercolor");

  expect(await screen.findByRole("textbox", { name: "Save this edit as a recipe" })).toBeInTheDocument();
});

it("offers no recipe for an exact edit", async () => {
  openOn({ provenance: { local_edit: { operation: "rotate_clockwise" } } }, "Rotate right");

  await screen.findByText("Workflow chooser");
  expect(screen.queryByRole("textbox", { name: "Save this edit as a recipe" })).toBeNull();
});
