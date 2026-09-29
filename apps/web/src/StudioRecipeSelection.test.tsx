/** A recipe made on a selection needs one again: the Studio switches to selecting, waits for it, and sends it the same way. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import type { EditTemplate } from "./types";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: {
    favoriteArtifact: vi.fn(),
    artifact: vi.fn().mockResolvedValue({ id: "art-1", favorite: false }),
    editTemplates: vi.fn(),
    studioCapabilities: vi.fn().mockResolvedValue({ tools: [] }),
  },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({ StudioWorkflowSelector: () => <div>Workflow chooser</div> }));
// The selection is encoded with a canvas jsdom does not have; what matters here is what goes with it.
vi.mock("./studioMasks", async (original) => ({
  ...(await original<typeof import("./studioMasks")>()),
  encodeMaskPng: vi.fn(async () => new Blob(["mask"], { type: "image/png" })),
}));

function recipe(overrides: Partial<EditTemplate>): EditTemplate {
  return {
    id: "tpl-sky",
    name: "Warm sky",
    description: "",
    instruction: "make the sky warmer",
    operation: "image_to_image",
    settings_json: {},
    workflow_revision_id: "rev-1",
    model_profile_id: null,
    mask_mode: "selection",
    trigger_words_json: [],
    content_rating: "general",
    builtin: false,
    enabled: true,
    ...overrides,
  };
}

const apply = vi.fn();

beforeEach(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.mocked(useStudioImage).mockReturnValue({
    bitmap: { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap,
    error: null,
    reload: vi.fn(),
  } as ReturnType<typeof useStudioImage>);
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [{ artifactId: "art-1", instruction: "", generationIdentity: null }],
    previewArtifactId: null,
    sessionId: "chat-studio",
    busy: false,
    error: null,
    apply,
  } as unknown as ReturnType<typeof useStudioSession>);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  apply.mockReset();
});

function open(chosen: EditTemplate) {
  vi.mocked(api.editTemplates).mockResolvedValue([chosen]);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
}

/** Brush a short stroke from the keyboard. */
function stroke() {
  const canvas = screen.getByRole("application");
  fireEvent.keyDown(canvas, { key: "Enter" });
  fireEvent.keyDown(canvas, { key: "ArrowRight" });
  fireEvent.keyDown(canvas, { key: "Enter" });
}

it("switches to selecting for a recipe made on a selection, and holds Apply until there is one", async () => {
  open(recipe({}));
  fireEvent.click(await screen.findByRole("button", { name: "Warm sky" }));

  expect(screen.getByRole("button", { name: "Brush a selection" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByText("Warm sky changes only a selected part. Select it first.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Apply edit" })).toHaveAttribute("aria-disabled", "true");

  stroke();
  expect(screen.queryByText("Warm sky changes only a selected part. Select it first.")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Apply to selection" }));

  await waitFor(() => expect(apply).toHaveBeenCalled());
  expect(apply.mock.calls[0][0]).toBe("make the sky warmer");
  expect(apply.mock.calls[0][2]).toEqual(expect.objectContaining({ invert: false }));
});

it("sends the selection inverted for a recipe made on everything outside one", async () => {
  open(recipe({ mask_mode: "inverse", name: "Soft surroundings" }));
  fireEvent.click(await screen.findByRole("button", { name: "Soft surroundings" }));
  expect(
    screen.getByText("Soft surroundings changes everything outside a selection. Select what to keep first."),
  ).toBeInTheDocument();

  stroke();
  fireEvent.click(screen.getByRole("button", { name: "Apply to selection" }));

  await waitFor(() => expect(apply).toHaveBeenCalled());
  expect(apply.mock.calls[0][2]).toEqual(expect.objectContaining({ invert: true }));
});

it("leaves the tool alone for a recipe made on the whole picture", async () => {
  open(recipe({ mask_mode: "none", name: "Watercolor" }));
  fireEvent.click(await screen.findByRole("button", { name: "Watercolor" }));

  expect(screen.getByRole("button", { name: "Instruct the whole image" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.queryByText(/Select it first|Select what to keep first/)).toBeNull();
});
