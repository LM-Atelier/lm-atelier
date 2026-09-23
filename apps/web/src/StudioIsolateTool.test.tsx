/** Isolate sends the picture to the workflow the report names, with nothing to select or describe. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import type { StudioToolCapability } from "./types";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: { favoriteArtifact: vi.fn(), artifact: vi.fn(), editTemplates: vi.fn(), studioCapabilities: vi.fn() },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
// What the studio's own workflow chooser reports. By default its workflow
// cannot run here, so only a workflow the report names can.
const chooser = vi.hoisted(() => ({ reason: "Choose an editing workflow." as string | null }));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(chooser.reason), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

let apply: ReturnType<typeof vi.fn>;

function openWith(isolate: Partial<StudioToolCapability>) {
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({
    tools: [
      {
        kind: "isolate",
        workflow_class: "matting",
        available: true,
        reason: null,
        workflow_revision_id: "wfrev_cutout",
        adapter_asset_id: null,
        ...isolate,
      },
    ],
  });
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
  fireEvent.click(screen.getByRole("button", { name: "Cut the subject out" }));
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  chooser.reason = "Choose an editing workflow.";
});

it("cuts the subject out with the workflow the report names and nothing else", async () => {
  openWith({});
  // Nothing to point at and nothing to describe.
  expect(screen.queryByRole("button", { name: "Invert" })).toBeNull();
  expect(screen.queryByRole("textbox")).toBeNull();

  const cut = await screen.findByRole("button", { name: "Cut out" });
  await waitFor(() => expect(cut).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(cut);

  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  const [words, artifactId, mask, settings, workflowRevisionId, , secondPicture] = apply.mock.calls[0];
  expect(words).toBe("Cut the subject out onto a transparent background.");
  expect(artifactId).toBe("art-1");
  expect(mask).toBeUndefined();
  expect(settings).toBeUndefined();
  // The named workflow, never the studio's chosen one, which would answer
  // with an ordinary edit instead of a cutout.
  expect(workflowRevisionId).toBe("wfrev_cutout");
  expect(secondPicture).toBeUndefined();
});

it("waits for the report to name a workflow even when the studio's own one could run", async () => {
  // The studio's chosen workflow is ready, and it is still not the one to use:
  // run for Isolate it would answer with an ordinary edit.
  chooser.reason = null;
  openWith({ workflow_revision_id: null });

  const cut = await screen.findByRole("button", { name: "Cut out" });
  await waitFor(() => expect(api.studioCapabilities).toHaveBeenCalled());
  expect(cut).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(cut);
  expect(apply).not.toHaveBeenCalled();
});

it("says what to install when nothing here can cut a subject out", async () => {
  openWith({
    available: false,
    reason: "Install a background removal workflow to cut a subject out of a picture.",
    workflow_revision_id: null,
  });

  expect(await screen.findByRole("status")).toHaveTextContent("Install a background removal workflow");
  const cut = screen.getByRole("button", { name: "Cut out" });
  expect(cut).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(cut);
  expect(apply).not.toHaveBeenCalled();
});
