/** Cropping: a box drawn on the canvas, kept with one deliberate press. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { useStudioImage } from "./useStudioImage";

vi.mock("./api", () => ({
  api: {
    openStudioSession: vi.fn(),
    studioSession: vi.fn(),
    studioLocalEdit: vi.fn(),
    sendTurn: vi.fn(),
    upload: vi.fn(),
    favoriteArtifact: vi.fn(),
    artifact: vi.fn(),
    editTemplates: vi.fn(),
    studioCapabilities: vi.fn(),
  },
}));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

function session() {
  return { id: "chat-studio", messages: [] } as never;
}

afterEach(() => {
  cleanup();
  localStorage.clear();
  vi.restoreAllMocks();
});

function openStudio() {
  vi.mocked(api.openStudioSession).mockResolvedValue(session());
  vi.mocked(api.studioSession).mockResolvedValue(session());
  vi.mocked(api.studioLocalEdit).mockResolvedValue(session());
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
  vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <StudioView sourceArtifactId="art-1" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
}

describe("the crop tool", () => {
  it("keeps exactly the box drawn on the canvas, with one press", async () => {
    openStudio();
    fireEvent.click(await screen.findByRole("button", { name: /^Crop the picture/ }));
    const crop = screen.getByRole("button", { name: "Crop to the box" });
    // Nothing is cut before there is a box to keep.
    expect(crop).toHaveAttribute("aria-disabled", "true");
    expect(screen.queryByRole("button", { name: "Apply edit" })).toBeNull();

    // From the middle of the 400 by 200 picture, two steps right and two down.
    const canvas = screen.getByRole("application");
    fireEvent.keyDown(canvas, { key: "Enter" });
    for (const key of ["ArrowRight", "ArrowRight", "ArrowDown", "ArrowDown"]) fireEvent.keyDown(canvas, { key });
    fireEvent.keyDown(canvas, { key: "Enter" });

    expect(await screen.findByText("Keeps 40 × 40 of 400 × 200.")).toBeInTheDocument();
    await waitFor(() => expect(crop).toHaveAttribute("aria-disabled", "false"));
    fireEvent.click(crop);

    await waitFor(() => expect(api.studioLocalEdit).toHaveBeenCalledTimes(1));
    expect(api.studioLocalEdit).toHaveBeenCalledWith("chat-studio", {
      source_artifact_id: "art-1",
      operation: "crop",
      crop: { left: 200, top: 100, width: 40, height: 40 },
    });
  });
});
