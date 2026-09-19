/** Relight sends the picture, a light map drawn at its size, and the adapter at full strength. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { lightMapGradient, renderLightMap } from "./studioLightMap";
import { initialToolState, studioToolReducer } from "./studioToolState";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: { favoriteArtifact: vi.fn(), artifact: vi.fn(), editTemplates: vi.fn(), studioCapabilities: vi.fn() },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./studioLightMap", async (original) => ({
  ...(await original<typeof import("./studioLightMap")>()),
  renderLightMap: vi.fn(),
}));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    // The chat's own workflow cannot run here; the report names one that can.
    useEffect(() => onAvailabilityChange("Choose an editing workflow."), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

let apply: ReturnType<typeof vi.fn>;

beforeEach(() => {
  // Set here rather than in the module mock: restoring mocks after each test
  // clears what a factory set, and a second test would see no tools at all.
  vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({
    tools: [
      {
        kind: "relight",
        workflow_class: "relight",
        available: true,
        reason: null,
        workflow_revision_id: "wfrev_light",
        adapter_asset_id: "asset_light",
      },
    ],
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  vi.mocked(renderLightMap).mockResolvedValue(new Blob(["light"], { type: "image/png" }));
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

it("relights from the chosen side with the adapter at full strength", async () => {
  fireEvent.click(screen.getByRole("button", { name: "Relight from a direction" }));
  // Nothing to select: the light comes from a side of the whole picture.
  expect(screen.queryByRole("button", { name: "Invert" })).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Top" }));
  fireEvent.change(screen.getByRole("slider"), { target: { value: "75" } });
  fireEvent.click(screen.getByRole("button", { name: "Warm" }));

  const relight = await screen.findByRole("button", { name: "Relight" });
  await waitFor(() => expect(relight).toBeEnabled());
  fireEvent.click(relight);

  await waitFor(() => expect(apply).toHaveBeenCalledTimes(1));
  const [words, artifactId, mask, settings, workflowRevisionId, , secondPicture] = apply.mock.calls[0];
  expect(words).toBe("Relight Figure 1 using the luminance map from Figure 2 (light source from the top).");
  expect(artifactId).toBe("art-1");
  expect(mask).toBeUndefined();
  expect(settings).toEqual({
    relight: { direction: "top", intensity: 0.75, kelvin: 4500 },
    loras: [{ asset_id: "asset_light", model_strength: 1 }],
  });
  expect(workflowRevisionId).toBe("wfrev_light");
  expect(secondPicture).toBeInstanceOf(Blob);
  expect(renderLightMap).toHaveBeenCalledWith(400, 200, "top");
});

it("refuses and says why when the light map cannot be drawn", async () => {
  vi.mocked(renderLightMap).mockResolvedValue(null);
  fireEvent.click(screen.getByRole("button", { name: "Relight from a direction" }));
  const relight = await screen.findByRole("button", { name: "Relight" });
  await waitFor(() => expect(relight).toBeEnabled());

  fireEvent.click(relight);

  expect(await screen.findByRole("alert")).toHaveTextContent("The light map could not be drawn");
  expect(apply).not.toHaveBeenCalled();
});

it("refuses the same way when drawing the light map throws", async () => {
  vi.mocked(renderLightMap).mockRejectedValue(new Error("canvas drawing failed"));
  fireEvent.click(screen.getByRole("button", { name: "Relight from a direction" }));
  const relight = await screen.findByRole("button", { name: "Relight" });
  await waitFor(() => expect(relight).toBeEnabled());

  fireEvent.click(relight);

  expect(await screen.findByRole("alert")).toHaveTextContent("The light map could not be drawn");
  expect(apply).not.toHaveBeenCalled();
});

it("draws the map bright on the side the light comes from", () => {
  expect(lightMapGradient(400, 200, "left")).toEqual([0, 0, 400, 0]);
  expect(lightMapGradient(400, 200, "right")).toEqual([400, 0, 0, 0]);
  expect(lightMapGradient(400, 200, "top")).toEqual([0, 0, 0, 200]);
});

it("keeps the strength between a quarter and all of the relit picture", () => {
  const start = { ...initialToolState(), kind: "relight" as const };
  expect(studioToolReducer(start, { type: "set-light-intensity", intensity: 0.1 }).lightIntensity).toBe(0.25);
  expect(studioToolReducer(start, { type: "set-light-intensity", intensity: 3 }).lightIntensity).toBe(1);
  expect(studioToolReducer(start, { type: "set-light-intensity", intensity: Number.NaN }).lightIntensity).toBe(0.5);
});
