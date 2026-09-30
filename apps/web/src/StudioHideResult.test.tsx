/** Hiding a result from the Studio's strip, and bringing it back. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { forgetHiddenStepsForTest } from "./studioHiddenSteps";
import { useStudioImage } from "./useStudioImage";
import { useStudioSession, type StudioStep } from "./useStudioSession";

vi.mock("./api", () => ({
  api: { favoriteArtifact: vi.fn(), artifact: vi.fn(), editTemplates: vi.fn(), studioCapabilities: vi.fn() },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
vi.mock("./useStudioImage", () => ({ useStudioImage: vi.fn() }));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: { onAvailabilityChange: (reason: string | null) => void }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

const source: StudioStep = {
  messageId: "source",
  artifactId: "art-source",
  instruction: "",
  beforeArtifactId: null,
  isSource: true,
  generationIdentity: null,
};
const first: StudioStep = {
  messageId: "answer-1",
  artifactId: "art-1",
  instruction: "calmer water",
  beforeArtifactId: "art-source",
  isSource: false,
  generationIdentity: null,
};
const second: StudioStep = { ...first, messageId: "answer-2", artifactId: "art-2" };

beforeEach(() => {
  localStorage.clear();
  forgetHiddenStepsForTest();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function open(steps: StudioStep[]) {
  vi.mocked(api.artifact).mockResolvedValue({ id: "art", favorite: false } as never);
  vi.mocked(api.editTemplates).mockResolvedValue([]);
  vi.mocked(api.studioCapabilities).mockResolvedValue({ tools: [] });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  const bitmap = { width: 400, height: 200, close: vi.fn() } as unknown as ImageBitmap;
  vi.mocked(useStudioImage).mockReturnValue({ bitmap, error: null, reload: vi.fn() });
  vi.mocked(useStudioSession).mockReturnValue({
    steps,
    previewArtifactId: null,
    sessionId: "chat-studio",
    session: null,
    busy: false,
    error: null,
    apply: vi.fn(),
  } as unknown as ReturnType<typeof useStudioSession>);
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <StudioView sourceArtifactId="art-source" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>,
  );
  return screen.getByRole("group", { name: "Edit history" });
}

it("takes the result on the canvas out of the strip, shows the one before it, and brings it back", () => {
  const strip = open([source, first, second]);
  expect(within(strip).getByRole("button", { pressed: true })).toHaveAccessibleName(/^Result of step 2/);

  fireEvent.click(screen.getByRole("button", { name: "Hide this result from the strip" }));

  expect(within(strip).queryByRole("button", { name: /^Result of step 2/ })).toBeNull();
  // The newest result still in the strip is on the canvas now, numbered as before.
  expect(within(strip).getByRole("button", { pressed: true })).toHaveAccessibleName(/^Result of step 1/);

  fireEvent.click(screen.getByRole("button", { name: "Show 1 hidden result" }));

  expect(within(strip).getByRole("button", { name: /^Result of step 2/ })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /hidden result/ })).toBeNull();
});

it("offers no hiding for the original", () => {
  open([source]);

  expect(screen.getByRole("group", { name: "Edit history" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Hide this result from the strip" })).toBeNull();
});
