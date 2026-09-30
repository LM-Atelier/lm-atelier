/** Closing the Studio loses nothing, so it asks nothing, even while an edit runs. */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { StudioCloseButton } from "./StudioCloseButton";
import { StudioView } from "./StudioView";
import { api } from "./api";
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

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function open(steps: StudioStep[], busy: boolean) {
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
    busy,
    error: null,
    apply: vi.fn(),
  } as unknown as ReturnType<typeof useStudioSession>);
  const onClose = vi.fn();
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <StudioView sourceArtifactId="art-source" onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={onClose} />
    </QueryClientProvider>,
  );
  return onClose;
}

it("puts an edited picture down at once, since opening it again brings its edits back", () => {
  const onClose = open([source, first], false);

  fireEvent.click(screen.getByRole("button", { name: "Close" }));

  expect(onClose).toHaveBeenCalledTimes(1);
  expect(screen.queryByRole("dialog")).toBeNull();
});

it("closes while an edit is still running, which carries on without the Studio", () => {
  const onClose = open([source, first], true);
  const close = screen.getByRole("button", { name: "Close" });
  expect(close).toHaveAttribute("aria-disabled", "false");

  fireEvent.click(close);

  expect(onClose).toHaveBeenCalledTimes(1);
});

it("waits while a replacement is between its two edits, and says why", () => {
  const onClose = vi.fn();
  const { rerender } = render(<StudioCloseButton halfway onClose={onClose} />);
  const close = screen.getByRole("button", { name: "Close" });
  expect(close).toHaveAttribute("aria-disabled", "true");
  expect(close).toHaveAttribute("title", "Closing waits until the replacement's second edit is sent");

  fireEvent.click(close);
  expect(onClose).not.toHaveBeenCalled();

  rerender(<StudioCloseButton halfway={false} onClose={onClose} />);
  fireEvent.click(close);
  expect(onClose).toHaveBeenCalledTimes(1);
});
