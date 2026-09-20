import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { StudioView } from "./StudioView";
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
vi.mock("./useStudioImage", () => ({
  useStudioImage: () => ({ bitmap: null, error: null, reload: vi.fn() }),
}));
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({ onAvailabilityChange }: {
    onAvailabilityChange: (reason: string | null) => void;
  }) => {
    useEffect(() => onAvailabilityChange(null), [onAvailabilityChange]);
    return <div>Workflow chooser</div>;
  },
}));

let session: ReturnType<typeof useStudioSession>;

beforeEach(() => {
  session = {
    steps: [{ artifactId: "art-1", messageId: "source", instruction: "", isSource: true, generationIdentity: null }],
    previewArtifactId: null,
    sessionId: "chat-studio",
    session: null,
    busy: false,
    error: null,
    apply: vi.fn(),
  };
  vi.mocked(useStudioSession).mockImplementation(() => session);
});

afterEach(cleanup);

function studio(sourceArtifactId: string | null, client: QueryClient) {
  return (
    <QueryClientProvider client={client}>
      <StudioView sourceArtifactId={sourceArtifactId} onOpenArtifact={vi.fn()} onOpenWorkflows={vi.fn()} onClose={vi.fn()} />
    </QueryClientProvider>
  );
}

it.each([
  [null, "art-1"],
  ["art-1", "art-2"],
  ["art-1", null],
])("moves focus into Studio when its source changes from %s to %s", async (from, to) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { rerender } = render(studio(from, client));
  const control = from
    ? screen.getByRole("textbox")
    : screen.getByRole("button", { name: "Choose an image" });
  control.focus();

  rerender(studio(to, client));

  await waitFor(() => expect(screen.getByRole("heading", { name: "Image Studio" })).toHaveFocus());
  const nextControl = to
    ? screen.getByRole("textbox")
    : screen.getByRole("button", { name: "Choose an image" });
  nextControl.focus();
  rerender(studio(to, client));
  expect(nextControl).toHaveFocus();
});

it("keeps Apply focused while ignoring unavailable activations", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { rerender } = render(studio("art-1", client));
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "make it blue" } });
  const apply = screen.getByRole("button", { name: "Apply edit" });
  await waitFor(() => expect(apply).not.toHaveAttribute("aria-disabled", "true"));
  apply.focus();
  fireEvent.click(apply);
  expect(session.apply).toHaveBeenCalledTimes(1);

  session = { ...session, busy: true };
  rerender(studio("art-1", client));

  expect(apply).toHaveAttribute("aria-disabled", "true");
  expect(apply).not.toBeDisabled();
  expect(apply).toHaveFocus();
  fireEvent.click(apply);
  expect(session.apply).toHaveBeenCalledTimes(1);

  session = { ...session, busy: false };
  rerender(studio("art-1", client));
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "" } });
  fireEvent.click(apply);
  expect(session.apply).toHaveBeenCalledTimes(1);
});
