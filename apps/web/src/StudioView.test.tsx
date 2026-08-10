import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useEffect } from "react";
import { StudioView } from "./StudioView";
import { api } from "./api";
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
let workflowAvailabilityReason: string | null = null;
vi.mock("./StudioWorkflowSelector", () => ({
  StudioWorkflowSelector: ({
    onAvailabilityChange,
  }: {
    onAvailabilityChange: (reason: string | null) => void;
    onSelectionChange: () => void;
  }) => {
    useEffect(
      () => onAvailabilityChange(workflowAvailabilityReason),
      [onAvailabilityChange],
    );
    return <div>Workflow chooser</div>;
  },
}));
// The canvas stack needs a real 2D context; none of it is under test here.
vi.mock("./StudioCanvas", () => ({ StudioCanvas: () => <div /> }));
vi.mock("./messageMedia", () => ({ artifactSource: () => "blob:picture" }));

function renderStudio() {
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [{ artifactId: "art-1", instruction: null }],
    sessionId: "chat-studio",
    busy: false,
    error: null,
    apply: vi.fn(),
  } as unknown as ReturnType<typeof useStudioSession>);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <StudioView
        sourceArtifactId="art-1"
        onOpenArtifact={vi.fn()}
        onOpenWorkflows={vi.fn()}
        onClose={vi.fn()}
      />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  workflowAvailabilityReason = null;
  cleanup();
  vi.clearAllMocks();
});

describe("applying an edit", () => {
  it("says when a pinned recipe overrides the displayed workflow choice", async () => {
    workflowAvailabilityReason = "Loading the current workflow choice.";
    vi.mocked(api.editTemplates).mockResolvedValue([{
      id: "recipe-1",
      name: "Pinned portrait edit",
      description: "",
      instruction: "make it warmer",
      operation: "image_to_image",
      settings_json: {},
      workflow_revision_id: "revision-pinned",
      model_profile_id: null,
      mask_mode: "none",
      trigger_words_json: [],
      content_rating: "general",
      builtin: false,
      enabled: true,
    }]);
    renderStudio();

    fireEvent.click(await screen.findByRole("button", { name: "Pinned portrait edit" }));

    expect(screen.getByRole("status")).toHaveTextContent(
      "Pinned portrait edit supplies the workflow for this edit.",
    );
    expect(screen.getByRole("button", { name: "Apply" })).toBeEnabled();

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "make it cooler" } });
    expect(screen.queryByText(/supplies the workflow for this edit/i)).toBeNull();
  });

  it("does not let an unpinned recipe bypass an unavailable workflow choice", async () => {
    workflowAvailabilityReason = "Loading the current workflow choice.";
    vi.mocked(api.editTemplates).mockResolvedValue([{
      id: "recipe-1",
      name: "Unpinned portrait edit",
      description: "",
      instruction: "make it warmer",
      operation: "image_to_image",
      settings_json: {},
      workflow_revision_id: null,
      model_profile_id: null,
      mask_mode: "none",
      trigger_words_json: [],
      content_rating: "general",
      builtin: false,
      enabled: true,
    }]);
    renderStudio();

    fireEvent.click(await screen.findByRole("button", { name: "Unpinned portrait edit" }));

    expect(screen.queryByText(/supplies the workflow for this edit/i)).toBeNull();
    expect(screen.getByRole("button", { name: "Apply" })).toBeDisabled();
  });

  it("keeps the instruction when the turn is refused", async () => {
    // The words were cleared at dispatch, so a refusal erased exactly what
    // would have been retyped to try again.
    const apply = vi.fn();
    vi.mocked(useStudioSession).mockReturnValue({
      steps: [{ artifactId: "art-1", instruction: null }],
      sessionId: "chat-studio",
      busy: false,
      error: null,
      apply,
    } as unknown as ReturnType<typeof useStudioSession>);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <StudioView
          sourceArtifactId="art-1"
          onOpenArtifact={vi.fn()}
          onOpenWorkflows={vi.fn()}
          onClose={vi.fn()}
        />
      </QueryClientProvider>,
    );

    const words = screen.getByRole("textbox");
    fireEvent.change(words, { target: { value: "make it warmer" } });
    fireEvent.click(screen.getByRole("button", { name: /apply/i }));

    // Dispatched, not accepted: nothing has called back yet.
    expect(apply).toHaveBeenCalled();
    expect((words as HTMLTextAreaElement).value).toBe("make it warmer");
  });
});

describe("marking a picture in the library", () => {
  it("says so when the request fails instead of looking unchanged", async () => {
    // The control reflects the artifact, so a failure leaves the picture
    // unmarked while the label still invites the same press - identical on
    // screen to never having pressed it.
    vi.mocked(api.favoriteArtifact).mockRejectedValue(new Error("artifact not found"));
    renderStudio();

    const button = await screen.findByRole("button", { name: /favorite/i });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() => expect(screen.getByText("artifact not found")).toBeTruthy());
    expect(screen.getByRole("button", { name: /favorite/i })).toBeTruthy();
  });

  it("shows what the artifact says rather than what this visit did", async () => {
    // Read locally, the mark was forgotten on every reopen and could never be
    // taken back. Read from the artifact, it survives leaving and returning.
    vi.mocked(api.artifact)
      .mockResolvedValueOnce({ id: "art-1", favorite: false } as never)
      .mockResolvedValue({ id: "art-1", favorite: true } as never);
    vi.mocked(api.favoriteArtifact).mockResolvedValue(undefined as never);
    renderStudio();

    const button = await screen.findByRole("button", { name: /favorite/i });
    await waitFor(() => expect(button).toBeEnabled());
    expect(button.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(button);

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /favorited/i })).toBeTruthy(),
    );
    expect(api.favoriteArtifact).toHaveBeenCalledWith("art-1", true);
  });

  it("takes the mark back rather than only ever adding it", async () => {
    vi.mocked(api.artifact).mockResolvedValue({ id: "art-1", favorite: true } as never);
    vi.mocked(api.favoriteArtifact).mockResolvedValue(undefined as never);
    renderStudio();

    const button = await screen.findByRole("button", { name: /favorited/i });
    await waitFor(() => expect(button).toBeEnabled());
    fireEvent.click(button);

    await waitFor(() => expect(api.favoriteArtifact).toHaveBeenCalledWith("art-1", false));
  });
});

describe("what the studio believes it can do", () => {
  it("re-asks on every entry, so an install made in between is seen", async () => {
    // The answer was held for a minute. Someone following the studio's own
    // "Browse workflows" button, installing the workflow, and coming straight
    // back was told by the same sentence that it still was not installed.
    vi.mocked(useStudioSession).mockReturnValue({
      steps: [{ artifactId: "art-1", instruction: null }],
      sessionId: "chat-studio",
      busy: false,
      error: null,
      apply: vi.fn(),
    } as unknown as ReturnType<typeof useStudioSession>);
    // One client across both visits: a fresh cache each time would prove
    // nothing about whether the cached answer is re-asked.
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const view = (
      <QueryClientProvider client={client}>
        <StudioView
          sourceArtifactId="art-1"
          onOpenArtifact={vi.fn()}
          onOpenWorkflows={vi.fn()}
          onClose={vi.fn()}
        />
      </QueryClientProvider>
    );

    const first = render(view);
    await waitFor(() => expect(api.studioCapabilities).toHaveBeenCalledTimes(1));
    first.unmount();
    render(view);

    await waitFor(() => expect(api.studioCapabilities).toHaveBeenCalledTimes(2));
  });
});
