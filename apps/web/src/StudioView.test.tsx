import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StudioView } from "./StudioView";
import { api } from "./api";
import { useStudioSession } from "./useStudioSession";

vi.mock("./api", () => ({
  api: {
    favoriteArtifact: vi.fn(),
    editTemplates: vi.fn().mockResolvedValue([]),
    studioCapabilities: vi.fn().mockResolvedValue({ tools: [] }),
  },
}));
vi.mock("./useStudioSession", () => ({ useStudioSession: vi.fn() }));
// The canvas stack needs a real 2D context; none of it is under test here.
vi.mock("./StudioCanvas", () => ({ StudioCanvas: () => <div /> }));
vi.mock("./messageMedia", () => ({ artifactSource: () => "blob:picture" }));

function renderStudio() {
  vi.mocked(useStudioSession).mockReturnValue({
    steps: [{ artifactId: "art-1", instruction: null }],
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
  cleanup();
  vi.clearAllMocks();
});

describe("applying an edit", () => {
  it("keeps the instruction when the turn is refused", async () => {
    // The words were cleared at dispatch, so a refusal erased exactly what
    // would have been retyped to try again.
    const apply = vi.fn();
    vi.mocked(useStudioSession).mockReturnValue({
      steps: [{ artifactId: "art-1", instruction: null }],
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

describe("saving a picture to the library", () => {
  it("says so when the save fails instead of looking unchanged", async () => {
    // The button only relabels on success, so a failure left the picture
    // unmarked while the label still invited the same press - identical on
    // screen to never having pressed it.
    vi.mocked(api.favoriteArtifact).mockRejectedValue(new Error("artifact not found"));
    renderStudio();

    fireEvent.click(screen.getByRole("button", { name: /save to library/i }));

    await waitFor(() => expect(screen.getByText("artifact not found")).toBeTruthy());
    expect(screen.getByRole("button", { name: /save to library/i })).toBeTruthy();
  });

  it("confirms the picture is kept when the save succeeds", async () => {
    vi.mocked(api.favoriteArtifact).mockResolvedValue(undefined as never);
    renderStudio();

    fireEvent.click(screen.getByRole("button", { name: /save to library/i }));

    await waitFor(() => expect(screen.getByRole("button", { name: /kept in the library/i })).toBeTruthy());
  });
});
