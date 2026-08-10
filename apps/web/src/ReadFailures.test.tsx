import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { JobsPanel } from "./JobsPanel";
import { WorkflowFamilyArchive } from "./WorkflowFamilyArchive";
import { api } from "./api";

vi.mock("./api", () => ({
  api: {
    jobs: vi.fn(),
    cancelJob: vi.fn(),
    pauseDownload: vi.fn(),
    resumeDownload: vi.fn(),
    retryJob: vi.fn(),
    workflowFamilyRemovalImpact: vi.fn(),
    updateWorkflowFamily: vi.fn(),
  },
}));

function wrap(node: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("a read that failed", () => {
  it("keeps the jobs panel rather than removing the only explanation", async () => {
    // Returning null took the status surface away at exactly the moment
    // something was wrong, leaving the workspace looking idle.
    vi.mocked(api.jobs).mockRejectedValue(new Error("jobs could not be read"));

    wrap(<JobsPanel />);

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("jobs could not be read"));
  });

  it("keeps image edit checks out of the ordinary jobs surface", async () => {
    vi.mocked(api.jobs).mockResolvedValue([
      {
        id: "check-running",
        kind: "edit_verify",
        status: "running",
      },
      {
        id: "check-failed",
        kind: "edit_verify",
        status: "failed",
        error: "assessment unavailable",
      },
    ] as never);

    wrap(<JobsPanel />);

    await waitFor(() => expect(api.jobs).toHaveBeenCalledTimes(1));
    expect(screen.queryByLabelText("Jobs")).toBeNull();
  });

  it("will not offer to archive a family whose impact is unknown", async () => {
    // An impact that could not be read is not an impact of nothing, and an
    // empty "what changes" list reads as a finding rather than a silence.
    vi.mocked(api.workflowFamilyRemovalImpact).mockRejectedValue(
      new Error("impact could not be read"),
    );

    wrap(
      <WorkflowFamilyArchive
        family={{ id: "fam-1", name: "Portraits" } as never}
        onClose={vi.fn()}
      />,
    );

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("impact could not be read"),
    );
    expect(screen.queryByRole("button", { name: "Archive it" })).toBeNull();
    expect(api.updateWorkflowFamily).not.toHaveBeenCalled();
  });
});
