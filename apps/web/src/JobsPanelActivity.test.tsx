import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { JobsPanel } from "./JobsPanel";
import { api } from "./api";
import type { Job, JobStatus } from "./types";

vi.mock("./api", () => ({ api: {
  jobs: vi.fn(), jobActivity: vi.fn(), cancelJob: vi.fn(), pauseDownload: vi.fn(),
  resumeDownload: vi.fn(), retryJob: vi.fn(),
} }));
const clients: QueryClient[] = [];
const dismissalKey = "lm-atelier-dismissed-job-issues-before";
function job(id: string, status: JobStatus = "queued"): Job {
  return { id, kind: "image", status, run_id: null, progress: 0, phase: "Waiting for execution",
    payload_json: {}, result_json: {}, error: null, attempt: 0, cancellable: true,
    created_at: "2026-09-01T00:00:00Z", updated_at: "2026-09-10T00:00:00Z",
    started_at: null, completed_at: null };
}
beforeEach(() => {
  vi.resetAllMocks();
  localStorage.removeItem(dismissalKey);
  vi.mocked(api.jobs).mockResolvedValue([]);
});
afterEach(() => {
  cleanup(); clients.splice(0).forEach((client) => client.clear());
  localStorage.removeItem(dismissalKey);
});
function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><JobsPanel /></QueryClientProvider>);
  return client;
}

describe("active jobs stay visible independently of recent history", () => {
  it("shows old active work and keeps its existing cancel action", async () => {
    vi.mocked(api.jobActivity).mockResolvedValue({ active: [job("old-active")], active_count: 1, recent_issues: [] });
    vi.mocked(api.cancelJob).mockResolvedValue(job("old-active", "cancelled"));
    open();
    expect(await screen.findByText("1 active job")).toBeInTheDocument();
    expect(screen.getByText("Waiting for execution")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel job" }));
    await waitFor(() => expect(api.cancelJob).toHaveBeenCalledOnce());
    expect(vi.mocked(api.cancelJob).mock.calls[0][0]).toBe("old-active");
    expect(api.jobs).not.toHaveBeenCalled();
  });

  it("reports the full active count while expanding the bounded visible list", async () => {
    const all = Array.from({ length: 601 }, (_, index) => job("active-" + index));
    vi.mocked(api.jobActivity).mockImplementation(async (limit) => ({
      active: all.slice(0, limit), active_count: all.length, recent_issues: [],
    }));
    open();
    expect(await screen.findByText("601 active jobs")).toBeInTheDocument();
    const showMore = screen.getByRole("button", { name: "Show more active jobs" });
    for (const count of [100, 200, 300, 400]) {
      expect(await screen.findByText(`Showing ${count} of 601 active jobs.`)).toBeInTheDocument();
      fireEvent.click(showMore);
    }
    expect(await screen.findByText("Showing 500 of 601 active jobs.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show more active jobs" })).not.toBeInTheDocument();
    expect(screen.getAllByRole("progressbar", { hidden: true })).toHaveLength(500);
    expect([...new Set(vi.mocked(api.jobActivity).mock.calls.map(([limit]) => limit))]).toEqual([100, 200, 300, 400, 500]);
  }, 15_000);

  it("keeps recent issues dismissible without hiding active work", async () => {
    vi.mocked(api.jobActivity).mockResolvedValue({
      active: [job("active")], active_count: 1,
      recent_issues: [{ ...job("failed", "failed"), phase: "Download stopped", error: "Transfer unavailable" }],
    });
    open();
    expect(await screen.findByText("Transfer unavailable")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear recent job issues" }));
    expect(screen.queryByText("Transfer unavailable")).not.toBeInTheDocument();
    expect(screen.getByText("1 active job")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("hides cached activity after a failed refresh and retries the activity endpoint", async () => {
    vi.mocked(api.jobActivity).mockResolvedValue({ active: [job("active")], active_count: 1, recent_issues: [] });
    const client = open();
    expect(await screen.findByText("1 active job")).toBeInTheDocument();
    vi.mocked(api.jobActivity).mockRejectedValue(new Error("Activity could not be read"));
    await client.invalidateQueries({ queryKey: ["jobs"] });
    expect(await screen.findByRole("alert")).toHaveTextContent("Activity could not be read");
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    vi.mocked(api.jobActivity).mockResolvedValue({ active: [job("active")], active_count: 1, recent_issues: [] });
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("1 active job")).toBeInTheDocument();
  });
});
