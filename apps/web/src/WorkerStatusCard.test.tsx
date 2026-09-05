import { afterEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";

import type { WorkerStatus } from "./types";
import { WorkerStatusCard } from "./WorkerStatusCard";

vi.mock("./api", () => ({ api: {} }));

const worker = (overrides: Partial<WorkerStatus> = {}): WorkerStatus => ({
  name: "media",
  state: "ready",
  managed: true,
  running: true,
  pid: 42,
  profile_id: null,
  command: [],
  exit_code: null,
  estimated_memory_bytes: null,
  current_memory_bytes: null,
  peak_memory_bytes: null,
  active_jobs: 1,
  queued_jobs: 0,
  ...overrides,
});

function renderCard(status: WorkerStatus) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const card = (value: WorkerStatus) => (
    <QueryClientProvider client={client}>
      <WorkerStatusCard
        worker={value}
        startPending={false}
        stopPending={false}
        onStart={() => undefined}
        onStop={() => undefined}
      />
    </QueryClientProvider>
  );
  const result = render(card(status));
  return (next: WorkerStatus) => result.rerender(card(next));
}

afterEach(cleanup);

describe("WorkerStatusCard progress reports", () => {
  it.each([
    { age: 0, duration: "0s" },
    { age: 95, duration: "1m 35s" },
    { age: 3_600, duration: "60m" },
  ])("shows the reported age $age while jobs are active", ({ age, duration }) => {
    renderCard(worker({ progress_age_seconds: age }));

    expect(screen.getByText(`Last progress report: ${duration} ago`)).toBeInTheDocument();
    expect(screen.queryByText(/awaiting first progress report/i)).not.toBeInTheDocument();
  });

  it.each([null, undefined])("keeps an unreported active job unknown for %s", (age) => {
    renderCard(worker({ progress_age_seconds: age }));

    expect(screen.getByText("Awaiting first progress report")).toBeInTheDocument();
    expect(screen.queryByText(/last progress report:|\bidle\b|\bstuck\b/i)).not.toBeInTheDocument();
  });

  it("does not show a previous report for queued work without an active job", () => {
    renderCard(worker({ active_jobs: 0, queued_jobs: 3, progress_age_seconds: 95 }));

    expect(screen.queryByText(/progress report/i)).not.toBeInTheDocument();
  });

  it("follows reported worker snapshots and hides the age after work ends", () => {
    const update = renderCard(worker({ progress_age_seconds: null }));
    expect(screen.getByText("Awaiting first progress report")).toBeInTheDocument();

    update(worker({ progress_age_seconds: 12 }));
    expect(screen.getByText("Last progress report: 12s ago")).toBeInTheDocument();
    expect(screen.queryByText(/awaiting first progress report/i)).not.toBeInTheDocument();

    update(worker({ active_jobs: 0, progress_age_seconds: 12 }));
    expect(screen.queryByText(/progress report/i)).not.toBeInTheDocument();
  });
});
