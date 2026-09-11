import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { QueuePlanSteps } from "./QueuePlanSteps";

vi.mock("./api", () => ({ api: { queuePlanSteps: vi.fn() } }));
const clients: QueryClient[] = [];
beforeEach(() => vi.resetAllMocks());
afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); });

it("shows the running step's reported percentage without inventing progress for other steps", async () => {
  vi.mocked(api.queuePlanSteps).mockResolvedValue({
    plan_id: "plan", total: 3, next_offset: null, observed_at: "2026-09-10T12:00:00Z",
    items: [
      { id: "running", ordinal: 1, label: "Image generation", status: "running",
        blocked_by: 0, progress: 0.42, progress_scope: "stage" },
      { id: "unknown", ordinal: 2, label: "Image generation", status: "running",
        blocked_by: 0, progress: null, progress_scope: null },
      { id: "queued", ordinal: 3, label: "Image generation", status: "queued",
        blocked_by: 0, progress: 0.5, progress_scope: "stage" },
    ],
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><QueuePlanSteps planId="plan" name="Example plan" /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Show steps for Example plan" }));
  const progress = await screen.findByRole("progressbar", { name: "Step 1 current-stage progress" });
  expect(progress).toHaveAttribute("aria-valuenow", "42");
  expect(screen.getByText("42% current-stage progress")).toBeInTheDocument();
  expect(screen.getAllByRole("progressbar")).toHaveLength(1);
});

it.each([0, 1])("keeps a reported boundary value of %s visible", async (value) => {
  vi.mocked(api.queuePlanSteps).mockResolvedValue({
    plan_id: "plan", total: 1, next_offset: null, observed_at: "2026-09-10T12:00:00Z",
    items: [{ id: "step", ordinal: 1, label: "Image generation", status: "running",
      blocked_by: 0, progress: value, progress_scope: "overall" }],
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><QueuePlanSteps planId="plan" name="Example plan" /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Show steps for Example plan" }));
  expect(await screen.findByRole("progressbar", { name: "Step 1 overall progress" })).toHaveAttribute("aria-valuenow", String(value * 100));
});
