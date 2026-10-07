import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { QueuePlanSteps } from "./QueuePlanSteps";

vi.mock("./api", () => ({ api: { queuePlanSteps: vi.fn() } }));
const clients: QueryClient[] = [];
afterEach(() => { cleanup(); clients.splice(0).forEach(client => client.clear()); vi.resetAllMocks(); });

it.each([0, 1, 2])("shows %s recorded media outputs separately from unfinished work", async count => {
  vi.mocked(api.queuePlanSteps).mockResolvedValue({
    plan_id: "plan", total: 3, next_offset: null, observed_at: "2026-09-11T02:00:00Z",
    items: [
      { id: "done", ordinal: 1, label: "Image generation", status: "complete", blocked_by: 0,
        progress: null, progress_scope: null, recorded_media_outputs: count },
      { id: "unknown", ordinal: 2, label: "Image generation", status: "complete", blocked_by: 0,
        progress: null, progress_scope: null, recorded_media_outputs: null },
      { id: "running", ordinal: 3, label: "Image generation", status: "running", blocked_by: 0,
        progress: null, progress_scope: null, recorded_media_outputs: 4 },
    ],
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><QueuePlanSteps planId="plan" name="Example plan" /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Show steps for Example plan" }));
  expect(await screen.findByText(String(count) + " recorded media " + (count === 1 ? "output" : "outputs"))).toBeInTheDocument();
  expect(screen.getAllByText(/recorded media output/)).toHaveLength(1);
});
