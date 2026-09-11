import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { JobsPanel } from "./JobsPanel";
import { api } from "./api";
import type { QueueActivityItem, QueueActivityPage } from "./types";

vi.mock("./api", () => ({ api: {
  jobActivity: vi.fn(), queueActivity: vi.fn(), queuePlanSteps: vi.fn(), cancelJob: vi.fn(),
  pauseDownload: vi.fn(), resumeDownload: vi.fn(), retryJob: vi.fn(),
} }));
const clients: QueryClient[] = [];
const stamp = "2026-09-01T00:00:00Z";
function item(id: string, lane: QueueActivityItem["lane"] = "generation"): QueueActivityItem {
  return { owner_type: "work_plan", owner_id: id, label: "Submitted work", lane,
    status: "running", chat_id: "example-chat", chat_title: "Example " + id,
    created_at: stamp, updated_at: stamp, step_count: 3, completed_steps: 1,
    blocked_steps: 1, active_jobs: 2, running_jobs: 1, queued_jobs: 1, paused_jobs: 0,
    progress: null };
}
function page(items: QueueActivityItem[], next: string | null = null, total = items.length): QueueActivityPage {
  return { items, next_cursor: next, total,
    lane_counts: { generation: total, transfer: 0, install: 0 }, observed_at: stamp };
}
beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.jobActivity).mockResolvedValue({ active: [], active_count: 0, recent_issues: [] });
});
afterEach(() => { cleanup(); clients.splice(0).forEach((client) => client.clear()); });
function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><JobsPanel /></QueryClientProvider>);
  return client;
}

it("opens grouped accepted work even when no standalone jobs are visible and restores focus", async () => {
  vi.mocked(api.queueActivity).mockResolvedValue(page([item("plan")]));
  open();
  const trigger = await screen.findByRole("button", { name: "View accepted work" });
  expect(api.queueActivity).not.toHaveBeenCalled();
  trigger.focus(); fireEvent.click(trigger);
  const dialog = await screen.findByRole("dialog", { name: "Accepted work" });
  expect(await screen.findByText("Example plan")).toBeInTheDocument();
  expect(screen.getByText("1 of 3 steps complete")).toBeInTheDocument();
  expect(screen.getByText("1 step waiting for prerequisites")).toBeInTheDocument();
  expect(screen.getByText(/Ordered by acceptance time/)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Hold|Reorder/ })).not.toBeInTheDocument();
  fireEvent.keyDown(dialog, { key: "Escape" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

it("pages without duplicating owners and starts a fresh query when changing lanes", async () => {
  vi.mocked(api.queueActivity).mockImplementation(async ({ lane, cursor }) => {
    if (lane === "transfer") return page([{ ...item("transfer", "transfer"),
      owner_type: "job", label: "Download", chat_id: null, chat_title: null,
      step_count: 0, completed_steps: 0, blocked_steps: 0, progress: 0.25 }]);
    return cursor ? page([item("first"), item("second")], null, 2) : page([item("first")], "next", 2);
  });
  open(); fireEvent.click(await screen.findByRole("button", { name: "View accepted work" }));
  expect(await screen.findByText("Showing 1 of 2 active items")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load more accepted work" }));
  expect(await screen.findByText("Showing 2 of 2 active items")).toBeInTheDocument();
  expect(screen.getAllByText("Example first")).toHaveLength(1);
  expect(screen.queryByRole("button", { name: "Load more accepted work" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "Work category" }), { target: { value: "transfer" } });
  expect(await screen.findByText("Download")).toBeInTheDocument();
  expect(screen.queryByText("Example first")).not.toBeInTheDocument();
  expect(screen.getByRole("progressbar", { name: "Download progress" })).toHaveAttribute("aria-valuenow", "25");
  expect(vi.mocked(api.queueActivity).mock.calls.at(-1)?.[0]).toMatchObject({ lane: "transfer", cursor: null, limit: 50 });
});

it("keeps a failed read distinct from an empty queue and retries from the first page", async () => {
  vi.mocked(api.queueActivity).mockRejectedValue(new Error("Accepted work is unavailable"));
  open(); fireEvent.click(await screen.findByRole("button", { name: "View accepted work" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Accepted work is unavailable");
  expect(screen.queryByText("No active accepted work in this category.")).not.toBeInTheDocument();
  vi.mocked(api.queueActivity).mockResolvedValue(page([]));
  fireEvent.click(screen.getByRole("button", { name: "Refresh accepted work" }));
  expect(await screen.findByText("No active accepted work in this category.")).toBeInTheDocument();
  await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
});

it("keeps the open category when the last job completes", async () => {
  vi.mocked(api.jobActivity).mockResolvedValue({ active_count: 1, recent_issues: [], active: [{
    id: "active", kind: "image", status: "queued", run_id: null, progress: 0,
    phase: "Waiting", payload_json: {}, result_json: {}, error: null, attempt: 0,
    cancellable: true, created_at: stamp, updated_at: stamp, started_at: null, completed_at: null,
  }] });
  vi.mocked(api.queueActivity).mockResolvedValue(page([item("plan")]));
  const client = open();
  await screen.findByText("1 active job");
  fireEvent.click(screen.getByRole("button", { name: "View accepted work" }));
  const dialog = await screen.findByRole("dialog", { name: "Accepted work" });
  fireEvent.change(screen.getByRole("combobox", { name: "Work category" }), { target: { value: "transfer" } });
  await screen.findByText("Example plan");
  await act(async () => {
    client.setQueryData(["jobs", "activity", 100], { active: [], active_count: 0, recent_issues: [] });
  });
  await waitFor(() => expect(screen.queryByText("1 active job")).not.toBeInTheDocument());
  expect(screen.getByRole("dialog", { name: "Accepted work" })).toBe(dialog);
  expect(screen.getByRole("combobox", { name: "Work category" })).toHaveValue("transfer");
  fireEvent.keyDown(dialog, { key: "Escape" });
  expect(screen.getByRole("button", { name: "View accepted work" })).toHaveFocus();
});

function steps(planId: string, offset = 0, total = 101, ordinalStart = 0) {
  return { plan_id: planId, total, next_offset: offset + 50 < total ? offset + 50 : null,
    observed_at: stamp,
    items: Array.from({ length: Math.min(50, total - offset) }, (_, index) => ({
      id: "step-" + (offset + index), ordinal: ordinalStart + offset + index, label: "Image generation",
      status: offset + index === 0 ? "complete" as const : offset + index === 2 ? "blocked" as const : "queued" as const,
      blocked_by: offset + index === 2 ? 1 : 0,
      progress: null, progress_scope: null,
    })),
  };
}

it.each([0, 1, 7])("numbers paged steps by position when stored ordinals start at %s", async (ordinalStart) => {
  vi.mocked(api.queueActivity).mockResolvedValue(page([{ ...item("plan"), step_count: 101 }]));
  vi.mocked(api.queuePlanSteps).mockImplementation(async (id, offset) => steps(id, offset, 101, ordinalStart));
  open(); fireEvent.click(await screen.findByRole("button", { name: "View accepted work" }));
  const expand = await screen.findByRole("button", { name: "Show steps for Example plan" });
  expect(api.queuePlanSteps).not.toHaveBeenCalled();
  fireEvent.click(expand);
  expect(await screen.findByText("Showing 50 of 101 steps")).toBeInTheDocument();
  const region = screen.getByRole("region", { name: "Steps for Example plan" });
  expect(within(region).getAllByRole("listitem")).toHaveLength(50);
  expect(within(region).getAllByRole("listitem")[0]).toHaveTextContent("Step 1 · Image generation");
  expect(within(region).getByText("1 prerequisite unfinished")).toBeInTheDocument();
  fireEvent.click(within(region).getByRole("button", { name: "Load more steps" }));
  expect(await screen.findByText("Showing 100 of 101 steps")).toBeInTheDocument();
  fireEvent.click(within(region).getByRole("button", { name: "Load more steps" }));
  expect(await screen.findByText("Showing 101 of 101 steps")).toBeInTheDocument();
  expect(within(region).getAllByRole("listitem").map(row => within(row).getByText(/^Step /).textContent))
    .toEqual(Array.from({ length: 101 }, (_, index) => "Step " + (index + 1) + " · Image generation"));
  expect(within(region).queryByRole("button", { name: "Load more steps" })).not.toBeInTheDocument();
  expect(vi.mocked(api.queuePlanSteps).mock.calls.map(([id, offset]) => [id, offset]))
    .toEqual([["plan", 0], ["plan", 50], ["plan", 100]]);
});

it("retries a failed step read without treating it as an empty plan", async () => {
  vi.mocked(api.queueActivity).mockResolvedValue(page([item("plan")]));
  vi.mocked(api.queuePlanSteps).mockRejectedValue(new Error("Step details unavailable"));
  open(); fireEvent.click(await screen.findByRole("button", { name: "View accepted work" }));
  fireEvent.click(await screen.findByRole("button", { name: "Show steps for Example plan" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Step details unavailable");
  expect(screen.queryByText("No steps recorded for this plan.")).not.toBeInTheDocument();
  vi.mocked(api.queuePlanSteps).mockResolvedValue(steps("plan", 0, 1));
  fireEvent.click(screen.getByRole("button", { name: "Retry step details" }));
  expect(await screen.findByText("Showing 1 of 1 step")).toBeInTheDocument();
});

it("cancels an in-flight step read when its disclosure closes", async () => {
  vi.mocked(api.queueActivity).mockResolvedValue(page([item("plan")]));
  let finish!: (value: ReturnType<typeof steps>) => void;
  vi.mocked(api.queuePlanSteps).mockImplementation(async () => new Promise(resolve => { finish = resolve; }));
  open(); fireEvent.click(await screen.findByRole("button", { name: "View accepted work" }));
  fireEvent.click(await screen.findByRole("button", { name: "Show steps for Example plan" }));
  await waitFor(() => expect(api.queuePlanSteps).toHaveBeenCalledOnce());
  const signal = vi.mocked(api.queuePlanSteps).mock.calls[0][2];
  fireEvent.click(screen.getByRole("button", { name: "Hide steps for Example plan" }));
  expect(signal?.aborted).toBe(true);
  await act(async () => finish(steps("plan")));
  expect(screen.queryByRole("region", { name: "Steps for Example plan" })).not.toBeInTheDocument();
});
