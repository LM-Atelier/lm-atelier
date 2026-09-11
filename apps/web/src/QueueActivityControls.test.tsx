import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { QueueActivityDialog } from "./QueueActivityDialog";
import type { QueueActivityItem } from "./types";

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: { queueActivity: vi.fn(), queueControl: vi.fn(), queuePlanSteps: vi.fn() },
}));

const clients: QueryClient[] = [];
let current: QueueActivityItem;
const stamp = "2026-09-01T00:00:00Z";

beforeEach(() => {
  vi.resetAllMocks();
  current = {
    owner_type: "work_plan", owner_id: "plan-a", label: "Submitted work", lane: "generation",
    status: "queued", chat_id: "chat-a", chat_title: "Example",
    created_at: stamp, updated_at: stamp, step_count: 2, completed_steps: 0,
    blocked_steps: 0, active_jobs: 2, running_jobs: 0, queued_jobs: 2, paused_jobs: 0,
    progress: null, control_state: "eligible", control_revision: 0, allowed_actions: ["hold"],
  };
  vi.mocked(api.queueActivity).mockImplementation(async () => ({
    items: [{ ...current }], total: 1, lane_counts: { generation: 1, transfer: 0, install: 0 },
    next_cursor: null, observed_at: stamp,
  }));
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
});

function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}><QueueActivityDialog onClose={() => undefined} /></QueryClientProvider>);
  return client;
}

it("holds and releases from authoritative state without losing button focus", async () => {
  vi.mocked(api.queueControl).mockImplementation(async (_plan, action) => {
    current = { ...current, control_state: action === "hold" ? "held" : "eligible",
      control_revision: (current.control_revision ?? 0) + 1,
      allowed_actions: action === "hold" ? ["release"] : ["hold"] };
    return { owner_id: current.owner_id, control_state: current.control_state!,
      control_revision: current.control_revision!, eligible_since: null };
  });
  open();
  const hold = await screen.findByRole("button", { name: "Hold Example" });
  hold.focus();
  fireEvent.click(hold);
  expect(await screen.findByText("Held — queued work will wait.")).toBeInTheDocument();
  const release = await screen.findByRole("button", { name: "Release Example" });
  expect(release).toHaveFocus();
  expect(api.queueControl).toHaveBeenLastCalledWith("plan-a", "hold", {
    expected_revision: 0, idempotency_key: expect.any(String),
  });
  fireEvent.click(release);
  await screen.findByRole("button", { name: "Hold Example" });
  expect(api.queueControl).toHaveBeenLastCalledWith("plan-a", "release", {
    expected_revision: 1, idempotency_key: expect.any(String),
  });
  expect(screen.queryByText("Held — queued work will wait.")).not.toBeInTheDocument();
});

it("retains one command key after a network failure and ignores an in-flight retry", async () => {
  let finish: (() => void) | undefined;
  vi.mocked(api.queueControl)
    .mockRejectedValueOnce(new Error("Connection interrupted"))
    .mockImplementationOnce(() => new Promise((resolve) => {
      finish = () => resolve({ owner_id: "plan-a", control_state: "held",
        control_revision: 1, eligible_since: null });
    }));
  open();
  fireEvent.click(await screen.findByRole("button", { name: "Hold Example" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Connection interrupted");
  const first = vi.mocked(api.queueControl).mock.calls[0]?.[2];
  fireEvent.click(screen.getByRole("button", { name: "Hold Example" }));
  await waitFor(() => expect(api.queueControl).toHaveBeenCalledTimes(2));
  expect(vi.mocked(api.queueControl).mock.calls[1]?.[2]).toEqual(first);
  expect(screen.getByRole("button", { name: "Hold Example" })).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(screen.getByRole("button", { name: "Hold Example" }));
  expect(api.queueControl).toHaveBeenCalledTimes(2);
  finish?.();
  await waitFor(() => expect(screen.getByRole("button", { name: "Hold Example" })).toHaveAttribute("aria-disabled", "false"));
});

it("refreshes a conflict and uses the refreshed revision for the next explicit action", async () => {
  vi.mocked(api.queueControl).mockImplementationOnce(async () => {
    current = { ...current, control_state: "held", control_revision: 1, allowed_actions: ["release"] };
    throw new ApiError(409, "The queue changed", "The queue changed", "queue-control-conflict");
  }).mockResolvedValue({ owner_id: "plan-a", control_state: "eligible",
    control_revision: 2, eligible_since: stamp });
  open();
  fireEvent.click(await screen.findByRole("button", { name: "Hold Example" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("The queue changed");
  const release = await screen.findByRole("button", { name: "Release Example" });
  fireEvent.click(release);
  await waitFor(() => expect(api.queueControl).toHaveBeenCalledTimes(2));
  const calls = vi.mocked(api.queueControl).mock.calls;
  expect(calls[1]?.[2].expected_revision).toBe(1);
  expect(calls[1]?.[2].idempotency_key).not.toBe(calls[0]?.[2].idempotency_key);
});

it("offers no control when the server advertises none", async () => {
  current = { ...current, owner_type: "job", chat_id: null, chat_title: null,
    control_state: null, control_revision: null, allowed_actions: [] };
  open();
  await screen.findByText("Submitted work");
  expect(screen.queryByRole("button", { name: /^(Hold|Release) / })).not.toBeInTheDocument();
  expect(api.queueControl).not.toHaveBeenCalled();
});


it.each([false, true])("keeps pending controls focusable and respects a deliberate focus change: %s", async (moveFocus) => {
  let finish: (() => void) | undefined;
  vi.mocked(api.queueControl).mockImplementationOnce(() => new Promise((resolve) => {
    finish = () => {
      current = { ...current, control_state: "held", control_revision: 1, allowed_actions: ["release"] };
      resolve({ owner_id: "plan-a", control_state: "held", control_revision: 1, eligible_since: null });
    };
  }));
  open();
  const hold = await screen.findByRole("button", { name: "Hold Example" });
  hold.focus();
  fireEvent.click(hold);
  await waitFor(() => expect(hold).toHaveAttribute("aria-disabled", "true"));
  expect(hold).toBeEnabled();
  expect(hold).toHaveFocus();
  const category = screen.getByRole("combobox", { name: "Work category" });
  if (moveFocus) category.focus();
  finish?.();
  const release = await screen.findByRole("button", { name: "Release Example" });
  await waitFor(() => expect(moveFocus ? category : release).toHaveFocus());
});
