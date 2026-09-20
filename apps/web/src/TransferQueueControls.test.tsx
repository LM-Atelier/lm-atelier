import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { QueueActivityDialog } from "./QueueActivityDialog";
import type { TransferQueuePolicy } from "./types";

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: {
    queueActivity: vi.fn(), queueControl: vi.fn(), queuePlanSteps: vi.fn(),
    generationQueuePolicy: vi.fn(), generationQueueControl: vi.fn(),
    transferQueuePolicy: vi.fn(), transferQueueControl: vi.fn(),
  },
}));

const clients: QueryClient[] = [];
let current: TransferQueuePolicy;

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(api.generationQueuePolicy).mockResolvedValue({ lane: "generation", dispatch_state: "open", revision: 0, running_jobs: 0, allowed_actions: ["pause_after_current"] });
  current = { lane: "transfer", dispatch_state: "open", revision: 0,
    running_jobs: 0, allowed_actions: ["pause_after_current"] };
  vi.mocked(api.queueActivity).mockResolvedValue({
    items: [], total: 0, lane_counts: { generation: 0, transfer: 0, install: 0 },
    next_cursor: null, observed_at: "2026-09-01T00:00:00Z",
  });
  vi.mocked(api.transferQueuePolicy).mockImplementation(async () => ({ ...current }));
});

afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  vi.useRealTimers();
});

function open() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}>
    <QueueActivityDialog onClose={() => undefined} />
  </QueryClientProvider>);
  return client;
}

it("names the transfer lane while its first state is still loading", async () => {
  vi.mocked(api.transferQueuePolicy).mockImplementation(() => new Promise(() => undefined));
  open();
  expect(await screen.findByText("Loading transfers state…")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Pause transfers after current work" })).not.toBeInTheDocument();
});

it.each([false, true])("preserves focus and prevents duplicate pause while saving: %s", async (moveFocus) => {
  let finish: (() => void) | undefined;
  vi.mocked(api.transferQueueControl).mockImplementationOnce(() => new Promise((resolve) => {
    finish = () => {
      current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
      resolve(current);
    };
  }));
  open();
  const pause = await screen.findByRole("button", { name: "Pause transfers after current work" });
  pause.focus();
  fireEvent.click(pause);
  const saving = await screen.findByRole("button", { name: "Saving transfers change" });
  expect(saving).toHaveFocus();
  expect(saving).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(saving);
  expect(api.transferQueueControl).toHaveBeenCalledTimes(1);
  if (moveFocus) screen.getByRole("combobox", { name: "Work category" }).focus();
  finish?.();
  const resume = await screen.findByRole("button", { name: "Resume transfers" });
  expect(await screen.findByText("Transfers paused. New submissions stay queued.")).toBeInTheDocument();
  if (moveFocus) expect(screen.getByRole("combobox", { name: "Work category" })).toHaveFocus();
  else expect(resume).toHaveFocus();
});

it("shows draining until the fetched durable state becomes paused, then resumes explicitly", async () => {
  current = { ...current, running_jobs: 1 };
  vi.mocked(api.transferQueueControl).mockImplementation(async (action) => {
    current = action === "pause_after_current"
      ? { ...current, dispatch_state: "draining", revision: 1, allowed_actions: ["resume"] }
      : { ...current, dispatch_state: "open", revision: 3, allowed_actions: ["pause_after_current"] };
    return current;
  });
  const client = open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause transfers after current work" }));
  expect(await screen.findByText("Finishing current transfers. New transfers will wait.")).toBeInTheDocument();
  expect(screen.queryByText("Transfers paused. New submissions stay queued.")).not.toBeInTheDocument();
  current = { ...current, dispatch_state: "paused", revision: 2, running_jobs: 0 };
  await act(() => client.invalidateQueries({ queryKey: ["jobs", "queue"] }));
  expect(await screen.findByText("Transfers paused. New submissions stay queued.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Resume transfers" }));
  await screen.findByRole("button", { name: "Pause transfers after current work" });
  expect(api.transferQueueControl).toHaveBeenLastCalledWith("resume", {
    expected_revision: 2, idempotency_key: expect.any(String),
  });
});

it("reuses the same command after an ambiguous network failure", async () => {
  vi.mocked(api.transferQueueControl)
    .mockRejectedValueOnce(new Error("Connection interrupted"))
    .mockImplementationOnce(async () => {
      current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
      return current;
    });
  open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause transfers after current work" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Connection interrupted");
  const original = vi.mocked(api.transferQueueControl).mock.calls[0]?.[1];
  fireEvent.click(screen.getByRole("button", { name: "Pause transfers after current work" }));
  await screen.findByRole("button", { name: "Resume transfers" });
  expect(api.transferQueueControl).toHaveBeenCalledTimes(2);
  expect(vi.mocked(api.transferQueueControl).mock.calls[1]?.[1]).toEqual(original);
});

it("refreshes a stale conflict and uses the new revision for the next chosen action", async () => {
  vi.mocked(api.transferQueueControl)
    .mockImplementationOnce(async () => {
      current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
      throw new ApiError(409, "Transfers state changed", "Transfers state changed", "queue-lane-conflict");
    })
    .mockImplementationOnce(async () => {
      current = { ...current, dispatch_state: "open", revision: 2, allowed_actions: ["pause_after_current"] };
      return current;
    });
  open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause transfers after current work" }));
  await screen.findByRole("button", { name: "Resume transfers" });
  fireEvent.click(screen.getByRole("button", { name: "Resume transfers" }));
  await waitFor(() => expect(api.transferQueueControl).toHaveBeenCalledTimes(2));
  const calls = vi.mocked(api.transferQueueControl).mock.calls;
  expect(calls[1]?.[1].expected_revision).toBe(1);
  expect(calls[1]?.[1].idempotency_key).not.toBe(calls[0]?.[1].idempotency_key);
});

it("polls transfers state when no queue event arrives", async () => {
  vi.useFakeTimers();
  open();
  await act(() => vi.advanceTimersByTimeAsync(50));
  expect(screen.getByRole("button", { name: "Pause transfers after current work" })).toBeInTheDocument();
  current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
  await act(() => vi.advanceTimersByTimeAsync(5_100));
  expect(screen.getByText("Transfers paused. New submissions stay queued.")).toBeInTheDocument();
  expect(api.transferQueueControl).not.toHaveBeenCalled();
});

it("blocks stale controls after a failed read and offers a state retry", async () => {
  const client = open();
  const pause = await screen.findByRole("button", { name: "Pause transfers after current work" });
  vi.mocked(api.transferQueuePolicy).mockRejectedValueOnce(new Error("State unavailable"));
  await act(() => client.invalidateQueries({ queryKey: ["jobs", "queue"] }));
  expect(await screen.findByRole("alert")).toHaveTextContent("State unavailable");
  expect(pause).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(pause);
  expect(api.transferQueueControl).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Retry transfers state" }));
  await waitFor(() => expect(pause).toHaveAttribute("aria-disabled", "false"));
});

it("clears only the conflict resolved by a newer authoritative policy", async () => {
  vi.mocked(api.transferQueueControl).mockRejectedValue(
    new ApiError(409, "Transfers state changed", "Transfers state changed", "queue-lane-conflict"),
  );
  const client = open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause transfers after current work" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Transfers state changed");
  await waitFor(() => expect(api.transferQueuePolicy).toHaveBeenCalledTimes(2));
  expect(screen.getByRole("alert")).toHaveTextContent("Transfers state changed");
  current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
  await act(() => client.invalidateQueries({ queryKey: ["jobs", "queue"] }));
  expect(await screen.findByRole("button", { name: "Resume transfers" })).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(api.transferQueueControl).toHaveBeenCalledTimes(1);
});

it("clears a second-click conflict after the first successful command refreshes", async () => {
  const refreshes: Array<(value: TransferQueuePolicy) => void> = [];
  vi.mocked(api.transferQueuePolicy)
    .mockResolvedValueOnce({ ...current })
    .mockImplementation(() => new Promise((resolve) => { refreshes.push(resolve); }));
  vi.mocked(api.transferQueueControl)
    .mockImplementationOnce(async () => {
      current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
      return current;
    })
    .mockRejectedValueOnce(new ApiError(409, "Transfers state changed",
      "Transfers state changed", "queue-lane-conflict"));
  open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause transfers after current work" }));
  await waitFor(() => expect(refreshes).toHaveLength(1));
  const again = await screen.findByRole("button", { name: "Pause transfers after current work" });
  await waitFor(() => expect(again).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(again);
  expect(await screen.findByRole("alert")).toHaveTextContent("Transfers state changed");
  await waitFor(() => expect(refreshes).toHaveLength(2));
  await act(async () => { refreshes.forEach((resolve) => resolve(current)); });
  expect(await screen.findByRole("button", { name: "Resume transfers" })).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(api.transferQueueControl).toHaveBeenCalledTimes(2);
});

it.each([
  new Error("Connection interrupted"),
  new ApiError(409, "Another refusal", "Another refusal", "another-refusal"),
])("keeps an unrelated failure after a newer policy arrives: %s", async (error) => {
  vi.mocked(api.transferQueueControl).mockRejectedValue(error);
  const client = open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause transfers after current work" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(error.message);
  current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
  await act(() => client.invalidateQueries({ queryKey: ["jobs", "queue"] }));
  expect(await screen.findByRole("button", { name: "Resume transfers" })).toBeInTheDocument();
  expect(screen.getByRole("alert")).toHaveTextContent(error.message);
});
