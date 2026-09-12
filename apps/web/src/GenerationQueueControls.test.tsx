import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { QueueActivityDialog } from "./QueueActivityDialog";
import type { GenerationQueuePolicy } from "./types";

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: {
    queueActivity: vi.fn(), queueControl: vi.fn(), queuePlanSteps: vi.fn(),
    generationQueuePolicy: vi.fn(), generationQueueControl: vi.fn(),
  },
}));

const clients: QueryClient[] = [];
let current: GenerationQueuePolicy;

beforeEach(() => {
  vi.resetAllMocks();
  current = { lane: "generation", dispatch_state: "open", revision: 0,
    running_jobs: 0, allowed_actions: ["pause_after_current"] };
  vi.mocked(api.queueActivity).mockResolvedValue({
    items: [], total: 0, lane_counts: { generation: 0, transfer: 0, install: 0 },
    next_cursor: null, observed_at: "2026-09-01T00:00:00Z",
  });
  vi.mocked(api.generationQueuePolicy).mockImplementation(async () => ({ ...current }));
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

it.each([false, true])("preserves focus and prevents duplicate pause while saving: %s", async (moveFocus) => {
  let finish: (() => void) | undefined;
  vi.mocked(api.generationQueueControl).mockImplementationOnce(() => new Promise((resolve) => {
    finish = () => {
      current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
      resolve(current);
    };
  }));
  open();
  const pause = await screen.findByRole("button", { name: "Pause generation after current work" });
  pause.focus();
  fireEvent.click(pause);
  const saving = await screen.findByRole("button", { name: "Saving generation change" });
  expect(saving).toHaveFocus();
  expect(saving).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(saving);
  expect(api.generationQueueControl).toHaveBeenCalledTimes(1);
  if (moveFocus) screen.getByRole("combobox", { name: "Work category" }).focus();
  finish?.();
  const resume = await screen.findByRole("button", { name: "Resume generation" });
  expect(await screen.findByText("Generation paused. New submissions stay queued.")).toBeInTheDocument();
  if (moveFocus) expect(screen.getByRole("combobox", { name: "Work category" })).toHaveFocus();
  else expect(resume).toHaveFocus();
});

it("shows draining until the fetched durable state becomes paused, then resumes explicitly", async () => {
  current = { ...current, running_jobs: 1 };
  vi.mocked(api.generationQueueControl).mockImplementation(async (action) => {
    current = action === "pause_after_current"
      ? { ...current, dispatch_state: "draining", revision: 1, allowed_actions: ["resume"] }
      : { ...current, dispatch_state: "open", revision: 3, allowed_actions: ["pause_after_current"] };
    return current;
  });
  const client = open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause generation after current work" }));
  expect(await screen.findByText("Finishing current generation. New generation will wait.")).toBeInTheDocument();
  expect(screen.queryByText("Generation paused. New submissions stay queued.")).not.toBeInTheDocument();
  current = { ...current, dispatch_state: "paused", revision: 2, running_jobs: 0 };
  await act(() => client.invalidateQueries({ queryKey: ["jobs", "queue"] }));
  expect(await screen.findByText("Generation paused. New submissions stay queued.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Resume generation" }));
  await screen.findByRole("button", { name: "Pause generation after current work" });
  expect(api.generationQueueControl).toHaveBeenLastCalledWith("resume", {
    expected_revision: 2, idempotency_key: expect.any(String),
  });
});

it("reuses the same command after an ambiguous network failure", async () => {
  vi.mocked(api.generationQueueControl)
    .mockRejectedValueOnce(new Error("Connection interrupted"))
    .mockImplementationOnce(async () => {
      current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
      return current;
    });
  open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause generation after current work" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Connection interrupted");
  const original = vi.mocked(api.generationQueueControl).mock.calls[0]?.[1];
  fireEvent.click(screen.getByRole("button", { name: "Pause generation after current work" }));
  await screen.findByRole("button", { name: "Resume generation" });
  expect(api.generationQueueControl).toHaveBeenCalledTimes(2);
  expect(vi.mocked(api.generationQueueControl).mock.calls[1]?.[1]).toEqual(original);
});

it("refreshes a stale conflict and uses the new revision for the next chosen action", async () => {
  vi.mocked(api.generationQueueControl)
    .mockImplementationOnce(async () => {
      current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
      throw new ApiError(409, "Generation state changed", "Generation state changed", "queue-lane-conflict");
    })
    .mockImplementationOnce(async () => {
      current = { ...current, dispatch_state: "open", revision: 2, allowed_actions: ["pause_after_current"] };
      return current;
    });
  open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause generation after current work" }));
  await screen.findByRole("button", { name: "Resume generation" });
  fireEvent.click(screen.getByRole("button", { name: "Resume generation" }));
  await waitFor(() => expect(api.generationQueueControl).toHaveBeenCalledTimes(2));
  const calls = vi.mocked(api.generationQueueControl).mock.calls;
  expect(calls[1]?.[1].expected_revision).toBe(1);
  expect(calls[1]?.[1].idempotency_key).not.toBe(calls[0]?.[1].idempotency_key);
});

it("polls generation state when no queue event arrives", async () => {
  vi.useFakeTimers();
  open();
  await act(() => vi.advanceTimersByTimeAsync(50));
  expect(screen.getByRole("button", { name: "Pause generation after current work" })).toBeInTheDocument();
  current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
  await act(() => vi.advanceTimersByTimeAsync(5_100));
  expect(screen.getByText("Generation paused. New submissions stay queued.")).toBeInTheDocument();
  expect(api.generationQueueControl).not.toHaveBeenCalled();
});

it("blocks stale controls after a failed read and offers a state retry", async () => {
  const client = open();
  const pause = await screen.findByRole("button", { name: "Pause generation after current work" });
  vi.mocked(api.generationQueuePolicy).mockRejectedValueOnce(new Error("State unavailable"));
  await act(() => client.invalidateQueries({ queryKey: ["jobs", "queue"] }));
  expect(await screen.findByRole("alert")).toHaveTextContent("State unavailable");
  expect(pause).toHaveAttribute("aria-disabled", "true");
  fireEvent.click(pause);
  expect(api.generationQueueControl).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Retry generation state" }));
  await waitFor(() => expect(pause).toHaveAttribute("aria-disabled", "false"));
});

it("clears only the conflict resolved by a newer authoritative policy", async () => {
  vi.mocked(api.generationQueueControl).mockRejectedValue(
    new ApiError(409, "Generation state changed", "Generation state changed", "queue-lane-conflict"),
  );
  const client = open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause generation after current work" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Generation state changed");
  await waitFor(() => expect(api.generationQueuePolicy).toHaveBeenCalledTimes(2));
  expect(screen.getByRole("alert")).toHaveTextContent("Generation state changed");
  current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
  await act(() => client.invalidateQueries({ queryKey: ["jobs", "queue"] }));
  expect(await screen.findByRole("button", { name: "Resume generation" })).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(api.generationQueueControl).toHaveBeenCalledTimes(1);
});

it("clears a second-click conflict after the first successful command refreshes", async () => {
  const refreshes: Array<(value: GenerationQueuePolicy) => void> = [];
  vi.mocked(api.generationQueuePolicy)
    .mockResolvedValueOnce({ ...current })
    .mockImplementation(() => new Promise((resolve) => { refreshes.push(resolve); }));
  vi.mocked(api.generationQueueControl)
    .mockImplementationOnce(async () => {
      current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
      return current;
    })
    .mockRejectedValueOnce(new ApiError(409, "Generation state changed",
      "Generation state changed", "queue-lane-conflict"));
  open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause generation after current work" }));
  await waitFor(() => expect(refreshes).toHaveLength(1));
  const again = await screen.findByRole("button", { name: "Pause generation after current work" });
  await waitFor(() => expect(again).toHaveAttribute("aria-disabled", "false"));
  fireEvent.click(again);
  expect(await screen.findByRole("alert")).toHaveTextContent("Generation state changed");
  await waitFor(() => expect(refreshes).toHaveLength(2));
  await act(async () => { refreshes.forEach((resolve) => resolve(current)); });
  expect(await screen.findByRole("button", { name: "Resume generation" })).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(api.generationQueueControl).toHaveBeenCalledTimes(2);
});

it.each([
  new Error("Connection interrupted"),
  new ApiError(409, "Another refusal", "Another refusal", "another-refusal"),
])("keeps an unrelated failure after a newer policy arrives: %s", async (error) => {
  vi.mocked(api.generationQueueControl).mockRejectedValue(error);
  const client = open();
  fireEvent.click(await screen.findByRole("button", { name: "Pause generation after current work" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(error.message);
  current = { ...current, dispatch_state: "paused", revision: 1, allowed_actions: ["resume"] };
  await act(() => client.invalidateQueries({ queryKey: ["jobs", "queue"] }));
  expect(await screen.findByRole("button", { name: "Resume generation" })).toBeInTheDocument();
  expect(screen.getByRole("alert")).toHaveTextContent(error.message);
});
