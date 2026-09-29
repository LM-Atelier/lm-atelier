import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { act, cleanup, createEvent, fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import { QueueActivityDialog } from "./QueueActivityDialog";
import { QueueOrderPanel } from "./QueueOrderPanel";
import type { QueueOrderPage } from "./queueOrderTypes";

vi.mock("./api", async (original) => ({
  ...(await original<typeof import("./api")>()),
  api: { queueOrder: vi.fn(), reorderQueue: vi.fn(), queueActivity: vi.fn(),
    generationQueuePolicy: vi.fn(), transferQueuePolicy: vi.fn(), installQueuePolicy: vi.fn() },
}));
const clients: QueryClient[] = [];
let page: QueueOrderPage;
const owners = ["a", "b", "c"].map((id) => ({ type: "job" as const, id }));
const neighbors = owners.map((_, i) => ({ before: owners[i - 1] ?? null, after: owners[i + 1] ?? null }));
const advance = () => act(() => vi.advanceTimersByTimeAsync(50));

beforeEach(() => {
  vi.resetAllMocks();
  vi.useFakeTimers();
  page = { lane: "transfer", revision: 7, total: 3, next_cursor: null,
    items: owners.map((owner, i) => ({
      owner, label: ["First transfer", "Second transfer", "Third transfer"][i],
      queued_at: "2026-09-01T00:00:00Z", priority: 0, cohort_id: "a".repeat(64),
      position: i + 1, cohort_length: 3, neighbors: neighbors[i],
      before_neighbors: neighbors[i - 1] ?? null, after_neighbors: neighbors[i + 1] ?? null,
      unavailable_reason: null,
    })),
  };
  vi.mocked(api.queueOrder).mockImplementation(async () => structuredClone(page));
  vi.mocked(api.reorderQueue).mockImplementation(async (lane, command) => {
    const moved = page.items.find((item) => item.owner.id === command.owner.id)!;
    const remaining = page.items.filter((item) => item !== moved);
    const anchor = command.before ?? command.after;
    const index = remaining.findIndex((item) => item.owner.id === anchor?.id);
    remaining.splice(index + (command.after ? 1 : 0), 0, moved);
    const updatedNeighbors = remaining.map((_, i) => ({ before: remaining[i - 1]?.owner ?? null,
      after: remaining[i + 1]?.owner ?? null }));
    page = { ...page, revision: page.revision + 1, items: remaining.map((item, i) => ({
      ...item, position: i + 1, neighbors: updatedNeighbors[i],
      before_neighbors: updatedNeighbors[i - 1] ?? null, after_neighbors: updatedNeighbors[i + 1] ?? null,
    })) };
    return { lane, revision: page.revision, owner: command.owner };
  });
  vi.mocked(api.queueActivity).mockResolvedValue({ items: [], total: 0, next_cursor: null,
    lane_counts: { generation: 0, transfer: 0, install: 0 }, observed_at: "2026-09-01T00:00:00Z" });
  vi.mocked(api.generationQueuePolicy).mockResolvedValue({ lane: "generation", revision: 0,
    dispatch_state: "open", running_jobs: 0, allowed_actions: ["pause_after_current"] });
  vi.mocked(api.transferQueuePolicy).mockResolvedValue({ lane: "transfer", revision: 0,
    dispatch_state: "open", running_jobs: 0, allowed_actions: ["pause_after_current"] });
  vi.mocked(api.installQueuePolicy).mockResolvedValue({ lane: "install", revision: 0,
    dispatch_state: "open", running_jobs: 0, allowed_actions: ["pause_after_current"] });
});
afterEach(() => {
  cleanup();
  clients.splice(0).forEach((client) => client.clear());
  vi.useRealTimers();
});
function open(dialog = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  clients.push(client);
  render(<QueryClientProvider client={client}>{dialog
    ? <QueueActivityDialog onClose={() => undefined} />
    : <QueueOrderPanel initialLane="transfer" onBack={() => undefined} />}</QueryClientProvider>);
  return client;
}
function row(name: string) { return within(screen.getByRole("listitem", { name })); }

it("opens dispatch ordering separately from acceptance-time activity", async () => {
  open(true);
  await advance();
  expect(screen.getByText(/Ordered by acceptance time/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Change dispatch order" }));
  await advance();
  expect(screen.getByRole("region", { name: "Dispatch order" })).toBeInTheDocument();
  expect(screen.queryByText(/Ordered by acceptance time/)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Back to accepted work" }));
  expect(screen.getByText(/Ordered by acceptance time/)).toBeInTheDocument();
});

it("sends one relative move and retains button focus while the server supplies the new order", async () => {
  open();
  await advance();
  const button = row("Third transfer").getByRole("button", { name: "Move earlier" });
  button.focus();
  fireEvent.click(button);
  fireEvent.click(button);
  expect(button).toHaveFocus();
  expect(button).toHaveAttribute("aria-disabled", "true");
  await advance();
  expect(api.reorderQueue).toHaveBeenCalledTimes(1);
  expect(api.reorderQueue).toHaveBeenCalledWith("transfer", {
    expected_revision: 7, idempotency_key: expect.any(String), cohort_id: "a".repeat(64),
    owner: owners[2], before: owners[1], after: null,
    expected_item_neighbors: neighbors[2], expected_anchor_neighbors: neighbors[1],
  });
  expect(row("Third transfer").getByRole("button", { name: "Move earlier" })).toHaveFocus();
  expect(screen.getByText(/Order saved/)).toBeInTheDocument();
  expect(screen.getAllByRole("listitem")[1]).toHaveAccessibleName("Third transfer");
});

it.each(["accepted", "stale"] as const)("submits a move during a background read and handles the %s result", async (result) => {
  const client = open();
  await advance();
  let finishRead: ((value: QueueOrderPage) => void) | undefined;
  vi.mocked(api.queueOrder).mockImplementationOnce(() => new Promise((resolve) => { finishRead = resolve; }));
  if (result === "stale") {
    vi.mocked(api.reorderQueue).mockRejectedValueOnce(new ApiError(409, "changed", "The queue changed", "queue-order-conflict"));
  }
  let refresh = Promise.resolve();
  await act(async () => { refresh = client.invalidateQueries({ queryKey: ["jobs", "queue"] }); });
  await advance();
  expect(client.isFetching()).toBe(1);
  try {
    fireEvent.click(row("Third transfer").getByRole("button", { name: "Move earlier" }));
    await advance();
    expect(api.reorderQueue).toHaveBeenCalledTimes(1);
    expect(api.reorderQueue).toHaveBeenCalledWith("transfer", expect.objectContaining({
      expected_revision: 7, cohort_id: "a".repeat(64),
      expected_item_neighbors: neighbors[2], expected_anchor_neighbors: neighbors[1],
    }));
    expect(screen.getByText(result === "accepted" ? /Order saved/ : /Review its latest order/)).toBeInTheDocument();
    if (result === "stale") expect(screen.queryByText(/Order saved/)).not.toBeInTheDocument();
  } finally {
    await act(async () => { finishRead?.(structuredClone(page)); await refresh; });
  }
});

it("retries an uncertain network result with the exact command and key", async () => {
  vi.mocked(api.reorderQueue).mockRejectedValueOnce(new Error("Connection interrupted"));
  open();
  await advance();
  fireEvent.click(row("Third transfer").getByRole("button", { name: "Move earlier" }));
  await advance();
  expect(screen.getByRole("alert")).toHaveTextContent("Connection interrupted");
  const command = vi.mocked(api.reorderQueue).mock.calls[0][1];
  fireEvent.click(row("First transfer").getByRole("button", { name: "Move later" }));
  expect(api.reorderQueue).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Retry the same move" }));
  await advance();
  expect(api.reorderQueue).toHaveBeenLastCalledWith("transfer", command);
  expect(api.reorderQueue).toHaveBeenCalledTimes(2);
});

it.each([false, true])("keeps focus after retrying unchanged queue data (focus moved: %s)", async (movedFocus) => {
  vi.mocked(api.reorderQueue)
    .mockRejectedValueOnce(new Error("Connection interrupted"))
    .mockResolvedValueOnce({ lane: "transfer", revision: page.revision, owner: owners[2] });
  open();
  await advance();
  fireEvent.click(row("Third transfer").getByRole("button", { name: "Move earlier" }));
  await advance();
  const command = vi.mocked(api.reorderQueue).mock.calls[0][1];
  const retry = screen.getByRole("button", { name: "Retry the same move" });
  retry.focus();
  fireEvent.click(retry);
  const category = screen.getByRole("combobox", { name: "Order category" });
  if (movedFocus) category.focus();
  await advance();
  expect(api.reorderQueue).toHaveBeenLastCalledWith("transfer", command);
  expect(screen.queryByRole("button", { name: "Retry the same move" })).not.toBeInTheDocument();
  expect(screen.getByText(/Order saved/)).toBeInTheDocument();
  expect(movedFocus ? category : screen.getByRole("button", { name: "Refresh dispatch order" })).toHaveFocus();
});

it("refreshes a conflict without claiming an optimistic move succeeded", async () => {
  vi.mocked(api.reorderQueue).mockImplementation(async () => {
    page = { ...page, revision: 8, items: page.items.map((item) => ({ ...item, unavailable_reason: "lane-busy" })) };
    throw new ApiError(409, "changed", "The queue changed", "queue-order-conflict");
  });
  open();
  await advance();
  row("Third transfer").getByRole("button", { name: "Move earlier" }).focus();
  fireEvent.click(row("Third transfer").getByRole("button", { name: "Move earlier" }));
  await advance();
  expect(screen.getByText(/Review its latest order/)).toBeInTheDocument();
  expect(screen.queryByText(/Order saved/)).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Move earlier" })).not.toBeInTheDocument();
  expect(screen.getAllByRole("listitem")[0]).toHaveAccessibleName("First transfer");
  expect(screen.getByRole("button", { name: "Refresh dispatch order" })).toHaveFocus();
});

it("returns focus to refresh when an event removes the focused item", async () => {
  const client = open();
  await advance();
  row("Third transfer").getByRole("button", { name: "Move earlier" }).focus();
  page = { ...page, revision: 8, items: page.items.slice(0, 2), total: 2 };
  await act(async () => { await client.invalidateQueries({ queryKey: ["jobs", "queue"] }); });
  await advance();
  expect(screen.queryByRole("listitem", { name: "Third transfer" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Refresh dispatch order" })).toHaveFocus();
});

it("does not reclaim focus after the user moves outside the changing list", async () => {
  const client = open();
  await advance();
  row("Third transfer").getByRole("button", { name: "Move earlier" }).focus();
  const category = screen.getByRole("combobox", { name: "Order category" });
  category.focus();
  page = { ...page, revision: 8, items: page.items.slice(0, 2), total: 2 };
  await act(async () => { await client.invalidateQueries({ queryKey: ["jobs", "queue"] }); });
  await advance();
  expect(category).toHaveFocus();
});

it("provides the exact unavailability reason and no move controls for held work", async () => {
  page.items = [{ ...page.items[0], unavailable_reason: "held", cohort_id: null, position: null }];
  open();
  await advance();
  expect(screen.getByText("Release this work before changing its order.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Move later" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Drag/ })).not.toBeInTheDocument();
});

it("moves beyond a cursor page using the neighbours supplied with that item", async () => {
  page = { ...page, items: [page.items[0]], next_cursor: "next-page" };
  vi.mocked(api.reorderQueue).mockResolvedValue({ lane: "transfer", revision: 8, owner: owners[0] });
  open();
  await advance();
  fireEvent.click(row("First transfer").getByRole("button", { name: "Move later" }));
  await advance();
  expect(api.reorderQueue).toHaveBeenCalledWith("transfer", expect.objectContaining({
    after: owners[1], expected_anchor_neighbors: neighbors[1],
  }));
});

it("requires a fresh page after a stale cursor rather than making stale rows actionable", async () => {
  page.next_cursor = "old-page";
  vi.mocked(api.queueOrder).mockImplementation(async (_lane, options) => {
    if (options.cursor) throw new ApiError(409, "stale", "Page changed", "queue-order-conflict");
    return structuredClone(page);
  });
  open();
  await advance();
  fireEvent.click(screen.getByRole("button", { name: "Next order page" }));
  await advance();
  expect(screen.getByRole("alert")).toHaveTextContent("Refresh dispatch order");
  expect(screen.queryByRole("button", { name: "Move later" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Refresh dispatch order" }));
  await advance();
  expect(row("First transfer").getByRole("button", { name: "Move later" })).toHaveAttribute("aria-disabled", "false");
});

it("explains an oversized queue without suggesting that its page merely changed", async () => {
  vi.mocked(api.queueOrder).mockRejectedValue(new ApiError(409, "too large",
    "Manual ordering supports up to 10,000 unfinished jobs in a category.", "queue-order-limit-exceeded"));
  open();
  await advance();
  expect(screen.getByRole("alert")).toHaveTextContent("Manual ordering supports up to 10,000 unfinished jobs");
  expect(screen.queryByText("This page changed. Refresh dispatch order to load it again.")).not.toBeInTheDocument();
});

it("explains a queue that grows past the ordering limit before a move", async () => {
  const error = new ApiError(409, "too large",
    "Manual ordering supports up to 10,000 unfinished jobs in a category.", "queue-order-limit-exceeded");
  vi.mocked(api.reorderQueue).mockImplementation(async () => {
    vi.mocked(api.queueOrder).mockRejectedValue(error);
    throw error;
  });
  open();
  await advance();
  fireEvent.click(row("Third transfer").getByRole("button", { name: "Move earlier" }));
  await advance();
  expect(screen.getByRole("status")).toHaveTextContent("Manual ordering supports up to 10,000 unfinished jobs");
  expect(screen.getByRole("alert")).toHaveTextContent("Manual ordering supports up to 10,000 unfinished jobs");
  expect(screen.queryByRole("button", { name: "Retry the same move" })).not.toBeInTheDocument();
  expect(screen.queryByText(/Order saved/)).not.toBeInTheDocument();
});

it("drops relative to a visible anchor and sends no browser-supplied full ordering", async () => {
  open();
  await advance();
  const dataTransfer = { setData: vi.fn(), effectAllowed: "", dropEffect: "" };
  fireEvent.dragStart(row("Third transfer").getByRole("button", { name: "Drag Third transfer" }), { dataTransfer });
  const target = screen.getByRole("listitem", { name: "First transfer" });
  const drop = createEvent.drop(target, { dataTransfer });
  Object.defineProperty(drop, "clientY", { value: -1 });
  fireEvent(target, drop);
  await advance();
  expect(api.reorderQueue).toHaveBeenCalledWith("transfer", expect.objectContaining({
    owner: owners[2], before: owners[0], after: null, expected_anchor_neighbors: neighbors[0],
  }));
});
