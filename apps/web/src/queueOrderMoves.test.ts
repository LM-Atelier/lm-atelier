import { expect, it } from "vitest";
import { adjacentMove, droppedMove } from "./queueOrderMoves";
import type { QueueOrderItem, QueueOrderPage } from "./queueOrderTypes";

const owners = ["a", "b", "c"].map((id) => ({ type: "job" as const, id }));
const neighbors = owners.map((_, index) => ({ before: owners[index - 1] ?? null, after: owners[index + 1] ?? null }));
const items: QueueOrderItem[] = owners.map((owner, index) => ({
  owner, label: "Download", queued_at: "2026-09-01T00:00:00Z", priority: 0, cohort_id: "a".repeat(64), position: index + 1, cohort_length: 3,
  neighbors: neighbors[index], before_neighbors: neighbors[index - 1] ?? null,
  after_neighbors: neighbors[index + 1] ?? null, unavailable_reason: null,
}));
const page: QueueOrderPage = { lane: "transfer", revision: 7, items, total: 3, next_cursor: null };

it("moves across a page boundary using the server's complete neighbour comparison", () => {
  const firstPage = { ...page, items: [items[0]], next_cursor: "next" };
  expect(adjacentMove(firstPage, items[0], "after", "once")).toEqual({
    expected_revision: 7, idempotency_key: "once", cohort_id: "a".repeat(64),
    owner: owners[0], before: null, after: owners[1],
    expected_item_neighbors: neighbors[0], expected_anchor_neighbors: neighbors[1],
  });
  expect(adjacentMove(page, items[0], "before", "once")).toBeNull();
  expect(adjacentMove(page, items[2], "after", "once")).toBeNull();
});

it.each(["before", "after"] as const)("expresses a drop %s one anchor without submitting the whole list", (direction) => {
  expect(droppedMove(page, items[2], items[0], direction, "once")).toEqual({
    expected_revision: 7, idempotency_key: "once", cohort_id: "a".repeat(64),
    owner: owners[2], before: direction === "before" ? owners[0] : null,
    after: direction === "after" ? owners[0] : null,
    expected_item_neighbors: neighbors[2], expected_anchor_neighbors: neighbors[0],
  });
});

it.each(["lane-busy", "held", "blocked", "mixed-resources", "unsupported"] as const)(
  "never proposes a move for %s work", (unavailable_reason) => {
    const refused = { ...items[1], unavailable_reason };
    expect(adjacentMove(page, refused, "before", "once")).toBeNull();
    expect(droppedMove(page, refused, items[0], "before", "once")).toBeNull();
    expect(droppedMove(page, items[0], refused, "before", "once")).toBeNull();
  },
);

it("refuses a cross-cohort or self drop", () => {
  expect(droppedMove(page, items[1], { ...items[0], cohort_id: "b".repeat(64) }, "before", "once")).toBeNull();
  expect(droppedMove(page, items[0], items[0], "after", "once")).toBeNull();
});
