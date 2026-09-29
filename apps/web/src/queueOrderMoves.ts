import type { QueueOrderCommand, QueueOrderItem, QueueOrderPage } from "./queueOrderTypes";

export const QUEUE_ORDER_REASONS = {
  "lane-busy": "Current work must finish before this category can be reordered. Pause after current work to keep it idle.",
  held: "Release this work before changing its order.",
  blocked: "This work is waiting for its prerequisites or cannot start yet.",
  "mixed-resources": "This plan needs different resources and cannot move as one item.",
  unsupported: "This work does not support manual ordering.",
};

export function orderOwnerKey(item: QueueOrderItem): string {
  return item.owner.type + ":" + item.owner.id;
}

export function adjacentMove(
  page: QueueOrderPage,
  item: QueueOrderItem,
  direction: "before" | "after",
  idempotencyKey: string,
): QueueOrderCommand | null {
  const anchor = item.neighbors[direction];
  const neighbors = direction === "before" ? item.before_neighbors : item.after_neighbors;
  if (!item.cohort_id || item.unavailable_reason || !anchor || !neighbors) return null;
  return {
    expected_revision: page.revision,
    idempotency_key: idempotencyKey,
    cohort_id: item.cohort_id,
    owner: item.owner,
    before: direction === "before" ? anchor : null,
    after: direction === "after" ? anchor : null,
    expected_item_neighbors: item.neighbors,
    expected_anchor_neighbors: neighbors,
  };
}

export function droppedMove(
  page: QueueOrderPage,
  item: QueueOrderItem,
  anchor: QueueOrderItem,
  direction: "before" | "after",
  idempotencyKey: string,
): QueueOrderCommand | null {
  if (!item.cohort_id || item.unavailable_reason || anchor.unavailable_reason
    || item.cohort_id !== anchor.cohort_id || orderOwnerKey(item) === orderOwnerKey(anchor)) return null;
  return {
    expected_revision: page.revision,
    idempotency_key: idempotencyKey,
    cohort_id: item.cohort_id,
    owner: item.owner,
    before: direction === "before" ? anchor.owner : null,
    after: direction === "after" ? anchor.owner : null,
    expected_item_neighbors: item.neighbors,
    expected_anchor_neighbors: anchor.neighbors,
  };
}
