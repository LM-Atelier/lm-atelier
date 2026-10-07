import type { QueueControlCommand } from "./types";

export type QueueLane = "generation" | "transfer" | "install" | "utility";
export interface QueueOrderOwner {
  type: "work_plan" | "job";
  id: string;
}
export interface QueueOrderNeighbours {
  before: QueueOrderOwner | null;
  after: QueueOrderOwner | null;
}
export interface QueueOrderCommand extends QueueControlCommand {
  cohort_id: string;
  owner: QueueOrderOwner;
  before?: QueueOrderOwner | null;
  after?: QueueOrderOwner | null;
  expected_item_neighbors: QueueOrderNeighbours;
  expected_anchor_neighbors: QueueOrderNeighbours;
}
export interface QueueOrderItem {
  owner: QueueOrderOwner;
  label: string;
  queued_at: string;
  priority: number | null;
  cohort_id: string | null;
  position: number | null;
  cohort_length: number;
  neighbors: QueueOrderNeighbours;
  before_neighbors?: QueueOrderNeighbours | null;
  after_neighbors?: QueueOrderNeighbours | null;
  unavailable_reason: "lane-busy" | "held" | "blocked" | "mixed-resources" | "unsupported" | null;
}
export interface QueueOrderPage {
  lane: QueueLane;
  revision: number;
  items: QueueOrderItem[];
  total: number;
  next_cursor: string | null;
}
export interface QueueOrderResult {
  lane: QueueLane;
  revision: number;
  owner: QueueOrderOwner;
}
