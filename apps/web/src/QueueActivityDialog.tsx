import { useState } from "react";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { api } from "./api";
import { QueuePlanControls } from "./QueuePlanControls";
import { QueuePlanSteps } from "./QueuePlanSteps";
import type { QueueActivityItem } from "./types";
import "./QueueActivityDialog.css";

type Lane = QueueActivityItem["lane"];

export function QueueActivityDialog({ onClose }: { onClose: () => void }) {
  const client = useQueryClient();
  const [lane, setLane] = useState<Lane | "all">("all");
  const queryKey = ["jobs", "queue", lane] as const;
  const activity = useInfiniteQuery({
    queryKey,
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) => api.queueActivity({
      lane: lane === "all" ? undefined : lane, cursor: pageParam, limit: 50,
    }, signal),
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    refetchInterval: 5_000,
  });
  const first = activity.data?.pages[0];
  const items = [...new Map((activity.data?.pages.flatMap((page) => page.items) ?? [])
    .map((item) => [item.owner_type + ":" + item.owner_id, item])).values()];
  const refresh = () => {
    if (activity.isFetching) return;
    void client.resetQueries({ queryKey, exact: true });
  };
  return (
    <AccessibleDialog title="Accepted work" eyebrow="Queue activity"
      closeLabel="Close accepted work" onClose={onClose} className="queue-activity-dialog">
      <p className="queue-activity-intro">
        Submitted plans stay grouped; transfers and installs appear separately.
        Ordered by acceptance time. Different resources can run at the same time.
      </p>
      <div className="queue-activity-toolbar">
        <label>Work category
          <select value={lane} onChange={(event) => setLane(event.target.value as Lane | "all")}>
            <option value="all">All work</option>
            <option value="generation">Generation</option>
            <option value="transfer">Transfers</option>
            <option value="install">Installs</option>
          </select>
        </label>
        <button className="secondary compact-button" aria-disabled={activity.isFetching}
          aria-busy={activity.isFetching} onClick={refresh}
          aria-label={activity.isFetching ? "Refreshing accepted work" : "Refresh accepted work"}>
          {activity.isFetching ? "Refreshing…" : "Refresh"}
        </button>
      </div>
      {activity.error && (
        <div role="alert" className="queue-activity-error">
          {activity.error.message}
          {first && <span> Last loaded items are shown below.</span>}
        </div>
      )}
      {activity.isPending && <p role="status">Loading accepted work…</p>}
      {first && (
        <>
          <p className="queue-activity-counts">
            {first.lane_counts.generation} generation · {first.lane_counts.transfer} transfer ·{" "}
            {first.lane_counts.install} install
          </p>
          <p role="status">Showing {items.length} of {first.total} active items</p>
          {!items.length && !activity.error && <p>No active accepted work in this category.</p>}
          <ol className="queue-activity-items" aria-label="Active accepted work">
            {items.map((item) => (
              <li key={item.owner_type + ":" + item.owner_id}>
                <div className="queue-activity-item-heading">
                  <strong>{item.chat_title || item.label}</strong>
                  <span className="queue-activity-status">{item.status}</span>
                </div>
                {item.chat_title && <span>{item.label}</span>}
                <small>Accepted {new Date(item.created_at).toLocaleString()}</small>
                {item.owner_type === "work_plan" ? (
                  <>
                    <span>{item.completed_steps} of {item.step_count} steps complete</span>
                    <span>{item.running_jobs} running · {item.queued_jobs} queued · {item.paused_jobs} paused jobs</span>
                    {item.blocked_steps > 0 && <span>
                      {item.blocked_steps} {item.blocked_steps === 1 ? "step" : "steps"} waiting for prerequisites
                    </span>}
                    <QueuePlanControls item={item} />
                    <QueuePlanSteps planId={item.owner_id} name={item.chat_title || item.label} />
                  </>
                ) : item.progress !== null ? (
                  <div className="progress-track" role="progressbar" aria-label={item.label + " progress"}
                    aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(item.progress * 100)}>
                    <div style={{ width: String(item.progress * 100) + "%" }} />
                  </div>
                ) : null}
              </li>
            ))}
          </ol>
          {activity.hasNextPage && (
            <button className="secondary" disabled={activity.isFetching}
              onClick={() => void activity.fetchNextPage()}>
              Load more accepted work
            </button>
          )}
          <small className="queue-activity-updated">
            Last checked {new Date(first.observed_at).toLocaleTimeString()}. Updates every five seconds.
          </small>
        </>
      )}
    </AccessibleDialog>
  );
}
