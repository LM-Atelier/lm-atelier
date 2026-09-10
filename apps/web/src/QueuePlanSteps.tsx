import { useId, useState } from "react";
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import "./QueueActivityDialog.css";

function QueuePlanStepList({ planId, name, regionId }: { planId: string; name: string; regionId: string }) {
  const client = useQueryClient();
  const queryKey = ["jobs", "queue-steps", planId] as const;
  const steps = useInfiniteQuery({
    queryKey, initialPageParam: 0,
    queryFn: ({ pageParam, signal }) => api.queuePlanSteps(planId, pageParam, signal),
    getNextPageParam: (page) => page.next_offset ?? undefined,
    refetchInterval: 5_000,
  });
  const first = steps.data?.pages[0];
  const items = [...new Map((steps.data?.pages.flatMap(page => page.items) ?? [])
    .map(step => [step.id, step])).values()];
  return (
    <section id={regionId} aria-label={"Steps for " + name} className="queue-plan-step-details">
      {steps.isPending && <p role="status">Loading step details…</p>}
      {steps.error && <div role="alert" className="queue-activity-error">
        <p>{steps.error.message}{first && " Last loaded steps are shown below."}</p>
        <button className="secondary compact-button" disabled={steps.isFetching}
          onClick={() => void client.resetQueries({ queryKey, exact: true })}>Retry step details</button>
      </div>}
      {first && <>
        <p role="status">Showing {items.length} of {first.total} {first.total === 1 ? "step" : "steps"}</p>
        {!items.length && !steps.error && <p>No steps recorded for this plan.</p>}
        <ol className="queue-step-list">
          {items.map(step => <li key={step.id}>
            <strong>Step {step.ordinal + 1} · {step.label}</strong>
            <span className="queue-activity-status">{step.status}</span>
            {step.blocked_by > 0 && <small>
              {step.blocked_by} {step.blocked_by === 1 ? "prerequisite" : "prerequisites"} unfinished
            </small>}
          </li>)}
        </ol>
        {steps.hasNextPage && <button className="secondary compact-button" disabled={steps.isFetching}
          onClick={() => void steps.fetchNextPage()}>Load more steps</button>}
      </>}
    </section>
  );
}

export function QueuePlanSteps({ planId, name }: { planId: string; name: string }) {
  const [expanded, setExpanded] = useState(false);
  const regionId = useId();
  return (
    <div className="queue-plan-steps">
      <button className="secondary compact-button" aria-expanded={expanded} aria-controls={regionId}
        aria-label={(expanded ? "Hide steps for " : "Show steps for ") + name}
        onClick={() => setExpanded(value => !value)}>{expanded ? "Hide steps" : "Show steps"}</button>
      {expanded && <QueuePlanStepList planId={planId} name={name} regionId={regionId} />}
    </div>
  );
}
