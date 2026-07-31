import { useEffect, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import type { QueryClient } from "@tanstack/react-query";
import { connectEvents } from "./api";
import type { AppEvent, Job, WorkPlan } from "./types";

const AUTHORITATIVE_QUERY_ROOTS = new Set([
  "about",
  "artifact-storage",
  "artifacts",
  "backups",
  "catalog",
  "chats",
  "chat",
  "credential",
  "custom-nodes",
  "engines",
  "jobs",
  "models",
  "model-storage",
  "presets",
  "profiles",
  "projects",
  "recipes",
  "runtimes",
  "setup-readiness",
  "system",
  "workers",
  "workflow-catalog-models",
  "workflows",
]);

function aggregateWorkPlanStatus(steps: WorkPlan["steps"]): string {
  const statuses = steps.map((step) => step.status);
  if (statuses.length === 0) return "queued";
  for (const active of ["running", "queued", "paused", "blocked"]) {
    if (statuses.includes(active)) return active;
  }
  if (statuses.every((status) => status === "complete")) return "complete";
  return new Set(statuses).size === 1 ? statuses[0] ?? "queued" : "partial";
}

/**
 * Keep the cache in step with the server's event stream, and report the link.
 *
 * The connection flag used to be thrown away - `connectEvents` was handed
 * `() => undefined` - while `refetchOnWindowFocus` is globally off. A dead
 * socket therefore left every view showing stale data with nothing to say so.
 */
export function useLiveEvents(
  client: QueryClient,
  setLiveText: Dispatch<SetStateAction<Record<string, string>>>,
): boolean {
  const [connected, setConnected] = useState(true);
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let mediaRefresh: number | undefined;
    let authoritativeRefresh: number | undefined;
    const scheduleMediaRefresh = () => {
      if (mediaRefresh !== undefined) return;
      mediaRefresh = window.setTimeout(() => {
        mediaRefresh = undefined;
        void client.invalidateQueries({ queryKey: ["chat"] });
      }, 100);
    };
    const scheduleAuthoritativeRefresh = () => {
      if (authoritativeRefresh !== undefined) return;
      authoritativeRefresh = window.setTimeout(() => {
        authoritativeRefresh = undefined;
        void client.invalidateQueries({
          predicate: (query) =>
            AUTHORITATIVE_QUERY_ROOTS.has(String(query.queryKey[0] ?? "")),
        });
      }, 100);
    };
    void connectEvents(
      (event: AppEvent) => {
        if (event.type === "events.replay_gap") {
          scheduleAuthoritativeRefresh();
          return;
        }
        if (event.type === "text.delta") {
          const messageId = String(event.payload.assistant_message_id ?? "");
          const text = String(event.payload.text ?? "");
          if (messageId) setLiveText((current) => ({ ...current, [messageId]: `${current[messageId] ?? ""}${text}` }));
          return;
        }
        if (event.type === "job.progress") {
          const snapshot = event.payload.job as Job | undefined;
          if (snapshot?.id) {
            client.setQueryData<Job[]>(["jobs"], (current) => {
              if (!current) return [snapshot];
              const index = current.findIndex((job) => job.id === snapshot.id);
              if (index < 0) return [snapshot, ...current];
              return current.map((job) => job.id === snapshot.id ? snapshot : job);
            });
            if (snapshot.work_plan_id) {
              client.setQueriesData<WorkPlan[]>(
                { queryKey: ["work-plans"] },
                (current) => current?.map((plan) => {
                  if (plan.id !== snapshot.work_plan_id) return plan;
                  const stepStatus = snapshot.progress_json?.stage
                    ?.startsWith("blocked by")
                    ? "blocked"
                    : snapshot.status;
                  const steps = plan.steps.map((step) => (
                    step.id === snapshot.work_step_id
                      ? {
                          ...step,
                          status: stepStatus,
                          error: stepStatus === "blocked"
                            ? "Waiting for required work to be retried."
                            : snapshot.error,
                        }
                      : step
                  ));
                  const statusCounts = steps.reduce<Record<string, number>>(
                    (counts, step) => ({
                      ...counts,
                      [step.status]: (counts[step.status] ?? 0) + 1,
                    }),
                    {},
                  );
                  return {
                    ...plan,
                    status: aggregateWorkPlanStatus(steps),
                    summary_json: {
                      ...plan.summary_json,
                      status_counts: statusCounts,
                    },
                    steps,
                  };
                }),
              );
            }
          }
          return;
        }
        if (event.type === "work_plan.created") {
          void client.invalidateQueries({ queryKey: ["work-plans"] });
        }
        if (event.type.includes("progress") || event.type.startsWith("download.")) void client.invalidateQueries({ queryKey: ["jobs"] });
        if (event.type.startsWith("download.") || event.type.startsWith("worker.") || event.type.startsWith("runtime.") || event.type.startsWith("setup.verification")) void client.invalidateQueries({ queryKey: ["setup-readiness"] });
        if (event.type === "run.progress") void client.invalidateQueries({ queryKey: ["chat"] });
        if (event.type === "download.completed") {
          void client.invalidateQueries({ queryKey: ["models"] });
          void client.invalidateQueries({ queryKey: ["profiles"] });
          void client.invalidateQueries({ queryKey: ["model-storage"] });
        }
        if (["generation.progress", "generation.preview"].includes(event.type)) {
          scheduleMediaRefresh();
        }
        if (["run.completed", "run.failed", "run.cancelled"].includes(event.type)) {
          if (mediaRefresh !== undefined) window.clearTimeout(mediaRefresh);
          mediaRefresh = undefined;
          void client.invalidateQueries({ queryKey: ["chat"] });
          void client.invalidateQueries({ queryKey: ["chats"] });
          void client.invalidateQueries({ queryKey: ["jobs"] });
          void client.invalidateQueries({ queryKey: ["artifacts"] });
          void client.invalidateQueries({ queryKey: ["artifact-storage"] });
          window.setTimeout(() => setLiveText({}), 200);
        }
      },
      setConnected,
      scheduleAuthoritativeRefresh,
    ).then((cleanup) => { dispose = cleanup; });
    return () => {
      if (mediaRefresh !== undefined) window.clearTimeout(mediaRefresh);
      if (authoritativeRefresh !== undefined) window.clearTimeout(authoritativeRefresh);
      dispose?.();
    };
  }, [client, setLiveText]);
  return connected;
}
