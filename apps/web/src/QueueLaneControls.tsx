import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import type { GenerationQueueAction, QueueControlCommand } from "./types";
import "./GenerationQueueControls.css";

type Attempt = { action: GenerationQueueAction; command: QueueControlCommand };

export function QueueLaneControls({ lane }: { lane: "generation" | "transfer" }) {
  const name = lane === "transfer" ? "transfers" : "generation";
  const title = lane === "transfer" ? "Transfers" : "Generation";
  const client = useQueryClient();
  const saving = useRef(false);
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const policy = useQuery({
    queryKey: ["jobs", "queue", lane + "-policy"],
    queryFn: async ({ signal }) => lane === "transfer"
      ? api.transferQueuePolicy(signal) : api.generationQueuePolicy(signal),
    refetchInterval: 5_000,
  });
  const refresh = () => void client.invalidateQueries({ queryKey: ["jobs", "queue"] });
  const mutation = useMutation({
    mutationFn: async (next: Attempt) => lane === "transfer"
      ? api.transferQueueControl(next.action, next.command)
      : api.generationQueueControl(next.action, next.command),
    onSuccess: () => {
      setAttempt(null);
      refresh();
    },
    onError: (error) => {
      if (error instanceof ApiError && error.status === 409) {
        setAttempt(null);
        refresh();
      }
    },
    onSettled: () => { saving.current = false; },
  });
  const value = policy.data;
  const conflictResolved = mutation.error instanceof ApiError
    && mutation.error.status === 409 && mutation.error.code === "queue-lane-conflict"
    && value !== undefined && mutation.variables !== undefined
    && value.revision > mutation.variables.command.expected_revision;
  const action = value?.allowed_actions[0];
  const unavailable = mutation.isPending || policy.isError;
  const submit = () => {
    if (!value || !action || unavailable || saving.current) return;
    const next = attempt?.action === action && attempt.command.expected_revision === value.revision
      ? attempt
      : { action, command: { expected_revision: value.revision, idempotency_key: crypto.randomUUID() } };
    saving.current = true;
    setAttempt(next);
    mutation.mutate(next);
  };

  return (
    <section className="generation-queue-controls" aria-label={title + " dispatch"}
      aria-busy={mutation.isPending}>
      <div>
        <strong>{title}</strong>
        {policy.isPending && <p role="status">Loading {name} state…</p>}
        {value && <p role="status">
          {value.dispatch_state === "open"
            ? title + " can start."
            : value.dispatch_state === "draining"
              ? "Finishing current " + name + ". New " + name + " will wait."
              : title + " paused. New submissions stay queued."}
        </p>}
        {value && <p>Active jobs: {value.running_jobs}</p>}
        <small>{lane === "transfer"
          ? "Pausing lets current downloads and their activation finish. Manually paused downloads stay paused when you resume."
          : "Pausing lets current generation finish. Held work stays held when you resume."}</small>
      </div>
      {action && (
        <button className="secondary compact-button" onClick={submit}
          aria-disabled={unavailable}
          aria-label={mutation.isPending ? "Saving " + name + " change"
            : action === "pause_after_current" ? "Pause " + name + " after current work" : "Resume " + name}>
          {mutation.isPending ? "Saving…" : action === "pause_after_current" ? "Pause after current" : "Resume"}
        </button>
      )}
      {policy.error && <div className="generation-queue-error">
        <p role="alert">{policy.error.message}</p>
        <button className="secondary compact-button" onClick={() => void policy.refetch()}>
          Retry {name} state
        </button>
      </div>}
      {mutation.error && !conflictResolved && <p role="alert" className="generation-queue-error">{mutation.error.message}</p>}
    </section>
  );
}
