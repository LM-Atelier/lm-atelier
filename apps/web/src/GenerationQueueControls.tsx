import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import type { GenerationQueueAction, QueueControlCommand } from "./types";
import "./GenerationQueueControls.css";

type Attempt = { action: GenerationQueueAction; command: QueueControlCommand };

export function GenerationQueueControls() {
  const client = useQueryClient();
  const saving = useRef(false);
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const policy = useQuery({
    queryKey: ["jobs", "queue", "generation-policy"],
    queryFn: ({ signal }) => api.generationQueuePolicy(signal),
    refetchInterval: 5_000,
  });
  const refresh = () => void client.invalidateQueries({ queryKey: ["jobs", "queue"] });
  const mutation = useMutation({
    mutationFn: (next: Attempt) => api.generationQueueControl(next.action, next.command),
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
    <section className="generation-queue-controls" aria-label="Generation dispatch"
      aria-busy={mutation.isPending}>
      <div>
        <strong>Generation</strong>
        {policy.isPending && <p role="status">Loading generation state…</p>}
        {value && <p role="status">
          {value.dispatch_state === "open"
            ? "Generation can start."
            : value.dispatch_state === "draining"
              ? "Finishing current generation. New generation will wait."
              : "Generation paused. New submissions stay queued."}
        </p>}
        <small>Pausing lets current generation finish. Held work stays held when you resume.</small>
      </div>
      {action && (
        <button className="secondary compact-button" onClick={submit}
          aria-disabled={unavailable}
          aria-label={mutation.isPending ? "Saving generation change"
            : action === "pause_after_current" ? "Pause generation after current work" : "Resume generation"}>
          {mutation.isPending ? "Saving…" : action === "pause_after_current" ? "Pause after current" : "Resume"}
        </button>
      )}
      {policy.error && <div className="generation-queue-error">
        <p role="alert">{policy.error.message}</p>
        <button className="secondary compact-button" onClick={() => void policy.refetch()}>
          Retry generation state
        </button>
      </div>}
      {mutation.error && !conflictResolved && <p role="alert" className="generation-queue-error">{mutation.error.message}</p>}
    </section>
  );
}
