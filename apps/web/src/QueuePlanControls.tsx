import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import type { QueueActivityItem, QueueControlCommand } from "./types";
import "./QueueActivityDialog.css";

type Action = "hold" | "release";
type Attempt = { action: Action; command: QueueControlCommand };

export function QueuePlanControls({ item }: { item: QueueActivityItem }) {
  const client = useQueryClient();
  const [attempt, setAttempt] = useState<Attempt | null>(null);
  const refresh = () => void client.invalidateQueries({ queryKey: ["jobs", "queue"] });
  const mutation = useMutation({
    mutationFn: (request: Attempt) => api.queueControl(item.owner_id, request.action, request.command),
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
  });
  const action = item.allowed_actions[0];
  const revision = item.control_revision;
  const name = item.chat_title || item.label;
  const submit = () => {
    if (!action || revision === null || mutation.isPending) return;
    const next = attempt?.action === action && attempt.command.expected_revision === revision
      ? attempt
      : { action, command: { expected_revision: revision, idempotency_key: crypto.randomUUID() } };
    setAttempt(next);
    mutation.mutate(next);
  };
  if (!action && item.control_state !== "held" && !mutation.error) return null;
  return (
    <div className="queue-plan-controls" aria-busy={mutation.isPending}>
      {item.control_state === "held" && <span role="status">Held — queued work will wait.</span>}
      {action && revision !== null && (
        <button className="secondary compact-button" aria-disabled={mutation.isPending}
          onClick={submit} aria-label={(action === "hold" ? "Hold " : "Release ") + name}>
          {mutation.isPending ? "Saving…" : action === "hold" ? "Hold" : "Release"}
        </button>
      )}
      {mutation.error && <p role="alert">{mutation.error.message}</p>}
    </div>
  );
}
