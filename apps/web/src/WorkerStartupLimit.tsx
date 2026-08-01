import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "./api";

// The startup time limit is the one worker setting a user must be able to fix
// unaided: a large model on a slow disk fails with a timeout whose remedy
// points at this control. Bounds mirror the server's (1-600 seconds) so a
// value accepted here is never rejected there.
export function WorkerStartupLimit() {
  const client = useQueryClient();
  const settings = useQuery({ queryKey: ["worker-settings"], queryFn: api.workerSettings });
  // null means "no edit in progress": the field tracks the saved value until touched.
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (settings.data ? String(settings.data.worker_startup_seconds) : "");
  const seconds = Number(shown);
  const valid = shown.trim() !== "" && Number.isFinite(seconds) && seconds >= 1 && seconds <= 600;
  const save = useMutation({
    mutationFn: () => api.updateWorkerSettings({ worker_startup_seconds: seconds }),
    onSuccess: (value) => {
      setDraft(null);
      client.setQueryData(["worker-settings"], value);
    },
  });
  return (
    <div className="row-actions">
      {save.error && (
        <div className="callout error" role="alert">
          <span>{save.error.message}</span>
        </div>
      )}
      <label className="inline-field">
        Startup time limit (seconds)
        <input
          type="number"
          min={1}
          max={600}
          aria-label="Worker startup time limit in seconds"
          value={shown}
          onChange={(event) => setDraft(event.target.value)}
        />
      </label>
      <button
        className="secondary"
        disabled={
          !valid
          || save.isPending
          || draft === null
          || !settings.data
          || seconds === settings.data.worker_startup_seconds
        }
        onClick={() => save.mutate()}
      >
        {save.isPending ? "Saving…" : "Save limit"}
      </button>
    </div>
  );
}
