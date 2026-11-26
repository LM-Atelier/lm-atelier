import { useRef } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

import { ApiError, api } from "./api";

/** Why the models were not unloaded, in words that say what to do about it. */
function refusalText(error: Error): string {
  if (error instanceof ApiError && error.code === "workers-busy") {
    return "Work is running or ready to start, so the models stay loaded. Let it finish, or pause its category under Accepted work in the sidebar, then try again.";
  }
  if (error instanceof ApiError && error.code === "media-worker-in-use") {
    return "ComfyUI is in use from its own page or a workflow editor, so it was left running. Finish that work or close the editor, then try again.";
  }
  return error.message;
}

// Frees the graphics card for another program in one press. Both workers stop,
// and nothing loads a model back on its own: the next message or generation
// does, as do Load on a chat profile, Start ComfyUI and restarting LM Atelier.
export function UnloadModelsButton() {
  const client = useQueryClient();
  // Set on the press itself, so a second press before the pending state renders
  // does not send the request twice.
  const sending = useRef(false);
  const unload = useMutation({
    mutationFn: () => api.unloadWorkers(),
    onSuccess: (workers) => client.setQueryData(["workers"], workers),
    onSettled: () => {
      sending.current = false;
      void client.invalidateQueries({ queryKey: ["workers"] });
    },
  });
  const press = () => {
    if (unload.isPending || sending.current) return;
    sending.current = true;
    unload.mutate();
  };
  return (
    <>
      <button
        className="secondary"
        aria-disabled={unload.isPending}
        title="Stops the chat and media workers so another program can use the graphics card"
        onClick={press}
      >
        {unload.isPending ? "Unloading…" : "Unload all models"}
      </button>
      {unload.isSuccess && (
        <p className="muted" role="status">
          Models unloaded. The next message or generation loads what it needs, and so do Load on a
          chat profile, Start ComfyUI and restarting LM Atelier.
        </p>
      )}
      {unload.error && (
        <div className="callout error" role="alert">
          <span>{refusalText(unload.error)}</span>
        </div>
      )}
    </>
  );
}
