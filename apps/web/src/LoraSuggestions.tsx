import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "./api";
import { ErrorCallout } from "./ErrorCallout";
import { InstallConfirmDialog } from "./InstallConfirmDialog";
import type { CatalogModel } from "./types";
import { useCatalogInstall } from "./useCatalogInstall";

const GAPS = {
  family_unknown: "The model this workflow runs could not be identified, so there are no suggestions for it.",
  family_unsupported: "There are no LoRA suggestions for this kind of model yet.",
} as const;

function count(value: number | null | undefined, noun: string): string | null {
  return typeof value === "number" ? `${value.toLocaleString()} ${noun}` : null;
}

/** Top-rated LoRAs on CivitAI for the model this workflow runs, to install on request.
 *
 * Asked for only when opened, so the settings panel makes no network request
 * until someone wants suggestions. Installing one goes through the same
 * preflight and confirmation as installing from Models; nothing is added to the
 * stack until the person adds it.
 */
export function LoraSuggestions({ revisionId }: { revisionId: string }) {
  const [open, setOpen] = useState(false);
  const suggestions = useQuery({
    queryKey: ["workflows", "lora-suggestions", revisionId],
    queryFn: ({ signal }) => api.workflowLoraSuggestions(revisionId, signal),
    enabled: open,
    staleTime: 5 * 60 * 1000,
  });
  const install = useCatalogInstall();
  const [queued, setQueued] = useState<string[]>([]);
  const data = suggestions.data;
  const name = (item: CatalogModel) => item.parent_model_name ?? item.name;
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Top-rated LoRAs for this model</summary>
      {suggestions.isPending && open && <p role="status">Finding LoRAs…</p>}
      <ErrorCallout message={suggestions.error?.message} />
      {data?.gap && <p className="muted">{GAPS[data.gap]}</p>}
      {data?.stale && <p className="muted" role="status">Showing saved suggestions while CivitAI is unavailable.</p>}
      {data && !data.gap && data.items.length === 0 && (
        <p className="muted">No suggestions that are not already installed.</p>
      )}
      {data && data.items.length > 0 && (
        <ul className="settings-list">
          {data.items.map((item) => {
            const facts = [item.author, count(item.likes, "likes"), count(item.downloads, "downloads")].filter(Boolean);
            const preparing = install.prepare.isPending && install.prepare.variables?.model.remote_id === item.remote_id;
            return (
              <li key={item.remote_id} className="lora-stack-item">
                <span>
                  <strong>{name(item)}</strong>
                  {facts.length > 0 && <small>{facts.join(" · ")}</small>}
                </span>
                {queued.includes(item.remote_id) ? (
                  <small role="status">Installing. It can be added once the download finishes.</small>
                ) : (
                  <button
                    type="button"
                    className="secondary"
                    aria-label={`Install ${name(item)}`}
                    disabled={install.prepare.isPending}
                    onClick={() => install.prepare.mutate({ model: item, selectedRole: "lora" })}
                  >
                    {preparing ? "Checking…" : "Install"}
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
      <ErrorCallout message={install.prepare.error?.message ?? install.confirm.error?.message} />
      {install.pendingInstall && (
        <InstallConfirmDialog
          name={name(install.pendingInstall.model)}
          preflight={install.pendingInstall.preflight}
          pending={install.confirm.isPending}
          onConfirm={() => {
            const pending = install.pendingInstall!;
            install.confirm.mutate(pending, {
              onSuccess: () => setQueued((current) => [...current, pending.model.remote_id]),
            });
          }}
          onCancel={install.cancel}
        />
      )}
    </details>
  );
}
