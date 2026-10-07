import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, ApiError } from "./api";
import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import { RecoveryBatchControls } from "./RecoveryBatchControls";
import { RecoveryImpactDetails as ImpactDetails } from "./RecoveryImpactDetails";
import type { RecoveryCommand, RecoveryItem, RecoveryKind, RecoveryState } from "./recoveryTypes";
import { useRecoveryBatch } from "./useRecoveryBatch";

const STATES: Record<RecoveryState, string> = {
  recoverable: "Recoverable", restoring: "Restoring", purging: "Deleting permanently",
  blocked: "Needs attention", purged: "Permanently deleted",
};

function dateText(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "long" }).format(new Date(value));
}

type Selection = { item: RecoveryItem; action: "restore" | "purge"; operationKey: string };

export function RecentlyDeleted() {
  const client = useQueryClient();
  const batch = useRecoveryBatch();
  const [state, setState] = useState<RecoveryState | "">("");
  const [kind, setKind] = useState<RecoveryKind | "">("");
  const [since, setSince] = useState("");
  const [selection, setSelection] = useState<Selection | null>(null);
  const [unfiled, setUnfiled] = useState(false);
  const [frozenCommand, setFrozenCommand] = useState<RecoveryCommand | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const page = useInfiniteQuery({
    queryKey: ["recovery-items", state, since, kind],
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam, signal }) => api.recoveryItems({
      cursor: pageParam, state: state || undefined, kind: kind || undefined,
      deletedSince: since ? `${since}T00:00:00Z` : undefined, signal,
    }),
    getNextPageParam: (last) => last.next_cursor ?? undefined,
  });
  useEffect(() => {
    const deadlines = page.data?.pages.flatMap((part) => part.items)
      .map((item) => new Date(item.purge_after).getTime()).filter((deadline) => deadline > now) ?? [];
    if (!deadlines.length) return;
    const delay = Math.max(0, Math.min(2_147_483_647, Math.min(...deadlines) - Date.now()));
    const timer = window.setTimeout(() => setNow(Date.now()), delay);
    return () => window.clearTimeout(timer);
  }, [page.data, now]);
  const impact = useQuery({
    queryKey: ["recovery-impact", selection?.item.deletion_id, selection?.operationKey],
    queryFn: ({ signal }) => api.recoveryImpact(selection!.item.deletion_id, signal),
    enabled: selection !== null,
    retry: false,
    refetchOnWindowFocus: false,
  });
  const transition = useMutation({
    mutationFn: ({ chosen, command, restoreUnfiled }: { chosen: Selection; command: RecoveryCommand; restoreUnfiled: boolean }) =>
      chosen.action === "restore"
        ? api.restoreRecovery(chosen.item.deletion_id, { ...command, restore_unfiled: restoreUnfiled })
        : api.purgeRecovery(chosen.item.deletion_id, { ...command, acknowledgement: "permanently-delete" }),
    onSuccess: (result) => {
      setNotice(result.kind === "project"
        ? result.action === "restore" ? "Project restored with its original settings. Chats moved elsewhere stay there." : "Project permanently deleted. Its chats and media remain available."
        : result.kind === "media_library_entry"
        ? result.action === "restore" ? "Media Library item restored with its favorites, collections and tags." : "Media Library item permanently removed. Shared and retained media remains available."
        : result.kind === "workflow_family"
        ? result.action === "restore" ? "Workflow restored. It remains disabled until you review and enable it." : "Workflow permanently deleted. Shared models and media remain available."
        : result.action === "restore" ? "Chat restored with its original history." : "Chat permanently deleted. Shared and retained media remains available.");
      setSelection(null);
      setFrozenCommand(null);
      for (const key of ["recovery-items", "chats", "chat-summaries", "projects", "empty-chats", "storage", "artifact-library-v1", "artifacts", "artifact-storage"])
        void client.invalidateQueries({ queryKey: [key] });
      void client.invalidateQueries({ queryKey: result.kind === "project" ? ["chat"] : ["chat", result.subject_id] });
      void client.invalidateQueries({ queryKey: ["chat-management"] });
      if (result.kind === "workflow_family") {
        for (const key of ["workflow-families", "workflow-family", "workflows", "workflow-ready-revisions"]) void client.invalidateQueries({ queryKey: [key] });
      }
    },
  });
  const busy = transition.isPending || batch.busy;
  const locked = busy || selection !== null || batch.intent !== null;
  const stale = transition.error instanceof ApiError && transition.error.status === 409;
  const missingProject = impact.data?.conflicts.includes("original_project_missing") ?? false;
  const actionAvailable = selection && impact.data?.available_actions.includes(selection.action);
  const canConfirm = Boolean(actionAvailable && !impact.isFetching && !impact.isError && !busy && !stale
    && (selection?.action !== "restore" || !missingProject || unfiled));

  function choose(item: RecoveryItem, action: Selection["action"]) {
    if (batch.intent) return;
    batch.clear();
    transition.reset();
    setFrozenCommand(null);
    setUnfiled(false);
    setSelection({ item, action, operationKey: crypto.randomUUID() });
  }

  function confirm() {
    if (!canConfirm || !selection || !impact.data) return;
    const command = frozenCommand ?? {
      expected_revision: impact.data.revision, impact_sha256: impact.data.impact_sha256,
      operation_key: selection.operationKey,
    };
    setFrozenCommand(command);
    transition.mutate({ chosen: selection, command, restoreUnfiled: unfiled });
  }

  const items = page.data?.pages.flatMap((part) => part.items) ?? [];
  const selectedMedia = selection?.item.kind === "media_library_entry";
  const selectedProject = selection?.item.kind === "project";
  const selectedWorkflow = selection?.item.kind === "workflow_family";
  return <section aria-labelledby="recently-deleted-title">
    <div className="detail-title">
      <h2 id="recently-deleted-title">Recently Deleted</h2>
      <button className="secondary" aria-disabled={page.isFetching || busy}
        onClick={() => { if (!page.isFetching && !busy) void page.refetch(); }}>Refresh</button>
    </div>
    <p className="muted">Restore deleted chats, projects, Media Library items and workflows within 30 days. Restoring does not restart any work.</p>
    <div className="row-actions storage-actions">
      <label>Type <select value={kind} aria-disabled={locked} onChange={(event) => { if (!locked) { batch.clear(); setKind(event.target.value as RecoveryKind | ""); } }}>
        <option value="">All types</option>
        <option value="chat">Chats</option>
        <option value="project">Projects</option>
        <option value="media_library_entry">Media Library items</option>
        <option value="workflow_family">Workflows</option>
      </select></label>
      <label>State <select value={state} aria-disabled={locked} onChange={(event) => { if (!locked) { batch.clear(); setState(event.target.value as RecoveryState | ""); } }}>
        <option value="">All states</option>
        {(["recoverable", "blocked", "restoring", "purging"] as const).map((value) => <option key={value} value={value}>{STATES[value]}</option>)}
      </select></label>
      <label>Deleted since (UTC) <input type="date" value={since} aria-disabled={locked}
        onChange={(event) => { if (!locked) { batch.clear(); setSince(event.target.value); } }} /></label>
    </div>
    <RecoveryBatchControls batch={batch} disabled={selection !== null || busy} />
    {notice && <p className="callout success" role="status">{notice}</p>}
    <ErrorCallout message={page.error?.message} />
    {page.isPending && <p role="status">Loading deleted items…</p>}
    {!page.isPending && !page.isError && items.length === 0 && <p>No deleted items match these filters.</p>}
    <div className="backup-list">
      {items.map((item) => {
        const recoverable = item.state === "recoverable";
        const expired = new Date(item.purge_after).getTime() <= now;
        const checked = batch.selected.some(value => value.deletion_id === item.deletion_id);
        const cannotSelect = locked || !recoverable || (!checked && batch.selected.length >= 20);
        return <article className="backup-row" key={item.deletion_id}>
          <div className="backup-copy">
            <label><input type="checkbox" aria-label={`Select ${item.display_label}`} checked={checked} aria-disabled={cannotSelect}
              onChange={() => { if (!cannotSelect) batch.toggle(item); }} /> Select</label>
            <strong>{item.display_label}</strong>
            <small>{STATES[item.state]} · {item.kind === "project" ? "Project" : item.kind === "media_library_entry" ? "Media Library" : item.kind === "workflow_family" ? "Workflow" : item.original_location.project_label ?? "Unfiled"}</small>
            <small>Deleted <time dateTime={item.deleted_at}>{dateText(item.deleted_at)}</time></small>
            <small>{expired ? "Recovery ended" : "Recover until"} <time dateTime={item.purge_after}>{dateText(item.purge_after)}</time></small>
            <ImpactDetails counts={item.counts} kind={item.kind} />
          </div>
          <div className="row-actions">
            <button className="secondary compact-button" aria-disabled={locked || !recoverable || expired}
              aria-label={`Restore ${item.display_label}`}
              onClick={() => { if (!locked && recoverable && !expired) choose(item, "restore"); }}>Restore</button>
            <button className="secondary compact-button danger" aria-disabled={locked || !recoverable}
              aria-label={`Permanently delete ${item.display_label}`}
              onClick={() => { if (!locked && recoverable) choose(item, "purge"); }}>Permanently delete</button>
          </div>
        </article>;
      })}
    </div>
    {page.hasNextPage && <button className="secondary" aria-disabled={page.isFetching}
      onClick={() => { if (!page.isFetching) void page.fetchNextPage(); }}>Load more deleted items</button>}
    {selection && <ConfirmDialog title={selection.action === "restore" ? selectedProject ? "Restore this project?" : selectedMedia ? "Restore this Media Library item?" : selectedWorkflow ? "Restore this workflow?" : "Restore this chat?" : selectedProject ? "Permanently delete this project?" : selectedMedia ? "Permanently remove this Media Library item?" : selectedWorkflow ? "Permanently delete this workflow?" : "Permanently delete this chat?"}
      question={selectedProject
        ? selection.action === "restore" ? `Restore “${selection.item.display_label}” with its original settings? Chats moved elsewhere stay there.` : `Permanently delete “${selection.item.display_label}”? Its chats and media remain available. This cannot be undone.`
        : selectedMedia
        ? selection.action === "restore" ? `Restore “${selection.item.display_label}” with its favorites, collections and tags?` : `Permanently remove “${selection.item.display_label}” from the Media Library? This cannot be undone.`
        : selectedWorkflow
        ? selection.action === "restore" ? `Restore “${selection.item.display_label}” without enabling it, granting trust or activating it?` : `Permanently delete “${selection.item.display_label}”? Shared models and media remain available. This cannot be undone.`
        : selection.action === "restore" ? `Restore “${selection.item.display_label}” with its original history?` : `Permanently delete “${selection.item.display_label}” and its history? This cannot be undone.`}
      tone={selection.action === "restore" ? "action" : "danger"}
      confirmLabel={busy ? "Working…" : transition.isError && !stale ? "Try again" : selection.action === "restore" ? selectedProject ? "Restore project" : selectedMedia ? "Restore item" : selectedWorkflow ? "Restore workflow" : "Restore chat" : "Delete permanently"}
      confirmDisabled={!canConfirm} onConfirm={confirm}
      onCancel={() => { if (!busy) { setSelection(null); setFrozenCommand(null); } }}
      detail={<>
        {impact.isFetching && <p role="status">Checking current recovery details…</p>}
        {impact.data && <ImpactDetails counts={impact.data.counts} kind={selection.item.kind} />}
        {selection.action === "purge" && <p>Shared and retained media remains available. No immediate storage reclamation is promised.</p>}
        {selection.action === "purge" && selection.item.kind === "chat" && impact.data?.delete_generated_media && <p>This chat's saved choice also removes exclusive generated media from the Media Library. Favorites and media retained elsewhere stay protected.</p>}
        {selection.action === "restore" && missingProject && <label>
          <input type="checkbox" checked={unfiled} aria-disabled={busy || frozenCommand !== null}
            onChange={(event) => { if (!busy && frozenCommand === null) setUnfiled(event.target.checked); }} />
          The original project is gone. Restore this chat unfiled.
        </label>}
        {!impact.isFetching && impact.data && !actionAvailable && <p role="alert">This action is unavailable. Refresh the list to check the current recovery state.</p>}
        <ErrorCallout message={(impact.error || transition.error)?.message} />
        {(impact.isError || stale) && <button className="secondary" onClick={() => {
          transition.reset(); setFrozenCommand(null);
          setSelection({ ...selection, operationKey: crypto.randomUUID() });
        }}>Recheck recovery details</button>}
      </>} />}
  </section>;
}
