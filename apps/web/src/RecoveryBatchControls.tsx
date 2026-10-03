import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import { RecoveryImpactDetails } from "./RecoveryImpactDetails";
import type { useRecoveryBatch } from "./useRecoveryBatch";
import "./RecoveryBatchControls.css";

function dateText(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "long" }).format(new Date(value));
}

export function RecoveryBatchControls({ batch, disabled }: { batch: ReturnType<typeof useRecoveryBatch>; disabled: boolean }) {
  const { intent, preview, frozen, busy } = batch;
  const document = batch.displayedPreview;
  const noticeRef = useRef<HTMLParagraphElement>(null);
  useEffect(() => { if (batch.notice) noticeRef.current?.focus(); }, [batch.notice]);
  const locked = disabled || intent !== null;
  const missingProject = intent?.action === "restore"
    && document?.items.some(item => item.impact.conflicts.includes("original_project_missing"));
  return <>
    {batch.notice && <p ref={noticeRef} tabIndex={-1} className="callout success" role="status">{batch.notice}</p>}
    {batch.selected.length > 0 && <div className="row-actions storage-actions" aria-label="Selected deleted items">
      <span role="status">{batch.selected.length} of 20 items selected</span>
      <button className="secondary" aria-disabled={locked} onClick={() => { if (!locked) batch.review("restore"); }}>Review restore selection</button>
      <button className="secondary danger" aria-disabled={locked} onClick={() => { if (!locked) batch.review("purge"); }}>Review permanent deletion</button>
      <button className="secondary" aria-disabled={locked} onClick={() => { if (!locked) batch.clear(); }}>Clear selection</button>
      {intent && !batch.open && <>
        <span>An earlier request still needs its result checked.</span>
        <button className="secondary" onClick={batch.reopen}>Review pending action</button>
      </>}
    </div>}
    {intent && batch.open && <ConfirmDialog
      title={intent.action === "restore" ? "Restore selected items?" : "Permanently delete selected items?"}
      question={intent.action === "restore"
        ? "All or nothing: every selected item is restored together. If any item changes, none are restored. No work restarts."
        : "All or nothing: every selected item is permanently deleted together. If any item changes, none are deleted. This cannot be undone."}
      tone={intent.action === "restore" ? "action" : "danger"}
      confirmLabel={busy ? "Working…" : batch.transition.isError && !batch.rejected ? "Try this request again"
        : intent.action === "restore" ? "Restore selected items" : "Delete selected permanently"}
      confirmDisabled={!batch.canConfirm} onConfirm={batch.confirm} onCancel={batch.dismiss}
      detail={<>
        {preview.isFetching && !frozen && <p role="status">Checking the whole selection…</p>}
        {document && <>
          <p>Confirm before <time dateTime={document.expires_at}>{dateText(document.expires_at)}</time>.</p>
          <ul className="recovery-batch-members">
            {document.items.map(item => <li key={item.deletion_id}>
              <strong>{item.display_label}</strong>
              <p>Recovery ends <time dateTime={item.purge_after}>{dateText(item.purge_after)}</time>.</p>
              <RecoveryImpactDetails counts={item.impact.counts} kind={item.impact.kind} />
              {item.impact.conflicts.includes("active_work") && <p>{item.impact.counts.active_work} active work items must finish before this item can change.</p>}
              {item.impact.conflicts.includes("active_selection") && <p>This workflow is selected or set as a default. Clear that use before changing it.</p>}
              {item.impact.conflicts.includes("subject_missing") && <p>The original item is unavailable.</p>}
              {item.impact.conflicts.includes("record_changed") && <p>The saved record changed. Check its recovery details again.</p>}
              {!item.impact.available_actions.includes(intent.action) && <p role="alert">This item cannot be changed now. None of the selected items will be changed.</p>}
              {intent.action === "purge" && item.impact.kind === "chat" && item.impact.delete_generated_media
                && <p>This chat's saved choice also removes exclusive generated media from the Media Library. Favorites and media retained elsewhere stay protected.</p>}
            </li>)}
          </ul>
          {!document.available && <p role="alert">This selection is not ready. Resolve the conflicts or choose a different selection.</p>}
        </>}
        {missingProject && <label>
          <input type="checkbox" checked={intent.restoreUnfiled} aria-disabled={busy || frozen !== null}
            onChange={event => { if (!busy && !frozen) batch.recheck(event.target.checked); }} />
          Restore chats whose original project is gone as unfiled. Check the updated selection before confirming.
        </label>}
        {intent.action === "purge" && <>
          <p>Shared and retained media remains available. No immediate storage reclamation is promised. Per-item storage estimates may overlap.</p>
          <label><input type="checkbox" checked={batch.acknowledged} aria-disabled={busy || frozen !== null}
            onChange={event => batch.acknowledge(event.target.checked)} /> I understand permanent deletion cannot be undone.</label>
        </>}
        {batch.expired && !frozen && <p role="alert">The selection preview expired. Check the recovery details again.</p>}
        <ErrorCallout message={(preview.error || batch.transition.error)?.message} />
        {frozen && batch.transition.isError && !batch.rejected && <p role="status">The result is uncertain. Try the same request again to check its result. The selection and choices are fixed.</p>}
        {(preview.isError || batch.rejected || (batch.expired && !frozen)) && <button className="secondary"
          aria-disabled={busy} onClick={() => { if (!busy) batch.recheck(); }}>Recheck selected recovery details</button>}
      </>} />}
  </>;
}
import { useEffect, useRef } from "react";
