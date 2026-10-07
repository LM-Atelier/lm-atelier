import { useRef } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { api } from "./api";
import { formatBytes } from "./format";
import type { WorkflowInstallOffer } from "./types";
import "./WorkflowInstallOfferDialog.css";

export function WorkflowInstallOfferDialog({
  offer, workflowName, onClose, onQueued,
}: {
  offer: WorkflowInstallOffer;
  workflowName: string;
  onClose: () => void;
  onQueued: () => void;
}) {
  const client = useQueryClient();
  const submitted = useRef(false);
  const install = useMutation({
    mutationFn: () => api.installWorkflowOffer(offer.id),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["workflow-families"] });
      void client.invalidateQueries({ queryKey: ["jobs"] });
      void client.invalidateQueries({ queryKey: ["workflows"] });
      onQueued();
    },
    onError: () => {
      void client.invalidateQueries({ queryKey: ["workflow-families"] });
    },
    onSettled: () => { submitted.current = false; },
  });
  const close = () => { if (!submitted.current) onClose(); };
  const submit = () => {
    if (submitted.current) return;
    submitted.current = true;
    install.mutate();
  };
  return (
    <AccessibleDialog title="Review workflow downloads" eyebrow={workflowName}
      closeLabel="Close download review" onClose={close} className="workflow-install-review">
      <p>Download the reviewed dependencies for this workflow. The current files and plans are checked again before downloads start.</p>
      <p><strong>Reviewed download size: {formatBytes(offer.total_bytes)}</strong></p>
      <h3>Workflow files</h3>
      <ul className="workflow-install-files">
        {offer.assets.map((asset) => <li key={asset.reference_filename}>
          <span><strong>{asset.reference_filename}</strong><small>{asset.kind}</small></span>
          <span>{formatBytes(asset.size_bytes)}</span>
        </li>)}
      </ul>
      <p className="muted">The total includes required companion files. Downloads may still need to finish before this workflow is ready.</p>
      <ErrorCallout message={install.error?.message} />
      <footer>
        <button className="secondary" disabled={install.isPending} onClick={close}>Cancel</button>
        <button className="primary" disabled={install.isPending} onClick={submit}>
          {install.isPending ? "Starting downloads…" : "Download reviewed files"}
        </button>
      </footer>
    </AccessibleDialog>
  );
}
