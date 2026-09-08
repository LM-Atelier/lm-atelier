import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { api } from "./api";
import { artifactSource } from "./messageMedia";
import type { ReferenceAsset, ReferenceAssetReview } from "./types";

export function ReferenceAssetReviewDialog({
  asset,
  subjectName,
  onClose,
}: {
  asset: ReferenceAsset;
  subjectName: string;
  onClose: () => void;
}) {
  const client = useQueryClient();
  const [outcome, setOutcome] = useState<ReferenceAssetReview["outcome"] | "">("");
  const [reason, setReason] = useState("");
  const queryKey = ["reference-assets", asset.reference_subject_id];
  const review = useMutation({
    mutationFn: (body: ReferenceAssetReview) =>
      api.reviewReferenceAsset(asset.reference_subject_id, asset.id, body),
    onSuccess: async (result) => {
      // The recorded response settles this image. An older list request must
      // not replace it with the unchecked state that preceded the decision.
      await client.cancelQueries({ queryKey });
      client.setQueryData<ReferenceAsset[]>(queryKey, (previous) =>
        previous?.map((item) => item.id === result.asset.id ? result.asset : item),
      );
      void client.invalidateQueries({ queryKey: ["references"] });
      onClose();
    },
    onError: () => {
      // Another window may have settled or detached the image. Refresh the
      // list while retaining this dialog's decision and the server's refusal.
      void client.invalidateQueries({ queryKey });
    },
  });
  const trimmed = reason.trim();
  const canSave = outcome !== "" && (outcome === "usable" || trimmed !== "") && !review.isPending;
  const close = () => {
    if (!review.isPending) onClose();
  };

  return (
    <AccessibleDialog
      title={`Review image ${asset.sort_order + 1}`}
      eyebrow={subjectName}
      closeLabel="Close image review"
      onClose={close}
    >
      <div className="media-card">
        <img
          src={artifactSource(asset.artifact_id) ?? undefined}
          alt={asset.caption ?? `${subjectName}, ${asset.purpose}`}
        />
      </div>
      <p className="muted">Review this image for its intended purpose: {asset.purpose}. A saved review cannot be changed.</p>
      <label className="prompt-field">
        Review outcome
        <select
          aria-label="Review outcome"
          value={outcome}
          disabled={review.isPending}
          onChange={(event) => {
            const value = event.target.value;
            if (value === "" || value === "usable" || value === "weak" || value === "rejected") setOutcome(value);
          }}
        >
          <option value="">Choose an outcome</option>
          <option value="usable">Usable</option>
          <option value="weak">Weak</option>
          <option value="rejected">Rejected</option>
        </select>
      </label>
      <label className="prompt-field">
        Review reason
        <textarea
          aria-label="Review reason"
          value={reason}
          maxLength={200}
          disabled={review.isPending}
          onChange={(event) => setReason(event.target.value)}
        />
      </label>
      <p className="muted">A reason is required for weak or rejected images.</p>
      {review.error ? (
        <ErrorCallout message={review.error instanceof Error ? review.error.message : "The review could not be saved."} />
      ) : null}
      <footer>
        <button className="secondary" disabled={review.isPending} onClick={close}>Cancel</button>
        <button
          className="primary"
          disabled={!canSave}
          onClick={() => {
            if (canSave && outcome) review.mutate({ outcome, reasons: trimmed ? [trimmed] : [] });
          }}
        >
          {review.isPending ? "Saving review…" : "Save review"}
        </button>
      </footer>
    </AccessibleDialog>
  );
}
