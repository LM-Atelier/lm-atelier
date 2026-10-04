import { ConfirmDialog } from "./ConfirmDialog";
import { ErrorCallout } from "./ErrorCallout";
import { formatBytes } from "./format";
import { type useMediaOrganization } from "./useMediaOrganization";

export function MediaOrganizationReview({ organization }: { organization: ReturnType<typeof useMediaOrganization> }) {
  const { review, pending, applyError, needsReview } = organization;
  if (!review) return null;
  const trash = review.command.action === "trash";
  const actions = { "add-to-album": "Add to album", "remove-from-album": "Remove from album", "reorder-album": "Reorder in album",
    "add-tag": "Add tag", "remove-tag": "Remove tag", "set-favorite": "Change favorite", "trash": "Move to Recently Deleted",
    "rename-album": "Rename album", "rename-tag": "Rename tag", "delete-album": "Remove album", "delete-tag": "Remove tag", "merge-tags": "Merge tags" };
  return <ConfirmDialog title="Review library changes" tone="action"
    question={`${review.preview.selected_count} media items · ${formatBytes(review.preview.size_bytes)}`}
    confirmLabel={pending ? "Applying…" : "Apply changes"} confirmDisabled={pending || needsReview}
    onCancel={organization.cancel} onConfirm={() => void organization.apply()}
    detail={<>
      <p>{actions[review.command.action]}{review.labels.targetLabel ? `: ${review.labels.targetLabel}` : ""}</p>
      {"changes" in review.command && <p>New name: {review.command.changes.name ?? review.command.changes.label}</p>}
      {review.labels.destinationLabel && <p>Merge into: {review.labels.destinationLabel}</p>}
      <p>{review.preview.changed_count} changes will be applied to this exact selection.</p>
      {trash && <p>Selected items move to Recently Deleted. You can restore them for 30 days with their favorites, albums and tags.</p>}
      <p>{trash ? "Existing chats and References keep their media. No media bytes are removed now."
        : "Media files and existing chats stay intact."}</p>
      <ErrorCallout message={applyError ? organization.applyMessage : undefined} />
      {needsReview && <p role="alert">The saved review is no longer usable. Close it and review the selection again.</p>}
    </>} />;
}
