import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { api, ApiError } from "./api";
import type { ArtifactLibraryEntry } from "./artifactLibraryPage";
import {
  organizationOperationKey, parseMediaAlbum, parseMediaTag,
  parseOrganizationPreview, parseOrganizationResult,
  type OrganizationCommand, type OrganizationPreview,
} from "./mediaOrganization";
import { useMediaOrganizationCatalog } from "./useMediaOrganizationCatalog";
import { useMediaOrganizationCreation } from "./useMediaOrganizationCreation";

export const ORGANIZATION_REVIEW_ERROR = "The selected media could not be reviewed. Refresh and select them again.";
export const ORGANIZATION_APPLY_ERROR = "The change could not be confirmed. Try again with this same selection.";
function organizationRefusal(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.status === 409) {
    if (error.code === "media-tag-conflict") return "This tag name is already in use. Choose a different name.";
    if (error.code === "media-tag-merge-trash") return "Restore media in Recently Deleted before merging this tag, then review the merge again.";
  }
  return fallback;
}
export type OrganizationReviewLabels = { targetLabel?: string; destinationLabel?: string };
type Review = { command: OrganizationCommand; preview: OrganizationPreview; operationKey: string; labels: OrganizationReviewLabels };

export function useMediaOrganization(onChanged: (command: OrganizationCommand) => void) {
  const client = useQueryClient();
  const [selection, setSelection] = useState<Map<string, ArtifactLibraryEntry>>(() => new Map());
  const [review, setReview] = useState<Review | null>(null);
  const [pending, setPending] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const [applyError, setApplyError] = useState(false);
  const [applyMessage, setApplyMessage] = useState(ORGANIZATION_APPLY_ERROR);
  const [needsReview, setNeedsReview] = useState(false);
  const [manage, setManage] = useState(false);
  const submitting = useRef(false);
  const albumCatalog = useMediaOrganizationCatalog("albums", parseMediaAlbum);
  const tagCatalog = useMediaOrganizationCatalog("tags", parseMediaTag);
  const refresh = () => {
    albumCatalog.refresh(); tagCatalog.refresh();
  };
  const creation = useMediaOrganizationCreation(refresh);
  const selected = Array.from(selection.values());
  const choose = (entry: ArtifactLibraryEntry) => {
    if (submitting.current || review) return;
    setSelection((current) => {
      const next = new Map(current);
      if (next.has(entry.id)) next.delete(entry.id);
      else if (next.size < 100) next.set(entry.id, entry);
      return next;
    });
    setReviewError(null);
  };
  const preview = async (command: OrganizationCommand, exactSelection?: ArtifactLibraryEntry[], labels: OrganizationReviewLabels = {}) => {
    if (submitting.current || review) return;
    submitting.current = true; setPending(true); setReviewError(null);
    const name = (id?: string) => {
      const choice = [...albumCatalog.items, ...tagCatalog.items].find((item) => item.id === id);
      return choice ? "name" in choice ? choice.name : choice.label : undefined;
    };
    const reviewLabels = {
      targetLabel: labels.targetLabel ?? name("target" in command ? command.target.id : undefined),
      destinationLabel: labels.destinationLabel ?? name("destination" in command ? command.destination.id : undefined),
    };
    try {
      const result = parseOrganizationPreview(await api.previewMediaOrganization(command), command, exactSelection);
      if (Date.parse(result.expires_at) <= Date.now()) throw new Error("preview-expired");
      setManage(false); setApplyError(false); setNeedsReview(false);
      setReview({ command, preview: result, operationKey: organizationOperationKey(), labels: reviewLabels });
    } catch (error) { setReviewError(organizationRefusal(error, ORGANIZATION_REVIEW_ERROR)); }
    finally { submitting.current = false; setPending(false); }
  };
  const apply = async () => {
    if (!review || submitting.current || needsReview) return;
    if (!applyError && Date.parse(review.preview.expires_at) <= Date.now()) { setNeedsReview(true); return; }
    submitting.current = true; setPending(true); setApplyError(false);
    try {
      const result = await api.applyMediaOrganization(review.preview.id, review.operationKey);
      parseOrganizationResult(result, review.preview, review.command);
      setReview(null); setSelection(new Map()); refresh(); onChanged(review.command);
      for (const key of ["recovery-items", "artifacts"]) void client.invalidateQueries({ queryKey: [key] });
    } catch (error) {
      setApplyError(true);
      setApplyMessage(organizationRefusal(error, ORGANIZATION_APPLY_ERROR));
      setNeedsReview(error instanceof ApiError && error.status >= 400 && error.status < 500);
    } finally { submitting.current = false; setPending(false); }
  };
  return {
    albums: albumCatalog.items, tags: tagCatalog.items, albumCatalog, tagCatalog,
    catalogFailed: albumCatalog.failed || tagCatalog.failed,
    catalogPending: albumCatalog.pending || tagCatalog.pending,
    selected, selection, choose, pending, review, preview, apply, reviewError, applyError, applyMessage, needsReview,
    manage, setManage, refresh, creation,
    clear: () => { if (!submitting.current && !review) setSelection(new Map()); },
    move: (id: string, offset: number) => {
      if (submitting.current || review) return;
      setSelection((current) => {
        const entries = Array.from(current.entries()); const index = entries.findIndex(([key]) => key === id);
        const next = index + offset;
        if (index < 0 || next < 0 || next >= entries.length) return current;
        [entries[index], entries[next]] = [entries[next], entries[index]];
        return new Map(entries);
      });
    },
    cancel: () => { if (!submitting.current) setReview(null); },
  };
}
