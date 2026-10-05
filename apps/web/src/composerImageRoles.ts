import type { ImageInputRole } from "./types";
import type { ComposerAttachment } from "./useComposerUploads";

export const IMAGE_PURPOSE_ERROR = "Choose a purpose for each picture before sending.";

export function imageRoleVectorShape(value: unknown, ids: unknown): boolean {
  return value === undefined || value === null || (Array.isArray(value) && Array.isArray(ids)
    && value.length === ids.length && new Set(ids).size === ids.length
    && value.every((role) => role === "edit_source" || role === "reference")
    && value.filter((role) => role === "edit_source").length <= 1);
}

/** A complete explicit selection, or the unchanged older positional convention. */
export function attachmentImageRoles(attachments: readonly ComposerAttachment[]): ImageInputRole[] | undefined {
  if (!attachments.some((item) => item.imageRole !== undefined)) return undefined;
  if (attachments.some((item) => item.kind !== "image" || item.imageRole === undefined)) throw new Error(IMAGE_PURPOSE_ERROR);
  const roles = attachments.map((item) => item.imageRole as ImageInputRole);
  if (roles.filter((role) => role === "edit_source").length > 1) throw new Error(IMAGE_PURPOSE_ERROR);
  return roles;
}

/** Choosing a canvas makes every other picture a reference, preserving attachment order. */
export function withImagePurpose(attachments: readonly ComposerAttachment[], id: string, role: ImageInputRole | undefined): ComposerAttachment[] {
  if (!attachments.some((item) => item.id === id && item.kind === "image")) return [...attachments];
  if (role === undefined) return attachments.map((item) => ({ ...item, imageRole: undefined }));
  return attachments.map((item) => item.kind !== "image" ? item : ({
    ...item,
    imageRole: item.id === id ? role : role === "edit_source" ? "reference" : item.imageRole ?? "reference",
  }));
}

export function editSourceId(attachments: readonly ComposerAttachment[]): string | null {
  if (attachments.some((item) => item.imageRole !== undefined)) {
    return attachments.find((item) => item.kind === "image" && item.imageRole === "edit_source")?.id ?? null;
  }
  return attachments[0]?.kind === "image" ? attachments[0].id : null;
}
