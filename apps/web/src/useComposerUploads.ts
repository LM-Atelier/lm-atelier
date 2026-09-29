import { useState, type DragEvent } from "react";

import { api } from "./api";
import { artifactOrigin } from "./messageMedia";
import type { MediaOrigin } from "./messageMedia";
import type { Artifact } from "./types";

export type ComposerAttachment = {
  id: string;
  kind: "image" | "video";
  artifact?: Artifact | null;
  origin: MediaOrigin;
};

/** Attach files to the composer, one request at a time.
 *
 * Sequential with per-file isolation: one rejected file (too large, wrong
 * type) must not abandon the rest of a selection, and the upload limit is
 * accounted per request rather than by a burst of parallel posts. Failures
 * used to be unhandled rejections, so a refused upload said nothing at all.
 */
export function useComposerUploads(
  onAttached: (attachment: ComposerAttachment) => void,
) {
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  const uploadFiles = async (files: File[]) => {
    if (!files.length) return;
    setUploading(true);
    try {
      for (const file of files) {
        try {
          const artifact = await api.upload(file);
          onAttached({
            id: artifact.id,
            kind: file.type.startsWith("video/") ? "video" : "image",
            artifact,
            origin: artifactOrigin(artifact) ?? "uploaded",
          });
        } catch (reason) {
          const detail = reason instanceof Error ? reason.message : "upload failed";
          setUploadError(`${file.name}: ${detail}`);
        }
      }
    } finally {
      setUploading(false);
    }
  };

  const uploadPastedImages = async (files: File[]) => {
    const images = files.filter((file) => file.type.startsWith("image/"));
    setUploadError(images.length < files.length ? "Only images can be pasted." : "");
    await uploadFiles(images);
  };

  return { uploading, uploadError, setUploadError, uploadFiles, uploadPastedImages };
}

/** Dropping pictures and videos onto the composer to attach them.
 *
 * Ignored while a request is being accepted. Anything that is neither a
 * picture nor a video is left out, and the composer says so rather than
 * dropping it silently.
 */
export function useComposerDrop(
  pending: { readonly current: boolean },
  uploadFiles: (files: File[]) => Promise<void>,
  setUploadError: (message: string) => void,
) {
  const [active, setActive] = useState(false);
  const handlers = {
    onDragOver: (event: DragEvent<HTMLElement>) => {
      if (pending.current) return;
      if (!Array.from(event.dataTransfer.types).includes("Files")) return;
      event.preventDefault();
      setActive(true);
    },
    onDragLeave: (event: DragEvent<HTMLElement>) => {
      if (event.currentTarget.contains(event.relatedTarget as Node)) return;
      setActive(false);
    },
    onDrop: (event: DragEvent<HTMLElement>) => {
      event.preventDefault();
      if (pending.current) return;
      setActive(false);
      const dropped = Array.from(event.dataTransfer.files);
      const files = dropped.filter(
        (file) => file.type.startsWith("image/") || file.type.startsWith("video/"),
      );
      setUploadError(
        files.length < dropped.length ? "Only images and videos can be attached." : "",
      );
      void uploadFiles(files);
    },
  };
  return { active, handlers };
}
