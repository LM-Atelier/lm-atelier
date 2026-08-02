import { useState } from "react";

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

  return { uploading, uploadError, setUploadError, uploadFiles };
}
