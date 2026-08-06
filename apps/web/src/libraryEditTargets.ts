import type { ArtifactLibraryItem } from "./types";
import type { ComposerAttachment } from "./useComposerUploads";
import { focusMainContent } from "./viewHelpers";

export type VisualTarget = {
  attachment: ComposerAttachment;
  // null means "attach only", used by Reference: quoting a picture says what
  // the next turn is about; it must not silently change the generation mode
  // the user chose, the way Edit and Animate deliberately do.
  mode: "image" | "video" | null;
  requestId: number;
  // Open the editing studio once the image is attached - the library's Edit
  // entry point, where no instruction has been drafted yet.
  studio?: boolean;
  // Companions from a library multi-select; the studio's "Apply to each"
  // path then runs one turn per image.
  extraAttachments?: ComposerAttachment[];
};

/** Turning a library selection into something a surface can open.
 *
 * Pure enough to live away from the shell: what a selection means is a
 * property of the selection, not of whichever view happened to make it.
 */
export function libraryEditTarget(artifacts: ArtifactLibraryItem[]): VisualTarget | null {
  const [first, ...rest] = artifacts.map((artifact): ComposerAttachment => ({
    id: artifact.id,
    kind: "image",
    artifact,
    // Stored uploads keep their filename; generated media has none.
    origin: artifact.original_name ? "uploaded" : "generated",
  }));
  if (!first) return null;
  return {
    attachment: first,
    mode: "image",
    requestId: Date.now(),
    studio: true,
    extraAttachments: rest,
  };
}

/** One image opens the studio canvas; a selection keeps the composer's
 * apply-to-each batch path. */
export function openLibraryEditTargets(
  artifacts: ArtifactLibraryItem[],
  handlers: {
    openStudio: (artifactId: string) => void;
    openComposer: (target: VisualTarget) => void;
  },
): void {
  if (artifacts.length === 1) {
    handlers.openStudio(artifacts[0].id);
    focusMainContent();
    return;
  }
  const target = libraryEditTarget(artifacts);
  if (!target) return;
  handlers.openComposer(target);
  focusMainContent();
}
