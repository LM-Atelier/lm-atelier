import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { api } from "./api";
import { artifactOrigin, artifactSource } from "./messageMedia";
import type { ArtifactLibraryItem } from "./types";
import type { ComposerAttachment } from "./useComposerUploads";

/** Attach something already in the library, not only something on disk.
 *
 * The composer could reach a file picker and nothing else, so using a picture
 * the app had itself generated meant exporting it and uploading it back - the
 * same bytes making a round trip through the filesystem to arrive where they
 * started, and arriving as a second artifact with a second identity.
 *
 * Picking from the library attaches the existing artifact by its id, so the
 * reference is the same object the library already holds.
 */
export function LibraryAttachPicker({
  onAttach,
  onClose,
}: {
  onAttach: (attachment: ComposerAttachment) => void;
  onClose: () => void;
}) {
  const [chosen, setChosen] = useState<string[]>([]);
  const library = useQuery({
    queryKey: ["artifacts", "composer-picker"],
    queryFn: () => api.artifacts("image", "", false),
  });

  const usable = (library.data ?? []).filter(
    (item): item is ArtifactLibraryItem => item.kind === "image",
  );

  return (
    <AccessibleDialog
      title="Attach from the library"
      eyebrow="Media"
      closeLabel="Close the library picker"
      onClose={onClose}
    >
      {library.error && <ErrorCallout message={(library.error as Error).message} />}
      {library.isPending && <p>Reading the library…</p>}
      {!library.isPending && usable.length === 0 && (
        <p>Nothing in the library yet. Anything generated or uploaded appears here.</p>
      )}
      <ul className="library-attach-grid">
        {usable.map((item) => {
          const picked = chosen.includes(item.id);
          return (
            <li key={item.id}>
              <button
                type="button"
                className={`library-attach-tile ${picked ? "picked" : ""}`}
                aria-pressed={picked}
                aria-label={item.original_name ?? `${item.kind} ${item.id}`}
                onClick={() =>
                  setChosen((current) =>
                    picked ? current.filter((id) => id !== item.id) : [...current, item.id],
                  )
                }
              >
                <img src={artifactSource(item.id) ?? undefined} alt="" loading="lazy" />
              </button>
            </li>
          );
        })}
      </ul>
      <footer>
        <button className="secondary" onClick={onClose}>
          Cancel
        </button>
        <button
          className="primary"
          disabled={chosen.length === 0}
          onClick={() => {
            for (const id of chosen) {
              const item = usable.find((candidate) => candidate.id === id);
              if (!item) continue;
              onAttach({
                id: item.id,
                kind: "image",
                artifact: item,
                // Whatever it already was. Attaching a generated picture does
                // not make it an upload, and the transcript should not say a
                // person supplied something the app produced.
                origin: artifactOrigin(item) ?? "generated",
              });
            }
            onClose();
          }}
        >
          Attach {chosen.length || ""}
        </button>
      </footer>
    </AccessibleDialog>
  );
}
