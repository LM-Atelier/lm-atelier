import { useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { api } from "./api";
import { artifactSource } from "./messageMedia";
import type { ArtifactLibraryItem } from "./types";

/** Choose images the app already holds.
 *
 * Two places need this - the composer, and a reference collecting the pictures
 * that show it - and they differ only in what they do with the result. The
 * grid, the selection, and the nothing-here-yet case are the same problem
 * twice, so they live here while each caller keeps its own verb.
 */
export function LibraryImagePicker({
  title,
  confirmLabel,
  onConfirm,
  onClose,
  children,
}: {
  title: string;
  confirmLabel: string;
  onConfirm: (items: ArtifactLibraryItem[]) => void;
  onClose: () => void;
  children?: ReactNode;
}) {
  const [chosen, setChosen] = useState<string[]>([]);
  const library = useQuery({
    queryKey: ["artifacts", "image-picker"],
    queryFn: () => api.artifacts("image", "", false),
  });

  const usable = (library.data ?? []).filter(
    (item): item is ArtifactLibraryItem => item.kind === "image",
  );

  return (
    <AccessibleDialog
      title={title}
      eyebrow="Media"
      closeLabel="Close the library picker"
      onClose={onClose}
    >
      {library.error && <ErrorCallout message={(library.error as Error).message} />}
      {library.isPending && <p>Reading the library…</p>}
      {!library.isPending && usable.length === 0 && (
        <p>Nothing in the library yet. Anything generated or uploaded appears here.</p>
      )}
      {children}
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
            // Selection order, not library order. Someone picking several
            // pictures of one subject is usually deciding a sequence as they
            // click, and re-sorting it under them would discard that.
            onConfirm(
              chosen
                .map((id) => usable.find((candidate) => candidate.id === id))
                .filter((item): item is ArtifactLibraryItem => item !== undefined),
            );
            onClose();
          }}
        >
          {confirmLabel} {chosen.length || ""}
        </button>
      </footer>
    </AccessibleDialog>
  );
}
