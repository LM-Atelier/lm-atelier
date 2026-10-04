import { useState, type ReactNode } from "react";
import { useSensitiveMediaChoice } from "./sensitiveMedia";
import { ShieldedThumbnail } from "./ShieldedThumbnail";
import { useInfiniteQuery } from "@tanstack/react-query";
import { AccessibleDialog } from "./AccessibleDialog";
import { ErrorCallout } from "./ErrorCallout";
import { api } from "./api";
import { artifactSource } from "./messageMedia";
import type { ArtifactLibraryItem } from "./types";

/** Choose images the app already holds.
 *
 * Several places need this - the composer, a reference collecting the pictures
 * that show it, and the Studio taking a new subject from one - and they differ
 * only in what they do with the result and how many it takes. The grid, the
 * selection, and the nothing-here-yet case are the same problem each time, so
 * they live here while each caller keeps its own verb.
 */
export function LibraryImagePicker({
  title,
  confirmLabel,
  onConfirm,
  onClose,
  single = false,
  children,
}: {
  title: string;
  confirmLabel: string;
  /** Only one picture is taken: choosing another replaces the one chosen. */
  single?: boolean;
  onConfirm: (items: ArtifactLibraryItem[]) => void;
  onClose: () => void;
  children?: ReactNode;
}) {
  const [chosen, setChosen] = useState<string[]>([]);
  const shielding = useSensitiveMediaChoice() !== "show";
  const library = useInfiniteQuery({
    queryKey: ["artifacts", "image-picker", { pageSize: 50 }],
    initialPageParam: 0,
    queryFn: ({ pageParam, signal }) =>
      api.artifacts("image", "", false, { limit: 50, offset: pageParam }, signal),
    getNextPageParam: (lastPage, _pages, lastOffset) =>
      lastPage.length === 50 ? lastOffset + 50 : undefined,
  });

  // Offset pages may overlap if images arrive while the picker is open.
  const usable = Array.from(new Map(
    (library.data?.pages.flat() ?? [])
      .filter((item) => item.kind === "image")
      .map((item) => [item.id, item] as const),
  ).values());

  return (
    <AccessibleDialog
      title={title}
      eyebrow="Media"
      closeLabel="Close the library picker"
      onClose={onClose}
    >
      {library.error && <ErrorCallout message={(library.error as Error).message} />}
      {library.isPending && <p>Reading the library…</p>}
      {!library.isPending && !library.error && usable.length === 0 && (
        <p>Nothing in the library yet. Anything generated or uploaded appears here.</p>
      )}
      {children}
      <ul className="library-attach-grid">
        {usable.map((item, index) => {
          const picked = chosen.includes(item.id);
          const name = shielding ? `Picture ${index + 1}` : item.original_name ?? `${item.kind} ${item.id}`;
          return (
            <li key={item.id}>
              <button
                type="button"
                className={`library-attach-tile ${picked ? "picked" : ""}`}
                aria-pressed={picked}
                aria-label={name}
                onClick={() =>
                  setChosen((current) =>
                    picked ? current.filter((id) => id !== item.id) : single ? [item.id] : [...current, item.id],
                  )
                }
              >
                <ShieldedThumbnail src={artifactSource(item.id) ?? undefined} kind="image" />
              </button>
            </li>
          );
        })}
      </ul>
      {(library.hasNextPage || library.isError) && (
        <button
          type="button"
          className="secondary"
          disabled={library.isFetching}
          onClick={() => {
            if (library.hasNextPage) void library.fetchNextPage();
            else void library.refetch();
          }}
        >
          {library.isFetching ? "Loading images…" : library.isError
            ? "Retry loading images" : "Load more images"}
        </button>
      )}
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
          {single ? confirmLabel : `${confirmLabel} ${chosen.length || ""}`}
        </button>
      </footer>
    </AccessibleDialog>
  );
}
