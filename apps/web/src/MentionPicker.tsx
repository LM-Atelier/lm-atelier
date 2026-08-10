import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import type { ReferenceSubject } from "./types";

/** Choose which subject an `@` means.
 *
 * The list is the only way a reference is attached. Typing the characters
 * never binds anything, because the alternative - matching prose against
 * names - silently binds whoever the text most resembles, which is the one
 * thing this whole path refuses to do.
 *
 * Archived subjects are excluded. Archiving is the removal a person expects to
 * be honoured everywhere, and offering one here would quietly undo it.
 */
export function MentionPicker({
  query,
  onChoose,
}: {
  /** The partial name after `@`. Empty means "everything"; null means closed. */
  query: string | null;
  onChoose: (subject: ReferenceSubject) => void;
}) {
  const open = query !== null;
  const subjects = useQuery({
    queryKey: ["references", query ?? "", false],
    queryFn: () => api.references(query ?? "", false),
    enabled: open,
  });

  if (!open) return null;

  const items = subjects.data?.items ?? [];

  return (
    <div className="mention-picker" role="listbox" aria-label="References">
      {subjects.isPending && <p className="muted">Looking…</p>}
      {!subjects.isPending && items.length === 0 && (
        // Says which of the two reasons applies, because "no references at
        // all" and "none match what you typed" need different next actions.
        <p className="muted">
          {query
            ? `Nothing here answers to @${query}.`
            : "No references yet. Create one in the References library."}
        </p>
      )}
      {items.map((subject) => (
        <button
          key={subject.id}
          type="button"
          role="option"
          aria-selected={false}
          className="mention-option"
          // Chosen on mousedown rather than click: the textarea loses focus on
          // blur first, which closes the picker before a click can land.
          onMouseDown={(event) => {
            event.preventDefault();
            onChoose(subject);
          }}
        >
          <strong>{subject.name}</strong>
          <code>@{subject.mention_slug}</code>
          <span className="badge">{subject.kind}</span>
        </button>
      ))}
    </div>
  );
}
