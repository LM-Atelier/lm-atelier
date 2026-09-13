import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { ConfirmDialog } from "./ConfirmDialog";
import { ApiError, api } from "./api";
import type { EmptyChatEntry, EmptyChatPreview } from "./types";

/** The key the workspace keeps the open chat under, so it is never offered as safe. */
export const CURRENT_CHAT_STORAGE_KEY = "local-lm-chat";

/** Why an empty chat is listed as somebody's choice rather than untouched.
 *
 * The server sends stable slugs, so the wording lives here. A slug this page
 * does not recognise is shown as itself rather than dropped: an unexplained row
 * in a delete list is worse than an awkward word, because the reader cannot tell
 * whether the list is complete.
 */
const REASON_TEXT: Record<string, string> = {
  custom_title: "you named it",
  in_project: "it is in a project",
  pinned: "you pinned it",
  archived: "you archived it",
  has_draft: "it holds an unsent draft",
  routing_chosen: "you chose how it routes",
  confirmation_changed: "you changed its confirmation setting",
  profile_chosen: "you chose a model for it",
  web_access_granted: "you gave it internet access",
  forked: "it was forked from another chat",
  settings_overridden: "you changed its generation settings",
  linked_chat: "another chat was forked or opened from it",
  stale_head: "its records disagree with each other",
};

/** Why a chosen chat is no longer part of the cleanup it was chosen for.
 *
 * A different vocabulary from the reasons above: these describe what changed
 * between looking and confirming.
 */
const CONFLICT_TEXT: Record<string, string> = {
  missing: "already gone",
  out_of_scope: "not a conversation this page manages",
  not_empty: "no longer empty",
  too_young: "too new to offer",
  archived_excluded: "archived",
  inconsistent: "its records disagree",
  filtered_out: "excluded by the filters in use",
};

function wordFor(table: Record<string, string>, slug: string): string {
  return table[slug] ?? slug;
}

/** What to do next after a refusal. A stale preview and a changed selection need
 *  different actions, so the wording keeps the codes' precision. */
function refusalText(code: string): string {
  switch (code) {
    case "empty-chat-preview-expired":
      return "That check is too old to act on. Nothing was deleted; check again.";
    case "empty-chat-preview-unknown":
      return "That selection was not checked first. Nothing was deleted; check again.";
    case "empty-chat-selection-drifted":
      return "Some of these chats changed while you were deciding. Nothing was deleted; check again.";
    case "empty-chat-count-mismatch":
      return "The number of chats changed since you were shown it. Nothing was deleted; check again.";
    case "empty-chat-configured-not-acknowledged":
      return "Chats you set up need their own confirmation. Nothing was deleted.";
    default:
      return "The chats were not deleted.";
  }
}

function kindText(entry: EmptyChatEntry): string {
  if (entry.classification === "strict_blank") return "Untouched";
  if (entry.classification === "configured_blank") return "Set up but unused";
  return "Records disagree";
}

function ageText(hours: number): string {
  const days = Math.floor(hours / 24);
  if (days >= 1) return `${days} ${days === 1 ? "day" : "days"} old`;
  const whole = Math.max(1, Math.floor(hours));
  return `${whole} ${whole === 1 ? "hour" : "hours"} old`;
}

function openChatId(): string | null {
  try {
    return localStorage.getItem(CURRENT_CHAT_STORAGE_KEY);
  } catch {
    return null;
  }
}

/** Review empty chats and delete the ones you choose, all of them or none.
 *
 * NOTHING IS SELECTED WHEN THE PAGE OPENS, and that is the load-bearing default.
 * A cleanup screen that arrives with everything ticked has already decided and
 * is asking for a rubber stamp; the work here is choosing. "Select untouched"
 * ticks only chats that carry no setting of anybody's and are not the chat open
 * in the workspace.
 *
 * Deleting goes through the server's check first. It binds the exact chats and
 * the filters they were chosen under, and says which of them no longer qualify.
 * The confirmation states the counts the server gave rather than counts worked
 * out here, and only the chats that still qualify are sent to be deleted - so
 * what is confirmed is exactly what the server will act on.
 */
export function EmptyChatMaintenance() {
  const client = useQueryClient();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [includeConfigured, setIncludeConfigured] = useState(false);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [pending, setPending] = useState<{
    preview: EmptyChatPreview;
    chatIds: string[];
    operationId: string;
  } | null>(null);
  const [acknowledgedConfigured, setAcknowledgedConfigured] = useState(false);
  const [outcome, setOutcome] = useState<{ kind: "success" | "error"; message: string } | null>(
    null,
  );

  const filters = { include_archived: includeArchived, include_configured: includeConfigured };
  const page = useQuery({
    queryKey: ["empty-chats", includeArchived, includeConfigured],
    queryFn: () => api.emptyChats(filters),
  });

  const entries: EmptyChatEntry[] = page.data?.entries ?? [];
  const offered = entries.filter((entry) => entry.deletable);
  // Only what is on screen can be chosen: a filter change that hides a chat
  // takes it out of the selection rather than deleting something unseen.
  const chosen = offered.filter((entry) => selected.has(entry.id)).map((entry) => entry.id);

  const toggle = (id: string, on: boolean) => {
    setSelected((current) => {
      const next = new Set(current);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });
  };

  const selectUntouched = () => {
    const open = openChatId();
    setSelected(
      new Set(
        offered
          .filter((entry) => entry.classification === "strict_blank" && entry.id !== open)
          .map((entry) => entry.id),
      ),
    );
  };

  const close = () => {
    setPending(null);
    setAcknowledgedConfigured(false);
  };

  const refused = (error: unknown) => {
    close();
    const code = error instanceof ApiError ? error.code ?? "" : "";
    setOutcome({ kind: "error", message: refusalText(code) });
    void client.invalidateQueries({ queryKey: ["empty-chats"] });
  };

  const check = useMutation({
    mutationFn: (chatIds: string[]) => api.previewEmptyChats({ chat_ids: chatIds, ...filters }),
    onSuccess: (preview, chatIds) => {
      setOutcome(null);
      const excluded = new Set(preview.conflicts.map((conflict) => conflict.chat_id));
      setPending({
        preview,
        // Only the chats the check bound. The rest were named as no longer
        // qualifying, and sending them would make the server refuse the lot.
        chatIds: chatIds.filter((id) => !excluded.has(id)),
        // Minted once per decision, not per attempt: a retry of the same
        // confirmation must be the same operation, or a lost response becomes
        // a second deletion.
        operationId: crypto.randomUUID(),
      });
    },
    onError: refused,
  });

  const remove = useMutation({
    mutationFn: () => {
      if (!pending) throw new Error("nothing to delete");
      return api.deleteEmptyChats({
        operation_id: pending.operationId,
        preview_id: pending.preview.preview_id,
        digest: pending.preview.digest,
        acknowledged_count: pending.preview.strict_count + pending.preview.configured_count,
        acknowledged_configured: acknowledgedConfigured,
        chat_ids: pending.chatIds,
        ...filters,
      });
    },
    onSuccess: (deletion) => {
      close();
      setSelected(new Set());
      const count = deletion.deleted_ids.length;
      setOutcome({
        kind: "success",
        message: `Deleted ${count} empty ${count === 1 ? "chat" : "chats"}.`,
      });
      void client.invalidateQueries({ queryKey: ["empty-chats"] });
      void client.invalidateQueries({ queryKey: ["chats"] });
    },
    onError: refused,
  });

  const bound = pending ? pending.preview.strict_count + pending.preview.configured_count : 0;
  const needsAcknowledgement = (pending?.preview.configured_count ?? 0) > 0;
  const busy = check.isPending || remove.isPending;

  return (
    <section>
      <div className="detail-title">
        <div>
          <h2>Empty chats</h2>
          <p>
            Conversations with nothing in them, older than a day. Each is described by why it is
            listed, never by anything written in it.
          </p>
        </div>
      </div>

      <div className="empty-chat-filters">
        <label>
          <input
            type="checkbox"
            checked={includeArchived}
            onChange={(event) => setIncludeArchived(event.target.checked)}
          />
          Include archived chats
        </label>
        <label>
          <input
            type="checkbox"
            checked={includeConfigured}
            onChange={(event) => setIncludeConfigured(event.target.checked)}
          />
          Include chats you set up
        </label>
      </div>

      {/* One live region for the whole result, so a screen reader hears the
          outcome once rather than hearing each row arrive. */}
      <p role="status" className="muted">
        {page.isPending
          ? "Looking for empty chats…"
          : page.isError
            ? "The list of empty chats could not be read."
            : entries.length === 0
              ? "No empty chats."
              : `${entries.length} empty ${entries.length === 1 ? "chat" : "chats"}, ${chosen.length} selected.`}
      </p>

      {entries.length > 0 && (
        <ul className="empty-chat-list">
          {entries.map((entry) => {
            const label = `${kindText(entry)}, ${ageText(entry.age_hours)}`;
            return (
              <li key={entry.id} className="empty-chat-row">
                {entry.deletable ? (
                  <input
                    type="checkbox"
                    aria-label={label}
                    checked={selected.has(entry.id)}
                    onChange={(event) => toggle(entry.id, event.target.checked)}
                  />
                ) : (
                  <span className="muted">Not offered</span>
                )}
                <span className="empty-chat-copy">
                  <strong>{kindText(entry)}</strong>
                  <small>
                    {[ageText(entry.age_hours), ...entry.reasons.map((reason) => wordFor(REASON_TEXT, reason))].join(" · ")}
                  </small>
                </span>
              </li>
            );
          })}
        </ul>
      )}

      {page.data?.next_cursor && (
        <p className="muted">More empty chats exist than are listed here. Delete these to see the rest.</p>
      )}

      {offered.length > 0 && (
        <div className="row-actions empty-chat-actions">
          <button type="button" className="secondary" onClick={selectUntouched}>
            Select untouched
          </button>
          <button type="button" className="secondary" onClick={() => setSelected(new Set())}>
            Clear selection
          </button>
          <button
            type="button"
            className="secondary danger"
            aria-disabled={busy || chosen.length === 0}
            onClick={() => {
              if (busy || chosen.length === 0) return;
              check.mutate(chosen);
            }}
          >
            {check.isPending ? "Checking…" : `Delete selected (${chosen.length})`}
          </button>
        </div>
      )}

      {outcome && (
        <div className={`callout ${outcome.kind}`} role={outcome.kind === "error" ? "alert" : "status"}>
          {outcome.message}
        </div>
      )}

      {pending && (
        <ConfirmDialog
          title="Delete these empty chats?"
          question={
            bound === 0
              ? "None of the selected chats can be deleted now."
              : pending.preview.configured_count > 0
                ? `${bound} ${bound === 1 ? "chat" : "chats"}, ${pending.preview.configured_count} of which you set up.`
                : `${bound} ${bound === 1 ? "chat" : "chats"}, none of which you set up.`
          }
          detail={
            <>
              {pending.preview.conflicts.length > 0 && (
                <p>
                  {`${pending.preview.conflicts.length} of the chats you selected no longer qualify and will not be touched: `}
                  {[...new Set(pending.preview.conflicts.map((conflict) => wordFor(CONFLICT_TEXT, conflict.reason)))]
                    .sort()
                    .join(", ")}
                  .
                </p>
              )}
              {needsAcknowledgement && (
                <label className="empty-chat-acknowledgement">
                  <input
                    type="checkbox"
                    checked={acknowledgedConfigured}
                    onChange={(event) => setAcknowledgedConfigured(event.target.checked)}
                  />
                  Also delete the chats I set up
                </label>
              )}
            </>
          }
          confirmLabel={remove.isPending ? "Deleting…" : "Delete"}
          /* The second confirmation is the server's rule, not a flourish: it
             refuses without it. Disabling the button says so before the
             request does. */
          confirmDisabled={
            bound === 0 || remove.isPending || (needsAcknowledgement && !acknowledgedConfigured)
          }
          onConfirm={() => remove.mutate()}
          onCancel={close}
        />
      )}
    </section>
  );
}
