import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "./api";
import { DRAFT_SAVE_DELAY_MS, composerDraftFrom, draftHasContent, storedDraft } from "./composerDraftStore";
import type { ComposerDraft, ComposerDraftUpdate } from "./composerPromptSource";
import type { ChatComposerDraft } from "./types";

/** Keep one chat's composer draft in the workspace, so it outlasts the page.
 *
 * On opening a chat, the stored draft is read and put into the composer if the
 * composer is still empty. Anything typed before it arrives wins, and nothing is
 * saved until the read has answered, so an empty composer cannot erase a stored
 * draft it has not seen yet.
 *
 * Afterwards the draft is saved whenever typing pauses, naming the revision this
 * page last saw, one save at a time and in order: a save still on its way when a
 * message is sent finishes before the emptied draft is saved, so it cannot bring
 * back what was sent. An emptied draft is saved as empty, which also lets go of
 * any file it held.
 *
 * If another window saved first, nothing is overwritten. Saving stops, this
 * page's draft stays in the composer, and the person chooses between it and the
 * saved one. Leaving the chat tries to save what is pending, but a page that is
 * closing is not guaranteed to finish.
 */
export function WorkspaceComposerDraft({
  chatId,
  draft,
  onDraft,
}: {
  chatId: string;
  draft: ComposerDraft;
  onDraft: (update: ComposerDraftUpdate) => void;
}) {
  const [conflict, setConflict] = useState<ChatComposerDraft | null>(null);
  const revision = useRef<number | null>(null);
  const saved = useRef<string | null>(null);
  const latest = useRef(draft);
  const applyDraft = useRef(onDraft);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const paused = useRef(false);

  useEffect(() => {
    latest.current = draft;
    applyDraft.current = onDraft;
  });

  useEffect(() => {
    let cancelled = false;
    // Started inside a promise so any failure, however it surfaces, lands in the
    // handler below rather than in the render of the chat.
    void Promise.resolve().then(() => api.composerDraft(chatId)).then(
      async (stored) => {
        if (cancelled) return;
        const fromStore = composerDraftFrom(stored);
        if (stored.revision === 0 || !draftHasContent(storedDraft(fromStore))) {
          saved.current = JSON.stringify(storedDraft(fromStore));
          revision.current = stored.revision;
          return;
        }
        const records = await Promise.all(
          stored.attachments.map((attachment) =>
            Promise.resolve().then(() => api.artifact(attachment.artifact_id)).catch(() => null)),
        );
        if (cancelled) return;
        const artifacts = new Map(records.flatMap((record) => (record ? [[record.id, record] as const] : [])));
        const restore = (current: ComposerDraft) =>
          draftHasContent(storedDraft(current)) ? current : composerDraftFrom(stored, artifacts);
        // Also what a save compares against before the next render, or it would
        // see the empty composer and empty the draft just read. Saving is only
        // armed once the restore is in place.
        latest.current = restore(latest.current);
        applyDraft.current(restore);
        saved.current = JSON.stringify(storedDraft(fromStore));
        revision.current = stored.revision;
      },
      () => {
        // Unreadable now: the composer keeps working, and nothing is saved over
        // a draft this page never saw.
      },
    );
    return () => {
      cancelled = true;
    };
  }, [chatId]);

  const save = useCallback(async () => {
    if (revision.current === null || paused.current) return;
    const next = storedDraft(latest.current);
    const serialized = JSON.stringify(next);
    if (serialized === saved.current) return;
    if (revision.current === 0 && !draftHasContent(next)) return;
    try {
      revision.current = (await api.saveComposerDraft(chatId, revision.current, next)).revision;
      saved.current = serialized;
    } catch (error) {
      if (!(error instanceof ApiError) || error.code !== "chat-draft-revision-stale") return;
      const current = await api.composerDraft(chatId).catch(() => null);
      // Unreadable: saving simply tries again later and meets the same refusal.
      if (!current) return;
      paused.current = true;
      setConflict(current);
    }
  }, [chatId]);

  const enqueueSave = useCallback(() => {
    queue.current = queue.current.then(save);
    return queue.current;
  }, [save]);

  const serializedDraft = JSON.stringify(storedDraft(draft));
  useEffect(() => {
    const timer = window.setTimeout(() => void enqueueSave(), DRAFT_SAVE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [serializedDraft, enqueueSave]);

  useEffect(() => () => void enqueueSave(), [enqueueSave]);

  if (!conflict) return null;
  const resolve = (keepThisWindow: boolean) => {
    revision.current = conflict.revision;
    if (keepThisWindow) {
      saved.current = null;
    } else {
      const theirs = composerDraftFrom(conflict);
      saved.current = JSON.stringify(storedDraft(theirs));
      latest.current = theirs;
      applyDraft.current(theirs);
    }
    paused.current = false;
    setConflict(null);
    if (keepThisWindow) void enqueueSave();
  };
  return (
    <div className="callout warning action-callout" role="alert">
      <span>This draft was also changed in another window. Nothing has been overwritten.</span>
      <button type="button" className="secondary compact-button" onClick={() => resolve(true)}>
        Keep this draft
      </button>
      <button type="button" className="secondary compact-button" onClick={() => resolve(false)}>
        Use the other one
      </button>
    </div>
  );
}
