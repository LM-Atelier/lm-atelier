import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";
import { saveStudioDraft, savedStudioDraftWords, takeSavedStudioDraft } from "./studioSavedDraft";
import type { StudioToolState } from "./studioToolState";

/** A Studio visit's unfinished work, kept while the person looks elsewhere.
 *
 * Leaving the Studio for another view unmounts it, and what was drawn and typed
 * there went with it: someone who followed a tool's guidance to install its
 * workflow came back to the same picture with the selection, the tool's
 * settings and the words gone. The latest visit's work is kept in the query
 * cache for as long as the app is open, and given back when the same picture of
 * the same session is on the canvas again. Any other picture starts clean,
 * because a selection drawn on one picture means nothing on another. It is
 * also written down in this browser as the Studio is left or the page is
 * hidden, so a reload or a restart gives back its words and selection too.
 */
export type StudioDraft = {
  sessionId: string;
  /** The picture on the canvas when the Studio was left: what the selection was drawn on. */
  artifactId: string;
  tools: StudioToolState;
  instruction: string;
  /** The step chosen in the history, or null for the newest. */
  selectedId: string | null;
};

const DRAFT_KEY = ["studio-draft"];

export function useStudioDraft(sessionId: string | null) {
  const client = useQueryClient();
  const leaving = useRef<StudioDraft | null>(null);
  // Whether this visit has had a picture on the canvas and so looked for a
  // kept draft. A visit that ends before then has nothing of its own, and
  // keeping its empty start would lose the draft still waiting: React's
  // development checks end every first mount once, before any picture loads.
  const looked = useRef(false);
  useEffect(() => {
    // Kept for as long as the app is open rather than the cache's usual few
    // minutes: installing a workflow can take longer than that.
    client.setQueryDefaults(DRAFT_KEY, { gcTime: Infinity });
    // Written down whenever the page may not come back: hidden, or unloading.
    const writeDown = () => {
      if (looked.current && leaving.current) saveStudioDraft(leaving.current);
    };
    const onHidden = () => {
      if (document.visibilityState === "hidden") writeDown();
    };
    document.addEventListener("visibilitychange", onHidden);
    window.addEventListener("pagehide", writeDown);
    return () => {
      document.removeEventListener("visibilitychange", onHidden);
      window.removeEventListener("pagehide", writeDown);
      if (looked.current && leaving.current) client.setQueryData(DRAFT_KEY, leaving.current);
      writeDown();
    };
  }, [client]);
  // Read once for each session, so what it gives stays put while the visit lasts: what this app
  // kept in memory first, and failing that what this browser wrote down.
  const ours = useMemo(() => {
    const kept = client.getQueryData<StudioDraft>(DRAFT_KEY);
    return kept && kept.sessionId === sessionId ? kept : savedStudioDraftWords(sessionId);
  }, [client, sessionId]);
  return {
    /** The step the kept draft had chosen, when it belongs to this session. */
    selectedId: ours?.selectedId ?? null,
    /** The words the kept draft held, when it belongs to this session: they
     * describe an edit rather than a place on one picture, so they come back
     * with the session. */
    instruction: ours?.instruction ?? "",
    /** What leaving now would keep; called after every render. */
    track: (now: Omit<StudioDraft, "sessionId" | "artifactId"> & { artifactId: string | null }) => {
      const { artifactId } = now;
      leaving.current = sessionId && artifactId ? { ...now, sessionId, artifactId } : null;
    },
    /** The kept draft made on this picture, given back once: its selection and tool settings still fit it. */
    take: (artifactId: string | null, size: { width: number; height: number }): StudioDraft | null => {
      looked.current = true;
      const draft = client.getQueryData<StudioDraft>(DRAFT_KEY);
      if (!draft || draft.sessionId !== sessionId || draft.artifactId !== artifactId) {
        return takeSavedStudioDraft(sessionId, artifactId, size);
      }
      client.removeQueries({ queryKey: DRAFT_KEY, exact: true });
      return draft;
    },
  };
}
