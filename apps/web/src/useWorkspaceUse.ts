import { useEffect } from "react";
import { api } from "./api";

/** The input that counts as someone using the workspace. Moving the pointer does not. */
const USE_EVENTS = ["pointerdown", "keydown", "touchstart"] as const;
/** The longest a report waits after a use, so the server's spell never runs far ahead of this page's. */
const LONGEST_REPORT_GAP_MS = 30_000;

/** Tell the server, now and then, that someone is using this window.
 *
 * With a quiet-spell lock chosen, the server locks once nobody has used the
 * workspace for that long, in any window. Only a key, a click or a touch here
 * starts the spell again; live updates, polling and running work never do. A
 * report goes at most once per sixth of the spell, and never more than half a
 * minute after a use, so a page in steady use never lets the lock arrive early
 * by more than that.
 */
export function useWorkspaceUse(idleSeconds: number | null): void {
  useEffect(() => {
    if (idleSeconds === null) return;
    const gap = Math.min(LONGEST_REPORT_GAP_MS, (idleSeconds * 1000) / 6);
    let lastReport = Number.NEGATIVE_INFINITY;
    const onUse = () => {
      const now = performance.now();
      if (now - lastReport < gap) return;
      lastReport = now;
      // A refused report says nothing a later request will not say better.
      void api.noteWorkspaceUse().catch(() => undefined);
    };
    for (const type of USE_EVENTS) window.addEventListener(type, onUse, { capture: true, passive: true });
    return () => {
      for (const type of USE_EVENTS) window.removeEventListener(type, onUse, { capture: true });
    };
  }, [idleSeconds]);
}
