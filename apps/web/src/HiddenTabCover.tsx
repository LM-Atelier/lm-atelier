import { useEffect, useRef, useState, type ReactNode } from "react";
import { useHiddenTabCover } from "./tabCoverChoice";
import { focusMainContent } from "./viewHelpers";

function CoveredTab({ onShow }: { onShow: () => void }) {
  const show = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    show.current?.focus();
  }, []);
  return (
    <main className="first-run-shell workspace-lock-shell">
      <div className="workspace-lock-card">
        <h1>LM Atelier is covered</h1>
        <p>This window was covered when it was hidden. Nothing was locked or stopped.</p>
        <div className="workspace-lock-actions">
          <button ref={show} type="button" className="primary" onClick={onShow}>
            Show
          </button>
        </div>
      </div>
    </main>
  );
}

/** Covers this tab from the moment it is hidden until someone chooses Show.
 *
 * The workspace stays mounted underneath, so a draft, a scroll position or
 * running work is exactly as it was, but it is taken out of the page's layout
 * and out of the accessibility tree while covered. Only this tab is covered:
 * hiding one tab never covers another that is still in view. The server lock
 * is separate and still applies on top.
 */
export function HiddenTabCover({ children }: { children: ReactNode }) {
  const enabled = useHiddenTabCover();
  const [covered, setCovered] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    const onVisibility = () => {
      if (document.visibilityState === "hidden") setCovered(true);
    };
    // A tab that is already hidden when the cover is turned on, in another
    // tab or as this one opens in the background, is covered as well.
    onVisibility();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      // Turning the cover off uncovers this tab, and turning it on again later
      // starts from what is on screen then.
      setCovered(false);
    };
  }, [enabled]);

  const hidden = enabled && covered;
  return (
    <>
      <div className={hidden ? undefined : "hidden-tab-contents"} hidden={hidden}>
        {children}
      </div>
      {hidden && (
        <CoveredTab
          onShow={() => {
            setCovered(false);
            focusMainContent();
          }}
        />
      )}
    </>
  );
}
