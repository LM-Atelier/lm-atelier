import { useEffect, type RefObject } from "react";

/** The custom property that says how much of the window's foot the composer takes. */
export const COMPOSER_CLEARANCE = "--composer-clearance";

/** Publish how far the composer reaches up from the bottom of its area.
 *
 * The jobs panel floats over the page and must stay clear of the composer, or
 * it covers the text box, the attach button and the send button while work
 * runs. The composer grows as it is typed into, so its height is measured, not
 * assumed. Only one composer sits at the foot of a view; an editor opened
 * inside the transcript passes `active` false and publishes nothing.
 */
export function useComposerClearance(wrap: RefObject<HTMLElement | null>, active: boolean): void {
  useEffect(() => {
    const element = wrap.current;
    if (!active || !element) return;
    const root = document.documentElement;
    const publish = () => {
      const box = element.querySelector(".composer") ?? element;
      const clearance = element.getBoundingClientRect().bottom - box.getBoundingClientRect().top;
      root.style.setProperty(COMPOSER_CLEARANCE, `${Math.max(0, Math.round(clearance))}px`);
    };
    publish();
    // Without a way to hear it grow, the first measure stands.
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(publish);
    observer?.observe(element);
    return () => {
      observer?.disconnect();
      root.style.removeProperty(COMPOSER_CLEARANCE);
    };
  }, [wrap, active]);
}
