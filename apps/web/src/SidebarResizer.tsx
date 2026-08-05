import { useCallback, useEffect, useRef } from "react";
import { MAX_SIDEBAR_WIDTH, MIN_SIDEBAR_WIDTH, type SidebarLayout } from "./sidebarLayout";

const KEYBOARD_STEP = 16;
/** Below this, a pointer press was a click rather than a drag. */
const DRAG_THRESHOLD = 3;

/** The edge between the sidebar and the work: drag it, or click it to hide.
 *
 * One control rather than two. A separate hide button is a second thing to
 * find and put somewhere, when the edge is already exactly where the hand is
 * and already means "this is the boundary".
 *
 * A click and a drag are told apart by distance, not by timing: a press that
 * never moves is a click however long it lasts, which is what someone who
 * pauses mid-gesture expects.
 */
export function SidebarResizer({ layout }: { layout: SidebarLayout }) {
  const dragging = useRef(false);
  const startedAt = useRef(0);
  const moved = useRef(false);

  const onMove = useCallback(
    (event: PointerEvent) => {
      if (!dragging.current) return;
      if (Math.abs(event.clientX - startedAt.current) > DRAG_THRESHOLD) moved.current = true;
      if (moved.current) layout.setWidth(event.clientX);
    },
    [layout],
  );

  const onUp = useCallback(() => {
    if (!dragging.current) return;
    dragging.current = false;
    document.body.classList.remove("resizing-sidebar");
    // A press that went nowhere was a click, and a click on the edge hides or
    // shows what it is the edge of.
    if (!moved.current) layout.toggle();
  }, [layout]);

  useEffect(() => {
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [onMove, onUp]);

  const label = layout.collapsed ? "Show sidebar" : "Hide sidebar, or drag to resize";
  return (
    // A focusable separator with aria-valuenow is a widget in ARIA - the
    // standard pattern for a splitter. The rule models `separator` as
    // decoration only, which is true of the static kind and not this one.
    /* eslint-disable jsx-a11y-x/no-noninteractive-element-interactions,
       jsx-a11y-x/no-noninteractive-tabindex */
    <div
      className="sidebar-resizer"
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      title={label}
      aria-valuenow={layout.width}
      aria-valuemin={MIN_SIDEBAR_WIDTH}
      aria-valuemax={MAX_SIDEBAR_WIDTH}
      tabIndex={0}
      onPointerDown={(event) => {
        event.preventDefault();
        dragging.current = true;
        moved.current = false;
        startedAt.current = event.clientX;
        document.body.classList.add("resizing-sidebar");
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") layout.toggle();
        else if (event.key === "ArrowLeft") layout.setWidth(layout.width - KEYBOARD_STEP);
        else if (event.key === "ArrowRight") layout.setWidth(layout.width + KEYBOARD_STEP);
        else return;
        event.preventDefault();
      }}
    />
  );
}
