import { useCallback, useEffect, useRef } from "react";
import { PanelLeftOpen } from "lucide-react";
import { MAX_SIDEBAR_WIDTH, MIN_SIDEBAR_WIDTH, type SidebarLayout } from "./sidebarLayout";

const KEYBOARD_STEP = 16;

/** The edge between the sidebar and the work, draggable and focusable.
 *
 * A separator is a real control, not decoration, so it carries the separator
 * role and its bounds - and it answers the arrow keys, because a width that
 * can only be set by dragging cannot be set by everyone.
 */
export function SidebarResizer({ layout }: { layout: SidebarLayout }) {
  const dragging = useRef(false);

  const onMove = useCallback(
    (event: PointerEvent) => {
      if (dragging.current) layout.setWidth(event.clientX);
    },
    [layout],
  );

  const onUp = useCallback(() => {
    dragging.current = false;
    document.body.classList.remove("resizing-sidebar");
  }, []);

  useEffect(() => {
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [onMove, onUp]);

  if (layout.collapsed) {
    // The way back cannot live inside the thing that is hidden.
    return (
      <button
        type="button"
        className="icon-button reveal-sidebar"
        aria-label="Show sidebar"
        title="Show sidebar"
        onClick={layout.toggle}
      >
        <PanelLeftOpen size={16} aria-hidden="true" />
      </button>
    );
  }
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
      aria-label="Resize sidebar"
      aria-valuenow={layout.width}
      aria-valuemin={MIN_SIDEBAR_WIDTH}
      aria-valuemax={MAX_SIDEBAR_WIDTH}
      tabIndex={0}
      onPointerDown={(event) => {
        event.preventDefault();
        dragging.current = true;
        document.body.classList.add("resizing-sidebar");
      }}
      onDoubleClick={layout.toggle}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft") layout.setWidth(layout.width - KEYBOARD_STEP);
        else if (event.key === "ArrowRight") layout.setWidth(layout.width + KEYBOARD_STEP);
        else return;
        event.preventDefault();
      }}
    />
  );
}
