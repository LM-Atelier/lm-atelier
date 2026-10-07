import { useLayoutEffect, useRef } from "react";

/** Reserve the space notices occupy so workspace and dialog controls stay clear. */
export function useNoticeSpace(connected: boolean, failure: Error | null) {
  const rail = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const stack = document.querySelector<HTMLElement>(".notification-stack");
    if (!stack && connected && !failure) return;
    const surface = stack ?? rail.current;
    if (!surface) return;
    const root = document.documentElement;
    const update = () => {
      const height = surface.getBoundingClientRect().height;
      root.style.setProperty("--notice-space", height > 0 ? `${Math.ceil(height) + 36}px` : "0px");
    };
    update();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    observer?.observe(surface);
    return () => {
      observer?.disconnect();
      root.style.removeProperty("--notice-space");
    };
  }, [connected, failure]);
  return rail;
}
