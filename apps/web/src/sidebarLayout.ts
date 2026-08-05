import { useCallback, useEffect, useState } from "react";
import { useAppearance, type Appearance } from "./theme";

/** How wide the sidebar is, and whether it is there at all.
 *
 * Both are the person's, both survive a reload, and both are applied as one
 * custom property on the document so the grid needs no conditional rules.
 * Collapsing sets the width to zero rather than unmounting: the tree keeps
 * its state, so reopening does not lose which projects were expanded.
 */
export const SIDEBAR_WIDTH_KEY = "local-lm-sidebar-width";
export const SIDEBAR_COLLAPSED_KEY = "local-lm-sidebar-collapsed";

export const MIN_SIDEBAR_WIDTH = 200;
export const MAX_SIDEBAR_WIDTH = 520;
export const DEFAULT_SIDEBAR_WIDTH = 272;

export function clampSidebarWidth(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_SIDEBAR_WIDTH;
  return Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, Math.round(value)));
}

function storedWidth(): number {
  return clampSidebarWidth(Number(localStorage.getItem(SIDEBAR_WIDTH_KEY)) || DEFAULT_SIDEBAR_WIDTH);
}

function storedCollapsed(): boolean {
  return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "true";
}

export interface SidebarLayout {
  width: number;
  collapsed: boolean;
  setWidth: (width: number) => void;
  toggle: () => void;
}

export function useSidebarLayout(): SidebarLayout {
  const [width, setWidthState] = useState<number>(storedWidth);
  const [collapsed, setCollapsed] = useState<boolean>(storedCollapsed);

  useEffect(() => {
    // Zero when collapsed, so the grid column closes without a second rule.
    document.documentElement.style.setProperty(
      "--sidebar-width",
      collapsed ? "0px" : `${width}px`,
    );
    document.documentElement.dataset.sidebar = collapsed ? "collapsed" : "open";
  }, [width, collapsed]);

  const setWidth = useCallback((next: number) => {
    const clamped = clampSidebarWidth(next);
    setWidthState(clamped);
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(clamped));
  }, []);

  const toggle = useCallback(() => {
    setCollapsed((current) => {
      const next = !current;
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(next));
      return next;
    });
  }, []);

  return { width, collapsed, setWidth, toggle };
}

/** Everything about the shape of the workspace, in one call.
 *
 * Appearance and sidebar layout are the same kind of thing: persisted,
 * document-level, and nothing to do with what is being worked on. Grouping
 * them keeps that boundary visible from the call site.
 */
export function useWorkspaceChrome(): { appearance: Appearance; sidebar: SidebarLayout } {
  return { appearance: useAppearance(), sidebar: useSidebarLayout() };
}
