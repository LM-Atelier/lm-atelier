import { useEffect, useSyncExternalStore } from "react";
import { useAppearance, type Appearance } from "./theme";

/** How wide the sidebar is, and whether it is there at all.
 *
 * Both are the person's, both survive a reload, and both are applied as one
 * custom property on the document so the grid needs no conditional rules.
 * Collapsing sets the width to zero rather than unmounting: the tree keeps
 * its state, so reopening does not lose which projects were expanded.
 *
 * Stored rather than held in component state, so Settings can put the
 * sidebar back without reaching into the workspace that draws it.
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

const listeners = new Set<() => void>();

function changed(): void {
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Another window resizing or resetting the sidebar moves this one too.
  const onStorage = (event: StorageEvent) => {
    if (event.key === null || event.key === SIDEBAR_WIDTH_KEY || event.key === SIDEBAR_COLLAPSED_KEY) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

function setSidebarWidth(next: number): void {
  localStorage.setItem(SIDEBAR_WIDTH_KEY, String(clampSidebarWidth(next)));
  changed();
}

function toggleSidebar(): void {
  localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(!storedCollapsed()));
  changed();
}

/** Put the sidebar back as a new workspace has it: shown, at its usual width. */
export function resetSidebarLayout(): void {
  localStorage.removeItem(SIDEBAR_WIDTH_KEY);
  localStorage.removeItem(SIDEBAR_COLLAPSED_KEY);
  changed();
}

/** The sidebar's width and whether it is hidden, for a reader that changes neither. */
export function useSidebarState(): { width: number; collapsed: boolean } {
  const width = useSyncExternalStore(subscribe, storedWidth, () => DEFAULT_SIDEBAR_WIDTH);
  const collapsed = useSyncExternalStore(subscribe, storedCollapsed, () => false);
  return { width, collapsed };
}

export interface SidebarLayout {
  width: number;
  collapsed: boolean;
  setWidth: (width: number) => void;
  toggle: () => void;
}

export function useSidebarLayout(): SidebarLayout {
  const { width, collapsed } = useSidebarState();

  useEffect(() => {
    // Zero when collapsed, so the grid column closes without a second rule.
    document.documentElement.style.setProperty(
      "--sidebar-width",
      collapsed ? "0px" : `${width}px`,
    );
    document.documentElement.dataset.sidebar = collapsed ? "collapsed" : "open";
  }, [width, collapsed]);

  return { width, collapsed, setWidth: setSidebarWidth, toggle: toggleSidebar };
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
