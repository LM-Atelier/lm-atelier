import { PanelLeftClose, Settings, Sparkles } from "lucide-react";
import { ThemeToggle } from "./ThemeToggle";
import type { SetupReadinessReport } from "./types";
import type { View } from "./rooms";
import type { Appearance } from "./theme";

const SETUP_LABEL: Record<string, string> = {
  ready: "Ready",
  in_progress: "Working",
};

/** What the workspace is set to, rather than what is in it.
 *
 * Appearance, whether the sidebar is there at all, setup, and settings. The
 * tree above answers "what am I working on"; this answers "how".
 */
export function SidebarFooter({
  appearance,
  onToggleSidebar,
  setupState,
  view,
  onSetup,
  onView,
  onNavigate,
}: {
  appearance: Appearance;
  onToggleSidebar: () => void;
  setupState?: SetupReadinessReport["state"] | undefined;
  view: View;
  onSetup: () => void;
  onView: (view: View) => void;
  onNavigate: () => void;
}) {
  return (
    <div className="sidebar-footer">
      <ThemeToggle appearance={appearance} />
      <button
        className="icon-button"
        aria-label="Hide sidebar"
        title="Hide sidebar"
        onClick={onToggleSidebar}
      >
        <PanelLeftClose size={15} aria-hidden="true" />
      </button>
      <button onClick={() => { onSetup(); onNavigate(); }}>
        <Sparkles />Setup
        {setupState && (
          <small className={`setup-nav-state ${setupState}`}>
            {SETUP_LABEL[setupState] ?? "Action needed"}
          </small>
        )}
      </button>
      <button
        className={view === "settings" ? "active" : ""}
        aria-current={view === "settings" ? "page" : undefined}
        onClick={() => { onView("settings"); onNavigate(); }}
      >
        <Settings />Settings
      </button>
    </div>
  );
}
