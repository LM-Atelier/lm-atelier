import type { EngineRole, RoutingMode } from "./types";

export function focusMainContent() {
  window.setTimeout(() => document.getElementById("main-content")?.focus(), 0);
}

export function roleForMode(mode: RoutingMode): EngineRole {
  if (mode === "video") return "video";
  if (mode === "image") return "image";
  return "chat";
}

// The settings drawer keys off the PERSISTED routing mode, the same source
// its caller's settingsRole comes from - the composer's local mode can trail
// it when something persists a mode without going through changeMode (the
// attachment Edit/Animate buttons, a failed routing PATCH rollback), and
// splitting the sources leaves the drawer's role picker visible but inert.
// In auto the drawer edits the picked role, so its schema and its
// edit-strength surface must both follow that role: with the image role
// picked and an image attached, the routed turn is an edit governed by that
// strength, and hiding it repeats the defect the drawer exists to fix.
export function drawerRoleView(
  routingMode: RoutingMode,
  settingsRole: EngineRole,
  editableImageAttached: boolean,
): { drawerMode: RoutingMode; drawerImageEdit: boolean } {
  const drawerMode: RoutingMode =
    routingMode === "auto" && settingsRole !== "chat" ? settingsRole : routingMode;
  return {
    drawerMode,
    drawerImageEdit: drawerMode === "image" && editableImageAttached,
  };
}
