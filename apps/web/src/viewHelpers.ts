import type { EngineRole, RoutingMode } from "./types";

export function focusMainContent() {
  window.setTimeout(() => document.getElementById("main-content")?.focus(), 0);
}

export function roleForMode(mode: RoutingMode): EngineRole {
  if (mode === "video") return "video";
  if (mode === "image") return "image";
  return "chat";
}
