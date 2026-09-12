import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import type { View } from "./rooms";
import { DEFAULT_SETTINGS_DESTINATION, settingsDestinationFor } from "./settingsDestinations";

const VIEWS: readonly View[] = ["chat", "media", "models", "references", "prompts", "workflows", "studio", "settings"];
const SETTINGS_KEY = "lm-atelier.settings-destination";

interface Navigation {
  view: View;
  destination: string;
  cause: "initial" | "view" | "settings" | "history";
  sequence: number;
}

function rememberedDestination(): string {
  try {
    return settingsDestinationFor(sessionStorage.getItem(SETTINGS_KEY)).id;
  } catch {
    return DEFAULT_SETTINGS_DESTINATION;
  }
}

function rememberDestination(destination: string) {
  try {
    sessionStorage.setItem(SETTINGS_KEY, destination);
  } catch {
    // Navigation still works when browser storage is unavailable.
  }
}

function locationNavigation(remembered: string): Pick<Navigation, "view" | "destination"> {
  const params = new URL(window.location.href).searchParams;
  const rawView = params.get("view");
  const view = VIEWS.find((candidate) => candidate === rawView) ?? "chat";
  const destination = view === "settings"
    ? settingsDestinationFor(params.get("settings") ?? remembered).id
    : remembered;
  return { view, destination };
}

function writeLocation(navigation: Navigation, method: "pushState" | "replaceState") {
  const url = new URL(window.location.href);
  url.searchParams.set("view", navigation.view);
  if (navigation.view === "settings") url.searchParams.set("settings", navigation.destination);
  else url.searchParams.delete("settings");
  if (url.href !== window.location.href) {
    window.history[method](window.history.state, "", url);
  }
}

/** Workspace history contains page names only; document selection stays local. */
export function useAppNavigation() {
  const [navigation, setNavigation] = useState<Navigation>(() => ({
    ...locationNavigation(rememberedDestination()), cause: "initial", sequence: 0,
  }));
  const current = useRef(navigation);

  const navigate = useCallback((next: Navigation, method: "pushState" | "replaceState") => {
    writeLocation(next, method);
    rememberDestination(next.destination);
    current.current = next;
    setNavigation(next);
  }, []);

  useEffect(() => {
    writeLocation(current.current, "replaceState");
    rememberDestination(current.current.destination);
    const onHistory = () => {
      const previous = current.current;
      navigate({
        ...locationNavigation(previous.destination),
        cause: "history", sequence: previous.sequence + 1,
      }, "replaceState");
    };
    window.addEventListener("popstate", onHistory);
    return () => window.removeEventListener("popstate", onHistory);
  }, [navigate]);

  const setView: Dispatch<SetStateAction<View>> = useCallback((action) => {
    const previous = current.current;
    const view = typeof action === "function" ? action(previous.view) : action;
    navigate({ ...previous, view, cause: "view", sequence: previous.sequence + 1 }, "pushState");
  }, [navigate]);

  const setSettingsDestination = useCallback((id: string) => {
    const previous = current.current;
    navigate({
      view: "settings", destination: settingsDestinationFor(id).id,
      cause: "settings", sequence: previous.sequence + 1,
    }, "pushState");
  }, [navigate]);

  return {
    view: navigation.view,
    setView,
    settingsDestination: navigation.destination,
    setSettingsDestination,
    settingsFocusRequest: navigation.view === "settings"
      && (navigation.cause === "settings" || navigation.cause === "history")
      ? navigation.sequence : undefined,
    mainFocusRequest: navigation.view !== "settings" && navigation.cause === "history"
      ? navigation.sequence : undefined,
  };
}
