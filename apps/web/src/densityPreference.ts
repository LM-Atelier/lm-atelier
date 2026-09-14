import { useSyncExternalStore } from "react";

export type InterfaceDensity = "compact" | "standard" | "comfortable";
export const DENSITY_KEY = "local-lm-density";
export const DENSITIES: readonly InterfaceDensity[] = ["compact", "standard", "comfortable"];

const listeners = new Set<() => void>();

function storedDensity(): InterfaceDensity {
  try {
    const value = localStorage.getItem(DENSITY_KEY);
    return DENSITIES.includes(value as InterfaceDensity) ? value as InterfaceDensity : "standard";
  } catch {
    return "standard";
  }
}

export function setInterfaceDensity(density: InterfaceDensity): void {
  try {
    localStorage.setItem(DENSITY_KEY, density);
  } catch {
    return;
  }
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (event: StorageEvent) => {
    if (event.key === DENSITY_KEY || event.key === null) listener();
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

export function useInterfaceDensity(): InterfaceDensity {
  return useSyncExternalStore(subscribe, storedDensity, () => "standard");
}
