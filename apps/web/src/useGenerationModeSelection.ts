import { useCallback, useRef, useState } from "react";
import type { RoutingMode } from "./types";

export function useGenerationModeSelection(
  initialMode: RoutingMode,
  onChange: (mode: RoutingMode) => void,
) {
  const [mode, setMode] = useState(initialMode);
  const selectedMode = useRef(initialMode);
  const changeMode = useCallback((nextMode: RoutingMode) => {
    selectedMode.current = nextMode;
    setMode(nextMode);
    onChange(nextMode);
  }, [onChange]);
  const currentMode = useCallback(() => selectedMode.current, []);
  return { mode, changeMode, currentMode };
}