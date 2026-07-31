import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useGenerationModeSelection } from "./useGenerationModeSelection";

describe("useGenerationModeSelection", () => {
  it("snapshots a selection before React rerenders", () => {
    const onChange = vi.fn();
    const { result } = renderHook(() => useGenerationModeSelection("auto", onChange));

    act(() => {
      result.current.changeMode("image");
      expect(result.current.currentMode()).toBe("image");
    });

    expect(result.current.mode).toBe("image");
    expect(onChange).toHaveBeenCalledWith("image");
  });
});