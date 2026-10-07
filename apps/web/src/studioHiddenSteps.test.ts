/** Results hidden from the Studio's strip, remembered per session in this browser. */

import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  forgetHiddenStepsForTest,
  HIDDEN_STEPS_KEY,
  hideStudioStep,
  showStudioSteps,
  useStudioHiddenSteps,
} from "./studioHiddenSteps";

beforeEach(() => {
  localStorage.clear();
  forgetHiddenStepsForTest();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("hidden results", () => {
  it("hides a result in its own session only, and shows them all again on request", () => {
    const { result: mine } = renderHook(() => useStudioHiddenSteps("chat-a"));
    const { result: other } = renderHook(() => useStudioHiddenSteps("chat-b"));

    act(() => hideStudioStep("chat-a", "art-2"));
    act(() => hideStudioStep("chat-a", "art-3"));
    act(() => hideStudioStep("chat-a", "art-2"));

    expect([...mine.current]).toEqual(["art-2", "art-3"]);
    expect(other.current.size).toBe(0);

    act(() => showStudioSteps("chat-a"));
    expect(mine.current.size).toBe(0);
  });

  it("remembers what was hidden when the Studio is opened again", () => {
    act(() => hideStudioStep("chat-a", "art-2"));
    forgetHiddenStepsForTest();

    const { result } = renderHook(() => useStudioHiddenSteps("chat-a"));
    expect([...result.current]).toEqual(["art-2"]);
  });

  it("keeps the twenty sessions used most recently and forgets the oldest", () => {
    for (let session = 0; session < 21; session += 1) hideStudioStep(`chat-${session}`, "art");
    forgetHiddenStepsForTest();

    expect(renderHook(() => useStudioHiddenSteps("chat-0")).result.current.size).toBe(0);
    expect(renderHook(() => useStudioHiddenSteps("chat-1")).result.current.size).toBe(1);
    expect(renderHook(() => useStudioHiddenSteps("chat-20")).result.current.size).toBe(1);
  });

  it("hides nothing when what is stored cannot be read", () => {
    localStorage.setItem(HIDDEN_STEPS_KEY, "{not json");
    expect(renderHook(() => useStudioHiddenSteps("chat-a")).result.current.size).toBe(0);
    forgetHiddenStepsForTest();

    localStorage.setItem(HIDDEN_STEPS_KEY, JSON.stringify([{ session: "chat-a", hidden: [7] }, "chat-b"]));
    expect(renderHook(() => useStudioHiddenSteps("chat-a")).result.current.size).toBe(0);
  });

  it("still hides for this visit when storage refuses to keep it", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("full");
    });
    const { result } = renderHook(() => useStudioHiddenSteps("chat-a"));

    act(() => hideStudioStep("chat-a", "art-2"));

    expect([...result.current]).toEqual(["art-2"]);
  });
});
