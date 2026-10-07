import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { useFirstRunSetup } from "./useFirstRunSetup";

afterEach(() => { cleanup(); window.history.replaceState(null, "", "/"); });

it("clears only the setup flag while preserving navigation and launcher state", () => {
  const state = { launcher: "neutral-fixture" };
  window.history.replaceState(state, "", "/?firstRunSetup=1&view=settings&settings=advanced&keep=1#anchor");
  const { result } = renderHook(useFirstRunSetup);
  expect(result.current[0]).toBe(true);
  act(() => result.current[1]());
  expect(result.current[0]).toBe(false);
  const params = new URL(window.location.href).searchParams;
  expect(params.has("firstRunSetup")).toBe(false);
  expect(params.get("view")).toBe("settings");
  expect(params.get("settings")).toBe("advanced");
  expect(params.get("keep")).toBe("1");
  expect(window.location.hash).toBe("#anchor");
  expect(window.history.state).toEqual(state);
});
